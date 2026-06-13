//! In-process CPU profiler → flamegraph SVG. No `perf` or root needed:
//! pprof samples this thread via SIGPROF and folds stacks into an SVG.
//!
//! Usage:
//!   cargo run --release --example profile -- [LSC] [WLI] [--vm] [--reps N] [--out FILE]
//!
//! Defaults to kharulian over pk_full_conjugated_verbs (~64k words), FST tier.
//! Pass `--vm` to pin the changer to the VM tier for an A/B of hot paths.

use std::time::Instant;

fn main() {
    let mut lsc = "kharulian".to_string();
    let mut wli = "pk_full_conjugated_verbs".to_string();
    let mut force_vm = false;
    let mut reps = 3usize;
    let mut out: Option<String> = None;
    let mut take: Option<usize> = None;

    let mut args = std::env::args().skip(1);
    let mut positional = 0;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--vm" => force_vm = true,
            "--reps" => reps = args.next().unwrap().parse().unwrap(),
            "--take" => take = Some(args.next().unwrap().parse().unwrap()),
            "--out" => out = Some(args.next().unwrap()),
            _ => {
                match positional {
                    0 => lsc = a,
                    1 => wli = a,
                    _ => panic!("unexpected arg {a:?}"),
                }
                positional += 1;
            }
        }
    }

    let out = out.unwrap_or_else(|| {
        format!(
            "flamegraph-{lsc}-{}.svg",
            if force_vm { "vm" } else { "fst" }
        )
    });

    let src = std::fs::read_to_string(format!("vendor/lexurgy/cli/test/{lsc}.lsc")).unwrap();
    let words = std::fs::read_to_string(format!("vendor/lexurgy/cli/test/{wli}.wli")).unwrap();
    let mut words: Vec<&str> = words.lines().filter(|l| !l.is_empty()).collect();
    if let Some(n) = take {
        words.truncate(n);
    }
    let statements = lauturgie::parse(&src).unwrap();

    let mut compiled = lauturgie::compiler::compile(&statements).unwrap();
    compiled.force_vm = force_vm;

    // warm caches so the profile reflects steady-state apply, not first-touch
    // DFA construction.
    for w in &words {
        let _ = compiled.apply(w);
    }

    let guard = pprof::ProfilerGuardBuilder::default()
        .frequency(2000)
        .blocklist(&["libc", "libgcc", "pthread", "vdso"])
        .build()
        .unwrap();

    let start = Instant::now();
    let mut ok = 0usize;
    for _ in 0..reps {
        for w in &words {
            ok += compiled.apply(w).is_ok() as usize;
        }
    }
    let elapsed = start.elapsed();

    let tier = if force_vm { "vm " } else { "fst" };
    let per = elapsed / (reps as u32 * words.len() as u32);
    eprintln!(
        "{lsc} x {} words [{tier}] x{reps}: {elapsed:.2?} total, {per:.0?}/word, {ok} ok/rep",
        words.len(),
        ok = ok / reps,
    );

    if let Ok(report) = guard.report().build() {
        let file = std::fs::File::create(&out).unwrap();
        report.flamegraph(file).unwrap();
        eprintln!("wrote {out}");

        // Self-time ranking: charge each sample to its leaf frame. This is the
        // "where is the CPU actually spending cycles" view the SVG hides.
        let mut self_time: std::collections::HashMap<String, isize> = Default::default();
        let mut total = 0isize;
        for (frames, count) in report.data.iter() {
            total += *count;
            if let Some(leaf) = frames.frames.first().and_then(|f| f.first()) {
                *self_time.entry(leaf.name()).or_default() += *count;
            }
        }
        let mut ranked: Vec<_> = self_time.into_iter().collect();
        ranked.sort_by_key(|(_, c)| std::cmp::Reverse(*c));
        eprintln!("\n=== self-time (leaf frame), top 25 of {total} samples ===");
        for (name, count) in ranked.into_iter().take(25) {
            eprintln!("{:6.2}%  {name}", 100.0 * *&count as f64 / total as f64);
        }
    }
}
