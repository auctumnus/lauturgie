//! Like bench_vs_kotlin, but for one arbitrary ruleset/wordlist pair.
//! Timing scope matches the Kotlin CLI: compile excluded, one cold pass,
//! romanization included.
//!
//! Usage: bench_one CHANGES.lsc WORDS.wli [OUT.wli]

use std::time::Instant;

fn main() {
    let mut args = std::env::args().skip(1);
    let lsc = args
        .next()
        .expect("usage: bench_one CHANGES.lsc WORDS.wli [OUT.wli]");
    let wli = args
        .next()
        .expect("usage: bench_one CHANGES.lsc WORDS.wli [OUT.wli]");
    let out_path = args.next();

    let src = std::fs::read_to_string(&lsc).unwrap();
    let words = std::fs::read_to_string(&wli).unwrap();
    let words: Vec<&str> = words.lines().filter(|l| !l.is_empty()).collect();
    let statements = lauturgie::parse(&src).unwrap();

    for (label, force_vm, par) in [
        ("fst", false, false),
        ("vm ", true, false),
        ("par", false, true),
    ] {
        let mut compiled = lauturgie::compiler::compile(&statements).unwrap();
        compiled.force_vm = force_vm;
        let start = Instant::now();
        let results = if par {
            compiled.apply_all(&words)
        } else {
            words.iter().map(|w| compiled.apply(w)).collect()
        };
        let secs = start.elapsed().as_secs_f64();
        let mut errors = 0usize;
        let mut out = Vec::with_capacity(words.len());
        for (w, r) in words.iter().zip(results) {
            match r {
                Ok(o) => out.push(o),
                Err(e) => {
                    errors += 1;
                    if errors <= 3 && !par {
                        eprintln!("error on {w:?}: {e:?}");
                    }
                    out.push("ERROR".into());
                }
            }
        }
        println!(
            "{:5} words [{label}]: {secs:.3} s ({errors} errors)",
            words.len(),
        );
        if label == "fst" {
            if let Some(p) = &out_path {
                std::fs::write(p, out.join("\n") + "\n").unwrap();
            }
        }
    }
}
