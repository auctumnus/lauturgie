//! Times lauturgie the same way the Kotlin lexurgy CLI times itself:
//! rule compilation excluded, one cold pass over the full word list,
//! romanization included. Compare against the CLI's console line
//! "Applied the changes to N words in S seconds".
//!
//! Usage: bench_vs_kotlin [DIR]   (default /tmp/lexurgy-bench)

use std::time::Instant;

fn main() {
    let dir = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "/tmp/lexurgy-bench".into());
    let pairs = [
        ("kharulian", "pk_full_conjugated_verbs"),
        ("kharulian", "pk_prefixed_verbs"),
        ("nitherwe", "pn_finite_verbs"),
        ("syllabian", "proto-syllabian"),
    ];
    for (lsc, wli) in pairs {
        let src = std::fs::read_to_string(format!("{dir}/{lsc}.lsc")).unwrap();
        let words = std::fs::read_to_string(format!("{dir}/{wli}.wli")).unwrap();
        let words: Vec<&str> = words.lines().filter(|l| !l.is_empty()).collect();
        let statements = lauturgie::parse(&src).unwrap();

        // serial fst, serial vm, then parallel fst (kotlin parallelStreams
        // across words by default, so `par` is the like-for-like row)
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
            let errors = results.iter().filter(|r| r.is_err()).count();
            println!(
                "{lsc:10} x {:5} words [{label}]: {secs:.3} s ({errors} errors)",
                words.len(),
            );
            if label == "fst" {
                let out: Vec<String> = results
                    .into_iter()
                    .map(|r| r.unwrap_or_else(|_| "ERROR".into()))
                    .collect();
                std::fs::write(format!("{dir}/{wli}_lauturgie.wli"), out.join("\n") + "\n")
                    .unwrap();
            }
        }
    }
}
