//! Rough tier benchmark: real rulesets over real word lists, FST tier
//! enabled vs forced onto the VM. Run with `--release`.

use std::time::Instant;

fn main() {
    let pairs = [
        ("kharulian", "pk_full_conjugated_verbs"),
        ("kharulian", "pk_prefixed_verbs"),
        ("nitherwe", "pn_finite_verbs"),
        ("syllabian", "proto-syllabian"),
    ];
    for (lsc, wli) in pairs {
        let src = std::fs::read_to_string(format!("vendor/lexurgy/cli/test/{lsc}.lsc")).unwrap();
        let words = std::fs::read_to_string(format!("vendor/lexurgy/cli/test/{wli}.wli")).unwrap();
        let words: Vec<&str> = words
            .lines()
            .filter(|l| !l.is_empty())
            .take(2_000)
            .collect();
        let statements = lauturgie::parse(&src).unwrap();

        let mut run = |force_vm: bool| {
            let mut compiled = lauturgie::compiler::compile(&statements).unwrap();
            compiled.force_vm = force_vm;
            // warm caches once, then time repeated application
            for w in &words {
                let _ = compiled.apply(w);
            }
            let reps = 5;
            let start = Instant::now();
            let mut ok = 0usize;
            for _ in 0..reps {
                for w in &words {
                    ok += compiled.apply(w).is_ok() as usize;
                }
            }
            (start.elapsed() / (reps as u32), ok / reps)
        };

        let (fst, ok_fst) = run(false);
        let (vm, ok_vm) = run(true);
        assert_eq!(ok_fst, ok_vm);
        println!(
            "{lsc:10} x {:4} words: vm {vm:>9.2?}  fst {fst:>9.2?}  ({:.1}x)",
            words.len(),
            vm.as_secs_f64() / fst.as_secs_f64(),
        );
    }
}
