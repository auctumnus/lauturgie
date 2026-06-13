//! A/B the tiers over the vendor corpora: every word through `apply` with
//! the FST tier enabled vs forced onto the VM. Run with `--release`.
fn main() {
    let pairs = [
        ("kharulian", "pk_full_conjugated_verbs"),
        ("kharulian", "pk_prefixed_verbs"),
        ("nitherwe", "pn_finite_verbs"),
        ("syllabian", "proto-syllabian"),
    ];
    let mut total_bad = 0usize;
    for (lsc, wli) in pairs {
        let src = std::fs::read_to_string(format!("vendor/lexurgy/cli/test/{lsc}.lsc")).unwrap();
        let words = std::fs::read_to_string(format!("vendor/lexurgy/cli/test/{wli}.wli")).unwrap();
        let statements = lauturgie::parse(&src).unwrap();
        let mut fast = lauturgie::compiler::compile(&statements).unwrap();
        let mut slow = lauturgie::compiler::compile(&statements).unwrap();
        slow.force_vm = true;
        let mut n = 0usize;
        let mut bad = 0usize;
        for w in words.lines().filter(|l| !l.is_empty()) {
            n += 1;
            let (f, v) = (fast.apply(w), slow.apply(w));
            if f != v {
                bad += 1;
                if bad <= 5 {
                    println!("  MISMATCH {lsc} {w:?}: fst={f:?} vm={v:?}");
                }
            }
        }
        total_bad += bad;
        println!("{lsc}/{wli}: {n} words, {bad} mismatches");
    }
    std::process::exit((total_bad > 0) as i32);
}
