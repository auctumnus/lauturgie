//! Per-rule tier report: which rules compile to the FST tier, why the rest
//! stay on the VM, and (with a word list) where the time actually goes.
//!
//! Timing works by compiling growing statement prefixes and diffing total
//! VM-tier apply time, so each rule is measured on the words it really
//! sees. Run with `--release`.
//!
//! Usage: tier_report [LSC WLI]   (default: the four vendor test rulesets)

use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let pairs: Vec<(String, Option<String>)> = if args.len() == 2 {
        vec![(args[0].clone(), Some(args[1].clone()))]
    } else {
        [
            ("kharulian", Some("pk_full_conjugated_verbs")),
            ("muipidan", None),
            ("nitherwe", Some("pn_finite_verbs")),
            ("syllabian", Some("proto-syllabian")),
        ]
        .into_iter()
        .map(|(l, w)| {
            (
                format!("vendor/lexurgy/cli/test/{l}.lsc"),
                w.map(|w| format!("vendor/lexurgy/cli/test/{w}.wli")),
            )
        })
        .collect()
    };

    for (lsc, wli) in pairs {
        let src = std::fs::read_to_string(&lsc).unwrap();
        let statements = lauturgie::parse(&src).unwrap();
        let compiled = lauturgie::compiler::compile(&statements).unwrap();

        let words: Vec<String> = match &wli {
            None => vec![],
            Some(path) => std::fs::read_to_string(path)
                .unwrap()
                .lines()
                .filter(|l| !l.is_empty())
                .take(500)
                .map(str::to_owned)
                .collect(),
        };
        let times = rule_times(&statements, &words, true);
        let fst_times = rule_times(&statements, &words, false);

        let eligible = compiled.fst_rules.iter().flatten().count();
        let fused_rules: usize = compiled.fused.iter().map(|f| f.rules.len()).sum();
        println!(
            "{lsc}: {}/{} rules on fst tier, {} fused runs covering {} rules",
            eligible,
            compiled.rules.len(),
            compiled.fused.len(),
            fused_rules,
        );
        let mut universes = vec![lauturgie::compiler::Universe::Real; compiled.rules.len()];
        for step in &compiled.steps {
            if let lauturgie::compiler::Step::Rule { rule, universe } = step {
                universes[*rule] = *universe;
            }
        }
        for (i, rule) in compiled.rules.iter().enumerate() {
            let (decls, segments) = match universes[i] {
                lauturgie::compiler::Universe::Real => (&compiled.decls, &compiled.segments),
                lauturgie::compiler::Universe::Literal => {
                    (&compiled.literal_decls, &compiled.literal_segments)
                }
            };
            let tier = match lauturgie::fst::compile_rule(rule, decls, segments) {
                Ok(f) => match f.splice_reason() {
                    None => "fst".to_string(),
                    Some(why) => format!("fst gate ({why})"),
                },
                Err(why) => format!("vm: {why}"),
            };
            let time = match times.get(rule.name.as_str()) {
                Some(ms) => format!("{ms:7.1}ms"),
                None => "      -  ".to_string(),
            };
            let ftime = match fst_times.get(rule.name.as_str()) {
                Some(ms) => format!("{ms:7.1}ms  "),
                None => "      -    ".to_string(),
            };
            println!("  vm{time} fst{ftime}{:<28}{tier}", rule.name);
        }
        println!();
    }
}

/// Per-rule VM-tier cost: time growing statement prefixes (everything
/// forced onto the VM) and attribute each increase to the statement that
/// joined. Only named rules are reported.
fn rule_times(
    statements: &[lauturgie::parser::ast::Statement],
    words: &[String],
    force_vm: bool,
) -> std::collections::HashMap<String, f64> {
    use lauturgie::parser::ast::Statement;
    let mut times = std::collections::HashMap::new();
    if words.is_empty() {
        return times;
    }
    let mut prev = 0.0_f64;
    for n in 1..=statements.len() {
        let Ok(mut compiled) = lauturgie::compiler::compile(&statements[..n]) else {
            continue; // prefix not yet self-contained
        };
        compiled.force_vm = force_vm;
        for w in words {
            let _ = compiled.apply(w);
        }
        let start = Instant::now();
        for w in words {
            let _ = compiled.apply(w);
        }
        let total = start.elapsed().as_secs_f64() * 1000.0;
        if let Statement::Rule(r) = &statements[n - 1] {
            times.insert(r.name.to_string(), total - prev);
        }
        prev = total;
    }
    times
}
