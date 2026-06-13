//! Differential testing of the FST tier against the reference VM.
//!
//! Every case extracted from lexurgy's test suite runs through two
//! pipelines (one preferring compiled FSTs, one forced onto the VM) and
//! the outputs must agree exactly (including errors). A floor on the
//! number of FST-compiled rules keeps the tier from silently regressing
//! into "nothing is eligible".

use std::path::PathBuf;

fn lexurgy_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("LEXURGY_DIR") {
        return PathBuf::from(dir);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("vendor/lexurgy")
}

/// Pull `(snippet, input)` pairs out of the lexurgy test corpus; a
/// lighter-weight version of the extraction in `tests/differential.rs`
/// (expected outputs don't matter here; the VM is the oracle).
fn extract(test_dir: &std::path::Path) -> Vec<(String, String)> {
    let mut cases = Vec::new();
    for entry in std::fs::read_dir(test_dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_none_or(|e| e != "kt") {
            continue;
        }
        let src = std::fs::read_to_string(&path).unwrap();
        let mut snippets: Vec<(usize, String)> = Vec::new();
        let mut search_from = 0;
        while let Some(call) = src[search_from..].find("lsc(") {
            let call_at = search_from + call;
            search_from = call_at + 4;
            let after = &src[call_at + 4..];
            let lead: usize = after
                .bytes()
                .take_while(|b| b.is_ascii_whitespace())
                .count();
            if let Some(body) = after[lead..].strip_prefix("\"\"\"") {
                if let Some(end) = body.find("\"\"\"") {
                    snippets.push((call_at, body[..end].replace("${'$'}", "$")));
                }
            }
        }
        let mut offset = 0;
        for line in src.lines() {
            let line_at = offset;
            offset += line.len() + 1;
            let Some((call, _)) = line.trim().split_once("\") shouldBe \"") else {
                continue;
            };
            let Some((_, input)) = call.split_once("(\"") else {
                continue;
            };
            if input.contains('\\') || input.contains('$') || input.contains(' ') {
                continue;
            }
            if let Some((_, snippet)) = snippets.iter().rev().find(|(at, _)| *at < line_at) {
                cases.push((snippet.clone(), input.to_string()));
            }
        }
    }
    cases
}

#[test]
fn fst_tier_agrees_with_vm() {
    let test_dir = lexurgy_dir().join("core/src/test/kotlin/com/meamoria/lexurgy/sc");
    assert!(test_dir.is_dir(), "missing lexurgy checkout");
    let cases = extract(&test_dir);
    assert!(cases.len() > 400, "extraction broke: {} cases", cases.len());

    let mut compared = 0usize;
    let mut fst_rules = 0usize;
    let mut fused_runs = 0usize;
    for (snippet, input) in &cases {
        let Ok(statements) = lauturgie::parse(snippet) else {
            continue;
        };
        let Ok(mut fast) = lauturgie::compiler::compile(&statements) else {
            continue;
        };
        let Ok(mut slow) = lauturgie::compiler::compile(&statements) else {
            continue;
        };
        slow.force_vm = true;
        fst_rules += fast.fst_rules.iter().flatten().count();
        fused_runs += fast.fused.len();
        let a = fast.apply(input);
        let b = slow.apply(input);
        assert_eq!(
            a, b,
            "tiers disagree on {input:?} under:\n---\n{snippet}\n---"
        );
        compared += 1;
    }
    eprintln!(
        "fst vs vm: {compared} cases compared, {fst_rules} rule instances on the fst tier, \
         {fused_runs} fused runs"
    );
    assert!(compared > 400, "only {compared} cases compared");
    assert!(
        fst_rules > 200,
        "only {fst_rules} rules took the fst tier; eligibility regressed"
    );
}
