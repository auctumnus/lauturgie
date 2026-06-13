//! Differential testing against lexurgy's own test suite.
//!
//! Lexurgy's Kotlin tests pair `val ch = lsc(""" ... """)` definitions with
//! `ch("input") shouldBe "output"` assertions: thousands of authoritative
//! input/output examples. This test extracts them and asserts that whatever
//! we can compile and run produces *exactly* lexurgy's output.
//!
//! Cases we can't run yet are skipped and counted, never silently wrong:
//! - multi-word phrases (spaces in input/output)
//! - constructs that fail loudly at compile or run time land in those
//!   counts
//!
//! The test fails on any output mismatch, and also if the number of
//! *matching* cases ever drops below a floor, so silently skipping
//! everything can't fake a pass.

use std::path::PathBuf;

use unicode_normalization::UnicodeNormalization;

fn lexurgy_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("LEXURGY_DIR") {
        return PathBuf::from(dir);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("vendor/lexurgy")
}

/// Kotlin's `trimMargin()`: strip everything through the first `|` on each
/// line. Used when every non-blank line carries a margin.
fn trim_margin(s: &str) -> Option<String> {
    let lines: Vec<&str> = s.split('\n').filter(|l| !l.trim().is_empty()).collect();
    if lines.is_empty() || !lines.iter().all(|l| l.trim_start().starts_with('|')) {
        return None;
    }
    Some(
        s.split('\n')
            .filter_map(|l| {
                let t = l.trim_start();
                t.strip_prefix('|').map(|rest| rest.to_string()).or({
                    if t.is_empty() {
                        None
                    } else {
                        Some(l.to_string())
                    }
                })
            })
            .collect::<Vec<_>>()
            .join("\n"),
    )
}

fn trim_indent(s: &str) -> String {
    let mut lines: Vec<&str> = s.split('\n').collect();
    if lines.first().is_some_and(|l| l.trim().is_empty()) {
        lines.remove(0);
    }
    if lines.last().is_some_and(|l| l.trim().is_empty()) {
        lines.pop();
    }
    let indent = lines
        .iter()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.len() - l.trim_start().len())
        .min()
        .unwrap_or(0);
    lines
        .iter()
        .map(|l| if l.len() >= indent { &l[indent..] } else { l })
        .collect::<Vec<_>>()
        .join("\n")
}

struct Case {
    file: String,
    line: usize,
    snippet: String,
    input: String,
    expected: String,
}

/// Extract `name("input") shouldBe "expected"` assertions, each resolved to
/// the most recent `val name = lsc(""" ... """)` binding above it.
fn extract_cases(test_dir: &std::path::Path) -> Vec<Case> {
    let mut cases = Vec::new();
    for entry in std::fs::read_dir(test_dir).expect("lexurgy test dir should be readable") {
        let path = entry.unwrap().path();
        if path.extension().is_none_or(|e| e != "kt") {
            continue;
        }
        let src = std::fs::read_to_string(&path).unwrap();
        let file = path.file_name().unwrap().to_string_lossy().into_owned();

        // bindings: (byte offset, name, snippet); a binding whose snippet we
        // can't extract (single-line string, builder call, ...) is recorded
        // as None so assertions on it are skipped instead of resolving to a
        // stale earlier binding
        let mut bindings: Vec<(usize, String, Option<String>)> = Vec::new();
        let mut search_from = 0;
        while let Some(call) = src[search_from..].find("lsc(") {
            let call_at = search_from + call;
            search_from = call_at + 4;
            // binding name: `val <name> =` just before the call
            let before = &src[..call_at];
            let Some(val_at) = before.rfind("val ") else {
                continue;
            };
            let between = &before[val_at + 4..];
            let mut parts = between.split_whitespace();
            let (Some(name), Some("="), None) = (parts.next(), parts.next(), parts.next()) else {
                continue;
            };
            // the triple-quoted snippet
            let after = &src[call_at + 4..];
            let lead: usize = after
                .bytes()
                .take_while(|b| b.is_ascii_whitespace())
                .count();
            let text = after[lead..].strip_prefix("\"\"\"").and_then(|body| {
                let end = body.find("\"\"\"")?;
                let body = body[..end].replace("${'$'}", "$");
                Some(trim_margin(&body).unwrap_or_else(|| trim_indent(&body)))
            });
            bindings.push((call_at, name.to_string(), text));
        }

        // assertions
        let mut offset = 0;
        for (line_number, line) in src.lines().enumerate() {
            let line_at = offset;
            offset += line.len() + 1;
            let Some((call, rest)) = line.trim().split_once("\") shouldBe \"") else {
                continue;
            };
            let Some((name, input)) = call.split_once("(\"") else {
                continue;
            };
            let Some(expected) = rest.split("\"").next() else {
                continue;
            };
            if !name.chars().all(|c| c.is_alphanumeric() || c == '_') || name.is_empty() {
                continue;
            }
            // Kotlin string escapes and templates: too clever, skip.
            if input.contains('\\') || expected.contains('\\') || input.contains('$') {
                continue;
            }
            let Some((_, _, snippet)) = bindings
                .iter()
                .rev()
                .find(|(at, n, _)| *at < line_at && n == name)
            else {
                continue;
            };
            let Some(snippet) = snippet else {
                continue;
            };
            // Expected strings are sometimes written NFD with an explicit
            // `.normalizeCompose()`; lexurgy's output is always composed.
            let expected: String = expected.nfc().collect();
            cases.push(Case {
                file: file.clone(),
                line: line_number + 1,
                snippet: snippet.clone(),
                input: input.to_string(),
                expected,
            });
        }
    }
    cases
}

#[test]
fn outputs_match_lexurgy() {
    let dir = lexurgy_dir();
    let test_dir = dir.join("core/src/test/kotlin/com/meamoria/lexurgy/sc");
    assert!(
        test_dir.is_dir(),
        "no lexurgy checkout at {}; run `git submodule update --init`, \
         or set $LEXURGY_DIR to a lexurgy checkout",
        dir.display()
    );
    let cases = extract_cases(&test_dir);
    assert!(
        cases.len() > 500,
        "expected to extract hundreds of cases, got {}",
        cases.len()
    );

    let mut matched = 0usize;
    let mut compile_errors = 0usize;
    let mut run_errors = 0usize;
    let mut parse_errors = 0usize;
    let mut mismatches: Vec<String> = Vec::new();

    for case in &cases {
        let Ok(statements) = lauturgie::parse(&case.snippet) else {
            if std::env::var("DIFF_DEBUG").is_ok() {
                eprintln!("parse {}:{}", case.file, case.line);
            }
            parse_errors += 1;
            continue;
        };
        let mut changer = match lauturgie::compiler::compile(&statements) {
            Ok(c) => c,
            Err(e) => {
                if std::env::var("DIFF_DEBUG").is_ok() {
                    eprintln!("compile {}:{}: {e}", case.file, case.line);
                }
                compile_errors += 1;
                continue;
            }
        };
        match changer.apply(&case.input) {
            Ok(output) => {
                if output == case.expected {
                    matched += 1;
                } else {
                    mismatches.push(format!(
                        "{}:{}: \"{}\" => \"{}\" (lexurgy: \"{}\")\n---\n{}\n---",
                        case.file, case.line, case.input, output, case.expected, case.snippet
                    ));
                }
            }
            Err(e) => {
                if std::env::var("DIFF_DEBUG").is_ok() {
                    eprintln!("run {}:{}: {e}", case.file, case.line);
                }
                run_errors += 1;
            }
        }
    }

    eprintln!(
        "differential vs lexurgy: {matched} matched, {} mismatched, \
         {compile_errors} compile-unsupported, {run_errors} run-unsupported, \
         {parse_errors} parse-failed, of {} total",
        mismatches.len(),
        cases.len(),
    );

    assert!(
        mismatches.is_empty(),
        "{} of {} runnable cases disagree with lexurgy (showing up to 20):\n\n{}",
        mismatches.len(),
        matched + mismatches.len(),
        mismatches
            .iter()
            .take(20)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n\n")
    );
    assert!(
        matched >= 380,
        "only {matched} cases actually ran and matched; the skip filters are eating everything"
    );
}
