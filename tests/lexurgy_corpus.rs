//! Parse every inline `.lsc` snippet from lexurgy's Kotlin test suite.
//!
//! This extracts the `lsc(""" ... """)` blocks from
//! `core/src/test/kotlin/com/meamoria/lexurgy/sc/*.kt` in a lexurgy checkout
//! and asserts that everything lexurgy parses, we parse; and that snippets
//! lexurgy rejects with `LscNotParsable`, we reject too.
//!
//! The checkout is the pinned `vendor/lexurgy` git submodule, overridable via
//! `$LEXURGY_DIR`. If the resolved checkout isn't present the test fails hard
//! (run `git submodule update --init` to fix), so the corpus always runs.

use std::path::{Path, PathBuf};

struct Snippet {
    file: String,
    expect_parse_failure: bool,
    text: String,
}

/// Locate the lexurgy checkout: `$LEXURGY_DIR` if set, otherwise the pinned
/// `vendor/lexurgy` submodule.
fn lexurgy_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("LEXURGY_DIR") {
        return PathBuf::from(dir);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("vendor/lexurgy")
}

/// Apply Kotlin's `trimIndent()`: drop blank first/last lines, strip the
/// common indentation.
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

/// Apply Kotlin's `trimMargin()` if the snippet uses `|` margins.
fn trim_margin(s: &str) -> String {
    let uses_margin = s.lines().any(|l| l.trim_start().starts_with('|'));
    if !uses_margin {
        return s.to_string();
    }
    s.lines()
        .map(|l| {
            let t = l.trim_start();
            t.strip_prefix('|').unwrap_or(l)
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn extract_snippets(test_dir: &Path) -> Vec<Snippet> {
    let mut snippets = Vec::new();
    for entry in std::fs::read_dir(test_dir).expect("lexurgy test dir should be readable") {
        let path = entry.unwrap().path();
        if path.extension().is_none_or(|e| e != "kt") {
            continue;
        }
        let src = std::fs::read_to_string(&path).unwrap();
        let file = path.file_name().unwrap().to_string_lossy().into_owned();
        let mut search_from = 0;
        while let Some(call) = src[search_from..].find("lsc(") {
            let call_at = search_from + call;
            search_from = call_at + 4;
            // The triple-quoted string must follow almost immediately.
            let after = &src[call_at + 4..];
            let lead: usize = after
                .bytes()
                .take_while(|b| b.is_ascii_whitespace())
                .count();
            let Some(body) = after[lead..].strip_prefix("\"\"\"") else {
                continue;
            };
            let Some(end) = body.find("\"\"\"") else {
                continue;
            };
            let raw = &body[..end];
            // Kotlin escapes a literal `$` in templates as ${'$'}.
            let text = trim_indent(&trim_margin(&raw.replace("${'$'}", "$")));
            let mut context_start = call_at.saturating_sub(200);
            while !src.is_char_boundary(context_start) {
                context_start -= 1;
            }
            let context = &src[context_start..call_at];
            snippets.push(Snippet {
                file: file.clone(),
                expect_parse_failure: context.contains("LscNotParsable"),
                text,
            });
        }
    }
    snippets
}

#[test]
fn lexurgy_test_suite_snippets() {
    let dir = lexurgy_dir();
    let test_dir = dir.join("core/src/test/kotlin/com/meamoria/lexurgy/sc");
    assert!(
        test_dir.is_dir(),
        "no lexurgy checkout at {}; run `git submodule update --init`, \
         or set $LEXURGY_DIR to a lexurgy checkout",
        dir.display()
    );
    let snippets = extract_snippets(&test_dir);
    assert!(
        snippets.len() > 300,
        "expected to extract hundreds of snippets, got {}",
        snippets.len()
    );

    let mut failures = Vec::new();
    for (i, s) in snippets.iter().enumerate() {
        let result = lauturgie::parse(&s.text);
        match (s.expect_parse_failure, result) {
            (false, Err(e)) => failures.push(format!(
                "#{i} ({}) failed to parse: {e}\n---\n{}\n---",
                s.file, s.text
            )),
            (true, Ok(_)) => failures.push(format!(
                "#{i} ({}) parsed but lexurgy rejects it:\n---\n{}\n---",
                s.file, s.text
            )),
            _ => {}
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} snippets disagree with lexurgy:\n{}",
        failures.len(),
        snippets.len(),
        failures.join("\n\n")
    );
}
