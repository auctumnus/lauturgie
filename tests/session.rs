// SPDX-License-Identifier: GPL-3.0-or-later
//
// Tests for the session path (`CompiledRules::change_with_intermediates`) and
// `rule_names`, the surface the HTTP API (`lauturgie-server`) is built on.
//
// The expectations are ported directly from lexurgy's own test suites:
//   - rule names  -> core `TestRuleNames`
//   - traces      -> core `TestTracing`
//   - the rest    -> api `sc/v1/Scv1Test` (the exact wire behaviour)
// plus a cross-check that the session output equals the trusted `apply` fast
// path (which the differential suite and fuzzer already pin to lexurgy).

use lauturgie::session::ChangeOptions;

fn compile(src: &str) -> lauturgie::compiler::CompiledRules {
    let statements = lauturgie::parse(src).expect("parse failed");
    lauturgie::compiler::compile(&statements).expect("compile failed")
}

fn rule_names(src: &str) -> Vec<String> {
    compile(src).rule_names()
}

fn run(src: &str, words: &[&str], options: ChangeOptions) -> lauturgie::session::ChangeOutput {
    compile(src)
        .change_with_intermediates(words, &options)
        .expect("session run failed")
}

// --- rule_names (lexurgy core TestRuleNames) --------------------------------

#[test]
fn rule_names_deromanizer_and_romanizer() {
    assert_eq!(
        rule_names("deromanizer:\n  a => b\n\nfoo:\n  b => c"),
        ["<deromanizer>", "foo"]
    );
    assert_eq!(
        rule_names("foo:\n  a => b\n\nromanizer:\n  b => c"),
        ["foo", "<romanizer>"]
    );
}

#[test]
fn rule_names_intermediate_romanizer() {
    assert_eq!(
        rule_names("romanizer-foo:\n  a => b\n\nfoo:\n  a => b"),
        ["<romanizer>-foo", "foo"]
    );
}

#[test]
fn rule_names_syllable_rules_counter() {
    assert_eq!(
        rule_names(
            "Syllables:\n    {ba, na}\n\nfoo:\n    a => o\n\n\
             Syllables:\n    {bon, on, o}\n\nSyllables:\n    {bon, no, o}"
        ),
        [
            "<syllables>/<initial>/1",
            "foo",
            "<syllables>/foo/1",
            "<syllables>/foo/2",
        ]
    );
}

#[test]
fn rule_names_cleanup_rules() {
    assert_eq!(
        rule_names("foo cleanup:\n    a => b\n\nbar cleanup:\n    b => c\n\nrule1:\n    c => a"),
        [
            "<cleanup>/<initial>/foo",
            "<cleanup>/<initial>/bar",
            "rule1",
            "<cleanup>/rule1/foo",
            "<cleanup>/rule1/bar",
        ]
    );
}

// --- session output / intermediates / startAt / stopBefore (api Scv1Test) ---

#[test]
fn applies_one_sound_change() {
    let out = run("rule:\no => a", &["foo", "oboe"], ChangeOptions::default());
    assert_eq!(out.rule_names, ["rule"]);
    assert_eq!(out.output_words, ["faa", "abae"]);
}

#[test]
fn intermediate_romanizer_returns_intermediate_forms() {
    let out = run(
        "rule-1:\n o => a\n\nromanizer-intermediate:\nunchanged\n\nrule-2:\n b => d",
        &["foo", "oboe"],
        ChangeOptions::default(),
    );
    assert_eq!(out.rule_names, ["rule-1", "<romanizer>-intermediate", "rule-2"]);
    assert_eq!(out.output_words, ["faa", "adae"]);
    assert_eq!(
        out.intermediate_words,
        [("intermediate".to_string(), vec!["faa".to_string(), "abae".to_string()])]
    );
}

#[test]
fn start_at_skips_earlier_rules() {
    let out = run(
        "a-to-b:\n a => b\n\nb-to-c:\n b => c\n\nc-to-d:\n c => d",
        &["aaa", "bbb", "ccc"],
        ChangeOptions {
            start_at: Some("b-to-c"),
            ..Default::default()
        },
    );
    // ruleNames is always the *full* list, independent of startAt.
    assert_eq!(out.rule_names, ["a-to-b", "b-to-c", "c-to-d"]);
    assert_eq!(out.output_words, ["aaa", "ddd", "ddd"]);
}

#[test]
fn stop_before_drops_later_rules() {
    let out = run(
        "a-to-b:\n a => b\n\nb-to-c:\n b => c\n\nc-to-d:\n c => d",
        &["aaa", "bbb", "ccc"],
        ChangeOptions {
            stop_before: Some("c-to-d"),
            ..Default::default()
        },
    );
    assert_eq!(out.output_words, ["ccc", "ccc", "ccc"]);
}

#[test]
fn unknown_start_at_is_an_error() {
    let err = compile("a-to-b:\n a => b")
        .change_with_intermediates(
            &["aaa"],
            &ChangeOptions {
                start_at: Some("nope"),
                ..Default::default()
            },
        )
        .unwrap_err();
    assert!(matches!(
        err,
        lauturgie::session::SessionError::RuleNotFound { .. }
    ));
}

// --- tracing (api Scv1Test / core TestTracing) ------------------------------

fn traces_of(out: &lauturgie::session::ChangeOutput) -> Vec<(String, Vec<(String, String)>)> {
    out.traces
        .iter()
        .map(|(w, steps)| {
            (
                w.clone(),
                steps.iter().map(|t| (t.rule.clone(), t.output.clone())).collect(),
            )
        })
        .collect()
}

#[test]
fn tracing_one_word() {
    let trace_words = vec!["aaa".to_string()];
    let out = run(
        "a-to-b:\n a => b\n\nb-to-c:\n b => c\n\nc-to-d:\n c => d",
        &["aaa", "bbb", "ccc"],
        ChangeOptions {
            trace_words: &trace_words,
            ..Default::default()
        },
    );
    assert_eq!(out.output_words, ["ddd", "ddd", "ddd"]);
    assert_eq!(
        traces_of(&out),
        [(
            "aaa".to_string(),
            vec![
                ("a-to-b".to_string(), "bbb".to_string()),
                ("b-to-c".to_string(), "ccc".to_string()),
                ("c-to-d".to_string(), "ddd".to_string()),
            ],
        )]
    );
}

#[test]
fn tracing_multiple_words() {
    let trace_words = vec!["aaa".to_string(), "ccc".to_string()];
    let out = run(
        "a-to-b:\n a => b\n\nb-to-c:\n b => c\n\nc-to-d:\n c => d",
        &["aaa", "bbb", "ccc"],
        ChangeOptions {
            trace_words: &trace_words,
            ..Default::default()
        },
    );
    assert_eq!(
        traces_of(&out),
        [
            (
                "aaa".to_string(),
                vec![
                    ("a-to-b".to_string(), "bbb".to_string()),
                    ("b-to-c".to_string(), "ccc".to_string()),
                    ("c-to-d".to_string(), "ddd".to_string()),
                ],
            ),
            (
                "ccc".to_string(),
                vec![("c-to-d".to_string(), "ddd".to_string())],
            ),
        ]
    );
}

// --- per-word failures (api Scv1Test) ---------------------------------------

#[test]
fn rule_failure_reports_word_and_rule() {
    let out = run("foo:\n{f, b} => {m, $1}", &["foo", "bar"], ChangeOptions::default());
    assert_eq!(out.output_words, ["moo", "ERROR"]);
    assert_eq!(out.errors.len(), 1);
    let e = &out.errors[0];
    assert_eq!(e.rule.as_deref(), Some("foo"));
    assert_eq!(e.original_word.as_deref(), Some("bar"));
    assert_eq!(e.current_word.as_deref(), Some("bar"));
}

#[test]
fn syllable_failure_reports_syllable_rule() {
    let out = run("Syllables:\n  x", &["xxx", "foo"], ChangeOptions::default());
    assert_eq!(out.rule_names, ["<syllables>/<initial>/1"]);
    assert_eq!(out.output_words, ["x.x.x", "ERROR"]);
    assert_eq!(out.errors.len(), 1);
    assert_eq!(out.errors[0].rule.as_deref(), Some("<syllables>/<initial>/1"));
}

#[test]
fn diverging_propagate_is_a_per_word_error() {
    // lexurgy times out here; lauturgie's internal budget reports a per-word
    // error instead. Only "bar" contains an `a`, so only it explodes.
    let out = run("explode propagate:\na => aa", &["foo", "bar"], ChangeOptions::default());
    assert_eq!(out.errors.len(), 1);
    assert_eq!(out.output_words[0], "foo");
    assert_eq!(out.output_words[1], "ERROR");
}

// --- the load-bearing guarantee: session output == apply fast path ----------

#[test]
fn session_output_matches_apply_fast_path() {
    let cases: &[(&str, &[&str])] = &[
        ("r0:\na => b\nb => a\n\nr1:\na => b\nb => a", &["ababab", "aaa", "bbb"]),
        (
            "deromanizer:\n a => b\n\nfoo:\n b => c\n\nromanizer:\n c => d",
            &["aaa", "abc", "xyz"],
        ),
        (
            "Syllables:\n {ba,na}\n\nfoo:\n a => o\n\nSyllables:\n {bon,on,o}",
            &["bana", "banabon", "baba"],
        ),
        ("c cleanup:\n x => y\n\nr1:\n a => x\n\nr2:\n b => x", &["aabb", "xab", "ba"]),
        (
            "rule-i:\n o => a\n\nromanizer-mid:\n unchanged\n\nr2:\n b => d",
            &["foo", "oboe", "bob"],
        ),
        ("longer ltr:\n a => e / _ b", &["ab", "aba", "bba"]),
    ];
    for (src, words) in cases {
        let compiled = compile(src);
        let apply_out: Vec<String> = words
            .iter()
            .map(|w| {
                let mut c = compiled.clone();
                c.apply(w).unwrap_or_else(|_| "ERROR".to_string())
            })
            .collect();
        let session = compiled
            .change_with_intermediates(words, &ChangeOptions::default())
            .expect("session run failed");
        assert_eq!(
            apply_out, session.output_words,
            "session disagreed with apply for ruleset starting {:?}",
            src.lines().next().unwrap()
        );
    }
}
