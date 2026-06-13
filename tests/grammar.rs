//! Targeted grammar tests, several ported from lexurgy's Kotlin test suite
//! (TestWhitespace, TestReusable, TestSyllables, ...).

use lauturgie::ast::*;
use lauturgie::{parse, Error};

fn ok(src: &str) -> Vec<Statement> {
    match parse(src) {
        Ok(stmts) => stmts,
        Err(e) => panic!("failed to parse {src:?}: {e}"),
    }
}

fn err(src: &str) -> Error {
    match parse(src) {
        Ok(stmts) => panic!("expected parse failure for {src:?}, got {stmts:?}"),
        Err(e) => e,
    }
}

fn single_rule(src: &str) -> ChangeRule {
    let stmts = ok(src);
    assert_eq!(stmts.len(), 1, "expected one statement: {stmts:?}");
    match stmts.into_iter().next().unwrap() {
        Statement::Rule(r) => r,
        other => panic!("expected rule, got {other:?}"),
    }
}

fn first_expression(src: &str) -> StandardExpression {
    let rule = single_rule(src);
    let BlockElement::Expressions(exprs) = rule.block.first else {
        panic!("expected expressions");
    };
    match exprs.into_iter().next().unwrap() {
        Expression::Standard(e) => e,
        other => panic!("expected standard expression, got {other:?}"),
    }
}

fn text(s: &str) -> Element {
    Element::Text(Text {
        text: s.into(),
        exact: false,
    })
}

#[test]
fn rules_can_break_lines_at_change_arrow() {
    let e = first_expression("rule:\n    a =>\n    b");
    assert_eq!(e.from.element, text("a"));
    assert_eq!(e.to, text("b"));
}

#[test]
fn rules_can_break_lines_at_condition_slash() {
    let e = first_expression("rule:\n    a => b /\n        _ c");
    let envs = e.environment.unwrap().condition.unwrap();
    assert_eq!(envs.len(), 1);
    assert!(envs[0].anchored);
    assert_eq!(envs[0].after, Some(text("c")));
}

#[test]
fn rules_cannot_break_across_blank_lines() {
    err("rule:\n    a =>\n    \n    b");
}

#[test]
fn rules_cannot_break_inside_sequences() {
    err("rule:\n    a\n    b => c");
}

#[test]
fn rules_cannot_break_inside_alternative_lists() {
    err("rule:\n    {a,\n    b} => c");
}

#[test]
fn class_decls_can_break_at_commas_and_open_brace() {
    let stmts = ok("Class letter {\n    a, b, c,\n    d, e, f,\n}\n\nrule:\n    @letter => *");
    assert!(matches!(
        &stmts[0],
        Statement::Class { name, elements } if name == "letter" && elements.len() == 6
    ));
}

#[test]
fn class_decl_trailing_comma_is_optional() {
    let stmts = ok("Class letter {\n    a, b, c,\n    d, e, f\n}");
    assert!(matches!(&stmts[0], Statement::Class { elements, .. } if elements.len() == 6));
}

#[test]
fn comments_are_ignored() {
    let stmts = ok("# leading comment\nrule: # trailing comment\n    a => b # another\n# done");
    assert_eq!(stmts.len(), 1);
}

#[test]
fn feature_decl_forms() {
    let stmts = ok(concat!(
        "Feature Type(cons, vowel)\n",
        "Feature Palatal(*unpalatal, palatal)\n",
        "Feature +long, +nasalized\n",
        "Feature (syllable) stress(*unstressed, primary, secondary)\n",
        "Feature (syllable) +heavy\n",
        "Feature tense",
    ));
    assert_eq!(stmts.len(), 6);
    assert!(matches!(
        &stmts[1],
        Statement::Feature(FeatureDecl::Full { null_alias: Some(a), .. }) if a == "unpalatal"
    ));
    assert!(matches!(
        &stmts[5],
        Statement::Feature(FeatureDecl::Plus(fs)) if fs.len() == 1 && !fs[0].plus
    ));
}

#[test]
fn diacritic_decl_modifier_positions() {
    let stmts = ok(concat!(
        "Diacritic ʰ [aspirated]\n",
        "Diacritic ː (floating) [+long]\n",
        "Diacritic ˈ (before) (floating) [stressed]\n",
        "Diacritic ² [+heavy] (first)",
    ));
    assert!(matches!(
        &stmts[1],
        Statement::Diacritic(d) if d.modifiers == vec![DiacriticModifier::Floating]
    ));
    assert!(matches!(
        &stmts[2],
        Statement::Diacritic(d)
            if d.modifiers == vec![DiacriticModifier::Before, DiacriticModifier::Floating]
    ));
    assert!(matches!(
        &stmts[3],
        Statement::Diacritic(d) if d.modifiers == vec![DiacriticModifier::First]
    ));
}

#[test]
fn symbol_decl_forms() {
    let stmts = ok("Symbol ts, dz, tɬ\nSymbol p [stop unvoiced labial]");
    assert!(matches!(&stmts[0], Statement::Symbol { names, matrix: None } if names.len() == 3));
    assert!(matches!(
        &stmts[1],
        Statement::Symbol { names, matrix: Some(m) } if names.len() == 1 && m.len() == 3
    ));
}

#[test]
fn element_decl() {
    let stmts = ok("Element fooOrBar {[+foo], [+bar]}\nElement baz @fooOrBar [+baz]");
    assert!(matches!(&stmts[0], Statement::Element { name, .. } if name == "fooOrBar"));
    assert!(matches!(
        &stmts[1],
        Statement::Element { element, .. }
            if matches!(&element.element, Element::Sequence(s) if s.len() == 2)
    ));
}

#[test]
fn rule_modifiers_and_hyphenated_names() {
    let rule = single_rule("vowel-harmony-2 [vowel] rtl propagate:\n    a => e");
    assert_eq!(rule.name, "vowel-harmony-2");
    assert_eq!(rule.modifiers.len(), 3);
    assert!(rule.modifiers.contains(&RuleModifier::Rtl));
}

#[test]
fn keyword_named_rules() {
    // Rule names that collide with keywords still parse when followed by `:`.
    for kw in ["feature", "class", "symbol", "element", "then", "off"] {
        let rule = single_rule(&format!("{kw}:\n    a => b"));
        assert_eq!(rule.name, kw);
    }
}

#[test]
fn rules_named_with_romanizer_keywords() {
    // From lexurgy's TestRobustness: a rule whose name is built out of
    // declaration keywords must not be mistaken for a romanizer.
    let rule =
        single_rule("deromanizer-romanizer-literal:\n    {deromanizer, romanizer} => literal");
    assert_eq!(rule.name, "deromanizer-romanizer-literal");

    // ...but `romanizer-x defer:` is a change rule (inter-romanizers take no
    // modifiers), while `romanizer-x:` is an inter-romanizer.
    let stmts = ok("romanizer-x defer:\n    a => b");
    assert!(matches!(&stmts[0], Statement::Rule(r) if r.name == "romanizer-x"));
}

#[test]
fn expressions_starting_with_then_keyword() {
    // From lexurgy's TestRobustness: `then` here is text, not a block type.
    let rule = single_rule("then-else:\n    then => else");
    assert_eq!(rule.name, "then-else");
    let BlockElement::Expressions(exprs) = &rule.block.first else {
        panic!()
    };
    let Expression::Standard(e) = &exprs[0] else {
        panic!()
    };
    assert_eq!(e.from.element, text("then"));
    // And a rule named `then-2` directly after another rule stays a rule.
    let stmts = ok("first:\n    a => b\n\nthen-2:\n    b => c");
    assert!(matches!(&stmts[1], Statement::Rule(r) if r.name == "then-2"));
}

#[test]
fn syllable_pattern_environment_before_arrow() {
    // From lexurgy's TestSyllables: `@cons? @vowel / _ $ => [heavy]`.
    let stmts = ok("Syllables:\n    @cons? @vowel / _ $ => [heavy]\n    @cons? @vowel => [light]");
    let Statement::Syllables(SyllableSpec::Patterns(pats)) = &stmts[0] else {
        panic!("{stmts:?}")
    };
    let SyllablePattern::Plain(re) = &pats[0].pattern else {
        panic!()
    };
    assert!(re.environment.is_some());
    assert!(pats[0].assign.is_some());
}

#[test]
fn deferred_rules_and_block_refs() {
    let stmts = ok("foo defer:\n    f => b\n\nreal-rule:\n    f => x\n    :foo");
    let Statement::Rule(foo) = &stmts[0] else {
        panic!()
    };
    assert!(foo.modifiers.contains(&RuleModifier::Defer));
    let Statement::Rule(real) = &stmts[1] else {
        panic!()
    };
    let BlockElement::Expressions(exprs) = &real.block.first else {
        panic!()
    };
    assert_eq!(exprs[1], Expression::BlockRef("foo".into()));
}

#[test]
fn then_and_else_blocks() {
    let rule = single_rule("rule:\n    a => b\n    Then:\n    b => c\n    Else:\n    c => d");
    assert_eq!(rule.block.rest.len(), 2);
    assert_eq!(rule.block.rest[0].0.kind, BlockKind::Then);
    assert_eq!(rule.block.rest[1].0.kind, BlockKind::Else);
}

#[test]
fn nested_parenthesized_blocks() {
    let rule = single_rule(concat!(
        "rule:\n",
        "    (\n",
        "        a => b\n",
        "        Else:\n",
        "        a => c\n",
        "    )\n",
        "    Then:\n",
        "    b => d",
    ));
    let BlockElement::Nested(inner) = &rule.block.first else {
        panic!()
    };
    assert_eq!(inner.rest.len(), 1);
    assert_eq!(inner.rest[0].0.kind, BlockKind::Else);
    assert_eq!(rule.block.rest.len(), 1);
}

#[test]
fn unchanged_and_off_expressions() {
    let stmts = ok("Romanizer-x:\n    unchanged\n\nrule:\n    off");
    assert!(matches!(
        &stmts[0],
        Statement::InterRomanizer { block, .. }
            if block.first == BlockElement::Expressions(vec![Expression::Unchanged])
    ));
    assert!(matches!(
        &stmts[1],
        Statement::Rule(r)
            if r.block.first == BlockElement::Expressions(vec![Expression::Off])
    ));
}

#[test]
fn romanizer_forms() {
    let stmts = ok(concat!(
        "Deromanizer:\n    x => ç\n\n",
        "Deromanizer literal:\n    c => k\n\n",
        "Romanizer-stage-1:\n    ç => x\n\n",
        "Romanizer:\n    ə => y",
    ));
    assert!(matches!(
        stmts[0],
        Statement::Deromanizer { literal: false, .. }
    ));
    assert!(matches!(
        stmts[1],
        Statement::Deromanizer { literal: true, .. }
    ));
    assert!(matches!(&stmts[2], Statement::InterRomanizer { name, .. } if name == "stage-1"));
    assert!(matches!(
        stmts[3],
        Statement::Romanizer { literal: false, .. }
    ));
}

#[test]
fn captures_repeaters_and_intersections() {
    let e =
        first_expression("rule:\n    [stop]$1 {a, b}* c+ d? e*2 f*(1-3) <syl>&[+heavy]&!x => $1");
    let Element::Sequence(seq) = e.from.element else {
        panic!()
    };
    assert_eq!(seq.len(), 7);
    assert!(matches!(&seq[0], Element::Capture { capture, .. } if capture.number == 1));
    assert!(matches!(
        &seq[1],
        Element::Repeat { kind: RepeaterKind::ZeroOrMore, element }
            if matches!(element.as_ref(), Element::List(items) if items.len() == 2)
    ));
    assert!(matches!(
        &seq[2],
        Element::Repeat {
            kind: RepeaterKind::AtLeastOne,
            ..
        }
    ));
    assert!(matches!(
        &seq[3],
        Element::Repeat {
            kind: RepeaterKind::Optional,
            ..
        }
    ));
    assert!(matches!(
        &seq[4],
        Element::Repeat {
            kind: RepeaterKind::Count(2),
            ..
        }
    ));
    assert!(matches!(
        &seq[5],
        Element::Repeat {
            kind: RepeaterKind::Range {
                min: Some(1),
                max: Some(3)
            },
            ..
        }
    ));
    let Element::Interfix { first, rest } = &seq[6] else {
        panic!("expected interfix, got {:?}", seq[6]);
    };
    assert_eq!(**first, Element::AnySyllable);
    assert_eq!(rest[0].0, InterfixKind::Intersection);
    assert_eq!(rest[1].0, InterfixKind::IntersectionNot);
    assert_eq!(
        e.to,
        Element::CaptureRef(CaptureRef {
            inexact: false,
            syllable: false,
            number: 1
        })
    );
}

#[test]
fn capture_variants() {
    let e = first_expression("rule:\n    a$1 => ~$1 $.1");
    let Element::Sequence(to) = e.to else {
        panic!()
    };
    assert_eq!(
        to[0],
        Element::CaptureRef(CaptureRef {
            inexact: true,
            syllable: false,
            number: 1
        })
    );
    assert_eq!(
        to[1],
        Element::CaptureRef(CaptureRef {
            inexact: false,
            syllable: true,
            number: 1
        })
    );
}

#[test]
fn negation_and_exact_text() {
    let e = first_expression("rule:\n    !{e!, o!} => x");
    let Element::Negated(inner) = e.from.element else {
        panic!()
    };
    let Element::List(items) = *inner else {
        panic!()
    };
    assert_eq!(
        items[0],
        ListItem::Element(RuleElement::from(Element::Text(Text {
            text: "e".into(),
            exact: true
        })))
    );
}

#[test]
fn boundaries_and_special_elements() {
    let e = first_expression("rule:\n    * => ʔ / $ [vowel] _ . $$");
    assert_eq!(e.from.element, Element::Empty);
    let envs = e.environment.unwrap().condition.unwrap();
    let env = &envs[0];
    assert!(matches!(
        env.before.as_ref().unwrap(),
        Element::Sequence(s) if s[0] == Element::WordBoundary
    ));
    assert!(matches!(
        env.after.as_ref().unwrap(),
        Element::Sequence(s) if s == &[Element::SyllableBoundary, Element::BetweenWords]
    ));
}

#[test]
fn fancy_matrix_values() {
    let e = first_expression("rule:\n    [cons +vcd -asp !lab !*pal $Place] => []");
    let Element::Matrix(m) = e.from.element else {
        panic!()
    };
    assert_eq!(m.len(), 6);
    assert_eq!(m[0].value, MatrixValueKind::Simple("cons".into()));
    assert_eq!(m[1].value, MatrixValueKind::Plus("vcd".into()));
    assert_eq!(m[2].value, MatrixValueKind::Minus("asp".into()));
    assert!(m[3].negated);
    assert_eq!(m[3].value, MatrixValueKind::Simple("lab".into()));
    assert!(m[4].negated);
    assert_eq!(m[4].value, MatrixValueKind::Absent("pal".into()));
    assert_eq!(m[5].value, MatrixValueKind::Variable("Place".into()));
    assert_eq!(e.to, Element::Matrix(vec![]));
}

#[test]
fn escapes_in_text() {
    let e = first_expression("rule:\n    \\+x => a\\\\b");
    assert_eq!(e.from.element, text("+x"));
    assert_eq!(e.to, text("a\\b"));
}

#[test]
fn environment_lists_unpack() {
    let e = first_expression("rule:\n    a => b / {_ x, y _, $ _ z} // {p _, _ q}");
    let env = e.environment.unwrap();
    assert_eq!(env.condition.as_ref().unwrap().len(), 3);
    assert_eq!(env.exclusion.as_ref().unwrap().len(), 2);
}

#[test]
fn plain_alternative_list_in_environment() {
    // `{j, w} _`: braces are an alternative list inside one environment.
    let e = first_expression("rule:\n    u => o // {j, w} _");
    let envs = e.environment.unwrap().exclusion.unwrap();
    assert_eq!(envs.len(), 1);
    assert!(matches!(envs[0].before.as_ref().unwrap(), Element::List(items) if items.len() == 2));
}

#[test]
fn anchor_free_environment() {
    // An environment that is just an element (no `_`).
    let e = first_expression("rule:\n    a => b / c");
    let envs = e.environment.unwrap().condition.unwrap();
    assert!(!envs[0].anchored);
    assert_eq!(envs[0].before, Some(text("c")));
}

#[test]
fn environment_on_the_from_element() {
    // `ruleElement` may carry its own environment before `=>`.
    let e = first_expression("rule:\n    (a / x _) => b");
    let Element::Group(g) = e.from.element else {
        panic!()
    };
    assert!(g.environment.is_some());
}

#[test]
fn anchored_item_outside_environment_is_rejected() {
    let e = err("rule:\n    {a _, b} => c");
    assert!(matches!(e, Error::Invalid(_)), "got {e:?}");
}

#[test]
fn syllable_modes() {
    let stmts = ok("Syllables:\n    explicit\n\nrule:\n    a => b\n\nSyllables:\n    clear");
    assert_eq!(stmts[0], Statement::Syllables(SyllableSpec::Explicit));
    assert_eq!(stmts[2], Statement::Syllables(SyllableSpec::Clear));
}

#[test]
fn structured_syllable_patterns() {
    let stmts = ok("Syllables:\n    s ?: cc? :: vv :: cc? / _ x");
    let Statement::Syllables(SyllableSpec::Patterns(pats)) = &stmts[0] else {
        panic!("{stmts:?}")
    };
    let SyllablePattern::Structured {
        reluctant_onset,
        parts,
    } = &pats[0].pattern
    else {
        panic!("{:?}", pats[0].pattern)
    };
    assert_eq!(reluctant_onset.as_ref(), Some(&text("s")));
    assert_eq!(parts.len(), 3);
    assert!(pats[0].environment.is_some());
}

#[test]
fn syllable_pattern_with_matrix() {
    let stmts = ok("Syllables:\n    @cons? @vowel @cons => [+heavy]\n    @cons? @vowel");
    let Statement::Syllables(SyllableSpec::Patterns(pats)) = &stmts[0] else {
        panic!("{stmts:?}")
    };
    assert_eq!(pats.len(), 2);
    assert!(pats[0].assign.is_some());
    assert!(pats[1].assign.is_none());
}

#[test]
fn top_level_bare_expression() {
    // The ANTLR grammar admits a bare standardExpression as a statement.
    let stmts = ok("a => b");
    assert!(matches!(&stmts[0], Statement::Expression(_)));
}

#[test]
fn empty_and_blank_files() {
    assert_eq!(ok("").len(), 0);
    assert_eq!(ok("\n\n   \n").len(), 0);
    assert_eq!(ok("# just a comment\n").len(), 0);
}

#[test]
fn rule_without_colon() {
    // `RULE_START?`: the colon after a rule header is optional.
    let rule = single_rule("rule\n    a => b");
    assert_eq!(rule.name, "rule");
}

#[test]
fn number_out_of_range_is_an_error() {
    err("rule:\n    a*99999999999999999999 => b");
}
