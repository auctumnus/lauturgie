//! Parse complete real-world sound change files from lexurgy's own
//! `cli/test/*.lsc`, read at runtime from the `vendor/lexurgy` submodule
//! (override with `$LEXURGY_DIR`) so no GPL-3 source is vendored into this
//! tree. Run `git submodule update --init` if the checkout is missing.

use lauturgie::ast::*;
use lauturgie::parse;
use std::path::PathBuf;

/// Read `cli/test/{name}.lsc` from the lexurgy checkout.
fn read_example(name: &str) -> String {
    let dir = match std::env::var("LEXURGY_DIR") {
        Ok(d) => PathBuf::from(d),
        Err(_) => PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("vendor/lexurgy"),
    };
    let path = dir.join("cli/test").join(format!("{name}.lsc"));
    std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "cannot read {} ({e}); run `git submodule update --init`, \
             or set $LEXURGY_DIR to a lexurgy checkout",
            path.display()
        )
    })
}

fn parse_ok(name: &str, src: &str) -> Vec<Statement> {
    match parse(src) {
        Ok(stmts) => stmts,
        Err(e) => panic!("{name} failed to parse: {e}"),
    }
}

#[test]
fn unicode() {
    let stmts = parse_ok("unicode", &read_example("unicode"));
    assert_eq!(stmts.len(), 1);
    let Statement::Rule(rule) = &stmts[0] else {
        panic!("expected a rule, got {:?}", stmts[0]);
    };
    assert_eq!(rule.name, "ph");
}

#[test]
fn kharulian() {
    let stmts = parse_ok("kharulian", &read_example("kharulian"));
    // 11 features + 3 diacritics + 36 symbols + 1 class + deromanizer
    // + 26 rules + romanizer (counts verified against the raw file)
    assert_eq!(stmts.len(), 79);
    assert!(
        matches!(&stmts[0], Statement::Feature(FeatureDecl::Full { name, .. }) if name == "Type")
    );
    assert!(stmts
        .iter()
        .any(|s| matches!(s, Statement::Deromanizer { .. })));
    assert!(stmts
        .iter()
        .any(|s| matches!(s, Statement::Romanizer { .. })));
    let rules: Vec<_> = stmts
        .iter()
        .filter_map(|s| match s {
            Statement::Rule(r) => Some(r),
            _ => None,
        })
        .collect();
    assert_eq!(rules.len(), 26);
    // `stress [vowel]:` has a matrix filter
    let stress = rules.iter().find(|r| r.name == "stress").unwrap();
    assert!(matches!(
        stress.modifiers.as_slice(),
        [RuleModifier::Filter(Element::Matrix(_))]
    ));
    // `boundary-vowels [vowel]:` contains two `Then:` continuations
    let bv = rules.iter().find(|r| r.name == "boundary-vowels").unwrap();
    assert_eq!(bv.block.rest.len(), 2);
    assert!(bv
        .block
        .rest
        .iter()
        .all(|(bt, _)| bt.kind == BlockKind::Then));
}

#[test]
fn syllabian() {
    let stmts = parse_ok("syllabian", &read_example("syllabian"));

    // `Feature +long, +nasalized`
    assert!(matches!(
        &stmts[0],
        Statement::Feature(FeatureDecl::Plus(fs))
            if fs.len() == 2 && fs[0].name == "long" && fs[0].plus && !fs[0].syllable
    ));
    // `Feature (syllable) stress(*unstressed, primary, secondary)`
    assert!(matches!(
        &stmts[1],
        Statement::Feature(FeatureDecl::Full { syllable: true, name, null_alias: Some(alias), values })
            if name == "stress" && alias == "unstressed" && values.len() == 2
    ));
    // `Feature (syllable) +heavy`
    assert!(matches!(
        &stmts[2],
        Statement::Feature(FeatureDecl::Plus(fs)) if fs.len() == 1 && fs[0].syllable
    ));

    // Three `Syllables:` declarations: patterns, patterns, clear.
    let sylls: Vec<_> = stmts
        .iter()
        .filter_map(|s| match s {
            Statement::Syllables(spec) => Some(spec),
            _ => None,
        })
        .collect();
    assert_eq!(sylls.len(), 3);
    assert!(matches!(&sylls[0], SyllableSpec::Patterns(p) if p.len() == 2));
    assert!(matches!(&sylls[1], SyllableSpec::Patterns(p) if p.len() == 1));
    assert_eq!(sylls[2], &SyllableSpec::Clear);

    // First syllable pattern assigns `[+heavy]`.
    let SyllableSpec::Patterns(pats) = &sylls[0] else {
        unreachable!()
    };
    assert!(matches!(
        pats[0].assign.as_deref(),
        Some([MatrixValue { negated: false, value: MatrixValueKind::Plus(n) }]) if n == "heavy"
    ));

    // `Romanizer-before-syncope:` with body `unchanged`.
    let inter: Vec<_> = stmts
        .iter()
        .filter_map(|s| match s {
            Statement::InterRomanizer { name, block, .. } => Some((name, block)),
            _ => None,
        })
        .collect();
    assert_eq!(inter.len(), 2);
    assert_eq!(inter[0].0, "before-syncope");
    assert_eq!(
        inter[0].1.first,
        BlockElement::Expressions(vec![Expression::Unchanged])
    );

    // `stress:` uses nested parenthesized blocks with `Else:` inside.
    let stress = stmts
        .iter()
        .find_map(|s| match s {
            Statement::Rule(r) if r.name == "stress" => Some(r),
            _ => None,
        })
        .unwrap();
    let BlockElement::Nested(first) = &stress.block.first else {
        panic!("expected nested block: {:?}", stress.block.first);
    };
    assert_eq!(first.rest.len(), 2);
    assert!(first.rest.iter().all(|(bt, _)| bt.kind == BlockKind::Else));
    // `Then propagate:` continuation
    assert!(stress.block.rest.iter().any(
        |(bt, _)| bt.kind == BlockKind::Then && bt.modifiers.contains(&RuleModifier::Propagate)
    ));
}

#[test]
fn muipidan() {
    let stmts = parse_ok("muipidan", &read_example("muipidan"));
    // `first-syllable-stress [vowel]:` is `[] => [str] / $ _ [unstr]+ $`
    let fss = stmts
        .iter()
        .find_map(|s| match s {
            Statement::Rule(r) if r.name == "first-syllable-stress" => Some(r),
            _ => None,
        })
        .unwrap();
    let BlockElement::Expressions(exprs) = &fss.block.first else {
        panic!()
    };
    let Expression::Standard(expr) = &exprs[0] else {
        panic!()
    };
    assert_eq!(expr.from.element, Element::Matrix(vec![]));
    let envs = expr
        .environment
        .as_ref()
        .unwrap()
        .condition
        .as_ref()
        .unwrap();
    assert_eq!(envs.len(), 1);
    assert!(envs[0].anchored);
    assert_eq!(envs[0].before, Some(Element::WordBoundary));

    // `vowel-harmony-by-height [vowel] propagate:` is filter + propagate
    let vh = stmts
        .iter()
        .find_map(|s| match s {
            Statement::Rule(r) if r.name == "vowel-harmony-by-height" => Some(r),
            _ => None,
        })
        .unwrap();
    assert_eq!(vh.modifiers.len(), 2);
    assert!(vh.modifiers.contains(&RuleModifier::Propagate));
}

#[test]
fn nitherwe() {
    let stmts = parse_ok("nitherwe", &read_example("nitherwe"));
    assert!(stmts.len() > 10);
    // Environment list: `{$ {@cons, j, w}* _, _ $}`-style conditions parse
    // into multiple environments.
    let vs = stmts
        .iter()
        .find_map(|s| match s {
            Statement::Rule(r) if r.name == "vowel-shift" => Some(r),
            _ => None,
        })
        .unwrap();
    let BlockElement::Expressions(exprs) = &vs.block.first else {
        panic!()
    };
    let Expression::Standard(first) = &exprs[0] else {
        panic!()
    };
    let envs = first
        .environment
        .as_ref()
        .unwrap()
        .condition
        .as_ref()
        .unwrap();
    assert_eq!(envs.len(), 2, "environment list should unpack: {envs:?}");
}
