// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) Autumn <auctumnus@pm.me>

//! Post-parse validation.
//!
//! The grammar parses braced lists as a cover that admits anchored
//! environment items (`{a _, b _}`), because only the surrounding context
//! decides whether braces mean "alternatives" or "environment list". The
//! parser reinterprets the cover right after `/` and `//`; any anchored item
//! that survives in element position is a syntax error the ANTLR grammar
//! would also have rejected.

use crate::parser::ast::*;

pub fn validate(statements: &[Statement]) -> Result<(), String> {
    check_statement_order(statements)?;
    for statement in statements {
        check_statement(statement)?;
    }
    Ok(())
}

/// Statement kinds have fixed pipeline positions (lexurgy's
/// `validateOrder` / `allowedStatementPositions`): features, then
/// diacritics/symbols, then classes/elements, then the deromanizer, then
/// rules (syllable declarations and intermediate romanizers may interleave
/// with them), then the final romanizer. Bare top-level expressions are
/// exempt (they get their own "rule needs a name" error later).
fn check_statement_order(statements: &[Statement]) -> Result<(), String> {
    fn position(s: &Statement) -> Option<(u32, &'static str)> {
        Some(match s {
            Statement::Feature(_) => (0, "feature declarations"),
            Statement::Diacritic(_) => (10, "diacritic declarations"),
            Statement::Symbol { .. } => (10, "symbol declarations"),
            Statement::Class { .. } => (30, "class declarations"),
            Statement::Element { .. } => (30, "element declarations"),
            Statement::Deromanizer { .. } => (40, "deromanizer"),
            Statement::Syllables(_) => (50, "syllable declarations"),
            Statement::Rule(_) => (50, "change rules"),
            Statement::InterRomanizer { .. } => (50, "intermediate romanizers"),
            Statement::Romanizer { .. } => (60, "final romanizer"),
            Statement::Expression(_) => return None,
        })
    }
    for pair in statements
        .iter()
        .filter_map(position)
        .collect::<Vec<_>>()
        .windows(2)
    {
        let ((prev, prev_name), (next, next_name)) = (pair[0], pair[1]);
        if prev > next {
            return Err(format!("the {prev_name} must come after the {next_name}"));
        }
    }
    Ok(())
}

fn check_statement(statement: &Statement) -> Result<(), String> {
    match statement {
        Statement::Feature(_) | Statement::Diacritic(_) | Statement::Symbol { .. } => Ok(()),
        Statement::Class { .. } => Ok(()),
        Statement::Element { element, .. } => check_rule_element(element, true),
        Statement::Syllables(SyllableSpec::Patterns(patterns)) => {
            for p in patterns {
                match &p.pattern {
                    SyllablePattern::Plain(re) => check_rule_element(re, true)?,
                    SyllablePattern::Structured {
                        reluctant_onset,
                        parts,
                    } => {
                        if let Some(o) = reluctant_onset {
                            check_element(o, true)?;
                        }
                        for part in parts {
                            check_element(part, true)?;
                        }
                    }
                }
                check_env_opt(&p.environment, true)?;
            }
            Ok(())
        }
        Statement::Syllables(_) => Ok(()),
        Statement::Deromanizer { block, .. }
        | Statement::Romanizer { block, .. }
        | Statement::InterRomanizer { block, .. } => check_block(block, true),
        Statement::Rule(rule) => {
            // A deferred rule is a template, validated only where `:name`
            // splices it (`check_deferred_rule`, called from the compiler's
            // splice lowering). Kotlin compiles a `defer`d rule lazily, so a
            // never-spliced deferred rule with an invalid body (e.g. a
            // peripheral repeater) is accepted, not rejected — see the matching
            // comment in `compiler::lower`. `check_statement_order` still
            // accounts for its statement position above.
            // A deferred rule still gets its *eager* checks now (transforming
            // `>`, structural) but skips the peripheral-repeater check, which
            // lexurgy defers to compile-at-splice — so an unspliced deferred
            // rule with only a peripheral repeater is accepted, and it
            // re-validates fully (peripheral included) at each `:name` splice
            // via `check_deferred_rule` called from `compiler::lower`.
            let periph = !rule.modifiers.contains(&RuleModifier::Defer);
            check_deferred_rule(rule, periph)
        }
        Statement::Expression(e) => check_expression_body(e, true),
    }
}

/// Structural validation of a single change rule's body (filters + block).
/// `periph` enables the peripheral-repeater check (a lexurgy compile-time
/// check): `true` for ordinary rules and for a deferred rule at its `:name`
/// splice (the compiler calls this from `lower_block_element` /
/// `lower_expression_into`), `false` for the eager pass over an unspliced
/// deferred rule. Eager-only checks (transforming `>`, structural) run either
/// way, mirroring kotlin's parse-time vs compile-at-splice split.
pub fn check_deferred_rule(rule: &ChangeRule, periph: bool) -> Result<(), String> {
    for m in &rule.modifiers {
        if let RuleModifier::Filter(e) = m {
            check_element(e, periph)?;
        }
    }
    check_block(&rule.block, periph)
}

/// Reject a transforming `>` interfix anywhere in a deferred rule. lexurgy
/// throws `LscFutureStructure("Transforming elements")` for *every* rule
/// including unspliced deferred ones, and it is a compile-stage error (not a
/// parse failure). The compiler lowers ordinary and spliced rules and so
/// rejects their `>` already; an unspliced deferred rule is never lowered, so
/// `compiler::lower` calls this on each deferred rule to keep reject-parity.
pub fn reject_deferred_transforming(rule: &ChangeRule) -> Result<(), String> {
    fn err() -> String {
        "transforming interfix (>) is not yet implemented".to_string()
    }
    fn el(e: &Element) -> Result<(), String> {
        match e {
            Element::Interfix { first, rest } => {
                if rest.iter().any(|(k, _)| *k == InterfixKind::Transforming) {
                    return Err(err());
                }
                el(first)?;
                for (_, e) in rest {
                    el(e)?;
                }
                Ok(())
            }
            Element::Sequence(items) => items.iter().try_for_each(el),
            Element::Group(re) => re_(re),
            Element::List(items) => items.iter().try_for_each(|it| match it {
                ListItem::Element(re) => re_(re),
                ListItem::Env(_) => Ok(()),
            }),
            Element::Negated(b)
            | Element::Capture { element: b, .. }
            | Element::Repeat { element: b, .. } => el(b),
            _ => Ok(()),
        }
    }
    fn re_(re: &RuleElement) -> Result<(), String> {
        el(&re.element)?;
        env_opt(&re.environment)
    }
    fn env_opt(env: &Option<CompoundEnvironment>) -> Result<(), String> {
        if let Some(env) = env {
            for list in [&env.condition, &env.exclusion].into_iter().flatten() {
                for e in list {
                    if let Some(b) = &e.before {
                        el(b)?;
                    }
                    if let Some(a) = &e.after {
                        el(a)?;
                    }
                }
            }
        }
        Ok(())
    }
    fn block(b: &Block) -> Result<(), String> {
        block_el(&b.first)?;
        b.rest.iter().try_for_each(|(_, e)| block_el(e))
    }
    fn block_el(be: &BlockElement) -> Result<(), String> {
        match be {
            BlockElement::Expressions(exprs) => exprs.iter().try_for_each(|x| match x {
                Expression::Standard(s) => {
                    re_(&s.from)?;
                    el(&s.to)?;
                    env_opt(&s.environment)
                }
                _ => Ok(()),
            }),
            BlockElement::Nested(b) => block(b),
        }
    }
    for m in &rule.modifiers {
        if let RuleModifier::Filter(e) = m {
            el(e)?;
        }
    }
    block(&rule.block)
}

fn check_block(block: &Block, periph: bool) -> Result<(), String> {
    check_block_element(&block.first, periph)?;
    for (_, el) in &block.rest {
        check_block_element(el, periph)?;
    }
    Ok(())
}

fn check_block_element(el: &BlockElement, periph: bool) -> Result<(), String> {
    match el {
        BlockElement::Expressions(exprs) => {
            for e in exprs {
                if let Expression::Standard(s) = e {
                    check_expression_body(s, periph)?;
                }
            }
            Ok(())
        }
        BlockElement::Nested(b) => check_block(b, periph),
    }
}

fn check_expression_body(e: &StandardExpression, periph: bool) -> Result<(), String> {
    check_rule_element(&e.from, periph)?;
    check_element(&e.to, periph)?;
    check_env_opt(&e.environment, periph)
}

fn check_rule_element(re: &RuleElement, periph: bool) -> Result<(), String> {
    check_element(&re.element, periph)?;
    check_env_opt(&re.environment, periph)
}

fn check_env_opt(env: &Option<CompoundEnvironment>, periph: bool) -> Result<(), String> {
    if let Some(env) = env {
        for list in [&env.condition, &env.exclusion].into_iter().flatten() {
            for e in list {
                check_environment(e, periph)?;
            }
        }
    }
    Ok(())
}

// `periph` gates the peripheral-repeater check: it is a compile-time check in
// lexurgy (thrown when building the matcher), so for a deferred rule's eager
// pass it is skipped (the rule re-validates with `periph = true` at its splice).
fn check_environment(env: &Environment, periph: bool) -> Result<(), String> {
    if let Some(b) = &env.before {
        check_element(b, periph)?;
        // The edge *away* from `_` (lexurgy's `rightBeforeAnchor` context).
        if periph {
            check_peripheral(b, true, false)?;
        }
    }
    if let Some(a) = &env.after {
        check_element(a, periph)?;
        if periph {
            check_peripheral(a, false, true)?;
        }
    }
    Ok(())
}

/// Reject repeaters at the open edge of an environment (lexurgy's
/// `LscPeripheralRepeater`): `x => y / a* _` can't mean anything, because
/// nothing constrains how far the repeater reaches. Exact-count repeaters
/// (`*2`) are allowed; an anchor (`$ a* _`) shields the repeater because the
/// `$` then occupies the edge. The flags mirror `ElementContext.butBetween`:
/// only the first/last elements of each (nested) sequence inherit edge-ness,
/// while wrappers (repeat/capture/negation/lists) pass it through.
fn check_peripheral(element: &Element, at_start: bool, at_end: bool) -> Result<(), String> {
    if !at_start && !at_end {
        return Ok(());
    }
    match element {
        Element::Sequence(items) => {
            let last = items.len() - 1;
            for (i, e) in items.iter().enumerate() {
                check_peripheral(e, at_start && i == 0, at_end && i == last)?;
            }
            Ok(())
        }
        Element::Group(re) => check_peripheral(&re.element, at_start, at_end),
        Element::List(items) => {
            for item in items {
                if let ListItem::Element(re) = item {
                    check_peripheral(&re.element, at_start, at_end)?;
                }
            }
            Ok(())
        }
        Element::Interfix { first, rest } => {
            check_peripheral(first, at_start, at_end)?;
            for (_, e) in rest {
                check_peripheral(e, at_start, at_end)?;
            }
            Ok(())
        }
        Element::Repeat { element, kind } => {
            let specific_multiple = match kind {
                RepeaterKind::Count(n) => *n > 1,
                RepeaterKind::Range {
                    min: Some(m),
                    max: Some(x),
                } => m == x && *m > 1,
                _ => false,
            };
            if !specific_multiple {
                return Err("a repeater is meaningless at the edge of the environment; \
                     anchor it (e.g. with $) or remove it"
                    .to_string());
            }
            check_peripheral(element, at_start, at_end)
        }
        Element::Negated(e) | Element::Capture { element: e, .. } => {
            check_peripheral(e, at_start, at_end)
        }
        _ => Ok(()),
    }
}

fn check_element(element: &Element, periph: bool) -> Result<(), String> {
    match element {
        Element::Sequence(items) => {
            for e in items {
                check_element(e, periph)?;
            }
            Ok(())
        }
        Element::Group(re) => check_rule_element(re, periph),
        Element::List(items) => {
            for item in items {
                match item {
                    ListItem::Element(re) => check_rule_element(re, periph)?,
                    ListItem::Env(_) => {
                        return Err(
                            "an anchored or empty alternative ('_') is only allowed in an \
                             environment list directly after '/' or '//'"
                                .to_string(),
                        )
                    }
                }
            }
            Ok(())
        }
        Element::Interfix { first, rest } => {
            // NB: a transforming `>` interfix is *not* rejected here. lexurgy's
            // `LscFutureStructure` is a post-parse (compile-stage) error, not an
            // `LscNotParsable`, so `parse` must accept `>`; the compiler rejects
            // it when it lowers the rule, and an unspliced deferred rule's `>`
            // is caught by `reject_deferred_transforming` at compile time.
            check_element(first, periph)?;
            for (_, e) in rest {
                check_element(e, periph)?;
            }
            Ok(())
        }
        Element::Negated(e)
        | Element::Capture { element: e, .. }
        | Element::Repeat { element: e, .. } => check_element(e, periph),
        _ => Ok(()),
    }
}
