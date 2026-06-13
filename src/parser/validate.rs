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
        Statement::Element { element, .. } => check_rule_element(element),
        Statement::Syllables(SyllableSpec::Patterns(patterns)) => {
            for p in patterns {
                match &p.pattern {
                    SyllablePattern::Plain(re) => check_rule_element(re)?,
                    SyllablePattern::Structured {
                        reluctant_onset,
                        parts,
                    } => {
                        if let Some(o) = reluctant_onset {
                            check_element(o)?;
                        }
                        for part in parts {
                            check_element(part)?;
                        }
                    }
                }
                check_env_opt(&p.environment)?;
            }
            Ok(())
        }
        Statement::Syllables(_) => Ok(()),
        Statement::Deromanizer { block, .. }
        | Statement::Romanizer { block, .. }
        | Statement::InterRomanizer { block, .. } => check_block(block),
        Statement::Rule(rule) => {
            for m in &rule.modifiers {
                if let RuleModifier::Filter(e) = m {
                    check_element(e)?;
                }
            }
            check_block(&rule.block)
        }
        Statement::Expression(e) => check_expression_body(e),
    }
}

fn check_block(block: &Block) -> Result<(), String> {
    check_block_element(&block.first)?;
    for (_, el) in &block.rest {
        check_block_element(el)?;
    }
    Ok(())
}

fn check_block_element(el: &BlockElement) -> Result<(), String> {
    match el {
        BlockElement::Expressions(exprs) => {
            for e in exprs {
                if let Expression::Standard(s) = e {
                    check_expression_body(s)?;
                }
            }
            Ok(())
        }
        BlockElement::Nested(b) => check_block(b),
    }
}

fn check_expression_body(e: &StandardExpression) -> Result<(), String> {
    check_rule_element(&e.from)?;
    check_element(&e.to)?;
    check_env_opt(&e.environment)
}

fn check_rule_element(re: &RuleElement) -> Result<(), String> {
    check_element(&re.element)?;
    check_env_opt(&re.environment)
}

fn check_env_opt(env: &Option<CompoundEnvironment>) -> Result<(), String> {
    if let Some(env) = env {
        for list in [&env.condition, &env.exclusion].into_iter().flatten() {
            for e in list {
                check_environment(e)?;
            }
        }
    }
    Ok(())
}

fn check_environment(env: &Environment) -> Result<(), String> {
    if let Some(b) = &env.before {
        check_element(b)?;
        // The edge *away* from `_` (lexurgy's `rightBeforeAnchor` context).
        check_peripheral(b, true, false)?;
    }
    if let Some(a) = &env.after {
        check_element(a)?;
        check_peripheral(a, false, true)?;
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

fn check_element(element: &Element) -> Result<(), String> {
    match element {
        Element::Sequence(items) => {
            for e in items {
                check_element(e)?;
            }
            Ok(())
        }
        Element::Group(re) => check_rule_element(re),
        Element::List(items) => {
            for item in items {
                match item {
                    ListItem::Element(re) => check_rule_element(re)?,
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
            check_element(first)?;
            for (_, e) in rest {
                check_element(e)?;
            }
            Ok(())
        }
        Element::Negated(e)
        | Element::Capture { element: e, .. }
        | Element::Repeat { element: e, .. } => check_element(e),
        _ => Ok(()),
    }
}
