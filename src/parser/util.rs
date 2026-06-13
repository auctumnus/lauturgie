// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) Autumn <auctumnus@pm.me>

//! Action-code helpers for the lalrpop grammar.

use crate::parser::ast::*;
use smol_str::{SmolStr, SmolStrBuilder};

/// Collapse a whitespace-separated run of elements: a single element stays
/// itself, two or more become a `Sequence`.
pub fn seq_of(mut elements: Vec<Element>) -> Element {
    if elements.len() == 1 {
        elements.pop().unwrap()
    } else {
        Element::Sequence(elements)
    }
}

pub fn join_rule_name(head: SmolStr, tail: Vec<SmolStr>) -> SmolStr {
    if tail.is_empty() {
        return head;
    }
    let mut builder = SmolStrBuilder::new();
    builder.push_str(&head);
    for part in tail {
        builder.push('-');
        builder.push_str(&part);
    }
    builder.finish()
}

/// Reinterpret the environment parsed after `/` or `//`.
///
/// The ANTLR grammar distinguishes `environment` from `environmentList` by
/// trying alternatives; we parse one cover [`Environment`] instead. If it
/// turns out to be exactly a braced list containing anchored items, it *was*
/// an environment list (`{a _, b _}`), so unpack it. Otherwise it's a single
/// environment (possibly with a leading alternative list: `{a, b} c _`).
pub fn reinterpret_env(env: Environment) -> Result<Vec<Environment>, String> {
    let is_bare_list = !env.anchored
        && env.after.is_none()
        && matches!(&env.before, Some(Element::List(items))
            if items.iter().any(|i| matches!(i, ListItem::Env(_))));
    if !is_bare_list {
        return Ok(vec![env]);
    }
    let Some(Element::List(items)) = env.before else {
        unreachable!()
    };
    items
        .into_iter()
        .map(|item| match item {
            ListItem::Env(e) => Ok(e),
            ListItem::Element(re) => {
                if re.environment.is_some() {
                    Err("an item in an environment list can't have its own condition".to_string())
                } else {
                    Ok(Environment {
                        before: Some(re.element),
                        anchored: false,
                        after: None,
                    })
                }
            }
        })
        .collect()
}
