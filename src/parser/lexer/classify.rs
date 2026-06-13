// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) Autumn <auctumnus@pm.me>

use super::{Spanned, Tok};

/// What surrounds the line currently being classified.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Ctx {
    /// Top level: expecting a statement.
    Top,
    /// Inside a change rule / romanizer / deromanizer block.
    Rule,
    /// Inside a `Syllables:` declaration.
    Syllables,
}

fn seg_has(seg: &[Spanned], pred: impl Fn(&Tok) -> bool) -> bool {
    seg.iter().any(|(_, t, _)| pred(t))
}

fn tok_after_change(seg: &[Spanned]) -> Option<&Tok> {
    let idx = seg.iter().position(|(_, t, _)| *t == Tok::Change)?;
    seg.get(idx + 1).map(|(_, t, _)| t)
}

/// Is this segment a declaration of the given keyword kind?
/// (`feature x(...)` yes; `feature ltr:` is a rule *named* "feature".)
fn is_decl_form(seg: &[Spanned]) -> bool {
    let first = &seg[0].1;
    let second = seg.get(1).map(|(_, t, _)| t);
    let has_colon = seg_has(seg, |t| *t == Tok::Colon);
    let has_change = seg_has(seg, |t| *t == Tok::Change);
    match first {
        Tok::KwSyllables(_) => second == Some(&Tok::Colon),
        Tok::KwRomanizer(_) => is_romanizer_header(&seg[1..], true),
        Tok::KwDeromanizer(_) => is_romanizer_header(&seg[1..], false),
        Tok::KwFeature(_)
        | Tok::KwClass(_)
        | Tok::KwElement(_)
        | Tok::KwSymbol(_)
        | Tok::KwDiacritic(_) => second == Some(&Tok::Ws) && !has_colon && !has_change,
        _ => false,
    }
}

fn is_name_part(t: &Tok) -> bool {
    matches!(t, Tok::Name(_) | Tok::Number(_)) || t.keyword_text().is_some()
}

/// Does the rest of a `romanizer`/`deromanizer` segment match the header
/// shape `(-name)* ( literal)? :`? Anything else (e.g. a rule *named*
/// `deromanizer-x` or `romanizer-x defer:`) falls through to a change rule,
/// matching the order ANTLR tries the statement alternatives in.
fn is_romanizer_header(mut rest: &[Spanned], allow_hyphen_name: bool) -> bool {
    if allow_hyphen_name {
        while let [(_, Tok::Hyphen, _), (_, part, _), tail @ ..] = rest {
            if !is_name_part(part) {
                return false;
            }
            rest = tail;
        }
    }
    if let [(_, Tok::Ws, _), (_, Tok::KwLiteral(_), _), tail @ ..] = rest {
        rest = tail;
    }
    matches!(rest, [(_, Tok::Colon, _)])
}

/// A `Then:`/`Else:` line is a block continuation only if it has the shape
/// `then (ws modifier)* :`, i.e. the `:` comes before any `=>` (otherwise
/// it's an expression like `then => else`) and the keyword is followed by
/// whitespace or `:` (otherwise it's a rule header like `then-2:`).
fn is_block_type(seg: &[Spanned]) -> bool {
    if !matches!(seg.get(1), Some((_, Tok::Ws | Tok::Colon, _))) {
        return false;
    }
    for (_, t, _) in seg {
        match t {
            Tok::Colon => return true,
            Tok::Change => return false,
            _ => {}
        }
    }
    false
}

/// Classify a statement-position segment. Returns (marker, relabel_first, new_ctx).
fn classify_statement(seg: &[Spanned]) -> (Tok, bool, Ctx) {
    if is_decl_form(seg) {
        let ctx = match &seg[0].1 {
            Tok::KwSyllables(_) => Ctx::Syllables,
            Tok::KwRomanizer(_) | Tok::KwDeromanizer(_) => Ctx::Rule,
            _ => Ctx::Top,
        };
        return (Tok::MarkStmt, false, ctx);
    }
    // Not a declaration: a keyword in first position is just a name
    // (e.g. a rule named `feature`).
    let relabel = seg[0].1.keyword_text().is_some();
    if seg_has(seg, |t| *t == Tok::Change) {
        (Tok::MarkTopExpr, relabel, Ctx::Top)
    } else {
        (Tok::MarkStmt, relabel, Ctx::Rule)
    }
}

fn classify_segment(ctx: Ctx, paren_depth: usize, seg: &[Spanned]) -> (Tok, bool, Ctx) {
    let first = &seg[0].1;
    match ctx {
        Ctx::Rule if paren_depth > 0 => {
            // Statements can't begin inside a parenthesized block; everything
            // is block content (anything else is a parse error either way).
            let marker = match first {
                Tok::CParen => Tok::MarkEnd,
                Tok::KwThen(_) | Tok::KwElse(_) if is_block_type(seg) => Tok::MarkBlockType,
                _ => Tok::MarkExpr,
            };
            (marker, false, ctx)
        }
        Ctx::Rule => match first {
            Tok::KwThen(_) | Tok::KwElse(_) if is_block_type(seg) => {
                (Tok::MarkBlockType, false, ctx)
            }
            Tok::Colon | Tok::OParen => (Tok::MarkExpr, false, ctx),
            Tok::KwUnchanged(_) | Tok::KwOff(_) if seg.len() == 1 => (Tok::MarkExpr, false, ctx),
            _ if seg_has(seg, |t| *t == Tok::Change) => (Tok::MarkExpr, false, ctx),
            _ => classify_statement(seg),
        },
        Ctx::Syllables => match first {
            Tok::KwExplicit(_) | Tok::KwClear(_) if seg.len() == 1 => {
                (Tok::MarkSyllMode, false, Ctx::Top)
            }
            _ if is_decl_form(seg) || seg_has(seg, |t| *t == Tok::Colon) => classify_statement(seg),
            // `a => b` exits the Syllables block (a syllable pattern's `=>`
            // must be followed by a matrix); `a => [x]` stays inside.
            _ if seg_has(seg, |t| *t == Tok::Change)
                && tok_after_change(seg) != Some(&Tok::MatrixStart) =>
            {
                classify_statement(seg)
            }
            _ => (Tok::MarkSyll, false, ctx),
        },
        Ctx::Top => classify_statement(seg),
    }
}

/// Pass 2: replace newline runs with classification markers.
///
/// Mirrors the greedy choices ANTLR's adaptive prediction makes: expression
/// lines after a rule header belong to that rule, `Then:`/`Else:` lines
/// continue the current block, syllable patterns extend the current
/// `Syllables:` declaration, and anything else starts a new statement.
pub fn classify(tokens: Vec<Spanned>) -> Vec<Spanned> {
    let mut out: Vec<Spanned> = Vec::with_capacity(tokens.len() + 16);
    let mut i = 0;
    let mut ctx = Ctx::Top;
    let mut paren_depth: usize = 0;

    // A leading Ws token can only occur at the very start of the file.
    if matches!(tokens.first(), Some((_, Tok::Ws, _))) {
        i = 1;
    }

    while i < tokens.len() {
        // Skip the newline run before the next segment.
        let run_start = tokens[i].0;
        while i < tokens.len() && tokens[i].1 == Tok::Newline {
            i += 1;
        }
        if i >= tokens.len() {
            break;
        }

        // Scan the segment: up to the next Newline at brace depth 0. Newlines
        // inside braces (multi-line class declarations) stay in the segment.
        let seg_start = i;
        let mut brace_depth: usize = 0;
        while i < tokens.len() {
            match &tokens[i].1 {
                Tok::Newline if brace_depth == 0 => break,
                Tok::ListStart | Tok::ClassStart => brace_depth += 1,
                Tok::ListEnd => brace_depth = brace_depth.saturating_sub(1),
                _ => {}
            }
            i += 1;
        }
        let seg = &tokens[seg_start..i];

        let (marker, relabel_first, new_ctx) = classify_segment(ctx, paren_depth, seg);
        if new_ctx != ctx {
            paren_depth = 0;
        }
        ctx = new_ctx;
        out.push((run_start, marker, seg[0].0));

        for (idx, (start, tok, end)) in seg.iter().enumerate() {
            let tok = if idx == 0 && relabel_first {
                Tok::Name(tok.keyword_text().expect("relabel implies keyword").into())
            } else {
                tok.clone()
            };
            if ctx == Ctx::Rule {
                match tok {
                    Tok::OParen => paren_depth += 1,
                    Tok::CParen => paren_depth = paren_depth.saturating_sub(1),
                    _ => {}
                }
            }
            out.push((*start, tok, *end));
        }
    }
    out
}
