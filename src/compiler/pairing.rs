// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) Autumn <auctumnus@pm.me>

//! Compile-time validation of `from => to` pairings, mirroring lexurgy's
//! `Matcher.transformerTo(Emitter, filtered)` dispatch tree, which runs at
//! rule-build time and rejects rules the VM would otherwise only fail (or
//! quietly run) per word.
//!
//! The model reproduces three behaviors of the Kotlin original exactly:
//!
//! 1. **Element-count matching.** `BaseMatcher.transformerTo` *tries* the
//!    structural pairing and catches `InvalidTransformation`: if the emitter
//!    is fully independent (text/`*`/captures, never matrices or
//!    alternative lists), the mismatch falls back to emitting the whole
//!    result per match. So `a b => x y z` is legal, but `a b => x [vcd] z`
//!    is "Found 2 elements ... but 3 elements".
//!
//! 2. **Filtered-rule restrictions.** Each matcher that directly pairs with
//!    an emitter is checked by `checkValidInFilterIfEmitterIs`: multi-segment
//!    text, `*` and `<syl>` are rejected (`LscIllegalStructureInFilteredRule-
//!    Input`, *not* catchable by the independence fallback). Sequence-shaped
//!    matchers can't take the independent-sequence fallback in filtered rules
//!    (`BaseMatcher.transformerToIndependentSequence` throws), but
//!    `SimpleMatcher` overrides that method *without* the filtered check, so
//!    a single matcher with a surplus sequence emitter (`@cv => z u`) slips
//!    through and silently drops the surplus at run time. We reproduce the
//!    quirk: rejecting it would diverge from lexurgy.
//!
//! 3. **Check placement.** The checks fire only where pairing recursion
//!    actually visits: members of sequences/alternatives pair recursively,
//!    but a group paired as a whole (`(gi u) => p`) never has its *contents*
//!    checked, and environments are never paired at all; multi-segment text
//!    is fine in both places even under a filter, exactly as in lexurgy.

use super::ir::{BlockIr, Emit, Pattern, RuleIr, SegTest};
use super::segments::SegmentInterner;
use super::CompileError;

/// Validate every expression of a lowered rule.
pub fn check_rule(rule: &RuleIr, segments: &SegmentInterner) -> Result<(), CompileError> {
    check_block(&rule.body, false, segments).map_err(|e| CompileError::Expression {
        rule: rule.name.to_string(),
        what: match e {
            PairErr::Invalid(msg) | PairErr::Fatal(msg) => msg,
        },
    })
}

fn check_block(block: &BlockIr, filtered: bool, segs: &SegmentInterner) -> Result<(), PairErr> {
    match block {
        BlockIr::Exprs { exprs, .. } => {
            for expr in exprs {
                // An expression-level environment wraps the match side in an
                // `EnvironmentMatcher` (a lifting matcher) before pairing,
                // which shields the inner element from the filtered-rule
                // checks whenever the emitter pairs independently: lexurgy
                // accepts filtered `sg => * / _ g` but rejects bare
                // `sg => *`. Model it as the `Look` it would lower to.
                if expr.condition.is_empty() && expr.exclusion.is_empty() {
                    transformer_to(&expr.from, &expr.to, filtered, segs)?;
                } else {
                    let wrapped = Pattern::Look {
                        inner: Box::new(expr.from.clone()),
                        condition: vec![],
                        exclusion: vec![],
                    };
                    transformer_to(&wrapped, &expr.to, filtered, segs)?;
                }
            }
            Ok(())
        }
        BlockIr::Sequential(children) | BlockIr::FirstMatching(children) => {
            for child in children {
                check_block(child, filtered, segs)?;
            }
            Ok(())
        }
        BlockIr::Propagate(inner) => check_block(inner, filtered, segs),
        BlockIr::Filter { inner, .. } => check_block(inner, true, segs),
    }
}

enum PairErr {
    /// Lexurgy's `InvalidTransformation`: caught by the independence
    /// fallback in enclosing pairings.
    Invalid(String),
    /// Lexurgy's `LscIllegalStructure` family: always propagates.
    Fatal(String),
}

/// `BaseMatcher.transformerTo`, including the catch-and-fall-back.
fn transformer_to(
    m: &Pattern,
    e: &Emit,
    filtered: bool,
    segs: &SegmentInterner,
) -> Result<(), PairErr> {
    if filtered {
        check_valid_in_filter(m, segs)?;
    }
    let attempt = match e {
        Emit::Alt(es) => to_alternatives(m, es, e, filtered, segs),
        Emit::Seq(es) => {
            if prefers_independent_seq(m) && is_independent(e) {
                to_independent_sequence(m, e, filtered, segs)
            } else {
                to_sequence(m, es, e, filtered, segs)
            }
        }
        _ => {
            if prefers_independent(m) && is_independent(e) {
                Ok(())
            } else if is_conditional(e) {
                to_conditional(m, e, filtered, segs)
            } else {
                Ok(())
            }
        }
    };
    match attempt {
        Err(PairErr::Invalid(_)) if is_independent(e) => match e {
            Emit::Seq(_) => to_independent_sequence(m, e, filtered, segs),
            _ => Ok(()),
        },
        other => other,
    }
}

/// `transformerToAlternatives` per matcher class.
fn to_alternatives(
    m: &Pattern,
    es: &[Emit],
    e: &Emit,
    filtered: bool,
    segs: &SegmentInterner,
) -> Result<(), PairErr> {
    match m {
        Pattern::Alt(ms) => {
            if ms.len() == es.len() {
                // Equal sizes pair element-wise (text-run grouping pairs the
                // grouped class with the matching emitter slice, which is the
                // same thing member-wise).
                for (mi, ei) in ms.iter().zip(es) {
                    transformer_to(mi, ei, filtered, segs)?;
                }
                Ok(())
            } else {
                // `AlternativeMatcher`'s fallback: each *grouped* member
                // pairs with the whole alternative emitter; consecutive text
                // members act as one class. Any failure becomes the
                // mismatched-lengths error.
                let attempt = (|| {
                    for group in group_texts(ms) {
                        match group {
                            TextGroup::Class(members) => {
                                // `ClassMatcher.transformerToAlternatives`
                                if members.len() == es.len() {
                                    for (mi, ei) in members.iter().zip(es) {
                                        transformer_to(mi, ei, filtered, segs)?;
                                    }
                                } else {
                                    return Err(PairErr::Invalid(String::new()));
                                }
                            }
                            TextGroup::Single(p) => transformer_to(p, e, filtered, segs)?,
                        }
                    }
                    Ok(())
                })();
                match attempt {
                    Err(PairErr::Invalid(_)) => Err(mismatched(ms.len(), m, es.len(), e, segs)),
                    other => other,
                }
            }
        }
        Pattern::Seq(ms) => {
            // `SequenceMatcher.transformerToAlternatives`: distribute the
            // alternatives over the sequence when every nested alternative
            // has as many members as the emitter list, and every emitter
            // alternative is a sequence as long as the match.
            let n = es.len();
            let alts_fit = ms.iter().all(|p| match p {
                Pattern::Alt(xs) => xs.len() == n,
                _ => true,
            });
            let seqs: Option<Vec<&Vec<Emit>>> = es
                .iter()
                .map(|a| match a {
                    Emit::Seq(parts) if parts.len() == ms.len() => Some(parts),
                    _ => None,
                })
                .collect();
            match seqs {
                Some(seqs) if alts_fit => {
                    for (i, emit_seq) in seqs.iter().enumerate() {
                        for (mp, ep) in ms.iter().zip(emit_seq.iter()) {
                            let mp_i = match mp {
                                Pattern::Alt(xs) => &xs[i],
                                other => other,
                            };
                            transformer_to(mp_i, ep, filtered, segs)?;
                        }
                    }
                    Ok(())
                }
                _ => Err(mismatched(ms.len(), m, 1, e, segs)),
            }
        }
        _ if is_lifting(m) => transformer_to(lifted(m), e, filtered, segs),
        _ => Err(mismatched(1, m, es.len(), e, segs)),
    }
}

/// `transformerToSequence` per matcher class.
fn to_sequence(
    m: &Pattern,
    es: &[Emit],
    e: &Emit,
    filtered: bool,
    segs: &SegmentInterner,
) -> Result<(), PairErr> {
    match m {
        Pattern::Seq(ms) => {
            if ms.len() == es.len() {
                for (mi, ei) in ms.iter().zip(es) {
                    transformer_to(mi, ei, filtered, segs)?;
                }
                Ok(())
            } else {
                Err(mismatched(ms.len(), m, es.len(), e, segs))
            }
        }
        // Alternative/class matchers pair every member with the whole
        // sequence emitter.
        Pattern::Alt(ms) => {
            for mi in ms {
                transformer_to(mi, e, filtered, segs)?;
            }
            Ok(())
        }
        _ if is_lifting(m) => transformer_to(lifted(m), e, filtered, segs),
        _ => Err(mismatched(1, m, es.len(), e, segs)),
    }
}

/// `transformerToConditional` per matcher class: sequences and alternatives
/// pair every element with the same conditional emitter; simple matchers
/// pair directly.
fn to_conditional(
    m: &Pattern,
    e: &Emit,
    filtered: bool,
    segs: &SegmentInterner,
) -> Result<(), PairErr> {
    match m {
        Pattern::Seq(ms) | Pattern::Alt(ms) => {
            for mi in ms {
                transformer_to(mi, e, filtered, segs)?;
            }
            Ok(())
        }
        _ if is_lifting(m) => transformer_to(lifted(m), e, filtered, segs),
        _ => Ok(()),
    }
}

/// `transformerToIndependentSequence`: rejected in filtered rules, except
/// that `SimpleMatcher` overrides it without the check (the quirk that
/// admits `@cv => z u` member-wise and `a => x y` directly).
fn to_independent_sequence(
    m: &Pattern,
    e: &Emit,
    filtered: bool,
    segs: &SegmentInterner,
) -> Result<(), PairErr> {
    if is_simple(m) || !filtered {
        Ok(())
    } else {
        let n = match m {
            Pattern::Seq(ms) => ms.len(),
            _ => 1,
        };
        let en = match e {
            Emit::Seq(es) => es.len(),
            _ => 1,
        };
        Err(mismatched(n, m, en, e, segs))
    }
}

/// `checkValidInFilterIfEmitterIs`: only the simple matchers that override
/// it: multi-segment text, `*`, and `<syl>`.
fn check_valid_in_filter(m: &Pattern, segs: &SegmentInterner) -> Result<(), PairErr> {
    match m {
        Pattern::Text(t) if t.tests.len() > 1 => Err(PairErr::Fatal(format!(
            "a multi-segment element like \"{}\" can't be used on the match side of filtered rules",
            render_pattern(m, segs),
        ))),
        Pattern::Empty => Err(PairErr::Fatal(
            "an empty element like \"*\" can't be used on the match side of filtered rules"
                .to_string(),
        )),
        Pattern::AnySyllable => Err(PairErr::Fatal(
            "a syllable element like \"<syl>\" can't be used on the match side of filtered rules"
                .to_string(),
        )),
        _ => Ok(()),
    }
}

// matcher/emitter classification (kotlin class hierarchy)

/// `SequenceMatcher` and `RepeaterMatcher` prefer independent emitters.
fn prefers_independent(m: &Pattern) -> bool {
    matches!(m, Pattern::Seq(_) | Pattern::Repeat { .. })
}

/// Only `RepeaterMatcher` prefers independent *sequence* emitters.
fn prefers_independent_seq(m: &Pattern) -> bool {
    matches!(m, Pattern::Repeat { .. })
}

/// `LiftingMatcher` subclasses: pairing passes through to the inner element.
fn is_lifting(m: &Pattern) -> bool {
    matches!(
        m,
        Pattern::Repeat { .. }
            | Pattern::Capture { .. }
            | Pattern::Look { .. }
            | Pattern::Intersect(_)
    )
}

fn lifted(m: &Pattern) -> &Pattern {
    match m {
        Pattern::Repeat { inner, .. } => inner,
        Pattern::Capture { inner, .. } => inner,
        Pattern::Look { inner, .. } => inner,
        // `IntersectionMatcher` lifts onto its initial matcher.
        Pattern::Intersect(parts) => &parts[0],
        _ => unreachable!("lifted() on non-lifting pattern"),
    }
}

/// `SimpleMatcher` subclasses: everything that isn't a container or lifter.
fn is_simple(m: &Pattern) -> bool {
    !matches!(m, Pattern::Seq(_) | Pattern::Alt(_)) && !is_lifting(m)
}

/// Kotlin `Emitter.isIndependent()`: alternative emitters never are (even
/// when every member is), sequences are when all members are.
fn is_independent(e: &Emit) -> bool {
    match e {
        Emit::Alt(_) | Emit::Matrix(_) => false,
        Emit::Seq(parts) => parts.iter().all(is_independent),
        Emit::Text { .. }
        | Emit::Empty
        | Emit::CaptureRef { .. }
        | Emit::SylCaptureRef { .. }
        | Emit::SyllableBoundary
        | Emit::WordBreak => true,
    }
}

/// Kotlin `Emitter.isConditional()`: plain text is *both* conditional and
/// independent (`SymbolEmitter`); exact text (`TextEmitter`) is only
/// independent.
fn is_conditional(e: &Emit) -> bool {
    match e {
        Emit::Text { exact, .. } => !*exact,
        Emit::Matrix(_) | Emit::Alt(_) => true,
        Emit::Seq(parts) => parts.iter().any(is_conditional),
        Emit::Empty
        | Emit::CaptureRef { .. }
        | Emit::SylCaptureRef { .. }
        | Emit::SyllableBoundary
        | Emit::WordBreak => false,
    }
}

/// `AlternativeMatcher` groups runs of consecutive text members into one
/// class, which changes how mismatched alternative emitters distribute.
enum TextGroup<'a> {
    Class(Vec<&'a Pattern>),
    Single(&'a Pattern),
}

fn group_texts(ms: &[Pattern]) -> Vec<TextGroup<'_>> {
    let is_text = |p: &Pattern| {
        matches!(
            p,
            Pattern::Text(_) | Pattern::Test(SegTest::Literal { .. } | SegTest::Exact(_))
        )
    };
    let mut out = Vec::new();
    let mut run: Vec<&Pattern> = Vec::new();
    for p in ms {
        if is_text(p) {
            run.push(p);
        } else {
            if !run.is_empty() {
                out.push(TextGroup::Class(std::mem::take(&mut run)));
            }
            out.push(TextGroup::Single(p));
        }
    }
    if !run.is_empty() {
        out.push(TextGroup::Class(run));
    }
    out
}

fn mismatched(n: usize, m: &Pattern, en: usize, e: &Emit, segs: &SegmentInterner) -> PairErr {
    let pl = |k: usize| if k == 1 { "element" } else { "elements" };
    PairErr::Invalid(format!(
        "found {n} {} (\"{}\") on the left side of the arrow but {en} {} (\"{}\") on the right side",
        pl(n),
        render_pattern(m, segs),
        pl(en),
        render_emit(e, segs),
    ))
}

// rendering for error messages

fn render_pattern(p: &Pattern, segs: &SegmentInterner) -> String {
    match p {
        Pattern::Empty => "*".to_string(),
        Pattern::Test(t) => render_test(t, segs),
        Pattern::Text(t) => t.tests.iter().map(|t| render_test(t, segs)).collect(),
        Pattern::Seq(parts) => parts
            .iter()
            .map(|p| render_pattern(p, segs))
            .collect::<Vec<_>>()
            .join(" "),
        Pattern::Alt(parts) => format!(
            "{{{}}}",
            parts
                .iter()
                .map(|p| render_pattern(p, segs))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Pattern::Repeat { inner, min, max } => {
            let suffix = match (min, max) {
                (0, Some(1)) => "?".to_string(),
                (0, None) => "*".to_string(),
                (1, None) => "+".to_string(),
                (n, Some(m)) if n == m => format!("*{n}"),
                (n, Some(m)) => format!("*({n}-{m})"),
                (n, None) => format!("*({n}-)"),
            };
            format!("({}){}", render_pattern(inner, segs), suffix)
        }
        Pattern::Capture { inner, slot } => {
            format!("({})${}", render_pattern(inner, segs), slot.0 + 1)
        }
        Pattern::CaptureRef { slot, .. } => format!("${}", slot.0 + 1),
        Pattern::Not(inner) | Pattern::NotAhead(inner) => {
            format!("!{}", render_pattern(inner, segs))
        }
        Pattern::NoBoundary => "!.".to_string(),
        Pattern::Look { inner, .. } => render_pattern(inner, segs),
        Pattern::Intersect(parts) => parts
            .iter()
            .map(|p| render_pattern(p, segs))
            .collect::<Vec<_>>()
            .join("&"),
        Pattern::WordBoundary | Pattern::WordStart | Pattern::WordEnd => "$".to_string(),
        Pattern::BetweenWords => "$$".to_string(),
        Pattern::SyllableBoundary => ".".to_string(),
        Pattern::AnySyllable => "<syl>".to_string(),
    }
}

fn render_test(t: &SegTest, segs: &SegmentInterner) -> String {
    match t {
        SegTest::Exact(id) | SegTest::Literal { id, .. } => segs.get(*id).text.to_string(),
        SegTest::Matrix(_) | SegTest::SylMatrix(_) => "[...]".to_string(),
        SegTest::Any => "[]".to_string(),
    }
}

fn render_emit(e: &Emit, segs: &SegmentInterner) -> String {
    match e {
        Emit::Empty => "*".to_string(),
        Emit::Text { word, .. } => word
            .segs
            .iter()
            .map(|&id| segs.get(id).text.as_str())
            .collect(),
        Emit::Matrix(_) => "[...]".to_string(),
        Emit::Seq(parts) => parts
            .iter()
            .map(|e| render_emit(e, segs))
            .collect::<Vec<_>>()
            .join(" "),
        Emit::Alt(parts) => format!(
            "{{{}}}",
            parts
                .iter()
                .map(|e| render_emit(e, segs))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Emit::CaptureRef { slot, .. } => format!("${}", slot.0 + 1),
        Emit::SylCaptureRef { slot } => format!("$.{}", slot.0 + 1),
        Emit::WordBreak => "$$".to_string(),
        Emit::SyllableBoundary => ".".to_string(),
    }
}
