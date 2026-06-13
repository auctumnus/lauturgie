// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) Autumn <auctumnus@pm.me>

//! The mid-level IR rules lower to.
//!
//! Shapes are regex-like trees whose leaves are *predicates on one segment*
//! ([`SegTest`]) rather than characters. Everything name-shaped is gone by
//! this point: classes and elements are inlined, text is interned to
//! [`SegmentId`]s, matrices are compiled to bit tests.
//!
//! The match side ([`Pattern`]) and the output side ([`Emit`]) are separate
//! types: lexurgy pairs matchers with emitters positionally (an `Alt` in
//! `from` selects the corresponding alternative in `to`, a `Matrix` emit
//! updates the segment its counterpart matched), and keeping the pairing
//! structural makes both tiers honest about it.
//!
//! This IR is what tier selection inspects: a rule whose patterns avoid
//! [`Pattern::Capture`]/[`Pattern::CaptureRef`] (and whose feature variables
//! range over small domains, which can be expanded into alternations) can
//! become an FST; everything else goes to the bytecode tier.

use smol_str::SmolStr;

use super::features::{MatrixTest, MatrixUpdate};
use super::segments::{CoreId, DiacriticMask, SegmentId};
use crate::word::Word;

/// A predicate on a single segment: the leaf of every pattern, and the
/// transition label of the FST tier (transitions carry predicates, not
/// concrete symbols, so the alphabet stays open).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SegTest {
    /// Literal text marked exact (`a!`): plain id equality.
    Exact(SegmentId),
    /// Literal text: id equality, or equality modulo floating diacritics
    /// (`core` matches, `required ⊆ seg.diacritics ⊆ allowed`, where
    /// `allowed = required | floating`). Lowering emits [`SegTest::Exact`]
    /// instead when no floating diacritics are declared.
    Literal {
        id: SegmentId,
        core: CoreId,
        required: DiacriticMask,
        allowed: DiacriticMask,
    },
    /// A feature matrix.
    Matrix(MatrixTest),
    /// A purely syllable-level matrix (lexurgy's `SyllableMatrixMatcher`):
    /// consumes one segment, testing the features of the syllable that
    /// contains it. As an intersection verifier it is *length-hinted*: it
    /// accepts any span that stays within one syllable.
    SylMatrix(MatrixTest),
    /// `[]` or a bare wildcard: any one segment.
    Any,
}

/// Literal text in match position: a run of segment tests matched as one
/// unit, plus the syllable-feature test induced by any syllable-level
/// diacritics written in the pattern text itself (lexurgy's
/// `SymbolMatcher` checks `target.syllableMatrix` against the text's).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextIr {
    pub tests: Vec<SegTest>,
    /// `a!`, lexurgy's `TextMatcher` (vs `SymbolMatcher`): no floating
    /// diacritic allowance, no syllable-feature test, and no floating
    /// transfer when paired with a text emitter.
    pub exact: bool,
    pub syl_mask: u128,
    pub syl_want: u128,
}

/// A capture slot (`$1` is slot 0).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CaptureSlot(pub u32);

/// The match side of an expression.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Pattern {
    /// `*`: matches the empty string.
    Empty,
    Test(SegTest),
    /// Literal text: a run of segment tests matched as one unit (lexurgy's
    /// `SymbolMatcher`/`TextMatcher`). Kept whole rather than lowered to a
    /// `Seq` because text pairs with text emitters through length-sensitive
    /// floating-diacritic transfer rules.
    Text(TextIr),
    Seq(Vec<Pattern>),
    Alt(Vec<Pattern>),
    /// `?`/`*`/`+`/`*n`/`*(n-m)`, with `max: None` for unbounded.
    /// Bounded repeats may simply be unrolled by codegen.
    Repeat {
        inner: Box<Pattern>,
        min: u32,
        max: Option<u32>,
    },
    /// `(...)$n`: record the matched span.
    Capture {
        inner: Box<Pattern>,
        slot: CaptureSlot,
    },
    /// `$n` in match position: match a copy of the captured span
    /// (`inexact` is `~$n`, matching modulo floating diacritics). This is
    /// the back-reference that forces the bytecode tier.
    CaptureRef {
        slot: CaptureSlot,
        inexact: bool,
    },
    /// `!x`: one segment that does *not* match the inner pattern
    /// (validation restricts the operand to single-segment shapes).
    Not(Box<Pattern>),
    /// `!x` at the edge of an environment (lexurgy's
    /// `NegatedLookaroundMatcher`): zero-width, any inner shape, requires a
    /// segment to exist at the position.
    NotAhead(Box<Pattern>),
    /// `!.`: zero-width "not at a syllable boundary" (lexurgy negates `.`
    /// specially, whether or not a syllabifier is in force).
    NoBoundary,
    /// A local environment attached to an element, `(a / x _ // y _)`:
    /// matches `inner`, then requires some `condition` (if any) and no
    /// `exclusion` to hold around the matched span. Zero-width on both
    /// sides; lexurgy's `EnvironmentMatcher`.
    Look {
        inner: Box<Pattern>,
        condition: Vec<EnvIr>,
        exclusion: Vec<EnvIr>,
    },
    /// `a&b`: all parts must match the same span.
    Intersect(Vec<Pattern>),
    /// `$`: start/end of word; which one is positional, so lowering
    /// resolves it to [`Pattern::WordStart`] or [`Pattern::WordEnd`] in
    /// environments. A `WordBoundary` reaching the VM (interior position or
    /// input position) is an error, as in lexurgy.
    WordBoundary,
    /// Zero-width match at the start of the word.
    WordStart,
    /// Zero-width match at the end of the word.
    WordEnd,
    /// `$$`: the boundary between words of a phrase.
    BetweenWords,
    /// `.`: a syllable break.
    SyllableBoundary,
    /// `<syl>`: exactly one whole syllable.
    AnySyllable,
}

/// The output side of an expression.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Emit {
    /// `*`: emit nothing (deletion).
    Empty,
    /// Verbatim text (lexurgy's `SymbolEmitter`, or `TextEmitter` when
    /// `exact`). The word keeps any syllable structure written in the text;
    /// `syl_mask`/`syl_want` are the explicit syllable-feature values its
    /// syllable diacritics carry (applied to every affected syllable).
    Text {
        word: Word,
        exact: bool,
        syl_mask: u128,
        syl_want: u128,
    },
    /// Rewrite the features of the corresponding matched segment, keeping
    /// the fields the matrix doesn't mention. Syllable-level fields update
    /// every syllable the match touches.
    Matrix(MatrixUpdate),
    Seq(Vec<Emit>),
    /// Paired with an `Alt` on the match side: the alternative that matched
    /// selects the emitter.
    Alt(Vec<Emit>),
    /// `$n`: replay a captured span (`~$n` strips floating diacritics
    /// the pattern didn't ask for). Syllable structure is dropped.
    CaptureRef {
        slot: CaptureSlot,
        inexact: bool,
    },
    /// `$.n`: replay a captured span *with* its syllable structure.
    SylCaptureRef {
        slot: CaptureSlot,
    },
    /// `.`: emit a syllable break.
    SyllableBoundary,
    /// `$$`: emit a word break (lexurgy's `BetweenWordsEmitter`: a phrase
    /// of two empty words, splitting the word at this point).
    WordBreak,
}

/// One environment: `before _ after`. `anchored` distinguishes `a _` items
/// in environment lists from bare alternatives.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EnvIr {
    pub before: Option<Pattern>,
    pub after: Option<Pattern>,
    pub anchored: bool,
}

/// One `from => to / condition // exclusion`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExprIr {
    pub from: Pattern,
    pub to: Emit,
    /// `/ {...}`: at least one must hold.
    pub condition: Vec<EnvIr>,
    /// `// {...}`: none may hold.
    pub exclusion: Vec<EnvIr>,
}

/// How a list of expressions scans the word (lexurgy's `MatchMode`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MatchMode {
    /// Find all (non-overlapping, leftmost-preferring) matches first, then
    /// apply them at once; the default.
    #[default]
    Simultaneous,
    /// Apply at the leftmost match, then rescan to the right of the
    /// replacement; `rtl` is the mirror image. Only legal on a plain
    /// expression list, never on a block (`LscIllegalNestedModifier`).
    Ltr,
    Rtl,
}

/// The body of a rule: a tree of blocks, mirroring lexurgy's
/// `ChangeRule` structure (`SequentialBlock`, `FirstMatchingBlock`,
/// `FilterBlock`, `PropagateBlock`). Filters and propagation can attach to
/// the whole rule *or* to individual `Then:`/`Else:` arms, which is why
/// they're wrapper nodes rather than rule fields.
///
/// "Matched" propagates outward: a `Sequential` block matched if any child
/// did, and a `FirstMatching` block applies only the first child that
/// matched, so the VM must report match/no-match for every node.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BlockIr {
    /// A leaf: expressions applied together in one pass over the word,
    /// earlier expressions winning overlaps.
    Exprs { mode: MatchMode, exprs: Vec<ExprIr> },
    /// `Then:`: apply every child in order.
    Sequential(Vec<BlockIr>),
    /// `Else:`: apply only the first child that matches.
    FirstMatching(Vec<BlockIr>),
    /// Re-apply until fixpoint (the VM caps iterations).
    Propagate(Box<BlockIr>),
    /// `[vowel]` / `@vowel` filter: only matching segments are visible
    /// inside (validation restricts the filter to single-segment shapes).
    Filter {
        filter: Pattern,
        inner: Box<BlockIr>,
    },
}

/// A change rule lowered against the declarations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuleIr {
    pub name: SmolStr,
    pub body: BlockIr,
}

/// A compiled `Syllables:` declaration (lexurgy's `Syllabifier`): patterns
/// tried by a shortest-path search over the word, with a preference order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyllabifierIr {
    pub exprs: Vec<SylExprIr>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SylExprIr {
    pub pattern: SylPatternIr,
    /// `=> [matrix]`: syllable-level features assigned to syllables this
    /// pattern forms, as an update.
    pub assign: Option<MatrixUpdate>,
}

impl SyllabifierIr {
    /// Do any of the patterns *read* existing syllable structure (syllable
    /// matrices; `.`/`<syl>` are already text/absent in syllable patterns)?
    /// If not, syllabification is idempotent: re-running it on its own
    /// untouched output reproduces that output exactly (the DP is
    /// deterministic over segments, the syllable map is the identity, and
    /// re-applying the assigned matrices is a fixpoint), so the apply
    /// loop may skip it.
    pub fn reads_structure(&self) -> bool {
        fn pattern(p: &Pattern) -> bool {
            match p {
                Pattern::Test(SegTest::SylMatrix(_)) => true,
                Pattern::Test(_)
                | Pattern::Empty
                | Pattern::WordBoundary
                | Pattern::WordStart
                | Pattern::WordEnd
                | Pattern::BetweenWords
                | Pattern::CaptureRef { .. } => false,
                Pattern::Text(t) => t.syl_mask != 0,
                Pattern::Seq(ps) | Pattern::Alt(ps) | Pattern::Intersect(ps) => {
                    ps.iter().any(pattern)
                }
                Pattern::Repeat { inner, .. }
                | Pattern::Capture { inner, .. }
                | Pattern::Not(inner)
                | Pattern::NotAhead(inner) => pattern(inner),
                Pattern::Look {
                    inner,
                    condition,
                    exclusion,
                } => pattern(inner) || condition.iter().any(env) || exclusion.iter().any(env),
                Pattern::SyllableBoundary | Pattern::NoBoundary | Pattern::AnySyllable => true,
            }
        }
        fn env(e: &EnvIr) -> bool {
            e.before.as_ref().is_some_and(pattern) || e.after.as_ref().is_some_and(pattern)
        }
        self.exprs.iter().any(|expr| match &expr.pattern {
            SylPatternIr::Simple(p) => pattern(p),
            SylPatternIr::Structured(s) => {
                let StructuredSyl {
                    reluctant_onset,
                    onset,
                    nucleus,
                    coda,
                    condition,
                    exclusion,
                } = &**s;
                reluctant_onset.as_ref().is_some_and(pattern)
                    || pattern(onset)
                    || pattern(nucleus)
                    || coda.as_ref().is_some_and(pattern)
                    || condition.iter().any(env)
                    || exclusion.iter().any(env)
            }
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SylPatternIr {
    /// A plain pattern matching one whole syllable (environments are baked
    /// in as [`Pattern::Look`]).
    Simple(Pattern),
    /// `reluctant? :: onset :: nucleus :: coda?`: parts matched in turn;
    /// the DP prefers shorter reluctant onsets and longer nuclei. Boxed
    /// because it is much larger than [`SylPatternIr::Simple`] and built once
    /// per syllabifier rule, never on a hot path.
    Structured(Box<StructuredSyl>),
}

/// The parts of a `reluctant? :: onset :: nucleus :: coda?` syllable pattern.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructuredSyl {
    pub reluctant_onset: Option<Pattern>,
    pub onset: Pattern,
    pub nucleus: Pattern,
    pub coda: Option<Pattern>,
    pub condition: Vec<EnvIr>,
    pub exclusion: Vec<EnvIr>,
}
