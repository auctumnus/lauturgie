// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) Autumn <auctumnus@pm.me>

//! The tree-walking executor for compiled rules.
//!
//! This is a faithful port of lexurgy's matcher/emitter machinery
//! (`sc/element/*.kt`, `ChangeRules.kt`, `Syllabifier.kt`), operating on
//! interned segments and bit-packed feature words instead of strings and
//! hash maps. It is the *reference* implementation: the FST and bytecode
//! tiers will be differentially tested against it.
//!
//! The semantics it preserves exactly:
//!
//! - Matchers return *all* possible match ends in preference order:
//!   alternatives in written order, repeaters longest-first (greedy).
//! - `claimAll` collects matches restarting one segment after each match's
//!   *start*; overlaps are resolved afterward by precedence (expression
//!   order, then position), with half-open ranges so zero-width claims
//!   never overlap anything.
//! - Environments check lookbehind by claiming the `before` pattern
//!   leftward from the match start; bindings flow before → after → result.
//! - The from/to pairing follows `BaseMatcher.transformerTo`: alternative
//!   emitters pair by index, sequence emitters pair element-wise (falling
//!   back to independent emission when lengths mismatch but the emitter
//!   doesn't need the match), matrices update what they matched, text
//!   emitters copy floating diacritics from the matched segment (excluding
//!   ones the pattern itself mentioned).
//! - Syllable structure: matched breaks (`.`) are *removed* from the output
//!   unless re-emitted; everything else is recovered from the original word
//!   (`recoverStructure`); syllable-level feature changes stitch across
//!   transformation boundaries exactly like `applyTransformations`.
//! - `ltr`/`rtl` rules never report "no match" (lexurgy's `invoke` returns
//!   the phrase unconditionally for directional rules), which matters for
//!   `Else:` blocks.
//! - Rules operate on whole phrases ([`Phrase`]), using its linear encoding
//!   of lexurgy's `PhraseIndex`: word gaps occupy one position slot each,
//!   ordinary matchers can't consume across them (no segment lives there),
//!   and `$$` steps over them. `$`-anchors match at *every* word's edges.
//!   `Else:` blocks run per word (lexurgy wraps every `FirstMatchingBlock`
//!   in a `WithinWordBlock`).

use std::collections::{BTreeMap, HashSet};
use std::rc::Rc;

use crate::compiler::decls::Declarations;
use crate::compiler::features::{FeatureWord, Level, MatrixTest, MatrixUpdate};
use crate::compiler::ir::{
    BlockIr, Emit, EnvIr, ExprIr, MatchMode, Pattern, SegTest, StructuredSyl, SylPatternIr,
    SyllabifierIr, TextIr,
};
use crate::compiler::segments::{
    render_syl_mods, syl_mods_bits, syl_mods_test, DiacriticMask, SegmentId, SegmentInterner,
};
use crate::compiler::{CompiledRules, Step, Universe};
use crate::word::{Phrase, Syl, Word};

/// Per-word filtered-index maps: `maps[w][i]` is the real segment index in
/// word `w` of its `i`th filter-passing segment (lexurgy's
/// `unfilterTransformations` bookkeeping). `None` when a rule has no filter.
type FilterMaps = Vec<Vec<usize>>;

/// Backtracking-option cap, so hostile patterns can't explode. Lexurgy
/// relies on wall-clock interruption instead.
const MAX_OPTIONS: usize = 65_536;
/// Budget for one `propagate` fixpoint loop, in *estimated work*: each
/// productive step charges its phrase length cubed, approximating the
/// worst-case backtracking cost of one rule application. Bare step counts
/// can't tell apart the two legitimate-vs-runaway regimes: lexurgy's own
/// binary-counter test does 2^10 cheap steps on an ~11-segment word
/// (≈ 1.4M work, inside the flat floor), while a divergent rule grinding
/// a growth-budget-bloated word pays its bloated length cubed per step and
/// trips within a few dozen steps instead of minutes of wall clock. The
/// `input_len⁴` term admits ~`len` productive steps at full length (a
/// harmony- or shift-style rule converges in at most ~one step per
/// segment), ×4 slack, so long real-world words don't false-positive.
/// Number of `Exprs` leaves under a block, in DFS order; the leaf
/// numbering shared with the FST gate.
fn leaf_count(block: &BlockIr) -> usize {
    match block {
        BlockIr::Exprs { .. } => 1,
        BlockIr::Sequential(children) | BlockIr::FirstMatching(children) => {
            children.iter().map(leaf_count).sum()
        }
        BlockIr::Propagate(inner) => leaf_count(inner),
        BlockIr::Filter { inner, .. } => leaf_count(inner),
    }
}

pub(crate) fn propagate_work_limit(input_len: usize) -> usize {
    6_000_000 + 4 * input_len.pow(4)
}
/// How large one rule application (an `ltr`/`rtl` scan or one `propagate`
/// fixpoint loop) may grow a word before we call it diverging. Kotlin
/// lexurgy has no such budget; its core spins forever on e.g.
/// `a* => a / _ u []` ltr and only the CLI's 1 s/step interrupt rescues it,
/// so any zero-width-source or growing-target rule under a rescanning mode
/// would otherwise hang us too. Real epenthesis/reduplication multiplies a
/// word by a small constant; 4× plus slack is far past any legitimate rule,
/// while tripping before backtracking costs blow up on the bloated word.
pub(crate) fn growth_limit(input_len: usize) -> usize {
    input_len * 4 + 100
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RunError {
    /// A pattern produced too many backtracking options.
    TooManyOptions,
    /// `$n` referenced before being bound.
    UnboundCapture(u32),
    /// `(x)$n` where `$n` is already bound.
    ReboundCapture(u32),
    /// `$Feature` used in output without being bound by the match.
    UnboundVariable,
    /// A rule output a feature bundle no symbol+diacritics can spell.
    InvalidMatrix,
    /// A `propagate` rule never settled (lexurgy's `LscDivergingPropagation`).
    DivergingPropagation,
    /// A rule kept growing the word past [`growth_limit`]. Lexurgy has no
    /// equivalent; its core diverges and the CLI kills it by wall clock.
    DivergingScan,
    /// The shapes on the two sides of `=>` don't line up.
    MismatchedEmitter,
    /// `$` somewhere it can't be resolved (input position).
    BoundaryInInput,
    /// `!x` where `x` isn't a single segment.
    MultiSegmentNegation,
    /// No way to divide the word into syllables
    /// (lexurgy's `SyllableStructureViolated`).
    SyllableStructure,
    /// Couldn't phonetically parse an input word.
    Word(String),
}

impl std::fmt::Display for RunError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RunError::TooManyOptions => write!(f, "too many possibilities while matching"),
            RunError::UnboundCapture(n) => {
                write!(
                    f,
                    "capture variable {} referenced before being bound",
                    n + 1
                )
            }
            RunError::ReboundCapture(n) => {
                write!(f, "capture variable {} bound more than once", n + 1)
            }
            RunError::UnboundVariable => {
                write!(f, "feature variable in output was never bound")
            }
            RunError::InvalidMatrix => {
                write!(f, "no symbol and diacritics spell the output of this rule")
            }
            RunError::DivergingPropagation => {
                write!(f, "propagating rule doesn't settle on a result")
            }
            RunError::DivergingScan => {
                write!(f, "rule applied indefinitely (the word keeps growing)")
            }
            RunError::MismatchedEmitter => {
                write!(f, "mismatched elements on the two sides of =>")
            }
            RunError::BoundaryInInput => {
                write!(f, "a word boundary can't appear in the input of a rule")
            }
            RunError::MultiSegmentNegation => {
                write!(f, "can't negate an element that isn't a single segment")
            }
            RunError::SyllableStructure => {
                write!(f, "the word doesn't fit the syllable structure")
            }
            RunError::Word(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for RunError {}

/// Feature-variable and capture bindings (lexurgy's `Bindings`). Later
/// bindings shadow earlier ones for the same key.
#[derive(Debug, Clone, Default)]
struct Bindings {
    features: Vec<(u16, u8)>,
    captures: Vec<(u32, Rc<Phrase>)>,
}

impl Bindings {
    fn feature(&self, feature: u16) -> Option<u8> {
        self.features
            .iter()
            .rev()
            .find(|(f, _)| *f == feature)
            .map(|&(_, code)| code)
    }

    fn bind_feature(&mut self, feature: u16, code: u8) {
        self.features.push((feature, code));
    }

    fn capture(&self, slot: u32) -> Option<&Rc<Phrase>> {
        self.captures
            .iter()
            .rev()
            .find(|(s, _)| *s == slot)
            .map(|(_, w)| w)
    }

    fn bind_capture(&mut self, slot: u32, phrase: Phrase) {
        self.captures.push((slot, Rc::new(phrase)));
    }
}

#[derive(Debug, Clone)]
struct MatchEnd {
    pos: usize,
    bindings: Bindings,
    /// Syllable breaks the pattern explicitly matched (positions of `.`),
    /// lexurgy's `matchedSyllableBreaks`.
    breaks: Vec<usize>,
}

impl MatchEnd {
    fn plain(pos: usize, bindings: Bindings) -> MatchEnd {
        MatchEnd {
            pos,
            bindings,
            breaks: Vec::new(),
        }
    }

    /// `PhraseMatchEnd.precededBy`: union matched breaks with an earlier
    /// match's.
    fn preceded_by(mut self, previous: &[usize]) -> MatchEnd {
        if !previous.is_empty() {
            let mut all = previous.to_vec();
            for b in self.breaks {
                if !all.contains(&b) {
                    all.push(b);
                }
            }
            self.breaks = all;
        }
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Dir {
    Fwd,
    Bwd,
}

/// A syllable-level feature change (a bound syllable matrix): overwrite the
/// mentioned fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SylChange {
    mask: u128,
    bits: u128,
}

/// A bound emitter result (lexurgy's `ChangeResult`): the replacement
/// phrase, positions where syllable breaks were *explicitly emitted*, and
/// explicit syllable feature changes by (phrase-linear) segment index.
#[derive(Debug, Clone)]
struct ChangeResult {
    phrase: Phrase,
    emits: Vec<usize>,
    changes: BTreeMap<usize, SylChange>,
    /// Lexurgy distinguishes a phrase with *no words* (an empty repeat's
    /// result) from one empty word (a deletion): the former is dropped from
    /// `fromSubTransformations`' neighbor logic entirely.
    vacuous: bool,
}

impl ChangeResult {
    fn of(phrase: Phrase) -> ChangeResult {
        ChangeResult {
            phrase,
            emits: vec![],
            changes: BTreeMap::new(),
            vacuous: false,
        }
    }
}

impl ChangeResult {
    fn emits_break_before(&self) -> bool {
        self.emits.first() == Some(&0)
    }

    fn emits_break_after(&self) -> bool {
        self.emits.last() == Some(&self.phrase.len())
    }
}

/// A pending replacement: `word[start..end]` becomes whatever `result`
/// produces once bound (lexurgy's `UnboundTransformation`).
#[derive(Debug, Clone)]
struct Transformation {
    start: usize,
    end: usize,
    result: ResultSpec,
    bindings: Bindings,
    /// Element-wise sub-transformations (sequence pairings); filtered rules
    /// need them to map each piece back to its real position.
    subs: Vec<Transformation>,
    /// Explicitly matched syllable breaks, suppressed in the output
    /// (lexurgy's `removesSyllableBreaks`).
    removes: Vec<usize>,
    /// Set by `unfilter` when the segment doesn't adjoin a real break.
    trim_leading: bool,
    trim_trailing: bool,
}

impl Transformation {
    fn removes_break_before(&self) -> bool {
        self.removes.first() == Some(&self.start)
    }

    fn removes_break_after(&self) -> bool {
        self.removes.last() == Some(&self.end)
    }
}

/// A delayed emission, bound with the final bindings after overlap
/// filtering (lexurgy's `UnboundResult`).
#[derive(Debug, Clone)]
enum ResultSpec {
    /// An independent emitter wrapped with the matched slice
    /// (`IndependentTransformer` / `IndependentSequenceTransformer`): the
    /// result recovers the slice's structure, except explicitly matched
    /// breaks.
    Independent {
        spec: IndepSpec,
        slice: Phrase,
        /// Matched breaks as flat segment indices relative to the slice.
        except: Vec<usize>,
    },
    /// `SymbolEmitter` paired conditionally: emitted word (floating
    /// transfer already applied), the *first word* of the matched slice
    /// (`ConditionalEmitter.result` takes `original.first()`), and the
    /// syllable feature change its diacritics carry.
    CondText {
        word: Word,
        original: Word,
        change: Option<SylChange>,
    },
    /// Matrix emitters (segment and/or syllable level).
    Matrix {
        update: MatrixUpdate,
        original: Word,
    },
    /// Element-wise concatenation of the sub-transformations
    /// (`UnboundTransformation.fromSubTransformations`).
    Subs,
}

/// What an independent emitter produces, before structure recovery.
#[derive(Debug, Clone)]
enum IndepSpec {
    /// Verbatim text (`TextEmitter` / independent `SymbolEmitter`).
    Text(Word),
    Empty,
    /// `.`: a bare syllable break.
    Boundary,
    /// `$$`: a word break (a phrase of two empty words).
    WordBreak,
    Capture(u32),
    SylCapture(u32),
    Seq(Vec<IndepSpec>),
}

pub struct Executor<'a> {
    decls: &'a Declarations,
    segments: &'a mut SegmentInterner,
    /// Mask of floating segment-level diacritics.
    floating: DiacriticMask,
    /// Union of all syllable-level feature fields.
    syl_fields: u128,
    /// FST-tier position gate: consulted before each `transform` attempt,
    /// skipping positions where no option of the expression can claim.
    /// Exact (never `false` where the VM would claim), so the output is
    /// identical with or without it, only faster.
    gate: Option<crate::fst::FstGate<'a>>,
}

pub(crate) fn emit_is_independent(emit: &Emit) -> bool {
    match emit {
        Emit::Text { .. }
        | Emit::Empty
        | Emit::CaptureRef { .. }
        | Emit::SylCaptureRef { .. }
        | Emit::SyllableBoundary
        | Emit::WordBreak => true,
        Emit::Matrix(_) => false,
        Emit::Seq(parts) | Emit::Alt(parts) => parts.iter().all(emit_is_independent),
    }
}

/// `SequenceMatcher` and `RepeaterMatcher` prefer independent emitters
/// (`a b => x` and `a* => x` emit one `x` for the whole match).
pub(crate) fn prefers_independent(pattern: &Pattern) -> bool {
    matches!(pattern, Pattern::Seq(_) | Pattern::Repeat { .. })
}

/// Only `RepeaterMatcher` prefers independent *sequence* emitters
/// (`a* => x y` emits one `x y`; `a b => x y` pairs element-wise).
fn prefers_independent_seq(pattern: &Pattern) -> bool {
    matches!(pattern, Pattern::Repeat { .. })
}

fn is_lifting(pattern: &Pattern) -> bool {
    matches!(
        pattern,
        Pattern::Capture { .. }
            | Pattern::Look { .. }
            | Pattern::Intersect(_)
            | Pattern::Repeat { .. }
    )
}

impl<'a> Executor<'a> {
    pub fn new(decls: &'a Declarations, segments: &'a mut SegmentInterner) -> Self {
        let floating = decls
            .diacritics
            .iter()
            .enumerate()
            .filter(|(_, d)| d.floating && d.level == Level::Segment)
            .fold(0, |mask, (i, _)| mask | (1 << i));
        let syl_fields = decls
            .features
            .defs()
            .filter(|(_, def)| def.level == Level::Syllable)
            .fold(0, |mask, (_, def)| mask | def.field_mask());
        Executor {
            decls,
            segments,
            floating,
            syl_fields,
            gate: None,
        }
    }

    pub fn with_gate(mut self, gate: crate::fst::FstGate<'a>) -> Self {
        self.gate = Some(gate);
        self
    }

    // syllable feature helpers

    /// The syllable feature word of the syllable containing linear `index`.
    fn syl_word_at(&self, phrase: &Phrase, index: usize) -> u128 {
        syl_mods_bits(self.decls, phrase.mods_at(index)).1
    }

    /// `word.syllableMatrix` of a slice: modifiers folded across syllables
    /// (and words) in order.
    fn slice_syl_word(&self, slice: &Phrase) -> u128 {
        let mut value = 0u128;
        for word in &slice.words {
            if let Some(syl) = &word.syl {
                for mods in syl.mods.values() {
                    let (mask, bits) = syl_mods_test(self.decls, mods);
                    value = value & !mask | bits;
                }
            }
        }
        value
    }

    /// Apply explicit syllable feature changes (keyed by segment index) to a
    /// word; lexurgy's `updateSyllableModifiers`.
    fn word_with_changes(
        &self,
        mut word: Word,
        changes: &BTreeMap<usize, SylChange>,
    ) -> Result<Word, RunError> {
        if word.syl.is_none() || changes.is_empty() {
            return Ok(word);
        }
        let pairs: Vec<(usize, SylChange)> = changes
            .iter()
            .filter(|(&i, _)| i < word.len().max(1))
            .map(|(&i, &c)| (word.syllable_number_at(i), c))
            .collect();
        let syl = word.syl.as_mut().unwrap();
        for (n, c) in pairs {
            let existing = syl.mods.get(&n).map_or(&[][..], |m| m.as_slice());
            let (_, value) = syl_mods_bits(self.decls, existing);
            let updated = value & !c.mask | c.bits;
            let mods = render_syl_mods(self.decls, updated).ok_or(RunError::InvalidMatrix)?;
            if mods.is_empty() {
                syl.mods.remove(&n);
            } else {
                syl.mods.insert(n, mods);
            }
        }
        Ok(word)
    }

    /// The phrase version: group changes by the word containing each linear
    /// index (lexurgy's `Phrase.updateSyllableModifiers`).
    fn phrase_with_changes(
        &self,
        phrase: Phrase,
        changes: &BTreeMap<usize, SylChange>,
    ) -> Result<Phrase, RunError> {
        if changes.is_empty() {
            return Ok(phrase);
        }
        let mut by_word: BTreeMap<usize, BTreeMap<usize, SylChange>> = BTreeMap::new();
        for (&i, &c) in changes {
            let (w, s) = phrase.locate(i);
            by_word.entry(w).or_default().insert(s, c);
        }
        let mut words = Vec::with_capacity(phrase.words.len());
        for (w, word) in phrase.words.into_iter().enumerate() {
            words.push(match by_word.get(&w) {
                Some(changes) => self.word_with_changes(word, changes)?,
                None => word,
            });
        }
        Ok(Phrase { words })
    }

    /// Resolve the syllable-level half of a matrix update against bindings.
    fn syl_change(
        &self,
        update: &MatrixUpdate,
        bindings: &Bindings,
    ) -> Result<Option<SylChange>, RunError> {
        let mut mask = update.syl_mask;
        let mut bits = update.syl_bits;
        for &feature in &update.vars {
            let def = self.decls.features.def(feature);
            if def.level != Level::Syllable {
                continue;
            }
            let code = bindings
                .feature(feature.0)
                .ok_or(RunError::UnboundVariable)?;
            mask |= def.field_mask();
            bits = bits & !def.field_mask() | def.encode(code);
        }
        Ok((mask != 0).then_some(SylChange { mask, bits }))
    }

    // blocks

    /// Run a block; `None` means it didn't match (lexurgy's nullable
    /// `ChangeRule.invoke`). Nested filters *compose*: a segment must pass
    /// every filter in scope.
    pub fn run_block<'b>(
        &mut self,
        block: &'b BlockIr,
        phrase: &Phrase,
        filters: &[&'b Pattern],
    ) -> Result<Option<Phrase>, RunError> {
        self.run_block_at(block, phrase, filters, 0)
    }

    /// `leaf` numbers the `Exprs` nodes in DFS order: the key the FST
    /// gate's leaves are indexed by (its block tree flattens filter
    /// wrappers but keeps the leaf order).
    fn run_block_at<'b>(
        &mut self,
        block: &'b BlockIr,
        phrase: &Phrase,
        filters: &[&'b Pattern],
        leaf: usize,
    ) -> Result<Option<Phrase>, RunError> {
        match block {
            BlockIr::Exprs { mode, exprs } => self.run_exprs(phrase, exprs, *mode, filters, leaf),
            BlockIr::Sequential(children) => {
                let mut matched = false;
                let mut current = phrase.clone();
                let mut base = leaf;
                for child in children {
                    if let Some(next) = self.run_block_at(child, &current, filters, base)? {
                        matched = true;
                        current = next;
                    }
                    base += leaf_count(child);
                }
                Ok(matched.then_some(current))
            }
            // Lexurgy wraps every `FirstMatchingBlock` in a
            // `WithinWordBlock`: each word picks its own first-matching arm.
            BlockIr::FirstMatching(children) => {
                let mut matched = false;
                let mut words = Vec::with_capacity(phrase.words.len());
                for word in &phrase.words {
                    let single = Phrase::single(word.clone());
                    let mut out = word.clone();
                    let mut base = leaf;
                    for child in children {
                        let arm = self.run_block_at(child, &single, filters, base)?;
                        base += leaf_count(child);
                        if let Some(result) = arm {
                            // kotlin calls `.single()` on the result, which
                            // throws (a per-word error) if a rule inside the
                            // block split the word.
                            if result.words.len() != 1 {
                                return Err(RunError::Word(
                                    "a rule in an Else: block split a word".to_string(),
                                ));
                            }
                            out = result.words.into_iter().next().unwrap();
                            matched = true;
                            break;
                        }
                    }
                    words.push(out);
                }
                Ok(matched.then_some(Phrase { words }))
            }
            BlockIr::Propagate(inner) => {
                let mut current = phrase.clone();
                let mut seen: HashSet<Phrase> = HashSet::new();
                seen.insert(current.clone());
                let mut first_step = true;
                let mut work = 0usize;
                let work_limit = propagate_work_limit(phrase.len());
                loop {
                    match self.run_block_at(inner, &current, filters, leaf)? {
                        None => return Ok(if first_step { None } else { Some(current) }),
                        Some(next) => {
                            if next == current {
                                return Ok(Some(next));
                            }
                            work += next.len().max(1).pow(3);
                            if !seen.insert(next.clone())
                                || work > work_limit
                                // A growing target under propagate diverges
                                // by construction; without a growth budget
                                // the seen-set OOMs long before the work cap.
                                || next.len() > growth_limit(phrase.len())
                            {
                                return Err(RunError::DivergingPropagation);
                            }
                            current = next;
                            first_step = false;
                        }
                    }
                }
            }
            BlockIr::Filter {
                filter: new_filter,
                inner,
            } => {
                let mut nested: Vec<&'b Pattern> = filters.to_vec();
                nested.push(new_filter);
                self.run_block_at(inner, phrase, &nested, leaf)
            }
        }
    }

    // expression lists (SimpleChangeRule)

    fn run_exprs(
        &mut self,
        phrase: &Phrase,
        exprs: &[ExprIr],
        mode: MatchMode,
        filters: &[&Pattern],
        leaf: usize,
    ) -> Result<Option<Phrase>, RunError> {
        match mode {
            MatchMode::Simultaneous => {
                let (fphrase, fmap) = self.filter_phrase(phrase, filters)?;
                let mut all = Vec::new();
                for (ei, expr) in exprs.iter().enumerate() {
                    all.extend(self.claim_all(&fphrase, expr, (leaf, ei))?);
                }
                let kept = filter_overlapping(all);
                let real = unfilter(phrase, fmap.as_deref(), kept)?;
                if real.is_empty() {
                    return Ok(None);
                }
                Ok(Some(self.apply_transformations(phrase, real)?))
            }
            // Directional rules always "match" (lexurgy returns the phrase
            // unconditionally), which matters for Else: blocks.
            MatchMode::Ltr => {
                let limit = growth_limit(phrase.len());
                let mut current = phrase.clone();
                let mut index = 0;
                // The cursor advances one position per step, so the scan
                // can only diverge by growing the phrase at least as fast as
                // the cursor moves; a growth budget is a full divergence
                // budget here. (Gap slots count toward the length, so a rule
                // that only inserts `$$` forever still trips it.)
                while index <= current.len() {
                    current = self.transform_once_at(&current, exprs, index, filters, leaf)?;
                    if current.len() > limit {
                        return Err(RunError::DivergingScan);
                    }
                    index += 1;
                }
                Ok(Some(current))
            }
            MatchMode::Rtl => {
                let limit = growth_limit(phrase.len());
                let mut current = phrase.clone();
                let mut index = current.len();
                loop {
                    current = self.transform_once_at(&current, exprs, index, filters, leaf)?;
                    if current.len() > limit {
                        return Err(RunError::DivergingScan);
                    }
                    if index == 0 {
                        break;
                    }
                    index -= 1;
                }
                Ok(Some(current))
            }
        }
    }

    fn transform_once_at(
        &mut self,
        phrase: &Phrase,
        exprs: &[ExprIr],
        index: usize,
        filters: &[&Pattern],
        leaf: usize,
    ) -> Result<Phrase, RunError> {
        let (fphrase, fmap) = self.filter_phrase(phrase, filters)?;
        let findex = match &fmap {
            None => index,
            Some(maps) => {
                // Convert a real position into filtered coordinates: the
                // same word, at the filtered index of this segment. Word
                // ends (and segments that don't pass the filter) can't
                // match (lexurgy's `indexOf` returns -1 there).
                let (w, s) = phrase.locate(index);
                match maps[w].iter().position(|&real| real == s) {
                    Some(fi) => fphrase.word_start(w) + fi,
                    None => return Ok(phrase.clone()),
                }
            }
        };
        if findex > fphrase.len() {
            return Ok(phrase.clone());
        }
        for (ei, expr) in exprs.iter().enumerate() {
            if let Some(t) = self.claim_expr_at(&fphrase, expr, findex, (leaf, ei))? {
                let real = unfilter(phrase, fmap.as_deref(), vec![t])?;
                return self.apply_transformations(phrase, real);
            }
        }
        Ok(phrase.clone())
    }

    fn filter_phrase(
        &mut self,
        phrase: &Phrase,
        filters: &[&Pattern],
    ) -> Result<(Phrase, Option<FilterMaps>), RunError> {
        if filters.is_empty() {
            return Ok((phrase.clone(), None));
        }
        let mut words = Vec::with_capacity(phrase.words.len());
        let mut maps = Vec::with_capacity(phrase.words.len());
        for word in &phrase.words {
            let mut fmap = Vec::new();
            for (i, &seg) in word.segs.iter().enumerate() {
                // Lexurgy tests filters on each segment in isolation (an
                // unsyllabified single-segment word).
                let alone = Phrase::single(Word::simple(vec![seg]));
                let mut passes = true;
                for pattern in filters {
                    if self
                        .claim(&alone, pattern, 0, &Bindings::default(), Dir::Fwd)?
                        .is_empty()
                    {
                        passes = false;
                        break;
                    }
                }
                if passes {
                    fmap.push(i);
                }
            }
            words.push(word.retain_indices(&fmap));
            maps.push(fmap);
        }
        Ok((Phrase { words }, Some(maps)))
    }

    /// All matches of one expression, scanning left to right and restarting
    /// one position after each match's *start* (lexurgy's `claimAll`).
    fn claim_all(
        &mut self,
        phrase: &Phrase,
        expr: &ExprIr,
        key: (usize, usize),
    ) -> Result<Vec<Transformation>, RunError> {
        let mut out = Vec::new();
        let mut from = 0;
        while from <= phrase.len() {
            let mut matched = None;
            for pos in from..=phrase.len() {
                if let Some(t) = self.claim_expr_at(phrase, expr, pos, key)? {
                    matched = Some(t);
                    break;
                }
            }
            // Resume scanning one position past the match's start, or stop
            // once a full pass found nothing.
            match matched {
                Some(t) => {
                    from = t.start + 1;
                    out.push(t);
                }
                None => break,
            }
        }
        Ok(out)
    }

    /// Match one expression anchored at `pos`: the first transformation
    /// option whose environment passes.
    fn claim_expr_at(
        &mut self,
        phrase: &Phrase,
        expr: &ExprIr,
        pos: usize,
        key: (usize, usize),
    ) -> Result<Option<Transformation>, RunError> {
        if let Some(gate) = &mut self.gate {
            let (w, s) = phrase.locate(pos);
            if !gate.may_claim(key.0, key.1, self.decls, self.segments, &phrase.words[w], s) {
                return Ok(None);
            }
        }
        let bindings = Bindings::default();
        let options = self.transform(phrase, &expr.from, &expr.to, pos, &bindings)?;
        for option in options {
            if let Some(bound) = self.check_envs(
                phrase,
                option.start,
                option.end,
                &expr.condition,
                &expr.exclusion,
                &option.bindings,
            )? {
                return Ok(Some(Transformation {
                    bindings: bound,
                    ..option
                }));
            }
        }
        Ok(None)
    }

    /// Lexurgy's `applyTransformations`: build the result word by
    /// alternating existing slices and bound results, stitching syllable
    /// modifiers at merged boundaries.
    fn apply_transformations(
        &mut self,
        phrase: &Phrase,
        mut transformations: Vec<Transformation>,
    ) -> Result<Phrase, RunError> {
        // Stable sort: precedence order survives among equal starts.
        transformations.sort_by_key(|t| t.start);
        let mut result = Phrase::default();
        // The previously applied transformation: (removes break after?,
        // changes, bound result phrase).
        let mut prev: Option<(bool, BTreeMap<usize, SylChange>, Phrase)> = None;
        let mut cursor = 0usize;

        for t in transformations {
            if cursor > t.start {
                continue;
            }
            result = self.concat_existing(result, phrase.slice(cursor, t.start), &prev)?;
            if t.removes_break_before() {
                result = result.remove_trailing_break();
            }
            let cr = self.bind(&t)?;
            // `finalResult`: apply the explicit changes to the result.
            let newbit = self.phrase_with_changes(cr.phrase.clone(), &cr.changes)?;
            result = self.concat_new(result, &newbit, &cr.changes)?;
            prev = Some((t.removes_break_after(), cr.changes, newbit));
            cursor = t.end;
        }
        result = self.concat_existing(result, phrase.slice(cursor, phrase.len()), &prev)?;
        Ok(result)
    }

    /// `addExistingSlice`: append an untouched slice of the original phrase,
    /// merging the boundary syllable's modifiers when the previous
    /// transformation's syllable bleeds into it.
    fn concat_existing(
        &self,
        result: Phrase,
        slice: Phrase,
        prev: &Option<(bool, BTreeMap<usize, SylChange>, Phrase)>,
    ) -> Result<Phrase, RunError> {
        let Some((removes_after, changes, newbit)) = prev else {
            return Ok(result.concat(&slice, |_, right| right.to_vec()));
        };
        let slice = if *removes_after {
            slice.remove_leading_break()
        } else {
            slice
        };
        let failed = std::cell::Cell::new(false);
        let combined = result.concat(&slice, |left, right| {
            // right.toMatrix().update(left.toMatrix()), then re-apply the
            // transformation's changes from the end of its result backwards
            // until a syllable break (word starts count as breaks).
            let (lm, lw) = syl_mods_test(self.decls, left);
            let (_, rw) = syl_mods_test(self.decls, right);
            let mut value = rw & !lm | lw;
            for i in (0..=newbit.len()).rev() {
                if let Some(c) = changes.get(&i) {
                    value = value & !c.mask | c.bits;
                }
                if i == 0 || newbit.has_break_before(i) {
                    break;
                }
            }
            match render_syl_mods(self.decls, value) {
                Some(mods) => mods,
                None => {
                    failed.set(true);
                    Vec::new()
                }
            }
        });
        if failed.get() {
            return Err(RunError::InvalidMatrix);
        }
        Ok(combined)
    }

    /// Append a transformation's result, merging the boundary syllable's
    /// modifiers and re-applying the result's leading changes.
    fn concat_new(
        &self,
        result: Phrase,
        newbit: &Phrase,
        changes: &BTreeMap<usize, SylChange>,
    ) -> Result<Phrase, RunError> {
        let failed = std::cell::Cell::new(false);
        let combined = result.concat(newbit, |left, right| {
            let (_, lw) = syl_mods_test(self.decls, left);
            let (rm, rw) = syl_mods_test(self.decls, right);
            let mut value = lw & !rm | rw;
            for i in 0..=newbit.len() {
                if let Some(c) = changes.get(&i) {
                    value = value & !c.mask | c.bits;
                }
                // hasSyllableBreakAfter(i)
                if newbit.has_break_after(i) {
                    break;
                }
            }
            match render_syl_mods(self.decls, value) {
                Some(mods) => mods,
                None => {
                    failed.set(true);
                    Vec::new()
                }
            }
        });
        if failed.get() {
            return Err(RunError::InvalidMatrix);
        }
        Ok(combined)
    }

    // environments

    fn check_envs(
        &mut self,
        phrase: &Phrase,
        lo: usize,
        hi: usize,
        condition: &[EnvIr],
        exclusion: &[EnvIr],
        bindings: &Bindings,
    ) -> Result<Option<Bindings>, RunError> {
        let positive = if condition.is_empty() {
            Some(bindings.clone())
        } else {
            let mut found = None;
            for env in condition {
                if let Some(b) = self.check_env(phrase, lo, hi, env, bindings)? {
                    found = Some(b);
                    break;
                }
            }
            found
        };
        let Some(positive) = positive else {
            return Ok(None);
        };
        for env in exclusion {
            if self.check_env(phrase, lo, hi, env, &positive)?.is_some() {
                return Ok(None);
            }
        }
        Ok(Some(positive))
    }

    fn check_env(
        &mut self,
        phrase: &Phrase,
        lo: usize,
        hi: usize,
        env: &EnvIr,
        bindings: &Bindings,
    ) -> Result<Option<Bindings>, RunError> {
        let after_before = match &env.before {
            None => bindings.clone(),
            Some(pattern) => {
                match self
                    .claim(phrase, pattern, lo, bindings, Dir::Bwd)?
                    .into_iter()
                    .next()
                {
                    Some(end) => end.bindings,
                    None => return Ok(None),
                }
            }
        };
        match &env.after {
            None => Ok(Some(after_before)),
            Some(pattern) => Ok(self
                .claim(phrase, pattern, hi, &after_before, Dir::Fwd)?
                .into_iter()
                .next()
                .map(|end| end.bindings)),
        }
    }

    // matching

    /// All possible match ends of `pattern` from `pos`, in preference
    /// order. `Bwd` walks leftward (mirrored sequences, original
    /// coordinates): used for lookbehind.
    fn claim(
        &mut self,
        phrase: &Phrase,
        pattern: &Pattern,
        pos: usize,
        bindings: &Bindings,
        dir: Dir,
    ) -> Result<Vec<MatchEnd>, RunError> {
        Ok(match pattern {
            Pattern::Empty => vec![MatchEnd::plain(pos, bindings.clone())],
            Pattern::Test(test) => {
                let Some(index) = self.index_at(phrase, pos, dir) else {
                    return Ok(vec![]);
                };
                match self.test_seg(test, phrase, index, bindings) {
                    Some(b) => vec![MatchEnd::plain(step(pos, dir), b)],
                    None => vec![],
                }
            }
            Pattern::Text(text) => self.claim_text(phrase, text, pos, bindings, dir)?,
            Pattern::Seq(parts) => self.claim_seq(phrase, parts, pos, bindings, dir)?,
            Pattern::Alt(parts) => {
                let mut ends = Vec::new();
                for part in parts {
                    ends.extend(self.claim(phrase, part, pos, bindings, dir)?);
                }
                ends
            }
            Pattern::Repeat { inner, min, max } => {
                let mut levels = vec![vec![MatchEnd::plain(pos, bindings.clone())]];
                loop {
                    let mut next = Vec::new();
                    for end in levels.last().unwrap() {
                        next.extend(
                            self.claim(phrase, inner, end.pos, &end.bindings, dir)?
                                .into_iter()
                                .map(|e| e.preceded_by(&end.breaks)),
                        );
                    }
                    if next.is_empty() {
                        break;
                    }
                    if next.len() > MAX_OPTIONS {
                        return Err(RunError::TooManyOptions);
                    }
                    levels.push(next);
                    if let Some(max) = max {
                        if levels.len() > *max as usize {
                            break;
                        }
                    }
                    // a zero-width inner pattern would repeat forever
                    if levels.len() > phrase.len() + 2 {
                        break;
                    }
                }
                if levels.len() <= *min as usize {
                    vec![]
                } else {
                    // greedy: longest first
                    levels.drain(*min as usize..).rev().flatten().collect()
                }
            }
            Pattern::Capture { inner, slot } => {
                if bindings.capture(slot.0).is_some() {
                    return Err(RunError::ReboundCapture(slot.0));
                }
                self.claim(phrase, inner, pos, bindings, dir)?
                    .into_iter()
                    .map(|end| {
                        let (lo, hi) = span(pos, end.pos);
                        let mut e = end;
                        e.bindings.bind_capture(slot.0, phrase.slice(lo, hi));
                        e
                    })
                    .collect()
            }
            Pattern::CaptureRef { slot, inexact } => {
                self.claim_capture_ref(phrase, slot.0, *inexact, pos, bindings, dir)?
            }
            Pattern::Not(inner) => {
                if self.pattern_length(inner, bindings) != Some(1) {
                    return Err(RunError::MultiSegmentNegation);
                }
                if self.index_at(phrase, pos, dir).is_none() {
                    return Ok(vec![]);
                }
                if self.claim(phrase, inner, pos, bindings, dir)?.is_empty() {
                    vec![MatchEnd::plain(step(pos, dir), bindings.clone())]
                } else {
                    vec![]
                }
            }
            Pattern::NotAhead(inner) => {
                // NegatedLookaroundMatcher: zero-width, requires a segment
                // to exist at the probe position.
                if self.index_at(phrase, pos, dir).is_none() {
                    return Ok(vec![]);
                }
                if self.claim(phrase, inner, pos, bindings, dir)?.is_empty() {
                    vec![MatchEnd::plain(pos, bindings.clone())]
                } else {
                    vec![]
                }
            }
            Pattern::Intersect(parts) => self.claim_intersect(phrase, parts, pos, bindings, dir)?,
            Pattern::Look {
                inner,
                condition,
                exclusion,
            } => {
                let ends = self.claim(phrase, inner, pos, bindings, dir)?;
                let mut out = Vec::new();
                for end in ends {
                    let (lo, hi) = span(pos, end.pos);
                    if let Some(b) =
                        self.check_envs(phrase, lo, hi, condition, exclusion, &end.bindings)?
                    {
                        out.push(MatchEnd {
                            pos: end.pos,
                            bindings: b,
                            breaks: end.breaks,
                        });
                    }
                }
                out
            }
            // `$` matches at *every* word's edges in a phrase
            // (`WordStartMatcher` / `WordEndMatcher` test the segment index
            // within the word).
            Pattern::WordStart => {
                let (_, s) = phrase.locate(pos);
                zero_width(s == 0, pos, bindings)
            }
            Pattern::WordEnd => {
                let (w, s) = phrase.locate(pos);
                zero_width(s == phrase.words[w].len(), pos, bindings)
            }
            Pattern::WordBoundary => return Err(RunError::BoundaryInInput),
            // `$$`: zero-length, but steps across the gap (end of word `w`
            // → start of word `w + 1`); mirrored for lookbehind.
            Pattern::BetweenWords => {
                let (w, s) = phrase.locate(pos);
                match dir {
                    Dir::Fwd => {
                        if s == phrase.words[w].len() && w + 1 < phrase.words.len() {
                            vec![MatchEnd::plain(pos + 1, bindings.clone())]
                        } else {
                            vec![]
                        }
                    }
                    Dir::Bwd => {
                        if s == 0 && w > 0 {
                            vec![MatchEnd::plain(pos - 1, bindings.clone())]
                        } else {
                            vec![]
                        }
                    }
                }
            }
            Pattern::SyllableBoundary => {
                if phrase.has_boundary_at(pos) {
                    vec![MatchEnd {
                        pos,
                        bindings: bindings.clone(),
                        breaks: vec![pos],
                    }]
                } else {
                    vec![]
                }
            }
            Pattern::NoBoundary => zero_width(!phrase.has_boundary_at(pos), pos, bindings),
            Pattern::AnySyllable => {
                let (w, s) = phrase.locate(pos);
                let word = &phrase.words[w];
                let end = match dir {
                    Dir::Fwd => self.syllable_from(word, s),
                    Dir::Bwd => self.syllable_back_from(word, s),
                };
                match end {
                    Some(end) => vec![MatchEnd::plain(
                        phrase.word_start(w) + end,
                        bindings.clone(),
                    )],
                    None => vec![],
                }
            }
        })
    }

    /// `SequenceMatcher` over a literal multi-segment unit: every `SegTest`
    /// must match in turn, with the optional syllable-feature mask checked
    /// over the whole span.
    fn claim_text(
        &mut self,
        phrase: &Phrase,
        text: &TextIr,
        pos: usize,
        bindings: &Bindings,
        dir: Dir,
    ) -> Result<Vec<MatchEnd>, RunError> {
        let mut p = pos;
        let mut b = bindings.clone();
        let order: Vec<&SegTest> = match dir {
            Dir::Fwd => text.tests.iter().collect(),
            Dir::Bwd => text.tests.iter().rev().collect(),
        };
        for test in order {
            let Some(index) = self.index_at(phrase, p, dir) else {
                return Ok(vec![]);
            };
            match self.test_seg(test, phrase, index, &b) {
                Some(nb) => {
                    b = nb;
                    p = step(p, dir);
                }
                None => return Ok(vec![]),
            }
        }
        if text.syl_mask != 0 {
            let (lo, hi) = span(pos, p);
            let target = self.slice_syl_word(&phrase.slice(lo, hi));
            if target & text.syl_mask != text.syl_want {
                return Ok(vec![]);
            }
        }
        Ok(vec![MatchEnd::plain(p, b)])
    }

    /// `SequenceMatcher`: thread the running match ends through each part in
    /// turn (`Bwd` reverses the order, original coordinates).
    fn claim_seq(
        &mut self,
        phrase: &Phrase,
        parts: &[Pattern],
        pos: usize,
        bindings: &Bindings,
        dir: Dir,
    ) -> Result<Vec<MatchEnd>, RunError> {
        let mut ends = vec![MatchEnd::plain(pos, bindings.clone())];
        let order: Vec<&Pattern> = match dir {
            Dir::Fwd => parts.iter().collect(),
            Dir::Bwd => parts.iter().rev().collect(),
        };
        for part in order {
            let mut next = Vec::new();
            for end in &ends {
                next.extend(
                    self.claim(phrase, part, end.pos, &end.bindings, dir)?
                        .into_iter()
                        .map(|e| e.preceded_by(&end.breaks)),
                );
            }
            if next.len() > MAX_OPTIONS {
                return Err(RunError::TooManyOptions);
            }
            if next.is_empty() {
                return Ok(vec![]);
            }
            ends = next;
        }
        Ok(ends)
    }

    /// `CaptureReferenceMatcher`: re-match a captured sub-phrase
    /// position-by-position (gaps must line up, `inexact` compares with
    /// diacritics stripped, lexurgy's `matchSubPhrase`).
    fn claim_capture_ref(
        &mut self,
        phrase: &Phrase,
        slot: u32,
        inexact: bool,
        pos: usize,
        bindings: &Bindings,
        dir: Dir,
    ) -> Result<Vec<MatchEnd>, RunError> {
        let captured = bindings
            .capture(slot)
            .ok_or(RunError::UnboundCapture(slot))?
            .clone();
        let n = captured.len();
        let start = match dir {
            Dir::Fwd if pos + n <= phrase.len() => pos,
            Dir::Bwd if pos >= n => pos - n,
            _ => return Ok(vec![]),
        };
        let mut ok = true;
        for i in 0..n {
            match (phrase.seg_at(start + i), captured.seg_at(i)) {
                (None, None) => {}
                (Some(w), Some(c)) => {
                    let (w, c) = if inexact {
                        (
                            self.segments
                                .without_diacritics(self.decls, w, self.floating),
                            self.segments
                                .without_diacritics(self.decls, c, self.floating),
                        )
                    } else {
                        (w, c)
                    };
                    if w != c {
                        ok = false;
                        break;
                    }
                }
                _ => {
                    ok = false;
                    break;
                }
            }
        }
        Ok(if ok {
            vec![MatchEnd::plain(
                match dir {
                    Dir::Fwd => pos + n,
                    Dir::Bwd => pos - n,
                },
                bindings.clone(),
            )]
        } else {
            vec![]
        })
    }

    /// `IntersectionMatcher`: the first part fixes the span; the rest are
    /// re-verified over that span (`verify_intersection`).
    fn claim_intersect(
        &mut self,
        phrase: &Phrase,
        parts: &[Pattern],
        pos: usize,
        bindings: &Bindings,
        dir: Dir,
    ) -> Result<Vec<MatchEnd>, RunError> {
        let (first, rest) = parts.split_first().expect("empty intersection");
        let ends = self.claim(phrase, first, pos, bindings, dir)?;
        let mut out = Vec::new();
        for end in ends {
            if let Some(b) =
                self.verify_intersection(phrase, rest, pos, end.pos, &end.bindings, dir)?
            {
                out.push(MatchEnd {
                    pos: end.pos,
                    bindings: b,
                    breaks: end.breaks,
                });
            }
        }
        Ok(out)
    }

    /// `SyllableMatcher`: match exactly one whole syllable rightward.
    fn syllable_from(&self, word: &Word, pos: usize) -> Option<usize> {
        if !word.is_syllabified() || word.is_empty() {
            return None;
        }
        let breaks = word.syllable_breaks();
        if pos == 0 && word.num_syllables() == 1 {
            return Some(word.len());
        }
        if pos == 0 {
            return Some(breaks[0]);
        }
        let bi = breaks.iter().position(|&b| b == pos)?;
        if bi + 1 == breaks.len() {
            Some(word.len())
        } else {
            Some(breaks[bi + 1])
        }
    }

    /// The mirror image (the reversed matcher on the reversed word).
    fn syllable_back_from(&self, word: &Word, pos: usize) -> Option<usize> {
        if !word.is_syllabified() || word.is_empty() {
            return None;
        }
        let breaks = word.syllable_breaks();
        if pos == word.len() && word.num_syllables() == 1 {
            return Some(0);
        }
        if pos == word.len() {
            return Some(*breaks.last()?);
        }
        let bi = breaks.iter().position(|&b| b == pos)?;
        if bi == 0 {
            Some(0)
        } else {
            Some(breaks[bi - 1])
        }
    }

    /// Subsequent `&`-parts must match the same span as the first; `&!`
    /// parts must *not* (lexurgy's match verifiers). Pure syllable matrices
    /// are length-hinted: they accept any span within one syllable.
    fn verify_intersection(
        &mut self,
        phrase: &Phrase,
        parts: &[Pattern],
        start: usize,
        end: usize,
        bindings: &Bindings,
        dir: Dir,
    ) -> Result<Option<Bindings>, RunError> {
        let mut bindings = bindings.clone();
        for part in parts {
            match part {
                Pattern::Test(SegTest::SylMatrix(test)) => {
                    match self.hinted_syl_claim(phrase, test, start, end, &bindings, dir) {
                        Some(b) => bindings = b,
                        None => return Ok(None),
                    }
                }
                Pattern::Not(inner) if matches!(**inner, Pattern::Test(SegTest::SylMatrix(_))) => {
                    let Pattern::Test(SegTest::SylMatrix(test)) = &**inner else {
                        unreachable!()
                    };
                    if self
                        .hinted_syl_claim(phrase, test, start, end, &bindings, dir)
                        .is_some()
                    {
                        return Ok(None);
                    }
                }
                Pattern::Not(inner) => {
                    let hit = self
                        .claim(phrase, inner, start, &bindings, dir)?
                        .iter()
                        .any(|e| e.pos == end);
                    if hit {
                        return Ok(None);
                    }
                }
                _ => {
                    // lexurgy keeps the *last* end at the expected position
                    // (`associate` overwrites earlier entries)
                    match self
                        .claim(phrase, part, start, &bindings, dir)?
                        .into_iter()
                        .filter(|e| e.pos == end)
                        .last()
                    {
                        Some(e) => bindings = e.bindings,
                        None => return Ok(None),
                    }
                }
            }
        }
        Ok(Some(bindings))
    }

    /// `SyllableMatrixMatcher`'s length-hinted claim: the span must stay
    /// within one word and one syllable, and the syllable's features must
    /// match.
    fn hinted_syl_claim(
        &self,
        phrase: &Phrase,
        test: &MatrixTest,
        start: usize,
        end: usize,
        bindings: &Bindings,
        dir: Dir,
    ) -> Option<Bindings> {
        let (lo, hi) = span(start, end);
        let (wl, sl) = phrase.locate(lo);
        let (wh, sh) = phrase.locate(hi);
        // Lexurgy's length-hinted claim rejects spans that cross words.
        if wl != wh {
            return None;
        }
        if phrase.words[wl]
            .syllable_breaks()
            .iter()
            .any(|&b| b > sl && b < sh)
        {
            return None;
        }
        let index = match dir {
            Dir::Fwd => lo,
            Dir::Bwd => hi.saturating_sub(1),
        };
        self.test_syl_matrix(test, phrase, index, bindings)
    }

    fn index_at(&self, phrase: &Phrase, pos: usize, dir: Dir) -> Option<usize> {
        match dir {
            Dir::Fwd => phrase.seg_at(pos).map(|_| pos),
            Dir::Bwd => pos.checked_sub(1).filter(|&p| phrase.seg_at(p).is_some()),
        }
    }

    fn test_seg(
        &self,
        test: &SegTest,
        phrase: &Phrase,
        index: usize,
        bindings: &Bindings,
    ) -> Option<Bindings> {
        let seg = phrase
            .seg_at(index)
            .expect("test_seg at a segment position");
        match test {
            SegTest::Any => Some(bindings.clone()),
            SegTest::Exact(id) => (seg == *id).then(|| bindings.clone()),
            SegTest::Literal {
                id,
                core,
                required,
                allowed,
            } => {
                if seg == *id {
                    return Some(bindings.clone());
                }
                let data = self.segments.get(seg);
                (data.core == *core
                    && data.diacritics & required == *required
                    && data.diacritics & !allowed == 0)
                    .then(|| bindings.clone())
            }
            SegTest::Matrix(test) => {
                let features = self.segments.get(seg).features;
                if !test.seg.matches(features) {
                    return None;
                }
                self.bind_vars(test, features.0, bindings)
            }
            SegTest::SylMatrix(test) => self.test_syl_matrix(test, phrase, index, bindings),
        }
    }

    fn test_syl_matrix(
        &self,
        test: &MatrixTest,
        phrase: &Phrase,
        index: usize,
        bindings: &Bindings,
    ) -> Option<Bindings> {
        let value = self.syl_word_at(phrase, index);
        if !test.syl.matches(FeatureWord(value)) {
            return None;
        }
        self.bind_vars(test, value, bindings)
    }

    /// Bind or check the feature variables of a matrix test against the
    /// feature word they read from.
    fn bind_vars(&self, test: &MatrixTest, value: u128, bindings: &Bindings) -> Option<Bindings> {
        let mut bindings = bindings.clone();
        for var in &test.vars {
            let def = self.decls.features.def(var.feature);
            let code = def.extract(FeatureWord(value));
            match bindings.feature(var.feature.0) {
                Some(bound) => {
                    let ok = if var.negated {
                        code != bound
                    } else {
                        code == bound
                    };
                    if !ok {
                        return None;
                    }
                }
                None => {
                    // A negated variable never matches unbound (lexurgy's
                    // `matchesNonBinding` discards the inner binding and
                    // negates the always-true result).
                    if var.negated {
                        return None;
                    }
                    bindings.bind_feature(var.feature.0, code);
                }
            }
        }
        Some(bindings)
    }

    fn pattern_length(&self, pattern: &Pattern, bindings: &Bindings) -> Option<usize> {
        match pattern {
            Pattern::Test(_) | Pattern::Not(_) => Some(1),
            Pattern::Text(text) => Some(text.tests.len()),
            Pattern::Empty
            | Pattern::WordStart
            | Pattern::WordEnd
            | Pattern::WordBoundary
            | Pattern::BetweenWords
            | Pattern::NotAhead(_)
            | Pattern::NoBoundary
            | Pattern::SyllableBoundary => Some(0),
            Pattern::Seq(parts) => {
                let mut total = 0;
                for part in parts {
                    total += self.pattern_length(part, bindings)?;
                }
                Some(total)
            }
            Pattern::Alt(parts) => {
                let lengths: Vec<_> = parts
                    .iter()
                    .map(|p| self.pattern_length(p, bindings))
                    .collect::<Option<_>>()?;
                lengths.windows(2).all(|w| w[0] == w[1]).then(|| lengths[0])
            }
            Pattern::Repeat { .. } | Pattern::AnySyllable => None,
            Pattern::Capture { inner, .. } | Pattern::Look { inner, .. } => {
                self.pattern_length(inner, bindings)
            }
            Pattern::CaptureRef { slot, .. } => bindings.capture(slot.0).map(|w| w.len()),
            Pattern::Intersect(parts) => self.pattern_length(&parts[0], bindings),
        }
    }

    // transforming (from/to pairing)

    /// All transformation options of `from => to` anchored at `pos`, in
    /// preference order; the port of `Matcher.transformerTo` dispatch plus
    /// the individual `Transformer`s.
    fn transform(
        &mut self,
        phrase: &Phrase,
        pattern: &Pattern,
        emit: &Emit,
        pos: usize,
        bindings: &Bindings,
    ) -> Result<Vec<Transformation>, RunError> {
        match emit {
            Emit::Alt(alternatives) => {
                self.transform_alt(phrase, pattern, emit, alternatives, pos, bindings)
            }
            Emit::Seq(parts) => self.transform_seq(phrase, pattern, emit, parts, pos, bindings),
            Emit::Matrix(_) => self.transform_conditional(phrase, pattern, emit, pos, bindings),
            // Exact text is lexurgy's `TextEmitter`: independent only.
            Emit::Text { exact, .. } => {
                if *exact || prefers_independent(pattern) {
                    self.transform_independent(phrase, pattern, emit, pos, bindings)
                } else {
                    self.transform_conditional(phrase, pattern, emit, pos, bindings)
                }
            }
            Emit::Empty
            | Emit::CaptureRef { .. }
            | Emit::SylCaptureRef { .. }
            | Emit::SyllableBoundary
            | Emit::WordBreak => self.transform_independent(phrase, pattern, emit, pos, bindings),
        }
    }

    /// Pair an alternation emitter `{...}` (lexurgy's
    /// `transformerToAlternatives` dispatch): positionally when the counts
    /// match, otherwise distributing the whole emitter into each branch, or
    /// spreading sequences into adjacent alternative lists.
    fn transform_alt(
        &mut self,
        phrase: &Phrase,
        pattern: &Pattern,
        emit: &Emit,
        alternatives: &[Emit],
        pos: usize,
        bindings: &Bindings,
    ) -> Result<Vec<Transformation>, RunError> {
        if is_lifting(pattern) {
            return self.transform_lifting(phrase, pattern, emit, pos, bindings);
        }
        match pattern {
            Pattern::Alt(parts) if parts.len() == alternatives.len() => {
                let mut out = Vec::new();
                for (part, alt) in parts.iter().zip(alternatives) {
                    out.extend(self.transform(phrase, part, alt, pos, bindings)?);
                }
                Ok(out)
            }
            // size mismatch: each branch gets the whole emitter
            // (lexurgy's fallback in `transformerToAlternatives`:
            // a nested alternative may pair with it on its own)
            Pattern::Alt(parts) => {
                let mut out = Vec::new();
                for part in parts {
                    out.extend(self.transform(phrase, part, emit, pos, bindings)?);
                }
                Ok(out)
            }
            // `a {e, o} => {e e, o o}`: sequences distribute into
            // adjacent alternative lists when the lengths line up
            // (`SequenceMatcher.transformerToAlternatives`)
            Pattern::Seq(parts) => {
                let n = alternatives.len();
                let alts_fit = parts.iter().all(|p| match p {
                    Pattern::Alt(xs) => xs.len() == n,
                    _ => true,
                });
                let seqs: Option<Vec<&Vec<Emit>>> = alternatives
                    .iter()
                    .map(|a| match a {
                        Emit::Seq(es) if es.len() == parts.len() => Some(es),
                        _ => None,
                    })
                    .collect();
                match seqs {
                    Some(seqs) if alts_fit => {
                        let mut out = Vec::new();
                        for (i, alt_emit) in seqs.iter().enumerate() {
                            let seq_i: Vec<Pattern> = parts
                                .iter()
                                .map(|p| match p {
                                    Pattern::Alt(xs) => xs[i].clone(),
                                    other => other.clone(),
                                })
                                .collect();
                            let emits: Vec<&Emit> = alt_emit.iter().collect();
                            out.extend(
                                self.transform_sequence(phrase, &seq_i, &emits, pos, bindings)?,
                            );
                        }
                        Ok(out)
                    }
                    _ => Err(RunError::MismatchedEmitter),
                }
            }
            _ => Err(RunError::MismatchedEmitter),
        }
    }

    /// Pair a sequence emitter `a b` (lexurgy's `SequenceMatcher`/`Transformer`
    /// dispatch): independent emission when every part is independent and the
    /// pattern prefers it, element-wise pairing against a same-length
    /// sequence pattern, or distribution into an alternation pattern.
    fn transform_seq(
        &mut self,
        phrase: &Phrase,
        pattern: &Pattern,
        emit: &Emit,
        parts: &[Emit],
        pos: usize,
        bindings: &Bindings,
    ) -> Result<Vec<Transformation>, RunError> {
        let independent = parts.iter().all(emit_is_independent);
        if independent && prefers_independent_seq(pattern) {
            return self.transform_independent(phrase, pattern, emit, pos, bindings);
        }
        if is_lifting(pattern) {
            return self.transform_lifting(phrase, pattern, emit, pos, bindings);
        }
        match pattern {
            Pattern::Seq(from_parts) if from_parts.len() == parts.len() => {
                let emits: Vec<&Emit> = parts.iter().collect();
                self.transform_sequence(phrase, from_parts, &emits, pos, bindings)
            }
            Pattern::Alt(from_parts) => {
                let mut out = Vec::new();
                for part in from_parts {
                    out.extend(self.transform(phrase, part, emit, pos, bindings)?);
                }
                Ok(out)
            }
            // lexurgy throws a length mismatch, then falls back to
            // independent emission if the emitter allows it
            _ if independent => self.transform_independent(phrase, pattern, emit, pos, bindings),
            _ => Err(RunError::MismatchedEmitter),
        }
    }

    /// Conditional pairing: the emitter needs to know what matched.
    fn transform_conditional(
        &mut self,
        phrase: &Phrase,
        pattern: &Pattern,
        emit: &Emit,
        pos: usize,
        bindings: &Bindings,
    ) -> Result<Vec<Transformation>, RunError> {
        if is_lifting(pattern) {
            return self.transform_lifting(phrase, pattern, emit, pos, bindings);
        }
        match pattern {
            // a sequence pairs every element with the same conditional
            // emitter (`a b => [+x]` updates both segments)
            Pattern::Seq(parts) => {
                let emits: Vec<&Emit> = parts.iter().map(|_| emit).collect();
                self.transform_sequence(phrase, parts, &emits, pos, bindings)
            }
            Pattern::Alt(parts) => {
                let mut out = Vec::new();
                for part in parts {
                    out.extend(self.transform(phrase, part, emit, pos, bindings)?);
                }
                Ok(out)
            }
            // leaves: claim one element, emit from what it matched
            _ => {
                let ends = self.claim(phrase, pattern, pos, bindings, Dir::Fwd)?;
                let mut out = Vec::new();
                for end in ends {
                    // `ConditionalEmitter.result` receives `original.first()`,
                    // the first word of the matched slice.
                    let original = phrase
                        .slice(pos, end.pos)
                        .words
                        .into_iter()
                        .next()
                        .unwrap_or_default();
                    let result = match emit {
                        Emit::Matrix(update) => ResultSpec::Matrix {
                            update: update.clone(),
                            original,
                        },
                        Emit::Text {
                            word: emit_word,
                            syl_mask,
                            syl_want,
                            ..
                        } => {
                            let segs = self.text_result(pattern, &original.segs, &emit_word.segs);
                            // When the matcher is a `SymbolMatcher` or
                            // `MatrixMatcher`, lexurgy builds a *fresh*
                            // unsyllabified result word (so the matched
                            // span's structure gets recovered); only other
                            // matchers emit the text with its own structure.
                            let from_symbol_or_matrix = matches!(
                                pattern,
                                Pattern::Text(TextIr { exact: false, .. })
                                    | Pattern::Test(SegTest::Matrix(_))
                                    | Pattern::Test(SegTest::Literal { .. })
                            );
                            let syl = if from_symbol_or_matrix {
                                None
                            } else {
                                emit_word.syl.clone()
                            };
                            // `SymbolEmitter.syllableFeatureChanges`: the
                            // matcher's syllable features reset to defaults,
                            // overridden by the emitted text's.
                            let matcher_mask = match pattern {
                                Pattern::Text(t) => t.syl_mask,
                                _ => 0,
                            };
                            let mask = matcher_mask | syl_mask;
                            ResultSpec::CondText {
                                word: Word { segs, syl },
                                original,
                                change: (mask != 0).then_some(SylChange {
                                    mask,
                                    bits: *syl_want,
                                }),
                            }
                        }
                        _ => unreachable!("conditional pairing with independent emitter"),
                    };
                    out.push(Transformation {
                        start: pos,
                        end: end.pos,
                        result,
                        bindings: end.bindings,
                        subs: vec![],
                        removes: end.breaks,
                        trim_leading: false,
                        trim_trailing: false,
                    });
                }
                Ok(out)
            }
        }
    }

    /// Wrappers connect their inner element to the emitter (lexurgy's
    /// `LiftingMatcher`), then add their own behavior on top.
    fn transform_lifting(
        &mut self,
        phrase: &Phrase,
        pattern: &Pattern,
        emit: &Emit,
        pos: usize,
        bindings: &Bindings,
    ) -> Result<Vec<Transformation>, RunError> {
        match pattern {
            Pattern::Capture { inner, slot } => {
                if bindings.capture(slot.0).is_some() {
                    return Err(RunError::ReboundCapture(slot.0));
                }
                Ok(self
                    .transform(phrase, inner, emit, pos, bindings)?
                    .into_iter()
                    // `CaptureTransformer` drops options whose span crosses
                    // a word boundary (unlike pure-matching captures).
                    .filter(|t| phrase.locate(t.start).0 == phrase.locate(t.end).0)
                    .map(|mut t| {
                        t.bindings
                            .bind_capture(slot.0, phrase.slice(t.start, t.end));
                        t
                    })
                    .collect())
            }
            Pattern::Look {
                inner,
                condition,
                exclusion,
            } => {
                let options = self.transform(phrase, inner, emit, pos, bindings)?;
                let mut out = Vec::new();
                for option in options {
                    if let Some(b) = self.check_envs(
                        phrase,
                        option.start,
                        option.end,
                        condition,
                        exclusion,
                        &option.bindings,
                    )? {
                        out.push(Transformation {
                            bindings: b,
                            ..option
                        });
                    }
                }
                Ok(out)
            }
            Pattern::Intersect(parts) => {
                let (first, rest) = parts.split_first().expect("empty intersection");
                let options = self.transform(phrase, first, emit, pos, bindings)?;
                let mut out = Vec::new();
                for option in options {
                    if let Some(b) = self.verify_intersection(
                        phrase,
                        rest,
                        option.start,
                        option.end,
                        &option.bindings,
                        Dir::Fwd,
                    )? {
                        out.push(Transformation {
                            bindings: b,
                            ..option
                        });
                    }
                }
                Ok(out)
            }
            Pattern::Repeat { inner, min, max } => {
                // RepeaterTransformer: each repetition transforms separately
                let mut levels: Vec<Vec<Vec<Transformation>>> = vec![vec![vec![]]];
                loop {
                    let mut next: Vec<Vec<Transformation>> = Vec::new();
                    for combo in levels.last().unwrap() {
                        let (p, b) = combo
                            .last()
                            .map(|t| (t.end, t.bindings.clone()))
                            .unwrap_or((pos, bindings.clone()));
                        for t in self.transform(phrase, inner, emit, p, &b)? {
                            let mut extended = combo.clone();
                            extended.push(t);
                            next.push(extended);
                        }
                    }
                    if next.is_empty() {
                        break;
                    }
                    if next.len() > MAX_OPTIONS {
                        return Err(RunError::TooManyOptions);
                    }
                    levels.push(next);
                    if let Some(max) = max {
                        if levels.len() > *max as usize {
                            break;
                        }
                    }
                    if levels.len() > phrase.len() + 2 {
                        break;
                    }
                }
                if levels.len() <= *min as usize {
                    return Ok(vec![]);
                }
                Ok(levels
                    .drain(*min as usize..)
                    .rev()
                    .flatten()
                    .map(|combo| combine_subs(pos, bindings, combo))
                    .collect())
            }
            _ => unreachable!("transform_lifting on non-wrapper"),
        }
    }

    /// Element-wise pairing of a from-sequence with to-emitters.
    fn transform_sequence(
        &mut self,
        phrase: &Phrase,
        from_parts: &[Pattern],
        emits: &[&Emit],
        pos: usize,
        bindings: &Bindings,
    ) -> Result<Vec<Transformation>, RunError> {
        let mut combos: Vec<Vec<Transformation>> = vec![vec![]];
        for (part, emit) in from_parts.iter().zip(emits) {
            let mut next = Vec::new();
            for combo in &combos {
                let (p, b) = combo
                    .last()
                    .map(|t| (t.end, t.bindings.clone()))
                    .unwrap_or((pos, bindings.clone()));
                for t in self.transform(phrase, part, emit, p, &b)? {
                    let mut extended = combo.clone();
                    extended.push(t);
                    next.push(extended);
                }
            }
            if next.len() > MAX_OPTIONS {
                return Err(RunError::TooManyOptions);
            }
            if next.is_empty() {
                return Ok(vec![]);
            }
            combos = next;
        }
        Ok(combos
            .into_iter()
            .map(|combo| combine_subs(pos, bindings, combo))
            .collect())
    }

    /// The emitter doesn't need the match: claim the pattern, conjure the
    /// result (recovering the matched span's structure at bind time).
    fn transform_independent(
        &mut self,
        phrase: &Phrase,
        pattern: &Pattern,
        emit: &Emit,
        pos: usize,
        bindings: &Bindings,
    ) -> Result<Vec<Transformation>, RunError> {
        let spec = independent_spec(emit)?;
        Ok(self
            .claim(phrase, pattern, pos, bindings, Dir::Fwd)?
            .into_iter()
            .map(|end| {
                let slice = phrase.slice(pos, end.pos);
                // Matched breaks become *flat* segment indices relative to
                // the slice (what `Phrase.recoverStructure` expects).
                let flat_pos = phrase.flat_index(pos);
                let except: Vec<usize> = end
                    .breaks
                    .iter()
                    .map(|&b| phrase.flat_index(b) - flat_pos)
                    .collect();
                // A sequence emitter contributes one sub-transformation per
                // (flattened) element, each spanning the whole match;
                // lexurgy's `IndependentSequenceTransformer.resultBits`.
                // Invisible to the unfiltered path (which binds the parent's
                // `result`); filtered rules unfilter these *individually*,
                // which is what drops surplus elements (`@cv => z u`).
                let subs = match &spec {
                    IndepSpec::Seq(parts) => flatten_indep(parts)
                        .into_iter()
                        .map(|part| Transformation {
                            start: pos,
                            end: end.pos,
                            result: ResultSpec::Independent {
                                spec: part.clone(),
                                slice: slice.clone(),
                                except: except.clone(),
                            },
                            bindings: end.bindings.clone(),
                            subs: vec![],
                            removes: vec![],
                            trim_leading: false,
                            trim_trailing: false,
                        })
                        .collect(),
                    _ => vec![],
                };
                Transformation {
                    start: pos,
                    end: end.pos,
                    result: ResultSpec::Independent {
                        spec: spec.clone(),
                        slice,
                        except,
                    },
                    bindings: end.bindings,
                    subs,
                    removes: end.breaks,
                    trim_leading: false,
                    trim_trailing: false,
                }
            })
            .collect())
    }

    /// Text in output position copies floating diacritics from the matched
    /// segments, excluding any the pattern itself mentioned; lexurgy's
    /// `SymbolEmitter.result`, with its three length cases.
    fn text_result(
        &mut self,
        pattern: &Pattern,
        original: &[SegmentId],
        text: &[SegmentId],
    ) -> Vec<SegmentId> {
        if self.floating == 0 {
            return text.to_vec();
        }
        // The pattern's own segments (`SymbolMatcher.text`). Exact text is a
        // `TextMatcher`, and matrices transfer without exclusions; anything
        // else emits the text verbatim.
        let pattern_segs: Option<Vec<SegmentId>> = match pattern {
            Pattern::Text(t) => t
                .tests
                .iter()
                .map(|test| match test {
                    SegTest::Literal { id, .. } => Some(*id),
                    _ => None,
                })
                .collect(),
            Pattern::Test(SegTest::Literal { id, .. }) => Some(vec![*id]),
            Pattern::Test(SegTest::Matrix(_)) => {
                if original.len() == 1 {
                    return text
                        .iter()
                        .map(|&id| self.with_floats(id, original[0], 0))
                        .collect();
                }
                return text.to_vec();
            }
            _ => None,
        };
        let Some(pattern_segs) = pattern_segs else {
            return text.to_vec();
        };
        if pattern_segs.len() == text.len() && original.len() == text.len() {
            // pairwise: each emitted segment from its counterpart
            text.iter()
                .zip(original)
                .zip(&pattern_segs)
                .map(|((&t, &o), &p)| {
                    let excluded = self.segments.get(p).diacritics;
                    self.with_floats(t, o, excluded)
                })
                .collect()
        } else if pattern_segs.len() == original.len() && text.len() == 1 {
            // collapse: the one emitted segment collects from all originals
            let mut out = text[0];
            for (&o, &p) in original.iter().zip(&pattern_segs) {
                let excluded = self.segments.get(p).diacritics;
                out = self.with_floats(out, o, excluded);
            }
            vec![out]
        } else if pattern_segs.len() == 1 && original.len() == 1 {
            // expand: every emitted segment from the one original
            let excluded = self.segments.get(pattern_segs[0]).diacritics;
            text.iter()
                .map(|&t| self.with_floats(t, original[0], excluded))
                .collect()
        } else {
            text.to_vec()
        }
    }

    fn with_floats(
        &mut self,
        target: SegmentId,
        source: SegmentId,
        excluded: DiacriticMask,
    ) -> SegmentId {
        let extra = self.segments.get(source).diacritics & self.floating & !excluded;
        if extra == 0 {
            target
        } else {
            self.segments
                .with_extra_diacritics(self.decls, target, extra)
        }
    }

    // binding results

    fn bind(&mut self, t: &Transformation) -> Result<ChangeResult, RunError> {
        let mut cr = match &t.result {
            ResultSpec::Independent {
                spec,
                slice,
                except,
            } => {
                let inner = self.bind_indep(spec, &t.bindings)?;
                let recovered = slice.recover_structure(inner.phrase, except);
                ChangeResult {
                    phrase: recovered,
                    emits: inner.emits,
                    changes: inner.changes,
                    vacuous: inner.vacuous,
                }
            }
            ResultSpec::CondText {
                word,
                original,
                change,
            } => {
                let recovered = original.recover_structure(word.clone(), &[]);
                let changes: BTreeMap<usize, SylChange> = match change {
                    Some(c) => (0..recovered.len().max(1)).map(|i| (i, *c)).collect(),
                    None => BTreeMap::new(),
                };
                let updated = self.word_with_changes(recovered, &changes)?;
                ChangeResult {
                    phrase: Phrase::single(updated),
                    emits: vec![],
                    changes,
                    vacuous: false,
                }
            }
            ResultSpec::Matrix { update, original } => {
                self.bind_matrix(update, original, &t.bindings)?
            }
            ResultSpec::Subs => self.bind_subs(t)?,
        };
        if t.trim_leading {
            cr.phrase = cr.phrase.remove_leading_break();
        }
        if t.trim_trailing {
            cr.phrase = cr.phrase.remove_trailing_break();
        }
        Ok(cr)
    }

    fn bind_indep(
        &mut self,
        spec: &IndepSpec,
        bindings: &Bindings,
    ) -> Result<ChangeResult, RunError> {
        Ok(match spec {
            IndepSpec::Text(word) => ChangeResult::of(Phrase::single(word.clone())),
            IndepSpec::Empty => ChangeResult::of(Phrase::single(Word::default())),
            IndepSpec::Boundary => ChangeResult {
                phrase: Phrase::single(Word::break_only()),
                emits: vec![0],
                changes: BTreeMap::new(),
                vacuous: false,
            },
            // `BetweenWordsEmitter`: a phrase of two empty words.
            IndepSpec::WordBreak => ChangeResult::of(Phrase {
                words: vec![Word::default(), Word::default()],
            }),
            IndepSpec::Capture(slot) => {
                let captured = bindings
                    .capture(*slot)
                    .ok_or(RunError::UnboundCapture(*slot))?;
                ChangeResult::of(captured.to_simple())
            }
            IndepSpec::SylCapture(slot) => {
                let captured = bindings
                    .capture(*slot)
                    .ok_or(RunError::UnboundCapture(*slot))?
                    .as_ref()
                    .clone()
                    .remove_bounding_breaks();
                let mut changes: BTreeMap<usize, SylChange> = BTreeMap::new();
                let mut offset = 0usize;
                for word in &captured.words {
                    for i in 0..word.len() {
                        let (_, bits) = syl_mods_bits(self.decls, word.mods_at(i));
                        changes.insert(
                            offset + i,
                            SylChange {
                                mask: self.syl_fields,
                                bits,
                            },
                        );
                    }
                    offset += word.len() + 1;
                }
                ChangeResult {
                    emits: captured.syllable_breaks_linear(),
                    phrase: captured,
                    changes,
                    vacuous: false,
                }
            }
            IndepSpec::Seq(parts) => {
                let mut phrase = Phrase::default();
                let mut emits = Vec::new();
                let mut changes = BTreeMap::new();
                let mut offset = 0usize;
                for part in parts {
                    let cr = self.bind_indep(part, bindings)?;
                    emits.extend(cr.emits.iter().map(|b| b + offset));
                    changes.extend(cr.changes.iter().map(|(&i, &c)| (i + offset, c)));
                    offset += cr.phrase.len();
                    phrase = phrase.concat(&cr.phrase, |left, _| left.to_vec());
                }
                ChangeResult {
                    phrase,
                    emits,
                    changes,
                    vacuous: parts.is_empty(),
                }
            }
        })
    }

    /// `MatrixEmitter` + `SyllableMatrixEmitter` (+ `MultiConditionalEmitter`
    /// chaining when a matrix has both levels).
    fn bind_matrix(
        &mut self,
        update: &MatrixUpdate,
        original: &Word,
        bindings: &Bindings,
    ) -> Result<ChangeResult, RunError> {
        let syl_change = self.syl_change(update, bindings)?;
        let has_seg = update.seg_mask != 0
            || update
                .vars
                .iter()
                .any(|&f| self.decls.features.def(f).level == Level::Segment)
            || syl_change.is_none();
        let result_word = if !has_seg {
            original.clone()
        } else if original.is_empty() {
            // zero-width match: emit the matrix alone if it spells a
            // segment, otherwise nothing (lexurgy catches LscInvalidMatrix)
            let value = self.updated_word(FeatureWord(0), update, bindings)?;
            match self.segments.render_features(self.decls, value) {
                Some(id) => Word::simple(vec![id]),
                None => Word::default(),
            }
        } else {
            let mut segs = Vec::with_capacity(original.len());
            for &seg in &original.segs {
                let data = self.segments.get(seg);
                let (features, core) = (data.features, data.core);
                let value = self.updated_word(features, update, bindings)?;
                // A featureless core keeps lexurgy's `UndeclaredSymbolValue`
                // in its matrix: the diacritic search starts from the core
                // itself, not from declared symbols.
                let id = if self.segments.core_is_featural(self.decls, core) {
                    self.segments.render_features(self.decls, value)
                } else {
                    self.segments.render_cored(self.decls, core, value)
                }
                .ok_or(RunError::InvalidMatrix)?;
                segs.push(id);
            }
            original.recover_structure(Word::simple(segs), &[])
        };
        let Some(change) = syl_change else {
            return Ok(ChangeResult::of(Phrase::single(result_word)));
        };
        // SyllableMatrixEmitter: rewrite the features of every syllable the
        // match touches, and force the result to be syllabified.
        let mut word = if result_word.is_syllabified() {
            result_word
        } else {
            Word {
                segs: result_word.segs,
                syl: Some(Syl::default()),
            }
        };
        let count = word.num_syllables();
        if let Some(syl) = word.syl.as_mut() {
            for n in 0..count {
                let existing = syl.mods.get(&n).map_or(&[][..], |m| m.as_slice());
                let (_, value) = syl_mods_bits(self.decls, existing);
                let updated = value & !change.mask | change.bits;
                let mods = render_syl_mods(self.decls, updated).ok_or(RunError::InvalidMatrix)?;
                if mods.is_empty() {
                    syl.mods.remove(&n);
                } else {
                    syl.mods.insert(n, mods);
                }
            }
        }
        let changes = (0..word.len()).map(|i| (i, change)).collect();
        Ok(ChangeResult {
            phrase: Phrase::single(word),
            emits: vec![],
            changes,
            vacuous: false,
        })
    }

    /// `UnboundTransformation.fromSubTransformations`'s bind: bind each sub,
    /// trim breaks at sub boundaries where a break was matched but not
    /// re-emitted, then concatenate.
    fn bind_subs(&mut self, t: &Transformation) -> Result<ChangeResult, RunError> {
        let mut bound: Vec<(usize, ChangeResult)> = Vec::new();
        for (i, sub) in t.subs.iter().enumerate() {
            let mut sub = sub.clone();
            sub.bindings = t.bindings.clone();
            let cr = self.bind(&sub)?;
            // an empty repeat's result vanishes entirely (lexurgy filters
            // `phrase.words.isNotEmpty()`), so its neighbors become adjacent
            if !cr.vacuous {
                bound.push((i, cr));
            }
        }
        let mut sub_phrases: Vec<Phrase> = Vec::new();
        for (bi, (i, cr)) in bound.iter().enumerate() {
            let sub = &t.subs[*i];
            let mut p = cr.phrase.clone();
            if !cr.emits_break_before() {
                let prev_removes_after = bi > 0 && t.subs[bound[bi - 1].0].removes_break_after();
                if sub.removes_break_before() || prev_removes_after {
                    p = p.remove_leading_break();
                }
            }
            if !cr.emits_break_after() {
                let next_removes_before =
                    bi + 1 < bound.len() && t.subs[bound[bi + 1].0].removes_break_before();
                if sub.removes_break_after() || next_removes_before {
                    p = p.remove_trailing_break();
                }
            }
            sub_phrases.push(p);
        }
        let phrase = Phrase::from_sub_phrases(&sub_phrases);
        let mut emits = Vec::new();
        let mut changes = BTreeMap::new();
        let mut offset = 0usize;
        for (_, cr) in &bound {
            emits.extend(cr.emits.iter().map(|b| b + offset));
            changes.extend(cr.changes.iter().map(|(&i, &c)| (i + offset, c)));
            offset += cr.phrase.len();
        }
        Ok(ChangeResult {
            phrase,
            emits,
            changes,
            vacuous: t.subs.is_empty(),
        })
    }

    fn updated_word(
        &self,
        features: FeatureWord,
        update: &MatrixUpdate,
        bindings: &Bindings,
    ) -> Result<FeatureWord, RunError> {
        let mut word = features.0 & !update.seg_mask | update.seg_bits;
        for &feature in &update.vars {
            let def = self.decls.features.def(feature);
            if def.level != Level::Segment {
                continue;
            }
            let code = bindings
                .feature(feature.0)
                .ok_or(RunError::UnboundVariable)?;
            word = word & !def.field_mask() | def.encode(code);
        }
        Ok(FeatureWord(word))
    }

    // syllabification (Syllabifier.kt)

    /// Divide the word into syllables using the best pattern sequence.
    pub fn syllabify(
        &mut self,
        syllabifier: &SyllabifierIr,
        word: &Word,
    ) -> Result<Word, RunError> {
        if syllabifier.exprs.is_empty() {
            return Ok(word.clone());
        }
        let matches = self.find_best_sequence(syllabifier, word)?;
        let new_breaks: Vec<usize> = matches
            .iter()
            .take(matches.len().saturating_sub(1))
            .map(|m| m.end)
            .collect();
        // Assigned syllable matrices, by new syllable number.
        let mut assigned: BTreeMap<usize, SylChange> = BTreeMap::new();
        for (n, m) in matches.iter().enumerate() {
            if let Some(expr) = m.assign {
                let update = syllabifier.exprs[expr]
                    .assign
                    .as_ref()
                    .expect("assign index points at an expression with a matrix");
                assigned.insert(
                    n,
                    SylChange {
                        mask: update.syl_mask,
                        bits: update.syl_bits,
                    },
                );
            }
        }
        // combineSyllableModifiers: carry the old modifiers onto the new
        // syllables, then apply the assigned matrices on top.
        let map = find_syllable_map(word, &new_breaks);
        let mut combined: BTreeMap<usize, (u128, u128)> = BTreeMap::new();
        if let Some(syl) = &word.syl {
            for (&n, mods) in &syl.mods {
                let nn = map.get(&n).copied().unwrap_or(n);
                let (em, ew) = combined.get(&nn).copied().unwrap_or((0, 0));
                let (mm, mw) = syl_mods_test(self.decls, mods);
                combined.insert(nn, (em | mm, ew & !mm | mw));
            }
        }
        for (n, change) in &assigned {
            let (em, ew) = combined.get(n).copied().unwrap_or((0, 0));
            combined.insert(*n, (em | change.mask, ew & !change.mask | change.bits));
        }
        let mut mods = BTreeMap::new();
        for (n, (_, value)) in combined {
            let rendered = render_syl_mods(self.decls, value).ok_or(RunError::InvalidMatrix)?;
            if !rendered.is_empty() {
                mods.insert(n, rendered);
            }
        }
        Ok(Word {
            segs: word.segs.clone(),
            syl: Some(Syl::new(new_breaks, mods)),
        })
    }

    /// The shortest-path search over pattern matches
    /// (`findBestSyllableSequence`).
    fn find_best_sequence(
        &mut self,
        syllabifier: &SyllabifierIr,
        word: &Word,
    ) -> Result<Vec<PatMatch>, RunError> {
        let phrase = Phrase::single(word.clone());
        let mut sequences: Vec<Option<Vec<PatMatch>>> = vec![None; word.len() + 1];
        sequences[0] = Some(Vec::new());
        for i in 0..word.len() {
            let Some(prev) = sequences[i].clone() else {
                continue;
            };
            for (pi, expr) in syllabifier.exprs.iter().enumerate() {
                for m in self.match_syl_pattern(&phrase, i, pi, expr)? {
                    let end = m.end;
                    let mut seq = prev.clone();
                    seq.push(m);
                    let better = match &sequences[end] {
                        None => true,
                        Some(existing) => seq_less(&seq, existing),
                    };
                    if better {
                        sequences[end] = Some(seq);
                    }
                }
            }
        }
        sequences[word.len()]
            .clone()
            .ok_or(RunError::SyllableStructure)
    }

    fn match_syl_pattern(
        &mut self,
        phrase: &Phrase,
        start: usize,
        pattern_number: usize,
        expr: &crate::compiler::ir::SylExprIr,
    ) -> Result<Vec<PatMatch>, RunError> {
        let assign = expr.assign.as_ref().map(|_| pattern_number);
        match &expr.pattern {
            SylPatternIr::Simple(pattern) => Ok(self
                .claim(phrase, pattern, start, &Bindings::default(), Dir::Fwd)?
                .into_iter()
                .map(|end| PatMatch {
                    pattern: pattern_number,
                    end: end.pos,
                    reluctant: None,
                    nucleus: None,
                    assign,
                })
                .collect()),
            SylPatternIr::Structured(s) => {
                let StructuredSyl {
                    reluctant_onset,
                    onset,
                    nucleus,
                    coda,
                    condition,
                    exclusion,
                } = &**s;
                // Match the parts in turn, recording the lengths the
                // preference order cares about.
                struct State {
                    end: usize,
                    bindings: Bindings,
                    reluctant: usize,
                    nucleus_end: Option<usize>,
                }
                let mut states = vec![State {
                    end: start,
                    bindings: Bindings::default(),
                    reluctant: 0,
                    nucleus_end: None,
                }];
                let empty = Pattern::Empty;
                let parts: [(&Pattern, u8); 4] = [
                    (reluctant_onset.as_ref().unwrap_or(&empty), 0),
                    (onset, 1),
                    (nucleus, 2),
                    (coda.as_ref().unwrap_or(&empty), 3),
                ];
                for (part, kind) in parts {
                    let mut next = Vec::new();
                    for state in &states {
                        for end in self.claim(phrase, part, state.end, &state.bindings, Dir::Fwd)? {
                            next.push(State {
                                end: end.pos,
                                reluctant: if kind == 0 {
                                    end.pos - start
                                } else {
                                    state.reluctant
                                },
                                nucleus_end: if kind == 2 {
                                    Some(end.pos - state.end)
                                } else {
                                    state.nucleus_end
                                },
                                bindings: end.bindings,
                            });
                        }
                    }
                    if next.len() > MAX_OPTIONS {
                        return Err(RunError::TooManyOptions);
                    }
                    if next.is_empty() {
                        return Ok(vec![]);
                    }
                    states = next;
                }
                let mut out = Vec::new();
                for state in states {
                    if self
                        .check_envs(
                            phrase,
                            start,
                            state.end,
                            condition,
                            exclusion,
                            &Bindings::default(),
                        )?
                        .is_none()
                    {
                        continue;
                    }
                    out.push(PatMatch {
                        pattern: pattern_number,
                        end: state.end,
                        reluctant: Some(state.reluctant),
                        nucleus: state.nucleus_end,
                        assign,
                    });
                }
                Ok(out)
            }
        }
    }
}

/// One syllable pattern match in the syllabifier's search.
#[derive(Debug, Clone)]
struct PatMatch {
    pattern: usize,
    end: usize,
    reluctant: Option<usize>,
    nucleus: Option<usize>,
    /// Expression index carrying an assigned matrix, if any.
    assign: Option<usize>,
}

/// `PatternMatchSequence.compareTo`: `true` if `a` is preferred over `b`:
/// shorter reluctant onsets, then longer nuclei, then earlier ends, then
/// earlier patterns.
fn seq_less(a: &[PatMatch], b: &[PatMatch]) -> bool {
    use std::cmp::Ordering;
    let mut i = 0;
    while i < a.len() && i < b.len() {
        if i + 1 < a.len() && i + 1 < b.len() && a[i + 1].reluctant != b[i + 1].reluctant {
            return a[i + 1].reluctant.unwrap_or(0) < b[i + 1].reluctant.unwrap_or(0);
        }
        if a[i].nucleus != b[i].nucleus {
            return b[i].nucleus.unwrap_or(0) < a[i].nucleus.unwrap_or(0);
        }
        match a[i].end.cmp(&b[i].end) {
            Ordering::Equal => {}
            other => return other == Ordering::Less,
        }
        match a[i].pattern.cmp(&b[i].pattern) {
            Ordering::Equal => {}
            other => return other == Ordering::Less,
        }
        i += 1;
    }
    false
}

/// Map old syllable numbers to the new syllable containing them
/// (`findSyllableMap`).
fn find_syllable_map(word: &Word, new_breaks: &[usize]) -> BTreeMap<usize, usize> {
    let mut result = BTreeMap::new();
    if !word.is_syllabified() {
        return result;
    }
    let mut old_ends: Vec<usize> = word.syllable_breaks().to_vec();
    old_ends.push(word.len());
    let mut new_ends: Vec<usize> = new_breaks.to_vec();
    new_ends.push(word.len());
    let mut new_index = 0usize;
    for (old_index, &old_end) in old_ends.iter().enumerate() {
        result.insert(old_index, new_index);
        while new_index < new_ends.len() && new_ends[new_index] <= old_end {
            new_index += 1;
        }
    }
    result
}

fn step(pos: usize, dir: Dir) -> usize {
    match dir {
        Dir::Fwd => pos + 1,
        Dir::Bwd => pos - 1,
    }
}

fn span(a: usize, b: usize) -> (usize, usize) {
    (a.min(b), a.max(b))
}

fn zero_width(matches: bool, pos: usize, bindings: &Bindings) -> Vec<MatchEnd> {
    if matches {
        vec![MatchEnd::plain(pos, bindings.clone())]
    } else {
        vec![]
    }
}

/// Fuse element-wise sub-transformations into one (lexurgy's
/// `UnboundTransformation.fromSubTransformations`).
fn combine_subs(pos: usize, bindings: &Bindings, subs: Vec<Transformation>) -> Transformation {
    let mut removes: Vec<usize> = Vec::new();
    for sub in &subs {
        for &b in &sub.removes {
            if !removes.contains(&b) {
                removes.push(b);
            }
        }
    }
    Transformation {
        start: pos,
        end: subs.last().map(|t| t.end).unwrap_or(pos),
        result: ResultSpec::Subs,
        bindings: subs
            .last()
            .map(|t| t.bindings.clone())
            .unwrap_or_else(|| bindings.clone()),
        subs,
        removes,
        trim_leading: false,
        trim_trailing: false,
    }
}

/// Flatten nested sequence specs into their leaf elements (lexurgy's
/// `resultBits` recursion over nested `SequenceEmitter`s).
fn flatten_indep(parts: &[IndepSpec]) -> Vec<&IndepSpec> {
    let mut out = Vec::new();
    for part in parts {
        match part {
            IndepSpec::Seq(inner) => out.extend(flatten_indep(inner)),
            other => out.push(other),
        }
    }
    out
}

fn independent_spec(emit: &Emit) -> Result<IndepSpec, RunError> {
    Ok(match emit {
        Emit::Text { word, .. } => IndepSpec::Text(word.clone()),
        Emit::Empty => IndepSpec::Empty,
        Emit::SyllableBoundary => IndepSpec::Boundary,
        Emit::WordBreak => IndepSpec::WordBreak,
        Emit::CaptureRef { slot, .. } => IndepSpec::Capture(slot.0),
        Emit::SylCaptureRef { slot } => IndepSpec::SylCapture(slot.0),
        Emit::Seq(parts) => IndepSpec::Seq(
            parts
                .iter()
                .map(independent_spec)
                .collect::<Result<_, _>>()?,
        ),
        Emit::Alt(_) | Emit::Matrix(_) => return Err(RunError::MismatchedEmitter),
    })
}

/// Keep transformations in precedence order, dropping any that overlap an
/// earlier claim. Half-open ranges: zero-width claims never overlap.
fn filter_overlapping(transformations: Vec<Transformation>) -> Vec<Transformation> {
    let mut claimed: Vec<(usize, usize)> = Vec::new();
    transformations
        .into_iter()
        .filter(|t| {
            let overlaps = claimed.iter().any(|&(s, e)| t.start < e && s < t.end);
            if !overlaps {
                claimed.push((t.start, t.end));
            }
            !overlaps
        })
        .collect()
}

/// Map transformations from filtered-word coordinates back to the real
/// word (lexurgy's `unfilterTransformations`). Each elemental piece becomes
/// a claim on exactly *one* real segment: the one its match *started* on
/// (`filterMap[sub.start]` .. `stepForward`), no matter how many filtered
/// segments it covered or how many segments its result emits. Pieces whose
/// claims then collide are dropped by `apply_transformations`' cursor skip,
/// which is how lexurgy silently discards surplus result elements (e.g.
/// filtered `@cv => z u` keeps only the `z`). Syllable breaks survive in
/// the result only where the real word has them.
fn unfilter(
    phrase: &Phrase,
    maps: Option<&[Vec<usize>]>,
    transformations: Vec<Transformation>,
) -> Result<Vec<Transformation>, RunError> {
    let Some(maps) = maps else {
        return Ok(transformations);
    };
    // Word offsets in the *filtered* phrase (word w has maps[w].len()
    // segments) and the real phrase.
    let mut filtered_starts = Vec::with_capacity(maps.len());
    let mut acc = 0usize;
    for m in maps {
        filtered_starts.push(acc);
        acc += m.len() + 1;
    }
    let mut out = Vec::new();
    for t in transformations {
        let bindings = t.bindings.clone();
        for sub in elemental(&t) {
            // Which filtered word does the piece start in?
            let w = filtered_starts
                .iter()
                .rposition(|&s| s <= sub.start)
                .unwrap_or(0);
            let s = sub.start - filtered_starts[w];
            // Lexurgy indexes `filterMap[wordIndex][segmentIndex]`
            // unguarded; a zero-width match at the end of a filtered word
            // throws (caught per word). Mirror that as a word-level error.
            let Some(real_seg) = maps[w].get(s).copied() else {
                return Err(RunError::Word(
                    "filtered rule matched past the last filtered segment".to_string(),
                ));
            };
            let start = phrase.word_start(w) + real_seg;
            let end = start + 1;
            let breaks = phrase.words[w].syllable_breaks();
            out.push(Transformation {
                start,
                end,
                result: sub.result.clone(),
                bindings: bindings.clone(),
                subs: vec![],
                removes: vec![],
                trim_leading: !breaks.contains(&real_seg),
                trim_trailing: !breaks.contains(&(real_seg + 1)),
            });
        }
    }
    Ok(out)
}

fn elemental(t: &Transformation) -> Vec<&Transformation> {
    if t.subs.is_empty() {
        vec![t]
    } else {
        t.subs.iter().flat_map(elemental).collect()
    }
}

/// Spell a word back out, including syllable breaks and syllable modifiers
/// (`Syllabification.string`).
fn render_word(decls: &Declarations, segments: &SegmentInterner, word: &Word) -> String {
    let Some(syl) = &word.syl else {
        return word
            .segs
            .iter()
            .map(|&id| segments.get(id).text.as_str())
            .collect();
    };
    if word.is_empty() {
        return if syl.breaks.is_empty() {
            String::new()
        } else {
            ".".to_string()
        };
    }
    let bounds = word.syllable_bounds();
    let mut out = String::new();
    if syl.breaks.first() == Some(&0) {
        out.push('.');
    }
    let syllables: Vec<String> = bounds
        .windows(2)
        .enumerate()
        .map(|(n, w)| {
            let body: String = word.segs[w[0]..w[1]]
                .iter()
                .map(|&id| segments.get(id).text.as_str())
                .collect();
            match syl.mods.get(&n) {
                None => body,
                Some(mods) => render_modified(decls, &body, mods),
            }
        })
        .collect();
    out.push_str(&syllables.join("."));
    if syl.breaks.last() == Some(&word.len()) {
        out.push('.');
    }
    out
}

/// Attach syllable modifiers to a syllable's text. Lexurgy joins multiple
/// modifiers in the same position with `", "` (the `joinToString` default
/// in `Syllabification`); we reproduce that for byte-compatible output.
fn render_modified(decls: &Declarations, body: &str, mods: &[u8]) -> String {
    use crate::compiler::decls::DiacriticPosition;
    let group = |position: DiacriticPosition| -> String {
        mods.iter()
            .map(|&i| &decls.diacritics[i as usize])
            .filter(|d| d.position == position)
            .map(|d| d.name.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    };
    let mut chars = body.chars();
    let first: String = chars.next().map(|c| c.to_string()).unwrap_or_default();
    format!(
        "{}{}{}{}{}",
        group(DiacriticPosition::Before),
        first,
        group(DiacriticPosition::First),
        chars.as_str(),
        group(DiacriticPosition::After),
    )
}

impl CompiledRules {
    /// Run one input line through the whole pipeline (lexurgy's
    /// `SoundChanger.change` for one cell). A line with spaces is a
    /// multi-word phrase: lexurgy trims it and splits on single spaces
    /// (consecutive spaces produce empty words, faithfully).
    pub fn apply(&mut self, text: &str) -> Result<String, RunError> {
        let mut universe = self.input_universe;
        let mut words = Vec::new();
        for cell in text.trim().split(' ') {
            let word = match universe {
                Universe::Real => {
                    self.segments
                        .parse_word(&self.decls, cell, self.input_syllabified)
                }
                Universe::Literal => {
                    self.literal_segments
                        .parse_word(&self.literal_decls, cell, false)
                }
            }
            .map_err(|e| RunError::Word(e.to_string()))?;
            words.push(word);
        }
        let mut phrase = Phrase { words };
        // Which syllabifier's *untouched* output `phrase` still is: while
        // set, re-running that syllabifier is a no-op (`syl_skippable`)
        // and the step is skipped. Cleared whenever anything changes the
        // phrase. Gated on `!force_vm` so the forced-VM side stays the
        // unoptimized reference; the fst-vs-vm fuzzer then checks this
        // optimization too.
        let mut fresh_syl: Option<usize> = None;

        let mut i = 0;
        while i < self.steps.len() {
            // A fused run of segment-map rules covers several steps at
            // once; like the per-rule FST path it only handles
            // unsyllabified words. Fused (and FST) rules never mention
            // `$$`, so applying them word by word is exact.
            if !self.force_vm && !phrase.is_syllabified() {
                if let Some(&f) = self.fused_at.get(&i) {
                    let run = &mut self.fused[f];
                    let (decls, segments) = match run.universe {
                        Universe::Real => (&self.decls, &mut self.segments),
                        Universe::Literal => (&self.literal_decls, &mut self.literal_segments),
                    };
                    for word in &mut phrase.words {
                        let next = run.apply(&mut self.fst_rules, decls, segments, word)?;
                        if next != *word {
                            fresh_syl = None;
                        }
                        *word = next;
                    }
                    i = run.end_step;
                    continue;
                }
            }
            match self.steps[i] {
                Step::Rule { rule, universe: u } => {
                    let (decls, segments) = match u {
                        Universe::Real => (&self.decls, &mut self.segments),
                        Universe::Literal => (&self.literal_decls, &mut self.literal_segments),
                    };
                    // Prefer the FST tier where the rule compiled for it;
                    // syllabified words stay on the (structure-aware) VM,
                    // as do multi-word phrases under `propagate` (its
                    // divergence budgets are phrase-global here but
                    // per-word there). Where the full FST path is off the
                    // table but the match side compiled, the VM runs with
                    // the FST as a position gate: same code paths, fewer
                    // `transform` attempts.
                    if !self.force_vm
                        && !phrase.is_syllabified()
                        && self.fst_rules[rule].as_ref().is_some_and(|f| {
                            f.splices() && (phrase.words.len() == 1 || !f.propagates())
                        })
                    {
                        let fst = self.fst_rules[rule].as_mut().expect("just checked");
                        for word in &mut phrase.words {
                            if let Some(next) = fst.apply(decls, segments, word)? {
                                if next != *word {
                                    fresh_syl = None;
                                }
                                *word = next;
                            }
                        }
                    } else {
                        let body = &self.rules[rule].body;
                        let mut executor = Executor::new(decls, segments);
                        if !self.force_vm {
                            if let Some(fst) = self.fst_rules[rule].as_mut() {
                                let gate = fst.gate();
                                debug_assert_eq!(gate.leaf_count(), leaf_count(body));
                                executor = executor.with_gate(gate);
                            }
                        }
                        if let Some(next) = executor.run_block(body, &phrase, &[])? {
                            if next != phrase {
                                fresh_syl = None;
                            }
                            phrase = next;
                        }
                    }
                }
                Step::StripBreaks => {
                    // `remove_bounding_breaks` is the identity unless some
                    // word actually has one.
                    let had = phrase.words.iter().any(|w| {
                        let b = w.syllable_breaks();
                        b.first() == Some(&0) || b.last() == Some(&w.len())
                    });
                    if had {
                        phrase = phrase.remove_bounding_breaks();
                        fresh_syl = None;
                    }
                }
                Step::Syllabify(index) => match index {
                    None => {
                        if phrase.is_syllabified() {
                            phrase = phrase.to_simple();
                        }
                        fresh_syl = None;
                    }
                    Some(si) => {
                        let skip =
                            !self.force_vm && fresh_syl == Some(si) && self.syl_skippable[si];
                        if !skip {
                            let syllabifier = &self.syllabifiers[si];
                            let mut executor = Executor::new(&self.decls, &mut self.segments);
                            let mut words = Vec::with_capacity(phrase.words.len());
                            for word in &phrase.words {
                                words.push(executor.syllabify(syllabifier, word)?);
                            }
                            phrase = Phrase { words };
                            fresh_syl = Some(si);
                        }
                    }
                },
                Step::Redeclare {
                    universe: u,
                    syllabified,
                } => {
                    let mut words = Vec::with_capacity(phrase.words.len());
                    for word in &phrase.words {
                        let rendered = match universe {
                            Universe::Real => render_word(&self.decls, &self.segments, word),
                            Universe::Literal => {
                                render_word(&self.literal_decls, &self.literal_segments, word)
                            }
                        };
                        let reparsed = match u {
                            Universe::Real => {
                                self.segments
                                    .parse_word(&self.decls, &rendered, syllabified)
                            }
                            Universe::Literal => self.literal_segments.parse_word(
                                &self.literal_decls,
                                &rendered,
                                false,
                            ),
                        }
                        .map_err(|e| RunError::Word(e.to_string()))?;
                        words.push(reparsed);
                    }
                    phrase = Phrase { words };
                    universe = u;
                    fresh_syl = None;
                }
            }
            i += 1;
        }

        let rendered = phrase
            .words
            .iter()
            .map(|word| match universe {
                Universe::Real => render_word(&self.decls, &self.segments, word),
                Universe::Literal => render_word(&self.literal_decls, &self.literal_segments, word),
            })
            .collect::<Vec<_>>()
            .join(" ");
        // lexurgy NFC-composes final output (`normalizeCompose`)
        if rendered.is_ascii() {
            Ok(rendered)
        } else {
            use unicode_normalization::UnicodeNormalization;
            Ok(rendered.nfc().collect())
        }
    }

    /// Run many input lines in parallel, results in input order. Lines are
    /// independent (lexurgy's `SoundChanger.change` maps over cells), so
    /// each rayon worker runs on its own clone of the changer: the
    /// interner, lazy DFA caches, and fused-run caches are per-worker
    /// runtime state, warmed independently and discarded at the end.
    pub fn apply_all<S: AsRef<str> + Sync>(&self, lines: &[S]) -> Vec<Result<String, RunError>> {
        use rayon::prelude::*;
        lines
            .par_iter()
            // Below this, cloning the changer would rival the work itself.
            .with_min_len(64)
            .map_init(
                || self.clone(),
                |changer, line| changer.apply(line.as_ref()),
            )
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse;

    fn changer(source: &str) -> CompiledRules {
        let statements = parse(source).expect("parse failed");
        crate::compiler::compile(&statements).expect("compile failed")
    }

    fn apply(source: &str, word: &str) -> String {
        changer(source).apply(word).expect("run failed")
    }

    #[test]
    fn basic_replacement() {
        assert_eq!(apply("shift:\n  a => e\n", "banana"), "benene");
    }

    #[test]
    fn multi_segment_and_sequences() {
        // 1 from-element, 2 to-elements: independent fallback
        assert_eq!(apply("break:\n  i => j a\n", "bi"), "bja");
        // element-wise pairing
        assert_eq!(apply("swap:\n  a b => o p\n", "ab"), "op");
    }

    #[test]
    fn simultaneous_overlap_resolution() {
        // claims at 0 and 2; the claim starting at 1 overlaps and drops
        assert_eq!(apply("rule:\n  a a => b\n", "aaaa"), "bb");
        assert_eq!(apply("rule:\n  a a => b\n", "aaa"), "ba");
    }

    #[test]
    fn expression_precedence() {
        // earlier expressions claim first; later ones get what's left
        let src = "rule:\n  a b => x\n  a => y\n";
        assert_eq!(apply(src, "aab"), "yx");
    }

    #[test]
    fn environments() {
        assert_eq!(apply("drop-final:\n  a => * / _ $\n", "banana"), "banan");
        assert_eq!(apply("drop-final:\n  a => * / _ $\n", "ab"), "ab");
        assert_eq!(apply("insert:\n  * => a / t _ t\n", "tt"), "tat");
        // exclusion wins over the condition
        let src = "lenite:\n  t => d / a _ a // _ a a\n";
        assert_eq!(apply(src, "ata"), "ada");
        assert_eq!(apply(src, "ataa"), "ataa");
    }

    #[test]
    fn environment_list_and_lookbehind() {
        let src = "drop:\n  k => * / {$ _, _ $}\n";
        assert_eq!(apply(src, "kakak"), "aka");
    }

    #[test]
    fn matrices_and_variables() {
        let src = "Feature Nasality(*oral, nasal)\n\
                   Feature Place(*alveolar, labial, velar)\n\
                   Symbol n [nasal]\n\
                   Symbol m [nasal labial]\n\
                   Symbol ŋ [nasal velar]\n\
                   Symbol b [labial]\n\
                   Symbol g [velar]\n\
                   assimilate:\n  [nasal] => [$Place] / _ [$Place]\n";
        assert_eq!(apply(src, "anba"), "amba");
        assert_eq!(apply(src, "anga"), "aŋga");
        assert_eq!(apply(src, "anta"), "anta");
    }

    #[test]
    fn captures() {
        assert_eq!(apply("redup:\n  (a b?)$1 => $1 $1\n", "ab"), "abab");
        assert_eq!(apply("redup:\n  (a b?)$1 => $1 $1\n", "a"), "aa");
        // back-reference in match position: only doubled segments reduce
        let src = "degeminate:\n  ([])$1 $1 => $1\n";
        assert_eq!(apply(src, "aabcc"), "abc");
        assert_eq!(apply(src, "abc"), "abc");
    }

    #[test]
    fn floating_diacritics_transfer() {
        let src = "Feature +stressed\n\
                   Diacritic ˈ (floating) [+stressed]\n\
                   shift:\n  a => e\n";
        assert_eq!(apply(src, "aˈb"), "eˈb");
        // exact text suppresses both the floating match and the transfer
        let exact = "Feature +stressed\n\
                     Diacritic ˈ (floating) [+stressed]\n\
                     shift:\n  a! => e\n";
        assert_eq!(apply(exact, "aˈb"), "aˈb");
    }

    #[test]
    fn blocks() {
        let sequential = "chain:\n  a => e\n  Then:\n  e => i\n";
        assert_eq!(apply(sequential, "a"), "i");
        let first_matching = "alt:\n  a => e\n  Else:\n  b => c\n";
        assert_eq!(apply(first_matching, "ab"), "eb");
        assert_eq!(apply(first_matching, "b"), "c");
    }

    #[test]
    fn propagate() {
        let src = "spread:\n  a => b / _ b\n";
        assert_eq!(apply(src, "aaab"), "aabb");
        let prop = "spread propagate:\n  a => b / _ b\n";
        assert_eq!(apply(prop, "aaab"), "bbbb");
    }

    #[test]
    fn ltr_feeds_its_own_output() {
        // simultaneous: only the `a` after the original `b` changes;
        // ltr rescans and feeds on the segment it just rewrote
        let simul = "rule:\n  a => b / b _\n";
        assert_eq!(apply(simul, "baa"), "bba");
        let ltr = "rule ltr:\n  a => b / b _\n";
        assert_eq!(apply(ltr, "baa"), "bbb");
    }

    #[test]
    fn filters() {
        let src = "Feature Type(*cons, vowel)\n\
                   Feature Height(*low, high)\n\
                   Symbol a [vowel]\n\
                   Symbol i [vowel high]\n\
                   harmony [vowel]:\n  a => i / i _\n";
        // the k is invisible: i…a count as adjacent
        assert_eq!(apply(src, "ika"), "iki");
        assert_eq!(apply(src, "aka"), "aka");
    }

    #[test]
    fn alternative_pairing() {
        let src = "rotate:\n  {p, t, k} => {t, k, p}\n";
        assert_eq!(apply(src, "ptk"), "tkp");
    }

    /// Parse or compile, expecting failure; returns the error message.
    fn reject(source: &str) -> String {
        let statements = match parse(source) {
            Err(e) => return e.to_string(),
            Ok(s) => s,
        };
        match crate::compiler::compile(&statements) {
            Err(e) => e.to_string(),
            Ok(_) => panic!("expected rejection, but compiled: {source:?}"),
        }
    }

    // statement-level validation (verified against the Kotlin CLI)

    #[test]
    fn inter_romanizers_get_pairing_checks() {
        // Inter-romanizer stages aren't pipeline steps, but kotlin still
        // builds their transformers; a from/to count mismatch there is a
        // compile error (fuzzer seeds 41001700/41002059, kotlin-verified:
        // "Found 3 elements ... but 4 elements"; a parenthesized group
        // counts as one element).
        // (`[stp]` keeps the emitter non-independent, so the count
        // mismatch can't take the independence fallback.)
        let bad = "Feature manner(stp, frc)\nFeature place(lab, alv)\nSymbol p [stp lab]\nSymbol t [stp alv]\n\
                   Class cv {t, p}\n\nr0:\n    unchanged\n\n\
                   romanizer-im:\n    p? (@cv @cv) a => e [stp] * o\n";
        assert!(reject(bad).contains("elements"), "{}", reject(bad));
        // …while a well-paired inter-romanizer still compiles.
        let good = "Feature manner(stp, frc)\nFeature place(lab, alv)\nSymbol p [stp lab]\nSymbol t [stp alv]\n\
                    Class cv {t, p}\n\nr0:\n    unchanged\n\n\
                    romanizer-im:\n    p? (@cv @cv) a => e [stp] o\n";
        let statements = parse(good).unwrap();
        crate::compiler::compile(&statements).expect("well-paired stage compiles");
    }

    #[test]
    fn duplicate_inter_romanizer_names_are_allowed() {
        // Unlike the deromanizer and final romanizer (kotlin rejects a second
        // of either via `extractRomanizerContext`'s `singleOrNullOrThrow` →
        // `LscDuplicateName`), *intermediate* romanizers are just collected as
        // a list, so two stages may share a name: both run, neither errors.
        // The stages aren't pipeline steps, so the word flows past unchanged.
        let src = "main:\n    a => b\n\
                   Romanizer-x:\n    b => c\n\
                   Romanizer-x:\n    c => d\n\
                   last:\n    b => z\n";
        assert_eq!(apply(src, "aa"), "zz");

        // …but a duplicate *final* romanizer is still rejected.
        let dup_final = "main:\n    a => b\n\
                         Romanizer:\n    b => c\n\
                         Romanizer:\n    c => d\n";
        assert!(reject(dup_final).contains("more than once"));
    }

    #[test]
    fn an_expression_outside_a_named_rule_is_rejected() {
        // A bare `a => b` with no `rule-name:` header isn't a statement on its
        // own (kotlin: "isn't in a rule; put it after a line like
        // \"rule-name:\""). lauturgie parses it as a top-level expression and
        // rejects it during lowering.
        assert!(reject("a => b\n").contains("outside a named rule"));
    }

    #[test]
    fn statements_must_be_ordered() {
        // lexurgy's validateOrder: features < diacritics/symbols <
        // classes/elements < deromanizer < syllables/rules < romanizer
        let msg = reject("class cv {a, b}\nfeature +lng\n\nr0:\n    a => b\n");
        assert!(msg.contains("must come after"), "got: {msg}");
        let msg = reject("symbol q\nfeature +lng\n\nr0:\n    a => b\n");
        assert!(msg.contains("must come after"), "got: {msg}");
        let msg = reject("syllables:\n    explicit\nderomanizer:\n    a => b\n");
        assert!(msg.contains("must come after"), "got: {msg}");
        // syllable declarations rank *with* the rules, not before them
        let src = "r0:\n    a => b\n\nsyllables:\n    explicit\n\nr1:\n    b => c\n";
        assert_eq!(apply(src, "a.a"), "c.c");
    }

    #[test]
    fn inexact_capture_refs_are_matchers_only() {
        // `~$1` re-matches a capture inexactly; lexurgy rejects it as an
        // emitter (CaptureReferenceElement.emitter → LscIllegalStructureInOutput)
        let msg = reject("r0:\n    (a)$1 b => ~$1 c\n");
        assert!(msg.contains("inexact capture reference"), "got: {msg}");
        // as a matcher it stays legal: `~$1` re-matches the capture with or
        // without floating diacritics
        let src = "feature +lng\ndiacritic ː (floating) [+lng]\n\nr0:\n    (a)$1 ~$1 => x *\n";
        assert_eq!(apply(src, "aaː"), "x");
        assert_eq!(apply(src, "aa"), "x");
        assert_eq!(apply(src, "ab"), "ab");
    }

    // declaration & matrix reject parity (verified against Kotlin CLI)

    #[test]
    fn element_redefinition_keeps_the_last_definition() {
        // Unlike a class (lexurgy rejects a duplicate class name), an
        // `Element` may be redefined; lexurgy's element loop just overwrites
        // `definedElements[name]`, so the LAST declaration wins. kotlin-
        // verified: with `@e => x`, input "a" is unchanged and "b" → "x".
        let src = "element e a\nelement e b\n\nr0:\n    @e => x\n";
        assert_eq!(apply(src, "a"), "a");
        assert_eq!(apply(src, "b"), "x");
        // An element overrides a class of the same name (elements resolve
        // first), matching lexurgy's `definedClasses + definedElements`.
        let clash = "class e {a}\nelement e b\n\nr0:\n    @e => x\n";
        assert_eq!(apply(clash, "a"), "a");
        assert_eq!(apply(clash, "b"), "x");
    }

    #[test]
    fn duplicate_declarations_are_rejected() {
        // lexurgy's LscDuplicateName: repeated feature/class/symbol/diacritic
        // *names*, and a feature *value* shared between two features.
        assert!(reject("feature +foo\nfeature +foo\n\nr0:\n    a => b\n")
            .contains("defined more than once"));
        assert!(
            reject("feature one(low, high)\nfeature two(low, deep)\n\nr0:\n    a => b\n")
                .contains("defined more than once")
        );
        assert!(reject("class c {a}\nclass c {b}\n\nr0:\n    a => b\n")
            .contains("defined more than once"));
        assert!(
            reject("symbol a\nsymbol a\n\nr0:\n    a => b\n").contains("defined more than once")
        );
        assert!(reject(
            "feature +long\ndiacritic \u{02d0} [+long]\ndiacritic \u{02d0} [+long]\n\nr0:\n    a => b\n"
        )
        .contains("defined more than once"));
        // …and two declarations with the *same matrix* (lexurgy: "both have
        // the matrix …; add features to make them distinct").
        assert!(reject(
            "feature +vowel\nsymbol a [+vowel]\nsymbol e [+vowel]\n\nr0:\n    a => b\n"
        )
        .contains("same matrix"));
        assert!(reject(
            "feature +long\ndiacritic \u{02d0} [+long]\ndiacritic \u{02d1} [+long]\n\nr0:\n    a => b\n"
        )
        .contains("same matrix"));
    }

    #[test]
    fn matrix_value_errors_are_rejected() {
        // `*name` resolves a feature by name; an undefined one is rejected.
        assert!(reject("r0:\n    a => [*nosuch]\n").contains("not defined"));
        // A negated value makes no sense in an *output* matrix.
        assert!(reject("feature +nas\n\nr0:\n    a => [!+nas]\n")
            .contains("negated value in output matrix"));
        // Two values of one feature in an output matrix is a repeat.
        assert!(
            reject("feature voice(vcd, vls)\n\nr0:\n    a => [vcd vls]\n")
                .contains("multiple values of the feature")
        );
    }

    #[test]
    fn wrong_level_feature_in_declaration_is_rejected() {
        // A syllable-level value can't appear in a (segment-level) symbol.
        assert!(
            reject("feature (syllable) +stress\nsymbol x [+stress]\n\nr0:\n    a => b\n")
                .contains("can't be used at this level")
        );
        // A diacritic mixing a segment- and a syllable-level feature.
        assert!(reject(
            "feature +long\nfeature (syllable) +stress\ndiacritic \u{0303} [+long +stress]\n\nr0:\n    a => b\n"
        )
        .contains("mixes segment- and syllable-level"));
        // Two values of one feature in a *declaration* matrix is a repeat.
        assert!(
            reject("feature voice(vcd, vls)\nsymbol x [vcd vls]\n\nr0:\n    a => b\n")
                .contains("multiple values of the feature")
        );
    }

    #[test]
    fn feature_space_exhaustion_is_a_resilience_guard() {
        // lexurgy has no such limit; lauturgie packs feature values into a
        // 128-bit word per level, so 129 binary segment features overflow it.
        // Not a parity test; a guard against pathological declarations.
        let mut src = String::new();
        for i in 0..129 {
            src.push_str(&format!("feature +f{i}\n"));
        }
        src.push_str("\nr0:\n    a => b\n");
        assert!(reject(&src).contains("too many segment-level"));
    }

    #[test]
    fn too_many_diacritics_is_a_resilience_guard() {
        // lexurgy has no such limit; lauturgie tracks attached diacritics in a
        // 64-bit mask, so a 65th distinct diacritic overflows it. Also not a
        // parity test. Each needs a distinct name (combining char) and matrix.
        let mut src = String::new();
        for i in 0..65 {
            src.push_str(&format!("feature +f{i}\n"));
        }
        for i in 0..65 {
            let mark = char::from_u32(0x0300 + i).unwrap();
            src.push_str(&format!("diacritic {mark} [+f{i}]\n"));
        }
        src.push_str("\nr0:\n    a => b\n");
        assert!(reject(&src).contains("too many diacritics"));
    }

    // filtered-rule semantics (verified against the Kotlin CLI)

    #[test]
    fn filtered_rule_drops_surplus_result_elements() {
        // `a` is the only filtered segment of "cabc"; the surplus `y` claims
        // the same real segment as `x` and is silently dropped; lexurgy's
        // unfilterTransformations maps every elemental bit onto one segment.
        let src = "class cv {a, b}\n\nr0 @cv:\n    a => x y\n";
        assert_eq!(apply(src, "cabc"), "cxbc");
        // unfiltered, the same expression emits both segments
        assert_eq!(apply("r0:\n    a => x y\n", "cabc"), "cxybc");
    }

    #[test]
    fn filtered_rule_multi_segment_piece_replaces_first_segment() {
        // the group `(b a)` matches two filtered segments but its result
        // claims only the first; the second survives untouched
        let src = "class cv {a, b}\n\nr0 @cv:\n    a (b a) => x y\n";
        assert_eq!(apply(src, "acba"), "xcya");
    }

    #[test]
    fn filtered_rule_zero_width_piece_consumes_a_segment() {
        // `b*` matches zero-width before "a"; its bit still claims the
        // segment, so the paired `a => f` bit collides and drops (kotlin
        // "iv" → "uv" with `i* i => u f`)
        let src = "class cv {a, b}\n\nr0 @cv:\n    b* a => u f\n";
        assert_eq!(apply(src, "ac"), "uc");
    }

    #[test]
    fn filtered_rule_zero_width_match_at_end_is_word_error() {
        // a zero-width match after the last filtered segment indexes past
        // the filter map: a per-word error in both engines
        let src = "class cv {a, b}\n\nr0 @cv:\n    a? => x\n";
        let mut compiled = changer(src);
        assert!(compiled.apply("ca").is_err());
    }

    // pairing validation (lexurgy's transformerTo, at compile time)

    #[test]
    fn mismatched_counts_need_independent_results() {
        // independent surplus: whole result replaces each whole match
        assert_eq!(apply("r0:\n    a b => x y z\n", "cabc"), "cxyzc");
        // a conditional element (matrix) forbids the fallback
        let src = "feature voice(uvc, vcd)\nsymbol p [vcd]\n\nr0:\n    a b => x [vcd] z\n";
        assert!(reject(src).contains("element"));
        // alternative emitters are never independent
        assert!(reject("r0:\n    a => {x, b}\n").contains("element"));
    }

    #[test]
    fn filtered_rules_reject_unpairable_shapes() {
        let reject_filtered =
            |expr: &str| reject(&format!("class cv {{a, b}}\n\nr0 @cv:\n    {expr}\n"));
        assert!(reject_filtered("ab => x").contains("multi-segment"));
        assert!(reject_filtered("* => x / a _").contains("empty element"));
        assert!(reject_filtered("a b => x y z").contains("element"));
        assert!(reject_filtered("a* => x y").contains("element"));
        // equal counts pair fine under a filter
        let src = "class cv {a, b}\n\nr0 @cv:\n    a b => x y\n";
        assert_eq!(apply(src, "cabc"), "cxyc");
    }

    #[test]
    fn expression_environment_shields_filtered_checks() {
        // the environment wraps the matcher in an EnvironmentMatcher, and an
        // independent non-conditional emitter pairs with it without ever
        // visiting the multi-segment text (kotlin accepts + runs this)
        let src = "class cv {s, g, b}\n\nr0 @cv:\n    sg => * / _ g\n";
        assert_eq!(apply(src, "sgg"), "gg");
        // without the environment the text is checked and rejected
        assert!(reject("class cv {s, g, b}\n\nr0 @cv:\n    sg => *\n").contains("multi-segment"));
        // a conditional emitter lifts through the wrapper and still rejects
        assert!(
            reject("class cv {s, g, b}\n\nr0 @cv:\n    sg => p / _ g\n").contains("multi-segment")
        );
    }

    #[test]
    fn peripheral_repeaters_rejected_at_open_env_edges() {
        assert!(reject("r0:\n    x => y / a* _\n").contains("repeater"));
        assert!(reject("r0:\n    x => y / _ a+\n").contains("repeater"));
        assert!(reject("r0:\n    x => y / b _ // a* _\n").contains("repeater"));
        // anchored, interior, and exact-count repeaters are fine
        assert_eq!(apply("r0:\n    x => y / $ a* _\n", "ax"), "ay");
        assert_eq!(apply("r0:\n    x => y / b a* _\n", "bax"), "bay");
        assert_eq!(apply("r0:\n    x => y / _ a*2\n", "xaa"), "yaa");
    }

    // divergence budgets

    #[test]
    fn diverging_scans_error_instead_of_hanging() {
        // zero-width source inserting under ltr grows forever
        let mut ltr = changer("r0 ltr:\n    a* => a / _ u\n");
        assert_eq!(ltr.apply("bub"), Err(RunError::DivergingScan));
        // growing target under ltr (hangs even kotlin's CLI rescue)
        let mut gem = changer("r0 ltr:\n    ([])$1 => $1 $1\n");
        assert_eq!(gem.apply("ab"), Err(RunError::DivergingScan));
        // growing target under propagate trips the growth budget
        let mut prop = changer("r0 propagate:\n    * => a / a _\n");
        assert_eq!(prop.apply("ab"), Err(RunError::DivergingPropagation));
    }

    #[test]
    fn repeaters() {
        assert_eq!(apply("squash:\n  a a+ => a\n", "aaab"), "ab");
        assert_eq!(apply("squash:\n  a a+ => a\n", "ab"), "ab");
    }

    // syllables

    #[test]
    fn explicit_syllables_pass_through() {
        let src = "Class vowel {a, e, i, o, u}\n\
                   Syllables:\n  explicit\n\
                   voice:\n  {p, t, k} => {b, d, g} / @vowel _ @vowel\n";
        assert_eq!(apply(src, "ko.li.mo"), "ko.li.mo");
        assert_eq!(apply(src, "ko.pi.ko"), "ko.bi.go");
    }

    #[test]
    fn insert_and_delete_breaks() {
        let insert = "Class vowel {a, e, i, o, u}\n\
                      Syllables:\n  explicit\n\
                      break:\n  * => . / @vowel _ @vowel\n";
        assert_eq!(apply(insert, "muo"), "mu.o");
        let delete = "Class vowel {a, e, i, o, u}\n\
                      Syllables:\n  explicit\n\
                      join:\n  . => * / @vowel _ @vowel\n";
        assert_eq!(apply(delete, "mu.o.ti"), "muo.ti");
    }

    #[test]
    fn dots_without_syllables_are_text() {
        let src = "Class vowel {a, e, i, o, u}\n\
                   break:\n  * => . / @vowel _ @vowel\n";
        assert_eq!(apply(src, "muo"), "mu.o");
    }

    #[test]
    fn automatic_syllabification() {
        let src = "Class vowel {a, e, i, o, u}\n\
                   Class cons {p, t, k, s, m, n, l}\n\
                   Syllables:\n  @cons? @vowel @cons?\n";
        assert_eq!(apply(src, "patan"), "pa.tan");
        assert_eq!(apply(src, "antas"), "an.tas");
    }

    #[test]
    fn syllable_level_features() {
        let src = "Feature (syllable) +stress\n\
                   Diacritic ˈ (before) [+stress]\n\
                   Class vowel {a, e, i, o, u}\n\
                   Class cons {p, t, k, s, m, n, l}\n\
                   Syllables:\n  @cons? @vowel @cons?\n\
                   stress-first:\n  <syl> => [+stress] / $ _\n";
        assert_eq!(apply(src, "patan"), "ˈpa.tan");
    }

    #[test]
    fn syllable_matrix_matching() {
        // closed syllables get [+heavy], spelled with the ˌ diacritic;
        // a syllable matrix in match position reads it back
        let src = "Feature (syllable) +heavy\n\
                   Diacritic ˌ [+heavy]\n\
                   Class vowel {a, e, i, o, u}\n\
                   Class cons {p, t, k, s, m, n, l}\n\
                   Syllables:\n  @cons? @vowel @cons => [+heavy]\n  @cons? @vowel\n";
        assert_eq!(apply(src, "patan"), "pa.tanˌ");
        let matching = "Feature (syllable) +heavy\n\
                        Diacritic ˌ [+heavy]\n\
                        Class vowel {a, e, i, o, u}\n\
                        Class cons {p, t, k, s, m, n, l}\n\
                        Syllables:\n  @cons? @vowel @cons => [+heavy]\n  @cons? @vowel\n\
                        raise:\n  a&[+heavy] => e\n";
        assert_eq!(apply(matching, "patan"), "pa.tenˌ");
    }

    #[test]
    fn syllable_features_assigned_with_conditions() {
        // conditional syllabifier assignment (light/heavy) feeds an
        // intersection-gated stress rule with an Else: fallback; exercises
        // syl_change + word_with_changes through the richer path.
        // Byte-identical to lexurgy's TestSyllables "assign syllable-level
        // features in syllable patterns with conditions".
        let src = "Feature (syllable) +stress\n\
                   Feature (syllable) sylType(light, heavy)\n\
                   Feature +long\n\
                   Diacritic ˈ (before) [+stress]\n\
                   Diacritic ¹ [light]\n\
                   Diacritic ² [heavy]\n\
                   Diacritic ː (floating) [+long]\n\
                   Class vowel {a, e, i, o, u}\n\
                   Class diphthong {aj, oj}\n\
                   Class cons {p, t, k, b, d, g, f, s, h, m, n, l, r, w}\n\
                   Syllables:\n\
                     @cons? {@vowel, @diphthong} @cons => [heavy]\n\
                     @cons? @diphthong => [heavy]\n\
                     @cons? @vowel&[+long] => [heavy]\n\
                     @cons? @vowel / _ $ => [heavy]\n\
                     @cons? @vowel => [light]\n\
                   stress:\n\
                     <syl>&[heavy] => [+stress] / _ <syl> $\n\
                     Else:\n\
                     <syl> => [+stress] / _ <syl> <syl> $\n\
                   coda-dropping:\n  @cons => * / _ @cons\n";
        assert_eq!(apply(src, "feːminaj"), "ˈfeː².mi¹.naj²");
        assert_eq!(apply(src, "wolukris"), "wo¹.ˈlu¹.ris²");
    }

    #[test]
    fn any_syllable_matcher() {
        let src = "Class vowel {a, e, i, o, u}\n\
                   Class cons {p, t, k, s, m, n, l}\n\
                   Syllables:\n  @cons? @vowel @cons?\n\
                   drop-final-syllable:\n  <syl> => * / _ $\n";
        assert_eq!(apply(src, "patan"), "pa");
    }

    #[test]
    fn syllables_clear() {
        let src = "Syllables:\n  explicit\n\
                   first:\n  a => e\n\
                   Syllables:\n  clear\n\
                   second:\n  e => i\n";
        // after the clear, words lose their structure (and their dots)
        assert_eq!(apply(src, "ka.ta"), "kiti");
    }

    // romanizers

    #[test]
    fn deromanizer_and_romanizer() {
        let src = "Deromanizer:\n  sh => ʃ\n\
                   shift:\n  ʃ => s\n\
                   Romanizer:\n  s => z\n";
        assert_eq!(apply(src, "ash"), "az");
    }

    #[test]
    fn literal_romanizer() {
        let src = "Symbol ts\n\
                   Deromanizer literal:\n  c => ts\n\
                   Romanizer literal:\n  ts => c\n";
        assert_eq!(apply(src, "aca"), "aca");
    }

    #[test]
    fn literal_romanizers_with_then_chains() {
        // (every value verified against the Kotlin CLI)
        // A literal *deromanizer* with a `Then:` chain runs its first block
        // in the literal universe (input as plain chars), redeclares to the
        // real universe, then runs the rest there.
        let derom = "Feature Type(*cons, vowel)\n\
                     Symbol a [vowel]\n\
                     Deromanizer literal:\n  c => k\n  Then:\n  k => g\n\
                     main:\n  g => x\n";
        // cca: c=>k (literal) → kka; k=>g (real) → gga; g=>x → xxa
        assert_eq!(apply(derom, "cca"), "xxa");
        assert_eq!(apply(derom, "kka"), "xxa");

        // A literal *romanizer* with a `Then:` chain runs its front blocks in
        // the real universe and only the *last* block in the literal one.
        let rom = "main:\n  a => b\n\
                   Romanizer literal:\n  b => d\n  Then:\n  d => q\n";
        // a=>b → bb; b=>d (real) → dd; d=>q (literal) → qq
        assert_eq!(apply(rom, "aa"), "qq");

        // A literal *inter*-romanizer with a `Then:` chain is validated but
        // not part of the pipeline, so the word flows past it unchanged.
        let inter = "main:\n  a => b\n\
                     Romanizer-mid literal:\n  b => c\n  Then:\n  c => d\n\
                     done:\n  b => z\n";
        assert_eq!(apply(inter, "aa"), "zz");
    }

    // deferred rules (`:name` splices, verified against the Kotlin CLI)

    #[test]
    fn deferred_rules_splice_at_their_reference() {
        // A `defer`d rule isn't a pipeline step; it only runs where `:name`
        // splices it.
        let src = "raise defer:\n  a => e\n\
                   main:\n  :raise\n  o => u\n";
        assert_eq!(apply(src, "ao"), "eu");
        // never referenced ⇒ never runs
        assert_eq!(
            apply("noop defer:\n  a => e\n\nmain:\n  o => u\n", "ao"),
            "au"
        );
    }

    #[test]
    fn duplicate_deferred_rule_names_keep_the_last() {
        // Unlike *plain* rules (kotlin rejects a duplicate name,
        // `duplicate_declarations_are_rejected`), a redeclared `defer`d rule
        // is *not* an error; kotlin's `resolveBlocks` does
        // `blocks.associate { it.rule.name to it.rule }`, so the last
        // declaration wins. Here `:r` splices the second `r` (`a => c`).
        let src = "r defer:\n  a => b\n\
                   r defer:\n  a => c\n\
                   main:\n  :r\n";
        assert_eq!(apply(src, "a"), "c");

        // …but a *plain* duplicate rule name is still rejected.
        assert!(reject("r:\n  a => b\n\nr:\n  a => c\n").contains("more than once"));
    }

    // multi-word phrases (every value verified against the Kotlin CLI)

    #[test]
    fn matches_fuse_words_across_the_gap() {
        let src = "fuse:\n a $$ b => c\n";
        assert_eq!(apply(src, "ta bu"), "tcu");
        assert_eq!(apply(src, "a b"), "c");
        // the gap must really be between an `a` and a `b`
        assert_eq!(apply(src, "ta ab"), "ta ab");
        // and `$$` never matches inside a single word
        assert_eq!(apply(src, "ab"), "ab");
    }

    #[test]
    fn emitted_gaps_split_words() {
        let src = "split:\n x => y $$ z\n";
        assert_eq!(apply(src, "axa"), "ay za");
        assert_eq!(apply(src, "x"), "y z");
        assert_eq!(apply(src, "tax"), "tay z");
    }

    #[test]
    fn deleting_the_gap_joins_words() {
        let src = "delgap:\n $$ => *\n";
        assert_eq!(apply(src, "ta bu"), "tabu");
        assert_eq!(apply(src, "a b c"), "abc");
        assert_eq!(apply(src, "solo"), "solo");
        // syllable structure stitches across the join
        let syl = "Syllables:\n explicit\nfusebreak:\n $$ => *\n";
        assert_eq!(apply(syl, "ta.ka bu.mi"), "ta.kabu.mi");
    }

    #[test]
    fn gap_replaced_by_text_joins_words_around_it() {
        let src = "gaptotext:\n $$ => a\n";
        assert_eq!(apply(src, "ta bu"), "taabu");
        assert_eq!(apply(src, "ta bu mi"), "taabuami");
        assert_eq!(apply(src, "solo"), "solo");
    }

    #[test]
    fn deleted_words_stay_as_empty_words() {
        // lexurgy keeps the empty word and renders the space around it
        let src = "delword:\n a => *\n";
        assert_eq!(apply(src, "a b"), " b");
        assert_eq!(apply(src, "b a"), "b ");
        assert_eq!(apply(src, "b a c"), "b  c");
    }

    #[test]
    fn word_anchors_match_at_every_word() {
        let src = "anchors:\n t => d / $ _\n k => g / _ $\n";
        assert_eq!(apply(src, "tak tak"), "dag dag");
        assert_eq!(apply(src, "atk atk"), "atg atg");
    }

    #[test]
    fn gaps_in_environments() {
        let src = "envgap:\n a => o / _ $$\n b => p / $$ _\n";
        assert_eq!(apply(src, "ba ba ba"), "bo po pa");
        assert_eq!(apply(src, "ab ab"), "ab ab");
        let ins = "insgap:\n * => i / $$ _\n";
        assert_eq!(apply(ins, "ta bu"), "ta ibu");
        assert_eq!(apply(ins, "solo"), "solo");
    }

    #[test]
    fn text_never_matches_across_the_gap() {
        let src = "nocross:\n ab => x\n";
        assert_eq!(apply(src, "a b"), "a b");
        assert_eq!(apply(src, "ab ab"), "x x");
    }

    #[test]
    fn directional_scans_walk_the_whole_phrase() {
        let ltr = "g ltr:\n aa => b\n";
        assert_eq!(apply(ltr, "aaa aaa"), "ba ba");
        assert_eq!(apply(ltr, "a aa"), "a b");
        let rtl = "g rtl:\n a => b / _ $$\n";
        assert_eq!(apply(rtl, "aa aa aa"), "ab ab aa");
    }

    #[test]
    fn else_blocks_run_per_word() {
        // lexurgy wraps FirstMatchingBlock in WithinWordBlock: each word
        // picks its own arm.
        let src = "e:\n t => d / a _\n Else:\n k => g\n";
        assert_eq!(apply(src, "atk kt"), "adk gt");
        assert_eq!(apply(src, "kk at"), "gg ad");
    }

    #[test]
    fn else_block_splitting_a_word_is_an_error() {
        // kotlin's WithinWordBlock calls `.single()` on the result
        let src = "e:\n a => *\n Else:\n k => g $$ g\n";
        assert!(changer(src).apply("kk kk").is_err());
    }

    #[test]
    fn captures_can_span_the_gap_in_environments() {
        let src = "cap:\n (a b)$1 => * / _ $$ $1\n";
        assert_eq!(apply(src, "ab ab"), " ab");
        assert_eq!(apply(src, "ab ba"), "ab ba");
    }

    #[test]
    fn filtered_rules_filter_each_word() {
        let src = "Class vowel {a, e, i, o, u}\nf @vowel:\n a => e / _ i\n";
        assert_eq!(apply(src, "tati tati"), "teti teti");
        // the filtered view never bridges words
        assert_eq!(apply(src, "tat ti"), "tat ti");
        assert_eq!(apply(src, "ta atti"), "ta etti");
        let two = "Class vowel {a, e, i, o, u}\nf @vowel:\n a i => e *\n";
        assert_eq!(apply(two, "tati tati"), "tet tet");
        assert_eq!(apply(two, "ta it"), "ta it");
    }

    #[test]
    fn gap_emitters_inside_sequences() {
        let src = "g:\n a $$ => * $$\n";
        assert_eq!(apply(src, "ta bu"), "t bu");
        assert_eq!(apply(src, "taa bu"), "ta bu");
    }

    #[test]
    fn phrase_input_splits_on_single_spaces() {
        // consecutive spaces produce empty words, like lexurgy's split(" ")
        let src = "x:\n a => b\n";
        assert_eq!(apply(src, "a  a"), "b  b");
        assert_eq!(apply(src, " a a "), "b b");
    }

    // environments on nested elements (verified against the Kotlin CLI)

    #[test]
    fn nested_environments_in_lists_and_groups() {
        // a list alternative with its own local environment
        let src = "r:\n {p / a _, t} => x\n";
        assert_eq!(apply(src, "apa"), "axa");
        assert_eq!(apply(src, "ipi"), "ipi");
        assert_eq!(apply(src, "patate"), "paxaxe");
        // a group, with condition and with exclusion
        assert_eq!(apply("r:\n (p / _ a) => b\n", "apap"), "abap");
        assert_eq!(apply("r:\n (p // _ a) => b\n", "apap"), "apab");
        // an element declaration carrying an environment
        let src = "element foo p / a _\nr:\n @foo => x\n";
        assert_eq!(apply(src, "apa pap"), "axa pax");
    }

    #[test]
    fn nested_environments_inside_environments() {
        // the env-before is itself environment-conditioned: `a` only
        // counts when preceded by `b`
        let src = "r:\n t => d / (a / b _) _\n";
        assert_eq!(apply(src, "bat"), "bad");
        assert_eq!(apply(src, "at"), "at");
        // insertion conditioned on a nested environment
        let src = "r:\n * => e / (t / a _) _\n";
        assert_eq!(apply(src, "atka tka"), "ateka tka");
    }

    #[test]
    fn nested_environments_compose_with_other_features() {
        // nested env + expression env at once
        let src = "r:\n (p / a _) => b / _ i\n";
        assert_eq!(apply(src, "api apa ipi"), "abi apa ipi");
        // capture inside the conditioned element
        let src = "class stop {p, t, k}\nr:\n (@stop$1 / a _) => $1 $1\n";
        assert_eq!(apply(src, "apa ipa"), "appa ipa");
        // under a filter
        let src = "class vowel {a, e, i, o, u}\nr @vowel:\n (a / _ i) => o\n";
        assert_eq!(apply(src, "kaki kak aia"), "koki kak oia");
        // rtl scan
        let src = "r rtl:\n (a / t _) => o / _ t\n";
        assert_eq!(apply(src, "tatat atat"), "totot atot");
    }

    #[test]
    fn nested_environments_rejected_in_output() {
        // lexurgy: EnvironmentElement is not a ResultElement
        // (LscIllegalStructureInOutput, "a nested environment")
        for src in [
            "r:\n p => (a / b _)\n",
            "r:\n p => {a / b _, c}\n",
            "element foo a / b _\nr:\n p => @foo\n",
        ] {
            let msg = reject(src);
            assert!(msg.contains("nested environment"), "{src:?}: {msg}");
        }
        // the peripheral-repeater check applies inside nested environments
        let msg = reject("r:\n t => d / (a / b* _) _\n");
        assert!(msg.contains("repeater"), "{msg}");
    }

    #[test]
    fn transforming_interfix_is_rejected() {
        // `>` is lexurgy's `LscFutureStructure("Transforming elements")`:
        // valid syntax, not implemented by either side, so both reject.
        let msg = reject("r:\n a>b => c\n");
        assert!(msg.contains("transforming interfix"), "{msg}");
    }
}
