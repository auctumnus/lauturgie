// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) Autumn <auctumnus@pm.me>

//! The FST tier: compiled symbolic automata for eligible rules.
//!
//! The VM ([`crate::vm`]) is the *reference* executor: a faithful but slow
//! tree-walk. This module is the fast tier: rules whose patterns fit a
//! regular subset compile to automata whose transitions carry *predicates
//! on one segment* ([`SegTest`]) rather than concrete symbols, so the
//! alphabet stays open (new segments can be interned at runtime without
//! recompiling).
//!
//! Correctness strategy: the automaton must reproduce the VM's semantics
//! *exactly*, so eligibility is conservative and everything else falls back
//! to the VM at compile time (pattern shapes outside the subset) or at run
//! time (syllabified words). The differential suite runs the whole lexurgy
//! corpus through `CompiledRules::apply`, which prefers this tier; any
//! divergence from the VM (and from lexurgy) shows up there, and
//! `tests/fst.rs` additionally A/Bs the two tiers directly.
//!
//! What makes a rule eligible:
//!
//! - body is a single plain expression list, `simultaneous`, `ltr`, or
//!   `rtl` (no `Then:`/`Else:`, filters, `propagate`);
//! - patterns are regular: literal text, matrices *without feature
//!   variables*, sequences, alternatives, repeaters, single-segment
//!   negation, word edges; no captures, lookarounds, intersections, or
//!   syllable elements;
//! - outputs are verbatim text (no floating-diacritic transfer in force,
//!   no syllable structure), variable-free matrix rewrites, or paired
//!   alternations of those (`{p, t} => {b, d}`).
//!
//! Match preference is encoded structurally: alternation forks try their
//! branches in written order and repeaters loop greedily, so a prioritized
//! depth-first traversal enumerates match ends in exactly the VM's
//! backtracking order. Paired alternations ride along as [`State::Tag`]
//! accumulators: the DFS path through the forks adds up a variant index
//! (mixed-radix over the paired alternation sites, most significant first,
//! mirroring the VM's lexicographic option order), and the first path to
//! reach an end fixes that end's emitter: the same option the VM's
//! backtracking would pick.
//!
//! On top of the NFAs sit two compiled layers:
//!
//! - **Determinization over predicate minterms** ([`Dfa`]): each machine
//!   lazily builds a DFA whose states are priority-ordered NFA state
//!   tuples *truncated at the first Match*; truncation is what preserves
//!   leftmost-first preference through the subset construction, so a
//!   single forward scan yields the same preferred end the backtracking
//!   NFA would find. Transition labels are minterm classes: the bitset of
//!   predicate outcomes for a segment, computed per interned segment on
//!   first sight and memoized (the alphabet is open, so satisfiable
//!   minterms are discovered from the data rather than enumerated).
//!   Everything is budgeted; a machine that outgrows its budget falls
//!   back to its NFA permanently.
//! - **Cross-rule composition** ([`FusedRun`]): adjacent rules that are
//!   context-free single-segment maps (the automaton provably consumes
//!   exactly one segment, no anchors or environments) fuse into one pass
//!   over the word. The composed transducer is materialized lazily as a
//!   segment → segments cache over the realized alphabet, so a run of n
//!   such rules costs one lookup per segment instead of n scans.
//!
//! Still on the VM: captures/variables, syllable structure, filters,
//! `propagate`, blocks, and multi-word phrases.

use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hasher};

use crate::compiler::decls::Declarations;
use crate::compiler::features::{BitTest, FeatureId, FeatureWord, Level, MatrixUpdate};
use crate::compiler::ir::{BlockIr, Emit, EnvIr, ExprIr, MatchMode, Pattern, RuleIr, SegTest};
use crate::compiler::segments::{DiacriticMask, SegmentId, SegmentInterner};
use crate::compiler::{Step, Universe};

/// A `HashMap` keyed by small integers (segment ids, DFA classes, state
/// tuples). The apply-path caches are hit once per segment per scan, and
/// SipHash dominated their cost in profiles (~12% of the FST tier on
/// kharulian); these keys don't need DoS resistance, so we hash them with
/// the multiply-rotate FxHash instead.
type FastMap<K, V> = HashMap<K, V, BuildHasherDefault<FxHasher>>;

/// The FxHash finalizer (the hasher rustc itself uses), inlined to avoid a
/// dependency. Integer keys arrive whole through the `write_*` paths; the
/// byte-slice fallback folds 8 bytes at a time for the few `Box<[StateId]>`
/// keys hashed at DFA-build time.
#[derive(Default)]
struct FxHasher {
    hash: u64,
}

const FX_SEED: u64 = 0x51_7c_c1_b7_27_22_0a_95;

impl FxHasher {
    #[inline]
    fn add(&mut self, i: u64) {
        self.hash = (self.hash.rotate_left(5) ^ i).wrapping_mul(FX_SEED);
    }
}

impl Hasher for FxHasher {
    #[inline]
    fn write_u8(&mut self, i: u8) {
        self.add(i as u64);
    }
    #[inline]
    fn write_u16(&mut self, i: u16) {
        self.add(i as u64);
    }
    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.add(i as u64);
    }
    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.add(i);
    }
    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.add(i as u64);
    }
    #[inline]
    fn write_u128(&mut self, i: u128) {
        self.add(i as u64);
        self.add((i >> 64) as u64);
    }
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for chunk in bytes.chunks(8) {
            let mut buf = [0u8; 8];
            buf[..chunk.len()].copy_from_slice(chunk);
            self.add(u64::from_le_bytes(buf));
        }
    }
    #[inline]
    fn finish(&self) -> u64 {
        self.hash
    }
}
use crate::vm::RunError;
use crate::word::Word;

/// State-count budget per compiled pattern: hostile repeat bounds can't
/// blow up compilation (the rule just stays on the VM tier).
const MAX_STATES: usize = 4_096;
/// Cap on emit variants from paired alternations per expression.
const MAX_VARIANTS: usize = 64;
/// Determinization budgets: a machine whose DFA outgrows these stays on
/// its NFA. Predicates above 64 don't fit a minterm-class word.
const MAX_DFA_STATES: usize = 160;
const MAX_DFA_PREDS: usize = 64;
/// Composed-map cache cap per fused run; segments past the cap are
/// computed without caching (hostile inputs can't grow memory unboundedly).
const MAX_FUSED_CACHE: usize = 4_096;

type StateId = usize;

/// Why a rule stayed on the VM tier: a short phrase naming the first
/// ineligible construct hit. Diagnostic only (surfaced by `tier_report`);
/// the apply path only cares whether compilation succeeded.
pub type VmReason = &'static str;

mod apply;
mod automaton;
mod pieces;
#[cfg(test)]
mod tests;
mod variables;

// Pull the submodule items into the `fst` namespace so each submodule's
// `use super::*` sees its siblings (and mod.rs's own prelude flows down the
// same way). `apply` is the top layer: nothing references back into it, so
// it needs no such import here.
use automaton::*;
use pieces::*;
use variables::*;

/// One expression, as one or more variable-free *instances*. A single
/// instance is the common case; feature variables expand into one instance
/// per assignment of their domains (`expand_variables`), relying on value
/// exclusivity: a segment has exactly one value per feature, so at most
/// one instance's binding tests can pass at a given spot.
#[derive(Debug, Clone)]
struct ExprFst {
    instances: Vec<InstFst>,
    /// Instances differ in their `from` machines (variables in match
    /// position; the expansion gate makes such froms rigid, so all
    /// instances match the same single width and instance-major iteration
    /// equals the VM's option order).
    from_varied: bool,
    /// Untagged union of varied instance froms: a deterministic prefilter
    /// so positions where no instance can match (the vast majority) cost
    /// one DFA step instead of one probe per instance.
    gate: Option<Machine>,
}

#[derive(Debug, Clone)]
struct InstFst {
    from: Machine,
    /// Piece lists by variant tag (one entry unless alternations pair).
    emits: Vec<PieceList>,
    conditions: Vec<CompiledEnv>,
    exclusions: Vec<CompiledEnv>,
}

impl ExprFst {
    /// First transformation option at exactly `pos`, in the VM's preference
    /// order: `(end, instance, variant tag)` of the first claim whose
    /// environment passes (lexurgy's `claimAt`).
    fn claim_at(
        &mut self,
        ctx: &Ctx<'_>,
        word: &Word,
        pos: usize,
        ends: &mut Vec<(usize, u32)>,
    ) -> Option<(usize, usize, u32)> {
        if self.instances.len() == 1 {
            return self.instances[0]
                .claim_at(ctx, word, pos, ends)
                .map(|(end, tag)| (end, 0, tag));
        }
        if self.from_varied {
            if let Some(gate) = &mut self.gate {
                if !gate.matches(ctx, word, pos, Dir::Fwd) {
                    return None;
                }
            }
            // Rigid froms with exclusive bindings: at most one instance
            // matches a given spot, so instance order is immaterial.
            for (i, inst) in self.instances.iter_mut().enumerate() {
                if let Some((end, tag)) = inst.claim_at(ctx, word, pos, ends) {
                    return Some((end, i, tag));
                }
            }
            return None;
        }
        // Shared from: candidate ends are instance-independent; iterate
        // end-major (the VM checks environments per option, and the
        // environment binding selects the instance).
        self.instances[0]
            .from
            .nfa
            .claim(ctx, word, pos, Dir::Fwd, ends);
        for &(end, tag) in ends.iter() {
            for (inst_i, inst) in self.instances.iter_mut().enumerate() {
                if inst.env_ok(ctx, word, pos, end) {
                    return Some((end, inst_i, tag));
                }
            }
        }
        None
    }
}

impl InstFst {
    fn claim_at(
        &mut self,
        ctx: &Ctx<'_>,
        word: &Word,
        pos: usize,
        ends: &mut Vec<(usize, u32)>,
    ) -> Option<(usize, u32)> {
        if self.conditions.is_empty() && self.exclusions.is_empty() && self.emits.len() <= 1 {
            // No environment can reject an option, so only the preferred
            // end matters; one deterministic scan.
            return self
                .from
                .preferred_end(ctx, word, pos, Dir::Fwd)
                .map(|end| (end, 0));
        }
        // While the DFA is alive there are no variant tags (tags kill
        // determinization), so the preferred end *is* option (end, 0):
        // probe it deterministically and fall back to the full
        // backtracking list only when the environment rejects it. Most
        // positions don't match at all and most matches pass, so the
        // common cases never pay the NFA walk.
        if !self.from.dfa.dead {
            match self
                .from
                .dfa
                .scan(&self.from.nfa, ctx, word, pos, Dir::Fwd, false)
            {
                Ok(None) => return None,
                Ok(Some(end)) => {
                    if self.env_ok(ctx, word, pos, end) {
                        return Some((end, 0));
                    }
                    self.from.nfa.claim(ctx, word, pos, Dir::Fwd, ends);
                    for &(e, tag) in ends.iter() {
                        if e == end {
                            continue; // just rejected
                        }
                        if self.env_ok(ctx, word, pos, e) {
                            return Some((e, tag));
                        }
                    }
                    return None;
                }
                Err(DfaBudget) => self.from.dfa.dead = true,
            }
        }
        self.from.nfa.claim(ctx, word, pos, Dir::Fwd, ends);
        let mut chosen = None;
        for &(end, tag) in ends.iter() {
            if self.env_ok(ctx, word, pos, end) {
                chosen = Some((end, tag));
                break;
            }
        }
        chosen
    }

    fn env_ok(&mut self, ctx: &Ctx<'_>, word: &Word, lo: usize, hi: usize) -> bool {
        let positive = self.conditions.is_empty()
            || self
                .conditions
                .iter_mut()
                .any(|env| env.check(ctx, word, lo, hi));
        positive
            && !self
                .exclusions
                .iter_mut()
                .any(|env| env.check(ctx, word, lo, hi))
    }
}

/// A whole rule compiled to the FST tier: the block tree with each leaf's
/// expressions compiled to machines.
#[derive(Debug, Clone)]
pub struct RuleFst {
    body: BlockFst,
    /// Mask of floating segment-level diacritics (for transfer pieces).
    floating: DiacriticMask,
    /// The rule contains a `propagate` block. Its divergence budgets
    /// (work, growth, seen-set) are *phrase-global* in the VM but per-word
    /// here, so multi-word phrases must stay on the VM to trip them at the
    /// same point.
    propagates: bool,
    /// `Ok` when the emit side compiled into pieces, so the full FST apply
    /// path (match *and* splice) is available, for unsyllabified words.
    /// `Err` carries why not; the rule then still serves as a match gate
    /// for the VM ([`FstGate`]).
    splice: Result<(), VmReason>,
}

impl RuleFst {
    pub fn propagates(&self) -> bool {
        self.propagates
    }

    pub fn splices(&self) -> bool {
        self.splice.is_ok()
    }

    pub fn splice_reason(&self) -> Option<VmReason> {
        self.splice.err()
    }

    /// Borrow the rule's expression machinery as a VM position gate.
    pub fn gate(&mut self) -> FstGate<'_> {
        let mut leaves = Vec::new();
        collect_leaves(&mut self.body, &mut leaves);
        FstGate {
            leaves,
            ends: Vec::new(),
        }
    }
}

fn collect_leaves<'g>(block: &'g mut BlockFst, out: &mut Vec<&'g mut Vec<ExprFst>>) {
    match block {
        BlockFst::Exprs { exprs, .. } => out.push(exprs),
        BlockFst::Sequential(children) | BlockFst::FirstMatching(children) => {
            children.iter_mut().for_each(|c| collect_leaves(c, out))
        }
        BlockFst::Propagate(inner) => collect_leaves(inner, out),
    }
}

/// An exact position gate for the VM tier, borrowed from a compiled rule:
/// "could expression `ei` of leaf `leaf` claim at this position?" The VM
/// consults it before attempting `transform` (where it spends most of its
/// time) and re-derives everything itself at the rare hits, so binding and
/// splicing (syllable structure included) stay on VM code paths. Leaves
/// are the `Exprs` nodes of the block tree in DFS order, which matches the
/// VM's traversal of the same `BlockIr` (filter wrappers are flattened on
/// both sides). The gate must never answer `false` where the VM would
/// claim; `true` merely costs a probe.
pub struct FstGate<'g> {
    leaves: Vec<&'g mut Vec<ExprFst>>,
    ends: Vec<(usize, u32)>,
}

impl FstGate<'_> {
    /// `pos` is word-local (the VM locates the word in its phrase; nothing
    /// in the compiled subset crosses word gaps, so word-local evaluation
    /// is exact). `word` is the word as the VM matches it, already
    /// filtered when the leaf is under a filter.
    pub fn may_claim(
        &mut self,
        leaf: usize,
        ei: usize,
        decls: &Declarations,
        segments: &SegmentInterner,
        word: &Word,
        pos: usize,
    ) -> bool {
        let ctx = Ctx { segments, decls };
        self.leaves[leaf][ei]
            .claim_at(&ctx, word, pos, &mut self.ends)
            .is_some()
    }

    pub fn leaf_count(&self) -> usize {
        self.leaves.len()
    }
}

/// The FST-tier image of [`BlockIr`]. Filters are pushed down to the
/// expression leaves at compile time (nested filters compose), so the
/// runtime never threads a filter stack.
#[derive(Debug, Clone)]
enum BlockFst {
    Exprs {
        mode: MatchMode,
        exprs: Vec<ExprFst>,
        /// In-scope filters; a segment must pass every one.
        filters: Vec<FilterFst>,
    },
    /// `Then:` apply every child in order.
    Sequential(Vec<BlockFst>),
    /// `Else:` apply only the first child that matches.
    FirstMatching(Vec<BlockFst>),
    /// Re-apply until fixpoint (same budgets as the VM).
    Propagate(Box<BlockFst>),
}

/// A compiled rule filter: a per-segment predicate (validation restricts
/// filters to single-segment shapes), memoized per interned segment;
/// lexurgy tests each segment in isolation, as a one-segment word.
#[derive(Debug, Clone)]
struct FilterFst {
    machine: Machine,
    cache: FastMap<SegmentId, bool>,
}

impl FilterFst {
    fn test(&mut self, decls: &Declarations, segments: &SegmentInterner, seg: SegmentId) -> bool {
        if let Some(&hit) = self.cache.get(&seg) {
            return hit;
        }
        let ctx = Ctx { segments, decls };
        let lone = Word::simple(vec![seg]);
        let hit = self.machine.matches(&ctx, &lone, 0, Dir::Fwd);
        self.cache.insert(seg, hit);
        hit
    }
}

/// Try to compile a rule for this tier; `Err` names the first construct
/// that keeps it on the VM. `segments` must be the interner of the
/// universe the rule runs in.
pub fn compile_rule(
    rule: &RuleIr,
    decls: &Declarations,
    segments: &SegmentInterner,
) -> Result<RuleFst, VmReason> {
    let floating = decls
        .diacritics
        .iter()
        .enumerate()
        .filter(|(_, d)| d.floating && d.level == Level::Segment)
        .fold(0, |mask, (i, _)| mask | (1 << i));
    let (body, splice) = match compile_block(&rule.body, floating, decls, segments, &[], true) {
        Ok(body) => (body, Ok(())),
        // The emit side didn't compile; retry match-only (the match
        // machinery still gates the VM). A match-side failure fails again
        // and keeps the rule on the plain VM.
        Err(why) => (
            compile_block(&rule.body, floating, decls, segments, &[], false)?,
            Err(why),
        ),
    };
    Ok(RuleFst {
        body,
        floating,
        propagates: contains_propagate(&rule.body),
        splice,
    })
}

fn contains_propagate(block: &BlockIr) -> bool {
    match block {
        BlockIr::Exprs { .. } => false,
        BlockIr::Sequential(children) | BlockIr::FirstMatching(children) => {
            children.iter().any(contains_propagate)
        }
        BlockIr::Propagate(_) => true,
        BlockIr::Filter { inner, .. } => contains_propagate(inner),
    }
}

fn compile_block(
    block: &BlockIr,
    floating: DiacriticMask,
    decls: &Declarations,
    segments: &SegmentInterner,
    filters: &[&Pattern],
    pieces: bool,
) -> Result<BlockFst, VmReason> {
    match block {
        BlockIr::Exprs { mode, exprs } => Ok(BlockFst::Exprs {
            mode: *mode,
            exprs: exprs
                .iter()
                .map(|expr| {
                    compile_expr(expr, floating, decls, segments, !filters.is_empty(), pieces)
                })
                .collect::<Result<Vec<_>, _>>()?,
            filters: filters
                .iter()
                .map(|filter| {
                    Ok(FilterFst {
                        machine: Machine::new(compile_nfa(filter, false)?),
                        cache: FastMap::default(),
                    })
                })
                .collect::<Result<Vec<_>, VmReason>>()?,
        }),
        BlockIr::Sequential(children) => Ok(BlockFst::Sequential(
            children
                .iter()
                .map(|child| compile_block(child, floating, decls, segments, filters, pieces))
                .collect::<Result<Vec<_>, _>>()?,
        )),
        BlockIr::FirstMatching(children) => Ok(BlockFst::FirstMatching(
            children
                .iter()
                .map(|child| compile_block(child, floating, decls, segments, filters, pieces))
                .collect::<Result<Vec<_>, _>>()?,
        )),
        BlockIr::Propagate(inner) => Ok(BlockFst::Propagate(Box::new(compile_block(
            inner, floating, decls, segments, filters, pieces,
        )?))),
        BlockIr::Filter { filter, inner } => {
            let mut nested = filters.to_vec();
            nested.push(filter);
            compile_block(inner, floating, decls, segments, &nested, pieces)
        }
    }
}

fn compile_expr(
    expr: &ExprIr,
    floating: DiacriticMask,
    decls: &Declarations,
    segments: &SegmentInterner,
    filtered: bool,
    pieces: bool,
) -> Result<ExprFst, VmReason> {
    let (instances, from_varied) = expand_variables(expr, decls)?;
    let gate = if from_varied && instances.len() > 1 {
        let union = Pattern::Alt(instances.iter().map(|inst| inst.from.clone()).collect());
        Some(Machine::new(compile_nfa(&union, false)?))
    } else {
        None
    };
    Ok(ExprFst {
        instances: instances
            .iter()
            .map(|inst| compile_instance(inst, floating, segments, filtered, pieces))
            .collect::<Result<Vec<_>, _>>()?,
        from_varied,
        gate,
    })
}

fn compile_instance(
    expr: &ExprIr,
    floating: DiacriticMask,
    segments: &SegmentInterner,
    filtered: bool,
    pieces: bool,
) -> Result<InstFst, VmReason> {
    let mut builder = Builder { states: Vec::new() };
    let matched = builder.push(State::Match)?;
    let (start, variants) =
        builder.build_paired(&expr.from, &expr.to, matched, 1, &Conjuncts::default())?;
    let nfa = Nfa {
        states: builder.states,
        start,
        scratch: ClaimScratch::default(),
    };
    let emits = if pieces {
        variants
            .iter()
            .map(|(pat, emit)| piece_list(pat, emit, floating, segments, filtered))
            .collect::<Result<Vec<_>, _>>()?
    } else {
        Vec::new()
    };
    Ok(InstFst {
        from: Machine::new(nfa),
        emits,
        conditions: expr
            .condition
            .iter()
            .map(compile_env)
            .collect::<Result<_, _>>()?,
        exclusions: expr
            .exclusion
            .iter()
            .map(compile_env)
            .collect::<Result<_, _>>()?,
    })
}

// feature-variable expansion

/// A run of adjacent seg-map rules fused into one pass. The composed
/// transducer is materialized lazily: each distinct input segment is pushed
/// through every stage once, and the segment → segments result is cached:
/// composition over the realized alphabet, under a cache budget.
#[derive(Debug, Clone)]
pub struct FusedRun {
    pub universe: Universe,
    pub rules: Vec<usize>,
    /// Index of the first step after the run (the steps it covers are only
    /// `Rule`s and `StripBreaks`, which are no-ops on unsyllabified words).
    pub end_step: usize,
    cache: FastMap<SegmentId, Vec<SegmentId>>,
}

impl FusedRun {
    pub fn apply(
        &mut self,
        fsts: &mut [Option<RuleFst>],
        decls: &Declarations,
        segments: &mut SegmentInterner,
        word: &Word,
    ) -> Result<Word, RunError> {
        debug_assert!(!word.is_syllabified());
        let mut out: Vec<SegmentId> = Vec::with_capacity(word.segs.len());
        for &seg in &word.segs {
            if let Some(mapped) = self.cache.get(&seg) {
                out.extend_from_slice(mapped);
                continue;
            }
            let mut current = vec![seg];
            for &rule in &self.rules {
                let fst = fsts[rule].as_mut().expect("fused rule lost its FST");
                let mut next = Vec::with_capacity(current.len());
                for &s in &current {
                    next.extend(fst.map_segment(decls, segments, s)?);
                }
                current = next;
            }
            out.extend_from_slice(&current);
            if self.cache.len() < MAX_FUSED_CACHE {
                self.cache.insert(seg, current);
            }
        }
        Ok(Word::simple(out))
    }
}

/// Find maximal runs of adjacent fusable rule steps. `StripBreaks` between
/// them is absorbed (a no-op on the unsyllabified words this path handles);
/// universes can't mix, and runs of fewer than two rules aren't worth a
/// detour.
pub fn fuse_steps(
    steps: &[Step],
    fsts: &[Option<RuleFst>],
) -> (Vec<FusedRun>, HashMap<usize, usize>) {
    let mut fused = Vec::new();
    let mut fused_at = HashMap::new();
    let mut i = 0;
    while i < steps.len() {
        let mut rules = Vec::new();
        let mut universe = None;
        let mut j = i;
        while j < steps.len() {
            match steps[j] {
                Step::StripBreaks => j += 1,
                Step::Rule { rule, universe: u }
                    if universe.is_none_or(|x| x == u)
                        && fsts[rule].as_ref().is_some_and(|f| f.seg_map_eligible()) =>
                {
                    universe = Some(u);
                    rules.push(rule);
                    j += 1;
                }
                _ => break,
            }
        }
        if rules.len() >= 2 {
            fused_at.insert(i, fused.len());
            fused.push(FusedRun {
                universe: universe.expect("run has rules"),
                rules,
                end_step: j,
                cache: FastMap::default(),
            });
            i = j;
        } else {
            i += 1;
        }
    }
    (fused, fused_at)
}
