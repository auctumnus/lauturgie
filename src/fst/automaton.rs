// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) Autumn <auctumnus@pm.me>

//! NFA/DFA matching core: predicate-transition automata, lazy
//! determinization over predicate minterms, and the syllable-aware
//! claim machinery shared by every compiled rule.
// (moved from the former monolithic fst.rs; see `super` for shared types)
use super::*;

#[derive(Debug, Clone)]
pub(super) enum State {
    /// Consume one segment matching the predicate.
    Pred(Leaf, StateId),
    /// Consume one segment *not* matching the predicate (`!x`).
    NegPred(Leaf, StateId),
    /// Zero-width position check.
    Anchor(Anchor, StateId),
    /// `<syl>`: consume exactly one whole syllable, a direct port of the
    /// VM's `syllable_from`/`syllable_back_from`, because the bounding-
    /// break cases (a break at position 0 or at the word end, as deletions
    /// and `.` emits leave behind mid-rule) make the span zero-width in
    /// direction-asymmetric ways no boundary-anchor automaton reproduces.
    /// Carries the syllable conjuncts of `<syl>&[...]`.
    Syl(Vec<(BitTest, bool)>, StateId),
    /// Priority fork: explore `first` before `second`.
    Split(StateId, StateId),
    /// Zero-width: add to the path's emit-variant index (paired
    /// alternations).
    Tag(u32, StateId),
    Match,
}

/// A transition predicate: a segment test plus any syllable-level
/// conjuncts (from `SegTest::SylMatrix`, or fused from an eligible
/// intersection like `@vowel&[+heavy]`). Syllable conjuncts test the
/// feature word of the syllable containing the segment; on an
/// unsyllabified word that's the all-defaults word, exactly like the VM's
/// `test_syl_matrix` on an empty modifier list.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct Leaf {
    seg: SegTest,
    /// Extra segment-level conjuncts (`{e, i}&[+nas]`): `(test, negated)`.
    segc: Vec<(SegTest, bool)>,
    /// `(test, negated)`: the containing syllable's word must (not) match.
    syl: Vec<(BitTest, bool)>,
}

impl Leaf {
    fn plain(seg: SegTest) -> Leaf {
        Leaf {
            seg,
            segc: Vec::new(),
            syl: Vec::new(),
        }
    }

    fn add(&mut self, conjuncts: &Conjuncts) {
        self.segc.extend(conjuncts.segc.iter().cloned());
        self.syl.extend(conjuncts.syl.iter().cloned());
    }
}

/// The verifier parts of an intersection, reduced to per-segment
/// conjuncts. Sound only where the span is forced to width 1 (or, for
/// `<syl>&[...]`, to one whole syllable): the VM verifies the *span*, and a
/// width-1 verifier can only reach the span end when the span is width 1,
/// and a syllable matrix is length-hinted to spans inside one syllable,
/// which a single segment or a whole syllable always satisfies.
#[derive(Debug, Default, Clone)]
pub(super) struct Conjuncts {
    segc: Vec<(SegTest, bool)>,
    syl: Vec<(BitTest, bool)>,
}

impl Conjuncts {
    fn is_empty(&self) -> bool {
        self.segc.is_empty() && self.syl.is_empty()
    }
}

/// Parse the rest parts of `first & rest…`, or say why the intersection
/// stays on the VM.
pub(super) fn intersect_conjuncts(rest: &[Pattern]) -> Result<Conjuncts, VmReason> {
    let mut out = Conjuncts::default();
    for part in rest {
        let (test, negated) = match part {
            Pattern::Not(inner) => match inner.as_ref() {
                Pattern::Test(test) => (test, true),
                _ => return Err("intersection (&)"),
            },
            Pattern::Test(test) => (test, false),
            Pattern::Text(t) if t.tests.len() == 1 && t.syl_mask == 0 => (&t.tests[0], false),
            // A mixed-level matrix (`[-long unstressed]`) lowers to a
            // nested intersection of its two halves; on a width-1 span
            // every part is itself a conjunct.
            Pattern::Intersect(sub) => {
                let inner = intersect_conjuncts(sub)?;
                out.segc.extend(inner.segc);
                out.syl.extend(inner.syl);
                continue;
            }
            _ => return Err("intersection (&)"),
        };
        match test {
            SegTest::SylMatrix(m) => {
                if !m.vars.is_empty() {
                    return Err("feature variables");
                }
                out.syl.push((m.syl.clone(), negated));
            }
            SegTest::Matrix(m) if !m.vars.is_empty() => return Err("feature variables"),
            other => out.segc.push((other.clone(), negated)),
        }
    }
    Ok(out)
}

#[derive(Debug, Clone, Copy)]
pub(super) enum Anchor {
    WordStart,
    WordEnd,
    /// `.`: a syllable boundary here (word edges count, unsyllabified
    /// words have none).
    Bound,
    /// `!.`
    NoBound,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Dir {
    Fwd,
    Bwd,
}

/// Per-`Nfa` reusable scratch for `claim`'s backtracking walk. Cleared (not
/// reallocated) at the start of each claim; `claim` is hit per position per
/// rule, so the three per-call `Vec`s it used to allocate were a measurable
/// slice of the FST tier's allocation churn. The whole apply path is
/// `&mut self` (each rayon worker owns its clone of the changer, see
/// `apply_all`), so these buffers can live inline without a cell; their
/// contents never carry meaning across calls.
#[derive(Debug, Clone, Default)]
pub(super) struct ClaimScratch {
    visited: Vec<bool>,
    seen_end: Vec<bool>,
    stack: Vec<(StateId, usize, u32)>,
}

/// A pattern compiled to a predicate-transition NFA.
#[derive(Debug, Clone)]
pub(super) struct Nfa {
    pub(super) states: Vec<State>,
    pub(super) start: StateId,
    pub(super) scratch: ClaimScratch,
}

pub(super) struct Builder {
    pub(super) states: Vec<State>,
}

impl Builder {
    pub(super) fn push(&mut self, state: State) -> Result<StateId, VmReason> {
        if self.states.len() >= MAX_STATES {
            return Err("the pattern outgrew the state budget");
        }
        self.states.push(state);
        Ok(self.states.len() - 1)
    }

    /// Emit one predicate state matching `test` fused with `conjuncts`
    /// (empty conjuncts for an ordinary, non-intersection leaf).
    fn leaf_pred(
        &mut self,
        test: &SegTest,
        conjuncts: &Conjuncts,
        next: StateId,
    ) -> Result<StateId, VmReason> {
        let mut leaf = eligible_test(test)?;
        leaf.add(conjuncts);
        self.push(State::Pred(leaf, next))
    }

    /// Compile `pattern` so that matching continues at `next`. Returns the
    /// entry state, or the reason the pattern is outside the eligible
    /// subset. `mirror` reverses concatenation order (for lookbehind, which
    /// the runtime walks leftward).
    pub(super) fn build(
        &mut self,
        pattern: &Pattern,
        next: StateId,
        mirror: bool,
    ) -> Result<StateId, VmReason> {
        match pattern {
            Pattern::Empty => Ok(next),
            Pattern::Test(test) => self.leaf_pred(test, &Conjuncts::default(), next),
            Pattern::Not(inner) => match inner.as_ref() {
                Pattern::Test(test) => self.push(State::NegPred(eligible_test(test)?, next)),
                _ => Err("negation of a non-leaf pattern"),
            },
            Pattern::Text(text) => {
                if text.syl_mask != 0 {
                    return Err("syllable features on match text");
                }
                let mut entry = next;
                let order: Vec<&SegTest> = if mirror {
                    text.tests.iter().collect()
                } else {
                    text.tests.iter().rev().collect()
                };
                for test in order {
                    entry = self.push(State::Pred(eligible_test(test)?, entry))?;
                }
                Ok(entry)
            }
            Pattern::SyllableBoundary => self.push(State::Anchor(Anchor::Bound, next)),
            Pattern::NoBoundary => self.push(State::Anchor(Anchor::NoBound, next)),
            Pattern::AnySyllable => self.any_syllable(&Conjuncts::default(), next),
            // `a&b`: the first part sets the span; the FST encodes the
            // shapes whose verifiers reduce to per-segment conjuncts: a
            // single-segment first part (possibly an alternative list, as
            // classes lower to), or `<syl>&[syllable matrix]`. Anything
            // else stays on the VM.
            Pattern::Intersect(parts) => {
                let (first, rest) = parts.split_first().ok_or("an empty intersection")?;
                let conjuncts = intersect_conjuncts(rest)?;
                self.build_leafy(first, &conjuncts, next)
            }
            Pattern::Seq(parts) => {
                let mut entry = next;
                // build back to front so each part chains into the next
                let order: Vec<&Pattern> = if mirror {
                    parts.iter().collect()
                } else {
                    parts.iter().rev().collect()
                };
                for part in order {
                    entry = self.build(part, entry, mirror)?;
                }
                Ok(entry)
            }
            Pattern::Alt(parts) => {
                let mut entries = Vec::with_capacity(parts.len());
                for part in parts {
                    entries.push(self.build(part, next, mirror)?);
                }
                self.fold_alternatives(entries)
            }
            Pattern::Repeat { inner, min, max } => {
                match max {
                    Some(max) => {
                        if max < min {
                            return Err("an empty repeat range");
                        }
                        // X^min (X?)^(max-min), optionals greedy
                        let mut entry = next;
                        for _ in 0..(max - min) {
                            let body = self.build(inner, entry, mirror)?;
                            entry = self.push(State::Split(body, entry))?;
                        }
                        for _ in 0..*min {
                            entry = self.build(inner, entry, mirror)?;
                        }
                        Ok(entry)
                    }
                    None => {
                        // X^min X*, the star greedy (loop branch first)
                        let split = self.push(State::Split(0, 0))?;
                        let body = self.build(inner, split, mirror)?;
                        self.states[split] = State::Split(body, next);
                        let mut entry = split;
                        for _ in 0..*min {
                            entry = self.build(inner, entry, mirror)?;
                        }
                        Ok(entry)
                    }
                }
            }
            Pattern::WordStart => self.push(State::Anchor(Anchor::WordStart, next)),
            Pattern::WordEnd => self.push(State::Anchor(Anchor::WordEnd, next)),
            // Everything else stays on the VM tier.
            Pattern::Capture { .. } | Pattern::CaptureRef { .. } => Err("captures"),
            Pattern::Look { .. } => Err("a local environment"),
            Pattern::NotAhead(_) => Err("a negated lookaround"),
            Pattern::BetweenWords => Err("the between-words boundary ($$)"),
            Pattern::WordBoundary => Err("an unresolved word boundary"),
        }
    }

    /// `<syl>`: exactly one whole syllable; a boundary, the syllable's
    /// segments (no interior boundary), a boundary. Mirrors the VM's
    /// `syllable_from`/`syllable_back_from` (which require the start
    /// position to be a syllable edge and return the next edge), and is
    /// direction-symmetric, so the same states serve mirrored machines.
    /// `conjuncts` are syllable tests fused from `<syl>&[...]` (the span
    /// is one whole syllable, so the VM's length-hinted claim always
    /// passes and only the feature test remains). Segment-level conjuncts
    /// would have to match the whole span (only possible for a one-segment
    /// syllable); those stay on the VM.
    fn any_syllable(&mut self, conjuncts: &Conjuncts, next: StateId) -> Result<StateId, VmReason> {
        if !conjuncts.segc.is_empty() {
            return Err("intersection (&)");
        }
        self.push(State::Syl(conjuncts.syl.clone(), next))
    }

    /// Compile a width-1 (per the conjunct soundness argument) pattern
    /// with intersection conjuncts fused into its leaves.
    fn build_leafy(
        &mut self,
        pattern: &Pattern,
        conjuncts: &Conjuncts,
        next: StateId,
    ) -> Result<StateId, VmReason> {
        match pattern {
            Pattern::Test(test) => self.leaf_pred(test, conjuncts, next),
            Pattern::Text(text) if text.tests.len() == 1 && text.syl_mask == 0 => {
                self.leaf_pred(&text.tests[0], conjuncts, next)
            }
            Pattern::Alt(parts) => {
                let mut entries = Vec::with_capacity(parts.len());
                for part in parts {
                    entries.push(self.build_leafy(part, conjuncts, next)?);
                }
                self.fold_alternatives(entries)
            }
            Pattern::AnySyllable => self.any_syllable(conjuncts, next),
            // nested `a&b&c`: merge
            Pattern::Intersect(parts) => {
                let (first, rest) = parts.split_first().ok_or("an empty intersection")?;
                let mut merged = intersect_conjuncts(rest)?;
                merged.segc.extend(conjuncts.segc.iter().cloned());
                merged.syl.extend(conjuncts.syl.iter().cloned());
                self.build_leafy(first, &merged, next)
            }
            _ => Err("intersection (&)"),
        }
    }

    /// Fold entry states into a right-leaning split chain, preserving
    /// written order as fork priority.
    fn fold_alternatives(&mut self, mut entries: Vec<StateId>) -> Result<StateId, VmReason> {
        let mut entry = entries.pop().ok_or("an empty alternative list")?;
        while let Some(prev) = entries.pop() {
            entry = self.push(State::Split(prev, entry))?;
        }
        Ok(entry)
    }

    /// Compile a `from`/`to` pair, resolving paired alternations: returns
    /// the entry state and the variants in tag order, each a *resolved*
    /// (pattern, emit) pair with the alternation choices substituted in
    /// (the pattern half is what piece analysis pairs the emit against).
    /// Tag deltas written into the NFA are scaled by `scale` so nested
    /// pairing sites compose into one mixed-radix variant index (matching
    /// the VM's lexicographic enumeration of options through
    /// `transform`/`transform_sequence`).
    pub(super) fn build_paired(
        &mut self,
        pattern: &Pattern,
        emit: &Emit,
        next: StateId,
        scale: u32,
        conj: &Conjuncts,
    ) -> Result<(StateId, Vec<(Pattern, Emit)>), VmReason> {
        if !emit_contains_alt(emit) {
            let entry = if conj.is_empty() {
                self.build(pattern, next, false)?
            } else {
                self.build_leafy(pattern, conj, next)?
            };
            return Ok((entry, vec![(pattern.clone(), emit.clone())]));
        }
        match (pattern, emit) {
            // `{p, t} => {b, d}` pairs positionally; a size mismatch
            // distributes the whole emitter to each branch (lexurgy's
            // fallback in `transformerToAlternatives`). A non-Alt emitter
            // with alternatives further in distributes the same way.
            (Pattern::Alt(parts), _) => {
                let pairs: Vec<(&Pattern, &Emit)> = match emit {
                    Emit::Alt(alts) if alts.len() == parts.len() => {
                        parts.iter().zip(alts.iter()).collect()
                    }
                    _ => parts.iter().map(|p| (p, emit)).collect(),
                };
                let mut entries = Vec::with_capacity(pairs.len());
                let mut variants: Vec<(Pattern, Emit)> = Vec::new();
                for (part, part_emit) in pairs {
                    let (entry, table) = self.build_paired(part, part_emit, next, scale, conj)?;
                    self.accumulate_variant(entry, table, scale, &mut entries, &mut variants)?;
                }
                Ok((self.fold_alternatives(entries)?, variants))
            }
            // `{e, i}&[+nas] => {ɛ, e}`: the VM pairs the *first* part
            // with the emitter (`transform_lifting`) and verifies the rest
            // per option; the verifiers become leaf conjuncts here, so they
            // ride along into the first part's branches.
            (Pattern::Intersect(iparts), _) => {
                let (first, rest) = iparts.split_first().ok_or("an empty intersection")?;
                let mut merged = intersect_conjuncts(rest)?;
                merged.segc.extend(conj.segc.iter().cloned());
                merged.syl.extend(conj.syl.iter().cloned());
                self.build_paired(first, emit, next, scale, &merged)
            }
            // `a b => x {y, z}`: element-wise pairing
            // (`SequenceMatcher.transformerTo`).
            (Pattern::Seq(parts), Emit::Seq(emits)) if parts.len() == emits.len() => {
                if !conj.is_empty() {
                    return Err("intersection (&)");
                }
                self.build_seq_paired(parts, emits, next, scale)
            }
            // `a {e, o} => {e e, o o}`: sequences distribute into adjacent
            // alternative lists when the lengths line up
            // (`SequenceMatcher.transformerToAlternatives`): branch i takes
            // the i-th member of every direct alternative.
            (Pattern::Seq(parts), Emit::Alt(alts)) => {
                if !conj.is_empty() {
                    return Err("intersection (&)");
                }
                let n = alts.len();
                if !parts.iter().all(|p| match p {
                    Pattern::Alt(xs) => xs.len() == n,
                    _ => true,
                }) {
                    return Err(UNPAIRABLE_ALT);
                }
                let seqs: Option<Vec<&Vec<Emit>>> = alts
                    .iter()
                    .map(|a| match a {
                        Emit::Seq(es) if es.len() == parts.len() => Some(es),
                        _ => None,
                    })
                    .collect();
                let seqs = seqs.ok_or(UNPAIRABLE_ALT)?;
                let mut entries = Vec::with_capacity(n);
                let mut variants: Vec<(Pattern, Emit)> = Vec::new();
                for (i, es) in seqs.iter().enumerate() {
                    let seq_i: Vec<Pattern> = parts
                        .iter()
                        .map(|p| match p {
                            Pattern::Alt(xs) => xs[i].clone(),
                            other => other.clone(),
                        })
                        .collect();
                    let (entry, table) = self.build_seq_paired(&seq_i, es, next, scale)?;
                    self.accumulate_variant(entry, table, scale, &mut entries, &mut variants)?;
                }
                Ok((self.fold_alternatives(entries)?, variants))
            }
            // Repeaters pair per repetition (unbounded variants), leaves
            // can't pair with alternative emitters at all: VM tier.
            _ => Err(UNPAIRABLE_ALT),
        }
    }

    /// Append one already-built paired branch to an alternation: tag its
    /// entry with the running variant offset (so the composed mixed-radix
    /// index stays distinct across branches) and fold in its variant table.
    fn accumulate_variant(
        &mut self,
        mut entry: StateId,
        table: Vec<(Pattern, Emit)>,
        scale: u32,
        entries: &mut Vec<StateId>,
        variants: &mut Vec<(Pattern, Emit)>,
    ) -> Result<(), VmReason> {
        let base = variants.len() as u32;
        if base > 0 {
            let delta = base.checked_mul(scale).ok_or(TOO_MANY_VARIANTS)?;
            entry = self.push(State::Tag(delta, entry))?;
        }
        entries.push(entry);
        variants.extend(table);
        if variants.len() > MAX_VARIANTS {
            return Err(TOO_MANY_VARIANTS);
        }
        Ok(())
    }

    /// Pair sequence elements, building back to front. Element j's tag
    /// deltas scale by the product of the later elements' variant counts,
    /// making the leftmost pairing site the most significant digit,
    /// exactly the VM's option order.
    fn build_seq_paired(
        &mut self,
        parts: &[Pattern],
        emits: &[Emit],
        next: StateId,
        scale: u32,
    ) -> Result<(StateId, Vec<(Pattern, Emit)>), VmReason> {
        let mut entry = next;
        let mut tables_rev: Vec<Vec<(Pattern, Emit)>> = Vec::with_capacity(parts.len());
        let mut running: u32 = 1;
        for (part, emit) in parts.iter().zip(emits.iter()).rev() {
            let inner_scale = scale.checked_mul(running).ok_or(TOO_MANY_VARIANTS)?;
            let (e, table) =
                self.build_paired(part, emit, entry, inner_scale, &Conjuncts::default())?;
            entry = e;
            running = running
                .checked_mul(table.len() as u32)
                .ok_or(TOO_MANY_VARIANTS)?;
            if running as usize > MAX_VARIANTS {
                return Err(TOO_MANY_VARIANTS);
            }
            tables_rev.push(table);
        }
        let tables: Vec<Vec<(Pattern, Emit)>> = tables_rev.into_iter().rev().collect();
        let mut variants = Vec::with_capacity(running as usize);
        for combo in 0..running as usize {
            let mut rem = combo;
            let mut pats_out = Vec::with_capacity(tables.len());
            let mut parts_out = Vec::with_capacity(tables.len());
            for table in tables.iter().rev() {
                let (pat, emit) = &table[rem % table.len()];
                pats_out.push(pat.clone());
                parts_out.push(emit.clone());
                rem /= table.len();
            }
            pats_out.reverse();
            parts_out.reverse();
            variants.push((Pattern::Seq(pats_out), Emit::Seq(parts_out)));
        }
        Ok((entry, variants))
    }
}

pub(super) const TOO_MANY_VARIANTS: VmReason = "too many paired-alternation variants";
pub(super) const UNPAIRABLE_ALT: VmReason = "an alternation pairing the FST can't encode";
pub(super) const UNPAIRABLE: VmReason = "a from/to pairing the FST can't encode";

pub(super) fn emit_contains_alt(emit: &Emit) -> bool {
    match emit {
        Emit::Alt(_) => true,
        Emit::Seq(parts) => parts.iter().any(emit_contains_alt),
        _ => false,
    }
}

/// A test is eligible when it has no feature variables (variables need
/// bindings). A pure syllable matrix consumes one segment and tests its
/// syllable's word (the VM's `test_syl_matrix`); like the VM's `test_seg`,
/// the syllable half of a segment-level `Matrix` is ignored.
pub(super) fn eligible_test(test: &SegTest) -> Result<Leaf, VmReason> {
    match test {
        SegTest::Exact(_) | SegTest::Literal { .. } | SegTest::Any => Ok(Leaf::plain(test.clone())),
        SegTest::Matrix(m) if m.vars.is_empty() => Ok(Leaf::plain(test.clone())),
        SegTest::Matrix(_) => Err("feature variables"),
        SegTest::SylMatrix(m) if m.vars.is_empty() => Ok(Leaf {
            seg: SegTest::Any,
            segc: Vec::new(),
            syl: vec![(m.syl.clone(), false)],
        }),
        SegTest::SylMatrix(_) => Err("feature variables"),
    }
}

pub(super) fn compile_nfa(pattern: &Pattern, mirror: bool) -> Result<Nfa, VmReason> {
    let mut builder = Builder { states: Vec::new() };
    let matched = builder.push(State::Match)?;
    let start = builder.build(pattern, matched, mirror)?;
    Ok(Nfa {
        states: builder.states,
        start,
        scratch: ClaimScratch::default(),
    })
}

// Consumption analysis bits (for the seg-map tier): which total segment
// counts a path from a state to Match can consume.
pub(super) const CONS_ZERO: u8 = 1;
pub(super) const CONS_ONE: u8 = 2;
pub(super) const CONS_MANY: u8 = 4;
pub(super) const CONS_ANCHOR: u8 = 8;

impl Nfa {
    /// All match ends from `pos`, in the VM's preference order (prioritized
    /// DFS; duplicate ends collapse to their first, highest-priority visit,
    /// fixing the emit variant tag the VM's first option would carry).
    pub(super) fn claim(
        &mut self,
        ctx: &Ctx<'_>,
        word: &Word,
        pos: usize,
        dir: Dir,
        ends: &mut Vec<(usize, u32)>,
    ) {
        ends.clear();
        let positions = word.len() + 1;
        // Split the borrows: the walk reads `states`/`start` while it
        // mutates the scratch buffers.
        let Nfa {
            states,
            start,
            scratch,
        } = self;
        let ClaimScratch {
            visited,
            seen_end,
            stack,
        } = scratch;
        visited.clear();
        visited.resize(states.len() * positions, false);
        seen_end.clear();
        seen_end.resize(positions, false);
        // explicit stack, children pushed in reverse priority order
        stack.clear();
        stack.push((*start, pos, 0));
        while let Some((state, at, tag)) = stack.pop() {
            let slot = state * positions + at;
            if visited[slot] {
                continue;
            }
            visited[slot] = true;
            match &states[state] {
                State::Match => {
                    if !seen_end[at] {
                        seen_end[at] = true;
                        ends.push((at, tag));
                    }
                }
                State::Split(first, second) => {
                    stack.push((*second, at, tag));
                    stack.push((*first, at, tag));
                }
                State::Tag(delta, next) => {
                    stack.push((*next, at, tag + delta));
                }
                State::Pred(test, next) => {
                    if let Some(index) = index_at(word, at, dir) {
                        if ctx.test(test, word, index) {
                            stack.push((*next, step(at, dir), tag));
                        }
                    }
                }
                State::NegPred(test, next) => {
                    if let Some(index) = index_at(word, at, dir) {
                        if !ctx.test(test, word, index) {
                            stack.push((*next, step(at, dir), tag));
                        }
                    }
                }
                State::Anchor(anchor, next) => {
                    let holds = match anchor {
                        Anchor::WordStart => at == 0,
                        Anchor::WordEnd => at == word.len(),
                        Anchor::Bound => word.has_boundary_at(at),
                        Anchor::NoBound => !word.has_boundary_at(at),
                    };
                    if holds {
                        stack.push((*next, at, tag));
                    }
                }
                State::Syl(conjuncts, next) => {
                    if let Some(end) = syllable_span(word, at, dir) {
                        // The VM's hinted claim reads the syllable at the
                        // span's matcher-side index (`lo` forward,
                        // `hi - 1` saturating backward).
                        let ok = conjuncts.is_empty() || {
                            let (lo, hi) = if at <= end { (at, end) } else { (end, at) };
                            let index = match dir {
                                Dir::Fwd => lo,
                                Dir::Bwd => hi.saturating_sub(1),
                            };
                            let value = FeatureWord(syl_word_at(ctx.decls, word, index));
                            conjuncts
                                .iter()
                                .all(|(test, negated)| test.matches(value) != *negated)
                        };
                        if ok {
                            stack.push((*next, end, tag));
                        }
                    }
                }
            }
        }
    }

    /// Segment counts consumable on start→Match paths, capped at "2+"
    /// (`CONS_MANY`), with `CONS_ANCHOR` set if any anchor is reachable.
    /// Cycles count as many. `CONS_ONE` alone means: every match consumes
    /// exactly one segment, position-independently: a segment map.
    pub(super) fn consumption(&self) -> u8 {
        fn go(nfa: &Nfa, s: StateId, memo: &mut [Option<u8>], busy: &mut [bool]) -> u8 {
            if let Some(m) = memo[s] {
                return m;
            }
            if busy[s] {
                return CONS_MANY;
            }
            busy[s] = true;
            let r = match &nfa.states[s] {
                State::Match => CONS_ZERO,
                // position-dependent width: never a segment map
                State::Syl(_, n) => CONS_ANCHOR | CONS_MANY | go(nfa, *n, memo, busy),
                State::Tag(_, n) => go(nfa, *n, memo, busy),
                State::Split(a, b) => go(nfa, *a, memo, busy) | go(nfa, *b, memo, busy),
                State::Anchor(_, n) => CONS_ANCHOR | go(nfa, *n, memo, busy),
                State::Pred(_, n) | State::NegPred(_, n) => {
                    let inner = go(nfa, *n, memo, busy);
                    let mut out = inner & CONS_ANCHOR;
                    if inner & CONS_ZERO != 0 {
                        out |= CONS_ONE;
                    }
                    if inner & (CONS_ONE | CONS_MANY) != 0 {
                        out |= CONS_MANY;
                    }
                    out
                }
            };
            busy[s] = false;
            memo[s] = Some(r);
            r
        }
        let mut memo = vec![None; self.states.len()];
        let mut busy = vec![false; self.states.len()];
        go(self, self.start, &mut memo, &mut busy)
    }
}

pub(super) fn index_at(word: &Word, pos: usize, dir: Dir) -> Option<usize> {
    match dir {
        Dir::Fwd => (pos < word.len()).then_some(pos),
        Dir::Bwd => pos.checked_sub(1),
    }
}

pub(super) fn step(pos: usize, dir: Dir) -> usize {
    match dir {
        Dir::Fwd => pos + 1,
        Dir::Bwd => pos - 1,
    }
}

/// One whole syllable from `pos`: a port of the VM's `syllable_from`
/// (forward) and `syllable_back_from` (backward), bounding breaks and all.
pub(super) fn syllable_span(word: &Word, pos: usize, dir: Dir) -> Option<usize> {
    if !word.is_syllabified() || word.is_empty() {
        return None;
    }
    let breaks = word.syllable_breaks();
    match dir {
        Dir::Fwd => {
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
        Dir::Bwd => {
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
    }
}

/// Read-only context for predicate evaluation. `decls` feed syllable
/// conjuncts (the syllable feature word is derived from modifier lists).
pub(super) struct Ctx<'a> {
    pub(super) segments: &'a SegmentInterner,
    pub(super) decls: &'a Declarations,
}

/// The feature word of the syllable containing `index`: all defaults on
/// an unsyllabified word, like the VM reading an empty modifier list.
pub(super) fn syl_word_at(decls: &Declarations, word: &Word, index: usize) -> u128 {
    crate::compiler::segments::syl_mods_bits(decls, word.mods_at(index)).1
}

impl<'a> Ctx<'a> {
    fn test(&self, leaf: &Leaf, word: &Word, index: usize) -> bool {
        let seg = word.segs[index];
        if !self.test_seg(&leaf.seg, seg) {
            return false;
        }
        if !leaf
            .segc
            .iter()
            .all(|(test, negated)| self.test_seg(test, seg) != *negated)
        {
            return false;
        }
        if leaf.syl.is_empty() {
            return true;
        }
        let value = FeatureWord(syl_word_at(self.decls, word, index));
        leaf.syl
            .iter()
            .all(|(test, negated)| test.matches(value) != *negated)
    }

    fn test_seg(&self, test: &SegTest, seg: SegmentId) -> bool {
        match test {
            SegTest::Any => true,
            SegTest::Exact(id) => seg == *id,
            SegTest::Literal {
                id,
                core,
                required,
                allowed,
            } => {
                if seg == *id {
                    return true;
                }
                let data = self.segments.get(seg);
                data.core == *core
                    && data.diacritics & required == *required
                    && data.diacritics & !allowed == 0
            }
            SegTest::Matrix(m) => m.seg.matches(self.segments.get(seg).features),
            SegTest::SylMatrix(_) => unreachable!("syllable matrices become leaf conjuncts"),
        }
    }
}

// determinization over predicate minterms

/// Budget marker: the DFA refused to grow; the caller falls back to the
/// NFA and stops asking.
pub(super) struct DfaBudget;

pub(super) const FLAG_START: u8 = 1;
pub(super) const FLAG_END: u8 = 2;
/// Only set when the machine has boundary anchors (`bounds_matter`), so
/// segment-only machines don't fragment their transition tables on
/// syllabified words.
pub(super) const FLAG_BOUND: u8 = 4;

pub(super) fn flags_at(at: usize, word: &Word, bounds_matter: bool) -> u8 {
    (at == 0) as u8 * FLAG_START
        + (at == word.len()) as u8 * FLAG_END
        + (bounds_matter && word.has_boundary_at(at)) as u8 * FLAG_BOUND
}

/// A lazily built DFA over minterm classes. States are priority-ordered
/// tuples of NFA states truncated at the first Match; truncation encodes
/// leftmost-first preference: once a match is live, all lower-priority
/// threads die, so the *last* matching position seen on a scan is exactly
/// the end the prioritized backtracking NFA would pick first.
#[derive(Debug, Clone)]
pub(super) struct Dfa {
    /// Distinct predicates; a segment's minterm class is the bitset of
    /// predicates it satisfies, discovered per segment and memoized.
    preds: Vec<Leaf>,
    /// NFA state id → index into `preds` (consuming states only).
    pred_of: Vec<u32>,
    states: Vec<DfaState>,
    index: HashMap<(Box<[StateId]>, bool), u32>,
    /// Start state per anchor-flag context.
    starts: [Option<u32>; 8],
    /// Minterm classes by segment; the hot key for ordinary machines.
    classes: FastMap<SegmentId, u64>,
    /// When any predicate carries syllable conjuncts (`syl_classify`), the
    /// class also depends on the position's syllable feature word, so a
    /// wider key replaces the segment-only cache.
    syl_classes: FastMap<(SegmentId, u128), u64>,
    /// Any predicate tests the containing syllable's features.
    syl_classify: bool,
    /// Any boundary anchor: position flags include `FLAG_BOUND`.
    bounds_matter: bool,
    /// Determinization unavailable (variant tags, too many predicates) or
    /// abandoned (budget); the machine uses its NFA directly.
    pub(super) dead: bool,
}

#[derive(Debug, Clone)]
pub(super) struct DfaState {
    tuple: Box<[StateId]>,
    matching: bool,
    /// (minterm class, anchor flags of the position stepped to) → state.
    trans: FastMap<(u64, u8), u32>,
}

impl Dfa {
    pub(super) fn new(nfa: &Nfa) -> Dfa {
        let mut preds: Vec<Leaf> = Vec::new();
        let mut pred_of = vec![0u32; nfa.states.len()];
        let mut dead = false;
        let mut bounds_matter = false;
        for (i, state) in nfa.states.iter().enumerate() {
            match state {
                State::Pred(test, _) | State::NegPred(test, _) => {
                    let at = match preds.iter().position(|p| p == test) {
                        Some(at) => at,
                        None => {
                            preds.push(test.clone());
                            preds.len() - 1
                        }
                    };
                    pred_of[i] = at as u32;
                }
                // Tags are path-dependent; a subset DFA would lose them.
                // `<syl>` spans are read off the break list, not stepped.
                State::Tag(..) | State::Syl(..) => dead = true,
                State::Anchor(Anchor::Bound | Anchor::NoBound, _) => bounds_matter = true,
                _ => {}
            }
        }
        if preds.len() > MAX_DFA_PREDS {
            dead = true;
        }
        let syl_classify = preds.iter().any(|p| !p.syl.is_empty());
        Dfa {
            preds,
            pred_of,
            states: Vec::new(),
            index: HashMap::new(),
            starts: [None; 8],
            classes: FastMap::default(),
            syl_classes: FastMap::default(),
            syl_classify,
            bounds_matter,
            dead,
        }
    }

    fn classify(&mut self, ctx: &Ctx<'_>, word: &Word, index: usize) -> u64 {
        let seg = word.segs[index];
        if !self.syl_classify {
            if let Some(&class) = self.classes.get(&seg) {
                return class;
            }
            let bits = self.compute_class(ctx, word, index);
            self.classes.insert(seg, bits);
            return bits;
        }
        let syl = syl_word_at(ctx.decls, word, index);
        if let Some(&class) = self.syl_classes.get(&(seg, syl)) {
            return class;
        }
        let bits = self.compute_class(ctx, word, index);
        self.syl_classes.insert((seg, syl), bits);
        bits
    }

    fn compute_class(&self, ctx: &Ctx<'_>, word: &Word, index: usize) -> u64 {
        let mut bits = 0u64;
        for (i, pred) in self.preds.iter().enumerate() {
            if ctx.test(pred, word, index) {
                bits |= 1 << i;
            }
        }
        bits
    }

    /// Priority-ordered epsilon closure with cut: walk forks in preference
    /// order collecting consuming states; a Match makes the state matching
    /// and discards everything of lower priority.
    fn closure(&mut self, nfa: &Nfa, frontier: &[StateId], flags: u8) -> Result<u32, DfaBudget> {
        let mut tuple: Vec<StateId> = Vec::new();
        let mut matching = false;
        let mut seen = vec![false; nfa.states.len()];
        let mut stack: Vec<StateId> = frontier.iter().rev().copied().collect();
        while let Some(s) = stack.pop() {
            if seen[s] {
                continue;
            }
            seen[s] = true;
            match &nfa.states[s] {
                State::Match => {
                    matching = true;
                    break;
                }
                State::Split(first, second) => {
                    stack.push(*second);
                    stack.push(*first);
                }
                State::Anchor(anchor, next) => {
                    let holds = match anchor {
                        Anchor::WordStart => flags & FLAG_START != 0,
                        Anchor::WordEnd => flags & FLAG_END != 0,
                        Anchor::Bound => flags & FLAG_BOUND != 0,
                        Anchor::NoBound => flags & FLAG_BOUND == 0,
                    };
                    if holds {
                        stack.push(*next);
                    }
                }
                State::Pred(..) | State::NegPred(..) => tuple.push(s),
                State::Tag(..) | State::Syl(..) => {
                    unreachable!("undeterminizable NFA determinized")
                }
            }
        }
        let key = (tuple.into_boxed_slice(), matching);
        if let Some(&id) = self.index.get(&key) {
            return Ok(id);
        }
        if self.states.len() >= MAX_DFA_STATES {
            return Err(DfaBudget);
        }
        let id = self.states.len() as u32;
        self.states.push(DfaState {
            tuple: key.0.clone(),
            matching,
            trans: FastMap::default(),
        });
        self.index.insert(key, id);
        Ok(id)
    }

    fn start_state(&mut self, nfa: &Nfa, flags: u8) -> Result<u32, DfaBudget> {
        if let Some(id) = self.starts[flags as usize] {
            return Ok(id);
        }
        let id = self.closure(nfa, &[nfa.start], flags)?;
        self.starts[flags as usize] = Some(id);
        Ok(id)
    }

    fn next_state(&mut self, nfa: &Nfa, sid: u32, class: u64, flags: u8) -> Result<u32, DfaBudget> {
        if let Some(&n) = self.states[sid as usize].trans.get(&(class, flags)) {
            return Ok(n);
        }
        let tuple = self.states[sid as usize].tuple.clone();
        let mut moved: Vec<StateId> = Vec::with_capacity(tuple.len());
        for &s in tuple.iter() {
            let hit = class >> self.pred_of[s] & 1 != 0;
            match &nfa.states[s] {
                State::Pred(_, next) => {
                    if hit {
                        moved.push(*next);
                    }
                }
                State::NegPred(_, next) => {
                    if !hit {
                        moved.push(*next);
                    }
                }
                _ => unreachable!("non-consuming state in DFA tuple"),
            }
        }
        let n = self.closure(nfa, &moved, flags)?;
        self.states[sid as usize].trans.insert((class, flags), n);
        Ok(n)
    }

    /// Scan from `pos`, returning the first (existence) or last (preferred,
    /// thanks to truncation) matching position.
    pub(super) fn scan(
        &mut self,
        nfa: &Nfa,
        ctx: &Ctx<'_>,
        word: &Word,
        pos: usize,
        dir: Dir,
        first: bool,
    ) -> Result<Option<usize>, DfaBudget> {
        let mut at = pos;
        let mut sid = self.start_state(nfa, flags_at(at, word, self.bounds_matter))?;
        let mut best = None;
        loop {
            let (matching, empty) = {
                let state = &self.states[sid as usize];
                (state.matching, state.tuple.is_empty())
            };
            if matching {
                best = Some(at);
                if first {
                    return Ok(best);
                }
            }
            if empty {
                return Ok(best);
            }
            let Some(index) = index_at(word, at, dir) else {
                return Ok(best);
            };
            let class = self.classify(ctx, word, index);
            at = step(at, dir);
            sid = self.next_state(nfa, sid, class, flags_at(at, word, self.bounds_matter))?;
        }
    }
}

/// A pattern machine: the NFA plus its lazily determinized DFA.
#[derive(Debug, Clone)]
pub(super) struct Machine {
    pub(super) nfa: Nfa,
    pub(super) dfa: Dfa,
}

impl Machine {
    pub(super) fn new(nfa: Nfa) -> Machine {
        let dfa = Dfa::new(&nfa);
        Machine { nfa, dfa }
    }

    /// Does any match exist? (Environment checks need no more; the
    /// eligible subset has no bindings.)
    pub(super) fn matches(&mut self, ctx: &Ctx<'_>, word: &Word, pos: usize, dir: Dir) -> bool {
        if !self.dfa.dead {
            match self.dfa.scan(&self.nfa, ctx, word, pos, dir, true) {
                Ok(end) => return end.is_some(),
                Err(DfaBudget) => self.dfa.dead = true,
            }
        }
        let mut ends = Vec::new();
        self.nfa.claim(ctx, word, pos, dir, &mut ends);
        !ends.is_empty()
    }

    /// The end the VM's backtracking would try first (untagged machines
    /// only; used when no environment can reject it).
    pub(super) fn preferred_end(
        &mut self,
        ctx: &Ctx<'_>,
        word: &Word,
        pos: usize,
        dir: Dir,
    ) -> Option<usize> {
        if !self.dfa.dead {
            match self.dfa.scan(&self.nfa, ctx, word, pos, dir, false) {
                Ok(end) => return end,
                Err(DfaBudget) => self.dfa.dead = true,
            }
        }
        let mut ends = Vec::new();
        self.nfa.claim(ctx, word, pos, dir, &mut ends);
        ends.first().map(|&(end, _)| end)
    }
}

#[derive(Debug, Clone)]
pub(super) struct CompiledEnv {
    before: Option<Machine>,
    after: Option<Machine>,
}

impl CompiledEnv {
    pub(super) fn check(&mut self, ctx: &Ctx<'_>, word: &Word, lo: usize, hi: usize) -> bool {
        if let Some(before) = &mut self.before {
            if !before.matches(ctx, word, lo, Dir::Bwd) {
                return false;
            }
        }
        if let Some(after) = &mut self.after {
            if !after.matches(ctx, word, hi, Dir::Fwd) {
                return false;
            }
        }
        true
    }
}

pub(super) fn compile_env(env: &EnvIr) -> Result<CompiledEnv, VmReason> {
    Ok(CompiledEnv {
        before: match &env.before {
            None => None,
            Some(p) => Some(Machine::new(compile_nfa(p, true)?)),
        },
        after: match &env.after {
            None => None,
            Some(p) => Some(Machine::new(compile_nfa(p, false)?)),
        },
    })
}
