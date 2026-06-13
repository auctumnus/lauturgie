// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) Autumn <auctumnus@pm.me>

//! The runtime apply path: claiming matches, resolving emit pieces,
//! and splicing results back into the word (filtered and unfiltered).
// (moved from the former monolithic fst.rs; see `super` for shared types)
use super::*;

/// A claim: `(start, end, expression index, instance index, emit variant
/// tag)`.
pub(super) type Claim = (usize, usize, usize, usize, u32);

impl RuleFst {
    /// Apply the rule to an *unsyllabified* word. `None` means it didn't
    /// match (the word is unchanged either way; matched-ness drives
    /// `Else:` and `propagate`).
    pub fn apply(
        &mut self,
        decls: &Declarations,
        segments: &mut SegmentInterner,
        word: &Word,
    ) -> Result<Option<Word>, RunError> {
        debug_assert!(!word.is_syllabified());
        debug_assert!(self.splice.is_ok(), "match-only rules gate the VM instead");
        run_block(&mut self.body, self.floating, decls, segments, word)
    }

    /// Is this rule a context-free single-segment map? (A plain
    /// simultaneous expression list whose every expression consumes
    /// exactly one segment wherever it stands, with no environments, so
    /// the rule is a pure function on segments and composes with its
    /// neighbors.)
    pub fn seg_map_eligible(&self) -> bool {
        if self.splice.is_err() {
            return false;
        }
        match &self.body {
            BlockFst::Exprs {
                mode: MatchMode::Simultaneous,
                exprs,
                filters,
            } if filters.is_empty() => exprs.iter().all(|expr| {
                expr.instances.iter().all(|inst| {
                    inst.conditions.is_empty()
                        && inst.exclusions.is_empty()
                        && inst.from.nfa.consumption() == CONS_ONE
                })
            }),
            _ => false,
        }
    }

    /// Map one segment through a seg-map-eligible rule.
    pub(super) fn map_segment(
        &mut self,
        decls: &Declarations,
        segments: &mut SegmentInterner,
        seg: SegmentId,
    ) -> Result<Vec<SegmentId>, RunError> {
        let floating = self.floating;
        let BlockFst::Exprs { exprs, .. } = &mut self.body else {
            unreachable!("seg-map rule with block structure");
        };
        let lone = Word::simple(vec![seg]);
        let mut ends = Vec::new();
        for expr in exprs {
            for inst in &mut expr.instances {
                {
                    let ctx = Ctx {
                        segments: &*segments,
                        decls,
                    };
                    inst.from.nfa.claim(&ctx, &lone, 0, Dir::Fwd, &mut ends);
                }
                if let Some(&(end, tag)) = ends.first() {
                    debug_assert_eq!(end, 1);
                    let mut out = Vec::new();
                    run_pieces(
                        &inst.emits[tag as usize],
                        floating,
                        decls,
                        segments,
                        &lone.segs,
                        &mut out,
                    )?;
                    return Ok(out);
                }
            }
        }
        Ok(vec![seg])
    }
}

/// Run a block; `None` means it didn't match. The FST-tier mirror of
/// `vm::Executor::run_block` over single unsyllabified words (no `$$`
/// emits in the subset, so words can't split).
pub(super) fn run_block(
    block: &mut BlockFst,
    floating: DiacriticMask,
    decls: &Declarations,
    segments: &mut SegmentInterner,
    word: &Word,
) -> Result<Option<Word>, RunError> {
    match block {
        BlockFst::Exprs {
            mode,
            exprs,
            filters,
        } => match mode {
            MatchMode::Simultaneous => {
                apply_simultaneous(exprs, filters, floating, decls, segments, word)
            }
            MatchMode::Ltr => {
                apply_directional(exprs, filters, floating, decls, segments, word, false)
            }
            MatchMode::Rtl => {
                apply_directional(exprs, filters, floating, decls, segments, word, true)
            }
        },
        BlockFst::Sequential(children) => {
            let mut matched = false;
            let mut current = word.clone();
            for child in children {
                if let Some(next) = run_block(child, floating, decls, segments, &current)? {
                    matched = true;
                    current = next;
                }
            }
            Ok(matched.then_some(current))
        }
        BlockFst::FirstMatching(children) => {
            for child in children {
                if let Some(result) = run_block(child, floating, decls, segments, word)? {
                    return Ok(Some(result));
                }
            }
            Ok(None)
        }
        BlockFst::Propagate(inner) => {
            let mut current = word.clone();
            let mut seen: HashSet<Vec<SegmentId>> = HashSet::new();
            seen.insert(current.segs.clone());
            let mut first_step = true;
            let mut work = 0usize;
            let work_limit = crate::vm::propagate_work_limit(word.len());
            let limit = crate::vm::growth_limit(word.len());
            loop {
                match run_block(inner, floating, decls, segments, &current)? {
                    None => return Ok(if first_step { None } else { Some(current) }),
                    Some(next) => {
                        if next.segs == current.segs {
                            return Ok(Some(next));
                        }
                        work += next.len().max(1).pow(3);
                        if !seen.insert(next.segs.clone())
                            || work > work_limit
                            || next.len() > limit
                        {
                            return Err(RunError::DivergingPropagation);
                        }
                        current = next;
                        first_step = false;
                    }
                }
            }
        }
    }
}

/// Build the filtered word: the segments passing every filter, plus the
/// map from filtered index to real index.
pub(super) fn filter_word(
    filters: &mut [FilterFst],
    decls: &Declarations,
    segments: &SegmentInterner,
    word: &Word,
) -> (Word, Vec<usize>) {
    let mut segs = Vec::new();
    let mut map = Vec::new();
    for (i, &seg) in word.segs.iter().enumerate() {
        if filters.iter_mut().all(|f| f.test(decls, segments, seg)) {
            segs.push(seg);
            map.push(i);
        }
    }
    (Word::simple(segs), map)
}

/// All claims of every expression over `word`, in precedence order
/// (expression-major, scanning left to right and restarting one past each
/// match's start; lexurgy's `claimAll`).
pub(super) fn claim_all(
    exprs: &mut [ExprFst],
    decls: &Declarations,
    segments: &SegmentInterner,
    word: &Word,
) -> Vec<Claim> {
    let mut all: Vec<Claim> = Vec::new();
    let mut ends = Vec::new();
    let ctx = Ctx { segments, decls };
    for (ei, expr) in exprs.iter_mut().enumerate() {
        let mut from_pos = 0;
        while from_pos <= word.len() {
            let mut matched = None;
            for pos in from_pos..=word.len() {
                if let Some((end, inst, tag)) = expr.claim_at(&ctx, word, pos, &mut ends) {
                    matched = Some((pos, end, inst, tag));
                    break;
                }
            }
            // Resume one position past the match, or stop when a full pass
            // found nothing.
            match matched {
                Some((pos, end, inst, tag)) => {
                    all.push((pos, end, ei, inst, tag));
                    from_pos = pos + 1;
                }
                None => break,
            }
        }
    }
    all
}

/// Keep claims in precedence order, dropping any that overlap an earlier
/// claim (half-open ranges: zero-width claims never overlap). The result
/// stays in *precedence* order; the stable by-start sort happens at
/// splice time, so claims sharing a start keep their precedence (the VM's
/// `applyTransformations` ordering, which decides who wins a collision).
pub(super) fn keep_nonoverlapping(all: Vec<Claim>) -> Vec<Claim> {
    let mut claimed: Vec<(usize, usize)> = Vec::new();
    let mut kept: Vec<Claim> = Vec::new();
    for claim in all {
        let overlaps = claimed.iter().any(|&(s, e)| claim.0 < e && s < claim.1);
        if !overlaps {
            claimed.push((claim.0, claim.1));
            kept.push(claim);
        }
    }
    kept
}

/// The VM's simultaneous path: `claimAll` per expression, overlap
/// filtering by precedence, then splicing (through the filter map when
/// filters are in scope).
pub(super) fn apply_simultaneous(
    exprs: &mut [ExprFst],
    filters: &mut [FilterFst],
    floating: DiacriticMask,
    decls: &Declarations,
    segments: &mut SegmentInterner,
    word: &Word,
) -> Result<Option<Word>, RunError> {
    if filters.is_empty() {
        let mut kept = keep_nonoverlapping(claim_all(exprs, decls, segments, word));
        if kept.is_empty() {
            return Ok(None);
        }
        kept.sort_by_key(|claim| claim.0);
        return Ok(Some(splice(exprs, floating, decls, segments, word, &kept)?));
    }
    let (fword, fmap) = filter_word(filters, decls, segments, word);
    let kept = keep_nonoverlapping(claim_all(exprs, decls, segments, &fword));
    if kept.is_empty() {
        return Ok(None);
    }
    let mut subs: Vec<(usize, SubEmit)> = Vec::new();
    for claim in &kept {
        unfilter_claim(exprs, &fword, &fmap, claim, &mut subs)?;
    }
    Ok(Some(splice_subs(floating, decls, segments, word, subs)?))
}

/// One unfiltered sub: a claim on a single real segment, with its output
/// left *unevaluated*; the VM only binds transformations that actually
/// apply, so a colliding sub whose matrix doesn't spell a segment must
/// drop silently instead of erroring.
pub(super) enum SubEmit {
    Verbatim(Vec<SegmentId>),
    Matrix {
        update: MatrixUpdate,
        orig: Vec<SegmentId>,
        insert_when_empty: bool,
    },
    Transfer {
        text: Vec<SegmentId>,
        mode: TransferMode,
        orig: Vec<SegmentId>,
    },
}

/// Translate one filtered-coordinate claim into real-segment subs;
/// lexurgy's `unfilterTransformations`: every elemental piece claims
/// exactly one real segment (the one its match *started* on), no matter
/// how many filtered segments it covered or how many segments it emits.
pub(super) fn unfilter_claim(
    exprs: &[ExprFst],
    fword: &Word,
    fmap: &[usize],
    claim: &Claim,
    subs: &mut Vec<(usize, SubEmit)>,
) -> Result<(), RunError> {
    let &(start, end, ei, inst, tag) = claim;
    let list = &exprs[ei].instances[inst].emits[tag as usize];
    let matched = &fword.segs[start..end];
    let var_width = matched.len().saturating_sub(list.fixed);
    let mut cursor = 0usize;
    for piece in &list.pieces {
        let w = match piece.width {
            Some(w) => w,
            None => var_width,
        };
        let orig = &matched[cursor..cursor + w];
        // Lexurgy indexes `filterMap[sub.start]` unguarded; a zero-width
        // match at the end of the filtered word throws (a per-word error).
        let Some(&real) = fmap.get(start + cursor) else {
            return Err(RunError::Word(
                "filtered rule matched past the last filtered segment".to_string(),
            ));
        };
        match &piece.emit {
            // One sub per unit, all claiming the piece's first segment:
            // surplus units collide and drop at splice time.
            PieceEmit::Verbatim(units) => {
                for unit in units {
                    subs.push((real, SubEmit::Verbatim(unit.clone())));
                }
            }
            PieceEmit::Matrix {
                update,
                insert_when_empty,
            } => {
                subs.push((
                    real,
                    SubEmit::Matrix {
                        update: update.clone(),
                        orig: orig.to_vec(),
                        insert_when_empty: *insert_when_empty,
                    },
                ));
            }
            PieceEmit::Transfer { text, mode } => {
                subs.push((
                    real,
                    SubEmit::Transfer {
                        text: text.clone(),
                        mode: mode.clone(),
                        orig: orig.to_vec(),
                    },
                ));
            }
        }
        cursor += w;
    }
    Ok(())
}

/// Splice unfiltered subs into the real word: each sub replaces one real
/// segment; colliding subs drop via the cursor skip (lexurgy's
/// `applyTransformations`), *before* their outputs are evaluated.
pub(super) fn splice_subs(
    floating: DiacriticMask,
    decls: &Declarations,
    segments: &mut SegmentInterner,
    word: &Word,
    mut subs: Vec<(usize, SubEmit)>,
) -> Result<Word, RunError> {
    subs.sort_by_key(|sub| sub.0);
    let mut out: Vec<SegmentId> = Vec::with_capacity(word.len());
    let mut cursor = 0usize;
    for (start, emit) in subs {
        if cursor > start {
            continue;
        }
        out.extend_from_slice(&word.segs[cursor..start]);
        match emit {
            SubEmit::Verbatim(segs) => out.extend_from_slice(&segs),
            SubEmit::Matrix {
                update,
                orig,
                insert_when_empty,
            } => {
                if orig.is_empty() {
                    if insert_when_empty {
                        let value = FeatureWord(update.seg_bits & update.seg_mask);
                        if let Some(id) = segments.render_features(decls, value) {
                            out.push(id);
                        }
                    }
                } else {
                    for &seg in &orig {
                        out.push(apply_matrix(decls, segments, &update, seg)?);
                    }
                }
            }
            SubEmit::Transfer { text, mode, orig } => {
                transfer(decls, segments, floating, &text, &mode, &orig, &mut out);
            }
        }
        cursor = start + 1;
    }
    out.extend_from_slice(&word.segs[cursor..]);
    Ok(Word::simple(out))
}

/// The VM's `ltr`/`rtl` path: walk the *evolving* word one index at a
/// time, applying the first expression that claims at exactly that
/// index. Directional rules always count as matched. With filters in
/// scope, the word is refiltered every step and the index converted to
/// filtered coordinates (positions that fail the filter can't match).
pub(super) fn apply_directional(
    exprs: &mut [ExprFst],
    filters: &mut [FilterFst],
    floating: DiacriticMask,
    decls: &Declarations,
    segments: &mut SegmentInterner,
    word: &Word,
    rtl: bool,
) -> Result<Option<Word>, RunError> {
    // Same divergence budget as the VM tier: a scan that grows the
    // word faster than the cursor moves never terminates.
    let limit = crate::vm::growth_limit(word.len());
    let mut current = word.clone();
    let mut ends = Vec::new();
    if rtl {
        let mut index = current.len();
        loop {
            current = once_at(
                exprs, filters, floating, decls, segments, current, index, &mut ends,
            )?;
            if current.len() > limit {
                return Err(RunError::DivergingScan);
            }
            if index == 0 {
                break;
            }
            index -= 1;
        }
    } else {
        let mut index = 0;
        while index <= current.len() {
            current = once_at(
                exprs, filters, floating, decls, segments, current, index, &mut ends,
            )?;
            if current.len() > limit {
                return Err(RunError::DivergingScan);
            }
            index += 1;
        }
    }
    Ok(Some(current))
}

/// `transform_once_at`: first expression claiming at `index` applies.
#[allow(clippy::too_many_arguments)]
pub(super) fn once_at(
    exprs: &mut [ExprFst],
    filters: &mut [FilterFst],
    floating: DiacriticMask,
    decls: &Declarations,
    segments: &mut SegmentInterner,
    word: Word,
    index: usize,
    ends: &mut Vec<(usize, u32)>,
) -> Result<Word, RunError> {
    // An earlier replacement may have shrunk the word (rtl keeps its
    // original indices); past-the-end positions can't match.
    if index > word.len() {
        return Ok(word);
    }
    if filters.is_empty() {
        let mut found: Option<Claim> = None;
        {
            let ctx = Ctx {
                segments: &*segments,
                decls,
            };
            for (ei, expr) in exprs.iter_mut().enumerate() {
                if let Some((end, inst, tag)) = expr.claim_at(&ctx, &word, index, ends) {
                    found = Some((index, end, ei, inst, tag));
                    break;
                }
            }
        }
        return match found {
            Some(claim) => splice(exprs, floating, decls, segments, &word, &[claim]),
            None => Ok(word),
        };
    }
    let (fword, fmap) = filter_word(filters, decls, segments, &word);
    // A position whose segment fails the filter (or the end of the word)
    // can't match (lexurgy's `indexOf` returns -1 there).
    let Ok(findex) = fmap.binary_search(&index) else {
        return Ok(word);
    };
    let mut found: Option<Claim> = None;
    {
        let ctx = Ctx {
            segments: &*segments,
            decls,
        };
        for (ei, expr) in exprs.iter_mut().enumerate() {
            if let Some((end, inst, tag)) = expr.claim_at(&ctx, &fword, findex, ends) {
                found = Some((findex, end, ei, inst, tag));
                break;
            }
        }
    }
    match found {
        Some(claim) => {
            let mut subs = Vec::new();
            unfilter_claim(exprs, &fword, &fmap, &claim, &mut subs)?;
            splice_subs(floating, decls, segments, &word, subs)
        }
        None => Ok(word),
    }
}

/// Build the result word from non-overlapping claims sorted by start.
pub(super) fn splice(
    exprs: &[ExprFst],
    floating: DiacriticMask,
    decls: &Declarations,
    segments: &mut SegmentInterner,
    word: &Word,
    claims: &[Claim],
) -> Result<Word, RunError> {
    let mut out: Vec<SegmentId> = Vec::with_capacity(word.len());
    let mut cursor = 0usize;
    for &(start, end, ei, inst, tag) in claims {
        if cursor > start {
            continue;
        }
        out.extend_from_slice(&word.segs[cursor..start]);
        run_pieces(
            &exprs[ei].instances[inst].emits[tag as usize],
            floating,
            decls,
            segments,
            &word.segs[start..end],
            &mut out,
        )?;
        cursor = end;
    }
    out.extend_from_slice(&word.segs[cursor..]);
    Ok(Word::simple(out))
}

/// Emit a claim's pieces over the matched slice. Each piece reads its own
/// sub-span: constant widths are compile-time, the one variable piece gets
/// the remainder.
pub(super) fn run_pieces(
    list: &PieceList,
    floating: DiacriticMask,
    decls: &Declarations,
    segments: &mut SegmentInterner,
    matched: &[SegmentId],
    out: &mut Vec<SegmentId>,
) -> Result<(), RunError> {
    let var_width = matched.len().saturating_sub(list.fixed);
    let mut cursor = 0usize;
    for (i, piece) in list.pieces.iter().enumerate() {
        let w = match piece.width {
            Some(w) => w,
            None => var_width,
        };
        debug_assert!(Some(i) == list.variable || piece.width.is_some());
        let orig = &matched[cursor..cursor + w];
        match &piece.emit {
            PieceEmit::Verbatim(units) => {
                for unit in units {
                    out.extend_from_slice(unit);
                }
            }
            PieceEmit::Matrix {
                update,
                insert_when_empty,
            } => {
                if orig.is_empty() {
                    // zero-width: emit the matrix alone if it spells a
                    // segment, otherwise nothing (only for a literal `*`;
                    // a zero-width repeat emits nothing)
                    if *insert_when_empty {
                        let value = FeatureWord(update.seg_bits & update.seg_mask);
                        if let Some(id) = segments.render_features(decls, value) {
                            out.push(id);
                        }
                    }
                } else {
                    for &seg in orig {
                        out.push(apply_matrix(decls, segments, update, seg)?);
                    }
                }
            }
            PieceEmit::Transfer { text, mode } => {
                transfer(decls, segments, floating, text, mode, orig, out)
            }
        }
        cursor += w;
    }
    Ok(())
}

/// Emit conditional text with floating diacritics collected from the
/// matched segments.
pub(super) fn transfer(
    decls: &Declarations,
    segments: &mut SegmentInterner,
    floating: DiacriticMask,
    text: &[SegmentId],
    mode: &TransferMode,
    orig: &[SegmentId],
    out: &mut Vec<SegmentId>,
) {
    match mode {
        TransferMode::Pairwise(excls) => {
            for ((&t, &o), &x) in text.iter().zip(orig).zip(excls) {
                out.push(with_floats(decls, segments, floating, t, o, x));
            }
        }
        TransferMode::Collapse(excls) => {
            let mut id = text[0];
            for (&o, &x) in orig.iter().zip(excls) {
                id = with_floats(decls, segments, floating, id, o, x);
            }
            out.push(id);
        }
        TransferMode::Expand(excl) => {
            for &t in text {
                out.push(with_floats(decls, segments, floating, t, orig[0], *excl));
            }
        }
    }
}

/// Copy `source`'s floating diacritics (minus `excluded`) onto `target`;
/// the FST-tier image of `vm::Executor::with_floats`.
pub(super) fn with_floats(
    decls: &Declarations,
    segments: &mut SegmentInterner,
    floating: DiacriticMask,
    target: SegmentId,
    source: SegmentId,
    excluded: DiacriticMask,
) -> SegmentId {
    let extra = segments.get(source).diacritics & floating & !excluded;
    if extra == 0 {
        target
    } else {
        segments.with_extra_diacritics(decls, target, extra)
    }
}

/// Rewrite one segment's features by a matrix update.
pub(super) fn apply_matrix(
    decls: &Declarations,
    segments: &mut SegmentInterner,
    update: &MatrixUpdate,
    seg: SegmentId,
) -> Result<SegmentId, RunError> {
    let data = segments.get(seg);
    let (features, core) = (data.features, data.core);
    let value = FeatureWord(features.0 & !update.seg_mask | update.seg_bits);
    if segments.core_is_featural(decls, core) {
        segments.render_features(decls, value)
    } else {
        segments.render_cored(decls, core, value)
    }
    .ok_or(RunError::InvalidMatrix)
}

// cross-rule composition
