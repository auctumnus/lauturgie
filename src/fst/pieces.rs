// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) Autumn <auctumnus@pm.me>

//! Emit "pieces": decomposing each `from => to` pairing into the
//! constant- and variable-width spans the apply path splices.
// (moved from the former monolithic fst.rs; see `super` for shared types)
use super::*;

/// One unit of output, paired with the span of original segments its
/// pattern counterpart matched: the compile-time image of the VM's
/// per-element `Transformation`s (`transform_sequence` chains them; here
/// the chain is flattened into a list whose spans are recovered from the
/// claim's total width).
#[derive(Debug, Clone, PartialEq)]
pub(super) struct Piece {
    /// Constant match width of the pattern counterpart, or `None` for the
    /// (single) piece whose width is `claim width − fixed widths`.
    pub(super) width: Option<usize>,
    pub(super) emit: PieceEmit,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) enum PieceEmit {
    /// Verbatim segments (independent emission, exact text, deletion, or
    /// conditional text with nothing to transfer), as one unit per
    /// flattened sequence-emitter element (lexurgy's
    /// `IndependentSequenceTransformer.resultBits`). Unfiltered, the units
    /// just concatenate; a filtered rule unfilters each unit *separately*
    /// (all claiming the same real segment, so surplus units drop; the
    /// `@cv => z u` semantics).
    Verbatim(Vec<Vec<SegmentId>>),
    /// Rewrite each matched segment's features. On an empty span, the VM
    /// renders the matrix alone iff the pattern counterpart is literally
    /// `*` (`bind_matrix` with an empty original); a zero-width repeat
    /// emits nothing.
    Matrix {
        update: MatrixUpdate,
        insert_when_empty: bool,
    },
    /// Conditional text under floating diacritics: the emitted segments
    /// collect floating diacritics from the matched ones; lexurgy's
    /// `SymbolEmitter.result` (`vm::Executor::text_result`), with the mode
    /// resolved at compile time from the pattern/text lengths.
    Transfer {
        text: Vec<SegmentId>,
        mode: TransferMode,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub(super) enum TransferMode {
    /// Equal lengths: `out[i]` takes floats from `matched[i]`, excluding
    /// the pattern segment's own diacritics.
    Pairwise(Vec<DiacriticMask>),
    /// One emitted segment collects floats from every matched segment.
    Collapse(Vec<DiacriticMask>),
    /// Every emitted segment takes floats from the one matched segment.
    Expand(DiacriticMask),
}

/// The emit side of one expression variant, as a piece list.
#[derive(Debug, Clone)]
pub(super) struct PieceList {
    pub(super) pieces: Vec<Piece>,
    /// Index of the piece whose width is recovered from the claim span.
    pub(super) variable: Option<usize>,
    /// Sum of the constant piece widths.
    pub(super) fixed: usize,
}

/// Match width of a pattern when it's the same for every match.
pub(super) fn const_width(pattern: &Pattern) -> Option<usize> {
    match pattern {
        Pattern::Empty | Pattern::WordStart | Pattern::WordEnd => Some(0),
        Pattern::Test(_) | Pattern::Not(_) => Some(1),
        Pattern::Text(t) => Some(t.tests.len()),
        Pattern::Seq(parts) => parts.iter().map(const_width).sum(),
        Pattern::Alt(parts) => {
            let widths: Option<Vec<usize>> = parts.iter().map(const_width).collect();
            let widths = widths?;
            widths.windows(2).all(|w| w[0] == w[1]).then(|| widths[0])
        }
        Pattern::Repeat { inner, min, max } => (*max == Some(*min))
            .then(|| const_width(inner).map(|w| w * *min as usize))
            .flatten(),
        // the span is the first part's (the VM lifts intersections)
        Pattern::Intersect(parts) => parts.first().and_then(const_width),
        _ => None,
    }
}

/// Does the pattern have a `*` leaf (a zero-width piece a matrix emitter
/// would render alone)?
pub(super) fn contains_empty(pattern: &Pattern) -> bool {
    match pattern {
        Pattern::Empty => true,
        Pattern::Seq(parts) | Pattern::Alt(parts) => parts.iter().any(contains_empty),
        Pattern::Repeat { inner, .. } => contains_empty(inner),
        _ => false,
    }
}

/// Resolve one variant's `from => to` pairing into pieces: the
/// compile-time mirror of the VM's `transform` dispatch over the eligible
/// subset (captures, lookarounds, intersections, and syllable elements are
/// already gone: `Builder::build` rejected them).
pub(super) fn piece_list(
    pattern: &Pattern,
    emit: &Emit,
    floating: DiacriticMask,
    segments: &SegmentInterner,
    filtered: bool,
) -> Result<PieceList, VmReason> {
    let mut pieces = Vec::new();
    collect_pieces(pattern, emit, floating, segments, filtered, &mut pieces)?;
    let mut variable = None;
    let mut fixed = 0usize;
    for (i, piece) in pieces.iter().enumerate() {
        match piece.width {
            Some(w) => fixed += w,
            None => {
                if variable.replace(i).is_some() {
                    return Err("two variable-width pieces in one pairing");
                }
            }
        }
    }
    Ok(PieceList {
        pieces,
        variable,
        fixed,
    })
}

pub(super) fn collect_pieces(
    pattern: &Pattern,
    emit: &Emit,
    floating: DiacriticMask,
    segments: &SegmentInterner,
    filtered: bool,
    out: &mut Vec<Piece>,
) -> Result<(), VmReason> {
    // The VM lifts intersections before pairing (`transform_lifting`
    // re-dispatches on the first part); verifier parts don't shape the
    // result.
    if let Pattern::Intersect(parts) = pattern {
        let first = parts.first().ok_or("an empty intersection")?;
        return collect_pieces(first, emit, floating, segments, filtered, out);
    }
    match emit {
        // Alternations were resolved into variants by `build_paired`.
        Emit::Alt(_) => Err(UNPAIRABLE_ALT),
        Emit::Seq(parts) => {
            let independent = parts.iter().all(crate::vm::emit_is_independent);
            // Only repeaters prefer independent *sequence* emitters
            // (`a* => x y` emits one `x y`; `a b => x y` pairs).
            if independent && matches!(pattern, Pattern::Repeat { .. }) {
                return push_verbatim(pattern, emit, out);
            }
            match pattern {
                Pattern::Seq(from_parts) if from_parts.len() == parts.len() => {
                    for (p, e) in from_parts.iter().zip(parts) {
                        collect_pieces(p, e, floating, segments, filtered, out)?;
                    }
                    Ok(())
                }
                // Each branch pairs with the whole emitter; piece lists can
                // differ per branch, which only matters when some piece
                // reads its span (floating transfer).
                Pattern::Alt(_) if independent && floating == 0 => {
                    push_verbatim(pattern, emit, out)
                }
                Pattern::Alt(_) => Err("an alternation paired with a sequence output"),
                _ if independent => push_verbatim(pattern, emit, out),
                _ => Err(UNPAIRABLE),
            }
        }
        Emit::Matrix(update) => {
            if !update.vars.is_empty() {
                return Err("feature variables in the output");
            }
            if update.syl_mask != 0 || update.syl_bits != 0 {
                return Err("syllable features in the output");
            }
            matrix_pieces(pattern, update, filtered, out)
        }
        Emit::Text {
            word,
            exact,
            syl_mask,
            ..
        } => {
            if word.syl.is_some() || *syl_mask != 0 {
                return Err("syllable structure in the output");
            }
            // Exact text is lexurgy's `TextEmitter`, independent only;
            // sequences and repeaters prefer independent emission.
            if *exact || crate::vm::prefers_independent(pattern) {
                out.push(Piece {
                    width: const_width(pattern),
                    emit: PieceEmit::Verbatim(vec![word.segs.clone()]),
                });
                return Ok(());
            }
            transfer_piece(pattern, &word.segs, floating, segments, out)
        }
        Emit::Empty => {
            out.push(Piece {
                width: const_width(pattern),
                emit: PieceEmit::Verbatim(vec![vec![]]),
            });
            Ok(())
        }
        Emit::CaptureRef { .. } | Emit::SylCaptureRef { .. } => Err("captures in the output"),
        Emit::SyllableBoundary => Err("syllable breaks in the output"),
        Emit::WordBreak => Err("word breaks in the output"),
    }
}

/// An independent emission: gather the emitted segments, one *unit* per
/// flattened element of a sequence emitter, a single unit otherwise
/// (`independent_spec` + `flatten_indep`); one piece spans the whole
/// pattern.
pub(super) fn push_verbatim(
    pattern: &Pattern,
    emit: &Emit,
    out: &mut Vec<Piece>,
) -> Result<(), VmReason> {
    fn gather(emit: &Emit, units: &mut Vec<Vec<SegmentId>>) -> Result<(), VmReason> {
        match emit {
            Emit::Empty => {
                units.push(vec![]);
                Ok(())
            }
            Emit::Text { word, syl_mask, .. } if word.syl.is_none() && *syl_mask == 0 => {
                units.push(word.segs.clone());
                Ok(())
            }
            Emit::Text { .. } => Err("syllable structure in the output"),
            Emit::Seq(parts) => parts.iter().try_for_each(|part| gather(part, units)),
            Emit::CaptureRef { .. } | Emit::SylCaptureRef { .. } => Err("captures in the output"),
            Emit::SyllableBoundary => Err("syllable breaks in the output"),
            Emit::WordBreak => Err("word breaks in the output"),
            Emit::Matrix(_) | Emit::Alt(_) => Err(UNPAIRABLE),
        }
    }
    let mut units = Vec::new();
    gather(emit, &mut units)?;
    out.push(Piece {
        width: const_width(pattern),
        emit: PieceEmit::Verbatim(units),
    });
    Ok(())
}

/// A conditional matrix emitter: a sequence pairs every element with the
/// same matrix, so decompose sequences (isolating `*` elements, which
/// render the matrix alone); any other Empty-free pattern is one
/// per-segment piece.
pub(super) fn matrix_pieces(
    pattern: &Pattern,
    update: &MatrixUpdate,
    filtered: bool,
    out: &mut Vec<Piece>,
) -> Result<(), VmReason> {
    match pattern {
        Pattern::Empty => {
            out.push(Piece {
                width: Some(0),
                emit: PieceEmit::Matrix {
                    update: update.clone(),
                    insert_when_empty: true,
                },
            });
            Ok(())
        }
        Pattern::Seq(parts) => {
            for part in parts {
                matrix_pieces(part, update, filtered, out)?;
            }
            Ok(())
        }
        Pattern::Intersect(parts) => {
            let first = parts.first().ok_or("an empty intersection")?;
            matrix_pieces(first, update, filtered, out)
        }
        // A `*` under an alternative or repeater would make zero-width
        // mean "insert" on some paths and "nothing" on others.
        _ if contains_empty(pattern) => Err("a zero-width branch under a matrix output"),
        // Filtered rules unfilter each VM sub-transformation separately;
        // repeaters (one sub per repetition) and sequences nested in
        // alternatives would have sub granularity a single piece can't
        // represent.
        _ if filtered && !sub_free(pattern) => {
            Err("a repeater under a matrix output in a filtered rule")
        }
        _ => {
            out.push(Piece {
                width: const_width(pattern),
                emit: PieceEmit::Matrix {
                    update: update.clone(),
                    insert_when_empty: false,
                },
            });
            Ok(())
        }
    }
}

/// Would a conditional pairing with this pattern produce exactly one VM
/// transformation (no `combine_subs` chaining)? Leaves do; alternatives do
/// iff every branch does; repeaters pair per repetition and sequences per
/// element.
pub(super) fn sub_free(pattern: &Pattern) -> bool {
    match pattern {
        Pattern::Test(_) | Pattern::Text(_) | Pattern::Not(_) => true,
        Pattern::Alt(parts) => parts.iter().all(sub_free),
        _ => false,
    }
}

/// A conditional text emitter at a leaf: resolve the floating-transfer
/// mode from the pattern shape and lengths (`vm::Executor::text_result`'s
/// three cases, decided at compile time).
pub(super) fn transfer_piece(
    pattern: &Pattern,
    text: &[SegmentId],
    floating: DiacriticMask,
    segments: &SegmentInterner,
    out: &mut Vec<Piece>,
) -> Result<(), VmReason> {
    let verbatim = |out: &mut Vec<Piece>| {
        out.push(Piece {
            width: const_width(pattern),
            emit: PieceEmit::Verbatim(vec![text.to_vec()]),
        });
        Ok(())
    };
    if floating == 0 {
        return verbatim(out);
    }
    match pattern {
        Pattern::Text(t) if !t.exact => {
            // The pattern's own diacritics are excluded from transfer;
            // non-literal tests mean no transfer at all.
            let excls: Option<Vec<DiacriticMask>> = t
                .tests
                .iter()
                .map(|test| match test {
                    SegTest::Literal { id, .. } => Some(segments.get(*id).diacritics),
                    _ => None,
                })
                .collect();
            let Some(excls) = excls else {
                return verbatim(out);
            };
            let p = t.tests.len();
            let mode = if text.len() == p {
                TransferMode::Pairwise(excls)
            } else if text.len() == 1 {
                TransferMode::Collapse(excls)
            } else if p == 1 {
                TransferMode::Expand(excls[0])
            } else {
                return verbatim(out);
            };
            out.push(Piece {
                width: Some(p),
                emit: PieceEmit::Transfer {
                    text: text.to_vec(),
                    mode,
                },
            });
            Ok(())
        }
        Pattern::Test(SegTest::Literal { id, .. }) => {
            out.push(Piece {
                width: Some(1),
                emit: PieceEmit::Transfer {
                    text: text.to_vec(),
                    mode: TransferMode::Expand(segments.get(*id).diacritics),
                },
            });
            Ok(())
        }
        Pattern::Test(SegTest::Matrix(_)) => {
            out.push(Piece {
                width: Some(1),
                emit: PieceEmit::Transfer {
                    text: text.to_vec(),
                    mode: TransferMode::Expand(0),
                },
            });
            Ok(())
        }
        // Exact text, `[]`, negations, and zero-width leaves emit verbatim
        // (`text_result` finds no pattern segments to transfer from).
        Pattern::Text(_)
        | Pattern::Test(_)
        | Pattern::Not(_)
        | Pattern::Empty
        | Pattern::WordStart
        | Pattern::WordEnd => verbatim(out),
        // Each branch pairs with the same text. The claim doesn't say
        // which branch matched, so this only works when every branch
        // produces the same piece (e.g. diacritic-free branches of equal
        // width; no exclusions to differ on).
        Pattern::Alt(parts) => {
            let mut agreed: Option<Piece> = None;
            for part in parts {
                let mut branch = Vec::new();
                transfer_piece(part, text, floating, segments, &mut branch)?;
                debug_assert_eq!(branch.len(), 1);
                let piece = branch.pop().expect("leaf transfer produces one piece");
                match &agreed {
                    None => agreed = Some(piece),
                    Some(prev) if *prev == piece => {}
                    Some(_) => return Err("an alternation paired with a transferring text output"),
                }
            }
            out.push(agreed.ok_or("an empty alternative list")?);
            Ok(())
        }
        _ => Err(UNPAIRABLE),
    }
}
