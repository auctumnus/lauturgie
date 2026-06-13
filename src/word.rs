// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) Autumn <auctumnus@pm.me>

//! The runtime word: interned segments plus optional syllable structure.
//!
//! A port of lexurgy's `StandardWord`/`Syllabification` pair. Segments are
//! [`SegmentId`]s; syllable structure is a list of break positions plus
//! per-syllable modifier lists (syllable-level diacritics, by declaration
//! index, in spelling order). Everything here is *structural*; the feature
//! math on syllable modifiers lives with the VM, which has the declarations.
//!
//! Syllable breaks may include positions 0 and `len` ("bounding breaks",
//! e.g. parsed from a leading `.`); rules remove them after each application
//! (`removeBoundingBreaks`), matching lexurgy.

use std::collections::BTreeMap;

use crate::compiler::segments::SegmentId;

/// Syllable modifiers on one syllable: syllable-level diacritic indices in
/// spelling order (lexurgy's `List<Modifier>`).
pub type SylMods = Vec<u8>;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct Syl {
    /// Ascending break positions; may include 0 and the word length.
    pub breaks: Vec<usize>,
    /// Syllable number → modifiers. Only non-empty entries are stored
    /// (lexurgy filters empty modifier lists out).
    pub mods: BTreeMap<usize, SylMods>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct Word {
    pub segs: Vec<SegmentId>,
    pub syl: Option<Syl>,
}

impl Syl {
    pub fn new(breaks: Vec<usize>, mods: BTreeMap<usize, SylMods>) -> Syl {
        Syl {
            breaks,
            mods: mods.into_iter().filter(|(_, m)| !m.is_empty()).collect(),
        }
    }
}

/// Lexurgy's `Syllabification` helpers, parameterized by the word length the
/// structure describes.
struct SylView<'a> {
    syl: &'a Syl,
    len: usize,
}

impl<'a> SylView<'a> {
    fn break_at_start(&self) -> bool {
        self.syl.breaks.first() == Some(&0)
    }

    fn break_at_end(&self) -> bool {
        self.syl.breaks.last() == Some(&self.len)
    }

    fn num_syllables(&self) -> usize {
        (self.syl.breaks.len() + 1)
            .saturating_sub(self.break_at_start() as usize)
            .saturating_sub(self.break_at_end() as usize)
    }

    /// Syllable boundaries including the word edges (unless a bounding break
    /// already covers them).
    fn breaks_and_bounds(&self) -> Vec<usize> {
        let mut out = Vec::with_capacity(self.syl.breaks.len() + 2);
        if !self.break_at_start() {
            out.push(0);
        }
        out.extend_from_slice(&self.syl.breaks);
        if !self.break_at_end() {
            out.push(self.len);
        }
        out
    }

    /// Which syllable the segment at `index` belongs to; a port of
    /// `syllableNumberAt`, with Kotlin's behavior at negative indices
    /// preserved (callers may probe `index - 1`).
    fn syllable_number_at(&self, index: isize) -> isize {
        if self.syl.breaks.is_empty() {
            return 0;
        }
        if index >= *self.syl.breaks.last().unwrap() as isize {
            return self.num_syllables() as isize - 1;
        }
        let mut position = -1;
        for (i, &b) in self
            .syl
            .breaks
            .iter()
            .chain(std::iter::once(&self.len))
            .enumerate()
        {
            if b as isize > index {
                position = i as isize;
                break;
            }
        }
        position - self.break_at_start() as isize
    }
}

impl Word {
    pub fn simple(segs: Vec<SegmentId>) -> Word {
        Word { segs, syl: None }
    }

    /// A zero-segment word holding a single syllable break; lexurgy's
    /// `StandardWord.SYLLABLE_BREAK_ONLY`, what `.` emits.
    pub fn break_only() -> Word {
        Word {
            segs: Vec::new(),
            syl: Some(Syl::new(vec![0], BTreeMap::new())),
        }
    }

    pub fn len(&self) -> usize {
        self.segs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.segs.is_empty()
    }

    pub fn is_syllabified(&self) -> bool {
        self.syl.is_some()
    }

    pub fn to_simple(&self) -> Word {
        Word::simple(self.segs.clone())
    }

    fn view(&self) -> Option<SylView<'_>> {
        self.syl.as_ref().map(|syl| SylView {
            syl,
            len: self.len(),
        })
    }

    /// The structure to use when concatenating: an unsyllabified word gets an
    /// empty syllabification (`forcedSyllabification`).
    fn forced(&self) -> Syl {
        self.syl.clone().unwrap_or_default()
    }

    pub fn num_syllables(&self) -> usize {
        self.view().map_or(0, |v| v.num_syllables())
    }

    pub fn syllable_breaks(&self) -> &[usize] {
        self.syl.as_ref().map_or(&[], |s| &s.breaks)
    }

    /// Syllable boundaries including the word edges
    /// (`syllableBreaksAndBoundaries`): syllable `n` spans
    /// `bounds[n]..bounds[n + 1]`.
    pub fn syllable_bounds(&self) -> Vec<usize> {
        match self.view() {
            None => vec![0, self.len()],
            Some(v) => v.breaks_and_bounds(),
        }
    }

    /// Whether a matcher `.` matches at `pos` (lexurgy's
    /// `hasSyllableBoundaryAt`): word edges count as boundaries.
    pub fn has_boundary_at(&self, pos: usize) -> bool {
        self.is_syllabified()
            && (pos == 0 || pos == self.len() || self.syllable_breaks().contains(&pos))
    }

    pub fn syllable_number_at(&self, index: usize) -> usize {
        self.view()
            .map_or(0, |v| v.syllable_number_at(index as isize).max(0) as usize)
    }

    /// Modifiers of the syllable containing `index` (empty when the word is
    /// unsyllabified; syllable feature tests then see all-defaults).
    pub fn mods_at(&self, index: usize) -> &[u8] {
        match &self.syl {
            None => &[],
            Some(syl) => {
                let n = self.syllable_number_at(index);
                syl.mods.get(&n).map_or(&[], |m| m.as_slice())
            }
        }
    }

    pub fn remove_leading_break(mut self) -> Word {
        if let Some(syl) = &mut self.syl {
            if syl.breaks.first() == Some(&0) {
                syl.breaks.remove(0);
            }
        }
        self
    }

    pub fn remove_trailing_break(mut self) -> Word {
        let len = self.len();
        if let Some(syl) = &mut self.syl {
            if syl.breaks.last() == Some(&len) {
                syl.breaks.pop();
            }
        }
        self
    }

    pub fn remove_bounding_breaks(self) -> Word {
        self.remove_leading_break().remove_trailing_break()
    }

    /// `word[lo..hi]` with its syllable structure (lexurgy's `slice`).
    pub fn slice(&self, lo: usize, hi: usize) -> Word {
        let segs = self.segs[lo..hi].to_vec();
        let syl = self.view().map(|v| {
            let breaks = v
                .syl
                .breaks
                .iter()
                .filter(|&&b| b >= lo && b <= hi)
                .map(|&b| b - lo)
                .collect();
            // Kotlin: keys in syllableNumberAt(first)..syllableNumberAt(last),
            // shifted down; `last` is hi-1, which may be lo-1 for an empty
            // slice (Kotlin's IntRange exposes those values regardless).
            let first_syl = v.syllable_number_at(lo as isize);
            let last_syl = v.syllable_number_at(hi as isize - 1);
            let mods = v
                .syl
                .mods
                .iter()
                .filter(|(&k, _)| (k as isize) >= first_syl && (k as isize) <= last_syl)
                .map(|(&k, m)| ((k as isize - first_syl) as usize, m.clone()))
                .collect();
            Syl::new(breaks, mods)
        });
        Word { segs, syl }
    }

    pub fn take(&self, n: usize) -> Word {
        self.slice_take_drop(n, true)
    }

    pub fn drop_front(&self, n: usize) -> Word {
        self.slice_take_drop(n, false)
    }

    /// Lexurgy's `take`/`drop` differ subtly from `slice` (they don't clamp
    /// modifier keys the same way), so port them separately.
    fn slice_take_drop(&self, n: usize, take: bool) -> Word {
        if take {
            if n == self.len() {
                return self.clone();
            }
            let segs = self.segs[..n].to_vec();
            let syl = self.view().map(|v| {
                if n == 0 {
                    return Syl::default();
                }
                Syl::new(
                    v.syl.breaks.iter().copied().filter(|&b| b <= n).collect(),
                    v.syl
                        .mods
                        .iter()
                        .filter(|(&k, _)| k as isize <= v.syllable_number_at(n as isize))
                        .map(|(&k, m)| (k, m.clone()))
                        .collect(),
                )
            });
            Word { segs, syl }
        } else {
            if n == 0 {
                return self.clone();
            }
            let segs = self.segs[n..].to_vec();
            let syl = self.view().map(|v| {
                if n == self.len() {
                    return Syl::default();
                }
                let shift = v.syllable_number_at(n as isize);
                Syl::new(
                    v.syl
                        .breaks
                        .iter()
                        .filter_map(|&b| (b as isize - n as isize).try_into().ok())
                        .collect(),
                    v.syl
                        .mods
                        .iter()
                        .filter_map(|(&k, m)| {
                            let nk = k as isize - shift;
                            (nk >= 0).then(|| (nk as usize, m.clone()))
                        })
                        .collect(),
                )
            });
            Word { segs, syl }
        }
    }

    /// Concatenate, resolving modifiers of a merged boundary syllable with
    /// `combiner(left, right)` (lexurgy's `concat`; the default combiner
    /// keeps the left side's modifiers).
    pub fn concat(&self, other: &Word, combiner: impl Fn(&[u8], &[u8]) -> SylMods) -> Word {
        let mut segs = self.segs.clone();
        segs.extend_from_slice(&other.segs);
        let syl = if self.is_syllabified() || other.is_syllabified() {
            // Empty words shouldn't influence syllable-level features.
            let combined = if self.is_empty() {
                self.concat_syl(other, |_, right| right.to_vec())
            } else if other.is_empty() {
                self.concat_syl(other, |left, _| left.to_vec())
            } else {
                self.concat_syl(other, combiner)
            };
            Some(combined)
        } else {
            None
        };
        Word { segs, syl }
    }

    fn concat_syl(&self, other: &Word, combiner: impl Fn(&[u8], &[u8]) -> SylMods) -> Syl {
        let a = self.forced();
        let b = other.forced();
        let av = SylView {
            syl: &a,
            len: self.len(),
        };
        let bv = SylView {
            syl: &b,
            len: other.len(),
        };
        if self.is_empty() && a.breaks.is_empty() {
            return b;
        }
        let mut other_breaks: Vec<usize> = b.breaks.iter().map(|&x| x + self.len()).collect();
        if av.break_at_end() && bv.break_at_start() {
            other_breaks.remove(0);
        }
        let mods = if av.break_at_end() || bv.break_at_start() {
            let mut m = a.mods.clone();
            for (&k, v) in &b.mods {
                m.insert(k + av.num_syllables(), v.clone());
            }
            m
        } else {
            // Stitch the last syllable of `a` and the first of `b`.
            let offset = av.num_syllables().saturating_sub(1);
            let mut m: BTreeMap<usize, SylMods> = a
                .mods
                .iter()
                .filter(|(&k, _)| k != offset)
                .map(|(&k, v)| (k, v.clone()))
                .collect();
            let merged = combiner(
                a.mods.get(&offset).map_or(&[], |v| v.as_slice()),
                b.mods.get(&0).map_or(&[], |v| v.as_slice()),
            );
            m.insert(offset, merged);
            for (&k, v) in b.mods.iter().filter(|(&k, _)| k != 0) {
                m.insert(k + offset, v.clone());
            }
            m
        };
        let mut breaks = a.breaks.clone();
        breaks.extend(other_breaks);
        Syl::new(breaks, mods)
    }

    /// Keep only the segments at `indices` (ascending), preserving syllable
    /// structure; lexurgy's `retainOnlyIndices`, the filtered-rule view.
    pub fn retain_indices(&self, indices: &[usize]) -> Word {
        let segs: Vec<SegmentId> = indices.iter().map(|&i| self.segs[i]).collect();
        let syl = self.view().map(|v| {
            let mut new_breaks: Vec<usize> = Vec::new();
            let mut new_mods: BTreeMap<usize, SylMods> = BTreeMap::new();
            let mut new_index = 0usize;
            for (syl_index, syl_break) in v.breaks_and_bounds().into_iter().enumerate() {
                let orig_new_index = new_index;
                while new_index < indices.len() && indices[new_index] < syl_break {
                    new_index += 1;
                }
                if new_index > orig_new_index {
                    if syl_index >= 1 {
                        if let Some(m) = v.syl.mods.get(&(syl_index - 1)) {
                            new_mods.insert(new_breaks.len(), m.clone());
                        }
                    }
                    if new_index < indices.len() {
                        new_breaks.push(new_index);
                    }
                }
            }
            Syl::new(new_breaks, new_mods)
        });
        Word { segs, syl }
    }

    /// Copy this word's structure onto `other` (same content, new segments);
    /// lexurgy's `recoverStructure`. Breaks listed in `except` aren't
    /// copied. No-op if `other` already has structure.
    pub fn recover_structure(&self, other: Word, except: &[usize]) -> Word {
        let Some(v) = self.view() else { return other };
        if other.is_syllabified() || other.is_empty() {
            return other;
        }
        let transfer: Vec<usize> = v
            .syl
            .breaks
            .iter()
            .copied()
            .filter(|b| !except.contains(b))
            .collect();
        let mut new_breaks: Vec<usize> = transfer
            .iter()
            .copied()
            .filter(|&b| b < self.len() && b < other.len())
            .collect();
        if transfer.contains(&self.len()) {
            new_breaks.push(other.len());
        }
        let new_count = (new_breaks.len() + 1)
            .saturating_sub((new_breaks.first() == Some(&0)) as usize)
            .saturating_sub((new_breaks.last() == Some(&other.len())) as usize);
        let mut new_mods: BTreeMap<usize, SylMods> = BTreeMap::new();
        for (&n, mods) in &v.syl.mods {
            let nn = if n >= new_count {
                new_count.saturating_sub(1)
            } else {
                n
            };
            new_mods.entry(nn).or_default().extend(mods.iter().copied());
        }
        Word {
            segs: other.segs,
            syl: Some(Syl::new(new_breaks, new_mods)),
        }
    }

    /// Word concatenation with the default (left-priority) combiner;
    /// lexurgy's `Word.join` / `plus`.
    pub fn join(words: &[Word]) -> Word {
        let mut result = Word::default();
        for (i, word) in words.iter().enumerate() {
            if i == 0 {
                result = word.clone();
            } else {
                result = result.concat(word, |left, _| left.to_vec());
            }
        }
        result
    }
}

/// A phrase: one or more words (lexurgy's `Phrase`). Rules operate on whole
/// phrases; matchers can't cross the gap between words except `$$`.
///
/// Positions use a *linear* encoding of lexurgy's `PhraseIndex(word, seg)`:
/// each word's positions `0..=len` map to consecutive integers, with each
/// between-words gap occupying one slot of its own. So for words of lengths
/// `l0, l1, …`, position `l0` is "(0, l0)" (the end of word 0) and `l0 + 1`
/// is "(1, 0)" (the start of word 1): distinct positions, exactly like
/// `PhraseIndex`, and lexurgy's relative-index arithmetic becomes plain
/// addition. Segment lookups return `None` at gap slots, which is what
/// keeps ordinary matchers from consuming across a word boundary.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct Phrase {
    pub words: Vec<Word>,
}

impl Phrase {
    pub fn single(word: Word) -> Phrase {
        Phrase { words: vec![word] }
    }

    /// Total linear positions: segments plus one slot per word gap. The
    /// valid positions are `0..=len()`.
    pub fn len(&self) -> usize {
        let segs: usize = self.words.iter().map(|w| w.len()).sum();
        segs + self.words.len().saturating_sub(1)
    }

    pub fn is_empty(&self) -> bool {
        self.words.iter().all(|w| w.is_empty())
    }

    /// Linear position → `(word, seg)`; `seg == word.len()` is the word's
    /// end (a gap slot, except for the last word). Positions past the end
    /// land in the last word (out of its range).
    pub fn locate(&self, pos: usize) -> (usize, usize) {
        let mut p = pos;
        for (w, word) in self.words.iter().enumerate() {
            if p <= word.len() || w + 1 == self.words.len() {
                return (w, p);
            }
            p -= word.len() + 1;
        }
        (0, p)
    }

    /// Linear position of `(word, 0)`.
    pub fn word_start(&self, word: usize) -> usize {
        self.words[..word].iter().map(|w| w.len() + 1).sum()
    }

    /// The segment at a linear position, or `None` at gap slots and the
    /// phrase end.
    pub fn seg_at(&self, pos: usize) -> Option<SegmentId> {
        let (w, s) = self.locate(pos);
        self.words.get(w)?.segs.get(s).copied()
    }

    /// Number of segments strictly before `pos` (gap slots don't count):
    /// the index of `pos` in the words' joined segments.
    pub fn flat_index(&self, pos: usize) -> usize {
        let (w, s) = self.locate(pos);
        let before: usize = self.words[..w].iter().map(|x| x.len()).sum();
        before + s
    }

    /// Lexurgy's `Phrase.slice`: both endpoints are linear positions; a
    /// slice spanning a gap keeps the word division.
    pub fn slice(&self, lo: usize, hi: usize) -> Phrase {
        let (wl, sl) = self.locate(lo);
        let (wh, sh) = self.locate(hi);
        if wl == wh {
            return Phrase::single(self.words[wl].slice(sl, sh));
        }
        let mut words = Vec::with_capacity(wh - wl + 1);
        words.push(self.words[wl].drop_front(sl));
        words.extend(self.words[wl + 1..wh].iter().cloned());
        words.push(self.words[wh].take(sh));
        Phrase { words }
    }

    /// Lexurgy's `Phrase.concat`: the last word of `self` merges with the
    /// first word of `other` (resolving the boundary syllable's modifiers
    /// with `combiner`); other word divisions are preserved.
    pub fn concat(&self, other: &Phrase, combiner: impl Fn(&[u8], &[u8]) -> SylMods) -> Phrase {
        if self.words.is_empty() {
            return other.clone();
        }
        if other.words.is_empty() {
            return self.clone();
        }
        let mut words: Vec<Word> = self.words[..self.words.len() - 1].to_vec();
        words.push(self.words.last().unwrap().concat(&other.words[0], combiner));
        words.extend(other.words[1..].iter().cloned());
        Phrase { words }
    }

    /// Lexurgy's `Phrase.fromSubPhrases`: concatenate, merging each junction
    /// with the default (left-priority) combiner. Empty (word-less) phrases
    /// vanish.
    pub fn from_sub_phrases<'a>(parts: impl IntoIterator<Item = &'a Phrase>) -> Phrase {
        let mut result = Phrase::default();
        for part in parts {
            result = result.concat(part, |left, _| left.to_vec());
        }
        result
    }

    pub fn remove_leading_break(mut self) -> Phrase {
        if let Some(first) = self.words.first_mut() {
            let w = std::mem::take(first);
            *first = w.remove_leading_break();
        }
        self
    }

    pub fn remove_trailing_break(mut self) -> Phrase {
        if let Some(last) = self.words.last_mut() {
            let w = std::mem::take(last);
            *last = w.remove_trailing_break();
        }
        self
    }

    /// Lexurgy's `Phrase.removeBoundingBreaks`: strips bounding breaks of
    /// *every* word.
    pub fn remove_bounding_breaks(self) -> Phrase {
        Phrase {
            words: self
                .words
                .into_iter()
                .map(|w| w.remove_bounding_breaks())
                .collect(),
        }
    }

    pub fn to_simple(&self) -> Phrase {
        Phrase {
            words: self.words.iter().map(|w| w.to_simple()).collect(),
        }
    }

    pub fn is_syllabified(&self) -> bool {
        self.words.iter().any(|w| w.is_syllabified())
    }

    /// `hasSyllableBoundaryAt`: whether `.` matches at a linear position
    /// (within the word containing it; word edges count).
    pub fn has_boundary_at(&self, pos: usize) -> bool {
        let (w, s) = self.locate(pos);
        self.words[w].has_boundary_at(s)
    }

    /// `hasSyllableBreakBefore`: a word start, or an explicit break.
    pub fn has_break_before(&self, pos: usize) -> bool {
        let (w, s) = self.locate(pos);
        s == 0 || self.words[w].syllable_breaks().contains(&s)
    }

    /// `hasSyllableBreakAfter`: a word end, or an explicit break after.
    pub fn has_break_after(&self, pos: usize) -> bool {
        let (w, s) = self.locate(pos);
        s + 1 == self.words[w].len() || self.words[w].syllable_breaks().contains(&(s + 1))
    }

    pub fn mods_at(&self, pos: usize) -> &[u8] {
        let (w, s) = self.locate(pos);
        self.words[w].mods_at(s)
    }

    /// Interior syllable breaks of every word, as linear positions
    /// (`Phrase.syllableBreaks`).
    pub fn syllable_breaks_linear(&self) -> Vec<usize> {
        let mut out = Vec::new();
        let mut offset = 0;
        for word in &self.words {
            out.extend(word.syllable_breaks().iter().map(|&b| offset + b));
            offset += word.len() + 1;
        }
        out
    }

    /// Join into a single word (`Phrase.join`).
    pub fn join(&self) -> Word {
        Word::join(&self.words)
    }

    /// Join, putting syllable breaks between the original words if any word
    /// is syllabified (`joinWithSyllableBreaks`).
    pub fn join_with_breaks(&self) -> Word {
        if !self.is_syllabified() {
            return self.join();
        }
        let mut result = self.words[0].clone();
        for word in &self.words[1..] {
            result = result.concat(&Word::break_only(), |left, _| left.to_vec());
            result = result.concat(word, |left, _| left.to_vec());
        }
        result
    }

    /// Lexurgy's `Phrase.recoverStructure`: same segments as `other`, with
    /// structure copied from `self` (word boundaries treated as syllable
    /// breaks), resplit along `other`'s word divisions. `except` lists
    /// *flat* segment indices (relative to `self`'s joined segments) whose
    /// breaks aren't copied.
    pub fn recover_structure(&self, other: Phrase, except: &[usize]) -> Phrase {
        let recovered = self
            .join_with_breaks()
            .recover_structure(other.join(), except);
        other.resplit(recovered)
    }

    /// `Phrase.resplit`: cut `joined` along this phrase's word lengths.
    fn resplit(&self, joined: Word) -> Phrase {
        let mut words = Vec::with_capacity(self.words.len());
        let mut remaining = joined;
        for word in &self.words {
            if word.len() >= remaining.len() {
                words.push(remaining.clone());
            } else {
                words.push(remaining.take(word.len()).remove_trailing_break());
                remaining = remaining.drop_front(word.len()).remove_leading_break();
            }
        }
        Phrase { words }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seg(n: u32) -> SegmentId {
        SegmentId(n)
    }

    fn word(n: usize, breaks: Vec<usize>) -> Word {
        Word {
            segs: (0..n as u32).map(seg).collect(),
            syl: Some(Syl::new(breaks, BTreeMap::new())),
        }
    }

    #[test]
    fn syllable_numbers() {
        // "ko.li.mo": breaks at 2 and 4
        let w = word(6, vec![2, 4]);
        assert_eq!(w.num_syllables(), 3);
        assert_eq!(w.syllable_number_at(0), 0);
        assert_eq!(w.syllable_number_at(1), 0);
        assert_eq!(w.syllable_number_at(2), 1);
        assert_eq!(w.syllable_number_at(4), 2);
        assert_eq!(w.syllable_number_at(5), 2);
        assert!(w.has_boundary_at(0));
        assert!(w.has_boundary_at(2));
        assert!(!w.has_boundary_at(3));
        assert!(w.has_boundary_at(6));
    }

    #[test]
    fn bounding_breaks() {
        // "k.opi": break at 1; leading break case ".ka": break at 0
        let w = word(3, vec![0]);
        assert_eq!(w.num_syllables(), 1);
        let w = w.remove_bounding_breaks();
        assert_eq!(w.syllable_breaks(), &[] as &[usize]);
    }

    #[test]
    fn slice_and_concat_roundtrip() {
        let w = word(6, vec![2, 4]);
        let left = w.slice(0, 3);
        let right = w.slice(3, 6);
        assert_eq!(left.syllable_breaks(), &[2]);
        assert_eq!(right.syllable_breaks(), &[1]);
        let joined = left.concat(&right, |l, _| l.to_vec());
        assert_eq!(joined, w);
    }

    #[test]
    fn recover_structure_transfers_breaks() {
        let w = word(4, vec![2]);
        let plain = Word::simple(vec![seg(7), seg(8), seg(9), seg(10)]);
        let recovered = w.recover_structure(plain.clone(), &[]);
        assert_eq!(recovered.syllable_breaks(), &[2]);
        // explicit matched breaks are not transferred
        let except = w.recover_structure(plain, &[2]);
        assert_eq!(except.syllable_breaks(), &[] as &[usize]);
    }

    #[test]
    fn retain_indices_keeps_structure() {
        // segments 0..6 with breaks [2, 4]; keep 1, 2, 5
        let w = word(6, vec![2, 4]);
        let filtered = w.retain_indices(&[1, 2, 5]);
        assert_eq!(filtered.len(), 3);
        // syllable boundaries fall between kept segments: after index 0
        // (break 2) and after index 1 (break 4)
        assert_eq!(filtered.syllable_breaks(), &[1, 2]);
    }
}
