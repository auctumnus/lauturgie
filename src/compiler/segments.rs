// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) Autumn <auctumnus@pm.me>

//! Segment identity and the phonetic parser.
//!
//! At runtime a word is a `Vec<SegmentId>`, not a string. Each distinct
//! segment (core symbol + diacritics) is interned exactly once. All the
//! expensive per-segment work lexurgy redoes constantly (phonetic parsing,
//! diacritic search, matrix computation) happens at intern time and is
//! cached forever after.
//!
//! Identity follows lexurgy's canonicalization (`ComplexSymbol.normalize` =
//! `toMatrix().toSymbol()`): a segment built from a symbol with a declared
//! matrix is identified by its *feature word alone* (declared symbols are
//! bijective with their matrices, so e.g. `a` + an acute carrying
//! `[+stressed]` is the same segment as a declared `á [vowel +stressed]`).
//! A segment whose core has no declared matrix (plain symbols, undeclared
//! graphemes) is identified by its core plus whatever features its
//! diacritics contribute. The canonical *spelling* keeps the diacritics in
//! the order the search added them, exactly like lexurgy's
//! `addDiacriticsToMatch`.
//!
//! [`SegmentInterner::parse_word`] is a port of lexurgy's
//! `Declarations.parsePhonetic`: longest-match segmentation of the *raw*
//! text first (`PhoneticParser`), then per-segment NFD normalization that
//! re-parses decomposed cores (`Segment.normalizeDecompose`), so an
//! undeclared `é` stays one opaque segment rather than splitting into
//! `e` + a stray combining acute.

use std::collections::{BTreeMap, HashMap, HashSet};

use smol_str::SmolStr;
use unicode_normalization::UnicodeNormalization;

use super::decls::{Declarations, DiacriticPosition};
use super::features::{FeatureWord, Level};
use crate::word::{Syl, Word};

/// A core symbol: index into the declared-symbol table, or an undeclared
/// grapheme interned on first sight.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CoreId(pub u32);

/// An interned segment: the runtime alphabet.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SegmentId(pub u32);

/// Bitmask of declared diacritics present on a segment, indexed by
/// declaration order. Floating-diacritic text matching is two mask tests
/// over these (lexurgy's `Segment.matches`):
/// `pattern ⊆ segment && segment \ pattern ⊆ floating`.
pub type DiacriticMask = u64;

pub const MAX_DIACRITICS: usize = DiacriticMask::BITS as usize;

/// Cap on states explored by the diacritic search, so a hostile declaration
/// set (dozens of interacting diacritics) can't blow up intern time.
/// Lexurgy relies on wall-clock timeouts instead.
const RENDER_SEARCH_CAP: usize = 10_000;

/// A diacritic with no symbol to attach to (lexurgy's `DanglingDiacritic`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DanglingDiacritic {
    pub word: String,
    /// Byte offset of the offending diacritic.
    pub position: usize,
    pub diacritic: SmolStr,
}

impl std::fmt::Display for DanglingDiacritic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "the diacritic {} at position {} in {} isn't attached to a symbol",
            self.diacritic, self.position, self.word
        )
    }
}

#[derive(Debug, Clone)]
pub struct SegmentData {
    /// Canonical rendering.
    pub text: SmolStr,
    pub core: CoreId,
    /// Which diacritics are present (for matching).
    pub diacritics: DiacriticMask,
    /// The same diacritics in canonical (search-addition) order. The order
    /// matters both for rendering and for re-deriving the feature word.
    pub order: Vec<u8>,
    /// Segment-level feature word: the core's matrix updated by each
    /// diacritic's matrix in turn.
    pub features: FeatureWord,
}

/// Canonical identity of a segment (see module docs).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum SegmentKey {
    /// Core has a declared matrix: features say everything.
    Featural(FeatureWord),
    /// Bare core: identity is the core itself plus diacritic features.
    Cored(CoreId, FeatureWord),
}

/// What a piece of input text can match during segmentation.
#[derive(Debug, Clone, Copy)]
enum Token {
    Symbol,
    Diacritic(u8),
}

/// Output of one segmentation pass, before interning.
#[derive(Debug, Default)]
struct RawParse {
    /// (core text, segment-level diacritics) per segment.
    segs: Vec<(String, Vec<u8>)>,
    breaks: Vec<usize>,
    syl_mods: BTreeMap<usize, Vec<u8>>,
}

#[derive(Debug, Clone)]
pub struct SegmentInterner {
    /// Core symbol texts; the declared symbols come first, in declaration
    /// order, with undeclared graphemes interned after them.
    cores: Vec<SmolStr>,
    core_by_text: HashMap<SmolStr, CoreId>,
    segments: Vec<SegmentData>,
    by_key: HashMap<SegmentKey, SegmentId>,
    /// (Raw input text, syllabified?) → parsed word, so each distinct
    /// string is segmented once per ruleset, ever.
    parse_cache: HashMap<(SmolStr, bool), Word>,
    /// Longest-match table over symbol and diacritic spellings, in both NFD
    /// (canonical) and NFC forms; lexurgy registers raw and normalized
    /// names side by side so pre-normalization text still segments.
    matcher: HashMap<SmolStr, Token>,
    max_token_chars: usize,
}

fn nfd(text: &str) -> String {
    text.nfd().collect()
}

impl SegmentInterner {
    /// Pre-intern every declared symbol as a bare segment and build the
    /// segmentation table.
    pub fn seed(decls: &Declarations) -> Self {
        let mut interner = SegmentInterner {
            cores: Vec::new(),
            core_by_text: HashMap::new(),
            segments: Vec::new(),
            by_key: HashMap::new(),
            parse_cache: HashMap::new(),
            matcher: HashMap::new(),
            max_token_chars: 0,
        };
        let register = |matcher: &mut HashMap<SmolStr, Token>, name: &SmolStr, token: Token| {
            matcher.insert(name.clone(), token);
            if !name.is_ascii() {
                let composed: String = name.nfc().collect();
                if composed != name.as_str() {
                    matcher.insert(SmolStr::from(composed), token);
                }
            }
        };
        for symbol in &decls.symbols {
            register(&mut interner.matcher, &symbol.name, Token::Symbol);
        }
        // Diacritics second: on a spelling collision the diacritic wins,
        // matching lexurgy's map-merge order in `PhoneticParser`.
        for (index, diacritic) in decls.diacritics.iter().enumerate() {
            register(
                &mut interner.matcher,
                &diacritic.name,
                Token::Diacritic(index as u8),
            );
        }
        interner.max_token_chars = interner
            .matcher
            .keys()
            .map(|k| k.chars().count())
            .max()
            .unwrap_or(0);

        for (index, symbol) in decls.symbols.iter().enumerate() {
            let core = CoreId(index as u32);
            interner.cores.push(symbol.name.clone());
            interner.core_by_text.insert(symbol.name.clone(), core);
            let key = match symbol.features {
                Some(word) => SegmentKey::Featural(word),
                None => SegmentKey::Cored(core, FeatureWord::default()),
            };
            let id = SegmentId(interner.segments.len() as u32);
            interner.segments.push(SegmentData {
                text: symbol.name.clone(),
                core,
                diacritics: 0,
                order: Vec::new(),
                features: symbol.features.unwrap_or_default(),
            });
            interner.by_key.insert(key, id);
        }
        interner
    }

    pub fn get(&self, id: SegmentId) -> &SegmentData {
        &self.segments[id.0 as usize]
    }

    pub fn len(&self) -> usize {
        self.segments.len()
    }

    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }

    pub fn core_text(&self, core: CoreId) -> &SmolStr {
        &self.cores[core.0 as usize]
    }

    /// Segment a string into an interned [`Word`], lexurgy's
    /// `Declarations.parsePhonetic`. With `syllabified` set (a syllabifier
    /// is in force), `.` separates syllables and syllable-level diacritics
    /// attach to syllables; otherwise `.` is ordinary text and syllable
    /// diacritics are consumed and dropped, matching lexurgy.
    pub fn parse_word(
        &mut self,
        decls: &Declarations,
        text: &str,
        syllabified: bool,
    ) -> Result<Word, DanglingDiacritic> {
        let cache_key = (SmolStr::from(text), syllabified);
        if let Some(word) = self.parse_cache.get(&cache_key) {
            return Ok(word.clone());
        }
        // Segment the *raw* text first, then NFD-normalize per segment
        // (lexurgy's `Segment.normalizeDecompose`): a decomposed core
        // re-parses, and either folds back into one segment (its diacritics
        // were declared) or stays whole as an opaque core.
        let parsed = self.segment(decls, text, syllabified)?;
        let mut normalized: Vec<(String, Vec<u8>)> = Vec::with_capacity(parsed.segs.len());
        for (core, modifiers) in parsed.segs {
            if core.is_ascii() {
                normalized.push((core, modifiers));
                continue;
            }
            let decomposed = nfd(&core);
            if decomposed == core {
                normalized.push((core, modifiers));
                continue;
            }
            let mut sub = self.segment(decls, &decomposed, false)?;
            if sub.segs.len() == 1 {
                let (sub_core, mut sub_modifiers) = sub.segs.pop().unwrap();
                sub_modifiers.extend(modifiers);
                normalized.push((sub_core, sub_modifiers));
            } else {
                normalized.push((decomposed, modifiers));
            }
        }

        let segs: Vec<SegmentId> = normalized
            .into_iter()
            .map(|(core_text, diacritics)| {
                let core = self.core_id(&core_text);
                self.intern(decls, core, &diacritics)
            })
            .collect();
        let word = Word {
            segs,
            syl: syllabified.then(|| Syl::new(parsed.breaks, parsed.syl_mods)),
        };
        self.parse_cache.insert(cache_key, word.clone());
        Ok(word)
    }

    /// One segmentation pass (lexurgy's `PhoneticParseRun`): longest-match
    /// against symbols/diacritics, unknown characters one at a time.
    fn segment(
        &self,
        decls: &Declarations,
        text: &str,
        syllabified: bool,
    ) -> Result<RawParse, DanglingDiacritic> {
        let dangling = |unparsed: &str, diacritic: &str| DanglingDiacritic {
            word: text.to_string(),
            position: text.len() - unparsed.len(),
            diacritic: SmolStr::from(diacritic),
        };

        let mut unparsed = text.to_string();
        let mut core: Option<String> = None;
        // Pending diacritics: `(before)` ones accumulate *ahead* of their
        // core, so this buffer belongs to whichever segment finishes next.
        let mut diacritics: Vec<u8> = Vec::new();
        let mut parsed = RawParse::default();
        // Pending syllable-level diacritics for the syllable in progress.
        let mut syl_diacritics: Vec<u8> = Vec::new();

        macro_rules! done_segment {
            () => {
                if let Some(done) = core.take() {
                    parsed.segs.push((done, std::mem::take(&mut diacritics)));
                }
            };
        }
        macro_rules! done_syllable {
            ($add_break:expr) => {
                parsed
                    .syl_mods
                    .entry(parsed.breaks.len())
                    .or_default()
                    .extend(std::mem::take(&mut syl_diacritics));
                if $add_break {
                    parsed.breaks.push(parsed.segs.len());
                }
            };
        }

        while !unparsed.is_empty() {
            if syllabified && unparsed.starts_with('.') {
                done_segment!();
                done_syllable!(true);
                unparsed.drain(..1);
                continue;
            }
            match self.try_match(&unparsed) {
                None => {
                    done_segment!();
                    let first = unparsed.chars().next().unwrap();
                    core = Some(first.to_string());
                    unparsed.drain(..first.len_utf8());
                }
                Some((len, Token::Symbol)) => {
                    done_segment!();
                    core = Some(unparsed[..len].to_string());
                    unparsed.drain(..len);
                }
                Some((len, Token::Diacritic(index))) => {
                    let def = &decls.diacritics[index as usize];
                    if def.level == Level::Syllable {
                        // `processSyllableModifier`: the modifier attaches to
                        // a syllable, closing segments/syllables as needed.
                        match def.position {
                            DiacriticPosition::Before => {
                                done_segment!();
                                if parsed.segs.len() != parsed.breaks.last().copied().unwrap_or(0) {
                                    done_syllable!(true);
                                }
                                if len >= unparsed.len() {
                                    return Err(dangling(&unparsed, &unparsed[..len]));
                                }
                                syl_diacritics.push(index);
                                unparsed.drain(..len);
                            }
                            DiacriticPosition::First => {
                                done_segment!();
                                if parsed.segs.is_empty() {
                                    return Err(dangling(&unparsed, &unparsed[..len]));
                                }
                                if parsed.segs.len()
                                    != parsed.breaks.last().copied().unwrap_or(0) + 1
                                {
                                    done_syllable!(false);
                                    parsed.breaks.push(parsed.segs.len() - 1);
                                }
                                syl_diacritics.push(index);
                                unparsed.drain(..len);
                            }
                            DiacriticPosition::After => {
                                done_segment!();
                                syl_diacritics.push(index);
                                unparsed.drain(..len);
                                if syllabified && !unparsed.is_empty() && !unparsed.starts_with('.')
                                {
                                    done_syllable!(true);
                                }
                            }
                        }
                        continue;
                    }
                    match def.position {
                        DiacriticPosition::Before => {
                            done_segment!();
                            if len >= unparsed.len() {
                                return Err(dangling(&unparsed, &unparsed[..len]));
                            }
                            diacritics.push(index);
                            unparsed.drain(..len);
                        }
                        DiacriticPosition::First => {
                            // Re-queue a one-character core so it can match
                            // again as the start of a longer symbol.
                            match core.take() {
                                Some(c) if c.chars().count() == 1 => {
                                    let rest = unparsed[len..].to_string();
                                    diacritics.push(index);
                                    unparsed = c;
                                    unparsed.push_str(&rest);
                                }
                                _ => {
                                    let diacritic = unparsed[..len].to_string();
                                    return Err(dangling(&unparsed, &diacritic));
                                }
                            }
                        }
                        DiacriticPosition::After => {
                            if core.is_some() {
                                diacritics.push(index);
                                unparsed.drain(..len);
                            } else {
                                return Err(dangling(&unparsed, &unparsed[..len]));
                            }
                        }
                    }
                }
            }
        }
        done_segment!();
        done_syllable!(false);
        Ok(parsed)
    }

    /// Longest match against the symbol/diacritic table; returns the byte
    /// length of the match.
    fn try_match(&self, text: &str) -> Option<(usize, Token)> {
        let mut ends: Vec<usize> = Vec::with_capacity(self.max_token_chars);
        for (i, c) in text.char_indices().take(self.max_token_chars) {
            ends.push(i + c.len_utf8());
        }
        for &end in ends.iter().rev() {
            if let Some(&token) = self.matcher.get(&text[..end]) {
                return Some((end, token));
            }
        }
        None
    }

    fn core_id(&mut self, text: &str) -> CoreId {
        if let Some(&core) = self.core_by_text.get(text) {
            return core;
        }
        let core = CoreId(self.cores.len() as u32);
        self.cores.push(SmolStr::from(text));
        self.core_by_text.insert(SmolStr::from(text), core);
        core
    }

    fn core_features(&self, decls: &Declarations, core: CoreId) -> Option<FeatureWord> {
        decls.symbols.get(core.0 as usize).and_then(|s| s.features)
    }

    /// Intern a (core, ordered diacritics) pair, canonicalizing it the way
    /// lexurgy's `fixDiacriticOrder` does: the segment's *matrix* is its
    /// identity, and the stored rendering is whatever the diacritic search
    /// finds first.
    pub fn intern(
        &mut self,
        decls: &Declarations,
        core: CoreId,
        parsed_diacritics: &[u8],
    ) -> SegmentId {
        let core_features = self.core_features(decls, core);
        let mut word = core_features.unwrap_or_default().0;
        for &index in parsed_diacritics {
            let def = &decls.diacritics[index as usize];
            if def.level == Level::Segment {
                word = word & !def.mask | def.bits;
            }
        }
        let word = FeatureWord(word);
        let key = match core_features {
            Some(_) => SegmentKey::Featural(word),
            None => SegmentKey::Cored(core, word),
        };
        self.intern_key(decls, key, core, parsed_diacritics)
    }

    /// Intern by canonical key, searching for the canonical spelling; falls
    /// back to the provided decomposition if the search gives up.
    fn intern_key(
        &mut self,
        decls: &Declarations,
        key: SegmentKey,
        fallback_core: CoreId,
        fallback_order: &[u8],
    ) -> SegmentId {
        if let Some(&id) = self.by_key.get(&key) {
            return id;
        }
        let word = match key {
            SegmentKey::Featural(w) | SegmentKey::Cored(_, w) => w,
        };
        let (core, order) =
            search_render(decls, key).unwrap_or_else(|| (fallback_core, fallback_order.to_vec()));
        let text = self.render_text(decls, core, &order);
        let mask = order.iter().fold(0, |m, &i| m | (1 << i));
        let id = SegmentId(self.segments.len() as u32);
        self.segments.push(SegmentData {
            text,
            core,
            diacritics: mask,
            order,
            features: word,
        });
        self.by_key.insert(key, id);
        id
    }

    /// Re-intern a segment with extra diacritics added (floating-diacritic
    /// transfer, lexurgy's `withFloatingDiacriticsFrom`).
    pub fn with_extra_diacritics(
        &mut self,
        decls: &Declarations,
        id: SegmentId,
        extra: DiacriticMask,
    ) -> SegmentId {
        let data = self.get(id);
        if extra & !data.diacritics == 0 {
            return id;
        }
        let mut order = data.order.clone();
        let existing = data.diacritics;
        for index in 0..decls.diacritics.len() as u8 {
            if extra & !existing & (1 << index) != 0 {
                order.push(index);
            }
        }
        let core = data.core;
        self.reintern(decls, core, order)
    }

    /// Re-intern a segment with some diacritics removed (lexurgy's
    /// `withoutFloatingDiacritics`, used by inexact capture references).
    pub fn without_diacritics(
        &mut self,
        decls: &Declarations,
        id: SegmentId,
        removed: DiacriticMask,
    ) -> SegmentId {
        let data = self.get(id);
        if data.diacritics & removed == 0 {
            return id;
        }
        let order: Vec<u8> = data
            .order
            .iter()
            .copied()
            .filter(|&i| removed & (1 << i) == 0)
            .collect();
        let core = data.core;
        self.reintern(decls, core, order)
    }

    fn reintern(&mut self, decls: &Declarations, core: CoreId, order: Vec<u8>) -> SegmentId {
        self.intern(decls, core, &order)
    }

    /// The segment for a feature word produced by a rule (lexurgy's
    /// `Matrix.toSymbol`), or `None` if no symbol-plus-diacritics
    /// combination has that matrix (`LscInvalidMatrix`).
    pub fn render_features(
        &mut self,
        decls: &Declarations,
        word: FeatureWord,
    ) -> Option<SegmentId> {
        self.render_key(decls, SegmentKey::Featural(word))
    }

    /// The segment for a feature word *anchored to a featureless core*:
    /// when a matrix update hits an undeclared grapheme (or a symbol
    /// declared without a matrix), the matrix keeps lexurgy's
    /// `UndeclaredSymbolValue`, so the diacritic search starts from that
    /// core rather than from declared symbols.
    pub fn render_cored(
        &mut self,
        decls: &Declarations,
        core: CoreId,
        word: FeatureWord,
    ) -> Option<SegmentId> {
        self.render_key(decls, SegmentKey::Cored(core, word))
    }

    fn render_key(&mut self, decls: &Declarations, key: SegmentKey) -> Option<SegmentId> {
        if let Some(&id) = self.by_key.get(&key) {
            return Some(id);
        }
        let word = match key {
            SegmentKey::Featural(w) | SegmentKey::Cored(_, w) => w,
        };
        let (core, order) = search_render(decls, key)?;
        let text = self.render_text(decls, core, &order);
        let mask = order.iter().fold(0, |m, &i| m | (1 << i));
        let id = SegmentId(self.segments.len() as u32);
        self.segments.push(SegmentData {
            text,
            core,
            diacritics: mask,
            order,
            features: word,
        });
        self.by_key.insert(key, id);
        Some(id)
    }

    /// Whether a core carries a declared matrix (its identity is featural).
    pub fn core_is_featural(&self, decls: &Declarations, core: CoreId) -> bool {
        self.core_features(decls, core).is_some()
    }

    /// Spell out core + diacritics (lexurgy's `String.modify`): before
    /// diacritics, first character, first-position diacritics, remaining
    /// characters, after diacritics; each group in the given order.
    fn render_text(&self, decls: &Declarations, core: CoreId, order: &[u8]) -> SmolStr {
        let core_text = self.core_text(core);
        if order.is_empty() {
            return core_text.clone();
        }
        let mut before = String::new();
        let mut first = String::new();
        let mut after = String::new();
        for &index in order {
            let def = &decls.diacritics[index as usize];
            match def.position {
                DiacriticPosition::Before => before.push_str(&def.name),
                DiacriticPosition::First => first.push_str(&def.name),
                DiacriticPosition::After => after.push_str(&def.name),
            }
        }
        let mut chars = core_text.chars();
        let mut text = before;
        if let Some(c) = chars.next() {
            text.push(c);
        }
        text.push_str(&first);
        text.push_str(chars.as_str());
        text.push_str(&after);
        SmolStr::from(text)
    }
}

/// Find the canonical (core, ordered diacritics) spelling of a feature
/// word, breadth-first over diacritic combinations; a port of lexurgy's
/// `addDiacriticsToMatch`. Trying diacritics in declaration order at each
/// level reproduces lexurgy's tie-breaking, and the *addition order* is the
/// canonical spelling order (an overriding diacritic renders after the one
/// it overrides, whatever their declaration order).
fn search_render(decls: &Declarations, key: SegmentKey) -> Option<(CoreId, Vec<u8>)> {
    let (target, frontier) = match key {
        SegmentKey::Featural(word) => {
            let starts: Vec<(CoreId, Vec<u8>, u128)> = decls
                .symbols
                .iter()
                .enumerate()
                .filter_map(|(i, s)| s.features.map(|w| (CoreId(i as u32), Vec::new(), w.0)))
                .collect();
            (word, starts)
        }
        SegmentKey::Cored(core, word) => (word, vec![(core, Vec::new(), 0u128)]),
    };

    // A diacritic is viable if, for every field it sets, it either sets the
    // target's value or the target's value is reachable by *some* diacritic
    // (lexurgy's `viableDiacriticsFor`).
    let viable: Vec<u8> = decls
        .diacritics
        .iter()
        .enumerate()
        .filter(|(_, d)| {
            d.level == Level::Segment
                && decls.features.defs().all(|(_, def)| {
                    let field = def.field_mask();
                    if def.level != Level::Segment || d.mask & field == 0 {
                        return true;
                    }
                    let target_value = target.0 & field;
                    d.bits & field == target_value
                        || decls.diacritics.iter().any(|d2| {
                            d2.level == Level::Segment
                                && d2.mask & field != 0
                                && d2.bits & field == target_value
                        })
                })
        })
        .map(|(i, _)| i as u8)
        .collect();

    let mut seen: HashSet<u128> = HashSet::new();
    let mut frontier = frontier;
    while !frontier.is_empty() {
        let mut next = Vec::new();
        for (core, order, word) in &frontier {
            if *word == target.0 {
                return Some((*core, order.clone()));
            }
            seen.insert(*word);
            for &index in &viable {
                if order.contains(&index) {
                    continue;
                }
                let d = &decls.diacritics[index as usize];
                let new_word = word & !d.mask | d.bits;
                // Dedup at push time: lexurgy keeps same-level duplicates in
                // its frontier, but a duplicate can never match before its
                // first occurrence, so dropping it preserves the BFS
                // tie-breaking while keeping the frontier at distinct words.
                if seen.insert(new_word) {
                    let mut new_order = order.clone();
                    new_order.push(index);
                    next.push((*core, new_order, new_word));
                }
                if seen.len() > RENDER_SEARCH_CAP {
                    return None;
                }
            }
        }
        frontier = next;
    }
    None
}

/// The feature word and field mask of a syllable's modifier list: lexurgy's
/// `List<Modifier>.toMatrix()`, as bits. Later modifiers override earlier
/// ones for the same field.
pub fn syl_mods_bits(decls: &Declarations, mods: &[u8]) -> (u128, u128) {
    let mut mask = 0u128;
    let mut word = 0u128;
    for &index in mods {
        let d = &decls.diacritics[index as usize];
        if d.level == Level::Syllable {
            mask |= d.mask;
            word = word & !d.mask | d.bits;
        }
    }
    (mask, word)
}

/// The explicit-value test a modifier list induces (lexurgy strips fields
/// whose final value is the default; `Matrix.update` removes default
/// values from the explicit list).
pub fn syl_mods_test(decls: &Declarations, mods: &[u8]) -> (u128, u128) {
    let (mut mask, word) = syl_mods_bits(decls, mods);
    for (_, def) in decls.features.defs() {
        if def.level == Level::Syllable {
            let field = def.field_mask();
            if mask & field != 0 && word & field == 0 {
                mask &= !field;
            }
        }
    }
    (mask, word & mask)
}

/// Find the canonical modifier list spelling a syllable-level feature word:
/// lexurgy's `Matrix.toModifiers` (`addDiacriticsToMatch` from an empty
/// symbol). `None` if no diacritic combination spells it.
pub fn render_syl_mods(decls: &Declarations, target: u128) -> Option<Vec<u8>> {
    let viable: Vec<u8> = decls
        .diacritics
        .iter()
        .enumerate()
        .filter(|(_, d)| {
            d.level == Level::Syllable
                && decls.features.defs().all(|(_, def)| {
                    let field = def.field_mask();
                    if def.level != Level::Syllable || d.mask & field == 0 {
                        return true;
                    }
                    let target_value = target & field;
                    d.bits & field == target_value
                        || decls.diacritics.iter().any(|d2| {
                            d2.level == Level::Syllable
                                && d2.mask & field != 0
                                && d2.bits & field == target_value
                        })
                })
        })
        .map(|(i, _)| i as u8)
        .collect();

    let mut seen: HashSet<u128> = HashSet::new();
    let mut frontier: Vec<(Vec<u8>, u128)> = vec![(Vec::new(), 0)];
    while !frontier.is_empty() {
        let mut next = Vec::new();
        for (order, word) in &frontier {
            if *word == target {
                return Some(order.clone());
            }
            seen.insert(*word);
            for &index in &viable {
                if order.contains(&index) {
                    continue;
                }
                let d = &decls.diacritics[index as usize];
                let new_word = word & !d.mask | d.bits;
                // Push-time dedup; see search_render.
                if seen.insert(new_word) {
                    let mut new_order = order.clone();
                    new_order.push(index);
                    next.push((new_order, new_word));
                }
                if seen.len() > RENDER_SEARCH_CAP {
                    return None;
                }
            }
        }
        frontier = next;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse;

    fn setup(source: &str) -> (Declarations, SegmentInterner) {
        let statements = parse(source).expect("parse failed");
        let (decls, _) = super::super::decls::resolve(&statements).expect("resolve failed");
        let interner = SegmentInterner::seed(&decls);
        (decls, interner)
    }

    #[test]
    fn multigraph_symbols_match_longest() {
        let (decls, mut interner) = setup("Symbol ts\n");
        let ids = interner.parse_word(&decls, "atsa", false).unwrap().segs;
        assert_eq!(ids.len(), 3);
        assert_eq!(interner.get(ids[1]).text, "ts");
        assert_eq!(ids[0], ids[2]);
        assert_eq!(interner.get(ids[0]).text, "a");
    }

    #[test]
    fn diacritics_update_features() {
        let (decls, mut interner) = setup(
            "Feature +aspirated\n\
             Diacritic ʰ [+aspirated]\n\
             Symbol t\n",
        );
        let ids = interner.parse_word(&decls, "tʰa", false).unwrap().segs;
        assert_eq!(ids.len(), 2);
        let t_asp = interner.get(ids[0]);
        assert_eq!(t_asp.text, "tʰ");
        assert_eq!(t_asp.diacritics, 0b1);
        assert_eq!(t_asp.features, FeatureWord(0b1));
        // same segment again interns to the same id
        let again = interner.parse_word(&decls, "tʰ", false).unwrap().segs;
        assert_eq!(again, vec![ids[0]]);
    }

    #[test]
    fn symbol_plus_diacritic_canonicalizes_to_declared_symbol() {
        // `a` + stress diacritic has the same matrix as the declared `á`,
        // so it must intern as the *same segment*, rendered "á".
        let (decls, mut interner) = setup(
            "Feature Type(*cons, vowel)\n\
             Feature +stressed\n\
             Diacritic ˈ [+stressed]\n\
             Symbol a [vowel]\n\
             Symbol á [vowel +stressed]\n",
        );
        let ids = interner.parse_word(&decls, "aˈ", false).unwrap().segs;
        assert_eq!(ids.len(), 1);
        // internal text is NFD ("a" + combining acute); NFC happens on output
        assert_eq!(interner.get(ids[0]).text, "a\u{301}");
        let direct = interner.parse_word(&decls, "á", false).unwrap().segs;
        assert_eq!(direct, ids);
    }

    #[test]
    fn undeclared_composed_characters_stay_whole() {
        // lexurgy: "letters with built-in diacritics should be treated as
        // symbols even if not declared": NFC é segments as one unit, and
        // its NFD decomposition stays an opaque single segment
        let (decls, mut interner) = setup("Symbol t\n");
        let ids = interner.parse_word(&decls, "tété", false).unwrap().segs;
        assert_eq!(ids.len(), 4);
        assert_eq!(interner.get(ids[1]).text, "e\u{301}");
        assert_eq!(ids[1], ids[3]);
        // and it is NOT the same segment as a plain e
        let e = interner.parse_word(&decls, "e", false).unwrap().segs;
        assert_ne!(ids[1], e[0]);
    }

    #[test]
    fn before_diacritic_attaches_forward() {
        let (decls, mut interner) = setup(
            "Feature +stressed\n\
             Diacritic ˈ (before) [+stressed]\n",
        );
        let ids = interner.parse_word(&decls, "ˈta", false).unwrap().segs;
        assert_eq!(ids.len(), 2);
        let t = interner.get(ids[0]);
        assert_eq!(t.features, FeatureWord(0b1));
        assert_eq!(t.text, "ˈt");
        assert_eq!(interner.get(ids[1]).features, FeatureWord(0));
    }

    #[test]
    fn dangling_diacritic_is_an_error() {
        let (decls, mut interner) = setup(
            "Feature +aspirated\n\
             Diacritic ʰ [+aspirated]\n",
        );
        let err = interner.parse_word(&decls, "ʰa", false).unwrap_err();
        assert_eq!(err.position, 0);
        assert_eq!(err.diacritic, "ʰ");
    }

    #[test]
    fn render_features_finds_diacritic_spelling() {
        let (decls, mut interner) = setup(
            "Feature Type(*cons, vowel)\n\
             Feature +long\n\
             Diacritic ː [+long]\n\
             Symbol a [vowel]\n",
        );
        // vowel + long: no declared symbol, so the search must produce aː
        let ty = decls.features.def(decls.features.feature("Type").unwrap());
        let long = decls.features.def(decls.features.feature("long").unwrap());
        let word = FeatureWord(ty.encode(1) | long.encode(1));
        let id = interner.render_features(&decls, word).unwrap();
        assert_eq!(interner.get(id).text, "aː");
        // and it round-trips with parsing
        let parsed = interner.parse_word(&decls, "aː", false).unwrap().segs;
        assert_eq!(parsed, vec![id]);
    }

    #[test]
    fn search_renders_in_addition_order() {
        // The shortest decomposition of [+a +b +c] is e (sets +a +b +c +d)
        // overridden by d (sets -d); the override renders *after* even
        // though d is declared first; lexurgy's TestDiacritics "xed".
        let (decls, mut interner) = setup(
            "Feature +a, +b, +c, +d, +x\n\
             Diacritic a [+a]\n\
             Diacritic b [+b]\n\
             Diacritic c [+c]\n\
             Diacritic d [-d]\n\
             Diacritic e [+a +b +c +d]\n\
             Symbol x [+x]\n",
        );
        let m = &decls.features;
        let word = FeatureWord(
            m.def(m.feature("x").unwrap()).encode(1)
                | m.def(m.feature("a").unwrap()).encode(1)
                | m.def(m.feature("b").unwrap()).encode(1)
                | m.def(m.feature("c").unwrap()).encode(1),
        );
        let id = interner.render_features(&decls, word).unwrap();
        assert_eq!(interner.get(id).text, "xed");
    }
}
