// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) Autumn <auctumnus@pm.me>

//! The packed feature model.
//!
//! Lexurgy matches feature matrices by probing hash sets of value objects,
//! filling in implicit defaults on every lookup. Here the whole feature space
//! is packed into one integer per segment: every declared feature owns a
//! bitfield in a [`FeatureWord`], and **code 0 in each field is that
//! feature's default value** (lexurgy's `Feature.default`: the explicit
//! default for `+feature`s and null-aliased features, otherwise the absent
//! value `*name`).
//!
//! Lexurgy normalizes absent → default at lookup time (`normalizeAbsent`) and
//! fills unmentioned features with defaults (`fullValueList`), so with this
//! encoding:
//!
//! - a segment carrying no feature information is feature word 0, and
//! - matching a whole matrix is `(word & mask) == want`: one AND and one
//!   compare, however many values it has (plus one test per negated value).
//!
//! Segment-level and syllable-level features pack into *separate* words,
//! mirroring lexurgy's `WordLevel` split: segment words live on segments,
//! syllable words on syllables.

use std::collections::HashMap;

use smol_str::SmolStr;

use super::CompileError;
use crate::ast;

/// Packed feature values for one segment (or one syllable). All-defaults is
/// `FeatureWord(0)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, PartialOrd, Ord)]
pub struct FeatureWord(pub u128);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Level {
    Segment,
    Syllable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FeatureId(pub u16);

#[derive(Debug, Clone)]
pub struct FeatureDef {
    pub name: SmolStr,
    pub level: Level,
    /// Bit offset of this feature's field within its level's word.
    pub shift: u8,
    /// Width of the field.
    pub bits: u8,
    /// Value names by code; `values[0]` is the default. If
    /// `default_is_absent`, `values[0]` is the synthesized `*name` rather
    /// than a declared name, and shouldn't be rendered in matrices.
    pub values: Vec<SmolStr>,
    pub default_is_absent: bool,
}

impl FeatureDef {
    pub fn field_mask(&self) -> u128 {
        ((1u128 << self.bits) - 1) << self.shift
    }

    pub fn encode(&self, code: u8) -> u128 {
        (code as u128) << self.shift
    }

    pub fn extract(&self, word: FeatureWord) -> u8 {
        ((word.0 >> self.shift) as u8) & ((1u8 << self.bits) - 1)
    }
}

/// A value resolved against the model: either a concrete (feature, code)
/// pair or a feature variable (`$Place`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolvedValue {
    Value(FeatureId, u8),
    Variable(FeatureId),
}

/// Mask-and-compare tests over one feature word.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct BitTest {
    /// All positive values fused into one test: `word & eq_mask == eq_want`.
    pub eq_mask: u128,
    pub eq_want: u128,
    /// One test per negated value: `word & mask != want`.
    pub ne: Vec<(u128, u128)>,
}

impl BitTest {
    pub fn matches(&self, word: FeatureWord) -> bool {
        word.0 & self.eq_mask == self.eq_want
            && self.ne.iter().all(|&(mask, want)| word.0 & mask != want)
    }

    pub fn is_trivial(&self) -> bool {
        self.eq_mask == 0 && self.ne.is_empty()
    }
}

/// A feature variable occurrence in match position. Unnegated, it reads the
/// segment's field and binds it (or compares against an existing binding);
/// negated (`!$Place`), it requires the field to differ from the binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VarTest {
    pub feature: FeatureId,
    pub negated: bool,
}

/// A matrix compiled for match position.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct MatrixTest {
    pub seg: BitTest,
    pub syl: BitTest,
    pub vars: Vec<VarTest>,
}

/// A matrix compiled for output position: overwrite the mentioned fields and
/// leave the rest, i.e. `word & !mask | bits` (this is exactly lexurgy's
/// `Matrix.update`), then store variable bindings into their fields.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct MatrixUpdate {
    pub seg_mask: u128,
    pub seg_bits: u128,
    pub syl_mask: u128,
    pub syl_bits: u128,
    /// `$Feature` in output position: copy the bound code into the field.
    pub vars: Vec<FeatureId>,
}

#[derive(Debug, Default, Clone)]
pub struct FeatureModel {
    features: Vec<FeatureDef>,
    by_feature_name: HashMap<SmolStr, FeatureId>,
    /// Value names form a single global namespace (lexurgy rejects the same
    /// value name in two features), so a bare value name resolves without
    /// knowing its feature.
    by_value_name: HashMap<SmolStr, (FeatureId, u8)>,
    /// Bits allocated so far in the segment / syllable words.
    used_bits: [u16; 2],
}

fn level_index(level: Level) -> usize {
    match level {
        Level::Segment => 0,
        Level::Syllable => 1,
    }
}

fn plus_level(syllable: bool) -> Level {
    if syllable {
        Level::Syllable
    } else {
        Level::Segment
    }
}

impl FeatureModel {
    pub fn def(&self, id: FeatureId) -> &FeatureDef {
        &self.features[id.0 as usize]
    }

    pub fn feature(&self, name: &str) -> Option<FeatureId> {
        self.by_feature_name.get(name).copied()
    }

    pub fn defs(&self) -> impl Iterator<Item = (FeatureId, &FeatureDef)> {
        self.features
            .iter()
            .enumerate()
            .map(|(i, def)| (FeatureId(i as u16), def))
    }

    /// `Feature +foo` (binary, `-foo` default) or `Feature foo`
    /// (plus-minus, absent default).
    pub fn add_plus(&mut self, pf: &ast::PlusFeature) -> Result<FeatureId, CompileError> {
        let name = &pf.name;
        if pf.plus {
            self.add(
                name.clone(),
                plus_level(pf.syllable),
                vec![
                    SmolStr::from(format!("-{name}")),
                    SmolStr::from(format!("+{name}")),
                ],
                false,
            )
        } else {
            self.add(
                name.clone(),
                plus_level(pf.syllable),
                vec![
                    SmolStr::from(format!("*{name}")),
                    SmolStr::from(format!("+{name}")),
                    SmolStr::from(format!("-{name}")),
                ],
                true,
            )
        }
    }

    /// `Feature Height(*low, mid, high)`: the null alias, if present,
    /// becomes the name of the default (code 0).
    pub fn add_full(
        &mut self,
        syllable: bool,
        name: &SmolStr,
        null_alias: Option<&SmolStr>,
        values: &[SmolStr],
    ) -> Result<FeatureId, CompileError> {
        let mut codes = vec![match null_alias {
            Some(alias) => alias.clone(),
            None => SmolStr::from(format!("*{name}")),
        }];
        codes.extend(values.iter().cloned());
        self.add(
            name.clone(),
            plus_level(syllable),
            codes,
            null_alias.is_none(),
        )
    }

    fn add(
        &mut self,
        name: SmolStr,
        level: Level,
        values: Vec<SmolStr>,
        default_is_absent: bool,
    ) -> Result<FeatureId, CompileError> {
        if self.by_feature_name.contains_key(&name) {
            return Err(CompileError::Duplicate {
                kind: "feature",
                name: name.to_string(),
            });
        }
        debug_assert!(values.len() >= 2);
        let bits = (u32::BITS - (values.len() as u32 - 1).leading_zeros()) as u8;
        let li = level_index(level);
        if self.used_bits[li] + bits as u16 > 128 {
            return Err(CompileError::FeatureSpaceExhausted {
                level: match level {
                    Level::Segment => "segment",
                    Level::Syllable => "syllable",
                },
            });
        }
        let id = FeatureId(self.features.len() as u16);
        for (code, value) in values.iter().enumerate() {
            // The absent value is reachable through `*name` (the feature-name
            // path in `resolve_value`), not the value namespace.
            if code == 0 && default_is_absent {
                continue;
            }
            if self
                .by_value_name
                .insert(value.clone(), (id, code as u8))
                .is_some()
            {
                return Err(CompileError::Duplicate {
                    kind: "feature value",
                    name: value.to_string(),
                });
            }
        }
        self.features.push(FeatureDef {
            name: name.clone(),
            level,
            shift: self.used_bits[li] as u8,
            bits,
            values,
            default_is_absent,
        });
        self.used_bits[li] += bits as u16;
        self.by_feature_name.insert(name, id);
        Ok(id)
    }

    /// Resolve one matrix value. `*name` resolves to code 0; lexurgy's
    /// `normalizeAbsent` maps the absent value to the feature's default, and
    /// the default is always code 0 here.
    pub fn resolve_value(
        &self,
        kind: &ast::MatrixValueKind,
    ) -> Result<ResolvedValue, CompileError> {
        use ast::MatrixValueKind::*;
        let lookup = |name: &str| {
            self.by_value_name
                .get(name)
                .copied()
                .ok_or_else(|| CompileError::Undefined {
                    kind: "feature value",
                    name: name.to_string(),
                })
        };
        let feature = |name: &SmolStr| {
            self.feature(name).ok_or_else(|| CompileError::Undefined {
                kind: "feature",
                name: name.to_string(),
            })
        };
        Ok(match kind {
            Plus(name) => {
                let (f, code) = lookup(&format!("+{name}"))?;
                ResolvedValue::Value(f, code)
            }
            Minus(name) => {
                let (f, code) = lookup(&format!("-{name}"))?;
                ResolvedValue::Value(f, code)
            }
            Simple(name) => {
                let (f, code) = lookup(name)?;
                ResolvedValue::Value(f, code)
            }
            Absent(name) => ResolvedValue::Value(feature(name)?, 0),
            Variable(name) => ResolvedValue::Variable(feature(name)?),
        })
    }

    /// Compile a matrix for match position.
    pub fn compile_test(&self, matrix: &ast::Matrix) -> Result<MatrixTest, CompileError> {
        let mut test = MatrixTest::default();
        for value in matrix {
            match self.resolve_value(&value.value)? {
                ResolvedValue::Value(f, code) => {
                    let def = self.def(f);
                    let bt = match def.level {
                        Level::Segment => &mut test.seg,
                        Level::Syllable => &mut test.syl,
                    };
                    let mask = def.field_mask();
                    let want = def.encode(code);
                    if value.negated {
                        bt.ne.push((mask, want));
                    } else {
                        if bt.eq_mask & mask != 0 {
                            return Err(CompileError::RepeatedFeature {
                                feature: def.name.to_string(),
                            });
                        }
                        bt.eq_mask |= mask;
                        bt.eq_want |= want;
                    }
                }
                ResolvedValue::Variable(f) => test.vars.push(VarTest {
                    feature: f,
                    negated: value.negated,
                }),
            }
        }
        Ok(test)
    }

    /// Compile a matrix for output position. Negated values make no sense
    /// here and are rejected.
    pub fn compile_update(&self, matrix: &ast::Matrix) -> Result<MatrixUpdate, CompileError> {
        let mut update = MatrixUpdate::default();
        for value in matrix {
            if value.negated {
                return Err(CompileError::Invalid {
                    what: "negated value in output matrix".to_string(),
                });
            }
            match self.resolve_value(&value.value)? {
                ResolvedValue::Value(f, code) => {
                    let def = self.def(f);
                    let (mask, bits) = match def.level {
                        Level::Segment => (&mut update.seg_mask, &mut update.seg_bits),
                        Level::Syllable => (&mut update.syl_mask, &mut update.syl_bits),
                    };
                    let field = def.field_mask();
                    if *mask & field != 0 {
                        return Err(CompileError::RepeatedFeature {
                            feature: def.name.to_string(),
                        });
                    }
                    *mask |= field;
                    *bits |= def.encode(code);
                }
                ResolvedValue::Variable(f) => update.vars.push(f),
            }
        }
        Ok(update)
    }

    /// Compile a declaration matrix (symbol or diacritic) to a plain feature
    /// word at a single level. Only positive concrete values are allowed.
    /// Returns the level the values live at, defaulting to `expect` when the
    /// matrix pins nothing down (e.g. all-default values).
    pub fn compile_word(
        &self,
        matrix: &ast::Matrix,
        allowed: &[Level],
    ) -> Result<(Level, FeatureWord), CompileError> {
        let mut level: Option<Level> = None;
        let mut seen_mask = 0u128;
        let mut word = 0u128;
        for value in matrix {
            if value.negated {
                return Err(CompileError::Invalid {
                    what: "negated value in declaration matrix".to_string(),
                });
            }
            let (f, code) = match self.resolve_value(&value.value)? {
                ResolvedValue::Value(f, code) => (f, code),
                ResolvedValue::Variable(_) => {
                    return Err(CompileError::Invalid {
                        what: "feature variable in declaration matrix".to_string(),
                    })
                }
            };
            let def = self.def(f);
            if !allowed.contains(&def.level) {
                return Err(CompileError::InvalidFeatureLevel {
                    value: def.values[code as usize].to_string(),
                });
            }
            match level {
                None => level = Some(def.level),
                Some(l) if l != def.level => {
                    return Err(CompileError::InvalidFeatureLevel {
                        value: def.values[code as usize].to_string(),
                    })
                }
                Some(_) => {}
            }
            let field = def.field_mask();
            if seen_mask & field != 0 {
                return Err(CompileError::RepeatedFeature {
                    feature: def.name.to_string(),
                });
            }
            seen_mask |= field;
            word |= def.encode(code);
        }
        Ok((level.unwrap_or(allowed[0]), FeatureWord(word)))
    }

    /// Like [`compile_word`](Self::compile_word) but also returns the mask of
    /// fields the matrix mentions, which the update form diacritics need
    /// (`word & !mask | bits`).
    pub fn compile_field_update(
        &self,
        matrix: &ast::Matrix,
        allowed: &[Level],
    ) -> Result<(Level, u128, u128), CompileError> {
        let (level, word) = self.compile_word(matrix, allowed)?;
        let mut mask = 0u128;
        for value in matrix {
            if let ResolvedValue::Value(f, _) = self.resolve_value(&value.value)? {
                mask |= self.def(f).field_mask();
            }
        }
        Ok((level, mask, word.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{MatrixValue, MatrixValueKind, PlusFeature};

    fn model() -> FeatureModel {
        let mut m = FeatureModel::default();
        m.add_plus(&PlusFeature {
            syllable: false,
            plus: true,
            name: "long".into(),
        })
        .unwrap();
        m.add_full(
            false,
            &"Height".into(),
            Some(&"low".into()),
            &["mid".into(), "high".into()],
        )
        .unwrap();
        m
    }

    fn value(kind: MatrixValueKind) -> MatrixValue {
        MatrixValue {
            negated: false,
            value: kind,
        }
    }

    #[test]
    fn packing() {
        let m = model();
        let long = m.def(m.feature("long").unwrap());
        assert_eq!((long.shift, long.bits), (0, 1));
        let height = m.def(m.feature("Height").unwrap());
        assert_eq!((height.shift, height.bits), (1, 2));
        // codes: 0 = default ("low" via the null alias), then declared values
        assert_eq!(height.values, vec!["low", "mid", "high"]);
        assert!(!height.default_is_absent);
    }

    #[test]
    fn matrix_test_is_one_compare() {
        let m = model();
        let test = m
            .compile_test(&vec![
                value(MatrixValueKind::Plus("long".into())),
                value(MatrixValueKind::Simple("high".into())),
            ])
            .unwrap();
        assert!(test.vars.is_empty() && test.syl.is_trivial());
        // +long → bit 0 = 1; high → code 2 in the 2-bit field at shift 1
        assert_eq!((test.seg.eq_mask, test.seg.eq_want), (0b111, 0b101));
        assert!(test.seg.matches(FeatureWord(0b101)));
        assert!(!test.seg.matches(FeatureWord(0))); // all defaults
    }

    #[test]
    fn defaults_match_word_zero() {
        let m = model();
        // [-long low] is all defaults: matches the zero word
        let test = m
            .compile_test(&vec![
                value(MatrixValueKind::Minus("long".into())),
                value(MatrixValueKind::Simple("low".into())),
            ])
            .unwrap();
        assert!(test.seg.matches(FeatureWord(0)));
        assert!(!test.seg.matches(FeatureWord(0b1)));
        // *Height also normalizes to the default: want code 0 in the field
        let absent = m
            .compile_test(&vec![value(MatrixValueKind::Absent("Height".into()))])
            .unwrap();
        let height = m.def(m.feature("Height").unwrap());
        assert_eq!(absent.seg.eq_mask, height.field_mask());
        assert_eq!(absent.seg.eq_want, 0);
    }

    #[test]
    fn negated_values() {
        let m = model();
        let test = m
            .compile_test(&vec![MatrixValue {
                negated: true,
                value: MatrixValueKind::Plus("long".into()),
            }])
            .unwrap();
        assert!(test.seg.matches(FeatureWord(0)));
        assert!(!test.seg.matches(FeatureWord(0b1)));
    }

    #[test]
    fn update_overwrites_only_mentioned_fields() {
        let m = model();
        let update = m
            .compile_update(&vec![value(MatrixValueKind::Simple("mid".into()))])
            .unwrap();
        // +long segment, height low → height becomes mid, long untouched
        let word = FeatureWord(0b001);
        let result = word.0 & !update.seg_mask | update.seg_bits;
        assert_eq!(result, 0b011);
    }

    #[test]
    fn declaration_matrices_reject_negated_and_variable_values() {
        // The grammar forbids `!` and `$` inside a symbol/diacritic matrix,
        // so these are defensive guards reached only via direct construction
        // `compile_word` still rejects them rather than silently dropping
        // the negation / binding a variable that declarations can't have.
        let m = model();
        let negated = vec![MatrixValue {
            negated: true,
            value: MatrixValueKind::Plus("long".into()),
        }];
        assert!(matches!(
            m.compile_word(&negated, &[Level::Segment]),
            Err(CompileError::Invalid { .. })
        ));
        let variable = vec![value(MatrixValueKind::Variable("Height".into()))];
        assert!(matches!(
            m.compile_word(&variable, &[Level::Segment]),
            Err(CompileError::Invalid { .. })
        ));
    }
}
