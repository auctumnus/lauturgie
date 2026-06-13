// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) Autumn <auctumnus@pm.me>

//! The declaration-resolution pass.
//!
//! Walks the statement list in order (lexurgy requires declare-before-use,
//! which conveniently means classes can't be recursive) and folds every
//! declaration statement into a [`Declarations`]: the packed
//! [`FeatureModel`], symbol and diacritic tables with their matrices
//! pre-compiled to feature words, and flattened class/element tables for the
//! lowering pass to inline. Non-declaration statements (rules, romanizers,
//! syllable specs) are returned untouched for the next pass.

use std::collections::HashMap;

use smol_str::SmolStr;
use unicode_normalization::UnicodeNormalization;

use super::features::{FeatureModel, FeatureWord, Level};
use super::CompileError;
use crate::ast;

#[derive(Debug, Clone)]
pub struct SymbolDef {
    pub name: SmolStr,
    /// `None` for symbols declared without a matrix (`Symbol ts, dz`):
    /// they exist as units for the phonetic parser but carry no feature
    /// information of their own (lexurgy's `UndeclaredSymbolValue`: their
    /// identity *is* their matrix).
    pub features: Option<FeatureWord>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiacriticPosition {
    /// `(before)`: renders before the core symbol.
    Before,
    /// Default: renders after.
    After,
    /// `(first)`: renders immediately after the first character.
    First,
}

#[derive(Debug, Clone)]
pub struct DiacriticDef {
    pub name: SmolStr,
    /// All of a diacritic's features must live at one level (lexurgy's
    /// `LscInvalidFeatureLevel` check).
    pub level: Level,
    /// Applying the diacritic overwrites the fields it mentions:
    /// `word & !mask | bits`, exactly lexurgy's `Matrix.update`.
    pub mask: u128,
    pub bits: u128,
    pub position: DiacriticPosition,
    /// Floating diacritics are ignored by non-exact text matching.
    pub floating: bool,
}

#[derive(Debug, Clone)]
pub struct Declarations {
    pub features: FeatureModel,
    pub symbols: Vec<SymbolDef>,
    pub diacritics: Vec<DiacriticDef>,
    /// Class refs are inlined at declaration time, so members are plain
    /// texts by the time anything looks here.
    pub classes: HashMap<SmolStr, Vec<ast::Text>>,
    /// `Element` declarations stay as AST; the lowering pass inlines them
    /// (they may reference classes and earlier elements, so they can't be
    /// resolved further without lowering machinery).
    pub elements: HashMap<SmolStr, ast::RuleElement>,
}

fn nfd(text: &str) -> SmolStr {
    if text.is_ascii() {
        return SmolStr::from(text);
    }
    text.nfd().collect::<String>().into()
}

/// Split a statement list into resolved declarations and the remaining
/// (rule-like) statements, in their original order.
pub fn resolve(
    statements: &[ast::Statement],
) -> Result<(Declarations, Vec<&ast::Statement>), CompileError> {
    let mut features = FeatureModel::default();
    let mut symbols: Vec<SymbolDef> = Vec::new();
    let mut symbol_matrices: HashMap<FeatureWord, SmolStr> = HashMap::new();
    let mut diacritics: Vec<DiacriticDef> = Vec::new();
    // Same-matrix detection is per level: a segment-level and a
    // syllable-level diacritic may have numerically identical masks (the
    // two feature words pack independently).
    let mut diacritic_matrices: HashMap<(Level, u128, u128), SmolStr> = HashMap::new();
    let mut classes: HashMap<SmolStr, Vec<ast::Text>> = HashMap::new();
    let mut elements: HashMap<SmolStr, ast::RuleElement> = HashMap::new();
    let mut rest: Vec<&ast::Statement> = Vec::new();

    for statement in statements {
        match statement {
            ast::Statement::Feature(decl) => match decl {
                ast::FeatureDecl::Plus(list) => {
                    for pf in list {
                        features.add_plus(pf)?;
                    }
                }
                ast::FeatureDecl::Full {
                    syllable,
                    name,
                    null_alias,
                    values,
                } => {
                    features.add_full(*syllable, name, null_alias.as_ref(), values)?;
                }
            },

            ast::Statement::Diacritic(decl) => {
                // lexurgy NFD-normalizes declaration text (`normalizeDecompose`)
                let name = nfd(&decl.text.text);
                if diacritics.len() >= super::segments::MAX_DIACRITICS {
                    return Err(CompileError::TooManyDiacritics);
                }
                if diacritics.iter().any(|d| d.name == name) {
                    return Err(CompileError::Duplicate {
                        kind: "diacritic",
                        name: name.to_string(),
                    });
                }
                let (level, mask, bits) = features
                    .compile_field_update(&decl.matrix, &[Level::Segment, Level::Syllable])
                    .map_err(|e| match e {
                        CompileError::InvalidFeatureLevel { .. } => {
                            CompileError::MixedDiacriticLevels {
                                name: name.to_string(),
                            }
                        }
                        other => other,
                    })?;
                if diacritic_matrices
                    .insert((level, mask, bits), name.clone())
                    .is_some()
                {
                    return Err(CompileError::DuplicateMatrix {
                        kind: "diacritic",
                        name: name.to_string(),
                    });
                }
                let mut position = DiacriticPosition::After;
                let mut floating = false;
                for modifier in &decl.modifiers {
                    match modifier {
                        ast::DiacriticModifier::Before => position = DiacriticPosition::Before,
                        ast::DiacriticModifier::First => position = DiacriticPosition::First,
                        ast::DiacriticModifier::Floating => floating = true,
                    }
                }
                diacritics.push(DiacriticDef {
                    name,
                    level,
                    mask,
                    bits,
                    position,
                    floating,
                });
            }

            ast::Statement::Symbol { names, matrix } => {
                let compiled = match matrix {
                    Some(matrix) => {
                        if names.len() != 1 {
                            return Err(CompileError::Invalid {
                                what: "a symbol with a matrix must be declared alone".to_string(),
                            });
                        }
                        // Symbol matrices must be segment-level (lexurgy's
                        // `checkNonSegmentFeatures`).
                        let (_, word) = features.compile_word(matrix, &[Level::Segment])?;
                        Some(word)
                    }
                    None => None,
                };
                for name in names {
                    let name = nfd(&name.text);
                    if symbols.iter().any(|s| s.name == name) {
                        return Err(CompileError::Duplicate {
                            kind: "symbol",
                            name: name.to_string(),
                        });
                    }
                    if let Some(word) = compiled {
                        if symbol_matrices.insert(word, name.clone()).is_some() {
                            return Err(CompileError::DuplicateMatrix {
                                kind: "symbol",
                                name: name.to_string(),
                            });
                        }
                    }
                    symbols.push(SymbolDef {
                        name,
                        features: compiled,
                    });
                }
            }

            ast::Statement::Class {
                name,
                elements: members,
            } => {
                if classes.contains_key(name) {
                    return Err(CompileError::Duplicate {
                        kind: "class",
                        name: name.to_string(),
                    });
                }
                let mut flattened = Vec::new();
                for member in members {
                    match member {
                        ast::ClassElement::Text(text) => flattened.push(text.clone()),
                        ast::ClassElement::ElementRef(reference) => {
                            let inner = classes.get(reference).ok_or(CompileError::Undefined {
                                kind: "class",
                                name: reference.to_string(),
                            })?;
                            flattened.extend(inner.iter().cloned());
                        }
                    }
                }
                classes.insert(name.clone(), flattened);
            }

            ast::Statement::Element { name, element } => {
                // Unlike classes (lexurgy rejects duplicate class names,
                // `Parser.kt`'s `LscDuplicateName("class", ...)`), a
                // redefined `Element` is *not* an error in lexurgy; the
                // element loop just overwrites `definedElements[name]`, so
                // the last declaration wins. We match that by overwriting.
                elements.insert(name.clone(), element.clone());
            }

            other => rest.push(other),
        }
    }

    Ok((
        Declarations {
            features,
            symbols,
            diacritics,
            classes,
            elements,
        },
        rest,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse;

    fn resolve_source(source: &str) -> Declarations {
        let statements = parse(source).expect("parse failed");
        let (decls, _) = resolve(&statements).expect("resolve failed");
        decls
    }

    #[test]
    fn small_file() {
        let decls = resolve_source(
            "Feature Type(*cons, vowel)\n\
             Feature +stressed\n\
             Diacritic ´ [+stressed]\n\
             Symbol a [vowel]\n\
             Class glide {w, j}\n",
        );

        // Type: 1 bit at shift 0; stressed: 1 bit at shift 1
        let ty = decls.features.def(decls.features.feature("Type").unwrap());
        assert_eq!((ty.shift, ty.bits), (0, 1));

        // `a [vowel]` → Type = vowel (code 1)
        assert_eq!(decls.symbols.len(), 1);
        assert_eq!(decls.symbols[0].features, Some(FeatureWord(0b01)));

        // ´ overwrites only the stressed field, setting it to +stressed
        assert_eq!(decls.diacritics.len(), 1);
        let acute = &decls.diacritics[0];
        assert_eq!((acute.mask, acute.bits), (0b10, 0b10));
        assert_eq!(acute.position, DiacriticPosition::After);
        assert!(!acute.floating);

        let glides = &decls.classes["glide"];
        let names: Vec<_> = glides.iter().map(|t| t.text.as_str()).collect();
        assert_eq!(names, vec!["w", "j"]);
    }

    #[test]
    fn class_refs_flatten() {
        let decls = resolve_source(
            "Class stop {p, t, k}\n\
             Class fricative {s, h}\n\
             Class obstruent {@stop, @fricative}\n",
        );
        let names: Vec<_> = decls.classes["obstruent"]
            .iter()
            .map(|t| t.text.as_str())
            .collect();
        assert_eq!(names, vec!["p", "t", "k", "s", "h"]);
    }

    #[test]
    fn plain_symbols_have_no_features() {
        let decls = resolve_source("Symbol ts, dz\n");
        assert_eq!(decls.symbols.len(), 2);
        assert!(decls.symbols.iter().all(|s| s.features.is_none()));
    }

    #[test]
    fn duplicate_symbol_matrix_rejected() {
        let statements = parse(
            "Feature +nasal\n\
             Symbol m [+nasal]\n\
             Symbol n [+nasal]\n",
        )
        .unwrap();
        assert!(matches!(
            resolve(&statements),
            Err(CompileError::DuplicateMatrix { .. })
        ));
    }
}
