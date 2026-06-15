// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) Autumn <auctumnus@pm.me>

//! The lowering pass: rule statements → [`ir`](super::ir) against resolved
//! declarations, sequenced into a pipeline of [`Step`]s.
//!
//! Everything name-shaped disappears here. Class and element refs are
//! inlined (declare-before-use makes that terminate), literal text runs
//! through the phonetic parser and comes out as interned segment ids, and
//! matrices compile to bit tests. Match position and output position lower
//! through separate functions because their vocabularies differ: a repeater
//! is meaningful in `from` but not in `to`, a plain matrix *tests* in `from`
//! but *updates* in `to`.
//!
//! Statement order matters beyond the rules themselves, and this pass also
//! ports lexurgy's sequencing (`LscWalker.visitRulesWithAnchoredStatements` +
//! `sequenceRules`): `Syllables:` declarations re-run after every following
//! rule, `cleanup` rules re-run until switched `off`, deromanizers run
//! first and romanizers last, and `literal` romanizer blocks run in a
//! separate "empty declarations" universe bridged by re-parses.

use std::collections::HashMap;

use smol_str::SmolStr;

use super::decls::Declarations;
use super::features::{Level, MatrixTest};
use super::ir::{
    BlockIr, CaptureSlot, Emit, EnvIr, ExprIr, MatchMode, Pattern, RuleIr, SegTest, StructuredSyl,
    SylExprIr, SylPatternIr, SyllabifierIr, TextIr,
};
use super::segments::{syl_mods_test, DiacriticMask, SegmentId, SegmentInterner};
use super::{CompileError, CompiledRules, Stage, StageKind, Step, Universe};
use crate::ast;
use crate::word::Word;

/// Lower all rule-like statements into a sequenced pipeline.
pub fn lower(
    decls: Declarations,
    segments: SegmentInterner,
    literal_decls: Declarations,
    literal_segments: SegmentInterner,
    statements: &[&ast::Statement],
) -> Result<CompiledRules, CompileError> {
    Build {
        decls,
        segments,
        literal_decls,
        literal_segments,
        rules: Vec::new(),
        syllabifiers: Vec::new(),
        deferred: HashMap::new(),
        validate_only: Vec::new(),
        syl_active: false,
    }
    .run(statements)
}

/// A rule statement that is anchored to the next plain rule rather than
/// being a pipeline stage itself (lexurgy's `AnchoredStep`).
enum Anchored {
    Cleanup {
        name: SmolStr,
        rule: usize,
    },
    CleanupOff(SmolStr),
    Syllabify(Option<usize>),
    /// Intermediate romanizers don't touch the main word stream (their
    /// output goes to a separate stage map), but their *position* matters
    /// for sequencing. `name` is the `ruleNames` entry (`<romanizer>-x`),
    /// `stage_name` the bare intermediates-map key (`x`), and `steps` the
    /// romanizer's own pipeline (run on a copy of the stream).
    InterRomanizer {
        name: SmolStr,
        stage_name: SmolStr,
        steps: Vec<Step>,
    },
}

/// One rule plus the anchored statements preceding it. `steps` is the
/// rule's own pipeline footprint: usually one `Step::Rule` + strip, empty
/// for the trailing virtual rule, several steps for literal romanizers.
struct Entry {
    /// The `ApplyRule` name for this entry's rule (`foo`, `<deromanizer>`,
    /// `<romanizer>`). Empty for the trailing virtual rule (no stage).
    name: SmolStr,
    steps: Vec<Step>,
    anchored: Vec<Anchored>,
    is_romanizer: bool,
}

struct Build {
    decls: Declarations,
    segments: SegmentInterner,
    literal_decls: Declarations,
    literal_segments: SegmentInterner,
    rules: Vec<RuleIr>,
    syllabifiers: Vec<SyllabifierIr>,
    deferred: HashMap<SmolStr, ast::ChangeRule>,
    /// Whether a syllabifier is in force at the current point in the file;
    /// decides whether `.` is a syllable boundary or plain text.
    syl_active: bool,
    /// Inter-romanizer bodies: not pipeline steps, but kotlin still builds
    /// their transformers, so `compile` must run the pairing checks on
    /// them too.
    validate_only: Vec<(SmolStr, BlockIr, Universe)>,
}

fn has_modifier(rule: &ast::ChangeRule, which: ast::RuleModifier) -> bool {
    rule.modifiers.contains(&which)
}

/// `name:\n  off` switches off a persistent cleanup rule.
fn is_off_rule(rule: &ast::ChangeRule) -> bool {
    rule.block.rest.is_empty()
        && matches!(
            &rule.block.first,
            ast::BlockElement::Expressions(exprs)
                if matches!(exprs.as_slice(), [ast::Expression::Off])
        )
}

impl Build {
    fn run(mut self, statements: &[&ast::Statement]) -> Result<CompiledRules, CompileError> {
        // Lexurgy inserts an implicit "Syllables: explicit" at the very
        // beginning when the first anchored statement is a syllable
        // declaration, so that explicit breaks in the input are preserved.
        let initial_syl = first_anchored_is_syllables(statements);
        self.syl_active = initial_syl;

        let mut deromanizer: Option<(bool, ast::Block)> = None;
        let mut romanizer: Option<(bool, ast::Block)> = None;
        let mut entries: Vec<Entry> = Vec::new();
        let mut cur_anchored: Vec<Anchored> = Vec::new();
        let mut rule_names: Vec<SmolStr> = Vec::new();

        for statement in statements {
            match statement {
                ast::Statement::Rule(rule) => {
                    if is_off_rule(rule) {
                        cur_anchored.push(Anchored::CleanupOff(rule.name.clone()));
                        continue;
                    }
                    if has_modifier(rule, ast::RuleModifier::Defer) {
                        // A deferred rule is a template, not a pipeline step: it
                        // is lowered (and thus validated) only where `:name`
                        // splices it (`lower_block_element` /
                        // `lower_expression_into`). We must NOT validate the body
                        // here. Kotlin compiles a `defer`d rule lazily at its
                        // splice, so a never-spliced deferred rule with an invalid
                        // body (a peripheral repeater, a matrix with two values of
                        // one feature, a nested env or `~$` in output, an
                        // undefined name, a mismatched `=>` count, ...) is
                        // *accepted*, not rejected — eager validation here was 18
                        // of the kotlin-oracle fuzz's findings, all unspliced
                        // deferred rules (kotlin-CLI-verified: bad-defer-unspliced
                        // accepts, the same body spliced rejects). A duplicate
                        // deferred-rule name is likewise not an error: kotlin's
                        // `resolveBlocks` does
                        // `blocks.associate { it.rule.name to it.rule }`, keeping
                        // the last, so the last declaration wins (unlike a plain
                        // rule, where a duplicate name is rejected).
                        self.deferred.insert(rule.name.clone(), (*rule).clone());
                        continue;
                    }
                    if rule_names.contains(&rule.name) {
                        return Err(CompileError::Duplicate {
                            kind: "rule",
                            name: rule.name.to_string(),
                        });
                    }
                    rule_names.push(rule.name.clone());
                    let body = self
                        .lower_rule_body(rule, Universe::Real)
                        .map_err(|e| e.in_rule(&rule.name))?;
                    let index = self.push_rule(rule.name.clone(), body);
                    if has_modifier(rule, ast::RuleModifier::Cleanup) {
                        cur_anchored.push(Anchored::Cleanup {
                            name: rule.name.clone(),
                            rule: index,
                        });
                    } else {
                        entries.push(Entry {
                            name: rule.name.clone(),
                            steps: vec![
                                Step::Rule {
                                    rule: index,
                                    universe: Universe::Real,
                                },
                                Step::StripBreaks,
                            ],
                            anchored: std::mem::take(&mut cur_anchored),
                            is_romanizer: false,
                        });
                    }
                }
                ast::Statement::Syllables(spec) => {
                    let index = self.lower_syllables(spec)?;
                    self.syl_active = index.is_some();
                    cur_anchored.push(Anchored::Syllabify(index));
                }
                ast::Statement::Deromanizer { literal, block } => {
                    if deromanizer.is_some() {
                        return Err(CompileError::Duplicate {
                            kind: "rule",
                            name: "<deromanizer>".to_string(),
                        });
                    }
                    deromanizer = Some((*literal, block.clone()));
                }
                ast::Statement::Romanizer { literal, block } => {
                    if romanizer.is_some() {
                        return Err(CompileError::Duplicate {
                            kind: "rule",
                            name: "<romanizer>".to_string(),
                        });
                    }
                    romanizer = Some((*literal, block.clone()));
                }
                ast::Statement::InterRomanizer {
                    name,
                    literal,
                    block,
                } => {
                    let rule_name = SmolStr::from(format!("<romanizer>-{name}"));
                    // Validate against the declarations in force here (kotlin
                    // builds inter-romanizer transformers at parse time, so
                    // their `from => to` count errors are compile errors).
                    self.lower_romanizer_blocks(&rule_name, *literal, block)
                        .map_err(|e| e.in_rule(&rule_name))?;
                    // The session path also *runs* the romanizer (on a copy of
                    // the stream) to capture its output, so lower it to runnable
                    // steps too. A duplicate inter-romanizer *name* is not an
                    // error: unlike the deromanizer/final romanizer (kotlin's
                    // `extractRomanizerContext` uses `singleOrNullOrThrow` →
                    // `LscDuplicateName`), intermediate romanizers are just
                    // collected as a list (`visitInterRomanizer`), so two
                    // `Romanizer-x:` stages both run.
                    let steps = self
                        .lower_romanizer_pipeline(rule_name.clone(), *literal, block)
                        .map_err(|e| e.in_rule(&rule_name))?;
                    cur_anchored.push(Anchored::InterRomanizer {
                        name: rule_name,
                        stage_name: name.clone(),
                        steps,
                    });
                }
                ast::Statement::Expression(_) => {
                    return Err(CompileError::Invalid {
                        what: "expression outside a named rule".to_string(),
                    })
                }
                _ => unreachable!("declaration statement survived resolve()"),
            }
        }

        // lexurgy throws `LscFutureStructure("Transforming elements")` for a
        // transforming `>` in *any* rule, including a never-spliced deferred
        // one (it is a compile-stage error, post-parse). The deferred bodies
        // above are otherwise validated lazily at their splice, but `>` is an
        // eager reject, so check every collected deferred rule for it now.
        for rule in self.deferred.values() {
            crate::parser::validate::reject_deferred_transforming(rule)
                .map_err(|what| CompileError::Invalid { what }.in_rule(&rule.name))?;
        }

        // Trailing anchored statements attach to a virtual rule at the end.
        entries.push(Entry {
            name: SmolStr::default(),
            steps: Vec::new(),
            anchored: cur_anchored,
            is_romanizer: false,
        });

        // The deromanizer runs first (linked against the initial
        // declarations) and the final romanizer last (final declarations).
        let mut input_universe = Universe::Real;
        if let Some((literal, block)) = &deromanizer {
            let mut steps = Vec::new();
            if *literal {
                // The first block sees the input as plain characters; input
                // words are parsed with *empty* declarations.
                input_universe = Universe::Literal;
                let (first, rest) = split_then_blocks(block);
                let body = self
                    .lower_block_in(&first, Universe::Literal)
                    .map_err(|e| e.in_rule("<deromanizer>"))?;
                let index = self.push_rule("<deromanizer>".into(), body);
                steps.push(Step::Rule {
                    rule: index,
                    universe: Universe::Literal,
                });
                steps.push(Step::Redeclare {
                    universe: Universe::Real,
                    syllabified: initial_syl,
                });
                if let Some(rest) = rest {
                    let prev = std::mem::replace(&mut self.syl_active, initial_syl);
                    let body = self
                        .lower_block_in(&rest, Universe::Real)
                        .map_err(|e| e.in_rule("<deromanizer>"))?;
                    self.syl_active = prev;
                    let index = self.push_rule("<deromanizer>".into(), body);
                    steps.push(Step::Rule {
                        rule: index,
                        universe: Universe::Real,
                    });
                }
            } else {
                let prev = std::mem::replace(&mut self.syl_active, initial_syl);
                let body = self
                    .lower_block_in(block, Universe::Real)
                    .map_err(|e| e.in_rule("<deromanizer>"))?;
                self.syl_active = prev;
                let index = self.push_rule("<deromanizer>".into(), body);
                steps.push(Step::Rule {
                    rule: index,
                    universe: Universe::Real,
                });
            }
            steps.push(Step::StripBreaks);
            entries.insert(
                0,
                Entry {
                    name: "<deromanizer>".into(),
                    steps,
                    anchored: Vec::new(),
                    is_romanizer: false,
                },
            );
        }
        if let Some((literal, block)) = &romanizer {
            let steps = self
                .lower_final_romanizer(*literal, block)
                .map_err(|e| e.in_rule("<romanizer>"))?;
            entries.push(Entry {
                name: "<romanizer>".into(),
                steps,
                anchored: Vec::new(),
                is_romanizer: true,
            });
        }

        let (steps, stages) = sequence(entries);
        Ok(CompiledRules {
            decls: self.decls,
            segments: self.segments,
            literal_decls: self.literal_decls,
            literal_segments: self.literal_segments,
            rules: self.rules,
            syllabifiers: self.syllabifiers,
            steps,
            stages,
            input_universe,
            input_syllabified: input_universe == Universe::Real && initial_syl,
            validate_only: self.validate_only,
            // filled in by `compile` once the rules are final
            syl_skippable: Vec::new(),
            fst_rules: Vec::new(),
            fused: Vec::new(),
            fused_at: std::collections::HashMap::new(),
            force_vm: false,
        })
    }

    fn push_rule(&mut self, name: SmolStr, body: BlockIr) -> usize {
        self.rules.push(RuleIr { name, body });
        self.rules.len() - 1
    }

    fn lower_rule_body(
        &mut self,
        rule: &ast::ChangeRule,
        universe: Universe,
    ) -> Result<BlockIr, CompileError> {
        let mut lowerer = self.lowerer(universe);
        let modifiers = lowerer.lower_modifiers(&rule.modifiers, true)?;
        let body = lowerer.lower_block(&rule.block)?;
        wrap_with_modifiers(body, modifiers)
    }

    fn lower_block_in(
        &mut self,
        block: &ast::Block,
        universe: Universe,
    ) -> Result<BlockIr, CompileError> {
        self.lowerer(universe).lower_block(block)
    }

    fn lower_final_romanizer(
        &mut self,
        literal: bool,
        block: &ast::Block,
    ) -> Result<Vec<Step>, CompileError> {
        self.lower_romanizer_pipeline("<romanizer>".into(), literal, block)
    }

    /// Lower a romanizer block (final or intermediate) into runnable steps.
    /// Same shape as lexurgy's romanizer rule: a literal romanizer runs its
    /// last block against empty declarations, bridged by a re-parse.
    fn lower_romanizer_pipeline(
        &mut self,
        name: SmolStr,
        literal: bool,
        block: &ast::Block,
    ) -> Result<Vec<Step>, CompileError> {
        let mut steps = Vec::new();
        if literal {
            // The *last* block runs against empty declarations.
            let (front, last) = split_then_blocks_back(block);
            if let Some(front) = front {
                let body = self.lower_block_in(&front, Universe::Real)?;
                let index = self.push_rule(name.clone(), body);
                steps.push(Step::Rule {
                    rule: index,
                    universe: Universe::Real,
                });
            }
            steps.push(Step::Redeclare {
                universe: Universe::Literal,
                syllabified: false,
            });
            let body = self.lower_block_in(&last, Universe::Literal)?;
            let index = self.push_rule(name, body);
            steps.push(Step::Rule {
                rule: index,
                universe: Universe::Literal,
            });
        } else {
            let body = self.lower_block_in(block, Universe::Real)?;
            let index = self.push_rule(name, body);
            steps.push(Step::Rule {
                rule: index,
                universe: Universe::Real,
            });
        }
        steps.push(Step::StripBreaks);
        Ok(steps)
    }

    /// Validate an intermediate romanizer's blocks. The lowered bodies are
    /// kept for the pairing checks (kotlin builds inter-romanizer
    /// transformers at parse time, so their `from => to` count errors are
    /// compile errors even though the stage output isn't part of the
    /// pipeline).
    fn lower_romanizer_blocks(
        &mut self,
        name: &SmolStr,
        literal: bool,
        block: &ast::Block,
    ) -> Result<(), CompileError> {
        if literal {
            let (front, last) = split_then_blocks_back(block);
            if let Some(front) = front {
                let body = self.lower_block_in(&front, Universe::Real)?;
                self.validate_only
                    .push((name.clone(), body, Universe::Real));
            }
            let body = self.lower_block_in(&last, Universe::Literal)?;
            self.validate_only
                .push((name.clone(), body, Universe::Literal));
        } else {
            let body = self.lower_block_in(block, Universe::Real)?;
            self.validate_only
                .push((name.clone(), body, Universe::Real));
        }
        Ok(())
    }

    fn lower_syllables(&mut self, spec: &ast::SyllableSpec) -> Result<Option<usize>, CompileError> {
        let exprs = match spec {
            ast::SyllableSpec::Clear => return Ok(None),
            ast::SyllableSpec::Explicit => Vec::new(),
            ast::SyllableSpec::Patterns(patterns) => {
                // Syllable patterns are linked against the *initial*
                // declarations (no syllabifier), so `.` inside them is text.
                let prev = std::mem::replace(&mut self.syl_active, false);
                let result: Result<Vec<SylExprIr>, CompileError> = patterns
                    .iter()
                    .map(|expr| {
                        let mut lowerer = self.lowerer(Universe::Real);
                        lowerer.lower_syllable_expr(expr)
                    })
                    .collect();
                self.syl_active = prev;
                result?
            }
        };
        self.syllabifiers.push(SyllabifierIr { exprs });
        Ok(Some(self.syllabifiers.len() - 1))
    }

    fn lowerer(&mut self, universe: Universe) -> Lowerer<'_> {
        match universe {
            Universe::Real => Lowerer::new(
                &self.decls,
                &mut self.segments,
                &self.deferred,
                self.syl_active,
            ),
            Universe::Literal => Lowerer::new(
                &self.literal_decls,
                &mut self.literal_segments,
                &self.deferred,
                false,
            ),
        }
    }
}

/// Whether the first anchored-type statement (before the first plain rule)
/// is a `Syllables:` declaration.
fn first_anchored_is_syllables(statements: &[&ast::Statement]) -> bool {
    for statement in statements {
        match statement {
            ast::Statement::Syllables(_) => return true,
            ast::Statement::InterRomanizer { .. } => return false,
            ast::Statement::Rule(rule) => {
                if is_off_rule(rule) || has_modifier(rule, ast::RuleModifier::Cleanup) {
                    return false;
                }
                if has_modifier(rule, ast::RuleModifier::Defer) {
                    continue;
                }
                return false;
            }
            _ => {}
        }
    }
    false
}

/// Split a `Then:` chain after the first block (literal deromanizers run
/// their first block in the literal universe).
fn split_then_blocks(block: &ast::Block) -> (ast::Block, Option<ast::Block>) {
    if block.rest.is_empty()
        || block
            .rest
            .iter()
            .any(|(t, _)| t.kind != ast::BlockKind::Then)
    {
        return (block.clone(), None);
    }
    let first = ast::Block {
        first: block.first.clone(),
        rest: Vec::new(),
    };
    let mut rest_iter = block.rest.iter().cloned();
    let (_, rest_first) = rest_iter.next().unwrap();
    let rest = ast::Block {
        first: rest_first,
        rest: rest_iter.collect(),
    };
    (first, Some(rest))
}

/// Split a `Then:` chain before the last block (literal romanizers run
/// their last block in the literal universe).
fn split_then_blocks_back(block: &ast::Block) -> (Option<ast::Block>, ast::Block) {
    if block.rest.is_empty()
        || block
            .rest
            .iter()
            .any(|(t, _)| t.kind != ast::BlockKind::Then)
    {
        return (None, block.clone());
    }
    let mut rest = block.rest.clone();
    let (_, last) = rest.pop().unwrap();
    let front = ast::Block {
        first: block.first.clone(),
        rest,
    };
    (
        Some(front),
        ast::Block {
            first: last,
            rest: Vec::new(),
        },
    )
}

/// Port of lexurgy's `sequenceRules`: interleave rules with persistent
/// cleanup and syllabification steps, emitting the flat [`Step`] pipeline and
/// the parallel named [`Stage`] list (lexurgy's `sequencedRules`, whose names
/// are `SoundChanger.ruleNames`).
fn sequence(mut entries: Vec<Entry>) -> (Vec<Step>, Vec<Stage>) {
    // Unless the last entry is a romanizer (or already a virtual rule),
    // append a virtual rule so trailing persistent steps run at the end.
    let needs_trailing = entries
        .last()
        .is_some_and(|e| !e.steps.is_empty() && !e.is_romanizer);
    if needs_trailing {
        entries.push(Entry {
            name: SmolStr::default(),
            steps: Vec::new(),
            anchored: Vec::new(),
            is_romanizer: false,
        });
    }

    let mut seq = Seq {
        steps: Vec::new(),
        stages: Vec::new(),
        persistent_cleanups: Vec::new(),
        persistent_syllabify: None,
        // lexurgy seeds `lastRuleName` with "<initial>" and resets the
        // syllable counter to 0 after each applied rule.
        last_rule_name: SmolStr::from("<initial>"),
        syl_counter: 0,
    };

    for entry in entries {
        // Persistent cleanup rules always run first: before any
        // syllabification, and one last time before being cancelled.
        for (name, rule) in seq.persistent_cleanups.clone() {
            seq.push_cleanup(&name, rule);
        }
        let before_syllabification = if matches!(
            entry.anchored.first(),
            Some(Anchored::InterRomanizer { .. })
        ) {
            0
        } else {
            entry.anchored.len().min(1)
        };
        for anchored in &entry.anchored[..before_syllabification] {
            seq.apply_anchored(anchored);
        }
        if let Some(syllabify) = seq.persistent_syllabify {
            if seq.steps.last() != Some(&syllabify) {
                let Step::Syllabify(index) = syllabify else {
                    unreachable!("persistent_syllabify is always a Syllabify step")
                };
                seq.push_syllabify(index);
            }
        }
        for anchored in &entry.anchored[before_syllabification..] {
            seq.apply_anchored(anchored);
        }
        // A non-empty entry is a real `ApplyRule` (rule/deromanizer/romanizer):
        // emit one Rule stage spanning its steps and advance `lastRuleName`.
        // The trailing virtual rule (empty steps) emits nothing.
        if !entry.steps.is_empty() {
            let start = seq.steps.len();
            seq.steps.extend(entry.steps);
            seq.stages.push(Stage {
                name: entry.name.clone(),
                kind: StageKind::Rule {
                    steps: start..seq.steps.len(),
                },
            });
            seq.last_rule_name = entry.name;
            seq.syl_counter = 0;
        }
    }
    (seq.steps, seq.stages)
}

/// Mutable state threaded through [`sequence`], so cleanup/syllabify pushes
/// can name their stages with the current `lastRuleName` and counter.
struct Seq {
    steps: Vec<Step>,
    stages: Vec<Stage>,
    persistent_cleanups: Vec<(SmolStr, usize)>,
    persistent_syllabify: Option<Step>,
    last_rule_name: SmolStr,
    syl_counter: u32,
}

impl Seq {
    /// Push a cleanup rule application (first declaration or persistent
    /// re-run) and its `<cleanup>/<lastRule>/<name>` stage.
    fn push_cleanup(&mut self, name: &SmolStr, rule: usize) {
        let start = self.steps.len();
        self.steps.push(Step::Rule {
            rule,
            universe: Universe::Real,
        });
        self.steps.push(Step::StripBreaks);
        self.stages.push(Stage {
            name: SmolStr::from(format!("<cleanup>/{}/{name}", self.last_rule_name)),
            kind: StageKind::Cleanup {
                steps: start..self.steps.len(),
            },
        });
    }

    /// Push a syllabification step and its `<syllables>/<lastRule>/<n>` stage
    /// (the counter advances on every syllabify since the last applied rule,
    /// including persistent re-runs).
    fn push_syllabify(&mut self, index: Option<usize>) {
        let start = self.steps.len();
        self.steps.push(Step::Syllabify(index));
        self.syl_counter += 1;
        self.stages.push(Stage {
            name: SmolStr::from(format!(
                "<syllables>/{}/{}",
                self.last_rule_name, self.syl_counter
            )),
            kind: StageKind::Syllabify {
                steps: start..self.steps.len(),
            },
        });
    }

    fn apply_anchored(&mut self, anchored: &Anchored) {
        match anchored {
            Anchored::Cleanup { name, rule } => {
                self.push_cleanup(name, *rule);
                self.persistent_cleanups.push((name.clone(), *rule));
            }
            Anchored::CleanupOff(name) => {
                self.persistent_cleanups.retain(|(n, _)| n != name);
            }
            Anchored::Syllabify(index) => {
                self.push_syllabify(*index);
                self.persistent_syllabify = Some(Step::Syllabify(*index));
            }
            Anchored::InterRomanizer {
                name,
                stage_name,
                steps,
            } => {
                // Side branch: no main `steps` are pushed; the romanizer's own
                // steps are carried on the stage and run on a copy of the
                // stream by the session.
                self.stages.push(Stage {
                    name: name.clone(),
                    kind: StageKind::IntermediateRomanize {
                        stage_name: stage_name.clone(),
                        steps: steps.clone(),
                    },
                });
            }
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Edge {
    Leading,
    Trailing,
    Interior,
}

/// Resolve positional elements in environments. `$` (lexurgy's
/// `WordBoundaryElement`) becomes the start of the word at the leading edge
/// of a *before* environment and the end at the trailing edge of an *after*
/// environment; anywhere else it's an error (`LscInteriorWordBoundary`).
/// `!x` at either edge becomes the zero-width `NegatedLookaroundMatcher`.
/// Boundaries in input position stay unresolved; the VM rejects them if a
/// rule ever tries to match one.
fn resolve_boundaries(pattern: Pattern, edge: Edge) -> Result<Pattern, CompileError> {
    Ok(match pattern {
        Pattern::WordBoundary => match edge {
            Edge::Leading => Pattern::WordStart,
            Edge::Trailing => Pattern::WordEnd,
            Edge::Interior => {
                return Err(CompileError::Invalid {
                    what: "a word boundary in the middle of an environment".to_string(),
                })
            }
        },
        Pattern::Seq(patterns) => {
            let last = patterns.len().saturating_sub(1);
            Pattern::Seq(
                patterns
                    .into_iter()
                    .enumerate()
                    .map(|(i, p)| {
                        let child_edge = match edge {
                            Edge::Leading if i == 0 => Edge::Leading,
                            Edge::Trailing if i == last => Edge::Trailing,
                            _ => Edge::Interior,
                        };
                        resolve_boundaries(p, child_edge)
                    })
                    .collect::<Result<_, _>>()?,
            )
        }
        Pattern::Alt(patterns) => Pattern::Alt(
            patterns
                .into_iter()
                .map(|p| resolve_boundaries(p, edge))
                .collect::<Result<_, _>>()?,
        ),
        // "Things inside an intersection have to consume what they match!"
        // lexurgy resets the context (`ElementContext.aloneInMain`), so
        // negation inside `&` is never a lookaround and `$` is illegal.
        Pattern::Intersect(patterns) => Pattern::Intersect(
            patterns
                .into_iter()
                .map(|p| resolve_boundaries(p, Edge::Interior))
                .collect::<Result<_, _>>()?,
        ),
        Pattern::Capture { inner, slot } => Pattern::Capture {
            inner: Box::new(resolve_boundaries(*inner, edge)?),
            slot,
        },
        Pattern::Look {
            inner,
            condition,
            exclusion,
        } => Pattern::Look {
            inner: Box::new(resolve_boundaries(*inner, edge)?),
            // nested environments were already resolved when lowered
            condition,
            exclusion,
        },
        Pattern::Repeat { inner, min, max } => Pattern::Repeat {
            inner: Box::new(resolve_boundaries(*inner, Edge::Interior)?),
            min,
            max,
        },
        // `!x` at the edge of an environment is a zero-width lookaround
        // negation in lexurgy; `!.` was already special-cased in lowering.
        Pattern::Not(inner) if edge != Edge::Interior => {
            Pattern::NotAhead(Box::new(resolve_boundaries(*inner, edge)?))
        }
        Pattern::Not(inner) => Pattern::Not(Box::new(resolve_boundaries(*inner, Edge::Interior)?)),
        other => other,
    })
}

#[derive(Default)]
struct Modifiers {
    filter: Option<Pattern>,
    mode: MatchMode,
    propagate: bool,
}

/// Wrap a block with its modifiers, in lexurgy's order
/// (`wrapWithModifierBlocks`): match mode reaches into the leaf, then
/// propagate wraps, then the filter wraps outermost.
fn wrap_with_modifiers(body: BlockIr, m: Modifiers) -> Result<BlockIr, CompileError> {
    let body = if m.mode != MatchMode::Simultaneous {
        match body {
            BlockIr::Exprs { exprs, .. } => BlockIr::Exprs {
                mode: m.mode,
                exprs,
            },
            _ => {
                return Err(CompileError::Invalid {
                    what: "ltr/rtl can't apply to a block".to_string(),
                })
            }
        }
    } else {
        body
    };
    let body = if m.propagate {
        BlockIr::Propagate(Box::new(body))
    } else {
        body
    };
    Ok(match m.filter {
        Some(filter) => BlockIr::Filter {
            filter,
            inner: Box::new(body),
        },
        None => body,
    })
}

struct Lowerer<'a> {
    decls: &'a Declarations,
    segments: &'a mut SegmentInterner,
    deferred: &'a HashMap<SmolStr, ast::ChangeRule>,
    /// Whether `.` is a syllable boundary (a syllabifier is in force).
    syl_active: bool,
    /// Mask of declared floating diacritics; non-exact literals match
    /// modulo these.
    floating: DiacriticMask,
}

impl<'a> Lowerer<'a> {
    fn new(
        decls: &'a Declarations,
        segments: &'a mut SegmentInterner,
        deferred: &'a HashMap<SmolStr, ast::ChangeRule>,
        syl_active: bool,
    ) -> Self {
        let floating = decls
            .diacritics
            .iter()
            .enumerate()
            .filter(|(_, d)| d.floating)
            .fold(0, |mask, (i, _)| mask | (1 << i));
        Lowerer {
            decls,
            segments,
            deferred,
            syl_active,
            floating,
        }
    }

    /// A `Then:`/`Else:` chain. Lexurgy requires the chain to be uniform
    /// (`LscMixedBlock`); each arm carries its own modifiers, the first arm
    /// implicitly none.
    fn lower_block(&mut self, block: &ast::Block) -> Result<BlockIr, CompileError> {
        let first = self.lower_block_element(&block.first)?;
        if block.rest.is_empty() {
            return Ok(first);
        }
        let kind = block.rest[0].0.kind;
        if block.rest.iter().any(|(t, _)| t.kind != kind) {
            return Err(CompileError::Invalid {
                what: "can't mix Then: and Else: in one block".to_string(),
            });
        }
        let mut children = vec![first];
        for (block_type, element) in &block.rest {
            let modifiers = self.lower_modifiers(&block_type.modifiers, false)?;
            let child = self.lower_block_element(element)?;
            children.push(wrap_with_modifiers(child, modifiers)?);
        }
        Ok(match kind {
            ast::BlockKind::Then => BlockIr::Sequential(children),
            ast::BlockKind::Else => BlockIr::FirstMatching(children),
        })
    }

    fn lower_block_element(
        &mut self,
        element: &ast::BlockElement,
    ) -> Result<BlockIr, CompileError> {
        match element {
            ast::BlockElement::Nested(block) => self.lower_block(block),
            ast::BlockElement::Expressions(expressions) => {
                // A deferred rule referenced as the *only* expression may be
                // a whole complex block; mixed in with other expressions it
                // must be a plain expression list and is inlined.
                if let [ast::Expression::BlockRef(name)] = expressions.as_slice() {
                    let rule = self.deferred_rule(name)?;
                    // Validate the deferred body now that it's actually spliced
                    // (kotlin compiles `defer`d rules at their reference). Nested
                    // `:name` refs inside it recurse through here, so transitive
                    // splices validate too.
                    crate::parser::validate::check_deferred_rule(&rule, true)
                        .map_err(|what| CompileError::Invalid { what })?;
                    let modifiers = self.lower_modifiers(&rule.modifiers, true)?;
                    let body = self.lower_block(&rule.block)?;
                    return wrap_with_modifiers(body, modifiers);
                }
                let mut exprs = Vec::new();
                for expression in expressions {
                    self.lower_expression_into(expression, &mut exprs)?;
                }
                Ok(BlockIr::Exprs {
                    mode: MatchMode::Simultaneous,
                    exprs,
                })
            }
        }
    }

    fn lower_expression_into(
        &mut self,
        expression: &ast::Expression,
        exprs: &mut Vec<ExprIr>,
    ) -> Result<(), CompileError> {
        match expression {
            // `unchanged` contributes nothing: zero expressions is the
            // identity rule.
            ast::Expression::Unchanged => {}
            ast::Expression::Off => {
                return Err(CompileError::Invalid {
                    what: "\"off\" outside a cleanup-off rule".to_string(),
                })
            }
            ast::Expression::BlockRef(name) => {
                let rule = self.deferred_rule(name)?;
                crate::parser::validate::check_deferred_rule(&rule, true)
                    .map_err(|what| CompileError::Invalid { what })?;
                if !rule
                    .modifiers
                    .iter()
                    .all(|m| matches!(m, ast::RuleModifier::Defer))
                {
                    return Err(CompileError::Invalid {
                        what: format!(
                            "complex block reference \":{name}\" in a group of expressions"
                        ),
                    });
                }
                match (&rule.block.first, rule.block.rest.is_empty()) {
                    (ast::BlockElement::Expressions(inner), true) => {
                        let inner = inner.clone();
                        for expression in &inner {
                            self.lower_expression_into(expression, exprs)?;
                        }
                    }
                    _ => {
                        return Err(CompileError::Invalid {
                            what: format!(
                                "complex block reference \":{name}\" in a group of expressions"
                            ),
                        })
                    }
                }
            }
            ast::Expression::Standard(expression) => {
                exprs.push(self.lower_expr(expression)?);
            }
        }
        Ok(())
    }

    fn deferred_rule(&self, name: &SmolStr) -> Result<ast::ChangeRule, CompileError> {
        self.deferred
            .get(name)
            .cloned()
            .ok_or_else(|| CompileError::Undefined {
                kind: "block",
                name: name.to_string(),
            })
    }

    fn lower_modifiers(
        &mut self,
        modifiers: &[ast::RuleModifier],
        rule_level: bool,
    ) -> Result<Modifiers, CompileError> {
        let mut lowered = Modifiers::default();
        for modifier in modifiers {
            match modifier {
                ast::RuleModifier::Filter(element) => {
                    lowered.filter = Some(self.lower_match(element)?);
                }
                ast::RuleModifier::Ltr => lowered.mode = MatchMode::Ltr,
                ast::RuleModifier::Rtl => lowered.mode = MatchMode::Rtl,
                ast::RuleModifier::Propagate => lowered.propagate = true,
                // `cleanup` and `defer` are handled by the sequencing pass;
                // they're only valid at rule level (lexurgy checks this in
                // `visitBlock`).
                ast::RuleModifier::Defer | ast::RuleModifier::Cleanup => {
                    if !rule_level {
                        return Err(CompileError::Invalid {
                            what: "cleanup/defer is not valid on a block".to_string(),
                        });
                    }
                }
                ast::RuleModifier::Name(name) => {
                    return Err(CompileError::Invalid {
                        what: format!("unknown rule modifier \"{name}\""),
                    })
                }
            }
        }
        Ok(lowered)
    }

    // syllabification patterns

    fn lower_syllable_expr(
        &mut self,
        expr: &ast::SyllableExpression,
    ) -> Result<SylExprIr, CompileError> {
        let assign = match &expr.assign {
            None => None,
            Some(matrix) => {
                let update = self.decls.features.compile_update(matrix)?;
                if update.seg_mask != 0 || !update.vars.is_empty() {
                    return Err(CompileError::Invalid {
                        what: "syllable pattern matrices must use syllable-level features"
                            .to_string(),
                    });
                }
                Some(update)
            }
        };
        let pattern = match &expr.pattern {
            ast::SyllablePattern::Plain(element) => {
                let inner = self.lower_rule_element(element)?;
                let wrapped = self.wrap_env(inner, expr.environment.as_ref())?;
                SylPatternIr::Simple(wrapped)
            }
            ast::SyllablePattern::Structured {
                reluctant_onset,
                parts,
            } => {
                let reluctant_onset = reluctant_onset
                    .as_ref()
                    .map(|e| self.lower_match(e))
                    .transpose()?;
                let mut lowered: Vec<Pattern> = parts
                    .iter()
                    .map(|e| self.lower_match(e))
                    .collect::<Result<_, _>>()?;
                let coda = if lowered.len() > 2 {
                    lowered.pop()
                } else {
                    None
                };
                let nucleus = lowered.pop().ok_or_else(|| CompileError::Invalid {
                    what: "syllable pattern needs an onset and a nucleus".to_string(),
                })?;
                let onset = lowered.pop().ok_or_else(|| CompileError::Invalid {
                    what: "syllable pattern needs an onset and a nucleus".to_string(),
                })?;
                let (condition, exclusion) = match &expr.environment {
                    Some(compound) => (
                        self.lower_envs(compound.condition.as_deref())?,
                        self.lower_envs(compound.exclusion.as_deref())?,
                    ),
                    None => (Vec::new(), Vec::new()),
                };
                SylPatternIr::Structured(Box::new(StructuredSyl {
                    reluctant_onset,
                    onset,
                    nucleus,
                    coda,
                    condition,
                    exclusion,
                }))
            }
        };
        Ok(SylExprIr { pattern, assign })
    }

    // expressions

    fn lower_expr(&mut self, expr: &ast::StandardExpression) -> Result<ExprIr, CompileError> {
        // An environment on the from-element (`a / x _ => b`) is a local
        // lookaround on the whole from; one after the to-element is the
        // expression-level condition. Both can appear at once.
        let from = self.lower_rule_element(&expr.from)?;
        let to = self.lower_emit(&expr.to)?;
        let (condition, exclusion) = match &expr.environment {
            Some(compound) => (
                self.lower_envs(compound.condition.as_deref())?,
                self.lower_envs(compound.exclusion.as_deref())?,
            ),
            None => (Vec::new(), Vec::new()),
        };
        Ok(ExprIr {
            from,
            to,
            condition,
            exclusion,
        })
    }

    fn lower_envs(
        &mut self,
        environments: Option<&[ast::Environment]>,
    ) -> Result<Vec<EnvIr>, CompileError> {
        environments
            .unwrap_or_default()
            .iter()
            .map(|env| {
                Ok(EnvIr {
                    before: env
                        .before
                        .as_ref()
                        .map(|e| {
                            self.lower_match(e)
                                .and_then(|p| resolve_boundaries(p, Edge::Leading))
                        })
                        .transpose()?,
                    after: env
                        .after
                        .as_ref()
                        .map(|e| {
                            self.lower_match(e)
                                .and_then(|p| resolve_boundaries(p, Edge::Trailing))
                        })
                        .transpose()?,
                    anchored: env.anchored,
                })
            })
            .collect()
    }

    // match position

    /// An element plus its optional local environment.
    fn lower_rule_element(&mut self, element: &ast::RuleElement) -> Result<Pattern, CompileError> {
        let inner = self.lower_match(&element.element)?;
        self.wrap_env(inner, element.environment.as_ref())
    }

    fn wrap_env(
        &mut self,
        inner: Pattern,
        environment: Option<&ast::CompoundEnvironment>,
    ) -> Result<Pattern, CompileError> {
        let Some(compound) = environment else {
            return Ok(inner);
        };
        Ok(Pattern::Look {
            inner: Box::new(inner),
            condition: self.lower_envs(compound.condition.as_deref())?,
            exclusion: self.lower_envs(compound.exclusion.as_deref())?,
        })
    }

    fn lower_match(&mut self, element: &ast::Element) -> Result<Pattern, CompileError> {
        Ok(match element {
            ast::Element::Sequence(elements) => Pattern::Seq(
                elements
                    .iter()
                    .map(|e| self.lower_match(e))
                    .collect::<Result<_, _>>()?,
            ),
            ast::Element::Group(inner) => self.lower_rule_element(inner)?,
            ast::Element::List(items) => {
                let mut alternatives: Vec<Pattern> = items
                    .iter()
                    .map(|item| match item {
                        ast::ListItem::Element(element) => self.lower_rule_element(element),
                        ast::ListItem::Env(_) => Err(CompileError::Invalid {
                            what: "environment list used as an element".to_string(),
                        }),
                    })
                    .collect::<Result<_, _>>()?;
                // one-element lists act like the element itself
                if alternatives.len() == 1 {
                    alternatives.pop().unwrap()
                } else {
                    Pattern::Alt(alternatives)
                }
            }
            ast::Element::Interfix { first, rest } => {
                let mut parts = vec![self.lower_match(first)?];
                for (kind, element) in rest {
                    match kind {
                        ast::InterfixKind::Intersection => parts.push(self.lower_match(element)?),
                        ast::InterfixKind::IntersectionNot => {
                            parts.push(Pattern::Not(Box::new(self.lower_match(element)?)))
                        }
                        ast::InterfixKind::Transforming => {
                            // lexurgy's `LscFutureStructure("Transforming
                            // elements")`: valid syntax, not yet implemented
                            // (by lexurgy either). Both reject.
                            return Err(CompileError::Invalid {
                                what: "transforming interfix (>) is not yet implemented"
                                    .to_string(),
                            });
                        }
                    }
                }
                Pattern::Intersect(parts)
            }
            // `!.` negates the syllable boundary specially (zero-width),
            // whether or not a syllabifier is active.
            ast::Element::Negated(inner) if matches!(**inner, ast::Element::SyllableBoundary) => {
                Pattern::NoBoundary
            }
            ast::Element::Negated(inner) => Pattern::Not(Box::new(self.lower_match(inner)?)),
            ast::Element::Capture { element, capture } => Pattern::Capture {
                inner: Box::new(self.lower_match(element)?),
                slot: self.capture_slot(capture)?,
            },
            ast::Element::Repeat { element, kind } => {
                let (min, max) = match *kind {
                    ast::RepeaterKind::ZeroOrMore => (0, None),
                    ast::RepeaterKind::AtLeastOne => (1, None),
                    ast::RepeaterKind::Optional => (0, Some(1)),
                    ast::RepeaterKind::Count(n) => (n, Some(n)),
                    ast::RepeaterKind::Range { min, max } => (min.unwrap_or(0), max),
                };
                Pattern::Repeat {
                    inner: Box::new(self.lower_match(element)?),
                    min,
                    max,
                }
            }
            ast::Element::Text(text) => self.lower_text_match(text)?,
            ast::Element::Matrix(matrix) => self.lower_matrix_match(matrix)?,
            ast::Element::ElementRef(name) => self.lower_ref_match(name)?,
            ast::Element::CaptureRef(capture) => Pattern::CaptureRef {
                slot: self.capture_slot(capture)?,
                inexact: capture.inexact,
            },
            ast::Element::Empty => Pattern::Empty,
            ast::Element::SyllableBoundary => {
                if self.syl_active {
                    Pattern::SyllableBoundary
                } else {
                    self.lower_text_match(&ast::Text {
                        text: ".".into(),
                        exact: false,
                    })?
                }
            }
            ast::Element::WordBoundary => Pattern::WordBoundary,
            ast::Element::BetweenWords => Pattern::BetweenWords,
            ast::Element::AnySyllable => Pattern::AnySyllable,
        })
    }

    /// Split a matrix by feature level, mirroring lexurgy's `MatrixElement`:
    /// segment-level values make a plain matcher, syllable-level values a
    /// syllable matcher; both together intersect (with the syllable part as
    /// a length-hinted verifier).
    fn lower_matrix_match(&mut self, matrix: &ast::Matrix) -> Result<Pattern, CompileError> {
        let test = self.decls.features.compile_test(matrix)?;
        let (syl_vars, seg_vars): (Vec<_>, Vec<_>) = test
            .vars
            .iter()
            .copied()
            .partition(|v| self.decls.features.def(v.feature).level == Level::Syllable);
        let has_syl = !test.syl.is_trivial() || !syl_vars.is_empty();
        let has_seg = !test.seg.is_trivial() || !seg_vars.is_empty();
        let seg_test = MatrixTest {
            seg: test.seg.clone(),
            syl: Default::default(),
            vars: seg_vars,
        };
        let syl_test = MatrixTest {
            seg: Default::default(),
            syl: test.syl.clone(),
            vars: syl_vars,
        };
        Ok(if !has_syl {
            Pattern::Test(SegTest::Matrix(seg_test))
        } else if !has_seg {
            Pattern::Test(SegTest::SylMatrix(syl_test))
        } else {
            Pattern::Intersect(vec![
                Pattern::Test(SegTest::Matrix(seg_test)),
                Pattern::Test(SegTest::SylMatrix(syl_test)),
            ])
        })
    }

    fn lower_text_match(&mut self, text: &ast::Text) -> Result<Pattern, CompileError> {
        let word = self.parse_text(text)?;
        let tests = word
            .segs
            .iter()
            .map(|&id| self.seg_test(id, text.exact))
            .collect();
        // Exact text is lexurgy's `TextMatcher`, which compares segments
        // only; non-exact `SymbolMatcher` also tests the text's own
        // syllable features against the matched span.
        let (syl_mask, syl_want) = if text.exact {
            (0, 0)
        } else {
            self.text_syl_test(&word)
        };
        Ok(Pattern::Text(TextIr {
            tests,
            exact: text.exact,
            syl_mask,
            syl_want,
        }))
    }

    /// The syllable-feature test carried by a pattern text's own syllable
    /// diacritics (`Word.syllableMatrix`): modifiers folded across syllables
    /// in order, default-valued fields stripped.
    fn text_syl_test(&self, word: &Word) -> (u128, u128) {
        match &word.syl {
            None => (0, 0),
            Some(syl) => {
                let all: Vec<u8> = syl.mods.values().flatten().copied().collect();
                syl_mods_test(self.decls, &all)
            }
        }
    }

    /// The test a literal segment compiles to: id equality, or (when
    /// floating diacritics exist and the text isn't `!`-exact) equality
    /// modulo extra floating diacritics on the word's segment.
    fn seg_test(&self, id: SegmentId, exact: bool) -> SegTest {
        if exact || self.floating == 0 {
            SegTest::Exact(id)
        } else {
            let data = self.segments.get(id);
            SegTest::Literal {
                id,
                core: data.core,
                required: data.diacritics,
                allowed: data.diacritics | self.floating,
            }
        }
    }

    fn lower_ref_match(&mut self, name: &SmolStr) -> Result<Pattern, CompileError> {
        if let Some(element) = self.decls.elements.get(name) {
            let element = element.clone();
            return self.lower_rule_element(&element);
        }
        if let Some(members) = self.decls.classes.get(name) {
            let members = members.clone();
            let alternatives = members
                .iter()
                .map(|text| self.lower_text_match(text))
                .collect::<Result<_, _>>()?;
            return Ok(Pattern::Alt(alternatives));
        }
        Err(CompileError::Undefined {
            kind: "class or element",
            name: name.to_string(),
        })
    }

    // output position

    fn lower_emit(&mut self, element: &ast::Element) -> Result<Emit, CompileError> {
        Ok(match element {
            ast::Element::Sequence(elements) => Emit::Seq(
                elements
                    .iter()
                    .map(|e| self.lower_emit(e))
                    .collect::<Result<_, _>>()?,
            ),
            ast::Element::Group(inner) => {
                self.no_local_env(inner)?;
                self.lower_emit(&inner.element)?
            }
            ast::Element::List(items) => {
                let mut alternatives: Vec<Emit> = items
                    .iter()
                    .map(|item| self.lower_emit(self.list_element(item)?))
                    .collect::<Result<_, _>>()?;
                // one-element lists act like the element itself
                if alternatives.len() == 1 {
                    alternatives.pop().unwrap()
                } else {
                    Emit::Alt(alternatives)
                }
            }
            ast::Element::Text(text) => self.lower_text_emit(text)?,
            ast::Element::Matrix(matrix) => {
                Emit::Matrix(self.decls.features.compile_update(matrix)?)
            }
            ast::Element::ElementRef(name) => self.lower_ref_emit(name)?,
            ast::Element::CaptureRef(capture) => {
                // `~$1` is a matcher-only construct (lexurgy's
                // `CaptureReferenceElement.emitter` → `LscIllegalStructureInOutput`)
                if capture.inexact {
                    return Err(CompileError::Invalid {
                        what: "an inexact capture reference (~$) can't be used in the output of a rule"
                            .to_string(),
                    });
                }
                if capture.syllable {
                    Emit::SylCaptureRef {
                        slot: CaptureSlot(capture.number - 1),
                    }
                } else {
                    Emit::CaptureRef {
                        slot: self.capture_slot(capture)?,
                        inexact: capture.inexact,
                    }
                }
            }
            ast::Element::Empty => Emit::Empty,
            ast::Element::SyllableBoundary => {
                if self.syl_active {
                    Emit::SyllableBoundary
                } else {
                    self.lower_text_emit(&ast::Text {
                        text: ".".into(),
                        exact: false,
                    })?
                }
            }
            ast::Element::BetweenWords => Emit::WordBreak,
            ast::Element::Interfix { .. }
            | ast::Element::Negated(_)
            | ast::Element::Capture { .. }
            | ast::Element::Repeat { .. }
            | ast::Element::WordBoundary
            | ast::Element::AnySyllable => {
                return Err(CompileError::Invalid {
                    what: format!("{element:?} is not valid in output position"),
                })
            }
        })
    }

    fn lower_text_emit(&mut self, text: &ast::Text) -> Result<Emit, CompileError> {
        let word = self.parse_text(text)?;
        let (syl_mask, syl_want) = self.text_syl_test(&word);
        Ok(Emit::Text {
            word,
            exact: text.exact,
            syl_mask,
            syl_want,
        })
    }

    fn lower_ref_emit(&mut self, name: &SmolStr) -> Result<Emit, CompileError> {
        if let Some(element) = self.decls.elements.get(name) {
            self.no_local_env(element)?;
            let element = element.element.clone();
            return self.lower_emit(&element);
        }
        if let Some(members) = self.decls.classes.get(name) {
            let members = members.clone();
            let alternatives = members
                .iter()
                .map(|text| self.lower_text_emit(text))
                .collect::<Result<_, _>>()?;
            return Ok(Emit::Alt(alternatives));
        }
        Err(CompileError::Undefined {
            kind: "class or element",
            name: name.to_string(),
        })
    }

    // shared helpers

    fn parse_text(&mut self, text: &ast::Text) -> Result<Word, CompileError> {
        self.segments
            .parse_word(self.decls, &text.text, self.syl_active)
            .map_err(|e| CompileError::Invalid {
                what: e.to_string(),
            })
    }

    fn capture_slot(&self, capture: &ast::CaptureRef) -> Result<CaptureSlot, CompileError> {
        if capture.syllable {
            return Err(CompileError::Invalid {
                what: "a syllable capture reference ($.n) can't appear in match position"
                    .to_string(),
            });
        }
        Ok(CaptureSlot(capture.number - 1))
    }

    fn list_element<'e>(&self, item: &'e ast::ListItem) -> Result<&'e ast::Element, CompileError> {
        match item {
            ast::ListItem::Element(rule_element) => {
                self.no_local_env(rule_element)?;
                Ok(&rule_element.element)
            }
            // validate() rejects anchored items outside environment lists;
            // an Env here means the list is an environment list in element
            // position, which has no meaning.
            ast::ListItem::Env(_) => Err(CompileError::Invalid {
                what: "environment list used as an element".to_string(),
            }),
        }
    }

    /// Lexurgy's `EnvironmentElement` is not a `ResultElement`, so an
    /// element with an attached environment is rejected anywhere in output
    /// position (`castToResultElement` → `LscIllegalStructureInOutput`).
    fn no_local_env(&self, element: &ast::RuleElement) -> Result<(), CompileError> {
        if element.environment.is_some() {
            return Err(CompileError::Invalid {
                what: "a nested environment can't be used in the output of a rule".to_string(),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse;

    fn compile_rules(source: &str) -> Vec<RuleIr> {
        let statements = parse(source).expect("parse failed");
        crate::compiler::compile(&statements)
            .expect("compile failed")
            .rules
    }

    /// The expressions of the first leaf, descending through wrappers.
    fn leaf(block: &BlockIr) -> &[ExprIr] {
        match block {
            BlockIr::Exprs { exprs, .. } => exprs,
            BlockIr::Filter { inner, .. } | BlockIr::Propagate(inner) => leaf(inner),
            BlockIr::Sequential(children) | BlockIr::FirstMatching(children) => leaf(&children[0]),
        }
    }

    #[test]
    fn basic_rule() {
        let rules = compile_rules("shift:\n  a => e\n");
        assert_eq!(rules.len(), 1);
        let rule = &rules[0];
        assert_eq!(rule.name, "shift");
        let exprs = leaf(&rule.body);
        assert_eq!(exprs.len(), 1);
        assert!(matches!(&exprs[0].from, Pattern::Text(text)
            if matches!(text.tests[..], [SegTest::Exact(_)])));
        assert!(matches!(&exprs[0].to, Emit::Text { word, .. } if word.len() == 1));
    }

    #[test]
    fn class_ref_becomes_alternation() {
        let rules = compile_rules(
            "Class stop {p, t, k}\n\
             lenite:\n  @stop => h / a _ a\n",
        );
        let expr = &leaf(&rules[0].body)[0];
        assert!(matches!(&expr.from, Pattern::Alt(alts) if alts.len() == 3));
        assert_eq!(expr.condition.len(), 1);
        let env = &expr.condition[0];
        assert!(env.before.is_some() && env.after.is_some());
    }

    #[test]
    fn matrix_rule_with_env() {
        let rules = compile_rules(
            "Feature Type(*cons, vowel)\n\
             Feature +stressed\n\
             Symbol a [vowel]\n\
             stress-initial:\n  [vowel] => [+stressed] / $ _\n",
        );
        let expr = &leaf(&rules[0].body)[0];
        assert!(matches!(&expr.from, Pattern::Test(SegTest::Matrix(_))));
        assert!(matches!(&expr.to, Emit::Matrix(_)));
        let env = &expr.condition[0];
        assert!(matches!(env.before, Some(Pattern::WordStart)));
        assert!(env.after.is_none());
    }

    #[test]
    fn modifiers_and_filter() {
        let rules = compile_rules(
            "Feature Type(*cons, vowel)\n\
             Symbol a [vowel]\n\
             harmony [vowel] rtl propagate:\n  a => a\n",
        );
        // lexurgy wrap order: filter outermost, then propagate, mode on leaf
        match &rules[0].body {
            BlockIr::Filter { filter, inner } => {
                assert!(matches!(filter, Pattern::Test(SegTest::Matrix(_))));
                match inner.as_ref() {
                    BlockIr::Propagate(inner) => {
                        assert!(matches!(
                            inner.as_ref(),
                            BlockIr::Exprs {
                                mode: MatchMode::Rtl,
                                ..
                            }
                        ));
                    }
                    other => panic!("expected Propagate, got {other:?}"),
                }
            }
            other => panic!("expected Filter, got {other:?}"),
        }
    }

    #[test]
    fn then_and_else_blocks() {
        let rules = compile_rules(
            "chain:\n  a => e\n  Then:\n  e => i\n\
             alternative:\n  a => e\n  Else:\n  e => i\n",
        );
        assert!(matches!(&rules[0].body, BlockIr::Sequential(c) if c.len() == 2));
        assert!(matches!(&rules[1].body, BlockIr::FirstMatching(c) if c.len() == 2));
    }

    #[test]
    fn mixed_blocks_rejected() {
        let statements =
            parse("bad:\n  a => e\n  Then:\n  e => i\n  Else:\n  i => u\n").expect("parse failed");
        assert!(matches!(
            crate::compiler::compile(&statements),
            Err(CompileError::Invalid { .. })
        ));
    }

    #[test]
    fn captures_and_repeaters() {
        let rules = compile_rules("reduplicate:\n  (a b?)$1 => $1 $1\n");
        let expr = &leaf(&rules[0].body)[0];
        match &expr.from {
            Pattern::Capture { inner, slot } => {
                assert_eq!(*slot, CaptureSlot(0));
                match inner.as_ref() {
                    Pattern::Seq(parts) => {
                        assert!(matches!(
                            &parts[1],
                            Pattern::Repeat {
                                min: 0,
                                max: Some(1),
                                ..
                            }
                        ));
                    }
                    other => panic!("expected Seq, got {other:?}"),
                }
            }
            other => panic!("expected Capture, got {other:?}"),
        }
        assert!(matches!(&expr.to, Emit::Seq(parts) if parts.len() == 2));
    }

    #[test]
    fn deletion_and_epenthesis() {
        let rules = compile_rules(
            "drop-final:\n  a => * / _ $\n\
             insert:\n  * => a / t _ t\n",
        );
        assert!(matches!(leaf(&rules[0].body)[0].to, Emit::Empty));
        assert!(matches!(leaf(&rules[1].body)[0].from, Pattern::Empty));
    }

    #[test]
    fn syllables_sequence_after_every_rule() {
        let compiled = crate::compiler::compile(
            &parse(
                "Syllables:\n  explicit\n\
                 first:\n  a => e\n\
                 second:\n  e => i\n",
            )
            .unwrap(),
        )
        .unwrap();
        assert!(compiled.input_syllabified);
        // syllabify, rule, strip, syllabify, rule, strip, syllabify
        let syllabify_count = compiled
            .steps
            .iter()
            .filter(|s| matches!(s, Step::Syllabify(_)))
            .count();
        assert_eq!(syllabify_count, 3);
    }

    #[test]
    fn cleanup_rules_rerun_until_off() {
        let compiled = crate::compiler::compile(
            &parse(
                "clean cleanup:\n  x => *\n\
                 first:\n  a => e\n\
                 clean:\n  off\n\
                 second:\n  e => i\n",
            )
            .unwrap(),
        )
        .unwrap();
        // clean declared (runs), first, clean (persistent), second; after
        // the off, no more clean
        let rule_steps: Vec<usize> = compiled
            .steps
            .iter()
            .filter_map(|s| match s {
                Step::Rule { rule, .. } => Some(*rule),
                _ => None,
            })
            .collect();
        // rules: 0 = clean, 1 = first, 2 = second
        assert_eq!(rule_steps, vec![0, 1, 0, 2]);
    }
}
