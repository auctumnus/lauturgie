// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) Autumn <auctumnus@pm.me>

//! AST → executable sound changer.
//!
//! The pipeline is a sequence of passes:
//!
//! 1. **resolve** ([`decls`]): collect `Feature`/`Diacritic`/`Symbol`/`Class`/
//!    `Element` declarations into a [`decls::Declarations`], packing the
//!    feature space into bitfields ([`features`]) and seeding the segment
//!    interner ([`segments`]).
//! 2. **lower** ([`ir`]): resolve each rule against the declarations into a
//!    regex-like IR whose leaves are [`ir::SegTest`] predicates: class and
//!    element refs inlined, text interned to segment ids, matrices compiled
//!    to mask-and-compare tests.
//! 3. **tier selection** ([`crate::fst`]): rules in the regular subset
//!    (text/matrix/alternation/repetition/environments, but no captures,
//!    variables, or syllable structure) compile to predicate-transition
//!    automata, lazily determinized over predicate minterms; adjacent
//!    context-free single-segment rules additionally fuse into composed
//!    segment maps. Everything else runs on the reference VM.
//!
//! [`compile`] runs the passes and returns the [`CompiledRules`] that the VM
//! executes against words.

pub mod decls;
pub mod features;
pub mod ir;
pub mod lower;
pub mod pairing;
pub mod segments;

use crate::ast;
use smol_str::SmolStr;

/// A compilation failure. These correspond to lexurgy's `Lsc*` user errors
/// (`LscUndefinedName`, `LscDuplicateName`, ...) so that error behavior can
/// stay compatible.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompileError {
    /// A name was used that isn't declared (or not declared yet; lexurgy
    /// requires declaration before use).
    Undefined { kind: &'static str, name: String },
    /// The same name was declared twice.
    Duplicate { kind: &'static str, name: String },
    /// Two declarations (symbols or diacritics) have the same matrix.
    DuplicateMatrix { kind: &'static str, name: String },
    /// One matrix gives multiple values for the same feature.
    RepeatedFeature { feature: String },
    /// The declared features don't fit in a feature word. Lexurgy has no
    /// such limit, but 128 bits of *distinct* feature values is far past any
    /// real file; this is a resilience guard, not a real restriction.
    FeatureSpaceExhausted { level: &'static str },
    /// More diacritics than fit in a [`segments::DiacriticMask`].
    TooManyDiacritics,
    /// A diacritic's matrix mixes segment-level and syllable-level features.
    MixedDiacriticLevels { name: String },
    /// A feature value is used at the wrong level (e.g. a syllable feature
    /// in a symbol matrix).
    InvalidFeatureLevel { value: String },
    /// Structurally invalid declaration (catch-all for shapes the grammar
    /// admits but the language rejects).
    Invalid { what: String },
    /// An expression whose `from => to` pairing lexurgy rejects at rule
    /// build time (`InvalidTransformation` / `LscIllegalStructure`).
    Expression { rule: String, what: String },
}

impl CompileError {
    /// Attribute an error that arose while lowering a *rule* body to that
    /// rule, so the API can report it as an `invalidExpression` (lexurgy's
    /// `LscInvalidRuleExpression`: any error linking a rule's expression
    /// carries the rule name). Declaration errors are raised in
    /// `decls::resolve` and never pass through here, so they stay analysis
    /// errors. Already-attributed (`Expression`) errors are left untouched.
    pub fn in_rule(self, rule: &str) -> CompileError {
        match self {
            // Already attributed, or a *structural* error that lexurgy reports
            // as a plain analysis error rather than an invalid expression
            // (e.g. `LscMixedBlock` — a `Then:`/`Else:` mix). Expression-level
            // errors (undefined names, bad matrices, …) are the ones that
            // become `LscInvalidRuleExpression`.
            CompileError::Expression { .. } | CompileError::Invalid { .. } => self,
            other => CompileError::Expression {
                rule: rule.to_string(),
                what: other.to_string(),
            },
        }
    }
}

impl std::fmt::Display for CompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompileError::Undefined { kind, name } => {
                write!(f, "the {kind} \"{name}\" is not defined")
            }
            CompileError::Duplicate { kind, name } => {
                write!(f, "the {kind} \"{name}\" is defined more than once")
            }
            CompileError::DuplicateMatrix { kind, name } => {
                write!(
                    f,
                    "the {kind} \"{name}\" has the same matrix as an earlier {kind}"
                )
            }
            CompileError::RepeatedFeature { feature } => {
                write!(f, "matrix has multiple values of the feature \"{feature}\"")
            }
            CompileError::FeatureSpaceExhausted { level } => {
                write!(f, "too many {level}-level feature values declared")
            }
            CompileError::TooManyDiacritics => write!(f, "too many diacritics declared"),
            CompileError::MixedDiacriticLevels { name } => {
                write!(
                    f,
                    "the diacritic \"{name}\" mixes segment- and syllable-level features"
                )
            }
            CompileError::InvalidFeatureLevel { value } => {
                write!(f, "the value \"{value}\" can't be used at this level")
            }
            CompileError::Invalid { what } => write!(f, "invalid declaration: {what}"),
            CompileError::Expression { rule, what } => {
                write!(f, "error in rule \"{rule}\": {what}")
            }
        }
    }
}

impl std::error::Error for CompileError {}

/// Which segment universe a step runs in. `literal` romanizer blocks run
/// against *empty* declarations (lexurgy links them with
/// `ParseTimeDeclarations.empty`), so their text interns into a separate
/// universe where every character is its own featureless segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Universe {
    Real,
    Literal,
}

/// One stage of the compiled pipeline, mirroring lexurgy's `SequencedRule`
/// list: rules (including re-run cleanup instances), syllabification steps,
/// and re-parses when crossing universes (lexurgy's `Redeclaration`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Step {
    Rule {
        rule: usize,
        universe: Universe,
    },
    /// Re-syllabify with the given syllabifier; `None` = `Syllables: clear`
    /// (strip syllable structure).
    Syllabify(Option<usize>),
    /// Render the word and re-parse it in the given universe.
    Redeclare {
        universe: Universe,
        syllabified: bool,
    },
    /// `removeBoundingBreaks`: runs after each applied rule, but not after
    /// syllabification steps.
    StripBreaks,
}

/// A *named* pipeline stage, mirroring lexurgy's `SequencedRule` granularity
/// (one entry per `ApplyRule`/`Syllabify`/`CleanUp`/`IntermediateRomanize`).
///
/// The flat [`Step`] list is finer-grained than this — a literal deromanizer
/// is one `<deromanizer>` stage spanning two `Step::Rule`s plus a re-parse —
/// so stages are what `rule_names`, tracing, `startAt`/`stopBefore`, and
/// intermediate-romanizer capture work over. The `apply` fast path ignores
/// them entirely; only the session path ([`crate::session`]) walks them.
#[derive(Debug, Clone)]
pub struct Stage {
    /// The name used in `ruleNames` and tracing (`foo`, `<deromanizer>`,
    /// `<syllables>/foo/1`, `<cleanup>/foo/bar`, `<romanizer>-x`).
    pub name: SmolStr,
    pub kind: StageKind,
}

#[derive(Debug, Clone)]
pub enum StageKind {
    /// A normal rule, deromanizer, or final romanizer (lexurgy's `ApplyRule`):
    /// the only kind `startAt`/`stopBefore` can name. Runs `steps` on the
    /// main word stream.
    Rule { steps: std::ops::Range<usize> },
    /// A `Syllables:` (re-)application. Mutates the stream; the `startAt`
    /// back-up-by-one quirk keys off this kind.
    Syllabify { steps: std::ops::Range<usize> },
    /// A persistent `cleanup` rule application. Mutates the stream but is
    /// not nameable by `startAt`/`stopBefore`.
    Cleanup { steps: std::ops::Range<usize> },
    /// An intermediate romanizer: runs `steps` on a *copy* of the stream and
    /// renders it into the intermediates map under `stage_name` (the bare
    /// name, without the `<romanizer>-` prefix). Does not mutate the stream.
    IntermediateRomanize {
        stage_name: SmolStr,
        steps: Vec<Step>,
    },
}

impl StageKind {
    /// `startAt`/`stopBefore` only match `Rule` stages (lexurgy's `ApplyRule`).
    pub fn is_rule(&self) -> bool {
        matches!(self, StageKind::Rule { .. })
    }
    pub fn is_syllabify(&self) -> bool {
        matches!(self, StageKind::Syllabify { .. })
    }
}

/// A compiled sound changer: everything the VM needs to run words.
///
/// `rules` is the lowered IR; tier selection and codegen will turn it into
/// executable stages (FST or bytecode) in a later pass.
#[derive(Debug, Clone)]
pub struct CompiledRules {
    pub decls: decls::Declarations,
    pub segments: segments::SegmentInterner,
    /// The empty-declaration universe for `literal` romanizer blocks.
    pub literal_decls: decls::Declarations,
    pub literal_segments: segments::SegmentInterner,
    pub rules: Vec<ir::RuleIr>,
    pub syllabifiers: Vec<ir::SyllabifierIr>,
    pub steps: Vec<Step>,
    /// Named stages mirroring lexurgy's `SequencedRule` list (see [`Stage`]).
    /// Drives `rule_names`, tracing, `startAt`/`stopBefore`, and
    /// intermediate-romanizer capture in the session path.
    pub stages: Vec<Stage>,
    /// How input words are parsed (set by a literal deromanizer and by an
    /// initial `Syllables:` declaration).
    pub input_universe: Universe,
    pub input_syllabified: bool,
    /// Inter-romanizer bodies, lowered for validation only (their stage
    /// output isn't part of the single-word pipeline, but kotlin builds
    /// their transformers, so pairing errors are still compile errors).
    pub validate_only: Vec<(smol_str::SmolStr, ir::BlockIr, Universe)>,
    /// Per-syllabifier: re-syllabifying an *untouched* phrase is a no-op
    /// (the patterns don't read existing structure), so the apply loop may
    /// skip the step.
    pub syl_skippable: Vec<bool>,
    /// Per-rule FST-tier compilations (`None` = the rule runs on the VM).
    pub fst_rules: Vec<Option<crate::fst::RuleFst>>,
    /// Runs of adjacent single-segment context-free rules fused into
    /// composed segment maps (cross-rule composition).
    pub fused: Vec<crate::fst::FusedRun>,
    /// Step index → index into `fused` of the run starting there.
    pub fused_at: std::collections::HashMap<usize, usize>,
    /// Force the VM tier even where an FST exists (for differential
    /// testing of the tiers against each other).
    pub force_vm: bool,
}

impl CompiledRules {
    /// The rule names used in tracing output, in application order — lexurgy's
    /// `SoundChanger.ruleNames`. One entry per [`Stage`].
    pub fn rule_names(&self) -> Vec<String> {
        self.stages.iter().map(|s| s.name.to_string()).collect()
    }
}

/// Compile a parsed sound-change file.
pub fn compile(statements: &[ast::Statement]) -> Result<CompiledRules, CompileError> {
    let (decls, rest) = decls::resolve(statements)?;
    let segments = segments::SegmentInterner::seed(&decls);
    let (literal_decls, _) = decls::resolve(&[])?;
    let literal_segments = segments::SegmentInterner::seed(&literal_decls);
    let mut compiled = lower::lower(decls, segments, literal_decls, literal_segments, &rest)?;
    // Validate `from => to` pairings the way lexurgy does when it builds
    // transformers (per rule, in the segment universe the rule runs in).
    let mut checked = vec![false; compiled.rules.len()];
    for step in &compiled.steps {
        if let Step::Rule { rule, universe } = step {
            if std::mem::replace(&mut checked[*rule], true) {
                continue;
            }
            let segments = match universe {
                Universe::Real => &compiled.segments,
                Universe::Literal => &compiled.literal_segments,
            };
            pairing::check_rule(&compiled.rules[*rule], segments)?;
        }
    }
    for (name, body, universe) in &compiled.validate_only {
        let segments = match universe {
            Universe::Real => &compiled.segments,
            Universe::Literal => &compiled.literal_segments,
        };
        let rule = ir::RuleIr {
            name: name.clone(),
            body: body.clone(),
        };
        pairing::check_rule(&rule, segments)?;
    }
    compiled.syl_skippable = compiled
        .syllabifiers
        .iter()
        .map(|s| !s.reads_structure())
        .collect();
    // FST-tier compilation needs the universe the rule runs in (transfer
    // pieces read segment diacritics from its interner).
    compiled.fst_rules = vec![None; compiled.rules.len()];
    for step in &compiled.steps {
        if let Step::Rule { rule, universe } = step {
            if compiled.fst_rules[*rule].is_some() {
                continue;
            }
            let (decls, segments) = match universe {
                Universe::Real => (&compiled.decls, &compiled.segments),
                Universe::Literal => (&compiled.literal_decls, &compiled.literal_segments),
            };
            compiled.fst_rules[*rule] =
                crate::fst::compile_rule(&compiled.rules[*rule], decls, segments).ok();
        }
    }
    let (fused, fused_at) = crate::fst::fuse_steps(&compiled.steps, &compiled.fst_rules);
    compiled.fused = fused;
    compiled.fused_at = fused_at;
    Ok(compiled)
}
