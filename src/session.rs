// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) Autumn <auctumnus@pm.me>

//! The session path: a faithful port of lexurgy's `SoundChangeSession`, which
//! runs the changer over a list of words and returns the full result map —
//! final output, intermediate-romanizer stages, per-word tracing, and
//! structured per-word errors. This is what the HTTP API (`scv1`) needs and
//! what the plain [`CompiledRules::apply`] fast path deliberately omits.
//!
//! It walks the named [`Stage`] list (lexurgy's `sequencedRules`) rather than
//! the flat [`Step`](crate::compiler::Step) pipeline, so it can name traces and
//! capture intermediate stages. Execution runs on the reference VM tier
//! ([`CompiledRules::run_steps_vm`]); output is identical to `apply` by
//! construction, just without the FST tier's speed.

use crate::compiler::{CompiledRules, StageKind, Universe};
use crate::word::Phrase;

/// Options mirroring lexurgy's `SoundChangeOptions` (the fields the `scv1`
/// API surfaces).
#[derive(Default)]
pub struct ChangeOptions<'a> {
    /// Ignore all rules before the (`ApplyRule`) rule with this name.
    pub start_at: Option<&'a str>,
    /// Ignore the (`ApplyRule`) rule with this name and everything after it.
    pub stop_before: Option<&'a str>,
    /// Input words whose rule-by-rule evolution to record as traces.
    pub trace_words: &'a [String],
}

/// One step of a traced word's evolution (lexurgy's `TraceStep`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TraceStep {
    pub rule: String,
    pub output: String,
}

/// A per-word failure (lexurgy's `RuleFailure` / `LscRuleNotApplicable`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuleFailure {
    pub message: String,
    pub rule: Option<String>,
    pub original_word: Option<String>,
    pub current_word: Option<String>,
}

/// The full result of a session run (lexurgy's `Map<String?, …>` flattened
/// into named fields). Insertion order is preserved for `intermediate_words`
/// (stage order) and `traces` (input-word order).
#[derive(Debug, Clone, Default)]
pub struct ChangeOutput {
    pub rule_names: Vec<String>,
    pub output_words: Vec<String>,
    pub intermediate_words: Vec<(String, Vec<String>)>,
    pub traces: Vec<(String, Vec<TraceStep>)>,
    pub errors: Vec<RuleFailure>,
}

/// A whole-run failure: nothing in the result map, an error response instead.
/// Mirrors the exceptions lexurgy throws before/around the per-word run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionError {
    /// `startAt`/`stopBefore` named a rule that doesn't exist (lexurgy's
    /// `LscRuleNotFound`, surfaced as a runtime error).
    RuleNotFound { name: String, action: &'static str },
    /// An input word couldn't be parsed (lexurgy throws at session
    /// construction; the API reports it as a runtime error).
    Word(String),
}

impl std::fmt::Display for SessionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SessionError::RuleNotFound { name, action } => {
                write!(f, "can't {action} rule {name}; there is no rule with that name")
            }
            SessionError::Word(e) => write!(f, "{e}"),
        }
    }
}

/// One input row's evolving state: a list of cells (tab-separated), each
/// either an evolving phrase + its current universe, or a failure.
type Cell = Result<(Phrase, Universe), RuleFailure>;

impl CompiledRules {
    /// Run the changer over `words`, returning the full result map. Faithful
    /// to lexurgy's `changeWithIntermediatesAndIndividualErrors`.
    pub fn change_with_intermediates(
        &self,
        words: &[&str],
        options: &ChangeOptions,
    ) -> Result<ChangeOutput, SessionError> {
        // `startAt`/`stopBefore` slice the named stage list (lexurgy's
        // `applyStartAndStop`); an unknown name is a whole-run error.
        let active = self.active_range(options.start_at, options.stop_before)?;

        // Which input indices to trace: the first occurrence of each
        // `traceWords` entry (lexurgy keys traces by the original word).
        let mut seen_trace: Vec<&str> = Vec::new();
        let trace_flags: Vec<bool> = words
            .iter()
            .map(|w| {
                if options.trace_words.iter().any(|t| t == w) && !seen_trace.contains(w) {
                    seen_trace.push(w);
                    true
                } else {
                    false
                }
            })
            .collect();

        // One mutable clone runs every word: the interner is append-only and
        // rendering depends only on declarations, so reusing it across words
        // is output-independent (the concurrency contract in CLAUDE.md).
        let mut changer = self.clone();

        // Stage names of the active intermediate romanizers, in order: these
        // become the `intermediate_words` keys (the bare stage name).
        let inter_stage_names: Vec<String> = self.stages[active.clone()]
            .iter()
            .filter_map(|s| match &s.kind {
                StageKind::IntermediateRomanize { stage_name, .. } => Some(stage_name.to_string()),
                _ => None,
            })
            .collect();

        let mut output_words = Vec::with_capacity(words.len());
        let mut errors = Vec::new();
        let mut traces = Vec::new();
        // Per intermediate stage (in order): the per-word rendered output.
        let mut inter_columns: Vec<Vec<String>> = vec![Vec::new(); inter_stage_names.len()];

        for (i, &word) in words.iter().enumerate() {
            let outcome = run_word(&mut changer, word, &self.stages[active.clone()], trace_flags[i])?;
            output_words.push(outcome.output);
            if let Some(failure) = outcome.error {
                errors.push(failure);
            }
            for (col, value) in outcome.intermediates.into_iter().enumerate() {
                inter_columns[col].push(value);
            }
            if trace_flags[i] && !outcome.traces.is_empty() {
                traces.push((word.to_string(), outcome.traces));
            }
        }

        // Build the intermediate map preserving stage order; a duplicate stage
        // name keeps its first position and takes the last value (lexurgy's
        // `LinkedHashMap` overwrite semantics).
        let mut intermediate_words: Vec<(String, Vec<String>)> = Vec::new();
        for (name, column) in inter_stage_names.into_iter().zip(inter_columns) {
            if let Some(slot) = intermediate_words.iter_mut().find(|(n, _)| *n == name) {
                slot.1 = column;
            } else {
                intermediate_words.push((name, column));
            }
        }

        Ok(ChangeOutput {
            rule_names: self.rule_names(),
            output_words,
            intermediate_words,
            traces,
            errors,
        })
    }

    /// Port of lexurgy's `applyStartAndStop`: the slice of `stages` to run.
    fn active_range(
        &self,
        start_at: Option<&str>,
        stop_before: Option<&str>,
    ) -> Result<std::ops::Range<usize>, SessionError> {
        let mut start = match start_at {
            None => 0,
            Some(name) => self
                .stages
                .iter()
                .position(|s| s.kind.is_rule() && s.name == name)
                .ok_or_else(|| SessionError::RuleNotFound {
                    name: name.to_string(),
                    action: "start at",
                })?,
        };
        // A `startAt` that lands right after a syllabification backs up one,
        // so the word is re-syllabified before the chosen rule runs.
        if start > 0 && self.stages[start - 1].kind.is_syllabify() {
            start -= 1;
        }
        let stop = match stop_before {
            None => self.stages.len(),
            Some(name) => self
                .stages
                .iter()
                .position(|s| s.kind.is_rule() && s.name == name)
                .ok_or_else(|| SessionError::RuleNotFound {
                    name: name.to_string(),
                    action: "stop before",
                })?,
        };
        Ok(start..stop.max(start))
    }
}

/// One word's results after running the active stages.
struct WordOutcome {
    output: String,
    error: Option<RuleFailure>,
    /// One entry per active intermediate-romanizer stage, in stage order.
    intermediates: Vec<String>,
    traces: Vec<TraceStep>,
}

fn run_word(
    changer: &mut CompiledRules,
    input: &str,
    stages: &[crate::compiler::Stage],
    trace: bool,
) -> Result<WordOutcome, SessionError> {
    // Parse the row into cells (tab-separated), each a phrase. An input parse
    // failure is fatal for the whole request (lexurgy throws at construction).
    let mut row: Vec<Cell> = Vec::new();
    for cell in input.split('\t') {
        let phrase = changer.parse_cell(cell).map_err(|e| SessionError::Word(e.to_string()))?;
        row.push(Ok((phrase, changer.input_universe)));
    }

    let mut intermediates = Vec::new();
    let mut traces = Vec::new();

    for stage in stages {
        match &stage.kind {
            StageKind::Rule { steps } | StageKind::Cleanup { steps } | StageKind::Syllabify { steps } => {
                let before = trace.then(|| render_trace(changer, &row));
                // Own the step slice so the changer can be borrowed mutably below.
                let stage_steps = changer.steps[steps.clone()].to_vec();
                for cell in &mut row {
                    let fail = if let Ok((phrase, universe)) = cell {
                        // The cell string *before* this rule, for the error report.
                        let before_cell = changer.render_phrase(phrase, *universe);
                        match changer.run_steps_vm(&stage_steps, phrase, universe) {
                            Ok(()) => None,
                            Err(e) => Some(RuleFailure {
                                message: e.to_string(),
                                rule: Some(stage.name.to_string()),
                                original_word: Some(input.to_string()),
                                current_word: Some(before_cell),
                            }),
                        }
                    } else {
                        None
                    };
                    if let Some(f) = fail {
                        *cell = Err(f);
                    }
                }
                if trace {
                    let after = render_trace(changer, &row);
                    if before.as_deref() != Some(after.as_str()) {
                        traces.push(TraceStep {
                            rule: stage.name.to_string(),
                            output: after,
                        });
                    }
                }
            }
            StageKind::IntermediateRomanize { steps, .. } => {
                let before = trace.then(|| render_trace(changer, &row));
                // Run on a copy of each cell; the main stream is untouched.
                let mut had_error = false;
                let mut cell_strings = Vec::with_capacity(row.len());
                for cell in &row {
                    match cell {
                        Err(_) => {
                            had_error = true;
                            cell_strings.push("ERROR".to_string());
                        }
                        Ok((phrase, universe)) => {
                            let mut copy = phrase.clone();
                            let mut u = *universe;
                            match changer.run_steps_vm(steps, &mut copy, &mut u) {
                                Ok(()) => cell_strings.push(changer.render_phrase(&copy, u)),
                                Err(_) => {
                                    had_error = true;
                                    cell_strings.push("ERROR".to_string());
                                }
                            }
                        }
                    }
                }
                // The intermediates-map value is whole-row ERROR if any cell
                // failed; the trace shows ERROR only for the failed cells.
                let trace_after = cell_strings.join("\t");
                let value = if had_error {
                    "ERROR".to_string()
                } else {
                    trace_after.clone()
                };
                intermediates.push(value);
                if trace && before.as_deref() != Some(trace_after.as_str()) {
                    traces.push(TraceStep {
                        rule: stage.name.to_string(),
                        output: trace_after,
                    });
                }
            }
        }
    }

    // Final output: whole-row ERROR if any cell failed (lexurgy's
    // `runCatching { join { getOrThrow } }`), else the cells joined by tab.
    let output = render_output(changer, &row);
    // The reported error for the row is the leftmost failed cell's failure.
    let error = row.iter().find_map(|c| c.as_ref().err().cloned());

    Ok(WordOutcome {
        output,
        error,
        intermediates,
        traces,
    })
}

/// Render the row for tracing: each cell rendered, or `ERROR` if failed,
/// joined by tab (lexurgy's `joinToString("\t") { it.string }`).
fn render_trace(changer: &CompiledRules, row: &[Cell]) -> String {
    row.iter()
        .map(|cell| match cell {
            Ok((phrase, universe)) => changer.render_phrase(phrase, *universe),
            Err(_) => "ERROR".to_string(),
        })
        .collect::<Vec<_>>()
        .join("\t")
}

/// Render the row's final/intermediate output: `ERROR` if *any* cell failed,
/// else the cells joined by tab.
fn render_output(changer: &CompiledRules, row: &[Cell]) -> String {
    if row.iter().any(|c| c.is_err()) {
        return "ERROR".to_string();
    }
    row.iter()
        .map(|cell| {
            let (phrase, universe) = cell.as_ref().expect("checked no errors");
            changer.render_phrase(phrase, *universe)
        })
        .collect::<Vec<_>>()
        .join("\t")
}
