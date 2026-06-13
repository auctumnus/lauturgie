// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) Autumn <auctumnus@pm.me>

//! Feature-variable expansion: turning a variable-bearing rule into
//! variable-free instances (cartesian product over value domains)
//! plus the soundness gates that decide when that is legal.
// (moved from the former monolithic fst.rs; see `super` for shared types)
use super::*;

/// Expand an expression's feature variables into variable-free instances:
/// the cartesian product over each variable's value domain. Sound because
/// a segment has exactly one value per feature: instances' binding tests
/// are mutually exclusive wherever the VM's binding would be
/// deterministic, which the gates below guarantee:
///
/// - no negated variable tests (`!$X` reads an existing binding, and
///   binding-order bugs are per-word errors the VM must keep producing);
/// - variables in match position require a *rigid* `from` (leaves and
///   sequences only), so every instance matches the same single width and
///   the binding spot is unambiguous;
/// - variables used in the output must be bound in `from`, or by a single
///   rigid condition environment (lexurgy binds during the env match; with
///   one condition and no backtracking shapes, "first value that passes"
///   is "the value of the segment at that spot");
/// - variables in exclusions must be bound by `from` or the condition
///   (a free variable in an exclusion means "excluded if *any* value
///   matches", which instance expansion would flip into "if *every*").
pub(super) fn expand_variables(
    expr: &ExprIr,
    decls: &Declarations,
) -> Result<(Vec<ExprIr>, bool), VmReason> {
    let mut from_vars = VarSet::default();
    pattern_vars(&expr.from, &mut from_vars, false);
    let mut to_vars = VarSet::default();
    emit_vars(&expr.to, &mut to_vars);
    let mut cond_vars = VarSet::default();
    for env in &expr.condition {
        env_vars(env, &mut cond_vars);
    }
    let mut excl_vars = VarSet::default();
    for env in &expr.exclusion {
        env_vars(env, &mut excl_vars);
    }
    if from_vars.is_empty() && to_vars.is_empty() && cond_vars.is_empty() && excl_vars.is_empty() {
        return Ok((vec![expr.clone()], false));
    }
    if from_vars.negated_test || cond_vars.negated_test || excl_vars.negated_test {
        return Err("a negated feature variable");
    }
    if !from_vars.is_empty() && !rigid(&expr.from) {
        return Err("feature variables in a non-rigid match pattern");
    }
    // Variables bound by the environment: a single condition whose sides
    // pin every variable test to a deterministic spot (the binding is then
    // "the value of the segment there", and "first value that passes" reads
    // exactly that segment).
    let cond_bound: Vec<FeatureId> = if expr.condition.len() == 1
        && expr.condition[0]
            .before
            .as_ref()
            .is_none_or(|p| binding_deterministic(p, true))
        && expr.condition[0]
            .after
            .as_ref()
            .is_none_or(|p| binding_deterministic(p, false))
    {
        let mut vars = VarSet::default();
        env_vars(&expr.condition[0], &mut vars);
        vars.vars
    } else {
        vec![]
    };
    let bound = |var: FeatureId| from_vars.vars.contains(&var) || cond_bound.contains(&var);
    for &var in &to_vars.vars {
        if !bound(var) {
            return Err("a feature variable in the output the FST can't see bound");
        }
    }
    // Environment matching *commits* to the first match's bindings
    // (kotlin doesn't backtrack an earlier alternative's binding when a
    // later part of the environment then fails), so instance expansion,
    // which effectively explores every value, is only sound where the
    // binding is deterministic: bound in `from`, or read off a fixed spot
    // in the single rigid condition.
    for &var in &cond_vars.vars {
        if !bound(var) {
            return Err("a feature variable in an environment the FST can't see bound");
        }
    }
    // Negated contexts mean "no value matches here": only sound once the
    // variable is already concrete when the negation is evaluated.
    // Evaluation order: `from` binds first (left to right), then
    // conditions, then exclusions (which see the condition's bindings).
    // Within-`from` and within-condition ordering isn't tracked, so those
    // need the binding to come from an earlier stage entirely.
    if !from_vars.neg_vars.is_empty() {
        return Err("a feature variable under negation in the match pattern");
    }
    for &var in &cond_vars.neg_vars {
        if !from_vars.vars.contains(&var) {
            return Err("a feature variable under negation the FST can't see bound");
        }
    }
    for &var in excl_vars.vars.iter().chain(&excl_vars.neg_vars) {
        if !bound(var) {
            return Err("a feature variable under negation the FST can't see bound");
        }
    }
    // The full variable set, in first-appearance order.
    let mut vars: Vec<FeatureId> = Vec::new();
    for set in [&from_vars, &cond_vars, &excl_vars, &to_vars] {
        for &v in &set.vars {
            if !vars.contains(&v) {
                vars.push(v);
            }
        }
    }
    let mut total = 1usize;
    for &var in &vars {
        let def = decls.features.def(var);
        if def.level != Level::Segment {
            return Err("a syllable-level feature variable");
        }
        total = total.saturating_mul(def.values.len());
        if total > MAX_VARIANTS {
            return Err("feature variables over too large a domain");
        }
    }
    let mut instances = Vec::with_capacity(total);
    let mut assigns: Vec<(FeatureId, u8)> = vars.iter().map(|&v| (v, 0)).collect();
    'outer: loop {
        instances.push(subst_expr(expr, &assigns, decls));
        for slot in assigns.iter_mut().rev() {
            let domain = decls.features.def(slot.0).values.len() as u8;
            slot.1 += 1;
            if slot.1 < domain {
                continue 'outer;
            }
            slot.1 = 0;
        }
        break;
    }
    Ok((instances, !from_vars.is_empty()))
}

#[derive(Default)]
pub(super) struct VarSet {
    /// Variables in positive (binding) positions.
    vars: Vec<FeatureId>,
    /// Variables inside a negated context (`!x`, a `Look` exclusion):
    /// matching there means "no value works", which instance expansion
    /// would flip into "some value works"; only sound when the variable
    /// is already bound (concrete in every instance) before the negation
    /// is evaluated.
    neg_vars: Vec<FeatureId>,
    /// A negated variable *test* (`!$X`): reads an existing binding;
    /// binding-order errors are per-word VM behavior we don't model.
    negated_test: bool,
}

impl VarSet {
    fn is_empty(&self) -> bool {
        self.vars.is_empty() && self.neg_vars.is_empty() && !self.negated_test
    }
    fn add_test(&mut self, test: &SegTest, neg: bool) {
        if let SegTest::Matrix(m) | SegTest::SylMatrix(m) = test {
            for var in &m.vars {
                if var.negated {
                    self.negated_test = true;
                } else {
                    let set = if neg {
                        &mut self.neg_vars
                    } else {
                        &mut self.vars
                    };
                    if !set.contains(&var.feature) {
                        set.push(var.feature);
                    }
                }
            }
        }
    }
}

pub(super) fn pattern_vars(pattern: &Pattern, out: &mut VarSet, neg: bool) {
    match pattern {
        Pattern::Test(test) => out.add_test(test, neg),
        Pattern::Text(t) => t.tests.iter().for_each(|test| out.add_test(test, neg)),
        Pattern::Seq(parts) | Pattern::Alt(parts) | Pattern::Intersect(parts) => {
            parts.iter().for_each(|p| pattern_vars(p, out, neg))
        }
        Pattern::Repeat { inner, .. } | Pattern::Capture { inner, .. } => {
            pattern_vars(inner, out, neg)
        }
        Pattern::Not(inner) | Pattern::NotAhead(inner) => pattern_vars(inner, out, true),
        Pattern::Look {
            inner,
            condition,
            exclusion,
        } => {
            pattern_vars(inner, out, neg);
            condition.iter().for_each(|env| env_vars_in(env, out, neg));
            exclusion.iter().for_each(|env| env_vars_in(env, out, true));
        }
        _ => {}
    }
}

pub(super) fn emit_vars(emit: &Emit, out: &mut VarSet) {
    match emit {
        Emit::Matrix(update) => {
            for &var in &update.vars {
                if !out.vars.contains(&var) {
                    out.vars.push(var);
                }
            }
        }
        Emit::Seq(parts) | Emit::Alt(parts) => parts.iter().for_each(|e| emit_vars(e, out)),
        _ => {}
    }
}

pub(super) fn env_vars(env: &EnvIr, out: &mut VarSet) {
    env_vars_in(env, out, false);
}

pub(super) fn env_vars_in(env: &EnvIr, out: &mut VarSet, neg: bool) {
    if let Some(p) = &env.before {
        pattern_vars(p, out, neg);
    }
    if let Some(p) = &env.after {
        pattern_vars(p, out, neg);
    }
}

/// No backtracking shapes: matching is a single deterministic walk, so a
/// variable test's spot (and the match width) can't depend on the binding.
pub(super) fn rigid(pattern: &Pattern) -> bool {
    match pattern {
        Pattern::Test(_) | Pattern::Text(_) | Pattern::Not(_) | Pattern::Empty => true,
        Pattern::WordStart | Pattern::WordEnd => true,
        Pattern::Seq(parts) => parts.iter().all(rigid),
        _ => false,
    }
}

/// One side of a condition environment: is every variable test pinned to a
/// unique position relative to the match, across *all* ways the side could
/// match? Rigid sides qualify trivially. Beyond that, a repeat is allowed
/// when its inner test is variable-free and provably disjoint from the
/// next element away from the match (`ə => [$Height] / [$Height !glide]
/// [glide]* _`): two different repeat counts would put that neighbor's
/// segment inside the repeat's span, contradicting disjointness, so the
/// count (and every variable spot beyond it) is forced. Unique spots
/// plus value exclusivity mean at most one instance's environment can
/// match at all, so "some instance matches" reads the same segment
/// kotlin's committed first match binds. Elements past the outermost
/// variable don't affect variable spots and are unconstrained.
pub(super) fn binding_deterministic(side: &Pattern, before: bool) -> bool {
    let mut elems: Vec<&Pattern> = match side {
        Pattern::Seq(parts) => parts.iter().collect(),
        other => vec![other],
    };
    if before {
        // A before side is anchored at the match start: walk outward, i.e.
        // right to left.
        elems.reverse();
    }
    let no_vars = |p: &Pattern| {
        let mut vars = VarSet::default();
        pattern_vars(p, &mut vars, false);
        vars.is_empty()
    };
    for (i, elem) in elems.iter().enumerate() {
        if elems[i..].iter().all(|p| no_vars(p)) {
            return true;
        }
        match elem {
            Pattern::Test(_) | Pattern::Text(_) | Pattern::Not(_) | Pattern::Empty => {}
            Pattern::WordStart | Pattern::WordEnd => {}
            Pattern::Repeat { inner, .. } => {
                let Pattern::Test(rep) = &**inner else {
                    return false;
                };
                if !no_vars(inner) {
                    return false;
                }
                let Some(adj) = elems
                    .get(i + 1)
                    .and_then(|next| adjacent_test(next, before))
                else {
                    return false;
                };
                if !seg_tests_disjoint(rep, adj) {
                    return false;
                }
            }
            _ => return false,
        }
    }
    true
}

/// The single-segment test of `elem` nearest the previous walk element
/// (which sits on the match side of it).
pub(super) fn adjacent_test(elem: &Pattern, before: bool) -> Option<&SegTest> {
    match elem {
        Pattern::Test(test) => Some(test),
        Pattern::Text(t) => {
            if before {
                t.tests.last()
            } else {
                t.tests.first()
            }
        }
        _ => None,
    }
}

/// Conservatively: can no segment satisfy both tests? Only the concrete
/// parts are consulted, so a variable on either side (which can only add
/// constraints) never breaks disjointness.
pub(super) fn seg_tests_disjoint(a: &SegTest, b: &SegTest) -> bool {
    let (SegTest::Matrix(a), SegTest::Matrix(b)) = (a, b) else {
        return false;
    };
    bit_tests_disjoint(&a.seg, &b.seg) || bit_tests_disjoint(&a.syl, &b.syl)
}

pub(super) fn bit_tests_disjoint(a: &BitTest, b: &BitTest) -> bool {
    // Positive values disagree on a shared field…
    if (a.eq_want ^ b.eq_want) & a.eq_mask & b.eq_mask != 0 {
        return true;
    }
    // …or one side's positive values pin a field to exactly a value the
    // other side negates.
    let violates = |neg: &BitTest, pos: &BitTest| {
        neg.ne
            .iter()
            .any(|&(mask, want)| mask & !pos.eq_mask == 0 && pos.eq_want & mask == want)
    };
    violates(a, b) || violates(b, a)
}

pub(super) fn subst_expr(
    expr: &ExprIr,
    assigns: &[(FeatureId, u8)],
    decls: &Declarations,
) -> ExprIr {
    ExprIr {
        from: subst_pattern(&expr.from, assigns, decls),
        to: subst_emit(&expr.to, assigns, decls),
        condition: expr
            .condition
            .iter()
            .map(|env| subst_env(env, assigns, decls))
            .collect(),
        exclusion: expr
            .exclusion
            .iter()
            .map(|env| subst_env(env, assigns, decls))
            .collect(),
    }
}

pub(super) fn subst_pattern(
    pattern: &Pattern,
    assigns: &[(FeatureId, u8)],
    decls: &Declarations,
) -> Pattern {
    match pattern {
        Pattern::Test(test) => Pattern::Test(subst_test(test, assigns, decls)),
        Pattern::Seq(parts) => Pattern::Seq(
            parts
                .iter()
                .map(|p| subst_pattern(p, assigns, decls))
                .collect(),
        ),
        Pattern::Alt(parts) => Pattern::Alt(
            parts
                .iter()
                .map(|p| subst_pattern(p, assigns, decls))
                .collect(),
        ),
        Pattern::Repeat { inner, min, max } => Pattern::Repeat {
            inner: Box::new(subst_pattern(inner, assigns, decls)),
            min: *min,
            max: *max,
        },
        Pattern::Not(inner) => Pattern::Not(Box::new(subst_pattern(inner, assigns, decls))),
        Pattern::Intersect(parts) => Pattern::Intersect(
            parts
                .iter()
                .map(|p| subst_pattern(p, assigns, decls))
                .collect(),
        ),
        other => other.clone(),
    }
}

pub(super) fn subst_test(
    test: &SegTest,
    assigns: &[(FeatureId, u8)],
    decls: &Declarations,
) -> SegTest {
    let SegTest::Matrix(m) = test else {
        return test.clone();
    };
    if m.vars.is_empty() {
        return test.clone();
    }
    let mut out = m.clone();
    out.vars.clear();
    for var in &m.vars {
        debug_assert!(!var.negated, "negated variables are gated out");
        let &(_, code) = assigns
            .iter()
            .find(|(f, _)| *f == var.feature)
            .expect("variable missing from assignment");
        let def = decls.features.def(var.feature);
        let field = def.field_mask();
        if out.seg.eq_mask & field != 0 {
            // The matrix also names a concrete value of this feature
            // (`[lab $place]`): the binding can only ever be that value,
            // so any other assignment is unsatisfiable.
            if out.seg.eq_want & field != def.encode(code) {
                out.seg.ne.push((0, 0)); // `word & 0 != 0` never holds
            }
        } else {
            out.seg.eq_mask |= field;
            out.seg.eq_want |= def.encode(code);
        }
    }
    SegTest::Matrix(out)
}

pub(super) fn subst_emit(emit: &Emit, assigns: &[(FeatureId, u8)], decls: &Declarations) -> Emit {
    match emit {
        Emit::Matrix(update) if !update.vars.is_empty() => {
            let mut out = update.clone();
            out.vars.clear();
            for &var in &update.vars {
                let &(_, code) = assigns
                    .iter()
                    .find(|(f, _)| *f == var)
                    .expect("variable missing from assignment");
                let def = decls.features.def(var);
                out.seg_mask |= def.field_mask();
                out.seg_bits = out.seg_bits & !def.field_mask() | def.encode(code);
            }
            Emit::Matrix(out)
        }
        Emit::Seq(parts) => Emit::Seq(
            parts
                .iter()
                .map(|e| subst_emit(e, assigns, decls))
                .collect(),
        ),
        Emit::Alt(parts) => Emit::Alt(
            parts
                .iter()
                .map(|e| subst_emit(e, assigns, decls))
                .collect(),
        ),
        other => other.clone(),
    }
}

pub(super) fn subst_env(env: &EnvIr, assigns: &[(FeatureId, u8)], decls: &Declarations) -> EnvIr {
    EnvIr {
        before: env
            .before
            .as_ref()
            .map(|p| subst_pattern(p, assigns, decls)),
        after: env.after.as_ref().map(|p| subst_pattern(p, assigns, decls)),
        anchored: env.anchored,
    }
}
