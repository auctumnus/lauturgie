//! Differential fuzzer: random (ruleset, word list) cases from a seed.
//!
//! Modes:
//!  - default: fst tier vs reference VM, in-process, fast. A driver process
//!    spawns sandboxed workers (memory ulimit + timeout) so panics, aborts,
//!    OOMs, and hangs all become findings instead of killing the fuzzer.
//!  - `--kotlin`: lauturgie vs the original Kotlin lexurgy CLI. Slower
//!    (one JVM start per case), but the authoritative oracle.
//!  - `--seed N`: reproduce one case verbosely.
//!
//! Usage:
//!   cargo run --release --example fuzz                     # fst-vs-vm, random start
//!   cargo run --release --example fuzz -- --start 0        # deterministic seeds
//!   cargo run --release --example fuzz -- --seed 12345     # reproduce one case
//!   cargo run --release --example fuzz -- --kotlin         # vs Kotlin CLI
//!   cargo run --release --example fuzz -- --kotlin --words 200
//!
//! Findings (mismatching or crashing cases) are printed and saved to
//! fuzz_findings/ as rerunnable .lsc/.wli pairs plus a report.
//! The Kotlin CLI is found via $LEXURGY_CLI or the vendor build; see the
//! memory notes / README for how to build it.

use std::fmt::Write as _;
use std::io::Read as _;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use unicode_normalization::UnicodeNormalization;

const CHUNK: u64 = 500;
/// Per-case watchdog inside the worker. True hangs are structurally
/// prevented now (divergence budgets in both tiers, `MAX_OPTIONS`,
/// `MAX_PROPAGATE_STEPS`), so this is a safety net; it's sized generously
/// because pathological-but-terminating cases (propagate + repeater
/// backtracking on budget-bloated words) legitimately run for seconds, and
/// a case covers *both* tiers; the worst observed terminating case takes
/// ~26 s (fst) + ~17 s (vm) for one word list.
const CASE_TIMEOUT: Duration = Duration::from_secs(60);
/// Driver-side backstop in case the watchdog machinery itself wedges (OOM
/// thrash etc.), normally never hit. Must comfortably exceed a chunk's
/// worth of slow-but-legal cases.
const WORKER_TIMEOUT: Duration = Duration::from_secs(600);
const OURS_TIMEOUT: Duration = Duration::from_secs(60);
const KOTLIN_TIMEOUT: Duration = Duration::from_secs(180);
const WORKER_MEM_KB: u64 = 4 * 1024 * 1024; // 4 GB address-space cap
const MAX_SAVED_FINDINGS: usize = 300;

/// Worker exit code: the per-case watchdog fired; finding already reported.
const EXIT_CASE_TIMEOUT: i32 = 3;
/// `--ours` exit code: lauturgie rejected the ruleset (parse/compile error).
const EXIT_OURS_REJECTED: i32 = 4;
/// `--ours` exit code: lauturgie panicked in parse/compile; message in out file.
const EXIT_OURS_PANICKED: i32 = 5;

// RNG (splitmix64), no deps, deterministic per seed

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    fn chance(&mut self, percent: u64) -> bool {
        self.next() % 100 < percent
    }

    fn pick<'a, T>(&mut self, xs: &'a [T]) -> &'a T {
        &xs[self.below(xs.len())]
    }
}

fn sample<T: Clone>(r: &mut Rng, pool: &[T], n: usize) -> Vec<T> {
    let mut pool = pool.to_vec();
    let mut out = Vec::new();
    for _ in 0..n.min(pool.len()) {
        out.push(pool.swap_remove(r.below(pool.len())));
    }
    out
}

// Case generator

/// (voiceless, voiced, place, manner): a coherent mini feature system so
/// matrix emitters have real symbols to land on.
const PAIRS: &[(&str, &str, &str, &str)] = &[
    ("p", "b", "lab", "stp"),
    ("t", "d", "alv", "stp"),
    ("k", "g", "vel", "stp"),
    ("f", "v", "lab", "frc"),
    ("s", "z", "alv", "frc"),
];
const VOWELS: &[&str] = &["a", "e", "i", "o", "u"];
/// Graphemes that words may contain but declarations never mention.
const STRAYS: &[&str] = &["h", "w", "y", "m", "n", "l"];

struct Phonology {
    letters: Vec<String>,
    /// (name, member count); target classes must pair by size.
    classes: Vec<(String, usize)>,
    /// Values usable inside `[...]`; empty when no features are declared.
    matrix_values: Vec<String>,
    /// Declared feature names, for `$Feature` variables in matrices.
    features: Vec<String>,
    decls: String,
    /// A declared length diacritic (`ː`) letters may carry.
    diacritic: Option<&'static str>,
    /// A declared syllable-level feature value (`+hvy`).
    syl_feature: Option<&'static str>,
    /// `syllables: explicit` is in effect (words may carry `.`).
    syl_explicit: bool,
    /// Pattern-based syllabification is in effect.
    syl_patterns: bool,
    /// Members of the forced `vow`/`con` classes (pattern mode only), for
    /// building words that actually fit the syllable structure.
    vow: Vec<String>,
    con: Vec<String>,
    /// The `syllables:` block; ranks with the rules, so the case assembler
    /// places it after the deromanizer rather than among the declarations.
    syl_block: String,
}

fn gen_phonology(r: &mut Rng) -> Phonology {
    let n_vowels = 2 + r.below(3);
    let vowels = sample(r, VOWELS, n_vowels);
    let n_pairs = 2 + r.below(3);
    let pairs = sample(r, PAIRS, n_pairs);
    let featureful = r.chance(55);

    let mut letters: Vec<String> = vowels.iter().map(|v| v.to_string()).collect();
    for (vl, vd, _, _) in &pairs {
        letters.push(vl.to_string());
        letters.push(vd.to_string());
    }

    // Statement kinds have fixed positions (features, then diacritics, then
    // symbols, then classes; lexurgy's `validateOrder`), so declarations
    // are collected per kind and concatenated in order; one case in thirty
    // deliberately scrambles a line to keep proving reject-parity.
    let mut feats = String::new();
    let mut diacritics = String::new();
    let mut symbols = String::new();
    let mut class_decls = String::new();
    let mut matrix_values: Vec<String> = Vec::new();
    let mut features: Vec<String> = Vec::new();
    if featureful {
        // A leading `*value` marks the feature's null alias (its default,
        // code 0); both compilers treat it identically, so it's varied per
        // feature for coverage of the `Full { null_alias }` decl path.
        let na = |r: &mut Rng| if r.chance(25) { "*" } else { "" };
        writeln!(feats, "feature voice({}uvc, vcd)", na(r)).unwrap();
        writeln!(feats, "feature place({}lab, alv, vel)", na(r)).unwrap();
        writeln!(feats, "feature manner({}stp, frc)", na(r)).unwrap();
        matrix_values.extend(["uvc", "vcd"].map(String::from));
        features.extend(["voice", "place", "manner"].map(String::from));
        for (vl, vd, place, manner) in &pairs {
            writeln!(symbols, "symbol {vl} [{place} {manner} uvc]").unwrap();
            writeln!(symbols, "symbol {vd} [{place} {manner} vcd]").unwrap();
            for v in [place, manner] {
                if !matrix_values.iter().any(|m| m == v) {
                    matrix_values.push(v.to_string());
                }
            }
        }
    }

    // A length diacritic: `aː` is one segment (base plus diacritic), matrix
    // targets like `[+lng]` have to *render* the diacritic, and floating
    // diacritics survive symbol rewrites.
    let mut diacritic = None;
    if featureful && r.chance(30) {
        feats += "feature +lng\n";
        // `(first)` diacritics attach/render after the first char of a
        // *multi*-char base (`tsː` → `ts` + length on `t`); lauturgie matches
        // kotlin per-word exactly (both reject `(first)` on a multi-char core
        // like the symbol `va`). The one quirk is the kotlin CLI aborting a
        // whole batch on such a word-parse error, handled by the per-line
        // fallback in `kotlin_loop`, so `ː` flows freely into words here.
        let modifier = match r.below(6) {
            0 | 1 => " (floating)",
            2 => " (first)",
            _ => "",
        };
        writeln!(diacritics, "diacritic ː{modifier} [+lng]").unwrap();
        matrix_values.push("+lng".to_string());
        matrix_values.push("-lng".to_string());
        diacritic = Some("ː");
    }

    // Multigraph symbols built from existing letters: tokenization must
    // prefer the longest declared symbol, so a word like "tsa" parses as
    // ts-a in both engines once `symbol ts` exists.
    if r.chance(30) {
        let mut multigraphs: Vec<String> = Vec::new();
        for _ in 0..1 + r.below(2) {
            let g = format!("{}{}", r.pick(&letters), r.pick(&letters));
            if g.chars().count() == 2 && !multigraphs.contains(&g) {
                multigraphs.push(g);
            }
        }
        for g in &multigraphs {
            writeln!(symbols, "symbol {g}").unwrap();
        }
        letters.extend(multigraphs);
    }

    let mut classes = Vec::new();
    for name in ["cv", "cn", "cm", "ck"].iter().take(r.below(4)) {
        let n_members = 2 + r.below(3);
        let members = sample(r, &letters, n_members);
        writeln!(class_decls, "class {name} {{{}}}", members.join(", ")).unwrap();
        classes.push((name.to_string(), members.len()));
    }
    // `Element` declarations: a distinct statement/handler from `class` (it
    // can hold structured elements a class can't). Bodies here are letter
    // alternatives (single-segment, known size) so they pair across `=>`
    // exactly like classes; the point is to exercise the decl path. They
    // share the `@name` reference namespace, so they go in the class pool.
    for name in ["e0", "e1"].iter().take(r.below(3)) {
        let n_members = 2 + r.below(3);
        let members = sample(r, &letters, n_members);
        if members.len() < 2 {
            continue;
        }
        writeln!(class_decls, "element {name} {{{}}}", members.join(", ")).unwrap();
        classes.push((name.to_string(), members.len()));
    }

    // Syllables: `explicit` (words carry `.`, rules may move/insert/delete
    // boundaries) or pattern-based (words re-syllabify after every rule;
    // words that stop fitting the structure become per-word errors). The
    // `syllables:` block ranks with the rules, so it's kept separate from
    // the declarations and placed after the deromanizer.
    let mut syl_explicit = false;
    let mut syl_patterns = false;
    let mut syl_feature = None;
    let mut syl_block = String::new();
    let mut vow: Vec<String> = Vec::new();
    let mut con: Vec<String> = Vec::new();
    if r.chance(30) {
        if r.chance(35) {
            syl_explicit = true;
        } else {
            syl_patterns = true;
            vow = vowels.iter().map(|v| v.to_string()).collect();
            con = letters
                .iter()
                .filter(|l| !vow.contains(l))
                .cloned()
                .collect();
            writeln!(class_decls, "class vow {{{}}}", vow.join(", ")).unwrap();
            writeln!(class_decls, "class con {{{}}}", con.join(", ")).unwrap();
            classes.push(("vow".to_string(), vow.len()));
            classes.push(("con".to_string(), con.len()));
        }
        if r.chance(35) {
            // a syllable-level feature, with the diacritic that renders it
            feats += "feature (syllable) +hvy\n";
            diacritics += "diacritic ˈ (before) [+hvy]\n";
            syl_feature = Some("+hvy");
        }
        syl_block += "syllables:\n";
        if syl_explicit {
            syl_block += "    explicit\n";
        } else {
            let shapes = [
                "@con? @vow @con?",
                "@con? @vow",
                "@con @vow @con?",
                "@vow @con?",
            ];
            // Structured patterns name the onset/nucleus/coda explicitly with
            // `::` (and an optional reluctant onset via `?:`), vs the plain
            // CV sequence shapes. Keep the onset optional so the words the
            // word-builder produces still syllabify. Both forms accept an
            // `=> [feature]` assignment after the whole pattern.
            let pattern = if r.chance(40) {
                let onset = if r.chance(60) { "@con?" } else { "@con" };
                let coda = if r.chance(55) { " :: @con?" } else { "" };
                let reluctant = if r.chance(30) { "@con? ?: " } else { "" };
                format!("{reluctant}{onset} :: @vow{coda}")
            } else {
                r.pick(&shapes).to_string()
            };
            let assign = if syl_feature.is_some() && r.chance(50) {
                " => [+hvy]"
            } else {
                ""
            };
            writeln!(syl_block, "    {pattern}{assign}").unwrap();
            if r.chance(25) {
                writeln!(syl_block, "    {}", r.pick(&shapes)).unwrap();
            }
        }
    }

    let mut decls = format!("{feats}{diacritics}{symbols}{class_decls}");
    if r.chance(3) && !decls.is_empty() {
        // move one declaration line out of place (usually rejected by both)
        let lines: Vec<&str> = decls.lines().collect();
        let from = r.below(lines.len());
        let to = r.below(lines.len());
        if from != to {
            let mut shuffled: Vec<&str> = Vec::new();
            for (i, l) in lines.iter().enumerate() {
                if i == to {
                    shuffled.push(lines[from]);
                }
                if i != from {
                    shuffled.push(l);
                }
            }
            decls = shuffled.join("\n") + "\n";
        }
    }

    Phonology {
        letters,
        classes,
        matrix_values,
        features,
        decls,
        diacritic,
        syl_feature,
        syl_explicit,
        syl_patterns,
        vow,
        con,
        syl_block,
    }
}

fn gen_literal(r: &mut Rng, ph: &Phonology) -> String {
    let mut s = if r.chance(12) {
        format!("{}{}", r.pick(&ph.letters), r.pick(&ph.letters))
    } else {
        r.pick(&ph.letters).to_string()
    };
    if let Some(d) = ph.diacritic {
        if r.chance(8) {
            s.push_str(d);
        }
        if r.chance(4) {
            // exact match (`!` disables the diacritic search); also an
            // exact *emitter*, which pairs differently across `=>`
            s.push('!');
        }
    }
    s
}

fn gen_matrix(r: &mut Rng, ph: &Phonology) -> String {
    let n_vals = 1 + r.below(2);
    let vals = sample(r, &ph.matrix_values, n_vals);
    let mut vals: Vec<String> = vals
        .iter()
        .map(|v| {
            if r.chance(12) {
                format!("!{v}")
            } else {
                v.to_string()
            }
        })
        .collect();
    // A feature variable (`$place`), sometimes negated. Random placement
    // produces the whole spectrum: agreement pairs, env-bound emits,
    // unbound uses (per-word errors), and shapes the FST tier must punt
    // back to the VM; all of which have to agree across tiers.
    if !ph.features.is_empty() && r.chance(10) {
        let var = r.pick(&ph.features);
        let neg = if r.chance(15) { "!" } else { "" };
        vals.push(format!("{neg}${var}"));
    }
    // An absent test (`*place`: the feature has no value, i.e. its code-0
    // default), sometimes negated (`!*place`: it has *some* value). The `*`
    // takes a feature NAME, not a value name; matrices only land in match/env
    // positions here, where negated absence is valid in both engines.
    if !ph.features.is_empty() && r.chance(8) {
        let feat = r.pick(&ph.features);
        let neg = if r.chance(20) { "!" } else { "" };
        vals.push(format!("{neg}*{feat}"));
    }
    format!("[{}]", vals.join(" "))
}

/// A single non-recursive matcher element.
fn gen_simple(r: &mut Rng, ph: &Phonology) -> String {
    let roll = r.below(100);
    if roll >= 55 && roll < 70 && !ph.classes.is_empty() {
        return format!("@{}", r.pick(&ph.classes).0);
    }
    if roll >= 70 && roll < 85 && !ph.matrix_values.is_empty() {
        return gen_matrix(r, ph);
    }
    if roll >= 85 && roll < 92 {
        return "[]".to_string();
    }
    gen_literal(r, ph)
}

/// A source/environment element, possibly structured (list, group, repeat,
/// negation). `$$` (the between-words gap) shows up at a low rate in both
/// positions; word lists contain multi-word phrases at a matching rate.
/// `depth` caps nested local environments (`(p / a _)` can hold another
/// one inside its own env sides, but not unboundedly).
fn gen_elem(r: &mut Rng, ph: &Phonology, in_env: bool, depth: u32) -> String {
    let roll = r.below(100);
    if roll < 12 {
        let n = 2 + r.below(2);
        let mut items: Vec<String> = (0..n).map(|_| gen_simple(r, ph)).collect();
        // a list alternative carrying its own local environment
        if depth < 2 && r.chance(10) {
            let i = r.below(items.len());
            let env = gen_env(r, ph, true, depth + 1);
            items[i] = format!("{}{}", items[i], env);
        }
        return format!("{{{}}}", items.join(", "));
    }
    if roll < 20 {
        return format!("({} {})", gen_simple(r, ph), gen_simple(r, ph));
    }
    if roll < 30 {
        // repeaters: bare, exact-count (`*N`), and ranges: bounded
        // (`*(1-3)`), min-only (`*(2-)`), and max-only (`*(-3)`, which is
        // emptyable like `*`). `is_repeater`/`emptyable_elem` know these.
        let rep = [
            "?", "*", "+", "*2", "*3", "*(1-3)", "*(2-3)", "*(2-)", "*(-2)", "*(-3)",
        ];
        return format!("{}{}", gen_simple(r, ph), r.pick(&rep));
    }
    if roll < 36 && (in_env || roll < 33) {
        // negation: freely inside environments, and on the match side too
        // (a single-segment `!x => y` matches any *other* segment; multi-
        // segment / boundary negations are runtime errors in both tiers).
        return format!("!{}", gen_simple(r, ph));
    }
    if roll < 41 {
        return "$$".to_string();
    }
    if roll < 46 && depth < 2 {
        // an element with an attached environment (lexurgy's
        // `EnvironmentElement` / our `Pattern::Look`)
        return format!("({}{})", gen_simple(r, ph), gen_env(r, ph, true, depth + 1));
    }
    if roll < 52 {
        // intersection: both parts must match the same segment(s); `&!`
        // requires the second part NOT to match
        let op = if r.chance(30) { "&!" } else { "&" };
        return format!("{}{op}{}", gen_simple(r, ph), gen_simple(r, ph));
    }
    if (ph.syl_explicit || ph.syl_patterns) && roll < 58 {
        // syllable structure: boundaries, whole syllables, and (when a
        // syllable-level feature exists) feature-tagged syllables
        return match (ph.syl_feature, r.below(4)) {
            (Some(f), 0) => format!("<syl>&[{f}]"),
            (Some(f), 1) => format!("[{f}]"),
            (_, 0 | 2) => "<syl>".to_string(),
            _ => ".".to_string(),
        };
    }
    gen_simple(r, ph)
}

/// A target element positionally paired with `src`. lexurgy requires the
/// same element count on both sides of `=>`, and pairs lists/classes by
/// position and size.
fn gen_tgt_for(r: &mut Rng, ph: &Phonology, src: &str) -> String {
    if src == "$$" {
        // usually keep the gap; sometimes replace it (fusing the words) or
        // delete it outright
        return match r.below(10) {
            0..=5 => "$$".to_string(),
            6..=7 => "*".to_string(),
            _ => gen_literal(r, ph),
        };
    }
    if src == "." {
        // keep the boundary, delete it, or (rarely) replace it with a segment
        return match r.below(10) {
            0..=5 => ".".to_string(),
            6..=8 => "*".to_string(),
            _ => gen_literal(r, ph),
        };
    }
    if src.starts_with("<syl>") || ph.syl_feature.is_some_and(|f| src == format!("[{f}]")) {
        // a whole-syllable match usually re-emits with a syllable feature
        if let Some(f) = ph.syl_feature {
            if r.chance(60) {
                return format!("[{f}]");
            }
        }
    }
    if src.starts_with('{') && r.chance(60) {
        let k = src.matches(", ").count() + 1;
        let items: Vec<String> = (0..k).map(|_| r.pick(&ph.letters).to_string()).collect();
        return format!("{{{}}}", items.join(", "));
    }
    if let Some(name) = src.strip_prefix('@') {
        if r.chance(55) {
            let size = ph.classes.iter().find(|(n, _)| n == name).map(|&(_, s)| s);
            if let Some(size) = size {
                let same: Vec<&(String, usize)> =
                    ph.classes.iter().filter(|&&(_, s)| s == size).collect();
                return format!("@{}", r.pick(&same).0);
            }
        }
    }
    if !ph.matrix_values.is_empty() && r.chance(20) {
        return format!("[{}]", r.pick(&ph.matrix_values));
    }
    if r.chance(8) {
        return "*".to_string(); // delete just this element
    }
    gen_literal(r, ph)
}

fn is_repeater(part: &str) -> bool {
    if part.ends_with(['?', '*', '+']) {
        return true;
    }
    if part.contains("*(") {
        return true; // a range repeater `*(m-n)` / `*(m-)` / `*(-n)`
    }
    // an exact-count repeater `*N`
    match part.rsplit_once('*') {
        Some((_, suf)) => !suf.is_empty() && suf.bytes().all(|b| b.is_ascii_digit()),
        None => false,
    }
}

/// One side of an environment. Both engines reject "meaningless" repeaters
/// at the open edge of an environment (`v* _`, `_ u+`) at compile time, so
/// unless the edge is anchored by `$` the edge element is *usually* kept
/// repeater-free, but one case in ten leaves the repeater in, to keep
/// fuzzing that both validators reject the same shapes.
fn gen_env_side(r: &mut Rng, ph: &Phonology, open_edge_last: Option<bool>, depth: u32) -> String {
    let n = 1 + r.below(2);
    let mut parts: Vec<String> = (0..n).map(|_| gen_elem(r, ph, true, depth)).collect();
    if let Some(last) = open_edge_last {
        if !r.chance(10) {
            let edge = if last { parts.len() - 1 } else { 0 };
            while is_repeater(&parts[edge]) {
                parts[edge] = gen_simple(r, ph);
            }
        }
    }
    parts.join(" ")
}

fn gen_env(r: &mut Rng, ph: &Phonology, force_some: bool, depth: u32) -> String {
    let anchor_before = r.chance(10);
    let anchor_after = r.chance(10);
    let before_edge = if anchor_before { None } else { Some(false) };
    let after_edge = if anchor_after { None } else { Some(true) };
    let mut before = if r.chance(55) {
        gen_env_side(r, ph, before_edge, depth)
    } else {
        String::new()
    };
    let mut after = if r.chance(55) {
        gen_env_side(r, ph, after_edge, depth)
    } else {
        String::new()
    };
    if force_some && before.is_empty() && after.is_empty() {
        if r.chance(50) {
            before = gen_env_side(r, ph, before_edge, depth);
        } else {
            after = gen_env_side(r, ph, after_edge, depth);
        }
    }
    if anchor_before {
        before = format!("$ {before}").trim_end().to_string();
    }
    if anchor_after {
        after = format!("{after} $").trim_start().to_string();
    }
    let mut env = String::from(" / ");
    if !before.is_empty() {
        env += &before;
        env += " ";
    }
    env += "_";
    if !after.is_empty() {
        env += " ";
        env += &after;
    }
    if r.chance(10) {
        let b = gen_env_side(r, ph, Some(false), depth);
        match r.below(3) {
            0 => write!(env, " // {b} _").unwrap(),
            1 => write!(env, " // _ {}", gen_env_side(r, ph, Some(true), depth)).unwrap(),
            _ => write!(env, " // {b} _ {}", gen_env_side(r, ph, Some(true), depth)).unwrap(),
        }
    }
    env
}

struct GenExpr {
    text: String,
    /// Can diverge under ltr/rtl/propagate: a source that can match the
    /// empty string (zero-width insertion loops), a target that grows the
    /// word, or a count mismatch (whole-result-per-claim can grow too).
    /// lauturgie now errors out via its `MAX_RULE_GROWTH` budget, so these
    /// are fuzzed freely in fst-vs-vm mode; the Kotlin oracle still hangs
    /// on them (its CLI's 1 s/step rescue is slow and not even reliable),
    /// so `--kotlin` mode keeps the rescanning modifiers away.
    divergent: bool,
    /// Usable in a filtered rule without tripping lexurgy's validator
    /// (`*` and multi-segment elements on the match side). Rules ignore
    /// this 25% of the time now that both compilers reject identically.
    filter_safe: bool,
}

fn gen_expr(r: &mut Rng, ph: &Phonology) -> GenExpr {
    let syl_active = ph.syl_explicit || ph.syl_patterns;
    if r.chance(7) {
        // insertion
        let tgt = gen_literal(r, ph);
        let env = gen_env(r, ph, true, 0);
        return GenExpr {
            text: format!("* => {tgt}{env}"),
            divergent: true,
            filter_safe: false,
        };
    }
    let n = 1 + r.below(3);
    let mut srcs: Vec<String> = Vec::new();
    let mut tgts: Vec<String> = Vec::new();
    for _ in 0..n {
        let s = gen_elem(r, ph, false, 0);
        tgts.push(gen_tgt_for(r, ph, &s));
        srcs.push(s);
    }
    // emptyable sources (can match zero width): `x?`, `x*`, the zero-width
    // syllable boundary, and max-only ranges `x*(-n)` (min defaults to 0).
    // `x*N` and `x*(m-)`/`x*(m-n)` are not emptyable (they end in a digit /
    // `)` and don't contain the `*(-` marker).
    let emptyable_elem =
        |p: &String| p.ends_with('?') || p.ends_with('*') || p == "." || p.contains("*(-");
    let emptyable = srcs.iter().all(emptyable_elem);
    let single_seg =
        |s: &String| s.starts_with('@') || s.starts_with('[') || s.chars().count() == 1;
    let filter_safe = srcs.iter().all(single_seg);
    let mut forced_env: Option<String> = None;
    if r.chance(12) {
        // whole-match deletion: one `*` per source element keeps counts equal
        tgts = vec!["*".to_string(); n];
    } else if r.chance(11) {
        if n >= 2 && r.chance(35) {
            // two captures; half the time the targets emit them swapped
            let j = 1 + r.below(n - 1);
            srcs[0] = format!("({})$1", srcs[0]);
            srcs[j] = format!("({})$2", srcs[j]);
            let swap = r.chance(50);
            // `$.n` replays the captured material *with* its syllable
            // structure (VM-only); legal only as an output reference, so the
            // match-side captures above stay plain `$n`.
            let dot = if syl_active && r.chance(25) { "." } else { "" };
            tgts[0] = if swap {
                format!("${dot}2")
            } else {
                format!("${dot}1")
            };
            tgts[j] = if swap {
                format!("${dot}1")
            } else {
                format!("${dot}2")
            };
        } else {
            let i = r.below(n);
            srcs[i] = format!("({})$1", srcs[i]);
            tgts[i] = if r.chance(3) {
                // `~$1` is a matcher, not an emitter; reject-parity
                "~$1".to_string()
            } else if syl_active && r.chance(25) {
                // syllable-structure-preserving reference (VM-only)
                "$.1".to_string()
            } else {
                "$1".to_string()
            };
            if r.chance(25) {
                // the environment re-matches the captured material
                // (gemination-style); `~` re-matches it inexactly
                let tilde = if r.chance(25) { "~" } else { "" };
                forced_env = Some(if r.chance(50) {
                    format!(" / _ {tilde}$1")
                } else {
                    format!(" / {tilde}$1 _")
                });
            }
        }
    } else if r.chance(8) {
        // mismatched element counts across `=>`: rejected unless the result
        // is fully independent (whole result replaces each whole match) or
        // the match is a single element (surplus result elements silently
        // drop in filtered rules); both compilers now agree either way.
        if tgts.len() > 1 && r.chance(50) {
            tgts.pop();
        } else {
            tgts.push(r.pick(&ph.letters).to_string());
        }
    } else if r.chance(5) {
        // an emitted `$$` splits the word at that point (count mismatch:
        // legal iff the whole result is independent, which it is unless a
        // matrix target snuck in; reject-parity fuzzing either way)
        let i = r.below(tgts.len() + 1);
        tgts.insert(i, "$$".to_string());
    } else if r.chance(3) {
        // a nested environment in output position: `EnvironmentElement` is
        // not a `ResultElement`, so both compilers must reject
        let i = r.below(tgts.len());
        let env = gen_env(r, ph, true, 1);
        tgts[i] = format!("({}{})", tgts[i], env);
    } else if r.chance(3) {
        // a transforming interfix (`a>b`) in output position: lexurgy
        // disables `>` as an emitter (LscFutureStructure) and lauturgie
        // doesn't implement `>` at all; both reject it (reject-parity, and
        // it exercises the `>` parse + the Transforming lowering reject).
        let i = r.below(tgts.len());
        tgts[i] = format!("{}>{}", tgts[i], gen_literal(r, ph));
    }
    // an emptyable source element whose paired target still emits a segment
    // grows the word on zero-width matches (`i* => u` is +1 per pass)
    let net_growth = srcs
        .iter()
        .zip(&tgts)
        .any(|(s, t)| emptyable_elem(s) && t != "*");
    let growing = net_growth || tgts.iter().any(|t| target_grows(t));
    let mismatched = srcs.len() != tgts.len();
    let src = srcs.join(" ");
    let tgt = tgts.join(" ");
    let env = forced_env.unwrap_or_else(|| {
        if r.chance(55) {
            gen_env(r, ph, false, 0)
        } else {
            String::new()
        }
    });
    GenExpr {
        text: format!("{src} => {tgt}{env}"),
        divergent: emptyable || growing || mismatched,
        filter_safe,
    }
}

/// Can this target element yield more segments than the source element it
/// replaces? Growing targets under ltr/rtl/propagate diverge (e.g.
/// `([])$1 => $1 $1` ltr hangs even kotlin's CLI straight through its
/// rescue timeout); lauturgie now errors via `MAX_RULE_GROWTH`, kotlin
/// still hangs, so these stay suppressed only in `--kotlin` mode.
fn target_grows(tgt: &str) -> bool {
    match tgt.chars().next() {
        // class/matrix/capture map 1:1; `*` deletes
        Some('@') | Some('[') | Some('$') | Some('*') => false,
        // a `{a, b}` list grows if any member is multi-segment; the
        // `ends_with('}')` guard keeps the byte slice on a char boundary
        // (a `{..}>x` interfix target starts with `{` but doesn't end `}`).
        Some('{') if tgt.ends_with('}') => tgt[1..tgt.len() - 1]
            .split(", ")
            .any(|i| i.chars().count() > 1),
        _ => tgt.chars().count() > 1, // multigraph / interfix literal
    }
}

/// A deferred rule available for `:name` splicing.
struct Splice {
    name: String,
    divergent: bool,
}

fn gen_rule(
    r: &mut Rng,
    ph: &Phonology,
    idx: usize,
    allow_divergent: bool,
    splice: Option<&Splice>,
) -> String {
    if splice.is_none() && r.chance(2) {
        return format!("r{idx}:\n    unchanged\n");
    }
    // Usually one expression list; sometimes a `Then:`/`Else:` chain. One
    // kind per level; a flat mix is LscMixedBlock in both engines, so it's
    // only emitted at a token rate for reject-parity.
    let n_blocks = if r.chance(18) { 2 + r.below(2) } else { 1 };
    let mut blocks: Vec<Vec<String>> = Vec::new();
    let mut divergent = false;
    let mut all_filter_safe = true;
    for _ in 0..n_blocks {
        let n_expr = if r.chance(22) { 2 } else { 1 };
        let mut lines = Vec::new();
        for _ in 0..n_expr {
            let e = gen_expr(r, ph);
            divergent |= e.divergent;
            all_filter_safe &= e.filter_safe;
            lines.push(e.text);
        }
        blocks.push(lines);
    }
    if let Some(sp) = splice {
        divergent |= sp.divergent;
        let b = r.below(blocks.len());
        let i = r.below(blocks[b].len() + 1);
        blocks[b].insert(i, format!(":{}", sp.name));
    }
    // Filters usually go on filter-safe expressions so the rule *runs*, but
    // a quarter of the time the gate is ignored; filtered-rule validation
    // (multi-segment text, `*`, count mismatches) is mirrored now, so the
    // unsafe shapes should keep proving both compilers reject identically.
    let filterable = all_filter_safe || r.chance(25);
    let mut head = format!("r{idx}");
    if r.chance(8) && !ph.classes.is_empty() && filterable {
        write!(head, " @{}", r.pick(&ph.classes).0).unwrap();
    } else if r.chance(5) && !ph.matrix_values.is_empty() && filterable {
        write!(head, " [{}]", r.pick(&ph.matrix_values)).unwrap();
    }
    if (allow_divergent || !divergent) && r.chance(16) {
        head += r.pick(&[" ltr", " rtl", " propagate"]);
    }
    let mut s = format!("{head}:\n");
    let kind = if r.chance(50) { "Then" } else { "Else" };
    let mixed = r.chance(4);
    for (bi, block) in blocks.iter().enumerate() {
        if bi > 0 {
            let k = if mixed && r.chance(50) {
                if kind == "Then" {
                    "Else"
                } else {
                    "Then"
                }
            } else {
                kind
            };
            // block headers can carry their own modifiers
            let m = if (allow_divergent || !divergent) && r.chance(6) {
                r.pick(&[" ltr", " rtl", " propagate"])
            } else {
                ""
            };
            writeln!(s, "    {k}{m}:").unwrap();
        }
        for line in block {
            writeln!(s, "    {line}").unwrap();
        }
    }
    s
}

fn gen_word(r: &mut Rng, ph: &Phonology) -> String {
    fn one(r: &mut Rng, ph: &Phonology) -> String {
        // Under pattern syllabification, words that don't fit the structure
        // are per-word errors; most words are built syllable-by-syllable so
        // they fit, while the rest keep proving the error paths agree.
        if ph.syl_patterns && !r.chance(30) {
            let mut w = String::new();
            for _ in 0..1 + r.below(3) {
                if r.chance(60) {
                    w += r.pick(&ph.con);
                }
                w += r.pick(&ph.vow);
                if let Some(d) = ph.diacritic {
                    if r.chance(8) {
                        w += d;
                    }
                }
                if r.chance(25) {
                    w += r.pick(&ph.con);
                }
            }
            return w;
        }
        let len = 1 + r.below(7);
        let mut w = String::new();
        for i in 0..len {
            if ph.syl_explicit && i > 0 && r.chance(12) && !w.ends_with('.') {
                w.push('.');
            }
            if r.chance(4) {
                w += r.pick(STRAYS);
            } else {
                w += r.pick(&ph.letters);
            }
            if let Some(d) = ph.diacritic {
                if r.chance(6) {
                    w += d;
                }
            }
        }
        w
    }
    // A fifth of the inputs are multi-word phrases.
    if r.chance(20) {
        let n = 2 + r.below(2);
        let words: Vec<String> = (0..n).map(|_| one(r, ph)).collect();
        words.join(" ")
    } else {
        one(r, ph)
    }
}

/// `allow_divergent` lets rules that diverge under ltr/rtl/propagate
/// through: fine against lauturgie's own tiers (both error via the growth
/// budget), fatal against the Kotlin CLI (it hangs ~1 s per word at best).
fn gen_case(seed: u64, n_words: usize, allow_divergent: bool) -> (String, Vec<String>) {
    let mut r = Rng(seed);
    let ph = gen_phonology(&mut r);
    let mut lsc = ph.decls.clone();
    if !lsc.is_empty() {
        lsc += "\n";
    }
    let n_rules = 1 + r.below(5);

    // Romanizers are rules at fixed pipeline positions: a deromanizer maps
    // the input spelling to phonemes, a final romanizer maps phonemes back
    // to a spelling, and `literal` variants ignore the declarations.
    if r.chance(10) {
        let lit = if r.chance(25) { " literal" } else { "" };
        let e = gen_expr(&mut r, &ph);
        writeln!(lsc, "deromanizer{lit}:\n    {}\n", e.text).unwrap();
    }
    // the `syllables:` block ranks with the rules: after the deromanizer
    if !ph.syl_block.is_empty() {
        lsc += &ph.syl_block;
        lsc += "\n";
    }
    let inter_romanizer_at = if r.chance(6) {
        Some(r.below(n_rules))
    } else {
        None
    };

    // A deferred rule executes only where a later rule splices it (`:d0`);
    // occasionally it's declared and never used.
    let mut splice: Option<Splice> = None;
    let mut splice_into: Option<usize> = None;
    if r.chance(12) {
        let e = gen_expr(&mut r, &ph);
        let mut divergent = e.divergent;
        let mut body = format!("d0 defer:\n    {}\n", e.text);
        if r.chance(25) {
            // deferred rules can hold Then:/Else: blocks of their own
            let e2 = gen_expr(&mut r, &ph);
            divergent |= e2.divergent;
            let kind = if r.chance(50) { "Then" } else { "Else" };
            write!(body, "    {kind}:\n    {}\n", e2.text).unwrap();
        }
        lsc += &body;
        lsc += "\n";
        splice = Some(Splice {
            name: "d0".to_string(),
            divergent,
        });
        if !r.chance(10) {
            splice_into = Some(r.below(n_rules));
        }
    }

    // A cleanup rule re-runs after every later rule; an `off` redeclaration
    // stops it partway through the ruleset.
    let mut cleanup_at: Option<usize> = None;
    let mut off_at: Option<usize> = None;
    if r.chance(12) {
        let at = r.below(n_rules + 1);
        cleanup_at = Some(at);
        if r.chance(35) {
            off_at = Some(at + 1 + r.below(n_rules + 1 - at));
        }
    }

    // A `syllables: clear` redeclaration partway through the rules drops
    // syllabification (and invalidates the skip-resyllabification cache);
    // only meaningful when syllables were declared in the first place.
    let clear_at = if !ph.syl_block.is_empty() && r.chance(15) {
        Some(1 + r.below(n_rules))
    } else {
        None
    };

    for i in 0..=n_rules {
        if clear_at == Some(i) {
            lsc += "syllables:\n    clear\n\n";
        }
        if cleanup_at == Some(i) {
            let e = gen_expr(&mut r, &ph);
            writeln!(lsc, "c0 cleanup:\n    {}\n", e.text).unwrap();
        }
        if off_at == Some(i) {
            lsc += "c0:\n    off\n\n";
        }
        if i == n_rules {
            break;
        }
        if inter_romanizer_at == Some(i) {
            let e = gen_expr(&mut r, &ph);
            writeln!(lsc, "romanizer-im:\n    {}\n", e.text).unwrap();
        }
        let sp = if splice_into == Some(i) {
            splice.as_ref()
        } else {
            None
        };
        lsc += &gen_rule(&mut r, &ph, i, allow_divergent, sp);
        lsc += "\n";
    }
    if r.chance(2) {
        // a bare `a => b` with no rule name: the grammar parses it but both
        // compilers reject it ("expression outside a named rule" / lexurgy's
        // "rule needs a name"): a pure reject-parity probe.
        let e = gen_expr(&mut r, &ph);
        writeln!(lsc, "{}\n", e.text).unwrap();
    }
    if r.chance(12) {
        let lit = if r.chance(25) { " literal" } else { "" };
        if r.chance(8) {
            writeln!(lsc, "romanizer{lit}:\n    unchanged\n").unwrap();
        } else {
            let e = gen_expr(&mut r, &ph);
            writeln!(lsc, "romanizer{lit}:\n    {}\n", e.text).unwrap();
        }
    }
    let words = (0..n_words).map(|_| gen_word(&mut r, &ph)).collect();
    (lsc, words)
}

// In-process check: fst tier vs reference VM

struct Finding {
    kind: &'static str,
    detail: String,
}

enum Outcome {
    ParseRej,
    CompRej,
    Ran { agree_ok: u64, agree_err: u64 },
}

/// Run `f` on a fresh thread; `None` if it doesn't finish in `timeout`.
/// The hung thread is leaked; callers must treat a `None` as "this process
/// is now tainted" and exit soon (worker processes do exactly that).
fn run_with_timeout<T: Send + 'static>(
    f: impl FnOnce() -> T + Send + 'static,
    timeout: Duration,
) -> Option<T> {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        tx.send(f()).ok();
    });
    rx.recv_timeout(timeout).ok()
}

fn panic_msg(p: Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = p.downcast_ref::<&str>() {
        s.to_string()
    } else if let Some(s) = p.downcast_ref::<String>() {
        s.clone()
    } else {
        "<non-string panic payload>".to_string()
    }
}

fn check_seed(seed: u64, n_words: usize, findings: &mut Vec<Finding>) -> Outcome {
    let (lsc, words) = gen_case(seed, n_words, true);
    let stmts = match catch_unwind(AssertUnwindSafe(|| lauturgie::parse(&lsc))) {
        Err(p) => {
            findings.push(Finding {
                kind: "parser-panic",
                detail: format!("{}\n---\n{lsc}", panic_msg(p)),
            });
            return Outcome::ParseRej;
        }
        Ok(Err(_)) => return Outcome::ParseRej,
        Ok(Ok(s)) => s,
    };
    let compiled = catch_unwind(AssertUnwindSafe(|| {
        (
            lauturgie::compiler::compile(&stmts),
            lauturgie::compiler::compile(&stmts),
        )
    }));
    let (mut fst, mut vm) = match compiled {
        Err(p) => {
            findings.push(Finding {
                kind: "compiler-panic",
                detail: format!("{}\n---\n{lsc}", panic_msg(p)),
            });
            return Outcome::CompRej;
        }
        Ok((Ok(a), Ok(b))) => (a, b),
        Ok(_) => return Outcome::CompRej,
    };
    vm.force_vm = true;

    let mut agree_ok = 0;
    let mut agree_err = 0;
    for w in &words {
        let a = catch_unwind(AssertUnwindSafe(|| fst.apply(w)));
        let b = catch_unwind(AssertUnwindSafe(|| vm.apply(w)));
        let ctx = |body: String| format!("word: {w:?}\n{body}\n---\n{lsc}");
        let finding = match (a, b) {
            (Err(p), _) => Some(("fst-panic", ctx(panic_msg(p)))),
            (_, Err(p)) => Some(("vm-panic", ctx(panic_msg(p)))),
            (Ok(Ok(x)), Ok(Ok(y))) => {
                if x == y {
                    agree_ok += 1;
                    None
                } else {
                    Some(("tier-mismatch", ctx(format!("fst: {x:?}\nvm:  {y:?}"))))
                }
            }
            (Ok(Err(_)), Ok(Err(_))) => {
                agree_err += 1;
                None
            }
            (Ok(Ok(x)), Ok(Err(e))) => {
                Some(("fst-ok-vm-err", ctx(format!("fst: {x:?}\nvm:  error: {e}"))))
            }
            (Ok(Err(e)), Ok(Ok(y))) => {
                Some(("fst-err-vm-ok", ctx(format!("fst: error: {e}\nvm:  {y:?}"))))
            }
        };
        if let Some((kind, detail)) = finding {
            findings.push(Finding { kind, detail });
            // one finding per case: later words usually share the root cause,
            // and a panicked changer's state is no longer trustworthy
            break;
        }
    }
    Outcome::Ran {
        agree_ok,
        agree_err,
    }
}

// Findings dir

fn findings_dir() -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fuzz_findings");
    std::fs::create_dir_all(&dir).ok();
    dir
}

fn save_case_files(dir: &Path, name: &str, lsc: &str, words: &[String], report: &str) {
    let existing = std::fs::read_dir(dir).map(|d| d.count()).unwrap_or(0);
    if existing > MAX_SAVED_FINDINGS {
        return; // keep printing findings, stop accumulating files
    }
    std::fs::write(dir.join(format!("{name}.lsc")), lsc).ok();
    std::fs::write(dir.join(format!("{name}.wli")), words.join("\n") + "\n").ok();
    std::fs::write(dir.join(format!("{name}.report.txt")), report).ok();
}

// Worker / driver (fst-vs-vm mode)

#[derive(Default)]
struct Stats {
    cases: u64,
    parse_rej: u64,
    comp_rej: u64,
    agree_ok: u64,
    agree_err: u64,
    findings: u64,
}

impl Stats {
    fn line(&self) -> String {
        format!(
            "STATS cases={} parse_rej={} comp_rej={} agree_ok={} agree_err={} findings={}",
            self.cases, self.parse_rej, self.comp_rej, self.agree_ok, self.agree_err, self.findings
        )
    }

    fn absorb(&mut self, line: &str) {
        for kv in line.split_whitespace() {
            let Some((k, v)) = kv.split_once('=') else {
                continue;
            };
            let Ok(v) = v.parse::<u64>() else { continue };
            match k {
                "cases" => self.cases += v,
                "parse_rej" => self.parse_rej += v,
                "comp_rej" => self.comp_rej += v,
                "agree_ok" => self.agree_ok += v,
                "agree_err" => self.agree_err += v,
                "findings" => self.findings += v,
                _ => {}
            }
        }
    }
}

fn worker_main(start: u64, count: u64, n_words: usize) {
    // panics are expected material, not noise; keep stderr to AT/STATS lines
    std::panic::set_hook(Box::new(|_| {}));
    let dir = findings_dir();
    let mut stats = Stats::default();
    for seed in start..start + count {
        eprintln!("AT {seed}");
        stats.cases += 1;
        let case = run_with_timeout(
            move || {
                let mut findings = Vec::new();
                let outcome = check_seed(seed, n_words, &mut findings);
                (outcome, findings)
            },
            CASE_TIMEOUT,
        );
        let Some((outcome, findings)) = case else {
            // watchdog fired: report here and exit; the hung thread is
            // still running and this process can't be trusted any further
            stats.findings += 1;
            let (lsc, words) = gen_case(seed, n_words, true);
            let detail = format!("case ran > {CASE_TIMEOUT:?} (likely diverging rule)\n---\n{lsc}");
            println!("\n=== FINDING seed {seed} [hang] ===\n{detail}");
            println!(
                "saved to fuzz_findings/{seed}.*; rerun: \
                 cargo run --release --example fuzz -- --seed {seed} --words {n_words}"
            );
            save_case_files(
                &dir,
                &seed.to_string(),
                &lsc,
                &words,
                &format!("seed {seed} [hang]\n{detail}\n"),
            );
            eprintln!("{}", stats.line());
            std::process::exit(EXIT_CASE_TIMEOUT);
        };
        match outcome {
            Outcome::ParseRej => stats.parse_rej += 1,
            Outcome::CompRej => stats.comp_rej += 1,
            Outcome::Ran {
                agree_ok,
                agree_err,
            } => {
                stats.agree_ok += agree_ok;
                stats.agree_err += agree_err;
            }
        }
        stats.findings += findings.len() as u64;
        if !findings.is_empty() {
            let (lsc, words) = gen_case(seed, n_words, true);
            for f in &findings {
                println!("\n=== FINDING seed {seed} [{}] ===\n{}", f.kind, f.detail);
                println!(
                    "saved to fuzz_findings/{seed}.*; rerun: \
                     cargo run --release --example fuzz -- --seed {seed} --words {n_words}"
                );
                save_case_files(
                    &dir,
                    &seed.to_string(),
                    &lsc,
                    &words,
                    &format!("seed {seed} [{}]\n{}\n", f.kind, f.detail),
                );
            }
        }
    }
    eprintln!("{}", stats.line());
}

/// Spawn ourselves with the given args, memory-capped via `ulimit -v`.
fn spawn_self(args: &[&str]) -> std::process::Child {
    let exe = std::env::current_exe().expect("current_exe");
    Command::new("/bin/sh")
        .arg("-c")
        .arg(format!(
            "ulimit -v {WORKER_MEM_KB} 2>/dev/null; exec \"$0\" \"$@\""
        ))
        .arg(exe)
        .args(args)
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn self")
}

fn spawn_worker(start: u64, count: u64, n_words: usize) -> std::process::Child {
    spawn_self(&[
        "--worker",
        "--start",
        &start.to_string(),
        "--count",
        &count.to_string(),
        "--words",
        &n_words.to_string(),
    ])
}

/// Wait for the child with a deadline. Returns (status, stderr, timed_out).
fn wait_with_timeout(
    mut child: std::process::Child,
    timeout: Duration,
) -> (std::process::ExitStatus, String, bool) {
    let deadline = Instant::now() + timeout;
    let mut timed_out = false;
    let status = loop {
        if let Some(st) = child.try_wait().expect("try_wait") {
            break st;
        }
        if Instant::now() > deadline {
            timed_out = true;
            child.kill().ok();
            break child.wait().expect("wait");
        }
        std::thread::sleep(Duration::from_millis(20));
    };
    let mut err = String::new();
    if let Some(mut pipe) = child.stderr.take() {
        pipe.read_to_string(&mut err).ok();
    }
    (status, err, timed_out)
}

fn drive(start: u64, n_words: usize) {
    let dir = findings_dir();
    println!(
        "fuzzing fst-vs-vm from seed {start} ({n_words} words/case)\n\
         findings -> {} ; stop with ctrl-c ; rerun a seed with --seed N",
        dir.display()
    );
    let t0 = Instant::now();
    let mut tot = Stats::default();
    let mut seed = start;
    loop {
        let child = spawn_worker(seed, CHUNK, n_words);
        let (status, errbuf, timed_out) = wait_with_timeout(child, WORKER_TIMEOUT);
        let last_at = errbuf
            .lines()
            .rev()
            .find_map(|l| l.strip_prefix("AT "))
            .and_then(|s| s.trim().parse::<u64>().ok());
        if let Some(st) = errbuf.lines().rev().find(|l| l.starts_with("STATS ")) {
            tot.absorb(st);
        }

        if status.code() == Some(EXIT_CASE_TIMEOUT) {
            // worker hit its per-case watchdog and already reported the
            // finding itself; just resume after the hung seed
            seed = last_at.map_or(seed + CHUNK, |at| at + 1);
        } else if timed_out || !status.success() {
            let bad = last_at.unwrap_or(seed);
            let kind = if timed_out { "hang-or-oom" } else { "crash" };
            let (lsc, words) = gen_case(bad, n_words, true);
            save_case_files(
                &dir,
                &bad.to_string(),
                &lsc,
                &words,
                &format!("seed {bad} [{kind}] worker status {status:?}\n---\n{lsc}"),
            );
            println!(
                "\n=== FINDING seed {bad} [{kind}] (status {status:?}) ===\n\
                 saved to fuzz_findings/{bad}.*; rerun: --seed {bad} --words {n_words}"
            );
            tot.findings += 1;
            tot.cases += 1;
            seed = bad + 1;
        } else {
            seed += CHUNK;
        }

        let secs = t0.elapsed().as_secs_f64();
        let pct = |n: u64| 100.0 * n as f64 / tot.cases.max(1) as f64;
        let words_ran = tot.agree_ok + tot.agree_err;
        println!(
            "[{:>8} cases] rej: parse {:.1}% compile {:.1}% | words agree: {} ok, {} err ({:.1}%) | {} findings | {:.0} cases/s | next seed {}",
            tot.cases,
            pct(tot.parse_rej),
            pct(tot.comp_rej),
            tot.agree_ok,
            tot.agree_err,
            100.0 * tot.agree_err as f64 / words_ran.max(1) as f64,
            tot.findings,
            tot.cases as f64 / secs.max(0.001),
            seed,
        );
    }
}

// Repro mode

#[derive(PartialEq)]
enum TierRes {
    Out(String),
    Error,
    Panic(String),
    Hang,
    Died(String),
}

impl TierRes {
    fn show(&self) -> String {
        match self {
            TierRes::Out(o) => format!("{o:?}"),
            TierRes::Error => "error".to_string(),
            TierRes::Panic(m) => format!("PANIC: {m}"),
            TierRes::Hang => format!("HANG (> {CASE_TIMEOUT:?})"),
            TierRes::Died(st) => format!("DIED ({st})"),
        }
    }
}

/// Apply one word on one tier in a sandboxed subprocess; a diverging rule
/// (which can allocate gigabytes in seconds) only costs that subprocess.
fn tier_run(lsc_path: &Path, wli_path: &Path, out_path: &Path, force_vm: bool) -> TierRes {
    std::fs::remove_file(out_path).ok();
    let child = spawn_self(&[
        "--ours",
        lsc_path.to_str().unwrap(),
        wli_path.to_str().unwrap(),
        out_path.to_str().unwrap(),
        if force_vm { "vm" } else { "fst" },
    ]);
    let (status, _, timed_out) = wait_with_timeout(child, CASE_TIMEOUT);
    if timed_out {
        return TierRes::Hang;
    }
    match status.code() {
        Some(0) => {
            let text = std::fs::read_to_string(out_path).unwrap_or_default();
            let line = text.lines().next().unwrap_or("").to_string();
            match line.as_str() {
                "ERROR" => TierRes::Error,
                "PANIC" => TierRes::Panic("panicked during apply".to_string()),
                _ => TierRes::Out(line),
            }
        }
        Some(c) if c == EXIT_OURS_PANICKED => {
            TierRes::Panic(std::fs::read_to_string(out_path).unwrap_or_default())
        }
        Some(c) if c == EXIT_OURS_REJECTED => TierRes::Died("ruleset rejected".to_string()),
        _ => TierRes::Died(format!("{status:?}")),
    }
}

fn repro(seed: u64, n_words: usize) {
    let (lsc, words) = gen_case(seed, n_words, true);
    println!(
        "--- lsc (seed {seed}) ---\n{lsc}--- words ---\n{}\n",
        words.join(" ")
    );
    // parse/compile never run user words, so checking them in-process is safe
    let stmts = match lauturgie::parse(&lsc) {
        Ok(s) => s,
        Err(e) => {
            println!("parse error: {e}");
            return;
        }
    };
    if let Err(e) = lauturgie::compiler::compile(&stmts) {
        println!("compile error: {e}");
        return;
    }
    let work = std::env::temp_dir().join(format!("lauturgie-fuzz-repro-{}", std::process::id()));
    std::fs::create_dir_all(&work).expect("create repro work dir");
    let lsc_path = work.join("r.lsc");
    let wli_path = work.join("r.wli");
    let out_path = work.join("r_out.wli");
    std::fs::write(&lsc_path, &lsc).unwrap();
    for w in &words {
        std::fs::write(&wli_path, format!("{w}\n")).unwrap();
        let a = tier_run(&lsc_path, &wli_path, &out_path, false);
        let b = tier_run(&lsc_path, &wli_path, &out_path, true);
        let verdict = match (&a, &b) {
            (TierRes::Out(x), TierRes::Out(y)) if x == y => "ok",
            (TierRes::Error, TierRes::Error) => "ok (both error)",
            (TierRes::Hang, TierRes::Hang) => "both hang (diverging rule)",
            _ => "*** MISMATCH ***",
        };
        println!("{w:12} fst: {:32} vm: {:32} {verdict}", a.show(), b.show());
    }
}

// Kotlin differential mode

fn nfc(s: &str) -> String {
    s.nfc().collect()
}

enum Ours {
    Rejected,
    Panicked(String),
    Lines(Vec<String>),
}

fn run_ours(lsc: &str, words: &[String], force_vm: bool) -> Ours {
    let stmts = match catch_unwind(AssertUnwindSafe(|| lauturgie::parse(lsc))) {
        Err(p) => return Ours::Panicked(format!("parser panic: {}", panic_msg(p))),
        Ok(Err(_)) => return Ours::Rejected,
        Ok(Ok(s)) => s,
    };
    let mut changer = match catch_unwind(AssertUnwindSafe(|| lauturgie::compiler::compile(&stmts)))
    {
        Err(p) => return Ours::Panicked(format!("compiler panic: {}", panic_msg(p))),
        Ok(Err(_)) => return Ours::Rejected,
        Ok(Ok(c)) => c,
    };
    changer.force_vm = force_vm;
    let lines = words
        .iter()
        .map(
            |w| match catch_unwind(AssertUnwindSafe(|| changer.apply(w))) {
                Ok(Ok(o)) => nfc(&o),
                Ok(Err(_)) => "ERROR".to_string(),
                Err(_) => "PANIC".to_string(),
            },
        )
        .collect();
    Ours::Lines(lines)
}

/// `--ours` mode: apply a ruleset file to a word list and write one output
/// line per word (or exit with a distinguishing code). Runs sandboxed so the
/// kotlin-differential driver survives hangs, panics, and OOMs on our side.
fn ours_main(lsc_path: &Path, wli_path: &Path, out_path: &Path, force_vm: bool) {
    std::panic::set_hook(Box::new(|_| {}));
    let lsc = std::fs::read_to_string(lsc_path).expect("read lsc");
    let text = std::fs::read_to_string(wli_path).expect("read wli");
    let words: Vec<String> = text
        .lines()
        .filter(|l| !l.is_empty())
        .map(String::from)
        .collect();
    match run_ours(&lsc, &words, force_vm) {
        Ours::Rejected => std::process::exit(EXIT_OURS_REJECTED),
        Ours::Panicked(msg) => {
            std::fs::write(out_path, msg).ok();
            std::process::exit(EXIT_OURS_PANICKED);
        }
        Ours::Lines(lines) => {
            std::fs::write(out_path, lines.join("\n") + "\n").expect("write ours out");
        }
    }
}

/// What the `--ours` subprocess reported, plus the failure modes only its
/// parent can observe (hangs and aborts).
enum OursProc {
    Hang,
    Crash(String),
    Rejected,
    Panicked(String),
    Lines(Vec<String>),
}

fn run_ours_sandboxed(lsc_path: &Path, wli_path: &Path, out_path: &Path) -> OursProc {
    std::fs::remove_file(out_path).ok();
    let child = spawn_self(&[
        "--ours",
        lsc_path.to_str().unwrap(),
        wli_path.to_str().unwrap(),
        out_path.to_str().unwrap(),
        "fst",
    ]);
    let (status, _, timed_out) = wait_with_timeout(child, OURS_TIMEOUT);
    if timed_out {
        return OursProc::Hang;
    }
    match status.code() {
        Some(0) => {
            let text = std::fs::read_to_string(out_path).unwrap_or_default();
            OursProc::Lines(text.lines().map(String::from).collect())
        }
        Some(c) if c == EXIT_OURS_REJECTED => OursProc::Rejected,
        Some(c) if c == EXIT_OURS_PANICKED => {
            OursProc::Panicked(std::fs::read_to_string(out_path).unwrap_or_default())
        }
        _ => OursProc::Crash(format!("{status:?}")),
    }
}

/// Run the kotlin CLI once per input line and collect per-line results (the
/// applied form, or `"ERROR"` when that line produced no output). The kotlin
/// CLI aborts the WHOLE batch on a single word-*parse* error (e.g. a `(first)`
/// diacritic that can't attach to its base), where our side reports just that
/// line as `ERROR`; so a batch failure needs a per-line re-check before it
/// counts as a real reject-disagreement. Slow (one JVM per line), but only
/// reached on the rare cases the batch run already flagged.
fn kotlin_per_line(cli: &Path, lsc_path: &Path, words: &[String], work: &Path) -> Vec<String> {
    let pw = work.join("pw.wli");
    let pw_out = work.join("pw_ev.wli");
    let pw_log = work.join("pw.log");
    words
        .iter()
        .map(|w| {
            std::fs::write(&pw, format!("{w}\n")).ok();
            std::fs::remove_file(&pw_out).ok();
            let log = std::fs::File::create(&pw_log).unwrap();
            let child = Command::new(cli)
                .args(["sc", "-e", "-S", "-V"])
                .arg(lsc_path)
                .arg(&pw)
                .env("LEXURGY_OPTS", "-XX:ActiveProcessorCount=1")
                .stdin(Stdio::null())
                .stdout(log.try_clone().unwrap())
                .stderr(log)
                .spawn()
                .expect("spawn kotlin per-line");
            let (status, _, timed_out) = wait_with_timeout(child, KOTLIN_TIMEOUT);
            if timed_out || !status.success() || !pw_out.exists() {
                return "ERROR".to_string();
            }
            std::fs::read_to_string(&pw_out)
                .unwrap_or_default()
                .lines()
                .next()
                .map(nfc)
                .unwrap_or_else(|| "ERROR".to_string())
        })
        .collect()
}

fn kotlin_cli() -> PathBuf {
    if let Ok(p) = std::env::var("LEXURGY_CLI") {
        return PathBuf::from(p);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("vendor/lexurgy/cli/build/install/lexurgy/bin/lexurgy")
}

fn kotlin_loop(start: u64, n_words: usize) {
    let cli = kotlin_cli();
    assert!(
        cli.is_file(),
        "Kotlin lexurgy CLI not found at {}; build it in vendor/lexurgy \
         (`./gradlew :cli:installDist`) or set $LEXURGY_CLI",
        cli.display()
    );
    let work = std::env::temp_dir().join(format!("lauturgie-fuzz-{}", std::process::id()));
    std::fs::create_dir_all(&work).expect("create work dir");
    let dir = findings_dir();
    println!(
        "fuzzing vs Kotlin lexurgy from seed {start} ({n_words} words/case)\n\
         cli: {} ; scratch: {} ; findings -> {}",
        cli.display(),
        work.display(),
        dir.display()
    );

    let lsc_path = work.join("t.lsc");
    let wli_path = work.join("t.wli");
    let out_path = work.join("t_ev.wli");
    let ours_path = work.join("t_ours.wli");
    let log_path = work.join("t.log");

    let mut agree = 0u64;
    let mut both_reject = 0u64;
    let mut findings = 0u64;
    let mut seed = start;
    loop {
        let (lsc, words) = gen_case(seed, n_words, false);
        std::fs::write(&lsc_path, &lsc).unwrap();
        std::fs::write(&wli_path, words.join("\n") + "\n").unwrap();

        let ours = run_ours_sandboxed(&lsc_path, &wli_path, &ours_path);

        std::fs::remove_file(&out_path).ok();
        let log = std::fs::File::create(&log_path).unwrap();
        let child = Command::new(&cli)
            .args(["sc", "-e", "-S", "-V"])
            .arg(&lsc_path)
            .arg(&wli_path)
            .env("LEXURGY_OPTS", "-XX:ActiveProcessorCount=1")
            .stdin(Stdio::null())
            .stdout(log.try_clone().unwrap())
            .stderr(log)
            .spawn()
            .expect("spawn kotlin cli");
        let (status, _, timed_out) = wait_with_timeout(child, KOTLIN_TIMEOUT);
        let loaded = status.success() && out_path.exists();
        let log_tail = || {
            let text = std::fs::read_to_string(&log_path).unwrap_or_default();
            let lines: Vec<&str> = text.lines().collect();
            lines[lines.len().saturating_sub(8)..].join("\n")
        };

        let mut report_finding = |kind: &str, detail: String| {
            findings += 1;
            let name = format!("kotlin_{seed}");
            println!("\n=== FINDING seed {seed} [{kind}] ===\n{detail}");
            println!(
                "saved to fuzz_findings/{name}.*; rerun: \
                 cargo run --release --example fuzz -- --seed {seed} --words {n_words}"
            );
            save_case_files(
                &dir,
                &name,
                &lsc,
                &words,
                &format!("seed {seed} [{kind}]\n{detail}\n---\n{lsc}"),
            );
        };

        let verdict;
        if timed_out {
            report_finding(
                "kotlin-timeout",
                format!("kotlin CLI ran > {KOTLIN_TIMEOUT:?}\n---\n{lsc}"),
            );
            verdict = "kotlin-timeout";
        } else {
            match (&ours, loaded) {
                (OursProc::Hang, _) => {
                    report_finding(
                        "lauturgie-hang",
                        format!(
                            "our side ran > {OURS_TIMEOUT:?} (kotlin's CLI rescues itself \
                             with a 1s/step timeout; we spin)\n---\n{lsc}"
                        ),
                    );
                    verdict = "lauturgie-hang";
                }
                (OursProc::Crash(st), _) => {
                    report_finding(
                        "lauturgie-crash",
                        format!("our side died (abort/oom?): {st}\n---\n{lsc}"),
                    );
                    verdict = "lauturgie-crash";
                }
                (OursProc::Panicked(msg), _) => {
                    report_finding("lauturgie-panic", format!("{msg}\n---\n{lsc}"));
                    verdict = "panic";
                }
                (OursProc::Rejected, false) => {
                    both_reject += 1;
                    verdict = "both-reject";
                }
                (OursProc::Rejected, true) => {
                    report_finding(
                        "we-reject-kotlin-accepts",
                        format!("lauturgie failed to parse/compile; kotlin ran fine\n---\n{lsc}"),
                    );
                    verdict = "reject-disagree";
                }
                (OursProc::Lines(ls), false) => {
                    // kotlin's batch load failed. Distinguish a real ruleset
                    // reject from the CLI's batch-abort-on-word-parse-error
                    // quirk by re-running kotlin line-by-line and comparing.
                    let klines = kotlin_per_line(&cli, &lsc_path, &words, &work);
                    let diffs: Vec<String> = words
                        .iter()
                        .zip(ls.iter().zip(&klines))
                        .filter(|(_, (a, b))| a != b)
                        .map(|(w, (a, b))| format!("  {w:?}: ours {a:?}, kotlin {b:?}"))
                        .collect();
                    if diffs.is_empty() {
                        // every line agrees once kotlin is run per-line; the
                        // batch failure was only the CLI granularity quirk
                        agree += 1;
                        verdict = "agree(per-line)";
                    } else {
                        let n = diffs.len();
                        let shown = diffs[..n.min(10)].join("\n");
                        report_finding(
                            "we-accept-kotlin-rejects",
                            format!(
                                "kotlin batch-rejected; per-line {n} of {} differ:\n{shown}\n\
                                 (batch log: {})\n---\n{lsc}",
                                ls.len(),
                                log_tail()
                            ),
                        );
                        verdict = "reject-disagree";
                    }
                }
                (OursProc::Lines(ls), true) => {
                    let ktext = std::fs::read_to_string(&out_path).unwrap_or_default();
                    let klines: Vec<String> = ktext.lines().map(nfc).collect();
                    if klines.len() != ls.len() {
                        report_finding(
                            "line-count-mismatch",
                            format!("ours {} lines, kotlin {} lines", ls.len(), klines.len()),
                        );
                        verdict = "mismatch";
                    } else {
                        let diffs: Vec<String> = words
                            .iter()
                            .zip(ls.iter().zip(&klines))
                            .filter(|(_, (a, b))| a != b)
                            .map(|(w, (a, b))| format!("  {w:?}: ours {a:?}, kotlin {b:?}"))
                            .collect();
                        if diffs.is_empty() {
                            agree += 1;
                            verdict = "agree";
                        } else {
                            let n = diffs.len();
                            let shown = diffs[..n.min(10)].join("\n");
                            report_finding(
                                "output-mismatch",
                                format!(
                                    "{n} of {} words differ:\n{shown}\n\
                                     (note: kotlin \"ERROR\" can be its 1s/step CLI timeout)",
                                    ls.len()
                                ),
                            );
                            verdict = "mismatch";
                        }
                    }
                }
            }
        }
        println!(
            "[kotlin] seed {seed}: {verdict:14} | {agree} agree, {both_reject} both-reject, {findings} findings"
        );
        seed += 1;
    }
}

// main

fn random_start() -> u64 {
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap();
    t.as_nanos() as u64 % 1_000_000_000_000
}

fn main() {
    let mut args = std::env::args().skip(1);
    let mut start: Option<u64> = None;
    let mut seed: Option<u64> = None;
    let mut count: u64 = CHUNK;
    let mut words: Option<usize> = None;
    let mut kotlin = false;
    let mut worker = false;
    let mut ours: Option<(String, String, String, String)> = None;
    let next_num = |args: &mut dyn Iterator<Item = String>, flag: &str| -> u64 {
        args.next()
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(|| panic!("{flag} needs a number"))
    };
    while let Some(a) = args.next() {
        match a.as_str() {
            "--worker" => worker = true,
            "--kotlin" => kotlin = true,
            "--ours" => {
                let mut p = || args.next().expect("--ours LSC WLI OUT fst|vm");
                ours = Some((p(), p(), p(), p()));
            }
            "--start" => start = Some(next_num(&mut args, "--start")),
            "--seed" => seed = Some(next_num(&mut args, "--seed")),
            "--count" => count = next_num(&mut args, "--count"),
            "--words" => words = Some(next_num(&mut args, "--words") as usize),
            other => {
                eprintln!(
                    "unknown arg {other}\nusage: fuzz [--start N] [--words K] \
                     | fuzz --seed N [--words K] | fuzz --kotlin [--start N] [--words K]"
                );
                std::process::exit(2);
            }
        }
    }
    if let Some((lsc, wli, out, tier)) = ours {
        ours_main(
            Path::new(&lsc),
            Path::new(&wli),
            Path::new(&out),
            tier == "vm",
        );
    } else if let Some(s) = seed {
        repro(s, words.unwrap_or(10));
    } else if worker {
        worker_main(
            start.expect("--worker needs --start"),
            count,
            words.unwrap_or(10),
        );
    } else if kotlin {
        kotlin_loop(start.unwrap_or_else(random_start), words.unwrap_or(120));
    } else {
        drive(start.unwrap_or_else(random_start), words.unwrap_or(10));
    }
}
