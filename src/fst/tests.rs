// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) Autumn <auctumnus@pm.me>

use super::*;
use crate::parse;

fn changer(source: &str) -> crate::compiler::CompiledRules {
    crate::compiler::compile(&parse(source).unwrap()).unwrap()
}

fn ab(source: &str, words: &[&str]) {
    let mut fast = changer(source);
    let mut slow = changer(source);
    slow.force_vm = true;
    for word in words {
        assert_eq!(
            fast.apply(word),
            slow.apply(word),
            "tiers disagree on {word:?} under:\n{source}"
        );
    }
}

#[test]
fn simple_rules_take_the_fst_tier() {
    let compiled = changer(
        "Feature Type(*cons, vowel)\n\
             Symbol a [vowel]\n\
             simple:\n  a => e / _ $\n\
             matrixy:\n  [vowel] => [cons]\n\
             captureful:\n  (a)$1 $1 => $1\n",
    );
    let tiers: Vec<bool> = compiled.fst_rules.iter().map(|f| f.is_some()).collect();
    assert_eq!(tiers, vec![true, true, false]);
}

#[test]
fn fst_and_vm_agree() {
    let source = "Class stop {p, t, k}\n\
                      Class vowel {a, e, i, o, u}\n\
                      lenite:\n  @stop => h / @vowel _ @vowel // _ {i, u}\n\
                      drop-final:\n  {a, e}+ => * / _ $\n";
    let words = ["papa", "patu", "katikate", "aaa", "pkat", ""];
    let mut fast = changer(source);
    assert!(fast.fst_rules.iter().all(|f| f.is_some()));
    let mut slow = changer(source);
    slow.force_vm = true;
    for word in words {
        assert_eq!(
            fast.apply(word).unwrap(),
            slow.apply(word).unwrap(),
            "tiers disagree on {word:?}"
        );
    }
}

#[test]
fn paired_alternations_take_the_fst_tier() {
    let compiled = changer("voice:\n  {p, t, k} => {b, d, g}\n");
    assert!(compiled.fst_rules[0].is_some());
    let mut compiled = compiled;
    assert_eq!(compiled.apply("ptak").unwrap(), "bdag");
    ab("voice:\n  {p, t, k} => {b, d, g}\n", &["ptak", "appa", ""]);
    // sequences distributing into alternative lists
    ab(
        "breaking:\n  a {e, o} => {e e, o o}\n",
        &["kaek", "kaok", "ka", "aeao"],
    );
    // mixed emitters per branch
    ab(
        "Feature Voice(*vl, vd)\n\
             Symbol b [vd]\n\
             mix:\n  {p, t} => {[vd], s}\n",
        &["pata", "tap"],
    );
    // priority: written order decides the variant for a shared end
    ab("pri:\n  {a, ab} => {x, y}\n", &["ab", "a", "aab"]);
    ab("pri:\n  {ab, a} => {x, y}\n", &["ab", "a", "aab"]);
}

#[test]
fn directional_rules_take_the_fst_tier() {
    for mode in ["ltr", "rtl"] {
        let source = format!("squash {mode}:\n  aa => b\n");
        let compiled = changer(&source);
        assert!(compiled.fst_rules[0].is_some(), "{mode} not on fst tier");
        ab(&source, &["aaaa", "aaa", "aa", "a", "baab", ""]);
    }
    // deletion shifts indices differently per direction
    ab("del ltr:\n  a => * / _ b\n", &["aabab", "abab"]);
    ab("del rtl:\n  a => * / _ b\n", &["aabab", "abab"]);
}

#[test]
fn zero_width_matrix_pieces_render_alone() {
    // `* a => [vowel]` inserts the rendered matrix for the `*` piece;
    // missing it was a pre-existing fst-tier bug (kotlin: "bqq").
    let source = "Feature Type(*cons, vowel)\nFeature Voice(*vl, vd)\n\
                      Symbol q [vowel]\nSymbol b [cons vd]\nSymbol a [cons]\n\n\
                      r:\n * a => [vowel]\n";
    let mut fast = changer(source);
    assert!(fast.fst_rules[0].is_some());
    assert_eq!(fast.apply("ba").unwrap(), "bqq");
    ab(source, &["ba", "ab", "a", "b", ""]);
    // a zero-width *repeat* emits nothing instead
    ab(
        "Feature Type(*cons, vowel)\nFeature Voice(*vl, vd)\n\
             Symbol q [vowel]\nSymbol b [cons vd]\nSymbol a [cons]\n\n\
             r:\n a* => [vowel]\n",
        &["ba", "b", ""],
    );
}

#[test]
fn floating_diacritics_transfer_on_the_fst_tier() {
    // SymbolEmitter.result's three length cases plus element-wise
    // sequence pairing; outputs kotlin-CLI-verified.
    let source = "Feature Stress(*unstressed, stressed)\n\
                      Feature Height(low, mid, high)\n\
                      Feature Type(*cons, vowel)\n\
                      Diacritic ́  (floating) [stressed]\n\
                      Symbol a [vowel low]\nSymbol e [vowel mid]\n\
                      Symbol x [cons low]\nSymbol y [cons mid]\nSymbol b [cons high]\n\n\
                      expand:\n    a => ee / b _\n\
                      collapse:\n    ea => x / _ $\n\
                      pairwise:\n    ba => xy / $ _\n\
                      seq-pairwise:\n    e x => y b / $ _\n";
    let mut fast = changer(source);
    assert!(fast.fst_rules.iter().all(|f| f.is_some()));
    for (word, want) in [
        ("bá", "béé"),
        ("beá", "bx́"),
        ("báeá", "bééx́"),
        ("éxe", "ýbe"),
    ] {
        assert_eq!(fast.apply(word).unwrap(), want, "on {word:?}");
    }
    ab(source, &["bá", "beá", "báeá", "éxe", "ba", ""]);
}

#[test]
fn blocks_and_filters_take_the_fst_tier() {
    // Filtered Then:, a filtered propagate harmony with an env-bound
    // feature variable, filtered surplus dropping, and Else:; outputs
    // kotlin-CLI-verified.
    let source = "Feature Height(low, mid, high)\n\
                      Feature Type(*cons, vowel)\n\
                      Symbol a [vowel low]\nSymbol e [vowel mid]\nSymbol i [vowel high]\n\
                      Symbol b [cons low]\nSymbol t [cons mid]\n\n\
                      raise [vowel]:\n    a => e / _ e\n    Then:\n    e => i / i _\n\
                      height-harmony [vowel] propagate:\n    [vowel] => [$Height] / [vowel $Height] _\n\
                      surplus [vowel]:\n    {a, e, i} i => t a\n\
                      else-pick:\n    a => e / $ _\n    Else:\n    e => a\n";
    let mut fast = changer(source);
    assert!(fast.fst_rules.iter().all(|f| f.is_some()));
    for (word, want) in [
        ("baebi", "baaba"),
        ("taeie", "taaaa"),
        ("biebia", "btabta"),
        ("aeia", "aaaa"),
        ("btaibi", "btaaba"),
    ] {
        assert_eq!(fast.apply(word).unwrap(), want, "on {word:?}");
    }
    ab(source, &["baebi", "taeie", "biebia", "aeia", "btaibi", ""]);
}

#[test]
fn variable_expansion_intersects_same_feature_values() {
    // `[lab $place]` can only ever bind place=lab; clobbering the
    // concrete value with each instance made strays match (fuzzer
    // seed 2204, 98 findings from one soak).
    let source = "Feature Voice(uvc, vcd)\nFeature Place(lab, alv, vel)\n\
                      Symbol p [lab uvc]\nSymbol b [lab vcd]\nSymbol t [alv uvc]\n\n\
                      r:\n    [lab $Place] u => a z\n";
    let mut fast = changer(source);
    assert!(fast.fst_rules[0].is_some());
    // m is a stray (featureless): must not match [lab $Place]
    assert_eq!(fast.apply("mu").unwrap(), "mu");
    assert_eq!(fast.apply("pu").unwrap(), "az");
    ab(source, &["mu", "pu", "bu", "tu", ""]);
}

#[test]
fn variables_under_negation_stay_on_the_vm() {
    // `![vel $manner]` means "no manner value matches here"; instance
    // expansion would flip that ∃/∀ (fuzzer seed 2097233).
    let source = "Feature Voice(uvc, vcd)\nFeature Place(lab, alv, vel)\n\
                      Feature Manner(stp, frc)\n\
                      Symbol k [vel stp uvc]\nSymbol g [vel stp vcd]\n\n\
                      r:\n    * => k / $ ![vel !uvc $Manner] _\n";
    let fast = changer(source);
    assert!(fast.fst_rules[0].is_none(), "must stay on the VM");
    ab(source, &["gpa", "yno", "kk"]);
}

#[test]
fn env_variables_with_ambiguous_bindings_stay_on_the_vm() {
    // The before-side alternative `[vcd $place]` binds place=lab at
    // the `v`, and kotlin *commits* to it even though the after side
    // then fails and the bindingless `@cn` branch would have passed;
    // instance expansion would explore that value, so the rule must
    // stay on the VM (fuzzer seed 10412276; kotlin: "eezavzez").
    let source = "feature voice(uvc, vcd)\nfeature place(lab, alv, vel)\n\
                      feature manner(stp, frc)\n\
                      symbol s [alv frc uvc]\nsymbol z [alv frc vcd]\n\
                      symbol f [lab frc uvc]\nsymbol v [lab frc vcd]\n\
                      class cv {f, z, s}\nclass cn {e, v}\n\n\
                      r0:\n    * => e / {[alv frc], [vcd $place], @cn} _ [!uvc vcd $place]\n";
    let mut fast = changer(source);
    assert!(fast.fst_rules[0].is_none(), "must stay on the VM");
    assert_eq!(fast.apply("ezavzz").unwrap(), "eezavzez");
    ab(source, &["ezavzz", "vz", "ez", ""]);
}

#[test]
fn disjoint_repeats_keep_env_bindings_deterministic() {
    // The echo-vowel shape: the variable's binder is separated from the
    // match by a repeat whose inner (`[gl]`) contradicts the binder's
    // own `!gl`, so the repeat count (and the binding spot) is forced
    // and the rule may expand onto the FST tier. All outputs
    // kotlin-CLI-verified.
    let decls = "Feature Type(cons, vowel)\nFeature Height(*mid, low, high)\n\
                     Feature Gl(*ngl, gl)\n\
                     Symbol a [vowel low]\nSymbol i [vowel high]\nSymbol e [vowel]\n\
                     Symbol j [vowel high gl]\nSymbol p [cons]\nSymbol t [cons high]\n";
    let before = format!("{decls}\nr1:\n    e => [$Height] / [$Height vowel !gl] [gl]* _\n");
    let mut fast = changer(&before);
    assert!(
        fast.fst_rules[0].is_some(),
        "binding is forced: fst-eligible"
    );
    for (word, want) in [
        ("paje", "paja"),
        ("pije", "piji"),
        ("pe", "pe"),
        ("paije", "paiji"),
        ("pajje", "pajja"),
        ("je", "je"),
    ] {
        assert_eq!(fast.apply(word).unwrap(), want);
    }
    ab(&before, &["paje", "pije", "pe", "paije", "pajje", "je", ""]);
    // Mirror image: the binder sits past the repeat on the after side.
    let after = format!("{decls}\nr3:\n    e => [$Height] / _ [gl]* [$Height vowel !gl]\n");
    let mut fast = changer(&after);
    assert!(fast.fst_rules[0].is_some());
    for (word, want) in [("eja", "aja"), ("ejji", "ijji"), ("ej", "ej"), ("ea", "aa")] {
        assert_eq!(fast.apply(word).unwrap(), want);
    }
    ab(&after, &["eja", "ejji", "ej", "ea"]);
    // A fixed element may sit between the repeat and the binder; the
    // disjointness check then applies to *it* (`[!gl cons]` vs `[gl]`).
    let spaced =
        format!("{decls}\nr4:\n    e => [$Height] / [$Height vowel !gl] [!gl cons] [gl]* _\n");
    let mut fast = changer(&spaced);
    assert!(fast.fst_rules[0].is_some());
    for (word, want) in [
        ("atje", "atja"),
        ("itje", "itji"),
        ("apje", "apja"),
        ("ate", "ata"),
        ("aje", "aje"),
    ] {
        assert_eq!(fast.apply(word).unwrap(), want);
    }
    ab(&spaced, &["atje", "itje", "apje", "ate", "aje"]);
}

#[test]
fn overlapping_repeats_keep_env_variables_on_the_vm() {
    // `[vowel]*` can also match the binder `[$Height vowel]`, so the
    // binding spot depends on the repeat count kotlin's greedy first
    // match picks: "paie" binds the `a` (repeat swallows `i`), where
    // instance expansion would happily accept the `i` binding too.
    // Kotlin: paie→paia, pie→pii, paiie→paiia.
    let source = "Feature Type(cons, vowel)\nFeature Height(*mid, low, high)\n\
                      Feature Gl(*ngl, gl)\n\
                      Symbol a [vowel low]\nSymbol i [vowel high]\nSymbol e [vowel]\n\
                      Symbol j [vowel high gl]\nSymbol p [cons]\n\n\
                      r2:\n    e => [$Height] / [$Height vowel] [vowel]* _\n";
    let mut fast = changer(source);
    assert!(
        fast.fst_rules[0].is_none(),
        "ambiguous binding: must stay on the VM"
    );
    for (word, want) in [
        ("paie", "paia"),
        ("pie", "pii"),
        ("paiie", "paiia"),
        ("ppe", "ppe"),
    ] {
        assert_eq!(fast.apply(word).unwrap(), want);
    }
}

#[test]
fn dropped_filtered_subs_are_never_evaluated() {
    // The colliding matrix piece lands on the featureless multigraph
    // `ui`; the VM drops it before binding, so no "doesn't spell"
    // error (fuzzer seed 1009050).
    let source = "Feature Place(lab, alv, vel)\n\
                      Symbol p [lab]\nSymbol t [alv]\nSymbol ui\n\
                      Class cv {ui, p, o}\n\n\
                      r @cv ltr:\n    @cv* ui => * [lab]\n";
    let mut fast = changer(source);
    assert!(fast.fst_rules[0].is_some());
    assert_eq!(fast.apply("uiu").unwrap(), "u");
    ab(source, &["uiu", "ui", "puiu", ""]);
}

#[test]
fn equal_start_subs_keep_precedence_order() {
    // Two filtered claims whose pieces collide on the same real
    // segment: the earlier *expression*'s piece must win even though
    // its claim starts later (fuzzer seed 2362808; kotlin: "bbgb").
    let source = "Class cv {e, t, p, d}\n\n\
                      r0 @cv:\n    d? e => b k\n    t p* => * *\n";
    let mut fast = changer(source);
    assert!(fast.fst_rules[0].is_some());
    assert_eq!(fast.apply("teege").unwrap(), "bbgb");
    ab(source, &["teege", "te", "et", "dpe", ""]);
}

#[test]
fn syllable_rules_gate_the_vm() {
    // Syllabified words run on the VM with the rule's FST as a
    // position gate: `.` in environments, `@vowel&[+hv]` intersections,
    // and `<syl>` with a syllable-feature emit all compile (match-only
    // where the emit touches structure), and outputs (including the
    // multi-word phrase) are byte-identical to the kotlin CLI.
    let source = "Feature (syllable) +hv\nFeature Type(cons, vowel)\n\
                      Feature Height(*low, high, round)\nFeature Place(*lab, alv)\n\
                      Diacritic ² [+hv]\n\
                      Symbol a [vowel]\nSymbol i [vowel high]\nSymbol u [vowel round]\n\
                      Symbol p [cons]\nSymbol t [cons alv]\n\
                      Class cons {p, t}\nClass vowel {a, i, u}\n\n\
                      Syllables:\n    @cons? @vowel @cons => [+hv]\n    @cons? @vowel\n\n\
                      boundary-env:\n    a => i / _ @cons .\n\n\
                      heavy-vowel:\n    @vowel&[+hv] => u\n\n\
                      initial-heavy:\n    <syl> => [+hv] / $ _\n";
    let mut fast = changer(source);
    for i in 0..3 {
        let fst = fast.fst_rules[i].as_ref().expect("compiles for the gate");
        assert!(!fst.splices(), "structure-touching emits are match-only");
    }
    for (word, want) in [
        ("patu", "pa².tu"),
        ("ata", "a².ta"),
        ("u", "u²"),
        ("apta", "up².ta"),
        ("tipat", "ti².put²"),
        ("tip ta", "tup² ta²"),
    ] {
        assert_eq!(fast.apply(word).unwrap(), want);
    }
    ab(
        source,
        &["patu", "ata", "u", "apta", "tipat", "tip ta", "ptu", ""],
    );
}

#[test]
fn syllable_matrices_on_unsyllabified_words_splice() {
    // Without a syllabifier in force, a mixed-level matrix tests the
    // all-defaults syllable word (`[-hv]` holds, `[+hv]` never), and
    // the rule splices on the full FST path. Kotlin: apa→ppp.
    let source = "Feature (syllable) +hv\nFeature Type(cons, vowel)\n\
                      Feature Height(*low, high)\nDiacritic ² [+hv]\n\
                      Symbol a [vowel]\nSymbol i [vowel high]\nSymbol p [cons]\n\n\
                      defaults-match:\n    [vowel -hv] => p\n\n\
                      defaults-never:\n    [vowel +hv] => i\n";
    let mut fast = changer(source);
    assert!(fast.fst_rules[0].as_ref().unwrap().splices());
    assert!(fast.fst_rules[1].as_ref().unwrap().splices());
    for (word, want) in [("apa", "ppp"), ("pap", "ppp"), ("aaa", "ppp")] {
        assert_eq!(fast.apply(word).unwrap(), want);
    }
    ab(source, &["apa", "pap", "aaa", "ipi", ""]);
}

#[test]
fn any_syllable_matches_zero_width_at_bounding_breaks() {
    // Deleting a word's tail mid-rule leaves a *bounding* break at the
    // end, and the VM's `syllable_from` then gives `<syl>` a zero-width
    // match there, so the rtl scan keeps deleting. A boundary-anchor
    // automaton can't express that (the cases are direction-asymmetric),
    // which is why `<syl>` is a direct `syllable_span` port (fuzzer
    // seed 30010144; kotlin: fivii → "").
    let source = "Feature Type(cons, vowel)\nFeature Voice(*uvc, vcd)\n\
                      Symbol f [cons]\nSymbol v [cons vcd]\nSymbol i [vowel]\n\
                      Symbol u [vowel vcd]\n\
                      Class con {f, v}\nClass vow {i, u}\n\n\
                      Syllables:\n    @con? @vow\n\n\
                      r1 rtl:\n    @con [] <syl> => * * *\n";
    let mut fast = changer(source);
    assert!(fast.fst_rules[0].is_some());
    for (word, want) in [("fivii", ""), ("i", "i"), ("uu", "u.u"), ("fifi", "")] {
        assert_eq!(fast.apply(word).unwrap(), want, "on {word:?}");
    }
    ab(source, &["fivii", "i", "uu", "fifi", "fiv", ""]);
}

#[test]
fn adjacent_seg_maps_fuse() {
    let source = "one:\n  p => b\n\
                      two:\n  b => v\n\
                      three:\n  {a, e} => {e, i}\n";
    let mut compiled = changer(source);
    assert_eq!(compiled.fused.len(), 1, "expected one fused run");
    assert_eq!(compiled.fused[0].rules.len(), 3);
    assert_eq!(compiled.apply("pat").unwrap(), "vet");
    assert_eq!(compiled.apply("bee").unwrap(), "vii");
    ab(source, &["pat", "bee", "ppbb", ""]);
    // contextful rules break the run
    let gated = changer("one:\n  p => b\ntwo:\n  b => v / a _\n");
    assert!(gated.fused.is_empty());
}
