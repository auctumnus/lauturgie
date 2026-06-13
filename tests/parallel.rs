//! `CompiledRules::apply_all` (rayon, per-worker cloned runtime state) must
//! agree with the sequential `apply` loop result-for-result, including
//! errors and ordering. The word lists are large enough to actually split
//! across workers, and the rulesets deliberately mix tiers: fst-eligible
//! rules, vm-only rules (captures), syllabification, filters.

fn changer(source: &str) -> lauturgie::compiler::CompiledRules {
    let statements = lauturgie::parse(source).expect("parse failed");
    lauturgie::compiler::compile(&statements).expect("compile failed")
}

/// Deterministic word soup over the ruleset's alphabet, sized well past
/// `apply_all`'s split threshold.
fn words() -> Vec<String> {
    let onsets = ["p", "t", "k", "s", "m", "n", ""];
    let nuclei = ["a", "e", "i", "o", "u"];
    let codas = ["", "n", "s", "t"];
    let mut out = Vec::new();
    for o1 in onsets {
        for n1 in nuclei {
            for c1 in codas {
                for n2 in nuclei {
                    out.push(format!("{o1}{n1}{c1}{n2}"));
                }
            }
        }
    }
    out
}

#[test]
fn parallel_agrees_with_sequential() {
    let source = "Feature Type(*cons, vowel)\n\
                  Feature Height(*none, low, mid, high)\n\
                  Feature +round\n\
                  Feature +nasal\n\
                  Symbol a [vowel low]\n\
                  Symbol e [vowel mid]\n\
                  Symbol i [vowel high]\n\
                  Symbol o [vowel mid +round]\n\
                  Symbol u [vowel high +round]\n\
                  Symbol m [+nasal +round]\n\
                  Symbol n [+nasal]\n\
                  Class stop {p, t, k}\n\
                  lenite:\n  @stop => {b, d, g} / [vowel] _ [vowel]\n\
                  echo:\n  ([vowel])$1 s => $1 $1\n\
                  Syllables:\n  explicit\n\
                  cluster-drop:\n  [cons] => * / _ [cons] [cons]\n\
                  filtered-shift [vowel]:\n  {a, e, i} => {e, i, a}\n";
    let words = words();
    assert!(
        words.len() > 500,
        "need enough words to split across workers"
    );

    let mut sequential = changer(source);
    let expected: Vec<_> = words.iter().map(|w| sequential.apply(w)).collect();
    assert!(expected.iter().any(|r| r.is_ok()));

    let parallel = changer(source).apply_all(&words);
    assert_eq!(parallel.len(), expected.len());
    for ((word, seq), par) in words.iter().zip(&expected).zip(&parallel) {
        assert_eq!(par, seq, "parallel and sequential disagree on {word:?}");
    }
}

/// Spell an index as a distinct all-vowel tail, so every word is unique and
/// any result-order slip shows up as a mismatch.
fn enc(mut i: usize) -> String {
    let digits = ['a', 'e', 'i'];
    let mut s = String::from("p");
    loop {
        s.push(digits[i % 3]);
        i /= 3;
        if i == 0 {
            return s;
        }
    }
}

#[test]
fn parallel_preserves_error_positions() {
    // every 7th word ends in an unsyllabifiable coda cluster and must
    // error *in place*, not shift its neighbors' results
    let source = "Feature Type(*cons, vowel)\n\
                  Feature Height(*none, low, mid, high)\n\
                  Symbol a [vowel low]\n\
                  Symbol e [vowel mid]\n\
                  Symbol i [vowel high]\n\
                  Syllables:\n  [cons]? [vowel] [cons]?\n\
                  shift:\n  a => e\n";
    let words: Vec<String> = (0..300)
        .map(|i| {
            if i % 7 == 3 {
                format!("{}strn", enc(i))
            } else {
                enc(i)
            }
        })
        .collect();

    let mut sequential = changer(source);
    let expected: Vec<_> = words.iter().map(|w| sequential.apply(w)).collect();
    assert!(expected.iter().any(|r| r.is_ok()));
    assert!(expected.iter().any(|r| r.is_err()));

    let parallel = changer(source).apply_all(&words);
    assert_eq!(parallel, expected);
}
