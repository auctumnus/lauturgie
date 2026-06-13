# lauturgie

Lauturgie is a Rust port / implementation of [Lexurgy] with improved
performance and identical[^1] output. On a real-world set of words and changes,
Lauturgie achieves a 340x speedup over base Lexurgy; for a larger, slightly
more synthetic workload, Lauturgie achieves between 16x and 43x speedup.

This performance comes from two areas:
- Reduced allocations and GC churn compared to Kotlin.
- A compiled layer which makes use of [finite-state transducers].

In the compiled layer, the largest win is compiling patterns and lazily
determinizing them into a DFA. This is significantly faster than the VM layer,
which is ~largely a port of the original Lexurgy rule application.

[^1]: This is informed by an extensive test suite with fuzzing to find any
differences in behavior. 

## Development

Most scripts you will want when developing are available in the `Justfile`.
Running the differential suite after changes to the rule application system is
generally a good idea, to catch stray bugs.

## Usage

Lauturgie ships both a command-line tool and a Rust library. The sound-change
language is the same as Lexurgy's `.lsc` format; see the [Lexurgy docs] for
help.

### As a CLI

Apply a `.lsc` change file to a word list (one word or phrase per line):

```sh
lauturgie CHANGES.lsc WORDS [OPTIONS]
```

For example, against one of the vendor test sets:

```sh
lauturgie vendor/lexurgy/cli/test/kharulian.lsc \
          vendor/lexurgy/cli/test/kharulian.wli
```

The changed words are written to stdout (or `-o FILE`), one per input line. A
word that can't be applied is reported on stderr and emitted as `ERROR` on its
line, so output stays aligned with input.

Read the word list from stdin with `-`:

```sh
echo "kasa" | lauturgie changes.lsc -
```

Options:

| Flag | Meaning |
| --- | --- |
| `-o, --output <FILE>` | write evolved words here (default: stdout) |
| `--vm` | force the reference VM tier (disable the FST tier) |
| `-1, --single-thread` | apply lines sequentially (default: all cores) |
| `-q, --quiet` | suppress the summary line on stderr |
| `-h, --help` | print help |
| `-V, --version` | print the version |

Exit status: `0` = all words applied, `1` = some words errored (`ERROR` in the
output, details on stderr), `2` = the changes file couldn't be read or compiled.

During development you can run it through the `Justfile` instead of installing:

```sh
just run CHANGES.lsc WORDS         # cargo run --release --bin lauturgie -- ...
```

### As a Rust library

Add the crate, then compile a ruleset once and apply it to words:

```rust
use lauturgie::{parse, compiler};

let src = std::fs::read_to_string("changes.lsc")?;
let statements = parse(&src)?;                 // parse + validate -> AST
let mut rules = compiler::compile(&statements)?; // lower into CompiledRules

// One word at a time (takes &mut self: interners/DFA caches fill in lazily):
let evolved = rules.apply("kasa")?;
println!("{evolved}");

// Or a whole list in parallel across cores (takes &self):
let words = ["kasa", "tupa", "milo"];
for result in rules.apply_all(&words) {
    println!("{}", result?);
}
# Ok::<(), Box<dyn std::error::Error>>(())
```

`compile` returns a [`CompiledRules`]; set `rules.force_vm = true` to pin it to
the reference VM tier (the FST tier is otherwise selected per rule
automatically). `apply` returns `Err` for a word the ruleset can't evolve;
`compile`/`parse` return `Err` for a malformed ruleset.

[Lexurgy docs]: https://lexurgy-app.vercel.app/sc/docs
[`CompiledRules`]: https://docs.rs/lauturgie/latest/lauturgie/compiler/struct.CompiledRules.html

## License

GPL-3, due to being kind-of-a-port.

## Naming

Lexurgy:
- lexis:  (A.Gr.) saying, word, phrase
- ourgia: (A.Gr.) working with

Lauturgy:
- Laut:   (Ger.)  sound, or as an adjective "loud"
- ourgia: (A.Gr.) (id.)

You could also make a case that it kind of sounds like "lauter", louder, so
it's got a bit of a comparatively-larger feel, like how it's comparatively
faster...

[Lexurgy]: https://lexurgy-app.vercel.app/sc
[finite-state transducers]: https://en.wikipedia.org/wiki/Finite-state_transducer
