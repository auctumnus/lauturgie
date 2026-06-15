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

[^1]: This is informed by an extensive test suite plus a differential fuzzer
(`examples/fuzz.rs`) that generates random rulesets and word lists and checks
output two ways: Lauturgie's own fast (FST) and reference (VM) tiers against
each other, and the whole engine against the original Kotlin Lexurgy CLI as an
oracle. In the most recent campaign the two tiers agreed on 326,000 random
cases, and the engine matched Lexurgy byte-for-byte over ~47,000 cases
(~5.7M word comparisons). The only divergences are
rulesets that don't terminate (zero-width insertions or unbounded growth),
where Lexurgy's CLI gives up at its one-second-per-step timeout while Lauturgie
either finishes or reports a bounded per-word error.

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

For the richer result that the HTTP API needs — intermediate-romanizer stages,
per-word tracing, `startAt`/`stopBefore`, and structured per-word errors — use
[`CompiledRules::change_with_intermediates`] (the [`session`] module), a
faithful port of Lexurgy's `SoundChangeSession`. `rules.rule_names()` returns
the trace/rule-name list.

### As an HTTP server

`lauturgie-server` exposes an API wire-compatible with [Lexurgy's `scv1`
endpoints][lexurgy-api] (everything but `inflectv1`), so existing Lexurgy
clients can point straight at it. It's behind the `server` feature so the
library and CLI stay dependency-light:

```sh
cargo run --features server --bin lauturgie-server   # listens on :8080
```

Endpoints: `GET /`, `GET /version`, `POST /scv1` (apply — outputs,
`intermediateWords`, `traces`, per-word `errors`, plus `startAt`/`stopBefore`
and `allowPolling` background runs), `GET /scv1/poll/{id}`, and `POST
/scv1/validate` (compile-only, returns the rule names). Config via env:
`PORT`, `API_KEY`, and `REQUEST_TIMEOUT`/`TOTAL_TIMEOUT` (seconds; defaults
`0.2`/`0.5`). The JSON request/response shapes match Lexurgy's exactly.

```sh
curl -s localhost:8080/scv1 -H 'content-type: application/json' \
  -d '{"changes":"rule:\no => a","inputWords":["foo","oboe"]}'
# {"ruleNames":["rule"],"outputWords":["faa","abae"]}
```

[Lexurgy docs]: https://lexurgy-app.vercel.app/sc/docs
[lexurgy-api]: https://github.com/def-gthill/lexurgy/tree/master/api
[`CompiledRules`]: https://docs.rs/lauturgie/latest/lauturgie/compiler/struct.CompiledRules.html
[`CompiledRules::change_with_intermediates`]: https://docs.rs/lauturgie/latest/lauturgie/compiler/struct.CompiledRules.html#method.change_with_intermediates
[`session`]: https://docs.rs/lauturgie/latest/lauturgie/session/index.html

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
