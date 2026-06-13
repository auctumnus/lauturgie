# lauturgie task runner. `just` (https://github.com/casey/just) lists recipes
# with `just` or `just --list`. The vendor test rulesets live under
# `vendor/lexurgy/cli/test/` and most examples default to them.

# the four vendor rulesets the benches/A-B tools default to
VENDOR := "vendor/lexurgy/cli/test"

# show the recipe list
default:
    @just --list

# ── build ──────────────────────────────────────────────────────────────────

# debug build of everything (lib, CLI, examples, tests)
build:
    cargo build --all-targets

# release build (frame pointers + symbols kept, see Cargo.toml)
release:
    cargo build --release --all-targets

# apply a ruleset to a word list; flags pass through (--vm, -1, -o); '-' = stdin
run lsc words *FLAGS:
    cargo run --release --bin lauturgie -- {{lsc}} {{words}} {{FLAGS}}

# ── tests ──────────────────────────────────────────────────────────────────

# the whole suite (differential, grammar, corpus, fst, parallel, parse_examples)
test:
    cargo test

# differential suite vs lexurgy's Kotlin test cases — run after any vm.rs change
differential:
    cargo test --test differential

# the corpus A/B and fst-tier integration tests
fst-tests:
    cargo test --test fst

# the rayon-vs-sequential equivalence test
parallel:
    cargo test --test parallel

# ── fuzzing ────────────────────────────────────────────────────────────────
# examples/fuzz.rs: seeded grammar-aware differential fuzzer. Findings land in
# fuzz_findings/ as rerunnable .lsc/.wli + report.

# fst-vs-vm, ~3k cases/s, runs forever (Ctrl-C to stop)
fuzz:
    cargo run --release --example fuzz

# deterministic seed range from 0 (reproducible sweep)
fuzz-from start="0":
    cargo run --release --example fuzz -- --start {{start}}

# reproduce/inspect one seed, verbose (prints the generated lsc + words)
fuzz-seed seed:
    cargo run --release --example fuzz -- --seed {{seed}}

# vs the real Kotlin CLI oracle (~2 cases/s); needs the CLI in vendor/lexurgy or $LEXURGY_CLI
fuzz-kotlin:
    cargo run --release --example fuzz -- --kotlin

# ── coverage ───────────────────────────────────────────────────────────────

# merged test + fuzzer-sweep coverage HTML; needs `cargo install cargo-llvm-cov`
coverage seeds="4000":
    ./coverage.sh {{seeds}}

# ── benchmarks & analysis (all release) ────────────────────────────────────

# tier benchmark: real rulesets, FST tier vs forced-VM
bench:
    cargo run --release --example bench_tiers

# time the way the Kotlin CLI times itself, over a dir of lsc/wli pairs
bench-kotlin dir="/tmp/lexurgy-bench":
    cargo run --release --example bench_vs_kotlin {{dir}}

# bench one arbitrary pair (fst, vm, parallel): `just bench-one a.lsc w.wli`
bench-one lsc words:
    cargo run --release --example bench_one {{lsc}} {{words}}

# per-rule tier report (which rules hit FST, why the rest don't, timings); no args = vendor rulesets
tiers *ARGS:
    cargo run --release --example tier_report {{ARGS}}

# A/B every vendor-corpus word through fst vs forced-VM, reporting mismatches
ab:
    cargo run --release --example ab_corpus

# in-process flamegraph SVG (no perf/root); e.g. `just profile --vm --out vm.svg`
profile *FLAGS:
    cargo run --release --example profile -- {{FLAGS}}

# does every vendor ruleset still parse + compile?
smoke:
    cargo run --release --example smoke_compile

# find which rule first makes a word error: `just bisect a.lsc someword`
bisect lsc word:
    cargo run --release --example bisect_error {{lsc}} {{word}}

# ── housekeeping ───────────────────────────────────────────────────────────

fmt:
    cargo fmt

clippy:
    cargo clippy --all-targets

# the pre-push gate: format check, lint, full suite
check: fmt clippy test

# remove fuzzer/oracle scratch logs left in the repo root
clean-logs:
    rm -f fuzz_run.log fuzz_run.err.log oracle_*.log
