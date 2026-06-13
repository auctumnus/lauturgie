#!/bin/sh
# Code-coverage harness for lauturgie.
#
# Merges two sources into one report:
#   1. the whole test suite (differential, grammar, corpus, fst, parallel),
#   2. a deterministic fuzzer sweep (the generator + both tiers), which is
#      where most expression-variant coverage actually comes from.
#
# The fuzzer runs as a single `--worker` process (coverage can't follow the
# sandboxed child processes the default driver spawns), over a fixed seed
# range so the numbers are reproducible.
#
# Usage:  ./coverage.sh [SEEDS]   (default 4000)   then open the printed path.
# Needs:  cargo install cargo-llvm-cov
set -e
cd "$(dirname "$0")"
SEEDS="${1:-4000}"

cargo llvm-cov clean --workspace
# 1. test suite
cargo llvm-cov --no-report --all-targets
# 2. fuzzer sweep (deterministic seeds 0..SEEDS, 30 words each)
cargo llvm-cov run --no-report --example fuzz -- \
    --worker --start 0 --count "$SEEDS" --words 30

cargo llvm-cov report --summary-only
cargo llvm-cov report --html
echo "HTML report: target/llvm-cov/html/index.html"
