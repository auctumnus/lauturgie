// SPDX-License-Identifier: GPL-3.0-or-later
//
// lauturgie: apply Lexurgy sound changes (a Rust reimplementation).
// Copyright (C) Autumn <auctumnus@pm.me>
//
// This program is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by the Free
// Software Foundation, either version 3 of the License, or (at your option)
// any later version.
//
// This program is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
// more details.
//
// You should have received a copy of the GNU General Public License along with
// this program. If not, see <https://www.gnu.org/licenses/>.
//
// Derived from Lexurgy (https://github.com/def-gthill/lexurgy), © its authors,
// also licensed under GPLv3.

//! Command-line front end: apply the sound changes in a `.lsc` file to a list
//! of words, mirroring the scope of lexurgy's own `sc` command (romanized
//! output, one cold pass). Words run in parallel across all cores by default.

use std::io::{Read, Write};
use std::process::ExitCode;
use std::time::Instant;

const USAGE: &str = "\
lauturgie — apply Lexurgy sound changes (a Rust reimplementation)

USAGE:
    lauturgie <CHANGES.lsc> <WORDS> [OPTIONS]

ARGS:
    <CHANGES>    a .lsc sound-change file
    <WORDS>      a word list, one word or phrase per line ('-' for stdin)

OPTIONS:
    -o, --output <FILE>    write the evolved words here (default: stdout)
        --vm               force the reference VM tier (disable the FST tier)
    -1, --single-thread    apply lines sequentially (default: all cores)
    -q, --quiet            suppress the summary line on stderr
    -h, --help             print this help and exit
    -V, --version          print the version and exit

Exit status: 0 = all words applied, 1 = some words errored (marked ERROR in
the output, details on stderr), 2 = the changes file couldn't be read/compiled.
";

fn main() -> ExitCode {
    match run() {
        Ok(code) => code,
        // Fatal: bad args, unreadable file, or a parse/compile failure.
        Err(message) => {
            eprintln!("lauturgie: {message}");
            ExitCode::from(2)
        }
    }
}

fn run() -> Result<ExitCode, String> {
    let mut changes: Option<String> = None;
    let mut words: Option<String> = None;
    let mut output: Option<String> = None;
    let mut force_vm = false;
    let mut single_thread = false;
    let mut quiet = false;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                print!("{USAGE}");
                return Ok(ExitCode::SUCCESS);
            }
            "-V" | "--version" => {
                println!("lauturgie {}", env!("CARGO_PKG_VERSION"));
                return Ok(ExitCode::SUCCESS);
            }
            "--vm" => force_vm = true,
            "-1" | "--single-thread" => single_thread = true,
            "-q" | "--quiet" => quiet = true,
            "-o" | "--output" => {
                output = Some(args.next().ok_or("-o/--output needs a FILE argument")?);
            }
            // '-' is the stdin sentinel, not an option.
            s if s.starts_with('-') && s != "-" => {
                return Err(format!("unknown option '{s}' (try --help)"));
            }
            _ if changes.is_none() => changes = Some(arg),
            _ if words.is_none() => words = Some(arg),
            _ => return Err(format!("unexpected extra argument '{arg}' (try --help)")),
        }
    }

    let changes = changes.ok_or("missing CHANGES.lsc argument (try --help)")?;
    let words_path = words.ok_or("missing WORDS argument (try --help)")?;

    // Compile the ruleset (parse + validate + lower). Errors here are fatal.
    let src = std::fs::read_to_string(&changes).map_err(|e| format!("reading {changes}: {e}"))?;
    let statements = lauturgie::parse(&src).map_err(|e| format!("parsing {changes}:\n{e}"))?;
    let mut compiled = lauturgie::compiler::compile(&statements)
        .map_err(|e| format!("compiling {changes}:\n{e}"))?;
    compiled.force_vm = force_vm;

    // Read the word list ('-' = stdin). Keep one word/phrase per line; drop only
    // the single trailing newline so output lines stay aligned with input.
    let raw = if words_path == "-" {
        let mut s = String::new();
        std::io::stdin()
            .read_to_string(&mut s)
            .map_err(|e| format!("reading stdin: {e}"))?;
        s
    } else {
        std::fs::read_to_string(&words_path).map_err(|e| format!("reading {words_path}: {e}"))?
    };
    let mut lines: Vec<&str> = raw.split('\n').collect();
    if lines.last() == Some(&"") {
        lines.pop();
    }

    // Apply. Parallel across words by default; --single-thread / --vm walk the
    // sequential path (handy when debugging a ruleset).
    let start = Instant::now();
    let results = if single_thread {
        lines.iter().map(|&w| compiled.apply(w)).collect::<Vec<_>>()
    } else {
        compiled.apply_all(&lines)
    };
    let elapsed = start.elapsed().as_secs_f64();

    // A per-word failure isn't fatal: mark the line ERROR, report it on stderr,
    // and finish the rest (exit 1 at the end).
    let mut out = Vec::with_capacity(lines.len());
    let mut errors = 0usize;
    for (word, result) in lines.iter().zip(&results) {
        match result {
            Ok(evolved) => out.push(evolved.clone()),
            Err(e) => {
                errors += 1;
                eprintln!("{word} => ERROR: {e}");
                out.push("ERROR".to_string());
            }
        }
    }
    let body = out.join("\n") + "\n";

    match &output {
        Some(path) => std::fs::write(path, &body).map_err(|e| format!("writing {path}: {e}"))?,
        None => std::io::stdout()
            .write_all(body.as_bytes())
            .map_err(|e| format!("writing stdout: {e}"))?,
    }

    if !quiet {
        eprintln!(
            "applied the changes to {} word{} in {elapsed:.3}s ({errors} error{})",
            lines.len(),
            if lines.len() == 1 { "" } else { "s" },
            if errors == 1 { "" } else { "s" },
        );
    }

    Ok(if errors > 0 {
        ExitCode::from(1)
    } else {
        ExitCode::SUCCESS
    })
}
