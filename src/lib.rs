// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) Autumn <auctumnus@pm.me>.
// Derived from Lexurgy (https://github.com/def-gthill/lexurgy), © its authors,
// also licensed under GPLv3. See the LICENSE file for the full text.

//! lauturgie: a Rust implementation of the Lexurgy sound change language.
//!
//! Parse a `.lsc` ruleset with [`parse`], lower it with [`compiler::compile`],
//! and apply the resulting `CompiledRules` to words. See the crate README for
//! a worked example.

pub mod compiler;
pub mod fst;
pub mod parser;
pub mod session;
pub mod vm;
pub mod word;

pub use parser::lexer;
pub use parser::{ast, parse, Error};
