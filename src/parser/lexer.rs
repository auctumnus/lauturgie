// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) Autumn <auctumnus@pm.me>

//! Hand-rolled lexer for the Lsc language.
//!
//! Two passes:
//!
//! 1. [`tokenize`] reproduces the ANTLR lexer from `Lsc.g4`, including maximal
//!    munch with rule-order tie-breaking and whitespace absorption into
//!    operator tokens.
//! 2. [`classify`] walks the token stream line-by-line and injects marker
//!    tokens ([`Tok::MarkStmt`] etc.) that resolve, ahead of time, the
//!    decisions ANTLR's ALL(*) prediction makes with unbounded lookahead.
//!    This is what makes the grammar LR(1).

pub mod classify;
pub mod token;
pub mod tokenize;

pub use token::Tok;

pub type Spanned = (usize, Tok, usize);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LexError {
    pub pos: usize,
    pub message: String,
}

impl std::fmt::Display for LexError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} at offset {}", self.message, self.pos)
    }
}

impl std::error::Error for LexError {}

/// Tokenize and classify in one go.
pub fn lex(s: &str) -> Result<Vec<Spanned>, LexError> {
    Ok(classify::classify(tokenize::tokenize(s)?))
}
