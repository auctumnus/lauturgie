// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) Autumn <auctumnus@pm.me>

pub mod ast;
pub mod lexer;

mod util;
mod validate;

use lalrpop_util::lalrpop_mod;

lalrpop_mod!(
    #[allow(clippy::all)]
    #[allow(unused)]
    lsc
);

use lexer::LexError;
use lexer::Tok;

pub type RawParseError = lalrpop_util::ParseError<usize, Tok, LexError>;

#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    /// Tokenization failed.
    Lex(LexError),
    /// The token stream didn't match the grammar.
    Parse(Box<RawParseError>),
    /// Parsed, but structurally invalid (e.g. `_` in an alternative list).
    Invalid(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Lex(e) => write!(f, "lex error: {e}"),
            Error::Parse(e) => write!(f, "parse error: {e}"),
            Error::Invalid(msg) => write!(f, "invalid structure: {msg}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<RawParseError> for Error {
    fn from(e: RawParseError) -> Self {
        Error::Parse(Box::new(e))
    }
}

/// Parse an Lsc sound-change file into a list of statements.
pub fn parse(source: &str) -> Result<Vec<ast::Statement>, Error> {
    let tokens = lexer::lex(source).map_err(Error::Lex)?;
    let statements = lsc::FileParser::new().parse(tokens.into_iter().map(Ok::<_, LexError>))?;
    validate::validate(&statements).map_err(Error::Invalid)?;
    Ok(statements)
}
