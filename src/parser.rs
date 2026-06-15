// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) Autumn <auctumnus@pm.me>

pub mod ast;
pub mod lexer;

mod util;
pub(crate) mod validate;

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

impl Error {
    /// The byte offset a parse/lex error points at, if any. `Invalid`
    /// (post-parse structural) errors have no single location.
    pub fn offset(&self) -> Option<usize> {
        use lalrpop_util::ParseError as P;
        match self {
            Error::Lex(e) => Some(e.pos),
            Error::Parse(e) => Some(match e.as_ref() {
                P::InvalidToken { location } => *location,
                P::UnrecognizedEof { location, .. } => *location,
                P::UnrecognizedToken { token: (start, ..), .. } => *start,
                P::ExtraToken { token: (start, ..) } => *start,
                P::User { error } => error.pos,
            }),
            Error::Invalid(_) => None,
        }
    }
}

/// Convert a byte offset into a (line, column) pair, both 1-based, for error
/// reporting (lexurgy's `LscNotParsable.line`/`.column`).
pub fn line_col(source: &str, offset: usize) -> (usize, usize) {
    let offset = offset.min(source.len());
    let mut line = 1;
    let mut col = 1;
    for (i, ch) in source.char_indices() {
        if i >= offset {
            break;
        }
        if ch == '\n' {
            line += 1;
            col = 1;
        } else {
            col += 1;
        }
    }
    (line, col)
}

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
