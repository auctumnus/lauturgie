// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) Autumn <auctumnus@pm.me>

//! Tokens for the Lsc language.
//!
//! These mirror the ANTLR lexer in lexurgy's `Lsc.g4`, including its quirk of
//! folding whitespace into adjacent operator tokens (e.g. `CHANGE: WHITESPACE?
//! '=>' (WHITESPACE | NEWLINE)?`). That folding is what makes whitespace-
//! sensitive constructs like sequences (`a b`) and captures (`(a)$1`) LR(1)-
//! parsable: whitespace only survives as a `Ws` token where it is structurally
//! meaningful.
//!
//! In addition to the ANTLR-equivalent tokens, the classifier pass
//! ([`crate::parser::lexer::classify`]) injects zero-width *marker* tokens at the start
//! of each logical line, encoding decisions that ANTLR makes with unbounded
//! adaptive lookahead (rule header vs. expression, block continuation vs. new
//! statement, ...). The lalrpop grammar keys off these markers.

use smol_str::SmolStr;

/// A token, carrying source text where the grammar can use it as a name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Tok {
    // synthetic markers injected by the classifier
    /// Start of a top-level statement (declaration or rule header).
    MarkStmt,
    /// Start of a top-level standalone expression (`a => b` outside any rule).
    MarkTopExpr,
    /// Start of an expression line inside a rule block.
    MarkExpr,
    /// Start of a `Then:`/`Else:` block-type line inside a rule block.
    MarkBlockType,
    /// Start of a syllable-pattern line inside a `Syllables:` declaration.
    MarkSyll,
    /// Start of an `Explicit`/`Clear` line inside a `Syllables:` declaration.
    MarkSyllMode,
    /// Newline run immediately before a closing `)` of a multi-line block.
    MarkEnd,

    /// A plain newline that survives inside a class declaration's braces.
    Newline,
    /// Whitespace that separates sequence elements / declaration parts.
    Ws,

    // punctuation (whitespace-absorbing forms noted)
    ListSep,      // ',' ws?
    ClassSep,     // ',' newline (absorbs the line break)
    Change,       // ws? '=>' (ws|newline)?
    Condition,    // ws? '/' (ws|newline)?
    Exclusion,    // ws? '//' (ws|newline)?
    Anchor,       // '_'
    OParen,       // '(' ws?
    CParen,       // ws? ')'
    Star,         // '*'
    MatrixStart,  // '[' ws?
    MatrixEnd,    // ws? ']'
    ListStart,    // '{' ws?
    ClassStart,   // '{' newline (absorbs the line break)
    ListEnd,      // ws? '}'
    Plus,         // '+'
    Question,     // '?'
    Hyphen,       // '-'
    Colon,        // ':'
    DoubleColon,  // ws? '::' ws?
    QmarkColon,   // ws? '?:' ws?
    Tilde,        // '~'
    Bang,         // '!'
    Dot,          // '.'
    Dollar,       // '$'
    DollarDollar, // '$$'
    At,           // '@'
    Amp,          // '&'
    AmpBang,      // '&!'
    Greater,      // '>'

    // keywords; these carry a SmolStr because in the first pass we can't tell
    // if these are being used as names or as keywords yet
    KwElement(SmolStr),
    KwClass(SmolStr),
    KwFeature(SmolStr),
    KwDiacritic(SmolStr),
    KwSymbol(SmolStr),
    KwSyllables(SmolStr),
    KwExplicit(SmolStr),
    KwClear(SmolStr),
    KwDeromanizer(SmolStr),
    KwRomanizer(SmolStr),
    KwThen(SmolStr),
    KwElse(SmolStr),
    KwLiteral(SmolStr),
    KwLtr(SmolStr),
    KwRtl(SmolStr),
    KwPropagate(SmolStr),
    KwCleanup(SmolStr),
    KwDefer(SmolStr),
    KwUnchanged(SmolStr),
    KwOff(SmolStr),

    // parenthesized keywords (single tokens, as in ANTLR)
    SyllableFeature, // '(Syllable)' / '(syllable)'
    DiaBefore,       // '(Before)' / '(before)'
    DiaFirst,        // '(First)' / '(first)'
    DiaFloating,     // '(Floating)' / '(floating)'
    AnySyllable,     // '<Syl>' / '<syl>'

    Number(SmolStr),
    /// ASCII alphanumeric run.
    Name(SmolStr),
    /// Any other run of non-special characters (escapes already resolved).
    Str(SmolStr),
}

impl Tok {
    /// Source text of a keyword token, for relabeling keywords to plain names.
    pub fn keyword_text(&self) -> Option<&str> {
        use Tok::*;
        match self {
            KwElement(s) | KwClass(s) | KwFeature(s) | KwDiacritic(s) | KwSymbol(s)
            | KwSyllables(s) | KwExplicit(s) | KwClear(s) | KwDeromanizer(s) | KwRomanizer(s)
            | KwThen(s) | KwElse(s) | KwLiteral(s) | KwLtr(s) | KwRtl(s) | KwPropagate(s)
            | KwCleanup(s) | KwDefer(s) | KwUnchanged(s) | KwOff(s) => Some(s),
            _ => None,
        }
    }
}

impl std::fmt::Display for Tok {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use Tok::*;
        match self {
            MarkStmt => write!(f, "<start of statement>"),
            MarkTopExpr => write!(f, "<start of expression>"),
            MarkExpr => write!(f, "<start of expression line>"),
            MarkBlockType => write!(f, "<start of Then/Else>"),
            MarkSyll => write!(f, "<start of syllable pattern>"),
            MarkSyllMode => write!(f, "<start of Explicit/Clear>"),
            MarkEnd => write!(f, "<end of block>"),
            Newline => write!(f, "<newline>"),
            Ws => write!(f, "<whitespace>"),
            ListSep | ClassSep => write!(f, "','"),
            Change => write!(f, "'=>'"),
            Condition => write!(f, "'/'"),
            Exclusion => write!(f, "'//'"),
            Anchor => write!(f, "'_'"),
            OParen => write!(f, "'('"),
            CParen => write!(f, "')'"),
            Star => write!(f, "'*'"),
            MatrixStart => write!(f, "'['"),
            MatrixEnd => write!(f, "']'"),
            ListStart | ClassStart => write!(f, "'{{'"),
            ListEnd => write!(f, "'}}'"),
            Plus => write!(f, "'+'"),
            Question => write!(f, "'?'"),
            Hyphen => write!(f, "'-'"),
            Colon => write!(f, "':'"),
            DoubleColon => write!(f, "'::'"),
            QmarkColon => write!(f, "'?:'"),
            Tilde => write!(f, "'~'"),
            Bang => write!(f, "'!'"),
            Dot => write!(f, "'.'"),
            Dollar => write!(f, "'$'"),
            DollarDollar => write!(f, "'$$'"),
            At => write!(f, "'@'"),
            Amp => write!(f, "'&'"),
            AmpBang => write!(f, "'&!'"),
            Greater => write!(f, "'>'"),
            SyllableFeature => write!(f, "'(syllable)'"),
            DiaBefore => write!(f, "'(before)'"),
            DiaFirst => write!(f, "'(first)'"),
            DiaFloating => write!(f, "'(floating)'"),
            AnySyllable => write!(f, "'<syl>'"),
            Number(s) | Name(s) | Str(s) => write!(f, "'{s}'"),
            other => match other.keyword_text() {
                Some(s) => write!(f, "'{s}'"),
                None => write!(f, "<token>"),
            },
        }
    }
}
