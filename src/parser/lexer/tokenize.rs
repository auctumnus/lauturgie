// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) Autumn <auctumnus@pm.me>

use smol_str::{SmolStr, SmolStrBuilder};

use super::{LexError, Spanned, Tok};

/// Whitespace per the ANTLR grammar: any Unicode whitespace except newlines.
fn is_inline_ws(c: char) -> bool {
    c.is_whitespace() && c != '\n' && c != '\r'
}

/// Characters excluded from `STR` runs (the `ANY` fragment's negated set).
fn is_special(c: char) -> bool {
    matches!(
        c,
        ',' | '.'
            | '='
            | '>'
            | '('
            | ')'
            | '*'
            | '['
            | ']'
            | '{'
            | '}'
            | '+'
            | '?'
            | '/'
            | '-'
            | '_'
            | ':'
            | '!'
            | '~'
            | '$'
            | '@'
            | '#'
            | '&'
    )
}

fn is_str_char(c: char) -> bool {
    !c.is_whitespace() && !is_special(c) && c != '\\'
}

fn char_at(s: &str, pos: usize) -> Option<char> {
    s[pos..].chars().next()
}

/// Scan an inline-whitespace run starting at `pos`; returns the end offset.
fn scan_ws(s: &str, mut pos: usize) -> usize {
    while let Some(c) = char_at(s, pos) {
        if !is_inline_ws(c) {
            break;
        }
        pos += c.len_utf8();
    }
    pos
}

/// Consume `\r\n` or `\n` at `pos` plus trailing inline whitespace
/// (the ANTLR `NEWLINE` token absorbs the next line's indentation).
fn scan_newline(s: &str, pos: usize) -> Result<usize, LexError> {
    let after = match char_at(s, pos) {
        Some('\n') => pos + 1,
        Some('\r') => {
            if char_at(s, pos + 1) == Some('\n') {
                pos + 2
            } else {
                return Err(LexError {
                    pos,
                    message: "lone carriage return".into(),
                });
            }
        }
        _ => {
            return Err(LexError {
                pos,
                message: "expected newline".into(),
            })
        }
    };
    Ok(scan_ws(s, after))
}

fn at_newline(s: &str, pos: usize) -> bool {
    matches!(char_at(s, pos), Some('\n') | Some('\r'))
}

/// Skip a `#` comment (to just before the newline / end of input).
fn skip_comment(s: &str, mut pos: usize) -> usize {
    while let Some(c) = char_at(s, pos) {
        if c == '\n' || c == '\r' {
            break;
        }
        pos += c.len_utf8();
    }
    pos
}

/// After `=>`, `/` or `//`: absorb `(WHITESPACE | NEWLINE)?`.
fn absorb_ws_or_newline(s: &str, pos: usize) -> usize {
    let ws_end = scan_ws(s, pos);
    if at_newline(s, ws_end) {
        // NEWLINE alternative is the longer match
        scan_newline(s, ws_end).unwrap_or(ws_end)
    } else {
        ws_end
    }
}

fn keyword(raw: &str) -> Option<Tok> {
    let tok = match raw {
        "Element" | "element" => Tok::KwElement(raw.into()),
        "Class" | "class" => Tok::KwClass(raw.into()),
        "Feature" | "feature" => Tok::KwFeature(raw.into()),
        "Diacritic" | "diacritic" => Tok::KwDiacritic(raw.into()),
        "Symbol" | "symbol" => Tok::KwSymbol(raw.into()),
        "Syllables" | "syllables" => Tok::KwSyllables(raw.into()),
        "Explicit" | "explicit" => Tok::KwExplicit(raw.into()),
        "Clear" | "clear" => Tok::KwClear(raw.into()),
        "Deromanizer" | "deromanizer" => Tok::KwDeromanizer(raw.into()),
        "Romanizer" | "romanizer" => Tok::KwRomanizer(raw.into()),
        "Then" | "then" => Tok::KwThen(raw.into()),
        "Else" | "else" => Tok::KwElse(raw.into()),
        "Literal" | "literal" => Tok::KwLiteral(raw.into()),
        "LTR" | "Ltr" | "ltr" => Tok::KwLtr(raw.into()),
        "RTL" | "Rtl" | "rtl" => Tok::KwRtl(raw.into()),
        "Propagate" | "propagate" => Tok::KwPropagate(raw.into()),
        "Cleanup" | "cleanup" => Tok::KwCleanup(raw.into()),
        "Defer" | "defer" => Tok::KwDefer(raw.into()),
        "Unchanged" | "unchanged" => Tok::KwUnchanged(raw.into()),
        "Off" | "off" => Tok::KwOff(raw.into()),
        _ => return None,
    };
    Some(tok)
}

/// Classify a maximal STR-character run, mirroring ANTLR's rule priority:
/// keywords > NUMBER > NAME > STR (ties broken by rule order; all of these
/// match the same maximal run, so priority order decides).
fn classify_word(raw: &str, unescaped: SmolStr, has_escape: bool) -> Tok {
    if !has_escape {
        if let Some(kw) = keyword(raw) {
            return kw;
        }
        if raw.chars().all(|c| c.is_ascii_digit()) {
            return Tok::Number(raw.into());
        }
        if raw.chars().all(|c| c.is_ascii_alphanumeric()) {
            return Tok::Name(raw.into());
        }
    }
    Tok::Str(unescaped)
}

/// Pass 1: tokenize, replicating the ANTLR lexer. Comments are skipped.
/// The output still contains `Ws` and (merged-per-physical-line) `Newline`
/// tokens; [`classify`] turns the latter into markers.
pub fn tokenize(s: &str) -> Result<Vec<Spanned>, LexError> {
    let mut out: Vec<Spanned> = Vec::new();
    let mut pos = 0;

    macro_rules! push {
        ($tok:expr, $end:expr) => {{
            let end = $end;
            out.push((pos, $tok, end));
            pos = end;
        }};
    }

    while pos < s.len() {
        let c = char_at(s, pos).unwrap();
        match c {
            c if is_inline_ws(c) => {
                let ws_end = scan_ws(s, pos);
                let rest = &s[ws_end..];
                if rest.is_empty() {
                    // trailing whitespace before EOF: allowed by lscFile, drop
                    pos = ws_end;
                } else if at_newline(s, ws_end) {
                    let end = scan_newline(s, ws_end)?;
                    push!(Tok::Newline, end);
                } else if rest.starts_with('#') {
                    pos = skip_comment(s, ws_end);
                } else if rest.starts_with("=>") {
                    push!(Tok::Change, absorb_ws_or_newline(s, ws_end + 2));
                } else if rest.starts_with("//") {
                    push!(Tok::Exclusion, absorb_ws_or_newline(s, ws_end + 2));
                } else if rest.starts_with('/') {
                    push!(Tok::Condition, absorb_ws_or_newline(s, ws_end + 1));
                } else if rest.starts_with("::") {
                    push!(Tok::DoubleColon, scan_ws(s, ws_end + 2));
                } else if rest.starts_with("?:") {
                    push!(Tok::QmarkColon, scan_ws(s, ws_end + 2));
                } else if rest.starts_with(')') {
                    push!(Tok::CParen, ws_end + 1);
                } else if rest.starts_with(']') {
                    push!(Tok::MatrixEnd, ws_end + 1);
                } else if rest.starts_with('}') {
                    push!(Tok::ListEnd, ws_end + 1);
                } else {
                    push!(Tok::Ws, ws_end);
                }
            }
            '\n' | '\r' => {
                push!(Tok::Newline, scan_newline(s, pos)?);
            }
            '#' => {
                pos = skip_comment(s, pos);
            }
            ',' => {
                let ws_end = scan_ws(s, pos + 1);
                if at_newline(s, ws_end) {
                    push!(Tok::ClassSep, scan_newline(s, ws_end)?);
                } else {
                    push!(Tok::ListSep, ws_end);
                }
            }
            '(' => {
                let rest = &s[pos..];
                let paren_kw = [
                    ("(Syllable)", Tok::SyllableFeature),
                    ("(syllable)", Tok::SyllableFeature),
                    ("(Before)", Tok::DiaBefore),
                    ("(before)", Tok::DiaBefore),
                    ("(First)", Tok::DiaFirst),
                    ("(first)", Tok::DiaFirst),
                    ("(Floating)", Tok::DiaFloating),
                    ("(floating)", Tok::DiaFloating),
                ]
                .into_iter()
                .find(|(lit, _)| rest.starts_with(lit));
                match paren_kw {
                    Some((lit, tok)) => push!(tok, pos + lit.len()),
                    None => push!(Tok::OParen, scan_ws(s, pos + 1)),
                }
            }
            '{' => {
                let ws_end = scan_ws(s, pos + 1);
                if at_newline(s, ws_end) {
                    push!(Tok::ClassStart, scan_newline(s, ws_end)?);
                } else {
                    push!(Tok::ListStart, ws_end);
                }
            }
            '[' => push!(Tok::MatrixStart, scan_ws(s, pos + 1)),
            ')' => push!(Tok::CParen, pos + 1),
            ']' => push!(Tok::MatrixEnd, pos + 1),
            '}' => push!(Tok::ListEnd, pos + 1),
            '*' => push!(Tok::Star, pos + 1),
            '+' => push!(Tok::Plus, pos + 1),
            '-' => push!(Tok::Hyphen, pos + 1),
            '~' => push!(Tok::Tilde, pos + 1),
            '!' => push!(Tok::Bang, pos + 1),
            '.' => push!(Tok::Dot, pos + 1),
            '@' => push!(Tok::At, pos + 1),
            '_' => push!(Tok::Anchor, pos + 1),
            '?' => {
                if char_at(s, pos + 1) == Some(':') {
                    push!(Tok::QmarkColon, scan_ws(s, pos + 2));
                } else {
                    push!(Tok::Question, pos + 1);
                }
            }
            ':' => {
                if char_at(s, pos + 1) == Some(':') {
                    push!(Tok::DoubleColon, scan_ws(s, pos + 2));
                } else {
                    push!(Tok::Colon, pos + 1);
                }
            }
            '$' => {
                if char_at(s, pos + 1) == Some('$') {
                    push!(Tok::DollarDollar, pos + 2);
                } else {
                    push!(Tok::Dollar, pos + 1);
                }
            }
            '&' => {
                if char_at(s, pos + 1) == Some('!') {
                    push!(Tok::AmpBang, pos + 2);
                } else {
                    push!(Tok::Amp, pos + 1);
                }
            }
            '>' => push!(Tok::Greater, pos + 1),
            '=' => {
                if char_at(s, pos + 1) == Some('>') {
                    push!(Tok::Change, absorb_ws_or_newline(s, pos + 2));
                } else {
                    return Err(LexError {
                        pos,
                        message: "unexpected '='".into(),
                    });
                }
            }
            '<' => {
                let rest = &s[pos..];
                if rest.starts_with("<syl>") || rest.starts_with("<Syl>") {
                    // ANTLR quirk: a longer STR run would win maximal munch,
                    // e.g. `<syl>a` lexes as one STR token. Check for that.
                    let run_end = str_run_end(s, pos);
                    if run_end <= pos + 5 {
                        push!(Tok::AnySyllable, pos + 5);
                        continue;
                    }
                }
                lex_word(s, &mut out, &mut pos)?;
            }
            c if is_str_char(c) || c == '\\' => {
                lex_word(s, &mut out, &mut pos)?;
            }
            _ => {
                return Err(LexError {
                    pos,
                    message: format!("unexpected character {c:?}"),
                });
            }
        }
    }
    Ok(out)
}

/// End offset of a maximal STR run (escapes included) starting at `pos`.
fn str_run_end(s: &str, mut pos: usize) -> usize {
    loop {
        match char_at(s, pos) {
            Some('\\') => match char_at(s, pos + 1) {
                Some(c) => pos += 1 + c.len_utf8(),
                None => break,
            },
            Some(c) if is_str_char(c) => pos += c.len_utf8(),
            _ => break,
        }
    }
    pos
}

fn lex_word(s: &str, out: &mut Vec<Spanned>, pos: &mut usize) -> Result<(), LexError> {
    let start = *pos;
    let mut has_escape = false;
    let mut p = start;
    loop {
        match char_at(s, p) {
            Some('\\') => {
                let c = char_at(s, p + 1).ok_or(LexError {
                    pos: p,
                    message: "dangling escape at end of input".into(),
                })?;
                has_escape = true;
                p += 1 + c.len_utf8();
            }
            Some(c) if is_str_char(c) => {
                p += c.len_utf8();
            }
            _ => break,
        }
    }
    if p == start {
        return Err(LexError {
            pos: start,
            message: format!("unexpected character {:?}", char_at(s, start).unwrap()),
        });
    }
    let raw = &s[start..p];
    let text = if has_escape {
        let mut builder = SmolStrBuilder::new();
        let mut chars = raw.chars();
        while let Some(c) = chars.next() {
            // The scan above guarantees every '\' has a following char.
            builder.push(if c == '\\' { chars.next().unwrap() } else { c });
        }
        builder.finish()
    } else {
        SmolStr::new(raw)
    };
    out.push((start, classify_word(raw, text, has_escape), p));
    *pos = p;
    Ok(())
}
