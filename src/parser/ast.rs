// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) Autumn <auctumnus@pm.me>

//! AST for Lsc sound-change files.
//!
//! Shapes follow the parse rules in lexurgy's `Lsc.g4` so that a tree-walking
//! consumer can be written against either parser. One deliberate difference:
//! braced lists are a *cover* construct ([`ListItem`]) that can hold anchored
//! environments, because `{a _, b _}` (an environment list) and `{a, b} c _`
//! (an alternative list inside an environment) can't be distinguished without
//! unbounded lookahead. The parser reinterprets the cover at the point of use;
//! [`crate::validate`] rejects anchored items that survive in element position.

use smol_str::SmolStr;

#[derive(Debug, Clone, PartialEq)]
pub enum Statement {
    Feature(FeatureDecl),
    Diacritic(DiacriticDecl),
    Symbol {
        /// Several plain symbols (`Symbol a, b, c`) or one symbol with a matrix.
        names: Vec<Text>,
        matrix: Option<Matrix>,
    },
    Class {
        name: SmolStr,
        elements: Vec<ClassElement>,
    },
    Element {
        name: SmolStr,
        element: RuleElement,
    },
    Syllables(SyllableSpec),
    Deromanizer {
        literal: bool,
        block: Block,
    },
    Romanizer {
        literal: bool,
        block: Block,
    },
    InterRomanizer {
        name: SmolStr,
        literal: bool,
        block: Block,
    },
    Rule(ChangeRule),
    /// A bare `a => b` at top level (allowed by the grammar; lexurgy rejects
    /// it later with a "rule needs a name" style error).
    Expression(StandardExpression),
}

#[derive(Debug, Clone, PartialEq)]
pub enum FeatureDecl {
    /// `Feature +long, (syllable) +heavy`
    Plus(Vec<PlusFeature>),
    /// `Feature Height(*low, mid, high)`
    Full {
        syllable: bool,
        name: SmolStr,
        null_alias: Option<SmolStr>,
        values: Vec<SmolStr>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct PlusFeature {
    pub syllable: bool,
    pub plus: bool,
    pub name: SmolStr,
}

#[derive(Debug, Clone, PartialEq)]
pub struct DiacriticDecl {
    pub text: Text,
    pub modifiers: Vec<DiacriticModifier>,
    pub matrix: Matrix,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiacriticModifier {
    Before,
    First,
    Floating,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ClassElement {
    ElementRef(SmolStr),
    Text(Text),
}

#[derive(Debug, Clone, PartialEq)]
pub enum SyllableSpec {
    Explicit,
    Clear,
    Patterns(Vec<SyllableExpression>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct SyllableExpression {
    pub pattern: SyllablePattern,
    /// The matrix after `=>`, assigning syllable-level features.
    pub assign: Option<Matrix>,
    pub environment: Option<CompoundEnvironment>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SyllablePattern {
    /// `onset ?: onset :: nucleus :: coda` (2 or 3 parts).
    Structured {
        reluctant_onset: Option<Element>,
        parts: Vec<Element>,
    },
    /// A plain pattern; it may carry its own environment before the `=>`
    /// (`@cons? @vowel / _ $ => [heavy]`).
    Plain(RuleElement),
}

#[derive(Debug, Clone, PartialEq)]
pub struct ChangeRule {
    pub name: SmolStr,
    pub modifiers: Vec<RuleModifier>,
    pub block: Block,
}

#[derive(Debug, Clone, PartialEq)]
pub enum RuleModifier {
    /// `rule [vowel]:` or `rule @vowel:`: only matching segments are visible.
    Filter(Element),
    Ltr,
    Rtl,
    Propagate,
    Defer,
    Cleanup,
    /// The grammar admits any name here; lexurgy validates it later.
    Name(SmolStr),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Block {
    pub first: BlockElement,
    pub rest: Vec<(BlockType, BlockElement)>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BlockType {
    pub kind: BlockKind,
    pub modifiers: Vec<RuleModifier>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlockKind {
    /// `Then:`: all matching.
    Then,
    /// `Else:`: first matching.
    Else,
}

#[derive(Debug, Clone, PartialEq)]
pub enum BlockElement {
    Expressions(Vec<Expression>),
    Nested(Box<Block>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expression {
    Unchanged,
    Off,
    /// `:rule-name`: splice in a deferred rule.
    BlockRef(SmolStr),
    Standard(StandardExpression),
}

#[derive(Debug, Clone, PartialEq)]
pub struct StandardExpression {
    pub from: RuleElement,
    pub to: Element,
    pub environment: Option<CompoundEnvironment>,
}

/// An element plus the environment that may be attached directly to it
/// (`ruleElement` in the ANTLR grammar).
#[derive(Debug, Clone, PartialEq)]
pub struct RuleElement {
    pub element: Element,
    pub environment: Option<CompoundEnvironment>,
}

impl From<Element> for RuleElement {
    fn from(element: Element) -> Self {
        RuleElement {
            element,
            environment: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Element {
    /// Whitespace-separated elements (always 2+).
    Sequence(Vec<Element>),
    /// `( ... )`; the inner element may carry its own environment.
    Group(Box<RuleElement>),
    /// `{ ... , ... }`: alternatives, or (in environment position) an
    /// environment list. See [`ListItem`].
    List(Vec<ListItem>),
    /// `a&b`, `a&!b`, `a>b` chains (no whitespace).
    Interfix {
        first: Box<Element>,
        rest: Vec<(InterfixKind, Element)>,
    },
    /// `!a`
    Negated(Box<Element>),
    /// `(a)$1`
    Capture {
        element: Box<Element>,
        capture: CaptureRef,
    },
    /// `a*`, `a+`, `a?`, `a*3`, `a*(1-3)`
    Repeat {
        element: Box<Element>,
        kind: RepeaterKind,
    },
    Text(Text),
    /// `[+hi -lo $Place !*foo]`
    Matrix(Matrix),
    /// `@vowel`
    ElementRef(SmolStr),
    /// `$1`, `~$1`, `$.1`
    CaptureRef(CaptureRef),
    /// `*`
    Empty,
    /// `.`
    SyllableBoundary,
    /// `$`
    WordBoundary,
    /// `$$`
    BetweenWords,
    /// `<syl>`
    AnySyllable,
}

/// Literal text. `exact` is the postfix `!` (match without diacritic search).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Text {
    pub text: SmolStr,
    pub exact: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterfixKind {
    /// `&`
    Intersection,
    /// `&!`
    IntersectionNot,
    /// `>`
    Transforming,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CaptureRef {
    pub inexact: bool,
    pub syllable: bool,
    pub number: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RepeaterKind {
    ZeroOrMore,
    AtLeastOne,
    Optional,
    Count(u32),
    Range { min: Option<u32>, max: Option<u32> },
}

pub type Matrix = Vec<MatrixValue>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatrixValue {
    /// Prefix `!` (only valid in "fancy" matrices inside rule elements).
    pub negated: bool,
    pub value: MatrixValueKind,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MatrixValueKind {
    /// `+name`
    Plus(SmolStr),
    /// `-name`
    Minus(SmolStr),
    /// `name`
    Simple(SmolStr),
    /// `*name`: feature absent.
    Absent(SmolStr),
    /// `$Name`: feature variable.
    Variable(SmolStr),
}

/// One item of a braced list (cover for alternatives and environments).
#[derive(Debug, Clone, PartialEq)]
pub enum ListItem {
    Element(RuleElement),
    /// An item with an anchor (or empty); only legal where the list is
    /// reinterpreted as an environment list.
    Env(Environment),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Environment {
    pub before: Option<Element>,
    pub anchored: bool,
    pub after: Option<Element>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct CompoundEnvironment {
    pub condition: Option<Vec<Environment>>,
    pub exclusion: Option<Vec<Environment>>,
}
