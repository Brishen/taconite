// SPDX-FileCopyrightText: Copyright (C) 2026 Brishen Hawkins
// SPDX-License-Identifier: Apache-2.0

//! Regular expressions as Python's `re` matches them, std-only: a
//! backtracking matcher over `char`s with `re`'s syntax and Unicode
//! semantics, for model front ends that filter or split text with
//! user-supplied Python patterns (GLiNER2's `RegexValidator`).
//!
//! Supported: literals and escapes (`\n \t \xhh \uhhhh \Uhhhhhhhh \0`,
//! escaped punctuation), `.`, classes (`[a-z]`, `[^...]`, `\d \w \s \D \W
//! \S` inside and out), anchors (`^ $ \A \Z \b \B`), groups (capturing,
//! `(?:...)`, `(?P<name>...)`), backreferences (`\1`, `(?P=name)`),
//! lookahead and fixed-width lookbehind, alternation, greedy / lazy /
//! possessive quantifiers (`* + ? {m} {m,} {,n} {m,n}`, a `{` that is no
//! quantifier being a literal), comments `(?#...)`, and the flags `i m s x
//! a` as leading `(?imsxa)` or scoped `(?i:...)` / `(?-i:...)`.
//!
//! Semantics follow CPython 3.13's `sre`: `\d` is `str.isdecimal`, `\w` is
//! `str.isalnum` or `_`, `\s` is `str.isspace` (ASCII-only under `a`); `$`
//! matches at the end and before a final `\n`; `.` takes anything but `\n`
//! unless `s`. Case-insensitive matching compares simple lowercase
//! mappings plus `sre`'s extra equivalences (`i`/`ı`, `s`/`ſ`, `µ`/`μ`,
//! `σ`/`ς`, the Greek symbol variants, ...), as `re._casefix` does.
//! Not supported (rejected): conditional groups `(?(1)...)`, `\N{...}`,
//! atomic groups `(?>...)`.

use crate::unicode;

/// Pattern flags (`re.IGNORECASE`, `re.MULTILINE`, `re.DOTALL`,
/// `re.VERBOSE`, `re.ASCII`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Flags {
    pub ignore_case: bool,
    pub multiline: bool,
    pub dotall: bool,
    pub verbose: bool,
    pub ascii: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Cat {
    Digit,
    Word,
    Space,
}

#[derive(Debug, Clone, PartialEq)]
enum Item {
    Range(char, char),
    Cat(Cat, bool),
}

#[derive(Debug, Clone, PartialEq)]
enum Node {
    /// a literal, compared case-insensitively when the flag is on
    Char(char, bool),
    /// `.`: dotall
    Any(bool),
    /// a class: items, negated, case-insensitive, ASCII categories
    Class(Vec<Item>, bool, bool, bool),
    Cat(Cat, bool, bool),
    /// `^` / `$`: multiline
    Bol(bool),
    Eol(bool),
    StartText,
    EndText,
    /// `\b` (false) / `\B` (true): ASCII
    Boundary(bool, bool),
    Group(Box<Node>, Option<usize>),
    Concat(Vec<Node>),
    Alt(Vec<Node>),
    Repeat {
        node: Box<Node>,
        min: usize,
        max: usize,
        greedy: bool,
        possessive: bool,
    },
    /// group, case-insensitive
    Backref(usize, bool),
    Look {
        node: Box<Node>,
        ahead: bool,
        negate: bool,
        width: usize,
    },
}

/// `re`'s extra case equivalences (`re._casefix._EXTRA_CASES`), keyed by
/// a lowercase character.
const EXTRA: &[(u32, &[u32])] = &[
    (0x69, &[0x131]),
    (0x73, &[0x17f]),
    (0xb5, &[0x3bc]),
    (0x131, &[0x69]),
    (0x17f, &[0x73]),
    (0x345, &[0x3b9, 0x1fbe]),
    (0x390, &[0x1fd3]),
    (0x3b0, &[0x1fe3]),
    (0x3b2, &[0x3d0]),
    (0x3b5, &[0x3f5]),
    (0x3b8, &[0x3d1]),
    (0x3b9, &[0x345, 0x1fbe]),
    (0x3ba, &[0x3f0]),
    (0x3bc, &[0xb5]),
    (0x3c0, &[0x3d6]),
    (0x3c1, &[0x3f1]),
    (0x3c2, &[0x3c3]),
    (0x3c3, &[0x3c2]),
    (0x3c6, &[0x3d5]),
    (0x3d0, &[0x3b2]),
    (0x3d1, &[0x3b8]),
    (0x3d5, &[0x3c6]),
    (0x3d6, &[0x3c0]),
    (0x3f0, &[0x3ba]),
    (0x3f1, &[0x3c1]),
    (0x3f5, &[0x3b5]),
    (0x432, &[0x1c80]),
    (0x434, &[0x1c81]),
    (0x43e, &[0x1c82]),
    (0x441, &[0x1c83]),
    (0x442, &[0x1c84, 0x1c85]),
    (0x44a, &[0x1c86]),
    (0x463, &[0x1c87]),
    (0x1c80, &[0x432]),
    (0x1c81, &[0x434]),
    (0x1c82, &[0x43e]),
    (0x1c83, &[0x441]),
    (0x1c84, &[0x442, 0x1c85]),
    (0x1c85, &[0x442, 0x1c84]),
    (0x1c86, &[0x44a]),
    (0x1c87, &[0x463]),
    (0x1c88, &[0xa64b]),
    (0x1e61, &[0x1e9b]),
    (0x1e9b, &[0x1e61]),
    (0x1fbe, &[0x345, 0x3b9]),
    (0x1fd3, &[0x390]),
    (0x1fe3, &[0x3b0]),
    (0xa64b, &[0x1c88]),
    (0xfb05, &[0xfb06]),
    (0xfb06, &[0xfb05]),
];

fn extra(l: char) -> &'static [u32] {
    EXTRA.binary_search_by_key(&(l as u32), |e| e.0).map_or(&[], |i| EXTRA[i].1)
}

fn lower(c: char) -> char {
    unicode::lower_simple(c)
}

/// Whether `ch` matches literal `lit` case-insensitively
/// (`LITERAL_UNI_IGNORE`, or the extra cases' `IN_UNI_IGNORE`).
fn ci_eq(ch: char, lit: char) -> bool {
    let (l, ll) = (lower(ch), lower(lit));
    l == ll || extra(ll).contains(&(l as u32))
}

fn upper(c: char) -> char {
    let mut u = c.to_uppercase();
    match (u.next(), u.next()) {
        (Some(x), None) => x,
        _ => c,
    }
}

fn cat(c: Cat, ch: char, ascii: bool) -> bool {
    match (c, ascii) {
        (Cat::Digit, false) => unicode::is_decimal(ch),
        (Cat::Digit, true) => ch.is_ascii_digit(),
        (Cat::Word, false) => ch == '_' || unicode::is_alnum(ch),
        (Cat::Word, true) => ch == '_' || ch.is_ascii_alphanumeric(),
        (Cat::Space, false) => unicode::is_python_space(ch),
        (Cat::Space, true) => matches!(ch, ' ' | '\t' | '\n' | '\r' | '\x0b' | '\x0c'),
    }
}

fn class_has(items: &[Item], ch: char, ci: bool, ascii: bool) -> bool {
    let plain = |c: char| {
        items.iter().any(|it| match *it {
            Item::Range(a, b) => a <= c && c <= b,
            Item::Cat(k, neg) => cat(k, c, ascii) != neg,
        })
    };
    if !ci {
        return plain(ch);
    }
    // a member m matches when lower(m) is lower(ch) or one of its extra
    // cases: try every character that could be such an m
    let l = lower(ch);
    let mut cands = vec![ch, l, upper(l)];
    for &e in extra(l) {
        if let Some(e) = char::from_u32(e) {
            cands.push(e);
            cands.push(upper(e));
        }
    }
    items.iter().any(|it| match *it {
        Item::Range(a, b) => {
            cands.iter().any(|&m| a <= m && m <= b && (lower(m) == l || extra(lower(m)).contains(&(l as u32))))
        }
        Item::Cat(k, neg) => cat(k, l, ascii) != neg,
    })
}

struct Parser<'a> {
    p: &'a [char],
    i: usize,
    flags: Flags,
    groups: usize,
    names: Vec<(String, usize)>,
}

type R<T> = Result<T, String>;

impl Parser<'_> {
    fn peek(&self) -> Option<char> {
        self.p.get(self.i).copied()
    }

    fn eat(&mut self, c: char) -> bool {
        if self.peek() == Some(c) {
            self.i += 1;
            true
        } else {
            false
        }
    }

    fn skip_verbose(&mut self) {
        if !self.flags.verbose {
            return;
        }
        while let Some(c) = self.peek() {
            if unicode::is_python_space(c) {
                self.i += 1;
            } else if c == '#' {
                while let Some(c) = self.peek() {
                    self.i += 1;
                    if c == '\n' {
                        break;
                    }
                }
            } else {
                break;
            }
        }
    }

    fn alt(&mut self) -> R<Node> {
        let mut alts = vec![self.concat()?];
        while self.eat('|') {
            alts.push(self.concat()?);
        }
        Ok(if alts.len() == 1 { alts.pop().unwrap() } else { Node::Alt(alts) })
    }

    fn concat(&mut self) -> R<Node> {
        let mut items = Vec::new();
        loop {
            self.skip_verbose();
            match self.peek() {
                None | Some('|') | Some(')') => break,
                _ => {}
            }
            let Some(atom) = self.atom()? else { continue };
            let atom = self.quantified(atom)?;
            items.push(atom);
        }
        Ok(if items.len() == 1 { items.pop().unwrap() } else { Node::Concat(items) })
    }

    /// `{m}`, `{m,}`, `{,n}`, `{m,n}` at the cursor, if one is there.
    fn braces(&mut self) -> Option<(usize, usize)> {
        let save = self.i;
        if !self.eat('{') {
            return None;
        }
        let num = |s: &mut Self| {
            let st = s.i;
            while s.peek().is_some_and(|c| c.is_ascii_digit()) {
                s.i += 1;
            }
            (s.i > st).then(|| s.p[st..s.i].iter().collect::<String>().parse::<usize>().ok()).flatten()
        };
        let lo = num(self);
        let r = if self.eat(',') {
            let hi = num(self);
            Some((lo.unwrap_or(0), hi.unwrap_or(usize::MAX)))
        } else {
            lo.map(|l| (l, l))
        };
        match r {
            Some(r) if self.eat('}') => Some(r),
            _ => {
                self.i = save;
                None
            }
        }
    }

    fn quantified(&mut self, atom: Node) -> R<Node> {
        let mut atom = atom;
        loop {
            self.skip_verbose();
            let (min, max) = match self.peek() {
                Some('*') => {
                    self.i += 1;
                    (0, usize::MAX)
                }
                Some('+') => {
                    self.i += 1;
                    (1, usize::MAX)
                }
                Some('?') => {
                    self.i += 1;
                    (0, 1)
                }
                Some('{') => match self.braces() {
                    Some(r) => r,
                    None => return Ok(atom),
                },
                _ => return Ok(atom),
            };
            if min > max {
                return Err("min repeat greater than max repeat".into());
            }
            if matches!(atom, Node::Repeat { .. }) {
                return Err("multiple repeat".into());
            }
            if matches!(atom, Node::Bol(_) | Node::Eol(_) | Node::StartText | Node::EndText | Node::Boundary(..)) {
                return Err("nothing to repeat".into());
            }
            let (greedy, possessive) = if self.eat('?') {
                (false, false)
            } else if self.eat('+') {
                (true, true)
            } else {
                (true, false)
            };
            atom = Node::Repeat { node: Box::new(atom), min, max, greedy, possessive };
            if !matches!(self.peek(), Some('*' | '+' | '?' | '{')) {
                return Ok(atom);
            }
        }
    }

    fn hex(&mut self, n: usize) -> R<char> {
        let s: String = self.p.get(self.i..self.i + n).ok_or("incomplete escape")?.iter().collect();
        let v = u32::from_str_radix(&s, 16).map_err(|_| format!("incomplete escape \\{s}"))?;
        self.i += n;
        char::from_u32(v).ok_or_else(|| "bad character code".to_string())
    }

    /// An escape after `\`: a character, or a category / anchor node.
    fn escape(&mut self, in_class: bool) -> R<Result<char, Node>> {
        let c = self.peek().ok_or("bad escape (end of pattern)")?;
        self.i += 1;
        let (ci, a) = (self.flags.ignore_case, self.flags.ascii);
        Ok(match c {
            'n' => Ok('\n'),
            't' => Ok('\t'),
            'r' => Ok('\r'),
            'f' => Ok('\x0c'),
            'v' => Ok('\x0b'),
            'a' => Ok('\x07'),
            'x' => Ok(self.hex(2)?),
            'u' => Ok(self.hex(4)?),
            'U' => Ok(self.hex(8)?),
            '0' => {
                let mut v = 0u32;
                for _ in 0..2 {
                    match self.peek() {
                        Some(d @ '0'..='7') => {
                            v = v * 8 + d.to_digit(8).unwrap();
                            self.i += 1;
                        }
                        _ => break,
                    }
                }
                Ok(char::from_u32(v).unwrap())
            }
            'd' => Err(Node::Cat(Cat::Digit, false, a)),
            'D' => Err(Node::Cat(Cat::Digit, true, a)),
            'w' => Err(Node::Cat(Cat::Word, false, a)),
            'W' => Err(Node::Cat(Cat::Word, true, a)),
            's' => Err(Node::Cat(Cat::Space, false, a)),
            'S' => Err(Node::Cat(Cat::Space, true, a)),
            '1'..='7' if in_class => {
                // an octal escape inside a class
                let mut v = c.to_digit(8).unwrap();
                for _ in 0..2 {
                    match self.peek() {
                        Some(d @ '0'..='7') => {
                            v = v * 8 + d.to_digit(8).unwrap();
                            self.i += 1;
                        }
                        _ => break,
                    }
                }
                Ok(char::from_u32(v).ok_or("bad octal escape")?)
            }
            'b' if in_class => Ok('\x08'),
            'b' => Err(Node::Boundary(false, a)),
            'B' if !in_class => Err(Node::Boundary(true, a)),
            'A' if !in_class => Err(Node::StartText),
            'Z' if !in_class => Err(Node::EndText),
            '1'..='9' if !in_class => {
                let mut n = c.to_digit(10).unwrap() as usize;
                if let Some(d) = self.peek().and_then(|d| d.to_digit(10)) {
                    n = n * 10 + d as usize;
                    self.i += 1;
                }
                if n > self.groups {
                    return Err(format!("invalid group reference {n}"));
                }
                Err(Node::Backref(n, ci))
            }
            c if c.is_ascii_alphanumeric() => return Err(format!("bad escape \\{c}")),
            c => Ok(c),
        })
    }

    fn class(&mut self) -> R<Node> {
        let neg = self.eat('^');
        let mut items = Vec::new();
        let mut first = true;
        loop {
            let c = self.peek().ok_or("unterminated character set")?;
            if c == ']' && !first {
                self.i += 1;
                break;
            }
            first = false;
            self.i += 1;
            let lo = if c == '\\' {
                match self.escape(true)? {
                    Ok(ch) => ch,
                    Err(Node::Cat(k, n, _)) => {
                        items.push(Item::Cat(k, n));
                        continue;
                    }
                    Err(_) => return Err("bad escape in class".into()),
                }
            } else {
                c
            };
            if self.peek() == Some('-') && self.p.get(self.i + 1).is_some_and(|&n| n != ']') {
                self.i += 1;
                let h = self.peek().ok_or("unterminated character set")?;
                self.i += 1;
                let hi = if h == '\\' {
                    match self.escape(true)? {
                        Ok(ch) => ch,
                        Err(_) => return Err("bad character range".into()),
                    }
                } else {
                    h
                };
                if hi < lo {
                    return Err(format!("bad character range {lo}-{hi}"));
                }
                items.push(Item::Range(lo, hi));
            } else {
                items.push(Item::Range(lo, lo));
            }
        }
        Ok(Node::Class(items, neg, self.flags.ignore_case, self.flags.ascii))
    }

    fn flag_letters(&mut self) -> Flags {
        let mut f = self.flags;
        while let Some(c) = self.peek() {
            match c {
                'i' => f.ignore_case = true,
                'm' => f.multiline = true,
                's' => f.dotall = true,
                'x' => f.verbose = true,
                'a' => f.ascii = true,
                'u' | 'L' => {}
                _ => break,
            }
            self.i += 1;
        }
        f
    }

    fn group_tail(&mut self, node: Node, idx: Option<usize>) -> R<Node> {
        if !self.eat(')') {
            return Err("missing ), unterminated subpattern".into());
        }
        Ok(Node::Group(Box::new(node), idx))
    }

    /// One atom, or None for something that matches nothing (a comment,
    /// leading flags).
    fn atom(&mut self) -> R<Option<Node>> {
        let c = self.peek().unwrap();
        self.i += 1;
        let f = self.flags;
        Ok(Some(match c {
            '.' => Node::Any(f.dotall),
            '^' => Node::Bol(f.multiline),
            '$' => Node::Eol(f.multiline),
            '[' => self.class()?,
            '\\' => match self.escape(false)? {
                Ok(ch) => Node::Char(ch, f.ignore_case),
                Err(n) => n,
            },
            '*' | '+' | '?' => return Err("nothing to repeat".into()),
            '{' => {
                self.i -= 1;
                if self.braces().is_some() {
                    return Err("nothing to repeat".into());
                }
                self.i += 1;
                Node::Char('{', f.ignore_case)
            }
            '(' => {
                if !self.eat('?') {
                    self.groups += 1;
                    let idx = self.groups;
                    let inner = self.alt()?;
                    return Ok(Some(self.group_tail(inner, Some(idx))?));
                }
                match self.peek() {
                    Some(':') => {
                        self.i += 1;
                        let inner = self.alt()?;
                        self.group_tail(inner, None)?
                    }
                    Some('#') => {
                        while self.peek().is_some_and(|c| c != ')') {
                            self.i += 1;
                        }
                        if !self.eat(')') {
                            return Err("missing ), unterminated comment".into());
                        }
                        return Ok(None);
                    }
                    Some('P') => {
                        self.i += 1;
                        if self.eat('<') {
                            let st = self.i;
                            while self.peek().is_some_and(|c| c != '>') {
                                self.i += 1;
                            }
                            let name: String = self.p[st..self.i].iter().collect();
                            if !self.eat('>') || name.is_empty() {
                                return Err("bad group name".into());
                            }
                            self.groups += 1;
                            let idx = self.groups;
                            self.names.push((name, idx));
                            let inner = self.alt()?;
                            self.group_tail(inner, Some(idx))?
                        } else if self.eat('=') {
                            let st = self.i;
                            while self.peek().is_some_and(|c| c != ')') {
                                self.i += 1;
                            }
                            let name: String = self.p[st..self.i].iter().collect();
                            self.i += 1;
                            let idx = self
                                .names
                                .iter()
                                .find(|(n, _)| *n == name)
                                .map(|&(_, i)| i)
                                .ok_or_else(|| format!("unknown group name {name:?}"))?;
                            Node::Backref(idx, f.ignore_case)
                        } else {
                            return Err("unknown extension ?P".into());
                        }
                    }
                    Some('=') | Some('!') => {
                        let negate = self.peek() == Some('!');
                        self.i += 1;
                        let inner = self.alt()?;
                        if !self.eat(')') {
                            return Err("missing ), unterminated subpattern".into());
                        }
                        Node::Look { node: Box::new(inner), ahead: true, negate, width: 0 }
                    }
                    Some('<') if matches!(self.p.get(self.i + 1), Some('=') | Some('!')) => {
                        let negate = self.p[self.i + 1] == '!';
                        self.i += 2;
                        let inner = self.alt()?;
                        if !self.eat(')') {
                            return Err("missing ), unterminated subpattern".into());
                        }
                        let width = fixed_width(&inner).ok_or("look-behind requires fixed-width pattern")?;
                        Node::Look { node: Box::new(inner), ahead: false, negate, width }
                    }
                    _ => {
                        // flags: global "(?imsx)" or scoped "(?i-s:...)"
                        let on = self.flag_letters();
                        let mut scoped = on;
                        if self.eat('-') {
                            while let Some(c) = self.peek() {
                                match c {
                                    'i' => scoped.ignore_case = false,
                                    'm' => scoped.multiline = false,
                                    's' => scoped.dotall = false,
                                    'x' => scoped.verbose = false,
                                    _ => break,
                                }
                                self.i += 1;
                            }
                        }
                        if self.eat(')') {
                            // global flags (Python requires them first)
                            self.flags = on;
                            return Ok(None);
                        }
                        if !self.eat(':') {
                            return Err("unknown extension".into());
                        }
                        let saved = self.flags;
                        self.flags = scoped;
                        let inner = self.alt()?;
                        self.flags = saved;
                        self.group_tail(inner, None)?
                    }
                }
            }
            ')' => return Err("unbalanced parenthesis".into()),
            c => Node::Char(c, f.ignore_case),
        }))
    }
}

/// The width of a pattern that always matches the same number of chars.
fn fixed_width(n: &Node) -> Option<usize> {
    match n {
        Node::Char(..) | Node::Any(_) | Node::Class(..) | Node::Cat(..) => Some(1),
        Node::Bol(_) | Node::Eol(_) | Node::StartText | Node::EndText | Node::Boundary(..) | Node::Look { .. } => {
            Some(0)
        }
        Node::Group(n, _) => fixed_width(n),
        Node::Concat(v) => v.iter().map(fixed_width).sum(),
        Node::Alt(v) => {
            let w = fixed_width(&v[0])?;
            v.iter().all(|x| fixed_width(x) == Some(w)).then_some(w)
        }
        Node::Repeat { node, min, max, .. } if min == max => fixed_width(node).map(|w| w * min),
        _ => None,
    }
}

/// A compiled pattern.
#[derive(Debug, Clone)]
pub struct Regex {
    root: Node,
    groups: usize,
}

struct M<'a> {
    s: &'a [char],
    caps: Vec<Option<(usize, usize)>>,
}

type K<'k> = &'k mut dyn FnMut(usize, &mut M) -> bool;

fn is_word(s: &[char], i: usize, ascii: bool) -> bool {
    i < s.len() && cat(Cat::Word, s[i], ascii)
}

impl Regex {
    /// Compiles `pattern` (`re.compile(pattern, flags)`).
    pub fn new(pattern: &str, flags: Flags) -> Result<Regex, String> {
        let chars: Vec<char> = pattern.chars().collect();
        let mut p = Parser { p: &chars, i: 0, flags, groups: 0, names: Vec::new() };
        let root = p.alt()?;
        if p.i < chars.len() {
            return Err("unbalanced parenthesis".into());
        }
        Ok(Regex { root, groups: p.groups })
    }

    /// `re.fullmatch`: the whole of `text`.
    pub fn fullmatch(&self, text: &str) -> bool {
        let s: Vec<char> = text.chars().collect();
        let n = s.len();
        let mut m = M { s: &s, caps: vec![None; self.groups + 1] };
        m_node(&self.root, 0, &mut m, &mut |j, _| j == n)
    }

    /// `re.search`: anywhere in `text`.
    pub fn search(&self, text: &str) -> bool {
        let s: Vec<char> = text.chars().collect();
        (0..=s.len()).any(|st| {
            let mut m = M { s: &s, caps: vec![None; self.groups + 1] };
            m_node(&self.root, st, &mut m, &mut |_, _| true)
        })
    }

    /// `re.match`: from the start of `text`.
    pub fn is_match_at_start(&self, text: &str) -> bool {
        let s: Vec<char> = text.chars().collect();
        let mut m = M { s: &s, caps: vec![None; self.groups + 1] };
        m_node(&self.root, 0, &mut m, &mut |_, _| true)
    }
}

fn m_seq(v: &[Node], i: usize, m: &mut M, k: K) -> bool {
    match v.split_first() {
        None => k(i, m),
        Some((first, rest)) => m_node(first, i, m, &mut |j, m| m_seq(rest, j, m, k)),
    }
}

/// One character-width test.
fn one(n: &Node, ch: char) -> bool {
    match *n {
        Node::Char(c, ci) => {
            if ci {
                ci_eq(ch, c)
            } else {
                ch == c
            }
        }
        Node::Any(dotall) => dotall || ch != '\n',
        Node::Class(ref items, neg, ci, ascii) => class_has(items, ch, ci, ascii) != neg,
        Node::Cat(k, neg, ascii) => cat(k, ch, ascii) != neg,
        _ => false,
    }
}

fn single(n: &Node) -> bool {
    matches!(n, Node::Char(..) | Node::Any(_) | Node::Class(..) | Node::Cat(..))
}

/// A quantifier's bounds and greediness.
#[derive(Clone, Copy)]
struct Rep {
    min: usize,
    max: usize,
    greedy: bool,
}

fn m_repeat(node: &Node, rep: Rep, count: usize, i: usize, m: &mut M, k: K) -> bool {
    let Rep { min, max, greedy } = rep;
    if greedy {
        if count < max
            && m_node(node, i, m, &mut |j, m| {
                // an empty iteration past the minimum cannot progress
                if j == i && count >= min {
                    return false;
                }
                m_repeat(node, rep, count + 1, j, m, k)
            })
        {
            return true;
        }
        count >= min && k(i, m)
    } else {
        if count >= min && k(i, m) {
            return true;
        }
        count < max
            && m_node(node, i, m, &mut |j, m| {
                if j == i && count >= min {
                    return false;
                }
                m_repeat(node, rep, count + 1, j, m, k)
            })
    }
}

fn m_node(n: &Node, i: usize, m: &mut M, k: K) -> bool {
    let len = m.s.len();
    match n {
        Node::Char(..) | Node::Any(_) | Node::Class(..) | Node::Cat(..) => i < len && one(n, m.s[i]) && k(i + 1, m),
        Node::Bol(ml) => (i == 0 || (*ml && m.s[i - 1] == '\n')) && k(i, m),
        Node::Eol(ml) => {
            let ok = i == len || (i + 1 == len && m.s[i] == '\n') || (*ml && m.s[i] == '\n');
            ok && k(i, m)
        }
        Node::StartText => i == 0 && k(i, m),
        Node::EndText => i == len && k(i, m),
        Node::Boundary(neg, ascii) => {
            let before = i > 0 && is_word(m.s, i - 1, *ascii);
            let at = before != is_word(m.s, i, *ascii);
            (at != *neg) && k(i, m)
        }
        Node::Concat(v) => m_seq(v, i, m, k),
        Node::Alt(v) => v.iter().any(|a| {
            let saved = m.caps.clone();
            if m_node(a, i, m, k) {
                return true;
            }
            m.caps = saved;
            false
        }),
        Node::Group(inner, idx) => match idx {
            None => m_node(inner, i, m, k),
            Some(g) => {
                let g = *g;
                m_node(inner, i, m, &mut |j, m| {
                    let old = m.caps[g];
                    m.caps[g] = Some((i, j));
                    if k(j, m) {
                        return true;
                    }
                    m.caps[g] = old;
                    false
                })
            }
        },
        Node::Repeat { node, min, max, greedy, possessive } => {
            if *possessive {
                // as many as match, never given back
                let mut pos = i;
                let mut count = 0;
                while count < *max {
                    let mut end = None;
                    if !m_node(node, pos, m, &mut |j, _| {
                        end = Some(j);
                        true
                    }) {
                        break;
                    }
                    let j = end.unwrap();
                    count += 1;
                    if j == pos {
                        break;
                    }
                    pos = j;
                }
                return count >= *min && k(pos, m);
            }
            if single(node) && *greedy {
                // a run of single-character matches, backed off one at a time
                let mut e = i;
                while e < len && e - i < *max && one(node, m.s[e]) {
                    e += 1;
                }
                let mut j = e;
                loop {
                    if j - i < *min {
                        return false;
                    }
                    if k(j, m) {
                        return true;
                    }
                    if j == i {
                        return false;
                    }
                    j -= 1;
                }
            }
            m_repeat(node, Rep { min: *min, max: *max, greedy: *greedy }, 0, i, m, k)
        }
        Node::Backref(g, ci) => {
            let Some((a, b)) = m.caps[*g] else {
                return false;
            };
            let w = b - a;
            if i + w > len {
                return false;
            }
            let ok = (0..w).all(|t| {
                let (x, y) = (m.s[i + t], m.s[a + t]);
                if *ci { lower(x) == lower(y) } else { x == y }
            });
            ok && k(i + w, m)
        }
        Node::Look { node, ahead, negate, width } => {
            let saved = m.caps.clone();
            let found = if *ahead {
                m_node(node, i, m, &mut |_, _| true)
            } else {
                i >= *width && m_node(node, i - width, m, &mut |j, _| j == i)
            };
            if *negate || !found {
                m.caps = saved;
            }
            (found != *negate) && k(i, m)
        }
    }
}
