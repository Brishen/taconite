// SPDX-FileCopyrightText: Copyright (C) 2026 Brishen Hawkins
// SPDX-License-Identifier: Apache-2.0

//! JSON as Python's `json` module reads and writes it.
//!
//! A model front end ported from Python that serialises its input with
//! `json.dumps(..., ensure_ascii=False)` and tokenizes the result (laya's
//! request state) has token ids that depend on the exact text Python prints: its separators (`", "`, `": "`), its string
//! escapes (only `"`, `\`, and control characters; everything else raw), its
//! float repr (shortest round-trip digits, `1e-05`, `1e+16`, `1.0`), its
//! arbitrary-size ints, and a dict's key order. [`parse`] + [`dumps`]
//! reproduce `json.dumps(json.loads(s), ensure_ascii=False)`, and
//! [`py_round`] is Python's `round(x, n)`; results print as Python
//! would print them (`py_float_repr`).
//!
//! Deviation: a lone UTF-16 surrogate escape (`"\ud800"`) becomes U+FFFD --
//! a Python `str` can hold it, a Rust `String` cannot.

use std::fmt::Write;

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Null,
    Bool(bool),
    /// An integer lexeme as Python's `int` prints it (any size; `-0` is `0`).
    Int(String),
    Float(f64),
    Str(String),
    Array(Vec<Value>),
    /// Insertion-ordered, like a Python dict: a repeated key keeps its
    /// first position and takes the last value.
    Object(Vec<(String, Value)>),
}

impl Value {
    pub fn as_str(&self) -> Option<&str> {
        if let Value::Str(s) = self { Some(s) } else { None }
    }

    pub fn as_bool(&self) -> Option<bool> {
        if let Value::Bool(b) = self { Some(*b) } else { None }
    }

    /// An `Int` or a `Float` as f64.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::Float(f) => Some(*f),
            Value::Int(s) => s.parse().ok(),
            _ => None,
        }
    }

    /// An `Int` that fits an i64.
    pub fn as_i64(&self) -> Option<i64> {
        if let Value::Int(s) = self { s.parse().ok() } else { None }
    }

    pub fn as_array(&self) -> Option<&[Value]> {
        if let Value::Array(a) = self { Some(a) } else { None }
    }

    pub fn as_object(&self) -> Option<&[(String, Value)]> {
        if let Value::Object(o) = self { Some(o) } else { None }
    }

    pub fn is_null(&self) -> bool {
        matches!(self, Value::Null)
    }

    /// An object's value for `key`.
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.as_object()?.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }
}

impl From<&str> for Value {
    fn from(s: &str) -> Self {
        Value::Str(s.to_string())
    }
}

impl From<String> for Value {
    fn from(s: String) -> Self {
        Value::Str(s)
    }
}

impl From<f64> for Value {
    fn from(f: f64) -> Self {
        Value::Float(f)
    }
}

impl From<bool> for Value {
    fn from(b: bool) -> Self {
        Value::Bool(b)
    }
}

impl From<i64> for Value {
    fn from(i: i64) -> Self {
        Value::Int(i.to_string())
    }
}

// ----------------------------------------------------------------------------
// parse
// ----------------------------------------------------------------------------

/// `json.loads(s)`: RFC 8259 plus Python's `NaN`, `Infinity` and
/// `-Infinity`; whitespace is space, tab, CR and LF; control characters
/// inside strings are errors (Python's `strict=True`).
pub fn parse(s: &str) -> Result<Value, String> {
    let mut p = Parser { s: s.as_bytes(), i: 0 };
    p.ws();
    let v = p.value(0)?;
    p.ws();
    if p.i != p.s.len() {
        return Err(p.err("extra data"));
    }
    Ok(v)
}

struct Parser<'a> {
    s: &'a [u8],
    i: usize,
}

impl Parser<'_> {
    fn err(&self, what: &str) -> String {
        format!("JSON: {what} at byte {}", self.i)
    }

    fn ws(&mut self) {
        while let Some(b' ' | b'\t' | b'\n' | b'\r') = self.s.get(self.i) {
            self.i += 1;
        }
    }

    fn eat(&mut self, lit: &str) -> bool {
        if self.s[self.i..].starts_with(lit.as_bytes()) {
            self.i += lit.len();
            true
        } else {
            false
        }
    }

    fn value(&mut self, depth: usize) -> Result<Value, String> {
        if depth > 512 {
            return Err(self.err("nesting too deep"));
        }
        match self.s.get(self.i) {
            None => Err(self.err("expecting value")),
            Some(b'{') => self.object(depth),
            Some(b'[') => self.array(depth),
            Some(b'"') => self.string().map(Value::Str),
            Some(b'-' | b'0'..=b'9') => {
                if self.eat("-Infinity") {
                    Ok(Value::Float(f64::NEG_INFINITY))
                } else {
                    self.number()
                }
            }
            _ if self.eat("null") => Ok(Value::Null),
            _ if self.eat("true") => Ok(Value::Bool(true)),
            _ if self.eat("false") => Ok(Value::Bool(false)),
            _ if self.eat("NaN") => Ok(Value::Float(f64::NAN)),
            _ if self.eat("Infinity") => Ok(Value::Float(f64::INFINITY)),
            _ => Err(self.err("expecting value")),
        }
    }

    fn object(&mut self, depth: usize) -> Result<Value, String> {
        self.i += 1;
        let mut out: Vec<(String, Value)> = Vec::new();
        // a repeated key is found by scanning while the object is small,
        // through a key -> position index once it is not (a tokenizer
        // vocabulary has 256k keys)
        let mut index: Option<std::collections::HashMap<String, usize>> = None;
        self.ws();
        if self.eat("}") {
            return Ok(Value::Object(out));
        }
        loop {
            self.ws();
            if self.s.get(self.i) != Some(&b'"') {
                return Err(self.err("expecting property name enclosed in double quotes"));
            }
            let k = self.string()?;
            self.ws();
            if !self.eat(":") {
                return Err(self.err("expecting ':' delimiter"));
            }
            self.ws();
            let v = self.value(depth + 1)?;
            if index.is_none() && out.len() >= 16 {
                index = Some(out.iter().enumerate().map(|(i, (k, _))| (k.clone(), i)).collect());
            }
            let seen = match &index {
                Some(ix) => ix.get(&k).copied(),
                None => out.iter().position(|(ok, _)| *ok == k),
            };
            match seen {
                Some(i) => out[i].1 = v,
                None => {
                    if let Some(ix) = &mut index {
                        ix.insert(k.clone(), out.len());
                    }
                    out.push((k, v));
                }
            }
            self.ws();
            if self.eat("}") {
                return Ok(Value::Object(out));
            }
            if !self.eat(",") {
                return Err(self.err("expecting ',' delimiter"));
            }
        }
    }

    fn array(&mut self, depth: usize) -> Result<Value, String> {
        self.i += 1;
        let mut out = Vec::new();
        self.ws();
        if self.eat("]") {
            return Ok(Value::Array(out));
        }
        loop {
            self.ws();
            out.push(self.value(depth + 1)?);
            self.ws();
            if self.eat("]") {
                return Ok(Value::Array(out));
            }
            if !self.eat(",") {
                return Err(self.err("expecting ',' delimiter"));
            }
        }
    }

    fn digits(&mut self) -> usize {
        let start = self.i;
        while self.s.get(self.i).is_some_and(u8::is_ascii_digit) {
            self.i += 1;
        }
        self.i - start
    }

    /// Python's `-?(0|[1-9]\d*)(\.\d+)?([eE][-+]?\d+)?`; a fraction or an
    /// exponent that does not complete is left unread (then "extra data").
    fn number(&mut self) -> Result<Value, String> {
        let start = self.i;
        self.eat("-");
        match self.s.get(self.i) {
            Some(b'0') => self.i += 1,
            Some(b'1'..=b'9') => {
                self.digits();
            }
            _ => return Err(self.err("expecting value")),
        }
        let int_end = self.i;
        if self.s.get(self.i) == Some(&b'.') {
            self.i += 1;
            if self.digits() == 0 {
                self.i = int_end;
            }
        }
        let frac_end = self.i;
        if let Some(b'e' | b'E') = self.s.get(self.i) {
            self.i += 1;
            if let Some(b'+' | b'-') = self.s.get(self.i) {
                self.i += 1;
            }
            if self.digits() == 0 {
                self.i = frac_end;
            }
        }
        let lex = std::str::from_utf8(&self.s[start..self.i]).expect("ASCII");
        if self.i == int_end {
            let neg = lex.starts_with('-');
            let digits = lex.trim_start_matches('-');
            return Ok(Value::Int(if neg && digits != "0" { lex.to_string() } else { digits.to_string() }));
        }
        lex.parse().map(Value::Float).map_err(|_| self.err("bad number"))
    }

    fn hex4(&mut self) -> Result<u32, String> {
        let h = self.s.get(self.i..self.i + 4).ok_or_else(|| self.err("invalid \\uXXXX escape"))?;
        let v = std::str::from_utf8(h).ok().filter(|h| h.bytes().all(|b| b.is_ascii_hexdigit()));
        let v = v.and_then(|h| u32::from_str_radix(h, 16).ok()).ok_or_else(|| self.err("invalid \\uXXXX escape"))?;
        self.i += 4;
        Ok(v)
    }

    fn string(&mut self) -> Result<String, String> {
        self.i += 1;
        let mut out = String::new();
        loop {
            let start = self.i;
            while let Some(&b) = self.s.get(self.i) {
                if b == b'"' || b == b'\\' || b < 0x20 {
                    break;
                }
                self.i += 1;
            }
            // The input is a &str, and the run stops at ASCII bytes only.
            out.push_str(std::str::from_utf8(&self.s[start..self.i]).expect("UTF-8"));
            match self.s.get(self.i) {
                None => return Err(self.err("unterminated string")),
                Some(b'"') => {
                    self.i += 1;
                    return Ok(out);
                }
                Some(b'\\') => {
                    self.i += 1;
                    let e = *self.s.get(self.i).ok_or_else(|| self.err("unterminated string"))?;
                    self.i += 1;
                    match e {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let mut cp = self.hex4()?;
                            if (0xD800..0xDC00).contains(&cp) && self.s[self.i..].starts_with(b"\\u") {
                                let save = self.i;
                                self.i += 2;
                                let lo = self.hex4()?;
                                if (0xDC00..0xE000).contains(&lo) {
                                    cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                                } else {
                                    self.i = save;
                                }
                            }
                            out.push(char::from_u32(cp).unwrap_or('\u{FFFD}'));
                        }
                        _ => {
                            self.i -= 1;
                            return Err(self.err("invalid \\escape"));
                        }
                    }
                }
                Some(_) => return Err(self.err("invalid control character")),
            }
        }
    }
}

// ----------------------------------------------------------------------------
// dumps
// ----------------------------------------------------------------------------

/// `json.dumps(v, ensure_ascii=False)`.
pub fn dumps(v: &Value) -> String {
    dumps_with(v, ", ", ": ")
}

/// `json.dumps(v, ensure_ascii=False, separators=(item_sep, key_sep))`.
pub fn dumps_with(v: &Value, item_sep: &str, key_sep: &str) -> String {
    let mut out = String::new();
    write_value(&mut out, v, item_sep, key_sep);
    out
}

fn write_value(out: &mut String, v: &Value, item_sep: &str, key_sep: &str) {
    match v {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Int(s) => out.push_str(s),
        Value::Float(f) => out.push_str(&json_float(*f)),
        Value::Str(s) => write_str(out, s),
        Value::Array(a) => {
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push_str(item_sep);
                }
                write_value(out, x, item_sep, key_sep);
            }
            out.push(']');
        }
        Value::Object(o) => {
            out.push('{');
            for (i, (k, x)) in o.iter().enumerate() {
                if i > 0 {
                    out.push_str(item_sep);
                }
                write_str(out, k);
                out.push_str(key_sep);
                write_value(out, x, item_sep, key_sep);
            }
            out.push('}');
        }
    }
}

/// A JSON string literal as `json.dumps(..., ensure_ascii=False)` writes it.
pub fn write_str(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

fn json_float(f: f64) -> String {
    if f.is_nan() {
        "NaN".into()
    } else if f.is_infinite() {
        if f > 0.0 { "Infinity" } else { "-Infinity" }.into()
    } else {
        py_float_repr(f)
    }
}

/// Python's `repr(f)`: the shortest digits that round-trip, positional
/// when the decimal exponent is in [-4, 16), else `d.ddde+XX` (at least two
/// exponent digits); a whole positional value keeps `.0`.
pub fn py_float_repr(f: f64) -> String {
    if f.is_nan() {
        return "nan".into();
    }
    if f.is_infinite() {
        return if f > 0.0 { "inf" } else { "-inf" }.into();
    }
    // `{:e}` is the shortest round-trip digits, as `d.ddde<exp>`.
    let e = format!("{:e}", f.abs());
    let (mant, exp) = e.split_once('e').expect("{:e} has an exponent");
    let exp: i32 = exp.parse().expect("{:e} exponent");
    let digits: String = mant.chars().filter(|c| *c != '.').collect();
    let sign = if f.is_sign_negative() { "-" } else { "" };
    if !(-4..16).contains(&exp) {
        let (d0, rest) = digits.split_at(1);
        let frac = if rest.is_empty() { String::new() } else { format!(".{rest}") };
        let es = if exp < 0 { '-' } else { '+' };
        return format!("{sign}{d0}{frac}e{es}{:02}", exp.abs());
    }
    let n = digits.len() as i32;
    let body = if exp < 0 {
        format!("0.{}{digits}", "0".repeat((-exp - 1) as usize))
    } else if exp + 1 >= n {
        format!("{digits}{}.0", "0".repeat((exp + 1 - n) as usize))
    } else {
        let (a, b) = digits.split_at((exp + 1) as usize);
        format!("{a}.{b}")
    };
    format!("{sign}{body}")
}

/// Python's `round(x, ndigits)`: `x` rounded to `ndigits` decimal places,
/// correctly -- the exact binary value's decimal expansion, ties to even
/// -- and read back as the nearest f64. Non-finite values pass through.
pub fn py_round(x: f64, ndigits: i32) -> f64 {
    // CPython's bounds: past them every f64 is unchanged / rounds to zero.
    if !x.is_finite() || x == 0.0 || ndigits > 323 {
        return x;
    }
    if ndigits < -308 {
        return 0.0f64.copysign(x);
    }
    let s = if ndigits >= 0 {
        round_decimal(x, ndigits as usize)
    } else {
        let sign = if x < 0.0 { "-" } else { "" };
        format!("{sign}{}e{}", round_decimal_scaled(x, (-ndigits) as usize), -ndigits)
    };
    let r: f64 = s.parse().expect("decimal string");
    // Python keeps the sign of a result that rounds to zero.
    if r == 0.0 { 0.0f64.copysign(x) } else { r }
}

/// The exact decimal expansion of finite `x`: (negative, integer digits,
/// fraction digits), no trailing fraction zeros.
fn exact_decimal(x: f64) -> (bool, Vec<u8>, Vec<u8>) {
    let bits = x.to_bits();
    let neg = bits >> 63 != 0;
    let exp = ((bits >> 52) & 0x7FF) as i32;
    let frac = bits & ((1u64 << 52) - 1);
    let (mant, e2) = if exp == 0 { (frac, -1074) } else { (frac | 1 << 52, exp - 1075) };
    // value = mant * 2^e2, as a little-endian base-1e9 big integer divided
    // by 2^-e2 (when e2 < 0).
    let mut big: Vec<u32> = vec![(mant % 1_000_000_000) as u32, (mant / 1_000_000_000) as u32];
    let mul = |big: &mut Vec<u32>, m: u32| {
        let mut carry = 0u64;
        for d in big.iter_mut() {
            let v = *d as u64 * m as u64 + carry;
            *d = (v % 1_000_000_000) as u32;
            carry = v / 1_000_000_000;
        }
        while carry > 0 {
            big.push((carry % 1_000_000_000) as u32);
            carry /= 1_000_000_000;
        }
    };
    let to_digits = |big: &[u32]| -> Vec<u8> {
        let mut s = String::new();
        for (i, d) in big.iter().rev().enumerate() {
            if i == 0 {
                let _ = write!(s, "{d}");
            } else {
                let _ = write!(s, "{d:09}");
            }
        }
        let s = s.trim_start_matches('0');
        if s.is_empty() { vec![b'0'] } else { s.bytes().collect() }
    };
    if e2 >= 0 {
        for _ in 0..e2 {
            mul(&mut big, 2);
        }
        return (neg, to_digits(&big), Vec::new());
    }
    // mant / 2^k = mant * 5^k / 10^k
    let k = (-e2) as usize;
    for _ in 0..k {
        mul(&mut big, 5);
    }
    let all = to_digits(&big);
    let (int, frac): (Vec<u8>, Vec<u8>) = if all.len() > k {
        (all[..all.len() - k].to_vec(), all[all.len() - k..].to_vec())
    } else {
        (vec![b'0'], std::iter::repeat_n(b'0', k - all.len()).chain(all).collect())
    };
    let mut frac = frac;
    while frac.last() == Some(&b'0') {
        frac.pop();
    }
    (neg, int, frac)
}

/// Increment a decimal digit string by one unit in its last place.
fn increment(d: &mut Vec<u8>) {
    for x in d.iter_mut().rev() {
        if *x == b'9' {
            *x = b'0';
        } else {
            *x += 1;
            return;
        }
    }
    d.insert(0, b'1');
}

/// `x` rounded half-to-even at `n` fraction digits, as a decimal string.
fn round_decimal(x: f64, n: usize) -> String {
    let (neg, int, frac) = exact_decimal(x);
    let mut digits: Vec<u8> = int.iter().chain(frac.iter().chain(std::iter::repeat(&b'0')).take(n)).copied().collect();
    let rest = frac.get(n..).unwrap_or(&[]);
    if round_up(rest, digits.last().copied()) {
        increment(&mut digits);
    }
    let split = digits.len() - n;
    let s = format!(
        "{}{}.{}",
        if neg { "-" } else { "" },
        std::str::from_utf8(&digits[..split]).expect("digits"),
        std::str::from_utf8(&digits[split..]).expect("digits")
    );
    s.trim_end_matches('.').to_string()
}

/// `|x| / 10^n` rounded half-to-even to an integer, as a decimal string.
fn round_decimal_scaled(x: f64, n: usize) -> String {
    let (_, int, frac) = exact_decimal(x);
    let int: Vec<u8> = std::iter::repeat_n(b'0', (n + 1).saturating_sub(int.len())).chain(int).collect();
    let mut digits = int[..int.len() - n].to_vec();
    let rest: Vec<u8> = int[int.len() - n..].iter().chain(&frac).copied().collect();
    if round_up(&rest, digits.last().copied()) {
        increment(&mut digits);
    }
    String::from_utf8(digits).expect("digits")
}

/// Whether the discarded digits `rest` round the kept ones (last digit
/// `last`) up: above one half, or exactly half and `last` is odd.
fn round_up(rest: &[u8], last: Option<u8>) -> bool {
    match rest.first() {
        None => false,
        Some(&d) if d > b'5' => true,
        Some(&d) if d < b'5' => false,
        Some(_) => rest[1..].iter().any(|&d| d != b'0') || last.is_some_and(|l| (l - b'0') % 2 == 1),
    }
}
