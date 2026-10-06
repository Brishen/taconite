// SPDX-FileCopyrightText: Copyright (C) 2026 Brishen Hawkins
// SPDX-License-Identifier: Apache-2.0

//! The Unicode character database as Python sees it, std-only.
//!
//! Model front ends ported from Python (laya's language detection, GLiNER2's
//! word splitter) run Python `str` methods and `re` classes over Python's
//! `unicodedata` (Unicode 15.0 in Python 3.12), and their tokenizers are HF
//! `tokenizers`, which carry tables of their own. Rust's `char` predicates
//! follow whatever Unicode version the compiler ships and differ from both
//! in places (`str.isalpha` is not `char::is_alphabetic`, `str.isnumeric`
//! takes CJK numerals, Rust has no NFC at all), so the tables here are
//! generated from Python by `scripts/gen_unicode.py` (`unicode_data.rs`)
//! and every predicate is named after the Python method it answers.
//!
//! Two HF variants sit alongside: [`onig_letter`] / [`onig_number`] are the
//! `\p{L}` / `\p{N}` of Oniguruma (Unicode 16.0) that the GPT-2
//! pre-tokenizer regex runs on, and [`nfc_hf`] is the NFC of
//! `unicode-normalization-alignments` (Unicode 9.0 data: combining marks
//! assigned since are class 0, and U+11938 never composes).

#[rustfmt::skip]
#[path = "unicode_data.rs"]
mod data;

pub use data::UNICODE_VERSION;

fn props(c: char) -> (u8, u16) {
    let cp = c as u32;
    let t = data::PROPS;
    match t.binary_search_by(|&(lo, hi, _, _)| {
        if hi < cp {
            std::cmp::Ordering::Less
        } else if lo > cp {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Equal
        }
    }) {
        Ok(i) => (t[i].2, t[i].3),
        Err(_) => (0, 0),
    }
}

fn flag(c: char, f: u16) -> bool {
    props(c).1 & f != 0
}

/// `unicodedata.category(c)`: `"Lu"`, `"Nd"`, ... (`"Cn"` if unassigned).
pub fn category(c: char) -> &'static str {
    data::CATEGORIES[props(c).0 as usize]
}

/// `c.isalpha()`: categories Lu Ll Lt Lm Lo.
pub fn is_alpha(c: char) -> bool {
    flag(c, data::ALPHA)
}

/// Any `L*` category (the same set as [`is_alpha`]).
pub fn is_letter(c: char) -> bool {
    category(c).starts_with('L')
}

/// Any `N*` category: Nd Nl No.
pub fn is_number(c: char) -> bool {
    category(c).starts_with('N')
}

/// `c.isdecimal()`: what Python's `re` takes for `\d`.
pub fn is_decimal(c: char) -> bool {
    flag(c, data::DECIMAL)
}

/// `c.isdigit()`.
pub fn is_digit(c: char) -> bool {
    flag(c, data::DIGIT)
}

/// `c.isnumeric()`: every character with a numeric value, CJK numerals
/// such as 一 and 萬 (category Lo) included.
pub fn is_numeric(c: char) -> bool {
    flag(c, data::NUMERIC)
}

/// `c.isalnum()`: Python's `re` takes this or `_` for `\w`.
pub fn is_alnum(c: char) -> bool {
    flag(c, data::ALPHA | data::DECIMAL | data::DIGIT | data::NUMERIC)
}

/// `c.isupper()` for a single character (the Uppercase property).
pub fn is_upper(c: char) -> bool {
    flag(c, data::UPPER)
}

/// `c.isspace()`: Python's whitespace (bidirectional classes WS/B/S and
/// category Zs, so U+001C..U+001F count and U+200B does not).
pub fn is_python_space(c: char) -> bool {
    flag(c, data::SPACE)
}

/// The White_Space property: `\s` of Rust's `regex` and of Oniguruma.
pub fn is_white_space(c: char) -> bool {
    matches!(
        c,
        '\t'..='\r'
            | ' '
            | '\u{85}'
            | '\u{A0}'
            | '\u{1680}'
            | '\u{2000}'..='\u{200A}'
            | '\u{2028}'
            | '\u{2029}'
            | '\u{202F}'
            | '\u{205F}'
            | '\u{3000}'
    )
}

/// Oniguruma's `\p{L}` (Unicode 16.0), as HF's GPT-2 regex sees it.
pub fn onig_letter(c: char) -> bool {
    flag(c, data::ONIG_LETTER)
}

/// Oniguruma's `\p{N}` (Unicode 16.0).
pub fn onig_number(c: char) -> bool {
    flag(c, data::ONIG_NUMBER)
}

/// `unicodedata.combining(c)`: the canonical combining class.
pub fn combining(c: char) -> u8 {
    ccc(c as u32, false)
}

fn ccc(cp: u32, hf: bool) -> u8 {
    if cp < 0x300 || hf && data::CCC_AFTER_9.iter().any(|&(lo, hi)| (lo..=hi).contains(&cp)) {
        return 0;
    }
    let t = data::CCC;
    match t.binary_search_by(|&(lo, hi, _)| {
        if hi < cp {
            std::cmp::Ordering::Less
        } else if lo > cp {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Equal
        }
    }) {
        Ok(i) => t[i].2,
        Err(_) => 0,
    }
}

const S_BASE: u32 = 0xAC00;
const L_BASE: u32 = 0x1100;
const V_BASE: u32 = 0x1161;
const T_BASE: u32 = 0x11A7;
const L_COUNT: u32 = 19;
const V_COUNT: u32 = 21;
const T_COUNT: u32 = 28;
const N_COUNT: u32 = V_COUNT * T_COUNT;
const S_COUNT: u32 = L_COUNT * N_COUNT;

fn ch(cp: u32) -> char {
    char::from_u32(cp).expect("table code points are scalar values")
}

/// Full canonical decomposition of `cp` onto `out` (with its class),
/// keeping each run of non-starters in canonical order.
fn decompose(cp: u32, hf: bool, out: &mut Vec<(char, u8)>) {
    if (S_BASE..S_BASE + S_COUNT).contains(&cp) {
        let s = cp - S_BASE;
        push_ordered(out, ch(L_BASE + s / N_COUNT), 0);
        push_ordered(out, ch(V_BASE + s % N_COUNT / T_COUNT), 0);
        if s % T_COUNT != 0 {
            push_ordered(out, ch(T_BASE + s % T_COUNT), 0);
        }
        return;
    }
    let found = if cp >= 0xC0 && !(hf && cp == data::AFTER_9_COMPOSITE) {
        data::DECOMP.binary_search_by_key(&cp, |d| d.0).ok()
    } else {
        None
    };
    if let Some(i) = found {
        let (_, a, b) = data::DECOMP[i];
        decompose(a, hf, out);
        if b != 0 {
            decompose(b, hf, out);
        }
        return;
    }
    push_ordered(out, ch(cp), ccc(cp, hf));
}

fn push_ordered(out: &mut Vec<(char, u8)>, c: char, class: u8) {
    let mut i = out.len();
    out.push((c, class));
    if class == 0 {
        return;
    }
    while i > 0 && out[i - 1].1 > class {
        out.swap(i - 1, i);
        i -= 1;
    }
}

fn compose(a: char, b: char, hf: bool) -> Option<char> {
    let (a, b) = (a as u32, b as u32);
    if (L_BASE..L_BASE + L_COUNT).contains(&a) && (V_BASE..V_BASE + V_COUNT).contains(&b) {
        return Some(ch(S_BASE + ((a - L_BASE) * V_COUNT + (b - V_BASE)) * T_COUNT));
    }
    if (S_BASE..S_BASE + S_COUNT).contains(&a)
        && (a - S_BASE) % T_COUNT == 0
        && (T_BASE + 1..T_BASE + T_COUNT).contains(&b)
    {
        return Some(ch(a + b - T_BASE));
    }
    let i = data::COMPOSE.binary_search_by_key(&(a, b), |c| (c.0, c.1)).ok()?;
    let c = data::COMPOSE[i].2;
    if hf && c == data::AFTER_9_COMPOSITE { None } else { Some(ch(c)) }
}

fn nfc_with(s: &str, hf: bool) -> String {
    // Below U+0300 every character is a starter that NFC leaves alone.
    if s.chars().all(|c| (c as u32) < 0x300) {
        return s.to_string();
    }
    let mut d: Vec<(char, u8)> = Vec::with_capacity(s.len());
    for c in s.chars() {
        decompose(c as u32, hf, &mut d);
    }
    let mut out: Vec<char> = Vec::with_capacity(d.len());
    // The last starter, and the class of the last character since it (None
    // while the two are adjacent): a mark is blocked from the starter by
    // anything in between of the same or a higher class, or by a starter.
    let mut starter: Option<usize> = None;
    let mut last: Option<u8> = None;
    for (c, class) in d {
        let composed =
            starter.filter(|_| last.is_none_or(|l| l < class)).and_then(|si| Some((si, compose(out[si], c, hf)?)));
        if let Some((si, k)) = composed {
            out[si] = k;
            continue;
        }
        if class == 0 {
            starter = Some(out.len());
            last = None;
        } else {
            last = Some(class);
        }
        out.push(c);
    }
    out.into_iter().collect()
}

/// `unicodedata.normalize("NFC", s)`: canonical decomposition, canonical
/// ordering, canonical composition (composition exclusions and Hangul
/// included).
pub fn nfc(s: &str) -> String {
    nfc_with(s, false)
}

/// The NFC HF `tokenizers` applies: [`nfc`] with Unicode 9.0's view of the
/// characters assigned since (see the module docs).
pub fn nfc_hf(s: &str) -> String {
    nfc_with(s, true)
}

/// The simple (one-character) lowercase mapping Python's `re` folds case
/// with (`_sre.unicode_tolower`): U+0130 İ is `i`.
pub fn lower_simple(c: char) -> char {
    if c == '\u{130}' { 'i' } else { lower_char(c) }
}

fn lower_char(c: char) -> char {
    let cp = c as u32;
    match data::LOWER.binary_search_by_key(&cp, |l| l.0) {
        Ok(i) => ch(data::LOWER[i].1),
        Err(_) => c,
    }
}

/// Python's `str.lower()`: the full lowercase mapping (U+0130 İ becomes
/// `i` + U+0307) with Python's Final_Sigma rule for Σ -- ς when a cased
/// letter precedes it and none follows (case-ignorable characters
/// skipped), σ otherwise. `str::to_lowercase` implements the same rules on
/// the compiler's Unicode version; this follows Python's tables.
pub fn to_lower_py(s: &str) -> String {
    if s.is_ascii() {
        return s.to_ascii_lowercase();
    }
    let chars: Vec<char> = s.chars().collect();
    let mut out = String::with_capacity(s.len());
    for (i, &c) in chars.iter().enumerate() {
        match c {
            '\u{3A3}' => out.push(if final_sigma(&chars, i) { '\u{3C2}' } else { '\u{3C3}' }),
            '\u{130}' => out.push_str("i\u{307}"),
            _ => out.push(lower_char(c)),
        }
    }
    out
}

fn final_sigma(chars: &[char], i: usize) -> bool {
    let ignorable = |c: char| flag(c, data::CASE_IGNORABLE);
    let cased = |c: char| flag(c, data::CASED);
    let before = chars[..i].iter().rev().find(|&&c| !ignorable(c));
    if !before.is_some_and(|&c| cased(c)) {
        return false;
    }
    chars[i + 1..].iter().find(|&&c| !ignorable(c)).is_none_or(|&c| !cased(c))
}
