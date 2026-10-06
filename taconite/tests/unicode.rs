// SPDX-FileCopyrightText: Copyright (C) 2026 Brishen Hawkins
// SPDX-License-Identifier: Apache-2.0

//! `unicode.rs` against Python (`tests/fixtures/unicode_*`, from
//! `scripts/gen_unicode_fixtures.py`): every code point's predicates,
//! category and combining class, and NFC / `str.lower()` of strings.

use std::path::Path;

use taconite::json::{self, Value};
use taconite::unicode;

fn fixture(name: &str) -> String {
    std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name)).unwrap()
}

/// Per code point, the value of `ranges` (`[[first, last, value?], ...]`).
fn expand(ranges: &Value, default: Value) -> Vec<Value> {
    let mut out = vec![default; 0x110000];
    for r in ranges.as_array().unwrap() {
        let r = r.as_array().unwrap();
        let (lo, hi) = (r[0].as_i64().unwrap() as usize, r[1].as_i64().unwrap() as usize);
        let v = r.get(2).cloned().unwrap_or(Value::Bool(true));
        out[lo..=hi].fill(v);
    }
    out
}

#[test]
fn properties() {
    let props = json::parse(&fixture("unicode_props.json")).unwrap();
    assert_eq!(props.get("unicode").unwrap().as_str(), Some(unicode::UNICODE_VERSION));
    type Pred = fn(char) -> bool;
    let preds: [(&str, Pred); 7] = [
        ("isalpha", unicode::is_alpha),
        ("isdecimal", unicode::is_decimal),
        ("isdigit", unicode::is_digit),
        ("isnumeric", unicode::is_numeric),
        ("isalnum", unicode::is_alnum),
        ("isspace", unicode::is_python_space),
        ("isupper", unicode::is_upper),
    ];
    let mut bad = Vec::new();
    for (name, f) in preds {
        let want = expand(props.get(name).unwrap(), Value::Bool(false));
        for c in (0..0x110000).filter_map(char::from_u32) {
            if want[c as usize].as_bool() != Some(f(c)) {
                bad.push(format!("{name}(U+{:04X})", c as u32));
            }
        }
    }
    let cat = expand(props.get("category").unwrap(), Value::Str("Cn".into()));
    let ccc = expand(props.get("combining").unwrap(), Value::Int("0".into()));
    for c in (0..0x110000).filter_map(char::from_u32) {
        if cat[c as usize].as_str() != Some(unicode::category(c)) {
            bad.push(format!("category(U+{:04X}) = {}", c as u32, unicode::category(c)));
        }
        if ccc[c as usize].as_i64() != Some(unicode::combining(c) as i64) {
            bad.push(format!("combining(U+{:04X}) = {}", c as u32, unicode::combining(c)));
        }
        if unicode::is_letter(c) != unicode::is_alpha(c) {
            bad.push(format!("is_letter(U+{:04X})", c as u32));
        }
    }
    println!("9 properties x {} code points", 0x110000 - 0x800);
    assert!(bad.is_empty(), "{} mismatches, first: {:?}", bad.len(), &bad[..bad.len().min(20)]);
}

#[test]
fn nfc_and_lower() {
    let mut bad = Vec::new();
    let text = fixture("unicode_cases.jsonl");
    for line in text.lines() {
        let c = json::parse(line).unwrap();
        let s = c.get("s").unwrap().as_str().unwrap();
        let (nfc, lower) = (unicode::nfc(s), unicode::to_lower_py(s));
        if Some(nfc.as_str()) != c.get("nfc").unwrap().as_str() {
            bad.push(format!("nfc({s:?}) = {nfc:?}"));
        }
        if Some(lower.as_str()) != c.get("lower").unwrap().as_str() {
            bad.push(format!("lower({s:?}) = {lower:?}"));
        }
    }
    println!("{} strings", text.lines().count());
    assert!(bad.is_empty(), "{} mismatches, first: {:?}", bad.len(), &bad[..bad.len().min(20)]);
}

#[test]
fn hf_nfc_differs_only_after_unicode_9() {
    // A mark Unicode 14 added (class 218) is class 0 to `tokenizers`, so it
    // blocks reordering; the Dives Akuru pair (Unicode 13) stays apart.
    assert_eq!(unicode::nfc("a\u{1DFA}\u{334}"), "a\u{334}\u{1DFA}");
    assert_eq!(unicode::nfc_hf("a\u{1DFA}\u{334}"), "a\u{1DFA}\u{334}");
    assert_eq!(unicode::nfc("\u{11935}\u{11930}"), "\u{11938}");
    assert_eq!(unicode::nfc_hf("\u{11935}\u{11930}"), "\u{11935}\u{11930}");
    assert_eq!(unicode::nfc_hf("e\u{301} \u{1100}\u{1161}\u{11A8}"), "\u{E9} \u{AC01}");
}
