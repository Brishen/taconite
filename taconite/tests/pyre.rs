// SPDX-FileCopyrightText: Copyright (C) 2026 Brishen Hawkins
// SPDX-License-Identifier: Apache-2.0

//! `pyre` against Python's `re` (`tests/fixtures/pyre.jsonl`, from
//! `scripts/gen_regex_fixtures.py`): patterns x texts x flags, their
//! `fullmatch` / `search` / `match` results and which patterns fail to
//! compile.

use std::path::Path;

use taconite::json::{self, Value};
use taconite::pyre::{Flags, Regex};

#[test]
fn matches_python_re() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/pyre.jsonl");
    let text = std::fs::read_to_string(path).unwrap();
    let mut lines = text.lines();
    let head = json::parse(lines.next().unwrap()).unwrap();
    let texts: Vec<&str> =
        head.get("texts").and_then(Value::as_array).unwrap().iter().map(|t| t.as_str().unwrap()).collect();
    let (mut n, mut bad) = (0, 0);
    for line in lines {
        let v = json::parse(line).unwrap();
        let pat = v.get("pattern").and_then(Value::as_str).unwrap();
        let mut flags = Flags::default();
        for f in v.get("flags").and_then(Value::as_array).unwrap() {
            match f.as_str().unwrap() {
                "IGNORECASE" => flags.ignore_case = true,
                "MULTILINE" => flags.multiline = true,
                "DOTALL" => flags.dotall = true,
                other => panic!("flag {other}"),
            }
        }
        n += 1;
        let rx = Regex::new(pat, flags);
        if v.get("error").is_some() {
            if rx.is_ok() {
                bad += 1;
                eprintln!("{pat:?} {flags:?}: Python rejects it, compiled");
            }
            continue;
        }
        let rx = match rx {
            Ok(rx) => rx,
            Err(e) => {
                bad += 1;
                eprintln!("{pat:?} {flags:?}: {e}");
                continue;
            }
        };
        let digits = v.get("results").and_then(Value::as_str).unwrap();
        for (t, d) in texts.iter().zip(digits.chars()) {
            let d = d.to_digit(8).unwrap();
            let want = (d & 4 != 0, d & 2 != 0, d & 1 != 0);
            let got = (rx.fullmatch(t), rx.search(t), rx.is_match_at_start(t));
            n += 1;
            if got != want {
                bad += 1;
                eprintln!("{pat:?} {flags:?} on {t:?}: python (full, search, match) = {want:?}, rust {got:?}");
            }
        }
    }
    assert_eq!(bad, 0, "{bad} of {n} differ");
}
