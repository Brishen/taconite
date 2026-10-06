// SPDX-FileCopyrightText: Copyright (C) 2026 Brishen Hawkins
// SPDX-License-Identifier: Apache-2.0

//! `json.rs` against Python's `json`, `repr(float)` and `round`
//! (`tests/fixtures/json.jsonl`, from `scripts/gen_json_fixtures.py`).

use std::path::Path;

use taconite::json::{self, Value, py_float_repr, py_round};

fn bits(v: &Value, key: &str) -> f64 {
    let s = match v.get(key) {
        Some(Value::Int(s)) => s,
        other => panic!("{key}: {other:?}"),
    };
    f64::from_bits(s.parse().unwrap())
}

#[test]
fn python_json() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/json.jsonl");
    let text = std::fs::read_to_string(path).unwrap();
    let (mut n, mut bad) = ([0usize; 4], Vec::new());
    for line in text.lines() {
        let c = json::parse(line).unwrap();
        let input = c.get("in").and_then(Value::as_str);
        match c.get("kind").and_then(Value::as_str).unwrap() {
            "loads" => {
                n[0] += 1;
                match json::parse(input.unwrap()) {
                    Ok(v) => {
                        let got = (json::dumps(&v), json::dumps_with(&v, ",", ":"));
                        let want =
                            (c.get("out").unwrap().as_str().unwrap(), c.get("compact").unwrap().as_str().unwrap());
                        if (got.0.as_str(), got.1.as_str()) != want {
                            bad.push(format!("loads {input:?}: got {got:?}, want {want:?}"));
                        }
                    }
                    Err(e) => bad.push(format!("loads {input:?}: {e}")),
                }
            }
            "error" => {
                n[1] += 1;
                if let Ok(v) = json::parse(input.unwrap()) {
                    bad.push(format!("{input:?} parsed as {v:?}"));
                }
            }
            "repr" => {
                n[2] += 1;
                let x = bits(&c, "bits");
                let want = c.get("out").unwrap().as_str().unwrap();
                if py_float_repr(x) != want {
                    bad.push(format!("repr {x:e}: got {}, want {want}", py_float_repr(x)));
                }
            }
            "round" => {
                n[3] += 1;
                let (x, want) = (bits(&c, "bits"), bits(&c, "out_bits"));
                let nd = c.get("n").unwrap().as_i64().unwrap() as i32;
                let got = py_round(x, nd);
                if got.to_bits() != want.to_bits() {
                    bad.push(format!("round({x:?}, {nd}): got {got:?}, want {want:?}"));
                }
            }
            k => panic!("unknown kind {k}"),
        }
    }
    println!("{} loads/dumps, {} errors, {} reprs, {} rounds", n[0], n[1], n[2], n[3]);
    assert!(bad.is_empty(), "{} mismatches:\n{}", bad.len(), bad.join("\n"));
}

#[test]
fn duplicate_keys_keep_first_position() {
    let v = json::parse(r#"{"a": 1, "b": 2, "a": 3}"#).unwrap();
    assert_eq!(json::dumps(&v), r#"{"a": 3, "b": 2}"#);
    assert_eq!(v.get("a").and_then(Value::as_i64), Some(3));
}

#[test]
fn duplicate_keys_in_a_large_object() {
    // past the size where repeated keys are looked up through an index
    let keys: Vec<String> = (0..40).map(|i| format!("\"k{i}\": {i}")).collect();
    let text = format!("{{{}, \"k3\": -1, \"k39\": -2, \"new\": 0}}", keys.join(", "));
    let v = json::parse(&text).unwrap();
    let o = v.as_object().unwrap();
    assert_eq!(o.len(), 41);
    assert_eq!((o[3].0.as_str(), o[3].1.as_i64()), ("k3", Some(-1)));
    assert_eq!((o[39].0.as_str(), o[39].1.as_i64()), ("k39", Some(-2)));
    assert_eq!(o[40].0, "new");
}
