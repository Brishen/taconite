// SPDX-FileCopyrightText: Copyright (C) 2026 Brishen Hawkins
// SPDX-License-Identifier: Apache-2.0

//! Runs each GEMM of a bundle once on zero buffers at the given chunk
//! counts (default 1), reporting which complete: a hardware probe.
//! `cargo run --release --example gemm_probe -- <bundle> [prefix] [chunks...]`

use std::path::Path;
use std::time::Instant;

use taconite_bundle::Manifest;
use taconite_qwen35::npu::{Npu, gemm_key};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let bundle = Path::new(args.first().ok_or("usage: gemm_probe <bundle> [prefix] [chunks...]")?);
    let prefix = args.get(1).map_or("", String::as_str);
    let counts: Vec<usize> = args.iter().skip(2).map(|a| a.parse()).collect::<Result<_, _>>()?;
    let counts = if counts.is_empty() { vec![1] } else { counts };
    let m = Manifest::load(bundle, taconite_qwen35::VERSION)?;
    let npu = Npu::open(&m)?;
    let mut keys: Vec<&String> = npu.gemms.keys().filter(|k| k.starts_with(prefix)).collect();
    keys.sort();
    for key in keys {
        let g = npu.gemm(key)?.clone();
        for &n in &counts {
            if n > g.a_elems.len() {
                continue;
            }
            let a = npu.zeros(g.a_elems[n - 1] + g.lda)?;
            let b = npu.zeros(g.b_bytes.div_ceil(2))?;
            let c = npu.zeros(g.c_elems[n - 1])?;
            let t0 = Instant::now();
            let r = npu.run(&gemm_key(key, n), &[&a, &b, &c]);
            println!("{key} x{n}: {:?} in {:.1} ms", r.map(|_| "ok"), t0.elapsed().as_secs_f64() * 1e3);
        }
    }
    Ok(())
}
