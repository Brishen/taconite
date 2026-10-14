// SPDX-FileCopyrightText: Copyright (C) 2026 Brishen Hawkins
// SPDX-License-Identifier: Apache-2.0

//! Feature `embed`: packs the bundle directory `$REUSE_EMBED_BUNDLE` (an
//! absolute path) into `$OUT_DIR/bundle.pack` for `embed.rs` to
//! include_bytes!, and its content hash into `REUSE_EMBED_HASH`.
//!
//! The pack: per file (sorted by path) a u32 path length, the path
//! (relative, `/`-separated), a u64 length and the bytes; little-endian.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

fn files(dir: &Path, out: &mut Vec<PathBuf>) {
    let mut v: Vec<_> = fs::read_dir(dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display())).map(|e| e.unwrap().path()).collect();
    v.sort();
    for p in v {
        if p.is_dir() {
            files(&p, out);
        } else {
            out.push(p);
        }
    }
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=REUSE_EMBED_BUNDLE");
    if std::env::var_os("CARGO_FEATURE_EMBED").is_none() {
        return;
    }
    let dir = PathBuf::from(std::env::var("REUSE_EMBED_BUNDLE").unwrap_or_else(|_| {
        panic!("feature `embed`: set REUSE_EMBED_BUNDLE to the (absolute) bundle directory to build in")
    }));
    assert!(dir.is_absolute(), "REUSE_EMBED_BUNDLE must be an absolute path: {}", dir.display());
    assert!(dir.join("manifest.txt").is_file(), "{}: no manifest.txt (not a bundle?)", dir.display());
    let mut list = Vec::new();
    files(&dir, &mut list);
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("bundle.pack");
    let mut w = std::io::BufWriter::new(fs::File::create(&out).unwrap());
    // FNV-1a over the pack: names the unpacked copy, so a rebuilt binary
    // with another bundle never reuses a stale one
    let mut h: u64 = 0xcbf29ce484222325;
    let mut put = |w: &mut std::io::BufWriter<fs::File>, b: &[u8]| {
        for &x in b {
            h = (h ^ x as u64).wrapping_mul(0x100000001b3);
        }
        w.write_all(b).unwrap();
    };
    for p in &list {
        println!("cargo:rerun-if-changed={}", p.display());
        let rel = p.strip_prefix(&dir).unwrap().components().map(|c| c.as_os_str().to_str().unwrap()).collect::<Vec<_>>().join("/");
        let data = fs::read(p).unwrap();
        put(&mut w, &(rel.len() as u32).to_le_bytes());
        put(&mut w, rel.as_bytes());
        put(&mut w, &(data.len() as u64).to_le_bytes());
        put(&mut w, &data);
    }
    w.flush().unwrap();
    println!("cargo:rustc-env=REUSE_EMBED_HASH={h:016x}");
    println!("cargo:rustc-env=REUSE_EMBED_FILES={}", list.len());
}
