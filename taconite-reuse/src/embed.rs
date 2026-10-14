// SPDX-FileCopyrightText: Copyright (C) 2026 Brishen Hawkins
// SPDX-License-Identifier: Apache-2.0

//! A bundle built into the binary (feature `embed`: `REUSE_EMBED_BUNDLE`
//! at build time, see build.rs). The loaders want files -- the weights are
//! mmapped, XRT loads xclbins from paths -- so the first use unpacks it to
//! `$XDG_CACHE_HOME/taconite-reuse/bundle-<hash>` (else
//! `~/.cache/...`), and later ones find it there.

use std::fs;
use std::path::PathBuf;

use crate::{Error, Reuse};

static PACK: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/bundle.pack"));

/// The built-in bundle's content hash (names its unpacked copy).
pub const HASH: &str = env!("REUSE_EMBED_HASH");

const DONE: &str = ".unpacked";

fn cache_root() -> Result<PathBuf, Error> {
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))
        .ok_or_else(|| Error::Input("no XDG_CACHE_HOME or HOME to unpack the built-in bundle under".into()))?;
    Ok(base.join("taconite-reuse"))
}

/// The built-in bundle's directory, unpacked if it isn't yet.
pub fn dir() -> Result<PathBuf, Error> {
    let root = cache_root()?;
    let dst = root.join(format!("bundle-{HASH}"));
    if dst.join(DONE).is_file() {
        return Ok(dst);
    }
    let io = |e: std::io::Error| Error::Input(format!("unpacking the built-in bundle under {}: {e}", root.display()));
    fs::create_dir_all(&root).map_err(io)?;
    // into a private directory, then renamed: a concurrent first run, or
    // one killed halfway, never leaves a partial bundle at `dst`
    let tmp = root.join(format!(".bundle-{HASH}.{}", std::process::id()));
    let _ = fs::remove_dir_all(&tmp);
    let bad = || Error::Input("the built-in bundle is corrupt".into());
    let mut i = 0;
    let take = |i: &mut usize, n: usize| -> Result<&[u8], Error> {
        let s = PACK.get(*i..*i + n).ok_or_else(bad)?;
        *i += n;
        Ok(s)
    };
    while i < PACK.len() {
        let n = u32::from_le_bytes(take(&mut i, 4)?.try_into().unwrap()) as usize;
        let rel = std::str::from_utf8(take(&mut i, n)?).map_err(|_| bad())?;
        let len = u64::from_le_bytes(take(&mut i, 8)?.try_into().unwrap()) as usize;
        let data = take(&mut i, len)?;
        if rel.split('/').any(|c| c.is_empty() || c == "." || c == "..") {
            return Err(bad());
        }
        let p = tmp.join(rel);
        fs::create_dir_all(p.parent().unwrap()).map_err(io)?;
        fs::write(&p, data).map_err(io)?;
    }
    fs::write(tmp.join(DONE), HASH).map_err(io)?;
    match fs::rename(&tmp, &dst) {
        Ok(()) => Ok(dst),
        // another process got there first
        Err(_) if dst.join(DONE).is_file() => {
            let _ = fs::remove_dir_all(&tmp);
            Ok(dst)
        }
        Err(e) => Err(io(e)),
    }
}

impl Reuse {
    /// Loads the bundle built into the binary (unpacking it on first use).
    pub fn load_embedded() -> Result<Self, Error> {
        Reuse::load(&dir()?)
    }
}
