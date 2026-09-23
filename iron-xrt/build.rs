// SPDX-FileCopyrightText: Copyright (C) 2026 Brishen Hawkins
// SPDX-License-Identifier: Apache-2.0

//! Compiles the C++ XRT shim (`iron_xrt_shim.cpp`) with `g++`/`ar` and links
//! it plus XRT's `libxrt_coreutil`, one of two ways:
//!
//! - **static** (the default when `$XRT_STATIC_ROOT`, or its default
//!   `~/npu/xrt-static`, holds `lib/libxrt_coreutil.a` — an XRT install tree
//!   built with static archives): coreutil and the two archives it references
//!   go *into* the binary. The binary then has no `DT_NEEDED` on any
//!   `libxrt_*`, starts on a machine without XRT, and only opening the NPU
//!   needs XRT's shared pieces — `libxrt_core.so.2` and the amdxdna driver
//!   plug-in, which coreutil dlopens from `/opt/xilinx/xrt` (or
//!   `$XILINX_XRT`) and reports as a clean error when they are missing.
//!   Those plug-ins name `libxrt_coreutil.so.2` as a dependency, so a second,
//!   shared copy of coreutil loads next to the static one; `xrt.dynlist` puts
//!   every XRT symbol of the executable into its dynamic symbol table so the
//!   plug-ins bind to the static copy and the shared one stays inert (which
//!   is also why coreutil is linked `whole-archive`: a member the shim never
//!   references would otherwise be missing from the executable and resolve
//!   to the shared copy — split state). The list only reaches the binaries
//!   of the package whose build script emits it, so it is published as
//!   `DEP_IRONXRT_DYNLIST` for a dependent's build script to re-emit as
//!   `-Wl,--dynamic-list=…`.
//!
//! - **dynamic** (`$XRT_ROOT`, default `/opt/xilinx/xrt`, the tree XRT's
//!   `setup.sh` puts on the loader path): links `libxrt_coreutil.so.2`,
//!   publishes the lib dir as `DEP_IRONXRT_LIBDIR` for dependents to re-emit
//!   as an rpath.
//!
//! XRT's C++ headers need `uuid/uuid.h` (libuuid's development headers). No
//! build dependencies (no `cc` crate) so this builds offline.
//!
//! Only with the `xrt` feature (the default). Without it, or on docs.rs
//! (`$DOCS_RS`), which has neither XRT nor its headers, nothing is built or
//! linked: the rest of the crate is plain Rust, and rustdoc needs only the
//! Rust sources.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=iron_xrt_shim.cpp");
    println!("cargo:rerun-if-changed=iron_xrt_shim.h");
    println!("cargo:rerun-if-changed=xrt.dynlist");
    println!("cargo:rerun-if-env-changed=XRT_ROOT");
    println!("cargo:rerun-if-env-changed=XRT_STATIC_ROOT");
    // Without the `xrt` feature there is nothing to build or link; on
    // docs.rs there is no XRT to build against.
    if env::var_os("CARGO_FEATURE_XRT").is_none() || env::var_os("DOCS_RS").is_some() {
        return;
    }

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let static_root = static_root();
    let xrt_root = match &static_root {
        Some(root) => root.clone(),
        None => PathBuf::from(env::var("XRT_ROOT").unwrap_or_else(|_| "/opt/xilinx/xrt".to_string())),
    };
    let inc = xrt_root.join("include");
    let lib = xrt_root.join("lib");

    let obj = out_dir.join("iron_xrt_shim.o");
    let archive = out_dir.join("libironxrt.a");

    let status = Command::new("g++")
        .args(["-std=c++17", "-fPIC", "-O2", "-c", "iron_xrt_shim.cpp", "-I"])
        .arg(&inc)
        .arg("-o")
        .arg(&obj)
        .status()
        .expect("failed to invoke g++ (is it on PATH?)");
    assert!(status.success(), "g++ failed to compile iron_xrt_shim.cpp against {}", inc.display());

    let status = Command::new("ar")
        .arg("rcs")
        .arg(&archive)
        .arg(&obj)
        .status()
        .expect("failed to invoke ar");
    assert!(status.success(), "ar failed");

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=ironxrt");
    println!("cargo:rustc-link-search=native={}", lib.display());

    if let Some(root) = &static_root {
        // A refreshed tree (XRT rebuilt into it) relinks. A tree that did
        // not exist at the last build does not: cargo re-runs a build script
        // on every build when a watched path is missing, so the default
        // location is not watched — `cargo clean -p iron-xrt` after creating
        // it.
        println!("cargo:rerun-if-changed={}", root.join("lib").join("libxrt_coreutil.a").display());
        // `-bundle`: passed to the final link from this directory rather
        // than copied into the rlib, so `+whole-archive` applies there —
        // that is where the plug-ins' symbol lookups are decided.
        println!("cargo:rustc-link-lib=static:+whole-archive,-bundle=xrt_coreutil");
        println!("cargo:rustc-link-lib=static:-bundle=aiebu");
        println!("cargo:rustc-link-lib=static:-bundle=cert_dtrace");
        println!("cargo:rustc-link-lib=dylib=uuid");
        println!("cargo:rustc-link-lib=dylib=stdc++");
        let dynlist = manifest_dir.join("xrt.dynlist");
        // Only effective for this package's own binaries (its examples and
        // tests); dependents re-emit it from DEP_IRONXRT_DYNLIST.
        println!("cargo:rustc-link-arg=-Wl,--dynamic-list={}", dynlist.display());
        println!("cargo:dynlist={}", dynlist.display());
    } else {
        println!("cargo:rustc-link-lib=dylib=xrt_coreutil");
        println!("cargo:rustc-link-lib=dylib=stdc++");
        // Only effective when this crate is the final artifact (it never
        // is); dependents re-emit it from DEP_IRONXRT_LIBDIR, or rely on
        // XRT's setup.sh having put the lib dir on the loader path.
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib.display());
        println!("cargo:libdir={}", lib.display());
    }
}

/// The static XRT tree to link, if any: `$XRT_STATIC_ROOT` when set (and it
/// must then hold the archive — a set variable is a request, not a hint;
/// set but empty asks for the dynamic link even when the default tree
/// exists), else `~/npu/xrt-static` when that holds one, else none
/// (dynamic link).
fn static_root() -> Option<PathBuf> {
    let has_archive = |root: &Path| root.join("lib").join("libxrt_coreutil.a").is_file();
    if let Some(root) = env::var_os("XRT_STATIC_ROOT") {
        if root.is_empty() {
            return None;
        }
        let root = PathBuf::from(root);
        assert!(
            has_archive(&root),
            "XRT_STATIC_ROOT={} has no lib/libxrt_coreutil.a — point it at an XRT tree built with static archives, or unset it to link XRT dynamically from $XRT_ROOT",
            root.display()
        );
        return Some(root);
    }
    let home = env::var_os("HOME")?;
    let root = PathBuf::from(home).join("npu").join("xrt-static");
    has_archive(&root).then_some(root)
}
