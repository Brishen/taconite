// SPDX-FileCopyrightText: Copyright (C) 2026 Brishen Hawkins
// SPDX-License-Identifier: Apache-2.0

//! Re-emits taconite's link arguments: `cargo:rustc-link-arg` does not
//! propagate across crates. With XRT linked statically that is the
//! `--dynamic-list` the driver plug-ins need to bind to the executable's
//! coreutil; linked dynamically, XRT's lib dir as an rpath (see taconite's
//! build.rs). Both are also republished (`links = "taconite_adaface"`) as
//! DEP_TACONITE_ADAFACE_DYNLIST / DEP_TACONITE_ADAFACE_LIBDIR for dependent
//! binaries. The `direct` path links nothing, so there is nothing to emit.

fn main() {
    println!("cargo:rerun-if-env-changed=DEP_TACONITE_XRT_LIBDIR");
    println!("cargo:rerun-if-env-changed=DEP_TACONITE_XRT_DYNLIST");
    if let Ok(list) = std::env::var("DEP_TACONITE_XRT_DYNLIST") {
        println!("cargo:rustc-link-arg=-Wl,--dynamic-list={list}");
        println!("cargo:dynlist={list}");
    } else if let Ok(lib) = std::env::var("DEP_TACONITE_XRT_LIBDIR") {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{lib}");
        println!("cargo:libdir={lib}");
    }
}
