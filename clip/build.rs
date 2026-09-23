// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Re-emits iron-xrt's link arguments for this package's binaries:
//! `cargo:rustc-link-arg` does not propagate across crates. With XRT linked
//! statically that is the `--dynamic-list` the driver plug-ins need to bind
//! to the executable's coreutil; linked dynamically, XRT's lib dir as an
//! rpath (see iron-xrt's build.rs).

fn main() {
    println!("cargo:rerun-if-env-changed=DEP_IRONXRT_LIBDIR");
    println!("cargo:rerun-if-env-changed=DEP_IRONXRT_DYNLIST");
    if let Ok(list) = std::env::var("DEP_IRONXRT_DYNLIST") {
        println!("cargo:rustc-link-arg=-Wl,--dynamic-list={list}");
    } else if let Ok(lib) = std::env::var("DEP_IRONXRT_LIBDIR") {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{lib}");
    }
}
