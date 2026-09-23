// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Compile the C++ XRT shim with g++/ar and link it plus libxrt_coreutil. No
// build dependencies (no `cc` crate) so the crate builds offline. XRT lives at
// $XRT_ROOT (default /opt/xilinx/xrt).

use std::env;
use std::path::Path;
use std::process::Command;

fn main() {
    let xrt = env::var("XRT_ROOT").unwrap_or_else(|_| "/opt/xilinx/xrt".to_string());
    let out_dir = env::var("OUT_DIR").unwrap();
    let inc = format!("{xrt}/include");
    let lib = format!("{xrt}/lib");

    println!("cargo:rerun-if-changed=iron_xrt_shim.cpp");
    println!("cargo:rerun-if-changed=iron_xrt_shim.h");
    println!("cargo:rerun-if-env-changed=XRT_ROOT");

    let obj = Path::new(&out_dir).join("iron_xrt_shim.o");
    let archive = Path::new(&out_dir).join("libtaconite_adaface.a");

    let status = Command::new("g++")
        .args([
            "-std=c++17",
            "-fPIC",
            "-O2",
            "-c",
            "iron_xrt_shim.cpp",
            "-I",
            &inc,
            "-o",
        ])
        .arg(&obj)
        .status()
        .expect("failed to invoke g++");
    assert!(status.success(), "g++ failed to compile iron_xrt_shim.cpp");

    let status = Command::new("ar")
        .arg("rcs")
        .arg(&archive)
        .arg(&obj)
        .status()
        .expect("failed to invoke ar");
    assert!(status.success(), "ar failed");

    println!("cargo:rustc-link-search=native={out_dir}");
    println!("cargo:rustc-link-lib=static=taconite_adaface");
    println!("cargo:rustc-link-search=native={lib}");
    println!("cargo:rustc-link-lib=dylib=xrt_coreutil");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    // Only effective when this crate is the final artifact; link-args do NOT
    // propagate to dependents (link-search/link-lib above do). Dependent
    // binaries re-emit the rpath from DEP_TACONITE_ADAFACE_LIBDIR (published via the
    // `links = "taconite_adaface"` key and the libdir line below) or rely on XRT's
    // setup.sh LD_LIBRARY_PATH.
    println!("cargo:rustc-link-arg=-Wl,-rpath,{lib}");
    println!("cargo:libdir={lib}");
}
