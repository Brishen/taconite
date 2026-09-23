<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# `iron-xrt` — replay IRON kernels through XRT from Rust

[IRON](https://github.com/amd/IRON) is an ahead-of-time compiler. What it
leaves behind for a kernel is an `.xclbin` plus an instruction stream
(`*.insts.bin`). This crate runs those with no Python, through a small C shim
over XRT's C++ API (`iron_xrt_shim.cpp`, compiled by `build.rs` with no `cc`
crate, so it builds offline). It provides three types:

- **`Session`**: the NPU.
- **`Kernel`**: a resident hardware context, shared by every kernel loaded
  from the same xclbin and released when the last of them is dropped. Runs
  are cached per argument tuple, with a blocking `run` and an async
  `start`/`wait`.
- **`Buffer`**: a host-visible BO the host writes in place, with `sub`-buffer
  views.

## Building

The crate targets Linux on x86-64 with an XDNA NPU (the `amdxdna` driver) and
[XRT](https://github.com/Xilinx/XRT). Building needs `g++`, `ar`, XRT's
headers and libuuid's (`uuid/uuid.h`); `build.rs` compiles the shim itself,
with no build dependencies, so it also builds offline. It links XRT one of
two ways:

- **Dynamically** (the usual case), from `$XRT_ROOT` (default
  `/opt/xilinx/xrt`). The binary then needs XRT's lib dir on the loader path,
  e.g. after sourcing XRT's `setup.sh`.
- **Statically**, from `$XRT_STATIC_ROOT` (default `~/npu/xrt-static`) when
  it holds `lib/libxrt_coreutil.a`, an XRT tree built with static archives.
  The binary then starts on a machine without XRT, and only opening the NPU
  needs XRT's driver plug-in. Set `XRT_STATIC_ROOT=` (empty) to force the
  dynamic link.

Linker arguments don't propagate across crates, so `build.rs` publishes what a
binary needs for a dependent's build script to re-emit: `DEP_IRONXRT_LIBDIR`
(dynamic; the rpath) and `DEP_IRONXRT_DYNLIST` (static; the symbol list the
driver plug-ins bind to):

```rust
// build.rs of a crate that depends on iron-xrt
fn main() {
    if let Ok(dir) = std::env::var("DEP_IRONXRT_LIBDIR") {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{dir}");
    }
    if let Ok(list) = std::env::var("DEP_IRONXRT_DYNLIST") {
        println!("cargo:rustc-link-arg=-Wl,--dynamic-list={list}");
    }
}
```

## Building kernels: `compile`

`iron_xrt::compile` builds the kernels too, still without Python. It runs
the two tools IRON's Python drives as subprocesses, with the flags IRON and
mlir-aie pass them:

- **`Toolchain::compile_kernel`**: compiles an AIE kernel (`aie_kernels/…/*.cc`)
  to an object with Peano's `clang++`, for `Arch::Aie2` (NPU1) or
  `Arch::Aie2p` (NPU2).
- **`Toolchain::compile_design`**: compiles a design's MLIR (what `design.py`
  generates) to the `.xclbin` + instruction stream that `load_kernel` takes,
  with the native `aiecc` binary. It links the design's `link_with` objects.

```rust
use iron_xrt::compile::{Arch, Design, KernelSource, Toolchain};

let tc = Toolchain::from_env()?; // $MLIR_AIE_INSTALL_DIR (+ $PEANO_INSTALL_DIR, $AIECC_PATH)
tc.compile_kernel(&KernelSource::new("aie_kernels/generic/mul.cc", Arch::Aie2p), "out/mul.o".as_ref())?;
tc.compile_design(&Design::new("mul.mlir", "out/work").link("out/mul.o"),
                  "out/mul.xclbin".as_ref(), "out/mul.insts.bin".as_ref())?;
```

aiecc packages the xclbin with XRT's `xclbinutil`, so that must be on its
`PATH`. mlir-aie's binaries are built for a generic Linux. Where they can't
run as-is (NixOS, say), run them in a container instead: `Toolchain::with_launcher` prefixes every tool
invocation with a command, for example a script that runs
`podman run … --entrypoint "$1" <image> "${@:2}"` with the same paths mounted.

`examples/eltwise_mul.rs` goes from source to NPU. It builds `mul.cc` and an
`ElementwiseMul` design MLIR, runs the result, and checks it against the
CPU. Built this way, the object and the instruction stream are byte-identical
to IRON's. The xclbin differs only in its timestamps and UUIDs.

Adapted from [RLX](https://github.com/MIT-RLX/rlx)'s `rlx-xdna` `compile.rs`
(MIT OR Apache-2.0).
