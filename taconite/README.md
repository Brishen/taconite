<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Brishen Hawkins
SPDX-License-Identifier: Apache-2.0
-->

# `taconite` — replay IRON kernels through XRT from Rust

[IRON](https://github.com/amd/IRON) is an ahead-of-time compiler. What it
leaves behind for a kernel is an `.xclbin` plus an instruction stream
(`*.insts.bin`). This crate runs those through a small C shim
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

The crate targets Linux on x86-64 with an XDNA NPU (the `amdxdna` driver).
The XRT types above need the `xrt` feature, on by default. Without it
(`default-features = false`) nothing below applies: the crate is plain Rust
(`compile`, `direct`, `Error`, the bf16 helpers) and builds with no XRT, no
C++ compiler and no system headers:

```toml
taconite = { version = "0.1", default-features = false }
```

With `xrt`, building needs [XRT](https://github.com/Xilinx/XRT): `g++`,
`ar`, XRT's headers and libuuid's (`uuid/uuid.h`). `build.rs` compiles the
shim itself, with no build dependencies, so it also builds offline. It links
XRT one of two ways:

- **Dynamically** (the usual case), from `$XRT_ROOT` (default
  `/opt/xilinx/xrt`). The binary then needs XRT's lib dir on the loader path,
  e.g. after sourcing XRT's `setup.sh`.
- **Statically**, from `$XRT_STATIC_ROOT` (default `~/npu/xrt-static`) when
  it holds `lib/libxrt_coreutil.a`, an XRT tree built with static archives.
  The binary then starts on a machine without XRT, and only opening the NPU
  needs XRT's driver plug-in. Set `XRT_STATIC_ROOT=` (empty) to force the
  dynamic link.

Linker arguments don't propagate across crates, so `build.rs` publishes what a
binary needs for a dependent's build script to re-emit: `DEP_TACONITE_XRT_LIBDIR`
(dynamic; the rpath) and `DEP_TACONITE_XRT_DYNLIST` (static; the symbol list the
driver plug-ins bind to):

```rust
// build.rs of a crate that depends on taconite
fn main() {
    if let Ok(dir) = std::env::var("DEP_TACONITE_XRT_LIBDIR") {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{dir}");
    }
    if let Ok(list) = std::env::var("DEP_TACONITE_XRT_DYNLIST") {
        println!("cargo:rustc-link-arg=-Wl,--dynamic-list={list}");
    }
}
```

## Building kernels: `compile`

`taconite::compile` builds the kernels too. It runs
the two tools IRON's Python drives as subprocesses, with the flags IRON and
mlir-aie pass them:

- **`Toolchain::compile_kernel`**: compiles an AIE kernel (`aie_kernels/…/*.cc`)
  to an object with Peano's `clang++`, for `Arch::Aie2` (NPU1) or
  `Arch::Aie2p` (NPU2).
- **`Toolchain::compile_design`**: compiles a design's MLIR (what `design.py`
  generates) to the `.xclbin` + instruction stream that `load_kernel` takes,
  with the native `aiecc` binary. It links the design's `link_with` objects.

```rust
use taconite::compile::{Arch, Design, KernelSource, Toolchain};

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

## Running without XRT: `direct`

`taconite::direct` runs the same kernels with no XRT and no C++ (build with
`default-features = false` to leave XRT out entirely). It is
Linux/x86-64 only, and it talks to the `amdxdna` driver's ioctls on
`/dev/accel/accel0` itself. Its types and methods match the XRT API, so a
model switches paths by changing its `use` line:

- **`Session`**: the accel node, plus the process's 64 MiB device heap.
- **`Kernel`**: an xclbin and an instruction stream. Kernels loaded from the
  same xclbin (and kernel name) share one hardware context. `run` waits for
  the kernel; `start` returns a `Run` to `wait` on, and several can be in
  flight at once.
- **`Buffer`**: shared buffers the host fills in place, with `sub`-buffer
  views. A kernel takes up to five of them, the `MLIR_AIE` kernel's
  `bo0`…`bo4`.

It issues the same ioctl sequence XRT does. The sequence and the command
packet were captured by tracing this crate's XRT path, and `direct` reproduces
it, so each run is one `EXEC_CMD` plus one fence wait. `Session` can also
query the firmware version and the hardware contexts (with their command
counters and fault state), and set the NPU power mode (this needs root).

On an NPU2 it computes the same results as XRT at the same speed:

- **Complete models.** Switched over by their `use` line, a VGG16-based
  image-cropping model (73 dispatches per image) gives the same boxes and
  scores as through XRT at ~50 ms per image, and SAM3 and CLIP ViT-H/14 give
  identical outputs at the same speed.
- **A single kernel.** `examples/direct_eltwise_mul.rs` multiplies 1,638,400
  bf16 values with a median run of 235 µs through `direct` and 232 µs through
  XRT, with 0 mismatches either way.
- **Hardware tests.** `tests/direct_hw.rs` (`--ignored`, needs an NPU) covers
  context sharing and release (60 contexts), sub-buffers with runs in flight,
  and the argument limit.

It does not cover xclbins with more than one PDI.

Adapted from [RLX](https://github.com/MIT-RLX/rlx)'s `rlx-xdna` `direct.rs`
(MIT OR Apache-2.0). RLX never saw a command complete on NPU1. The likely
cause is its fence wait's timeout: the DRM syncobj ioctls read the timeout as
an absolute deadline, and RLX passed a relative one. That makes the wait
return at once with the command still `NEW`, and it reproduces here.
