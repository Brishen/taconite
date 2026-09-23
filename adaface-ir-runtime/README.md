<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# adaface-ir-runtime

An **embeddable** Rust library, with no crates.io dependencies, that runs
AdaFace IR face embedders (IR-18, IR-101) on the AIE NPU with no Python at
inference time. It replays a *bundle* — compiled conv kernels, packed
weights, and the network's steps in order, in the format every IRON bundle
shares (read with the in-repo, std-only `iron/rust/iron-bundle`) — exported
ahead of time by
`iron/applications/adaface_ir18/export_ir18.py` /
`iron/applications/adaface_ir101/export_ir101.py` (iron acts purely as an
AOT compiler). The `run_ir18` / `run_ir101` binaries under the app
directories are thin CLI wrappers over this crate.

## Using it from another Rust app

```toml
[dependencies]
adaface-ir-runtime = { path = ".../iron/rust/adaface-ir-runtime" }  # or a git dep
```

```rust
use adaface_ir_runtime::IrEmbedder;

fn main() -> Result<(), adaface_ir_runtime::Error> {
    // Load once at startup; keep it for the life of the process.
    let mut emb = IrEmbedder::load("/path/to/ir101_bundle")?;

    // Per face: an aligned 112x112 crop, channel-major (CHW) f32, with the
    // cvlface preprocessing ((rgb/255) - 0.5) / 0.5. Alignment (landmark
    // detection + similarity warp) is the caller's job, as with the
    // original model.
    let chw: Vec<f32> = vec![0.0; 3 * 112 * 112]; // your image here
    let embedding = emb.embed(&chw)?; // 512-d face embedding

    // Compare identities by cosine similarity.
    let _ = adaface_ir_runtime::cosine(&embedding, &embedding);
    Ok(())
}
```

Everything returns `Result<_, Error>` — the library never prints, panics on
bad bundles, or exits the host process.

### Performance model

`IrEmbedder::load` reads every weight into RAM and checks every step against
them (a malformed bundle fails here, naming the manifest line). The first
`embed()` creates one XRT hardware context per distinct conv kernel (16 for
IR-18/IR-101 — exactly the NPU2's concurrent-context cap); the session keeps
them all resident, so **every
later forward reloads nothing** (measured: IR-18 ~0.15 s, IR-101 ~0.31 s
warm). This is the whole point of the runtime — so:

- Construct **one** `IrEmbedder` per process and reuse it. A second
  simultaneous session fights over the 16-context cap and forces reloads.
- `IrEmbedder` is `Send` but not `Sync`: move it into a worker thread, or
  share it behind a `Mutex`. Forwards are serialized either way — the NPU
  path is one hardware queue.

`IRON_XRT_TIMING=1` makes the shim print context/BO/run timing at session
close; `IRON_XRT_CTX_CACHE` overrides the resident-context cap (default 16).

### Build requirements

- **XRT** headers/libs at `$XRT_ROOT` (default `/opt/xilinx/xrt`) and `g++`
  at build time: this crate's `build.rs` compiles the bundled C++ shim
  (`iron_xrt_shim.cpp`, one `xrt::hw_context` LRU) and links
  `libxrt_coreutil`. No crates.io dependencies (only the in-repo
  `iron-bundle`), so it builds offline.
- At link time the flags propagate to your binary automatically. The
  **rpath does not** (cargo drops `rustc-link-arg` across crates): either
  run with XRT's `setup.sh` sourced (`LD_LIBRARY_PATH`), or emit the rpaths
  in your own `build.rs` — `$ORIGIN/lib` for the portable layout below,
  plus the installed-XRT libdir this crate publishes via its
  `links = "ironxrt"` key (this is what the `run_ir*` binaries do):

  ```rust
  // build.rs of your app
  fn main() {
      println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/lib");
      if let Ok(lib) = std::env::var("DEP_IRONXRT_LIBDIR") {
          println!("cargo:rustc-link-arg=-Wl,-rpath,{lib}");
      }
  }
  ```

- At run time: the NPU device (`/dev/accel/accel0`, amdxdna driver) and the
  bundle directory. The bundle is plain data — `manifest.txt`, `kernels/`
  (~10 MB of xclbins/insts) and `tensors.bin` (the weights and reference
  tensors; 47 MB in all for IR-18, 126 MB for IR-101); ship it next to your
  binary and pass its path to `IrEmbedder::load`.

### Portable deployment (no XRT install on the target)

Static linking of XRT is not possible with stock XRT: the NPU backend
(`libxrt_driver_xdna.so`, from the amd/xdna-driver repo) is shared-only and
is *discovered* at run time — XRT locates its root via `dladdr()` on
`libxrt_coreutil.so` (on Linux `XILINX_XRT` is ignored), then scans that
same `lib/` directory for `libxrt_driver_*.so.2` and `dlopen`s them. But
that exact mechanism makes a fully self-contained app directory work with
no install, no environment variables:

```
myapp/
  run_ir101              (rpath $ORIGIN/lib)
  lib/libxrt_coreutil.so.2
  lib/libxrt_driver_xdna.so.2
  lib/libxrt_core.so.2   (dependency of the plugin)
  bundle/...
```

`package_portable.sh <binary> <outdir> [bundle_dir]` (in this directory)
assembles it from an installed XRT (`$XRT_ROOT`, default `/opt/xilinx/xrt`).
The `lib/` **subdirectory** is load-bearing — XRT strips the lib component
from coreutil's location to find its root, so the `.so` files must not sit
beside the binary.

Verified on NPU2: the packaged directory runs IR-101 inference in a stock
`ubuntu:24.04` container (no `/opt/xilinx`, no XRT packages, no
`LD_LIBRARY_PATH`) with identical results. What the target still needs, and
what can never be linked in: the amdxdna kernel driver + NPU firmware,
access to `/dev/accel/accel0`, a sufficient memlock limit, and distro
basics (glibc, `libstdc++`, `libuuid`).

## API summary

| item | purpose |
|---|---|
| `IrEmbedder::load(dir)` | read and check the bundle, open the NPU device |
| `embed(&mut self, chw: &[f32])` | embed a caller image (pads channels, rounds to bf16 round-to-nearest-even — bit-identical to torch) |
| `embed_recorded(&mut self)` | replay the bundle's recorded input (self-check / benchmark) |
| `input_dims() / embed_dim() / model() / num_conv_dispatches()` | bundle properties |
| `cosine(a, b)` | embedding similarity |
| `cli(net_name)` | the whole `run_ir*` CLI, built on the API above |
| `Error` | `Io` / `Plan` / `Xrt` / `Input` |

## Correctness

The exporter records the steps by *running* the Python NPU implementation, so
the steps and Python share numerics; `ref.npu_embedding` (recorder embedding)
and `ref.cpu_embedding` (fp32 CPU reference) ride along in the bundle, and the
CLI checks both every run. `ref.input_chw` (the raw f32 input) additionally routes
through the public `embed()` API — its pad + round + tile path reproduces
the recorded replay bit-for-bit (verified cosine 1.00000 on NPU2 for both
IR-18 and IR-101; cosine vs the fp32 reference is 0.9993 / 0.9974).

See `iron/applications/adaface_ir18/README.md` for how the network maps onto
NPU kernels and why the glue/head run on the CPU.
