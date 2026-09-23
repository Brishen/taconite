<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# `gaic` — GAIC image cropping on the NPU, from Rust

Replays the bundle
[`iron/applications/gaic/export_gaic.py`](../../applications/gaic/README.md)
writes. The GAIC (VGG16) backbone and first FC layer run on the NPU, the glue
runs on the host, and there is no Python, PyTorch or ONNX at run time. The
library is std-only. The `gaic` binary adds image decoding and writing through
the `image` crate (feature `cli`, on by default).

The bundle uses the format every IRON bundle shares, read with
[`iron-bundle`](../iron-bundle): `manifest.txt` (the kernels, the conv
layer table and the head, parsed and cross-checked against the tensors at
load), `tensors.txt` / `tensors.bin` (weights and the self-check's `ref.*`
references), and `kernels/`. `export_gaic.py`'s docstring lists every record
and tensor. Bundles in the older one-file-per-tensor format (`gaic 1`
manifest) are rejected; re-export them.

XRT is reached through [`iron-xrt`](../iron-xrt), vendored from
image-organizer's `crates/iron`. It is a C shim over XRT's C++ API with a
`Session` / `Kernel` / `Buffer` interface: hardware contexts stay resident,
runs are cached per argument tuple, and there are sub-buffers and async
launches.

## Build and run

The host needs XRT and `uuid/uuid.h`. On this NixOS setup, image-organizer's
`shell.nix` provides both, and XRT is linked statically from `~/npu/xrt-static`
when that tree exists (see `iron-xrt/build.rs`).

```bash
cd iron/rust/gaic
nix-shell ~/image-organizer/shell.nix --run "cargo build --release"

# Self-check against the bundle's reference image:
./target/release/gaic ~/npu/gaic/bundle-v1 check --reps 5

# GAIC-Pytorch's demo: best crop overall and at 1:1, 4:3, 16:9.
./target/release/gaic ~/npu/gaic/bundle-v1 crop --out crops/ photo.jpg ...
```

`check` compares against what the exporter recorded:

```
resize 1024x683 -> 384x256: 294912/294912 values exact, max |diff| 0e0
feature map 32x16x24: cosine vs CPU 0.999984, vs Python NPU 1.000000
83 anchors: score cosine vs CPU 1.000000, spearman 0.99994; vs Python NPU 1.000000; best 8 (CPU 8, Python NPU 8)
CHECK OK
```

To run without XRT, build with `--no-default-features --features cli,direct`.
The kernels then go through the `amdxdna` driver's ioctls
(`iron_xrt::direct`); the build needs no XRT headers or library, and the
binary links none. `check` gives the
same numbers, and `crop` the same boxes and scores.

- **Resize.** The input resize reproduces Pillow's LANCZOS bit for bit.
- **Feature map.** It is identical to the Python NPU path (cosine 1.000000).
- **Crop.** The chosen crop matches the f32 CPU reference.

## API

```rust
let mut g = gaic::Gaic::load(Path::new("gaic-bundle"))?;
let (x, (w, h)) = gaic::preprocess::preprocess(&rgb8, src_w, src_h);
let f = g.features(&x, w, h)?;                       // once per image
for (name, boxes) in gaic::anchors::demo_sets(w, h) {
    let boxes: Vec<[f32; 4]> = boxes.iter().map(|b| b.map(|v| v as f32)).collect();
    let scores = g.score(&f, &boxes)?;               // any number of boxes
}
```

`features` runs the backbone and DimRed. `score` can be called any number of
times against the same features, for example with your own candidate boxes. One
`Gaic` per process: its 9 kernels occupy 9 of the NPU's 16 hardware
contexts. It is `Send`, not `Sync`.

## Per layer

For each conv, `features` does three things:

1. **Build A.** It writes that conv's A source into one shared buffer, from
   the previous activation. Y rows (or the stem's im2col rows) are built in
   cached memory and stored a block at a time, since the BO mapping is
   uncached.
2. **Queue the chunks.** It queues the layer's M-chunks, up to 8 in flight.
   Each chunk is a (sub-buffer of A, weights, sub-buffer of the shared output)
   triple. Syncs cover each chunk's sub-buffer only: syncing the whole shared
   buffers, which are sized for the largest layer, was most of the host time
   at first.
3. **Epilogue.** It computes bias + ReLU (+ the next layer's 2×2 max-pool)
   into the next bf16 activation. For f3/f4 it keeps f32 instead.

Glue loops are threaded over rows (`--threads`, default: the machine's, at
most 16).

## Performance

NPU2 (Ryzen AI 9 HX 370), 384×256 input, warm, 8 host threads:

| stage | ms |
|---|---|
| NPU (13 convs, 73 dispatches incl. FC1) | ~33 |
| build A (Y) | ~7 |
| epilogue | ~9 |
| DimRed + align + FC2/3 | ~5 |
| **total per image (83 candidates)** | **~54** |

These were measured while other jobs shared the CPU. Under heavy host load,
the glue stages stretch several-fold.
