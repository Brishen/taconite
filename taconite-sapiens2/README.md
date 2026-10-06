<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Brishen Hawkins
SPDX-License-Identifier: Apache-2.0
-->

# `taconite-sapiens2` — Sapiens2-Pose keypoints on the NPU, from Rust

A Rust runtime for [`facebook/sapiens2-pose-0.4b`](https://huggingface.co/facebook/sapiens2-pose-0.4b):
an image and a person box in, 308 keypoints (body, feet, hands, face) in
image coordinates with their scores out, as HF's `Sapiens2ForPoseEstimation`
+ `post_process_pose_estimation` compute them. The backbone and the
heatmap head run on the AMD XDNA NPU (NPU2) through IRON kernels replayed
with [`taconite`](https://crates.io/crates/taconite) — through XRT, or
directly through the amdxdna driver (`--features direct`). It is the
forward of [IRON](https://github.com/amd/IRON)'s Python app
(`iron/applications/sapiens2_pose`), kernel for kernel.

```bash
sapiens2 pose <bundle> person.jpg --box 150,130,275,480 -o overlay.png --json kp.json
#   box ...: 4200 ms; npu 2400 ms (mha 800, gemms 1600), host 1600 ms ...
#   box 150,130,275,480: 290 of 308 keypoints above 0.3 (mean score 0.656)
sapiens2 check <bundle>   # the reference cases against the float32 model
```

```rust
let mut m = taconite_sapiens2::Sapiens2::load(Path::new("bundle"))?;
let b = taconite_sapiens2::BBox { x: 150.0, y: 130.0, w: 275.0, h: 480.0 };
let kps = m.pose(&rgb, width, height, &b)?; // 308 x {x, y, score}
```

Boxes are COCO `x, y, width, height`; the model is top-down, so they come
from a person detector (without `--box` the CLI uses the whole image).
Download the bundle (~730 MB: six xclbins, their instruction streams, the
packed weights, the reference cases) from Hugging Face:

```bash
hf download brishen/iron-sapiens2-pose-0.4b-npu2 --local-dir sapiens2-pose
sapiens2 check sapiens2-pose
```

It is exported by IRON's `iron/applications/sapiens2_pose/export_sapiens2.py`.

## What runs where

| | NPU | host (this crate) |
|---|---|---|
| backbone (ViT, 24 layers, 1024 wide, 3081 tokens) | patch embedding, qkv, o, the SwiGLU gate+up (one `swiglu_pair` GEMM) and down as `flm.GEMM`s, one dispatch each; the attention (MHA operator, 16 heads of 64) | the box crop (`preprocess.rs`), RMSNorms (their weights folded into the GEMMs), q / k norms, 2D RoPE, residual adds |
| head (to 308 heatmaps of 256 x 192) | both transposed convolutions — each one GEMM over its input's 2 x 2 windows (K = 4096, the four output phases in N) — the three 1 x 1 convs and the predictor | the window layout, InstanceNorm + SiLU |
| keypoints | | argmax + DARK refinement, back through the crop (`post.rs`) |

Six hardware contexts of NPU2's 16 (shared across processes). When the
device has no free slot, the least recently used context is dropped and
reloaded on its next use (`Npu::swaps` counts these).

## Accuracy

`sapiens2 check` on 5 COCO val2017 person boxes, against the float32 HF
model (ROCm): backbone features cosine 0.99995-0.99997, heatmaps
0.9993-0.9996, mean keypoint error 0.12-1.03 px (at most 0.23% of the
box) over the keypoints the reference scores above 0.3.

## License

The crate is Apache-2.0; `preprocess.rs` follows PyTorch's `grid_sample`
(BSD-3-Clause, `LICENSE-PYTORCH`). The model weights (in the bundle) are
under the [Sapiens2 License](https://github.com/facebookresearch/sapiens2/blob/main/LICENSE.md).
