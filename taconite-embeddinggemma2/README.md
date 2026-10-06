<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Brishen Hawkins
SPDX-License-Identifier: Apache-2.0
-->

# `taconite-embeddinggemma2` — EmbeddingGemma 2 image embeddings on the NPU, from Rust

A Rust runtime for [`google/embeddinggemma-2`](https://huggingface.co/google/embeddinggemma-2)'s
image path: an image in, the L2-normalized 768-d embedding
sentence-transformers computes for it out, with both of the model's
transformers on the AMD XDNA NPU (NPU2) through IRON kernels replayed with
[`taconite`](https://crates.io/crates/taconite). It is the forward of
[IRON](https://github.com/amd/IRON)'s Python app
(`iron/applications/embeddinggemma2`), kernel for kernel. The embeddings
share the model's space with its text (and audio / video) embeddings, so
they compare by cosine against text embeddings made anywhere else.

A prebuilt NPU2 bundle is on Hugging Face:
[`brishen/iron-embeddinggemma2-npu2`](https://huggingface.co/brishen/iron-embeddinggemma2-npu2)
(`hf download brishen/iron-embeddinggemma2-npu2 --local-dir embeddinggemma2`).

```bash
cargo install taconite-embeddinggemma2   # over XRT
# or, with no XRT at all (straight to the amdxdna driver):
cargo install taconite-embeddinggemma2 --no-default-features --features cli,direct
```

```bash
embeddinggemma2 embed <bundle> cats.jpg car.png [--dim 256] [-o emb.f32]
#   cats.jpg: 1200 ms; npu 620 ms (mha 255), ...
#   cosine similarity: ...
embeddinggemma2 check <bundle>   # every stage against the float32 references
```

```rust
let mut m = taconite_embeddinggemma2::EmbeddingGemma2::load(Path::new("bundle"))?;
let e = m.embed_rgb(&rgb, width, height)?;              // [768], unit length
let e256 = taconite_embeddinggemma2::truncate(&e, 256); // Matryoshka, re-normalized
```

The bundle comes from `iron/applications/embeddinggemma2/export_eg2.py`
(~770 MB: three xclbins, their instruction streams, the packed weights).

## What runs where

| | NPU | host (this crate) |
|---|---|---|
| vision tower (16 layers, 768 wide, up to 2520 patches) | every projection as an `flm.GEMM` (patch embedding, qkv, o, the GeGLU gate+up as one `geglu_pair` GEMM, down as four K slices read in place, embed_vision) and the bidirectional attention (the MHA operator, 12 heads of 64, one dispatch an image, a 128-row length bucket's stream with its key count rewritten) | resize + patchify (`preprocess.rs`), position embeddings, RMSNorms, q / k / v norms, 2D RoPE, residual adds, 3 x 3 pooling |
| text encoder (24 layers, 512 wide, ~270 tokens) | every projection: the per-layer-input projection, qkv, o, GeGLU, down, the PLE gate (tanh-GELU epilogue) and projection | RMSNorms, RoPE, attention (head dims 256 / 512), the per-layer gating, mean pooling, the 512 -> 768 projection |

Three hardware contexts (NPU2 has 16, shared by every process). A kernel
loads on first use; when the device has no free slot for a context, the
others' kernels are dropped and reloaded when needed, which keeps it
working (~33 swaps an image) beside programs holding most of the slots.

- **Preprocessing is HF's `Gemma4ImageProcessor`, bit for bit**: the
  aspect-ratio-preserving size (sides multiples of 48, at most 280 x 9
  patches), torchvision's antialiased bicubic resize of the uint8 image
  (its fixed-point two-pass resampler), x / 255.
- Every RMSNorm feeding a projection has its weight folded into the packed
  weights; GEMM operands are bfp16 hi + lo pairs (`split_a`, `split_b`).

## Validation

`embeddinggemma2 check` on the four reference images (HF transformers +
sentence-transformers 6.1, float32, on the ROCm iGPU): pixels and patch
positions identical; embedding cosine 0.99977-0.99993 vs the float32
model; each image ranks the same of five text queries first, image x text
cosines within 0.0014. ~1.1-1.2 s an image on a Ryzen AI 9 HX 370 (NPU ~0.6
s, of which the vision attention ~0.25 s).

## Features

`xrt` (default) runs the kernels through XRT; `direct` through the amdxdna
driver's ioctls with no XRT (`--no-default-features --features
cli,direct`). `cli` builds the binary (the `image` crate decodes files);
the library takes RGB8 pixels.

## License

Apache-2.0. `preprocess.rs`'s resampler ports torch's antialiased uint8
resize (BSD-3-Clause, `LICENSE-PYTORCH`), which follows Pillow's
(MIT-CMU, `LICENSE-PILLOW`). The model's weights are Google's, under
Apache-2.0.
