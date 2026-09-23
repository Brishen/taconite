<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# `clip` — CLIP ViT-H/14 on the NPU, from Rust

A zero-Python runtime for
[`laion/CLIP-ViT-H-14-laion2B-s32B-b79K`](https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K):
image and text embeddings and zero-shot classification, with both
transformers on the AMD XDNA NPU (NPU2) through IRON kernels replayed with
[`iron-xrt`](../iron-xrt). It is the forward of the Python app
[`iron/applications/clip_vit_h14`](../../applications/clip_vit_h14), kernel
for kernel.

```bash
clip classify <bundle> cats.jpg car.png --label cat --label dog --label car
#   cats.jpg: cat 1.000, dog 0.000, car 0.000
#   car.png: car 0.999, cat 0.000, dog 0.000
clip embed <bundle> --image a.jpg --text "a photo of a cat" -o emb.f32
clip check <bundle>          # every stage against the references
```

```rust
let mut clip = clip::Clip::load(Path::new("bundle"))?;
let px = clip.preprocess(&rgb, width, height);          // [3, 224, 224]
let img = clip.encode_images(&px)?;                      // [n, 1024]
let txt = clip.encode_texts(&["a photo of a cat"])?;     // [m, 1024]
let logits = clip::logits(&img, &txt, 1024, clip.cfg.logit_scale);
```

## What runs where

| | NPU | host (this crate) |
|---|---|---|
| image | all 32 layers: qkv / o / fc1 (+GELU) / fc2 GEMMs, attention (16 heads of 80), residual adds + LayerNorms, each kernel reading the last one's output in place (`tower.rs`) | decode (the `image` crate), resize + crop + normalise (`preprocess.rs`), the patch embedding (f32), CLS + position embedding, pre-/post-LayerNorm, projection |
| text | all 24 layers, the same kernels, causal attention | BPE tokenizer (`sam3`'s), token + position embedding, final LayerNorm at the end token, projection |

8 hardware contexts (NPU2 has 16), all loaded at start-up; 4 images and 8
prompts a pass. The library is std-only; the CLI adds the `image` crate.

- **Preprocessing is HF's `CLIPImageProcessor`, bit for bit**: torchvision's
  antialiased bicubic resize of the uint8 image (its fixed-point two-pass
  resampler, int16 weights), center crop, the fused `(x - 255 mean) / (255
  std)` in f32. On a PNG every one of the 150528 values matches; JPEGs
  differ only through decoding (the `image` crate vs PIL, a level here and
  there).
- **The patch embedding runs on the host in f32**, as the model's conv. As
  an NPU GEMM its bf16 output, rounded before the position embedding and
  pre-LayerNorm, cost ~0.0025 image-embedding cosine for ~25 ms.
- Images sit 264 rows apart (257 tokens rounded up to 8), so an image's
  embedding does not depend on the others in its pass; see the Python app's
  README.

## Validation

`clip check <bundle>` on NPU2 (`~/npu/clip/bundle`, two reference sets of 4
images from the float32 model on the Radeon 890M via ROCm):

```
tokenizer:
  [ok] 8/8 texts tokenize as HF's CLIPTokenizer
reference set 0: 4 images x 8 prompts (cat, dog, car, person, laptop, kitchen, bicycle, pizza)
  [ok] preprocess 0_car.png: 0 of 150528 values differ from HF's (max 0.0000)
  [ok] image embeddings: cosine to float32 min 0.99944 (mean 0.99966); to the Python NPU app min 0.99984
  [ok] text embeddings: cosine to float32 min 0.99912 (mean 0.99954); to the Python NPU app min 1.00000
  [ok] zero-shot: top-1 agrees 4/4; logits max |diff| 0.350, probabilities 0.0002
  [ok] 0_cats.jpg: cat 1.000 (reference cat 1.000)
  ...
reference set 1: 4 images x 12 prompts (tabby cat, siamese cat, kitten, ...)
  [ok] zero-shot: top-1 agrees 4/4; logits max |diff| 0.344, probabilities 0.0057
  [ok] 1_cats.jpg: tabby cat 0.884 (reference tabby cat 0.871)
  ...
ALL CHECKS PASSED
```

The text tower matches the Python app exactly; the image side differs from
it only by the host's float arithmetic (patch embedding, LayerNorms).

## Performance

Warm, 4 images (decoded from files) x 8 labels, `clip classify`, Ryzen AI 9
HX 370: **~1.3-1.5 s** a forward (the spread is the machine's power state),
of which the NPU is ~1.2-1.4 s (vision fc1 0.27, qkv 0.12, o 0.10, fc2
0.12, MHA 0.08, AddLN 0.07; text 0.45) and the host ~80 ms (decode and
preprocess ~50, the patch embedding ~30). Loading the bundle takes 0.7 s
(the Python app compiles and packs for ~30 s). The float32 model on the
890M iGPU takes 2.75 s for the same forward.

## Building and running

```bash
# the bundle (in the NPU container, see the Python app)
python -m iron.applications.clip_vit_h14.export_clip --out bundle --ref ref_coarse.npz --ref ref_fine.npz

# the binary (native; XRT as for the other IRON Rust runtimes)
cargo build --release
./target/release/clip check bundle
```

To run without XRT, build with `--features direct`. The kernels then go
through the `amdxdna` driver's ioctls (`iron_xrt::direct`). On NPU2, `check`
prints the same numbers as the XRT build, at the same speed (~1.29 s for 4
images × 8 labels).

The bundle (1.3 GiB) holds every kernel, the packed weights, the tokenizer,
the reference sets and their images; its format is `iron-bundle`'s plus
the records `export_clip.py`'s docstring lists.

## Not done

- Batches are fixed at export (4 images, 8 prompts a pass); a smaller
  request still runs the full pass.
- The tokenizer and host math come from the `sam3` crate; a shared
  runtime-helpers crate would be the cleaner home.
