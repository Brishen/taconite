<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Brishen Hawkins
SPDX-License-Identifier: Apache-2.0
-->

# taconite-qwen35

[Qwen3.5-2B](https://huggingface.co/Qwen/Qwen3.5-2B) chatting about
text and images on an AMD XDNA NPU (Ryzen AI, NPU2), with every weight matrix on
the device: IRON kernels replayed through XRT, or directly through the
`amdxdna` driver (`direct` feature, no XRT).

The model is a hybrid of 18 Gated DeltaNet (linear attention) layers and 6
gated softmax-attention layers, each followed by a SwiGLU MLP.

| | NPU | host |
| --- | --- | --- |
| the prompt | every projection as an `flm.GEMM` over 256-row chunks, one dispatch a projection, one hardware context | tokenizer, embedding, norms, the DeltaNet's causal conv, gates and recurrence, partial RoPE, GQA attention and its KV cache |
| each generated token | every projection and the tied LM head (248320 rows, 5 dispatches) as a `GEMVbfp16`, a second context | the same, one row |
| an image (the 24-block vision tower) | every projection as an `flm.GEMM` (K = 1024), a third context | JPEG / PNG decoding (CLI), the processor's resize and patchify (torchvision-exact), LayerNorms, 2D RoPE, attention |

The multi-token-prediction head is not used.

## Bundle

A prebuilt NPU2 bundle is on Hugging Face,
[`brishen/iron-qwen3.5-2b-npu2`](https://huggingface.co/brishen/iron-qwen3.5-2b-npu2):

```bash
hf download brishen/iron-qwen3.5-2b-npu2 --local-dir qwen3.5-2b
```

The bundle (~2.5 GB: every weight once, bfp16; the vision tower's int8
per channel inside bfp16 blocks) holds the kernels, the packed weights,
the checkpoint's `tokenizer.json`, tokenizer test strings and the float32
references `check` compares against. IRON's exporter
(`iron/applications/qwen3_5/export_qwen35.py`) builds one from the
checkpoint.

The repo's `bfp6s` branch holds a 2.0 GB bundle whose text weights are
bfp6s16 (6.5 bits a weight, one copy for prefill and decode, unpacked on
the NPU's cores; the decode GEMVs on two cores a column): ~18 tokens/s,
no disagreement with float32 beyond near-ties. It needs this crate >=
0.1.1 (the embeddings are decoded from the head's bfp6s rows):

```bash
hf download brishen/iron-qwen3.5-2b-npu2 --revision bfp6s --local-dir qwen3.5-2b-bfp6s
```

## Use

```bash
cargo install taconite-qwen35                                             # XRT
cargo install taconite-qwen35 --no-default-features --features cli,direct # no XRT

qwen35 <bundle> --prompt "Why is the sky blue?"          # streams the answer
qwen35 <bundle> --thinking --max-new 1024 --prompt "..." # reason in <think> first
qwen35 <bundle> --image photo.jpg --prompt "Describe this image."   # --image repeats
qwen35 <bundle> --interactive                            # one prompt a line
qwen35 check <bundle>                                    # verify the bundle
```

`--timing` breaks a turn down per projection and host stage. The bundle
path can also come from `QWEN35_BUNDLE`.

As a library:

```rust
let mut q = taconite_qwen35::Qwen35::load(Path::new("bundle"), 8192)?;
let (text, stats) = q.chat("Why is the sky blue?", &Default::default(), |s| print!("{s}"))?;
```

## Accuracy and speed

`qwen35 check` on a Ryzen AI 9 HX 370, against transformers float32 on the
ROCm iGPU, with the next token teacher-forced on the reference's:

| prompt | tokens | steps | top-1 agrees | flips (reference's top-2 margin) | max \|dlogit\| (reference's top 16) |
| --- | --- | --- | --- | --- | --- |
| the example | 23 | 97 | 93/97 | 4, all ties (<= 0.08) | 0.35 |
| model-card summary | 778 | 110 | 108/110 | 2, all ties (<= 0.13) | 0.52 |
| a kitchen photo (640x427, 1040 patches) + "Describe this image in detail." | 280 | 160 | 154/160 | 6, all ties (<= 0.20) | 1.52 |

With the image, the patches match IRON's Python app bit for bit. The
vision tower's weights are int8 per output channel (`v.weights w8`: exact
bfp16 blocks on the NPU, a per-channel factor applied here), its output
at cosine 0.992 against float32 (~2-2.5 s: NPU ~0.25 s, host attention
~1.7 s). A `bfp16x2` bundle (hi + lo weights, 0.4 GB larger) gets 0.9989
and 157/160.

The tokenizer and chat template reproduce HF's ids on every test string
and both prompts. The free greedy run follows the reference up to its first
tie and writes the same answer as IRON's Python app.

Speed on a Ryzen AI 9 HX 370: setup ~2 s, prefill 0.2 s for 23 tokens
and 1.3 s for 778, decode ~66 ms a token (~15 tokens/s) with the bfp16
bundle and ~55 ms (~18 tokens/s) with the bfp6s one. A step is bound by
the weights it streams from DDR (~2.1 GB in bfp16, ~1.5 GB in bfp6s16).

NPU2 has 16 hardware-context slots shared by every process. When other
programs hold all but one, the prefill and decode contexts take turns
(each reloaded when next needed; `--timing` reports the swaps) rather than
failing.

## License

Apache-2.0, except the image resampler in `src/image.rs` (shared with
taconite-clip), a port of PyTorch's antialiased resize (BSD-3-Clause,
`LICENSE-PYTORCH`) after Pillow's (MIT-CMU, `LICENSE-PILLOW`). The Qwen3.5
weights are Apache-2.0.
