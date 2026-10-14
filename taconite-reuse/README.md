<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Brishen Hawkins
SPDX-License-Identifier: Apache-2.0
-->

# `taconite-reuse` — RE-USE speech enhancement on the NPU, from Rust

A Rust runtime for [`nvidia/RE-USE`](https://huggingface.co/nvidia/RE-USE),
NVIDIA's universal speech enhancement model (SEMamba: a dense
convolutional encoder, 30 bidirectional time-frequency Mamba blocks, two
decoders): noisy, reverberant, clipped, band-limited or codec-damaged
speech at 8, 16, 22.05, 24, 32, 44.1 or 48 kHz in, enhanced speech at the
same rate out, as the checkpoint's `inference.py` computes it. Every
convolution, projection and Mamba scan runs on the AMD XDNA NPU (NPU2)
through IRON kernels replayed with [`taconite`](https://crates.io/crates/taconite)
— through XRT, or directly through the amdxdna driver (`--features
direct`). It is the forward of [IRON](https://github.com/amd/IRON)'s Python
app (`iron/applications/reuse`), kernel for kernel.

Download the bundle (~55 MB: seven xclbins, their instruction streams,
the packed weights, reference cases) and install:

```bash
hf download brishen/iron-reuse-npu2 --local-dir reuse-bundle
cargo install taconite-reuse --features decode,encode           # over XRT
# or, with no XRT at all (straight to the amdxdna driver):
cargo install taconite-reuse --no-default-features --features cli,direct,decode,encode
reuse check reuse-bundle                                         # the reference cases against the float32 network
```

```bash
reuse enhance <bundle> noisy.wav enhanced.wav   # every channel, same rate, 16-bit PCM (--float: f32)
```

`reuse-clean` (feature `decode`) takes compressed audio too -- MP3, M4A /
AAC, FLAC or WAV, decoded with [symphonia](https://crates.io/crates/symphonia)
-- and writes the cleaned audio back in the input's format, rate and
channel count (with feature `encode`; WAV without it, and for FLAC):

```bash
export REUSE_BUNDLE=$PWD/reuse-bundle
reuse-clean interview.m4a                    # -> interview.clean.m4a (AAC, the input's bitrate)
reuse-clean memo.mp3 --mono --rate 16000     # -> memo.clean.mp3
reuse-clean memo.mp3 -o memo.wav             # the extension picks the format
```

MP3 goes out through LAME and M4A as AAC-LC through Fraunhofer's FDK AAC,
at the input's bitrate (`--bitrate` overrides), gapless: the encoder delay
is recorded (the LAME tag; the M4A's edit list), so the file decodes to
exactly the cleaned samples. Rates the bundle has no kernels for are
resampled to the nearest one above and back; `--rate 16000` runs ~2.5x
faster than 44.1 kHz (speech up to 8 kHz kept). A stereo file of identical
channels is enhanced once (`--mono`: the channels' mean). On a terminal
it draws a progress bar with an estimate of the time left
(`Reuse::enhance_with` reports the fraction done as it goes). The input's
encoder delay is trimmed (MP3's LAME tag, MP4's edit list or iTunSMPB), so
the output lines up with it.

```rust
let mut m = taconite_reuse::Reuse::load(Path::new("bundle"))?;
let enhanced = m.enhance(&samples, 16000)?; // one channel, f32 in [-1, 1]
```

Any length: up to a little over 5 s in one pass (`inference.py`), longer
audio in 5 s chunks at a 2.5 s hop overlap-added under a Hann window
(`inference_chunk.py`). The bundle is written by IRON's
`iron/applications/reuse/export_reuse.py` (`--rates` to export fewer
sample rates).

## What runs where

| | NPU | host (this crate) |
|---|---|---|
| STFT (n_fft 320 at 8 kHz, scaled with the rate) | | `stft.rs` and its inverse (f64), muting mostly-zero frames |
| encoder: 1 x 1 conv, a dense block of four dilated 3 x 3 convs, a strided (1, 3) conv | the 3 x 3 and strided convs (`flm.GEMM`s) | the 1 x 1 conv, the map layouts, the dilated taps' sums, InstanceNorm + PReLU |
| 30 x (time Mamba, frequency Mamba), both directions | `MambaScan`'s phases: in_proj, causal conv1d + x_proj, the selective scan with the output projection | token layouts (the backward direction time-reversed), both directions summed, LayerNorm, residual |
| decoders: a dense block, two sub-pixel upsampling convs | every conv (`flm.GEMM`s) | InstanceNorm + PReLU, the 1 x 1 heads, atan2 |

Seven hardware contexts of NPU2's 16 (shared across processes). When the
device has no free slot, the least recently used context is dropped and
reloaded on its next use (`Npu::swaps` counts these).

## Accuracy

Two float32 runs of this model agree to ~43 dB of waveform SNR (the
network reads the phase of near-silent STFT bins, which 1e-7 of input
noise moves). `reuse check` against the float32 network on a GPU, the
checkpoint's example clip at five rates (8 to 48 kHz): 35-46 dB of
waveform SNR, on a Ryzen AI 9 HX 370 in 1.2 s (1 s at 22.05 kHz) to 7.4
s (5.1 s at 44.1 kHz).

## Building

The default `xrt` feature needs XRT's headers and library; `direct` needs
neither: `cargo build --release --no-default-features --features cli,direct`
(add `decode` for `reuse-clean`, and `encode` for its MP3 / M4A output:
`--features cli,direct,decode,encode`). `encode` builds LAME (LGPL-2.0)
and FDK AAC (Fraunhofer's license) from their vendored sources and links
them statically, so a binary built with it carries their terms too.

`embed` builds a bundle into the binaries (~55 MB), so they need no
`--bundle` / `$REUSE_BUNDLE`; it is unpacked to
`$XDG_CACHE_HOME/taconite-reuse/bundle-<hash>` (else `~/.cache/...`) on the
first run, and `reuse check` with no argument checks it:

```bash
REUSE_EMBED_BUNDLE=$PWD/reuse-bundle \
  cargo build --release --no-default-features --features cli,direct,decode,encode,embed
```

(an absolute path; `Reuse::load_embedded` from the library). The binary
then carries the RE-USE weights, and their license.

The model weights (in the bundle, with the license as its `LICENSE.md`)
are NVIDIA's, under the
[NVIDIA One-Way Noncommercial License (NSCLv1)](https://github.com/NVlabs/HMAR/blob/main/LICENSE):
non-commercial research and educational use only. This crate (which only
loads a bundle) is Apache-2.0.
