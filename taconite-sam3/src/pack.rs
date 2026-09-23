// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-FileCopyrightText: Copyright (C) 2026 Brishen Hawkins
// SPDX-License-Identifier: Apache-2.0

//! `flm.GEMM`'s B packing (`iron/operators/flm/packing.py`), for the weights
//! built per prompt (the folded cross-attentions and the mask head): a
//! row-major `[K, N]` matrix reordered into the order the compute tiles read
//! it and quantized to bfp16ebs8 (8 values along k share an exponent, 9
//! bytes a block), optionally with a per-column bias chunk in front of each
//! column-block. Checked byte for byte against the Python packer by
//! `sam3 check` (`pack_test.*`).

use crate::bundle::GemmSpec;

/// The mmul's register tiling (`S`, `T` in `flm/gemm/design.py`).
const S: usize = 8;
const T: usize = 8;

/// 8 f32 -> one bfp16ebs8 block, rounding to nearest-even like the core's
/// `conv_even` mode (see `f32_to_bfp16ebs8`).
fn block(v: &[f32; 8], out: &mut [u8]) {
    let mut exp = [0i64; 8];
    let mut mag = [0i64; 8];
    let mut neg = [false; 8];
    for (i, x) in v.iter().enumerate() {
        let u = x.to_bits();
        neg[i] = u & 0x8000_0000 != 0;
        exp[i] = ((u >> 23) & 0xff) as i64;
        let mut man = (u & 0x007f_ffff) as i64;
        if exp[i] != 0 {
            man |= 0x0080_0000;
        }
        mag[i] = if neg[i] { -man } else { man };
    }
    let max_exp = *exp.iter().max().unwrap();
    out[0] = max_exp as u8;
    for i in 0..8 {
        let shift = max_exp - exp[i];
        let v8 = if shift >= 32 {
            if neg[i] { -1 } else { 0 }
        } else {
            // round-half-to-even of mag / 2^total; total >= 17
            let total = (17 + shift).clamp(0, 62) as u32;
            let q = mag[i] >> total; // floor
            let rem = mag[i] - (q << total);
            let half = 1i64 << (total - 1);
            if rem > half || (rem == half && q & 1 == 1) { q + 1 } else { q }
        };
        out[1 + i] = v8.clamp(-128, 127) as i8 as u8;
    }
}

/// `pack_b(B, k_tile, n_tile, 8, 8, ct_k, bfp16=True, bias=bias)` for
/// `spec`'s tiling. `b` is row-major `[K, N]`.
pub fn pack_b(spec: &GemmSpec, b: &[f32], bias: Option<&[f32]>) -> Vec<u8> {
    let (k, n) = (spec.k, spec.n);
    let (k_tile, n_tile, ct_k) = (spec.tile_k, spec.tile_n, spec.ct_k);
    assert_eq!(b.len(), k * n, "B is not [{k}, {n}]");
    assert!(k % k_tile == 0 && n % n_tile == 0 && k_tile % ct_k == 0 && ct_k % S == 0 && n_tile % T == 0);
    let col_a = ct_k / S;
    let n_blocks = n / n_tile;
    let per_block = k * n_tile / 8 * 9;
    let chunk = if bias.is_some() { ct_k * n_tile / 8 * 9 } else { 0 };
    let mut out = vec![0u8; n_blocks * (chunk + per_block)];
    let mut vals = [0f32; 8];
    for cb in 0..n_blocks {
        let base = cb * (chunk + per_block);
        if let Some(bias) = bias {
            for j in 0..n_tile {
                let h = taconite::f32_to_bf16(bias[cb * n_tile + j]).to_le_bytes();
                out[base + 2 * j..base + 2 * j + 2].copy_from_slice(&h);
            }
        }
        let mut o = base + chunk;
        // (cb, kb, kslice, tb, i, t_in, s_in): 8 consecutive k for one n
        for kb in 0..k / k_tile {
            for ks in 0..k_tile / ct_k {
                for tb in 0..n_tile / T {
                    for i in 0..col_a {
                        let k0 = kb * k_tile + ks * ct_k + i * S;
                        for t_in in 0..T {
                            let col = cb * n_tile + tb * T + t_in;
                            for (s_in, v) in vals.iter_mut().enumerate() {
                                *v = b[(k0 + s_in) * n + col];
                            }
                            block(&vals, &mut out[o..o + 9]);
                            o += 9;
                        }
                    }
                }
            }
        }
    }
    debug_assert_eq!(out.len(), spec.b_bytes);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rounds_half_to_even_on_the_shared_exponent() {
        // 14.9375 -> 15 (rounds up); ties go to even (106.5 -> 106, 94.5 -> 94),
        // measured on hardware (see packing.py). The block's largest value,
        // 106.5 (exponent 133), puts every mantissa on a grid of 1.
        let v = [64.0, 14.9375, 106.5, 94.5, 0.0, -1.5, -2.5, 3.0];
        let mut out = [0u8; 9];
        block(&v, &mut out);
        let m: Vec<i8> = out[1..].iter().map(|&b| b as i8).collect();
        assert_eq!(out[0], 133);
        assert_eq!(m, vec![64, 15, 106, 94, 0, -2, -2, 3]);
    }
}
