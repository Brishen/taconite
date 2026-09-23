// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The CLIP text encoder (24 pre-LN layers, width 1024, causal attention)
//! and SAM3's projection to the DETR width, on the host.
//!
//! Only the valid tokens are encoded: attention is causal and the padding
//! follows them, so they never see it, and every consumer of the text
//! features masks the padded positions out (the DETR cross-attentions, the
//! mask decoder's, the scoring's mean pool). Their rows are left zero. A
//! short prompt is a handful of tokens, so this is weight-bandwidth bound
//! (the weights ship as bf16).

use crate::cpu::{self, Attn, Rows, W};
use crate::{Error, Sam3};

/// An encoded prompt: `feats [L, 256]` (L = 32), `valid [L]`.
pub struct Text {
    pub feats: Vec<f32>,
    pub valid: Vec<bool>,
}

impl Sam3 {
    pub fn text(&mut self, ids: &[u32], mask: &[u32]) -> Result<Text, Error> {
        let st = &self.store;
        let c = &self.cfg;
        let (dim, l) = (c.text_dim, c.text_len);
        let n = mask.iter().take_while(|&&m| m != 0).count();
        if n == 0 || mask[n..].iter().any(|&m| m != 0) || ids.len() != l {
            return Err(Error::Input("the attention mask must be a non-empty prefix".into()));
        }
        let tok = st.bf16("t.tok_emb")?;
        let pos = st.f32("t.pos_emb")?;
        let mut x = vec![0f32; n * dim];
        for (t, row) in x.chunks_mut(dim).enumerate() {
            let e = &tok[ids[t] as usize * dim..(ids[t] as usize + 1) * dim];
            for (j, v) in row.iter_mut().enumerate() {
                *v = taconite::bf16_to_f32(e[j]) + pos[t * dim + j];
            }
        }
        let eps = c.text_eps;
        for i in 0..c.text_layers {
            let p = |s: &str| format!("t.{i}.{s}");
            let h = cpu::layer_norm(&x, dim, st.f32(&p("ln1.w"))?, st.f32(&p("ln1.b"))?, eps);
            let qkv = cpu::linear(&h, dim, W::Bf16(st.bf16(&p("qkv.w"))?), Some(st.f32(&p("qkv.b"))?));
            let q: Vec<f32> = qkv.chunks(3 * dim).flat_map(|r| r[..dim].iter().copied()).collect();
            let a = Attn { heads: c.text_heads, causal: true, ..Default::default() };
            let o = cpu::attention(
                &q,
                dim,
                Rows::strided(&qkv, n, 3 * dim, dim),
                Rows::strided(&qkv, n, 3 * dim, 2 * dim),
                &a,
            );
            let o = cpu::linear(&o, dim, W::Bf16(st.bf16(&p("o.w"))?), Some(st.f32(&p("o.b"))?));
            cpu::add_(&mut x, &o);
            let h = cpu::layer_norm(&x, dim, st.f32(&p("ln2.w"))?, st.f32(&p("ln2.b"))?, eps);
            let mut f = cpu::linear(&h, dim, W::Bf16(st.bf16(&p("fc1.w"))?), Some(st.f32(&p("fc1.b"))?));
            for v in f.iter_mut() {
                *v = cpu::gelu(*v);
            }
            let f = cpu::linear(&f, c.text_ffn, W::Bf16(st.bf16(&p("fc2.w"))?), Some(st.f32(&p("fc2.b"))?));
            cpu::add_(&mut x, &f);
        }
        let x = cpu::layer_norm(&x, dim, st.f32("t.final_ln.w")?, st.f32("t.final_ln.b")?, eps);
        let y = cpu::linear(&x, dim, W::F32(st.f32("t.proj.w")?), Some(st.f32("t.proj.b")?));
        let d = c.d_model;
        let mut feats = vec![0f32; l * d];
        feats[..n * d].copy_from_slice(&y);
        Ok(Text { feats, valid: mask.iter().map(|&m| m != 0).collect() })
    }
}
