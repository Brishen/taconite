// SPDX-FileCopyrightText: Copyright (C) 2026 Brishen Hawkins
// SPDX-License-Identifier: Apache-2.0

//! The model's STFT and its inverse, as `torch.stft` / `torch.istft`
//! compute them in the checkpoint's inference.py: a periodic Hann window
//! of the FFT size, centred frames (the signal reflect-padded by n_fft /
//! 2), one-sided spectra; the magnitude compressed by log1p. Transforms in
//! f64.

use std::sync::Arc;

use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};
use taconite::cpu::par_rows;

/// One rate's transform: n_fft (= the window) and hop.
pub struct Stft {
    pub n_fft: usize,
    pub hop: usize,
    window: Vec<f64>,
    fwd: Arc<dyn Fft<f64>>,
    inv: Arc<dyn Fft<f64>>,
}

impl Stft {
    pub fn new(n_fft: usize, hop: usize, win: usize) -> Self {
        assert_eq!(n_fft, win, "the model's window is its FFT size");
        let window = (0..win).map(|n| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / win as f64).cos()).collect();
        let mut p = FftPlanner::new();
        Stft { n_fft, hop, window, fwd: p.plan_fft_forward(n_fft), inv: p.plan_fft_inverse(n_fft) }
    }

    pub fn bins(&self) -> usize {
        self.n_fft / 2 + 1
    }

    /// Frames of a signal of `len` samples.
    pub fn frames(&self, len: usize) -> usize {
        1 + len / self.hop
    }

    /// wav -> (log1p magnitude, phase), each [F, T] (bin major).
    pub fn forward(&self, wav: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let (n, h) = (self.n_fft, self.n_fft / 2);
        let len = wav.len();
        assert!(len > h, "a signal shorter than half a window ({h} samples) cannot be reflect-padded");
        // reflect padding (the edge sample not repeated)
        let at = |i: isize| -> f64 {
            let l = len as isize;
            let mut j = i;
            if j < 0 {
                j = -j;
            }
            if j >= l {
                j = 2 * (l - 1) - j;
            }
            wav[j as usize] as f64
        };
        let t = self.frames(len);
        let f = self.bins();
        // frame-major spectra, then transposed
        let mut spec = vec![0f64; t * 2 * f];
        par_rows(&mut spec, 2 * f, |t0, rows| {
            let mut buf = vec![Complex::new(0.0, 0.0); n];
            let mut scratch = vec![Complex::new(0.0, 0.0); self.fwd.get_inplace_scratch_len()];
            for (ti, row) in rows.chunks_mut(2 * f).enumerate() {
                let s0 = ((t0 + ti) * self.hop) as isize - h as isize;
                for (k, b) in buf.iter_mut().enumerate() {
                    *b = Complex::new(at(s0 + k as isize) * self.window[k], 0.0);
                }
                self.fwd.process_with_scratch(&mut buf, &mut scratch);
                for k in 0..f {
                    row[2 * k] = buf[k].re;
                    row[2 * k + 1] = buf[k].im;
                }
            }
        });
        let mut mag = vec![0f32; f * t];
        let mut pha = vec![0f32; f * t];
        for ti in 0..t {
            for k in 0..f {
                let (re, im) = (spec[(ti * f + k) * 2], spec[(ti * f + k) * 2 + 1]);
                mag[k * t + ti] = (re.hypot(im)).ln_1p() as f32;
                pha[k * t + ti] = im.atan2(re) as f32;
            }
        }
        (mag, pha)
    }

    /// (linear magnitude, phase) [F, T] -> the signal, `hop (T - 1)`
    /// samples (torch.istft, center=True, no length).
    pub fn inverse(&self, mag: &[f32], pha: &[f32], t: usize) -> Vec<f32> {
        let (n, h) = (self.n_fft, self.n_fft / 2);
        let f = self.bins();
        // each frame's windowed inverse transform
        let mut frames = vec![0f64; t * n];
        par_rows(&mut frames, n, |t0, rows| {
            let mut buf = vec![Complex::new(0.0, 0.0); n];
            let mut scratch = vec![Complex::new(0.0, 0.0); self.inv.get_inplace_scratch_len()];
            for (ti, row) in rows.chunks_mut(n).enumerate() {
                let tt = t0 + ti;
                for k in 0..f {
                    let (m, p) = (mag[k * t + tt] as f64, pha[k * t + tt] as f64);
                    buf[k] = Complex::new(m * p.cos(), m * p.sin());
                }
                // a one-sided spectrum: DC and Nyquist real (irfft)
                buf[0].im = 0.0;
                if n % 2 == 0 {
                    buf[n / 2].im = 0.0;
                }
                for k in f..n {
                    buf[k] = buf[n - k].conj();
                }
                self.inv.process_with_scratch(&mut buf, &mut scratch);
                for (k, o) in row.iter_mut().enumerate() {
                    *o = buf[k].re / n as f64 * self.window[k];
                }
            }
        });
        let total = n + self.hop * (t - 1);
        let mut y = vec![0f64; total];
        let mut env = vec![0f64; total];
        for tt in 0..t {
            let s0 = tt * self.hop;
            for k in 0..n {
                y[s0 + k] += frames[tt * n + k];
                env[s0 + k] += self.window[k] * self.window[k];
            }
        }
        (h..h + self.hop * (t - 1)).map(|i| (y[i] / env[i]) as f32).collect()
    }
}
