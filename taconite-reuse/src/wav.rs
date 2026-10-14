// SPDX-FileCopyrightText: Copyright (C) 2026 Brishen Hawkins
// SPDX-License-Identifier: Apache-2.0

//! WAV files: PCM 16 / 24 / 32-bit and IEEE float 32 in (as torchaudio
//! scales them: int16 / 32768, ...), 16-bit PCM or float 32 out.

use std::path::Path;

use crate::Error;

/// Samples, channels separate, and the rate.
pub struct Audio {
    pub channels: Vec<Vec<f32>>,
    pub sr: u32,
}

fn u16le(b: &[u8], i: usize) -> u16 {
    u16::from_le_bytes([b[i], b[i + 1]])
}

fn u32le(b: &[u8], i: usize) -> u32 {
    u32::from_le_bytes([b[i], b[i + 1], b[i + 2], b[i + 3]])
}

pub fn read(path: &Path) -> Result<Audio, Error> {
    let b = std::fs::read(path).map_err(|e| Error::Input(format!("{}: {e}", path.display())))?;
    let bad = |m: &str| Error::Input(format!("{}: {m}", path.display()));
    if b.len() < 12 || &b[0..4] != b"RIFF" || &b[8..12] != b"WAVE" {
        return Err(bad("not a RIFF WAVE file"));
    }
    let (mut fmt, mut data) = (None, None);
    let mut i = 12;
    while i + 8 <= b.len() {
        let id = &b[i..i + 4];
        let n = u32le(&b, i + 4) as usize;
        let body = i + 8..(i + 8 + n).min(b.len());
        match id {
            b"fmt " => fmt = Some(body),
            b"data" => data = Some(body),
            _ => {}
        }
        i += 8 + n + (n & 1);
    }
    let fmt = fmt.ok_or_else(|| bad("no fmt chunk"))?;
    let data = data.ok_or_else(|| bad("no data chunk"))?;
    let f = &b[fmt];
    let mut tag = u16le(f, 0);
    let ch = u16le(f, 2) as usize;
    let sr = u32le(f, 4);
    let bits = u16le(f, 14);
    if tag == 0xFFFE && f.len() >= 26 {
        tag = u16le(f, 24); // WAVE_FORMAT_EXTENSIBLE: the sub-format
    }
    let d = &b[data];
    let width = bits as usize / 8;
    if ch == 0 || width == 0 {
        return Err(bad("no channels"));
    }
    let frames = d.len() / (width * ch);
    let mut channels = vec![Vec::with_capacity(frames); ch];
    for fr in 0..frames {
        for (c, out) in channels.iter_mut().enumerate() {
            let o = (fr * ch + c) * width;
            let v = match (tag, bits) {
                (1, 16) => i16::from_le_bytes([d[o], d[o + 1]]) as f32 / 32768.0,
                (1, 24) => ((i32::from_le_bytes([0, d[o], d[o + 1], d[o + 2]]) >> 8) as f32) / 8388608.0,
                (1, 32) => i32::from_le_bytes([d[o], d[o + 1], d[o + 2], d[o + 3]]) as f32 / 2147483648.0,
                (3, 32) => f32::from_le_bytes([d[o], d[o + 1], d[o + 2], d[o + 3]]),
                _ => return Err(bad(&format!("unsupported sample format (tag {tag}, {bits} bits)"))),
            };
            out.push(v);
        }
    }
    Ok(Audio { channels, sr })
}

/// Writes 16-bit PCM (`float` false) or IEEE float 32.
pub fn write(path: &Path, a: &Audio, float: bool) -> Result<(), Error> {
    let ch = a.channels.len();
    let frames = a.channels.first().map_or(0, Vec::len);
    let width = if float { 4 } else { 2 };
    let n = frames * ch * width;
    let mut b = Vec::with_capacity(44 + n);
    b.extend_from_slice(b"RIFF");
    b.extend_from_slice(&((36 + n) as u32).to_le_bytes());
    b.extend_from_slice(b"WAVEfmt ");
    b.extend_from_slice(&16u32.to_le_bytes());
    b.extend_from_slice(&(if float { 3u16 } else { 1u16 }).to_le_bytes());
    b.extend_from_slice(&(ch as u16).to_le_bytes());
    b.extend_from_slice(&a.sr.to_le_bytes());
    b.extend_from_slice(&(a.sr * (ch * width) as u32).to_le_bytes());
    b.extend_from_slice(&((ch * width) as u16).to_le_bytes());
    b.extend_from_slice(&((width * 8) as u16).to_le_bytes());
    b.extend_from_slice(b"data");
    b.extend_from_slice(&(n as u32).to_le_bytes());
    for fr in 0..frames {
        for c in &a.channels {
            let v = c[fr];
            if float {
                b.extend_from_slice(&v.to_le_bytes());
            } else {
                let q = (v.clamp(-1.0, 1.0 - 1.0 / 32768.0) * 32768.0).round() as i16;
                b.extend_from_slice(&q.to_le_bytes());
            }
        }
    }
    std::fs::write(path, b).map_err(|e| Error::Input(format!("{}: {e}", path.display())))
}
