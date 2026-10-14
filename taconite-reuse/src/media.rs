// SPDX-FileCopyrightText: Copyright (C) 2026 Brishen Hawkins
// SPDX-License-Identifier: Apache-2.0

//! Compressed audio in (feature `decode`): MP3, AAC in MP4 / M4A, FLAC and
//! WAV through symphonia, and a resampler for rates the bundle has no
//! kernels for.

use std::f64::consts::PI;
use std::fs::File;
use std::path::Path;

use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::{CODEC_TYPE_NULL, DecoderOptions};
use symphonia::core::errors::Error as SymError;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

use crate::Error;
use crate::wav::Audio;

/// Decodes the first audio track of `path` (the container and codec are
/// probed; the extension is only a hint), channels separate, in [-1, 1].
pub fn read(path: &Path) -> Result<Audio, Error> {
    let bad = |m: String| Error::Input(format!("{}: {m}", path.display()));
    let file = File::open(path).map_err(|e| bad(e.to_string()))?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
        hint.with_extension(ext);
    }
    // gapless: MP3 / AAC encoder delay and padding trimmed, so the output
    // lines up with the original
    let fo = FormatOptions { enable_gapless: true, ..Default::default() };
    let probed = symphonia::default::get_probe()
        .format(&hint, mss, &fo, &MetadataOptions::default())
        .map_err(|e| bad(format!("unrecognised audio ({e})")))?;
    let mut format = probed.format;
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| bad("no audio track".into()))?;
    let id = track.id;
    let mut sr = track.codec_params.sample_rate;
    let mut dec = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(|e| bad(format!("unsupported codec ({e})")))?;
    let mut channels: Vec<Vec<f32>> = Vec::new();
    let mut sb: Option<SampleBuffer<f32>> = None;
    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(SymError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(SymError::ResetRequired) => break,
            Err(e) => return Err(bad(e.to_string())),
        };
        if packet.track_id() != id {
            continue;
        }
        let buf = match dec.decode(&packet) {
            Ok(b) => b,
            // a corrupt frame: skip it, as players do
            Err(SymError::DecodeError(_)) => continue,
            Err(e) => return Err(bad(e.to_string())),
        };
        let spec = *buf.spec();
        sr.get_or_insert(spec.rate);
        let nc = spec.channels.count();
        if channels.is_empty() {
            channels = vec![Vec::new(); nc];
        } else if channels.len() != nc {
            return Err(bad("the channel count changes mid-stream".into()));
        }
        if sb.as_ref().is_none_or(|s| s.capacity() < buf.capacity() * nc) {
            sb = Some(SampleBuffer::new(buf.capacity() as u64, spec));
        }
        let s = sb.as_mut().unwrap();
        s.copy_interleaved_ref(buf);
        for fr in s.samples().chunks_exact(nc) {
            for (c, &v) in channels.iter_mut().zip(fr) {
                c.push(v);
            }
        }
    }
    let sr = sr.ok_or_else(|| bad("no sample rate".into()))?;
    if channels.first().is_none_or(Vec::is_empty) {
        return Err(bad("no audio decoded".into()));
    }
    // symphonia's MP4 reader ignores the encoder delay (AAC's priming
    // frames, 1024 or 2112 samples): trim it, and the padding, as players do
    if probed_mp4(path) {
        if let Some((skip, keep)) = std::fs::read(path).ok().and_then(|b| mp4_trim(&b, sr)) {
            for c in &mut channels {
                let s = skip.min(c.len());
                c.drain(..s);
                if let Some(k) = keep {
                    c.truncate(k);
                }
            }
        }
    }
    Ok(Audio { channels, sr })
}

fn probed_mp4(path: &Path) -> bool {
    let mut h = [0u8; 12];
    File::open(path).and_then(|mut f| std::io::Read::read_exact(&mut f, &mut h)).is_ok() && &h[4..8] == b"ftyp"
}

/// The samples an MP4's audio track starts with that aren't audio (its
/// edit list's media time, else iTunes' iTunSMPB priming), and the count
/// after them that is (the edit's duration / iTunSMPB's), at `sr`.
fn mp4_trim(b: &[u8], sr: u32) -> Option<(usize, Option<usize>)> {
    let be32 = |i: usize| b.get(i..i + 4).map(|s| u32::from_be_bytes(s.try_into().unwrap()) as u64);
    let be64 = |i: usize| b.get(i..i + 8).map(|s| u64::from_be_bytes(s.try_into().unwrap()));
    // the boxes in [lo, hi): (type, body start, end)
    let boxes = |lo: usize, hi: usize| {
        let mut v = Vec::new();
        let mut i = lo;
        while i + 8 <= hi {
            let Some(mut n) = be32(i) else { break };
            let mut body = i + 8;
            if n == 1 {
                n = be64(i + 8).unwrap_or(0);
                body += 8;
            } else if n == 0 {
                n = (hi - i) as u64;
            }
            let end = i.saturating_add(n as usize).min(hi);
            if end <= body.min(hi) && n < 8 {
                break;
            }
            v.push((&b[i + 4..i + 8], body, end));
            i = end;
        }
        v
    };
    let find = |lo: usize, hi: usize, t: &[u8]| boxes(lo, hi).into_iter().find(|x| x.0 == t);
    let (_, ml, mh) = find(0, b.len(), b"moov")?;
    // the movie timescale (mvhd), which the edit durations are in
    let movie_ts = find(ml, mh, b"mvhd").and_then(|(_, l, _)| if b[l] == 1 { be32(l + 20) } else { be32(l + 12) });
    for (t, tl, th) in boxes(ml, mh) {
        if t != b"trak" {
            continue;
        }
        let Some((_, al, ah)) = find(tl, th, b"mdia") else { continue };
        let sound = find(al, ah, b"hdlr").is_some_and(|(_, l, _)| b.get(l + 8..l + 12) == Some(b"soun"));
        if !sound {
            continue;
        }
        let media_ts = find(al, ah, b"mdhd").and_then(|(_, l, _)| if b[l] == 1 { be32(l + 20) } else { be32(l + 12) })?;
        let elst = find(tl, th, b"edts").and_then(|(_, el, eh)| find(el, eh, b"elst"));
        if let Some((_, l, _)) = elst {
            let v1 = b[l] == 1;
            let n = be32(l + 4)?;
            let w = if v1 { 20 } else { 12 };
            for e in 0..n as usize {
                let o = l + 8 + e * w;
                let (dur, mt) = if v1 { (be64(o)?, be64(o + 8)? as i64) } else { (be32(o)?, be32(o + 4)? as u32 as i32 as i64) };
                if mt < 0 {
                    continue; // an empty edit
                }
                let skip = (mt as u128 * sr as u128 / media_ts as u128) as usize;
                let keep = movie_ts.filter(|&m| m > 0 && dur > 0).map(|m| (dur as u128 * sr as u128 / m as u128) as usize);
                return Some((skip, keep));
            }
        }
    }
    // iTunes: "iTunSMPB" = " 00000000 <priming> <padding> <samples> ..."
    let at = b.windows(8).position(|w| w == b"iTunSMPB")?;
    let tail = &b[at..(at + 200).min(b.len())];
    let txt = String::from_utf8_lossy(tail);
    let hex: Vec<u64> =
        txt.split_whitespace().filter_map(|w| u64::from_str_radix(w.trim_matches(char::from(0)), 16).ok()).take(4).collect();
    let (pri, len) = (*hex.get(1)?, *hex.get(3)?);
    Some((pri as usize, Some(len as usize)))
}

/// The channels' mean, as one channel.
pub fn mono(a: &Audio) -> Audio {
    let n = a.channels.len() as f32;
    let len = a.channels[0].len();
    let m = (0..len).map(|i| a.channels.iter().map(|c| c[i]).sum::<f32>() / n).collect();
    Audio { channels: vec![m], sr: a.sr }
}

/// The rate of `rates` to run `sr` at: `sr` itself when it is one, else
/// the lowest above it (no band lost), else the highest.
pub fn nearest_rate(sr: u32, rates: &[u32]) -> Option<u32> {
    if rates.contains(&sr) {
        return Some(sr);
    }
    rates.iter().copied().filter(|&r| r > sr).min().or_else(|| rates.iter().copied().max())
}

/// `x` from `from` Hz to `to` Hz: a Kaiser-windowed sinc (32 zero
/// crossings a side, cut off at 0.95 of the lower Nyquist rate), evaluated
/// at each output instant. Speech-quality, not mastering-grade.
pub fn resample(x: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to || x.is_empty() {
        return x.to_vec();
    }
    const ZC: f64 = 32.0;
    const BETA: f64 = 8.6;
    let ratio = to as f64 / from as f64;
    let cut = 0.95 * ratio.min(1.0); // in input-rate Nyquist units
    let half = ZC / cut; // the kernel's half width, input samples
    let i0 = bessel_i0(BETA);
    let n_out = ((x.len() as f64) * ratio).round() as usize;
    (0..n_out)
        .map(|j| {
            let t = j as f64 / ratio;
            let lo = (t - half).ceil().max(0.0) as usize;
            let hi = ((t + half).floor() as usize).min(x.len() - 1);
            let mut acc = 0.0;
            for (i, &v) in x.iter().enumerate().take(hi + 1).skip(lo) {
                let d = i as f64 - t;
                let w = 1.0 - (d / half).powi(2);
                if w <= 0.0 {
                    continue;
                }
                let a = PI * cut * d;
                let sinc = if a.abs() < 1e-9 { 1.0 } else { a.sin() / a };
                acc += v as f64 * cut * sinc * bessel_i0(BETA * w.sqrt()) / i0;
            }
            acc as f32
        })
        .collect()
}

fn bessel_i0(x: f64) -> f64 {
    let (mut sum, mut term, q) = (1.0, 1.0, x * x / 4.0);
    for k in 1..64 {
        term *= q / (k * k) as f64;
        sum += term;
        if term < sum * 1e-16 {
            break;
        }
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resample_keeps_a_tone() {
        let (from, to, f) = (11025u32, 16000u32, 440.0);
        let x: Vec<f32> = (0..11025).map(|i| (2.0 * PI * f * i as f64 / from as f64).sin() as f32).collect();
        let y = resample(&x, from, to);
        assert_eq!(y.len(), 16000);
        // away from the edges, the tone at the new rate
        let err = (2000..14000)
            .map(|i| (y[i] as f64 - (2.0 * PI * f * i as f64 / to as f64).sin()).abs())
            .fold(0.0, f64::max);
        assert!(err < 2e-3, "{err}");
    }

    #[test]
    fn rates() {
        let r = [8000, 16000, 22050, 24000, 32000, 44100, 48000];
        assert_eq!(nearest_rate(44100, &r), Some(44100));
        assert_eq!(nearest_rate(11025, &r), Some(16000));
        assert_eq!(nearest_rate(96000, &r), Some(48000));
    }
}
