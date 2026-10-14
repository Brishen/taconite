// SPDX-FileCopyrightText: Copyright (C) 2026 Brishen Hawkins
// SPDX-License-Identifier: Apache-2.0

//! Compressed audio out (feature `encode`): MP3 through LAME and M4A
//! (AAC-LC in MP4) through Fraunhofer's FDK AAC, both gapless -- the
//! encoder delay and padding are recorded (MP3's LAME tag, the M4A's edit
//! list), so a decoder that honours them (ffmpeg, symphonia + `media`,
//! browsers) returns exactly the samples written.
//!
//! LAME is LGPL-2.0 and FDK AAC is under Fraunhofer's own license; both are
//! linked statically from their vendored sources.

use std::path::Path;

use crate::Error;
use crate::wav::Audio;

fn io(path: &Path) -> impl Fn(std::io::Error) -> Error + '_ {
    move |e| Error::Input(format!("{}: {e}", path.display()))
}

fn check_channels(a: &Audio, what: &str) -> Result<usize, Error> {
    match a.channels.len() {
        n @ (1 | 2) => Ok(n),
        n => Err(Error::Input(format!("{what} out takes 1 or 2 channels, not {n}"))),
    }
}

/// MP3, constant `kbps` (LAME picks the nearest legal rate for the sample
/// rate), joint stereo for two channels, with a LAME / Info tag.
pub fn write_mp3(path: &Path, a: &Audio, kbps: u32) -> Result<(), Error> {
    use mp3lame_sys as l;
    let nc = check_channels(a, "MP3")?;
    let n = a.channels[0].len();
    let fail = |m: &str| Error::Input(format!("{}: LAME: {m}", path.display()));
    struct Lame(l::lame_t);
    impl Drop for Lame {
        fn drop(&mut self) {
            unsafe { l::lame_close(self.0) };
        }
    }
    let g = Lame(unsafe { l::lame_init() });
    if g.0.is_null() {
        return Err(fail("init"));
    }
    let mut out = Vec::with_capacity(n * kbps as usize / 8 / a.sr.max(1) as usize * 1000 + 8192);
    unsafe {
        l::lame_set_num_channels(g.0, nc as i32);
        l::lame_set_in_samplerate(g.0, a.sr as i32);
        l::lame_set_out_samplerate(g.0, a.sr as i32);
        l::lame_set_mode(g.0, if nc == 1 { l::MPEG_mode::MONO } else { l::MPEG_mode::JOINT_STEREO });
        l::lame_set_VBR(g.0, l::vbr_mode::vbr_off);
        l::lame_set_brate(g.0, kbps as i32);
        l::lame_set_quality(g.0, 2); // near best, still fast
        l::lame_set_bWriteVbrTag(g.0, 1);
        if l::lame_init_params(g.0) < 0 {
            return Err(fail("unsupported parameters"));
        }
        const CH: usize = 8192;
        let mut buf = vec![0u8; CH * 5 / 4 + 7200];
        let (left, right) = (&a.channels[0], &a.channels[nc - 1]);
        for s in (0..n).step_by(CH) {
            let e = (s + CH).min(n);
            let k = l::lame_encode_buffer_ieee_float(
                g.0,
                left[s..e].as_ptr(),
                right[s..e].as_ptr(),
                (e - s) as i32,
                buf.as_mut_ptr(),
                buf.len() as i32,
            );
            if k < 0 {
                return Err(fail("encode"));
            }
            out.extend_from_slice(&buf[..k as usize]);
        }
        let k = l::lame_encode_flush(g.0, buf.as_mut_ptr(), buf.len() as i32);
        if k < 0 {
            return Err(fail("flush"));
        }
        out.extend_from_slice(&buf[..k as usize]);
        // the first frame was the tag's placeholder: now it can say how
        // many samples there are, and the delay and padding around them
        let k = l::lame_get_lametag_frame(g.0, buf.as_mut_ptr(), buf.len());
        if k > 0 && k <= out.len() {
            out[..k].copy_from_slice(&buf[..k]);
        }
    }
    std::fs::write(path, out).map_err(io(path))
}

/// M4A: AAC-LC at constant `kbps` in an MP4 whose edit list skips the
/// encoder's delay and ends at the last sample written.
pub fn write_m4a(path: &Path, a: &Audio, kbps: u32) -> Result<(), Error> {
    use fdk_aac::enc::{AudioObjectType, BitRate, ChannelMode, Encoder, EncoderParams, Transport};
    let nc = check_channels(a, "M4A")?;
    let n = a.channels[0].len();
    let fail = |e: fdk_aac::enc::EncoderError| Error::Input(format!("{}: FDK AAC: {e}", path.display()));
    let enc = Encoder::new(EncoderParams {
        bit_rate: BitRate::Cbr(kbps * 1000),
        sample_rate: a.sr,
        transport: Transport::Raw,
        channels: if nc == 1 { ChannelMode::Mono } else { ChannelMode::Stereo },
        audio_object_type: AudioObjectType::Mpeg4LowComplexity,
    })
    .map_err(fail)?;
    let info = enc.info().map_err(fail)?;
    let frame = info.frameLength as usize;
    let delay = info.nDelay as usize;
    let asc = info.confBuf[..info.confSize as usize].to_vec();
    // interleaved i16, then silence past the end so the delayed tail comes
    // out (the wrapper can't signal end of stream; the edit list trims it)
    let total = n + delay + frame;
    let mut pcm = Vec::with_capacity(total.div_ceil(frame) * frame * nc);
    for i in 0..total.div_ceil(frame) * frame {
        for c in &a.channels {
            let v = c.get(i).copied().unwrap_or(0.0);
            pcm.push((v.clamp(-1.0, 1.0) * 32767.0).round() as i16);
        }
    }
    let mut units: Vec<Vec<u8>> = Vec::new();
    let mut buf = vec![0u8; (info.maxOutBufBytes as usize).max(8192)];
    let need = (n + delay).div_ceil(frame); // access units covering delay + audio
    let mut at = 0;
    while units.len() < need && at < pcm.len() {
        let r = enc.encode(&pcm[at..(at + frame * nc).min(pcm.len())], &mut buf).map_err(fail)?;
        at += r.input_consumed;
        if r.output_size > 0 {
            units.push(buf[..r.output_size].to_vec());
        }
        if r.input_consumed == 0 && r.output_size == 0 {
            break;
        }
    }
    if units.len() < need {
        return Err(Error::Input(format!("{}: FDK AAC: {} of {need} frames", path.display(), units.len())));
    }
    let avg = (units.iter().map(Vec::len).sum::<usize>() as u64 * 8 * a.sr as u64 / (units.len() * frame) as u64) as u32;
    let max_unit = units.iter().map(Vec::len).max().unwrap_or(0) as u32;
    let mp4 = mp4_aac(&units, &asc, a.sr, nc as u16, frame as u32, delay as u32, n as u32, avg, max_unit);
    std::fs::write(path, mp4).map_err(io(path))
}

/// A box: size, type, body.
fn bx(t: &[u8; 4], body: &[&[u8]]) -> Vec<u8> {
    let n: usize = 8 + body.iter().map(|b| b.len()).sum::<usize>();
    let mut v = Vec::with_capacity(n);
    v.extend_from_slice(&(n as u32).to_be_bytes());
    v.extend_from_slice(t);
    for b in body {
        v.extend_from_slice(b);
    }
    v
}

/// A full box: version 0, `flags`.
fn fbx(t: &[u8; 4], flags: u32, body: &[&[u8]]) -> Vec<u8> {
    let vf = flags.to_be_bytes();
    let mut parts: Vec<&[u8]> = vec![&vf];
    parts.extend_from_slice(body);
    bx(t, &parts)
}

/// An MPEG-4 descriptor: tag, a one-byte length (every one here is short).
fn desc(tag: u8, body: &[u8]) -> Vec<u8> {
    assert!(body.len() < 128);
    let mut v = vec![tag, body.len() as u8];
    v.extend_from_slice(body);
    v
}

const MATRIX: [u32; 9] = [0x10000, 0, 0, 0, 0x10000, 0, 0, 0, 0x40000000];

#[allow(clippy::too_many_arguments)]
fn mp4_aac(units: &[Vec<u8>], asc: &[u8], sr: u32, nc: u16, frame: u32, delay: u32, n: u32, avg_bps: u32, max_unit: u32) -> Vec<u8> {
    let be = |v: u32| v.to_be_bytes();
    let matrix: Vec<u8> = MATRIX.iter().flat_map(|v| v.to_be_bytes()).collect();
    let ftyp = bx(b"ftyp", &[b"M4A ", &be(0), b"M4A ", b"mp42", b"isom"]);
    let data: usize = units.iter().map(Vec::len).sum();
    let mdat_body_at = ftyp.len() + 8;
    let media_dur = units.len() as u32 * frame;
    // the movie and the track at the sample rate: their duration is what
    // plays, `n` samples
    let mvhd = fbx(
        b"mvhd",
        0,
        &[&be(0), &be(0), &be(sr), &be(n), &be(0x10000), &[0x01, 0x00], &[0; 10], &matrix, &[0; 24], &be(2)],
    );
    let tkhd = fbx(
        b"tkhd",
        7,
        &[&be(0), &be(0), &be(1), &be(0), &be(n), &[0; 8], &[0; 2], &[0; 2], &[0x01, 0x00], &[0; 2], &matrix, &be(0), &be(0)],
    );
    let elst = fbx(b"elst", 0, &[&be(1), &be(n), &be(delay), &be(0x10000)]);
    let edts = bx(b"edts", &[&elst]);
    let mdhd = fbx(b"mdhd", 0, &[&be(0), &be(0), &be(sr), &be(media_dur), &[0x55, 0xC4], &[0; 2]]);
    let hdlr = fbx(b"hdlr", 0, &[&be(0), b"soun", &[0; 12], b"SoundHandler\0"]);
    let smhd = fbx(b"smhd", 0, &[&[0; 4]]);
    let url = fbx(b"url ", 1, &[]);
    let dref = fbx(b"dref", 0, &[&be(1), &url]);
    let dinf = bx(b"dinf", &[&dref]);
    let dsi = desc(0x05, asc);
    let mut dcd = vec![0x40, 0x15];
    dcd.extend_from_slice(&max_unit.to_be_bytes()[1..]); // bufferSizeDB, 24 bits
    dcd.extend_from_slice(&be(avg_bps.max(max_unit * 8 * sr / frame)));
    dcd.extend_from_slice(&be(avg_bps));
    dcd.extend_from_slice(&dsi);
    let mut es = vec![0x00, 0x01, 0x00]; // ES_ID 1, no flags
    es.extend_from_slice(&desc(0x04, &dcd));
    es.extend_from_slice(&desc(0x06, &[0x02]));
    let esds = fbx(b"esds", 0, &[&desc(0x03, &es)]);
    let sr_fixed = if sr < 65536 { sr << 16 } else { 0 };
    let mp4a = bx(b"mp4a", &[&[0; 6], &[0, 1], &[0; 8], &nc.to_be_bytes(), &16u16.to_be_bytes(), &[0; 4], &be(sr_fixed), &esds]);
    let stsd = fbx(b"stsd", 0, &[&be(1), &mp4a]);
    let stts = fbx(b"stts", 0, &[&be(1), &be(units.len() as u32), &be(frame)]);
    let stsc = fbx(b"stsc", 0, &[&be(1), &be(1), &be(units.len() as u32), &be(1)]);
    let sizes: Vec<u8> = units.iter().flat_map(|u| (u.len() as u32).to_be_bytes()).collect();
    let stsz = fbx(b"stsz", 0, &[&be(0), &be(units.len() as u32), &sizes]);
    let stco = fbx(b"stco", 0, &[&be(1), &be(mdat_body_at as u32)]);
    let stbl = bx(b"stbl", &[&stsd, &stts, &stsc, &stsz, &stco]);
    let minf = bx(b"minf", &[&smhd, &dinf, &stbl]);
    let mdia = bx(b"mdia", &[&mdhd, &hdlr, &minf]);
    let trak = bx(b"trak", &[&tkhd, &edts, &mdia]);
    let moov = bx(b"moov", &[&mvhd, &trak]);
    let mut v = ftyp;
    v.extend_from_slice(&((8 + data) as u32).to_be_bytes());
    v.extend_from_slice(b"mdat");
    for u in units {
        v.extend_from_slice(u);
    }
    v.extend_from_slice(&moov);
    v
}

#[cfg(all(test, feature = "decode"))]
mod tests {
    use super::*;

    /// A tone and a sweep, `n` samples at `sr`.
    fn signal(n: usize, sr: u32) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let t = i as f64 / sr as f64;
                (0.3 * (2.0 * std::f64::consts::PI * 440.0 * t).sin()
                    + 0.2 * (2.0 * std::f64::consts::PI * (300.0 + 900.0 * t) * t).sin()) as f32
            })
            .collect()
    }

    /// The lag of `x` behind `y` (the best of -4096 ..= 4096, by
    /// correlation) and the SNR once aligned.
    fn lag_snr(x: &[f32], y: &[f32]) -> (i64, f64) {
        let at = |l: i64, i: usize| x.get((i as i64 + l) as usize).copied();
        let corr = |l: i64| (0..y.len()).filter_map(|i| at(l, i).map(|u| u as f64 * y[i] as f64)).sum::<f64>();
        let best = (-4096i64..=4096).step_by(1).max_by(|&a, &b| corr(a).total_cmp(&corr(b))).unwrap();
        let (mut s, mut e) = (0.0, 0.0);
        for (i, &v) in y.iter().enumerate() {
            if let Some(u) = at(best, i) {
                s += (v as f64).powi(2);
                e += (u as f64 - v as f64).powi(2);
            }
        }
        (best, 10.0 * (s / e).log10())
    }

    fn round_trip(ext: &str, sr: u32, nc: usize) {
        let n = sr as usize * 3 / 2 + 123; // not a whole number of frames
        let x = signal(n, sr);
        let a = Audio { channels: vec![x.clone(); nc], sr };
        let p = std::env::temp_dir().join(format!("taconite-reuse-rt-{}-{sr}-{nc}.{ext}", std::process::id()));
        match ext {
            "mp3" => write_mp3(&p, &a, 128).unwrap(),
            _ => write_m4a(&p, &a, 128).unwrap(),
        }
        let b = crate::media::read(&p).unwrap();
        std::fs::remove_file(&p).ok();
        assert_eq!((b.sr, b.channels.len()), (sr, nc));
        assert_eq!(b.channels[0].len(), n, "{ext} at {sr}: length");
        let (lag, snr) = lag_snr(&b.channels[0], &x);
        assert_eq!(lag, 0, "{ext} at {sr}: lag");
        assert!(snr > 15.0, "{ext} at {sr}: {snr:.1} dB");
    }

    #[test]
    fn mp3_round_trip() {
        round_trip("mp3", 44100, 2);
        round_trip("mp3", 16000, 1);
    }

    #[test]
    fn m4a_round_trip() {
        round_trip("m4a", 48000, 2);
        round_trip("m4a", 16000, 1);
    }
}
