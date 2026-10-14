// SPDX-FileCopyrightText: Copyright (C) 2026 Brishen Hawkins
// SPDX-License-Identifier: Apache-2.0

//! `reuse-clean`: clean up a recording (MP3, M4A / AAC, FLAC or WAV) with
//! RE-USE on the NPU.
//!
//! ```text
//! reuse-clean <in.mp3|m4a|flac|wav> [-o out.{mp3,m4a,wav}] [--bundle DIR] [--rate HZ]
//!             [--bitrate KBPS] [--mono] [--float]
//! ```
//!
//! Writes `<in>.clean.<ext>` next to the input in its own format
//! (`noisy.mp3` -> `noisy.clean.mp3`; FLAC comes out as WAV) unless `-o`
//! says otherwise (its extension picks the format), at the input's rate.
//! MP3 and M4A (feature `encode`; without it, WAV) go out at the input's
//! bitrate (`--bitrate` overrides), gapless; WAV is 16-bit PCM (`--float`:
//! 32-bit float). The model runs at the input's rate when the bundle has kernels for it,
//! else the nearest rate above; `--rate` picks one of the bundle's rates
//! instead (16000 runs ~2.5x faster than 44100 and keeps speech up to 8
//! kHz). Every channel is enhanced, once when they are all the same
//! (`--mono`: their mean). The bundle comes from `--bundle`, else
//! `$REUSE_BUNDLE`, else the one built in (feature `embed`).

use std::path::{Path, PathBuf};
use std::fs;
use std::process::ExitCode;
use std::time::Instant;

use taconite_reuse::Reuse;
use taconite_reuse::media;
use taconite_reuse::wav::{self, Audio};

type R<T> = Result<T, Box<dyn std::error::Error>>;

const USAGE: &str = "usage: reuse-clean <in.mp3|m4a|flac|wav> [-o out.{mp3,m4a,wav}] [--bundle DIR] [--rate HZ] \
                     [--bitrate KBPS] [--mono] [--float]";

#[derive(Clone, Copy, PartialEq, Debug)]
enum Fmt {
    Wav,
    Mp3,
    M4a,
}

impl Fmt {
    /// From a file's extension: the format it gets written in (FLAC and
    /// the rest as WAV).
    fn of(p: &Path) -> Fmt {
        match p.extension().and_then(|e| e.to_str()).map(str::to_ascii_lowercase).as_deref() {
            Some("mp3") => Fmt::Mp3,
            Some("m4a" | "mp4" | "aac" | "m4b") => Fmt::M4a,
            _ => Fmt::Wav,
        }
    }
    fn ext(self) -> &'static str {
        match self {
            Fmt::Wav => "wav",
            Fmt::Mp3 => "mp3",
            Fmt::M4a => "m4a",
        }
    }
}

fn run() -> R<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (mut inp, mut out, mut bundle) = (None, None, std::env::var_os("REUSE_BUNDLE").map(PathBuf::from));
    let (mut mono, mut float, mut rate, mut kbps) = (false, false, None::<u32>, None::<u32>);
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "-o" | "--out" => out = Some(PathBuf::from(it.next().ok_or("-o <out.mp3|m4a|wav>")?)),
            "--bitrate" => kbps = Some(it.next().ok_or("--bitrate <kbps>")?.parse()?),
            "--bundle" => bundle = Some(PathBuf::from(it.next().ok_or("--bundle <dir>")?)),
            "--rate" => rate = Some(it.next().ok_or("--rate <Hz>")?.parse()?),
            "--mono" => mono = true,
            "--float" => float = true,
            "-h" | "--help" => {
                println!("{USAGE}");
                return Ok(());
            }
            s if s.starts_with('-') => return Err(format!("unknown option {s}\n{USAGE}").into()),
            _ if inp.is_none() => inp = Some(PathBuf::from(a)),
            _ => return Err(USAGE.into()),
        }
    }
    let inp = inp.ok_or(USAGE)?;
    let mut out = out.unwrap_or_else(|| inp.with_extension(format!("clean.{}", Fmt::of(&inp).ext())));
    let mut fmt = Fmt::of(&out);
    if fmt != Fmt::Wav && !cfg!(feature = "encode") {
        out = out.with_extension("wav");
        eprintln!("(built without the `encode` feature: writing {} instead)", out.display());
        fmt = Fmt::Wav;
    }

    let t0 = Instant::now();
    let mut a = media::read(&inp)?;
    let secs = a.channels[0].len() as f64 / a.sr as f64;
    eprintln!(
        "{}: {secs:.2} s, {} channel(s) at {} Hz (decoded in {:.0} ms)",
        inp.display(),
        a.channels.len(),
        a.sr,
        t0.elapsed().as_secs_f64() * 1e3
    );
    if mono && a.channels.len() > 1 {
        a = media::mono(&a);
    }

    let t0 = Instant::now();
    let mut m = match &bundle {
        Some(b) => Reuse::load(b)?,
        None => load_builtin()?,
    };
    eprintln!("bundle loaded in {:.1} s ({} hardware contexts)", t0.elapsed().as_secs_f64(), m.contexts());
    let sr = match rate {
        Some(r) if m.cfg.rates.contains(&r) => r,
        Some(r) => return Err(format!("--rate {r}: the bundle has kernels for {:?} Hz", m.cfg.rates).into()),
        None => media::nearest_rate(a.sr, &m.cfg.rates).ok_or("the bundle lists no rates")?,
    };
    if sr != a.sr {
        eprintln!("running at {sr} Hz (resampled from and back to {} Hz)", a.sr);
    }

    // dual mono (a mono recording in a stereo file): enhance it once
    let same = a.channels.iter().all(|c| c == &a.channels[0]);
    let n = if same { 1 } else { a.channels.len() };
    if !same && near_mono(&a.channels) {
        eprintln!("(the channels are nearly the same: --mono would take 1/{} the time)", a.channels.len());
    }
    let mut channels = Vec::new();
    let mut bar = Bar::new(n);
    for (ci, ch) in a.channels.iter().take(n).enumerate() {
        let x = media::resample(ch, a.sr, sr);
        let t0 = Instant::now();
        let y = m.enhance_with(&x, sr, &mut |f| bar.set(ci, f))?;
        bar.clear();
        let dt = t0.elapsed().as_secs_f64();
        eprintln!("channel {ci}: cleaned in {:.0} ms (x{:.1} real time)", dt * 1e3, secs / dt);
        let mut y = media::resample(&y, sr, a.sr);
        y.resize(ch.len(), 0.0);
        channels.push(y);
    }
    if same && a.channels.len() > 1 {
        eprintln!("(the {} channels are identical: enhanced once)", a.channels.len());
        channels = vec![channels[0].clone(); a.channels.len()];
    }
    let clean = Audio { channels, sr: a.sr };
    match fmt {
        Fmt::Wav => wav::write(&out, &clean, float)?,
        _ => {
            // the input's bitrate (its size over its length: close enough
            // for CBR / ABR files), for a like-for-like file; 64 kbps a
            // channel from uncompressed input
            let est = (fs::metadata(&inp)?.len() as f64 * 8.0 / secs / 1000.0).round() as u32;
            let pcm_in = matches!(Fmt::of(&inp), Fmt::Wav);
            let k = snap_kbps(kbps.unwrap_or(if pcm_in { 64 * clean.channels.len() as u32 } else { est }));
            eprintln!("encoding {} at {k} kbps", fmt.ext());
            write_compressed(fmt, &out, &clean, k)?;
        }
    }
    eprintln!("wrote {}", out.display());
    Ok(())
}

/// Every channel within 20 dB of the first (a mono source through a
/// joint-stereo codec, say).
fn near_mono(ch: &[Vec<f32>]) -> bool {
    let e0: f64 = ch[0].iter().map(|&v| (v as f64).powi(2)).sum();
    ch[1..].iter().all(|c| {
        let d: f64 = c.iter().zip(&ch[0]).map(|(&a, &b)| (a as f64 - b as f64).powi(2)).sum();
        d * 100.0 < e0
    })
}

/// A progress bar on stderr over every channel's enhancement, with the
/// time so far and an estimate of the rest; drawn only on a terminal (and
/// at most ~10 times a second), so logs and pipes stay clean.
struct Bar {
    channels: usize,
    t0: Instant,
    last: Option<Instant>,
    /// (time, fraction) at the first report: the rate is measured from
    /// there, past the kernels' one-off first loads
    first: Option<(f64, f64)>,
    tty: bool,
}

impl Bar {
    fn new(channels: usize) -> Self {
        use std::io::IsTerminal;
        Bar { channels, t0: Instant::now(), last: None, first: None, tty: std::io::stderr().is_terminal() }
    }

    /// Channel `ch` is `f` (0 to 1) done.
    fn set(&mut self, ch: usize, f: f64) {
        let done = ((ch as f64 + f.clamp(0.0, 1.0)) / self.channels as f64).min(1.0);
        let el = self.t0.elapsed().as_secs_f64();
        let (e0, d0) = *self.first.get_or_insert((el, done));
        if !self.tty || self.last.is_some_and(|t| t.elapsed().as_millis() < 100 && f < 1.0) {
            return;
        }
        self.last = Some(Instant::now());
        const W: usize = 30;
        let fill = (done * W as f64).round() as usize;
        let eta = if done - d0 > 0.02 {
            format!(", ~{:.1} s left", (el - e0) / (done - d0) * (1.0 - done))
        } else {
            String::new()
        };
        let chan = if self.channels > 1 { format!(" channel {}/{}", ch + 1, self.channels) } else { String::new() };
        eprint!("\r\x1b[2Kcleaning{chan} [{}{}] {:3.0}%  {el:.1} s{eta}", "█".repeat(fill), "░".repeat(W - fill), done * 100.0);
    }

    fn clear(&mut self) {
        if self.tty {
            eprint!("\r\x1b[2K");
        }
    }
}

/// The nearest of MP3's bitrates (AAC takes any; these are the familiar ones).
fn snap_kbps(k: u32) -> u32 {
    const R: [u32; 14] = [32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320];
    *R.iter().min_by_key(|&&r| r.abs_diff(k)).unwrap()
}

#[cfg(feature = "encode")]
fn write_compressed(fmt: Fmt, out: &Path, a: &Audio, kbps: u32) -> R<()> {
    match fmt {
        Fmt::Mp3 => taconite_reuse::encode::write_mp3(out, a, kbps)?,
        Fmt::M4a => taconite_reuse::encode::write_m4a(out, a, kbps)?,
        Fmt::Wav => unreachable!(),
    }
    Ok(())
}

#[cfg(not(feature = "encode"))]
fn write_compressed(_: Fmt, _: &Path, _: &Audio, _: u32) -> R<()> {
    unreachable!("formats other than WAV need the `encode` feature")
}

#[cfg(feature = "embed")]
fn load_builtin() -> R<Reuse> {
    Ok(Reuse::load_embedded()?)
}

#[cfg(not(feature = "embed"))]
fn load_builtin() -> R<Reuse> {
    Err("no bundle: pass --bundle <dir> or set REUSE_BUNDLE (or build one in: feature `embed`)".into())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
