// SPDX-FileCopyrightText: Copyright (C) 2026 Brishen Hawkins
// SPDX-License-Identifier: Apache-2.0

//! Per-stage wall time, as the model crates report it.

use std::fmt;
use std::time::Duration;

/// Wall time per stage, in first-seen order; NPU dispatch time is kept
/// under `npu:<kernel>`.
#[derive(Default, Debug, Clone)]
pub struct Timing {
    entries: Vec<(String, Duration)>,
}

impl Timing {
    pub fn add(&mut self, key: &str, d: Duration) {
        match self.entries.iter_mut().find(|(k, _)| k == key) {
            Some((_, t)) => *t += d,
            None => self.entries.push((key.to_string(), d)),
        }
    }

    pub fn get(&self, key: &str) -> Duration {
        self.entries.iter().find(|(k, _)| k == key).map_or(Duration::ZERO, |e| e.1)
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }

    pub fn npu_total(&self) -> Duration {
        self.entries.iter().filter(|(k, _)| k.starts_with("npu:")).map(|e| e.1).sum()
    }
}

impl fmt::Display for Timing {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ms = |d: &Duration| d.as_secs_f64() * 1e3;
        let (npu, host): (Vec<_>, Vec<_>) = self.entries.iter().partition(|(k, _)| k.starts_with("npu:"));
        write!(f, "stages:")?;
        for (k, d) in &host {
            write!(f, " {k} {:.0}", ms(d))?;
        }
        write!(f, " ms; npu {:.0} ms:", ms(&self.npu_total()))?;
        for (k, d) in &npu {
            write!(f, " {} {:.0}", &k[4..], ms(d))?;
        }
        Ok(())
    }
}
