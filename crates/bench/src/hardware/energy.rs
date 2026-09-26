// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: In-window GPU energy, the pure half: integrate the power readings
//! that [`super::energy_sampler`] streams and turn a measured window into record keys.
//!
//! Owner: bench hardware.
//! Invariants:
//! - No I/O: every function works on readings it is handed.
//! - Every key this module writes carries `gpu_rail` in its name.
//! - A window with no reading inside it, and a J/token ratio with no tokens or no
//!   joules, yield `None`: absent, never zero.
//!
//! # Store joules and tokens, derive the ratios
//!
//! Joules and tokens add across windows; a ratio does not. The record carries
//! `gpu_rail_energy_j` and the window's token count. It also carries one ratio,
//! `gpu_rail_joules_per_token`, because a BENCH.toml ceiling can only bound a key
//! the record carries (the per-rung J/token ceilings in
//! `kernels/gb10/qwen3.8-27b/BENCH.toml`). Code that re-aggregates windows adds
//! joules and tokens and divides once, with [`joules_per_token`].
//!
//! # Which rail
//!
//! The reading is nvidia-smi's `power.draw.average`, the GPU rail only. Measured
//! 2026-09-20 on GB10, driver 580.126.09: `power.limit`, the module power readings
//! and GPU memory power answer `N/A`, so the Grace cores and LPDDR5X are outside
//! this number. `gpu_rail` in every key name keeps a reader from setting it beside
//! a discrete card's board power.
//!
//! # Why not the driver's energy counter
//!
//! The telemetry crate reads the driver's cumulative counter
//! (`nvmlDeviceGetTotalEnergyConsumption`, `crates/telemetry/src/device.rs`).
//! Switching this reader to it would change what every recorded `gpu_rail_*` key
//! means under an unchanged name, while the BENCH.toml J/token ceilings bound values
//! of this integral. Such a switch needs new key names and re-measured ceilings.

use std::collections::BTreeMap;
use std::time::Instant;

/// 2026-09-26: Sampler cadence (`nvidia-smi -lms`), written beside every run's
/// joules as `gpu_rail_sample_period_ms`.
pub const SAMPLE_PERIOD_MS: u64 = 250;

/// 2026-09-26: Length of the idle-baseline window that
/// [`super::energy_sampler::EnergySampler::idle_baseline`] integrates.
pub const IDLE_BASELINE_SECS: u64 = 2;

/// 2026-09-26: Logged once per run when sampling starts (`EnergyMeter::start`).
pub const RAIL_NOTE: &str = "energy: GPU RAIL ONLY (nvidia-smi power.draw.average, 250 ms cadence). \
     On GB10 the module rail — Grace cores + LPDDR5X — reads N/A and is NOT in this number; on a \
     bandwidth-bound unified-memory decode a dominant share of real energy is outside it. Do not \
     compare to a discrete card's board power.";

/// 2026-09-26: One reading from the sampler.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PowerSample {
    pub at: Instant,
    pub power_w: f64,
    /// 2026-09-26: `clocks_event_reasons.sw_power_cap`; `None` when the cell was
    /// neither `Active` nor `Not Active`.
    pub sw_power_cap: Option<bool>,
    /// 2026-09-26: `clocks_event_reasons.hw_power_brake_slowdown`, same encoding.
    pub hw_power_brake: Option<bool>,
}

/// 2026-09-26: Parse one CSV row of
/// `--query-gpu=power.draw.average,clocks_event_reasons.sw_power_cap,clocks_event_reasons.hw_power_brake_slowdown`
/// with `--format=csv,noheader,nounits`, e.g. `4.76, Not Active, Active`.
///
/// `None` when the watt cell is not a finite, non-negative number, so a `[N/A]`
/// rail never integrates as zero watts. A missing or unrecognised flag cell
/// leaves that flag `None` rather than "not active".
pub fn parse_line(line: &str, at: Instant) -> Option<PowerSample> {
    let cells: Vec<&str> = line.split(',').map(str::trim).collect();
    let power_w = cells
        .first()?
        .parse::<f64>()
        .ok()
        .filter(|w| w.is_finite() && *w >= 0.0)?;
    let flag = |i: usize| match cells.get(i).copied() {
        Some("Active") => Some(true),
        Some("Not Active") => Some(false),
        _ => None,
    };
    Some(PowerSample {
        at,
        power_w,
        sw_power_cap: flag(1),
        hw_power_brake: flag(2),
    })
}

/// 2026-09-26: The energy integral over one measured window.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct EnergyWindow {
    /// 2026-09-26: Window length the joules span, seconds.
    pub window_s: f64,
    /// 2026-09-26: Readings the integral stood on.
    pub samples: usize,
    /// 2026-09-26: `Σ P_i · dt_i`: each reading held forward to the next, the
    /// first back-filled to the window start, the last held to the window end,
    /// so `Σ dt_i` is exactly `window_s`.
    pub energy_j: f64,
    pub mean_power_w: f64,
    pub max_power_w: f64,
    /// 2026-09-26: Of the readings that reported the SW power cap flag, the
    /// fraction (by count, not time) with it asserted; `None` when none reported it.
    pub sw_power_cap_frac: Option<f64>,
    /// 2026-09-26: Same, for the HW power brake.
    pub hw_power_brake_frac: Option<f64>,
}

/// 2026-09-26: Integrate the readings inside `[start, end]`, both ends included.
///
/// `None` when `end <= start` or no reading falls inside, so a window with no
/// evidence never reports zero joules.
pub fn integrate(samples: &[PowerSample], start: Instant, end: Instant) -> Option<EnergyWindow> {
    if end <= start {
        return None;
    }
    let inside: Vec<&PowerSample> = samples
        .iter()
        .filter(|s| s.at >= start && s.at <= end)
        .collect();
    if inside.is_empty() {
        return None;
    }
    let window_s = end.duration_since(start).as_secs_f64();
    let mut energy_j = 0.0;
    let mut max_power_w = 0.0f64;
    for (i, s) in inside.iter().enumerate() {
        let from = if i == 0 { start } else { s.at };
        let to = inside.get(i + 1).map_or(end, |n| n.at);
        energy_j += s.power_w * to.duration_since(from).as_secs_f64();
        max_power_w = max_power_w.max(s.power_w);
    }
    let frac = |pick: fn(&PowerSample) -> Option<bool>| {
        let reported: Vec<bool> = inside.iter().filter_map(|s| pick(s)).collect();
        (!reported.is_empty())
            .then(|| reported.iter().filter(|b| **b).count() as f64 / reported.len() as f64)
    };
    Some(EnergyWindow {
        window_s,
        samples: inside.len(),
        energy_j,
        mean_power_w: energy_j / window_s,
        max_power_w,
        sw_power_cap_frac: frac(|s| s.sw_power_cap),
        hw_power_brake_frac: frac(|s| s.hw_power_brake),
    })
}

/// 2026-09-26: GPU-rail joules per delivered output token, or `None` when the
/// pair cannot support a ratio: no tokens, or a joule count that is not a finite
/// positive number (a delivered token never costs zero energy, so 0 J means the
/// window measured nothing). `None` rather than 0.0 or ∞ because a ceiling bounds
/// this value, and any sentinel would read as a pass or as a real reading.
pub fn joules_per_token(energy_j: f64, tokens: usize) -> Option<f64> {
    (tokens > 0 && energy_j.is_finite() && energy_j > 0.0).then(|| energy_j / tokens as f64)
}

impl EnergyWindow {
    /// 2026-09-26: Joules above what the box would have drawn idle for the same
    /// duration: `energy_j − idle.mean_power_w × window_s`. Negative when the
    /// window drew less than the baseline; not clamped.
    pub fn above_idle_j(&self, idle: &EnergyWindow) -> f64 {
        self.energy_j - idle.mean_power_w * self.window_s
    }

    /// 2026-09-26: Several windows as one: joules, seconds and samples add, the
    /// mean is re-derived, the max is the largest max, and each flag fraction is
    /// weighted by the sample counts of the windows that report it. `None` for
    /// no windows.
    pub fn sum(windows: &[EnergyWindow]) -> Option<EnergyWindow> {
        if windows.is_empty() {
            return None;
        }
        let window_s: f64 = windows.iter().map(|w| w.window_s).sum();
        let energy_j: f64 = windows.iter().map(|w| w.energy_j).sum();
        let samples: usize = windows.iter().map(|w| w.samples).sum();
        let weighted = |pick: fn(&EnergyWindow) -> Option<f64>| {
            let (num, den) = windows
                .iter()
                .fold((0.0, 0usize), |(n, d), w| match pick(w) {
                    Some(f) => (n + f * w.samples as f64, d + w.samples),
                    None => (n, d),
                });
            (den > 0).then(|| num / den as f64)
        };
        Some(EnergyWindow {
            window_s,
            samples,
            energy_j,
            mean_power_w: if window_s > 0.0 {
                energy_j / window_s
            } else {
                0.0
            },
            max_power_w: windows.iter().map(|w| w.max_power_w).fold(0.0, f64::max),
            sw_power_cap_frac: weighted(|w| w.sw_power_cap_frac),
            hw_power_brake_frac: weighted(|w| w.hw_power_brake_frac),
        })
    }

    /// 2026-09-26: The record keys for one window, under `prefix` (`"c{C}_"` from
    /// the concurrency sweep, `""` from decode-floor). `tokens` is the caller's
    /// count of output tokens delivered inside this window; it is written beside
    /// the joules, and J/token is derived from the pair.
    pub fn metrics(
        &self,
        prefix: &str,
        tokens: usize,
        idle: Option<&EnergyWindow>,
        m: &mut BTreeMap<String, f64>,
    ) {
        let put = |m: &mut BTreeMap<String, f64>, k: &str, v: f64| {
            m.insert(format!("{prefix}{k}"), v);
        };
        put(m, "gpu_rail_energy_j", self.energy_j);
        put(m, "gpu_rail_energy_window_tokens", tokens as f64);
        // 2026-09-26: Absent when the window cannot support the ratio, so a J/token
        // ceiling fails as missing from the record rather than passing.
        if let Some(r) = joules_per_token(self.energy_j, tokens) {
            put(m, "gpu_rail_joules_per_token", r);
        }
        put(m, "gpu_rail_mean_power_w", self.mean_power_w);
        put(m, "gpu_rail_max_power_w", self.max_power_w);
        put(m, "gpu_rail_power_samples", self.samples as f64);
        put(m, "gpu_rail_energy_window_s", self.window_s);
        if let Some(f) = self.sw_power_cap_frac {
            put(m, "gpu_rail_sw_power_cap_frac", f);
        }
        if let Some(f) = self.hw_power_brake_frac {
            put(m, "gpu_rail_hw_power_brake_frac", f);
        }
        if let Some(idle) = idle {
            put(m, "gpu_rail_energy_above_idle_j", self.above_idle_j(idle));
        }
    }

    /// 2026-09-26: One log line: joules, seconds, mean and max watts, samples,
    /// above-idle joules when a baseline is given, the SW cap fraction, and the
    /// HW brake fraction when it is above zero.
    pub fn one_line(&self, idle: Option<&EnergyWindow>) -> String {
        let mut s = format!(
            "gpu rail {:.1} J over {:.1} s ({:.1} W mean, {:.1} W max, {} samples)",
            self.energy_j, self.window_s, self.mean_power_w, self.max_power_w, self.samples
        );
        if let Some(idle) = idle {
            s.push_str(&format!(
                " · {:.1} J above the {:.1} W idle baseline",
                self.above_idle_j(idle),
                idle.mean_power_w
            ));
        }
        match self.sw_power_cap_frac {
            Some(f) => s.push_str(&format!(" · sw power cap {:.0}% of samples", f * 100.0)),
            None => s.push_str(" · sw power cap unreported"),
        }
        if let Some(f) = self.hw_power_brake_frac.filter(|f| *f > 0.0) {
            s.push_str(&format!(" · HW POWER BRAKE {:.0}% of samples", f * 100.0));
        }
        s
    }
}

/// 2026-09-26: What the sampler itself cost, written to the record as
/// `gpu_rail_sample*` keys so the instrument's overhead is visible.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SamplerCost {
    /// 2026-09-26: CPU seconds the sampler child consumed, from the first field
    /// of `/proc/<pid>/schedstat` read as nanoseconds. `None` when procfs could
    /// not be read.
    pub cpu_s: Option<f64>,
    /// 2026-09-26: Wall seconds it ran.
    pub wall_s: f64,
    /// 2026-09-26: Readings it produced.
    pub samples: usize,
    /// 2026-09-26: Lines it emitted that did not parse as a reading, such as an
    /// `[N/A]` watt cell. Written to the record when non-zero.
    pub rejected_lines: u64,
}

impl SamplerCost {
    pub fn metrics(&self, m: &mut BTreeMap<String, f64>) {
        m.insert(
            "gpu_rail_sample_period_ms".to_string(),
            SAMPLE_PERIOD_MS as f64,
        );
        if let Some(cpu) = self.cpu_s {
            m.insert("gpu_rail_sampler_cpu_s".to_string(), cpu);
        }
        m.insert("gpu_rail_sampler_wall_s".to_string(), self.wall_s);
        if self.rejected_lines > 0 {
            m.insert(
                "gpu_rail_sampler_rejected_lines".to_string(),
                self.rejected_lines as f64,
            );
        }
    }

    pub fn one_line(&self) -> String {
        format!(
            "energy sampler cost: {} CPU over {:.0} s wall ({} samples at {SAMPLE_PERIOD_MS} ms{})",
            self.cpu_s
                .map(|c| format!("{c:.3} s"))
                .unwrap_or_else(|| "unmeasured".into()),
            self.wall_s,
            self.samples,
            if self.rejected_lines > 0 {
                format!(", {} unparseable lines", self.rejected_lines)
            } else {
                String::new()
            }
        )
    }
}

#[cfg(test)]
#[path = "energy_tests.rs"]
mod tests;
