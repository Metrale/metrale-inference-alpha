// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Pure parsers for the nvidia-smi, procfs and sysfs text that
//! [`super::collect`] reads.
//!
//! Owner: bench hardware.
//! Invariants:
//! - `&str` in, data out: no I/O, so the fixtures in `fixtures/` exercise every parser.
//! - A number that does not parse is `None`, or its row or zone is dropped; no parser
//!   substitutes a default, so a caller can tell "0 µs of throttling" from "no idea".

use super::state::{GpuComputeApp, ThermalZone, ThrottleActive, ThrottleCounters};

/// 2026-09-26: The trimmed cell, or `None` for a blank cell or one of nvidia-smi's
/// not-a-value spellings: `[N/A]` (`--format=csv`, `fixtures/gb10_query_gpu_na.csv`),
/// `N/A` (`-q`, `fixtures/gb10_performance_unsupported.txt`) and `[Not Supported]`.
fn value(cell: &str) -> Option<&str> {
    let c = cell.trim();
    (!c.is_empty() && c != "[N/A]" && c != "N/A" && c != "[Not Supported]").then_some(c)
}

/// 2026-09-26: The first non-blank CSV row of
/// `--query-gpu=name,driver_version,clocks.sm,clocks.max.sm,temperature.gpu,persistence_mode`
/// with `--format=csv,noheader,nounits`.
///
/// Fields are positional; a short row leaves the missing tail unknown rather
/// than shifting values into the wrong slots.
#[derive(Debug, Default, PartialEq)]
pub struct GpuQuery {
    pub name: Option<String>,
    pub driver: Option<String>,
    pub sm_clock_mhz: Option<f64>,
    pub sm_clock_max_mhz: Option<f64>,
    pub gpu_temp_c: Option<f64>,
    pub persistence_mode: Option<bool>,
}

pub fn gpu_query(text: &str) -> GpuQuery {
    let Some(line) = text.lines().find(|l| !l.trim().is_empty()) else {
        return GpuQuery::default();
    };
    let cells: Vec<&str> = line.split(',').collect();
    let cell = |i: usize| cells.get(i).copied().and_then(value);
    let num = |i: usize| {
        cell(i)
            .and_then(|c| c.parse::<f64>().ok())
            .filter(|n| n.is_finite())
    };
    GpuQuery {
        name: cell(0).map(str::to_string),
        driver: cell(1).map(str::to_string),
        sm_clock_mhz: num(2),
        sm_clock_max_mhz: num(3),
        gpu_temp_c: num(4),
        // 2026-09-26: Anything that is neither spelling is unknown, not "off".
        persistence_mode: cell(5).and_then(|c| match c {
            "Enabled" => Some(true),
            "Disabled" => Some(false),
            _ => None,
        }),
    }
}

/// 2026-09-26: The device count from `nvidia-smi -L`, one line per visible device.
///
/// ```text
/// GPU 0: NVIDIA H100 80GB HBM3 (UUID: GPU-2f1c…)
/// GPU 1: NVIDIA H100 80GB HBM3 (UUID: GPU-9a44…)
/// ```
///
/// Only lines that begin with `GPU ` and contain `:` are counted, so the indented
/// `  MIG 1g.10gb  Device 0: …` lines of a MIG-partitioned box are not.
///
/// `None` when no such line is present: an unreadable answer is not a box with
/// zero GPUs, and a count of one from a missing tool would claim a single-GPU
/// topology for a run that may have used more.
pub fn gpu_count(text: &str) -> Option<u32> {
    let n = text
        .lines()
        .filter(|l| l.starts_with("GPU ") && l.contains(':'))
        .count();
    (n > 0).then_some(n as u32)
}

/// 2026-09-26: The rows of
/// `--query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits`.
///
/// A row whose pid does not parse is dropped rather than guessed at: the count
/// this feeds can refuse a run. An unparseable memory cell keeps the row with
/// `used_mib: None`, because the pid is the part the check reads.
pub fn compute_apps(text: &str) -> Vec<GpuComputeApp> {
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .filter_map(|line| {
            let cells: Vec<&str> = line.split(',').collect();
            let pid = value(cells.first()?)?.parse::<u32>().ok()?;
            Some(GpuComputeApp {
                pid,
                name: cells
                    .get(1)
                    .and_then(|c| value(c))
                    .unwrap_or("")
                    .to_string(),
                used_mib: cells
                    .get(2)
                    .and_then(|c| value(c))
                    .and_then(|c| c.split_whitespace().next())
                    .and_then(|c| c.parse().ok()),
            })
        })
        .collect()
}

/// 2026-09-26: The counters and the asserted reasons from `nvidia-smi -q -d PERFORMANCE`.
///
/// The two sections share key names: "SW Thermal Slowdown" appears under
/// `Clocks Event Reasons` as `Not Active` and under `Clocks Event Reasons
/// Counters` as `502088297 us` (`fixtures/gb10_performance.txt`). A flat key scan
/// would read whichever came last, so the section is tracked explicitly. The
/// counters header is tested first because the reasons header is a prefix of it.
pub fn performance(text: &str) -> (ThrottleCounters, ThrottleActive) {
    let (mut counters, mut active) = (ThrottleCounters::default(), ThrottleActive::default());
    let mut in_counters = false;
    let mut in_reasons = false;
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("Clocks Event Reasons Counters") {
            (in_counters, in_reasons) = (true, false);
            continue;
        }
        if trimmed.starts_with("Clocks Event Reasons") {
            (in_counters, in_reasons) = (false, true);
            continue;
        }
        let Some((key, raw)) = trimmed.split_once(':') else {
            // 2026-09-26: A non-blank line with no `:` ends both sections, so
            // keys under a later heading are not read as clock reasons.
            if !trimmed.is_empty() {
                (in_counters, in_reasons) = (false, false);
            }
            continue;
        };
        let key = key.trim();
        let Some(raw) = value(raw) else { continue };
        if in_counters {
            let us = raw
                .split_whitespace()
                .next()
                .and_then(|n| n.parse::<u64>().ok());
            match key {
                "SW Power Capping" => counters.sw_power_cap_us = us,
                "SW Thermal Slowdown" => counters.sw_thermal_us = us,
                "HW Thermal Slowdown" => counters.hw_thermal_us = us,
                "HW Power Braking" => counters.hw_power_brake_us = us,
                "Sync Boost" => counters.sync_boost_us = us,
                _ => {}
            }
        } else if in_reasons {
            // 2026-09-26: "Active"/"Not Active" only. Any other spelling is unknown.
            let flag = match raw {
                "Active" => Some(true),
                "Not Active" => Some(false),
                _ => None,
            };
            match key {
                "SW Power Cap" => active.sw_power_cap = flag,
                "SW Thermal Slowdown" => active.sw_thermal = flag,
                "HW Thermal Slowdown" => active.hw_thermal = flag,
                "HW Power Brake Slowdown" => active.hw_power_brake = flag,
                _ => {}
            }
        }
    }
    (counters, active)
}

/// 2026-09-26: `MemTotal`, `MemAvailable` and `Cached` from `/proc/meminfo`, kB.
///
/// GB10 reports its framebuffer usage as `N/A` (`fixtures/gb10_performance.txt`),
/// so memory is read from here.
#[derive(Debug, Default, PartialEq)]
pub struct MemInfo {
    pub total_kb: Option<u64>,
    pub available_kb: Option<u64>,
    pub cached_kb: Option<u64>,
}

pub fn meminfo(text: &str) -> MemInfo {
    let mut out = MemInfo::default();
    for line in text.lines() {
        let Some((key, raw)) = line.split_once(':') else {
            continue;
        };
        let kb = raw
            .split_whitespace()
            .next()
            .and_then(|n| n.parse::<u64>().ok());
        match key.trim() {
            "MemTotal" => out.total_kb = kb,
            "MemAvailable" => out.available_kb = kb,
            // 2026-09-26: An exact key, so `SwapCached` never lands here.
            "Cached" => out.cached_kb = kb,
            _ => {}
        }
    }
    out
}

/// 2026-09-26: A `/sys/class/thermal/thermal_zone*/temp` file, milli-degrees
/// Celsius, as °C; `None` when it is not a finite number.
pub fn milli_celsius(text: &str) -> Option<f64> {
    text.trim()
        .parse::<f64>()
        .ok()
        .filter(|m| m.is_finite())
        .map(|m| m / 1000.0)
}

/// 2026-09-26: Assemble one zone from its `type` and `temp` file contents.
///
/// A zone whose temperature does not parse is dropped: an unreadable zone is not
/// a 0 °C zone, and the hottest zone feeds the precheck and the equivalence test.
pub fn thermal_zone(type_text: Option<&str>, temp_text: &str) -> Option<ThermalZone> {
    Some(ThermalZone {
        name: type_text.unwrap_or("").trim().to_string(),
        temp_c: milli_celsius(temp_text)?,
    })
}

#[cfg(test)]
#[path = "parse_tests.rs"]
mod tests;
