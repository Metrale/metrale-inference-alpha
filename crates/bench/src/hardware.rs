// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The hardware fingerprint: which box a benchmark number was
//! measured on, and the box-class key gate baselines are indexed by.
//!
//! * [`Hardware`] names the box. A gate record's copy comes from the serving
//!   endpoint's `/hardware` (`http::fetch_hardware`), so it describes the
//!   serving box even when the benchmark CLI runs elsewhere;
//!   [`Hardware::probe`] reads the local box.
//! * [`state::HardwareState`] records the box's state. The executor collects
//!   it before a run and, unless that capture fails or the pre-check refuses
//!   the run, after; [`policy`] decides what is gated on it.
//!
//! Owner: bench (hardware).
//! Invariants:
//! - [`Hardware::probe`] never fails; with no tool answering it returns
//!   [`Hardware::unknown`].
//! - [`Hardware::gate_key`] is never empty: a name that yields no key gives
//!   `"unknown"`.

use serde::{Deserialize, Serialize};

pub mod collect;
pub mod energy;
pub mod energy_sampler;
pub mod equivalence;
pub mod ids;
pub mod limits;
pub mod parse;
pub mod policy;
pub mod report;
pub mod state;
pub mod throttle_monitor;

pub use policy::{Decision, Sensitivity, Validity};
pub use report::HardwareStateReport;
pub use state::{HardwareState, HardwareStateDelta};

/// 2026-09-26: A box's fingerprint, as its serving endpoint reported it or
/// as [`Hardware::probe`] read it.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Hardware {
    /// 2026-09-26: Device model, e.g. "NVIDIA GB10", or `pci:<vendor>:<device>`
    /// from sysfs. Empty when unknown.
    #[serde(default)]
    pub gpu: String,
    /// 2026-09-26: Driver version, e.g. "580.126.09". Empty when unknown.
    #[serde(default)]
    pub driver: String,
    /// 2026-09-26: GPU clock in MHz at probe time: nvidia-smi's `clocks.sm`,
    /// or the clock `rocm-smi --showclocks` reports. `None` for `[N/A]`, an
    /// unparseable cell, and every sysfs reading.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sm_clock_mhz: Option<f64>,
    /// 2026-09-26: How many GPUs the box exposed at probe time, counted from
    /// `nvidia-smi -L` (`parse::gpu_count`); `gpu` names only the first. `None`
    /// means unmeasured, never one: rocm-smi and sysfs leave it `None`, and it
    /// is left out of the JSON.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_count: Option<u32>,
    /// 2026-09-26: Where the reading came from ("nvidia-smi", "rocm-smi",
    /// "sysfs"). Empty when unknown.
    #[serde(default)]
    pub source: String,
}

impl Hardware {
    pub fn unknown() -> Self {
        Self::default()
    }

    /// 2026-09-26: The box-class key a gate baseline is indexed by, e.g.
    /// `"gb10"`: a class, not a host.
    ///
    /// A name with a token in the SKU table ([`ids::hardware_id_from_gpu_name`])
    /// gets that id, so `"NVIDIA H100 80GB HBM3"` is `h100`. Any other name is
    /// lowercased with `nvidia` and `amd` removed and only alphanumerics kept.
    /// An empty `gpu`, or a name that leaves nothing, gives `"unknown"`, which
    /// no baseline defines; `fetch_hardware` returns [`Hardware::unknown`] on
    /// every error path without surfacing one.
    pub fn gate_key(&self) -> String {
        if self.gpu.is_empty() {
            return "unknown".to_string();
        }
        if let Some(id) = ids::hardware_id_from_gpu_name(&self.gpu) {
            return id.to_string();
        }
        let key: String = self
            .gpu
            .to_lowercase()
            .replace("nvidia", "")
            .replace("amd", "")
            .chars()
            .filter(|c| c.is_alphanumeric())
            .collect();
        if key.is_empty() {
            "unknown".to_string()
        } else {
            key
        }
    }

    /// 2026-09-26: True when `gpu`, `driver` and `sm_clock_mhz` are all empty;
    /// `gpu_count` and `source` are not consulted.
    pub fn is_unknown(&self) -> bool {
        self.gpu.is_empty() && self.driver.is_empty() && self.sm_clock_mhz.is_none()
    }

    /// 2026-09-26: One-line summary for reports, e.g. "NVIDIA GB10 · driver
    /// 580.126.09 · sm 208 MHz"; a GPU count above one follows the part as
    /// `×n`.
    pub fn one_line(&self) -> String {
        if self.is_unknown() {
            return "unknown hardware".to_string();
        }
        let mut parts = Vec::new();
        if !self.gpu.is_empty() {
            parts.push(match self.gpu_count {
                Some(n) if n > 1 => format!("{} ×{n}", self.gpu),
                _ => self.gpu.clone(),
            });
        }
        if !self.driver.is_empty() {
            parts.push(format!("driver {}", self.driver));
        }
        if let Some(clock) = self.sm_clock_mhz {
            parts.push(format!("sm {clock:.0} MHz"));
        }
        parts.join(" · ")
    }

    /// 2026-09-26: Probe the local box: `nvidia-smi`, then `rocm-smi`, then
    /// sysfs, returning the first that answers, or [`Hardware::unknown`].
    pub fn probe() -> Self {
        nvidia_smi()
            .or_else(rocm_smi)
            .or_else(sysfs)
            .unwrap_or_else(Self::unknown)
    }
}

/// 2026-09-26: A tool's trimmed stdout; `None` when it cannot start, exits
/// non-zero or prints nothing.
fn run(tool: &str, args: &[&str]) -> Option<String> {
    let out = std::process::Command::new(tool)
        .args(args)
        .stdin(std::process::Stdio::null())
        .output()
        .ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
}

/// 2026-09-26: Name, driver and SM clock of the first GPU from one CSV query
/// (an `[N/A]` clock is `None`), and the GPU count from `nvidia-smi -L`.
fn nvidia_smi() -> Option<Hardware> {
    let line = run(
        "nvidia-smi",
        &[
            "--query-gpu=name,driver_version,clocks.sm",
            "--format=csv,noheader,nounits",
        ],
    )?
    .lines()
    .next()?
    .to_string();
    let mut cells = line.split(',').map(str::trim);
    let gpu = cells.next().unwrap_or_default().to_string();
    let driver = cells.next().unwrap_or_default().to_string();
    let sm_clock_mhz = cells
        .next()
        .filter(|v| *v != "[N/A]")
        .and_then(|v| v.parse().ok());
    let gpu_count = run("nvidia-smi", &["-L"]).and_then(|out| parse::gpu_count(&out));
    Some(Hardware {
        gpu,
        driver,
        sm_clock_mhz,
        gpu_count,
        source: "nvidia-smi".into(),
    })
}

/// 2026-09-26: rocm-smi: one call for the card and driver, one for the clock.
/// The GPU count is left `None`.
fn rocm_smi() -> Option<Hardware> {
    let card = run(
        "rocm-smi",
        &["--showproductname", "--showdriverversion", "--csv"],
    )?;
    let mut gpu = String::new();
    let mut driver = String::new();
    for line in card.lines().skip(1) {
        let cells: Vec<&str> = line.split(',').map(str::trim).collect();
        if cells.len() >= 3 {
            gpu = cells[1].to_string();
            driver = cells[2].to_string();
            break;
        }
    }
    let sm_clock_mhz = run("rocm-smi", &["--showclocks", "--csv"]).and_then(|out| {
        out.lines()
            .find_map(|l| l.split(',').nth(2))
            .and_then(|v| v.trim().parse().ok())
    });
    Some(Hardware {
        gpu,
        driver,
        sm_clock_mhz,
        gpu_count: None,
        source: "rocm-smi".into(),
    })
}

/// 2026-09-26: Last resort: vendor and device ids of the first device under
/// `/sys/bus/pci/devices` whose class starts with `0x03`, with no driver or
/// clock.
fn sysfs() -> Option<Hardware> {
    for entry in std::fs::read_dir("/sys/bus/pci/devices").ok()?.flatten() {
        let path = entry.path();
        let Ok(class) = std::fs::read_to_string(path.join("class")) else {
            continue;
        };
        if !class.trim().starts_with("0x03") {
            continue;
        }
        let vendor = std::fs::read_to_string(path.join("vendor")).unwrap_or_default();
        let device = std::fs::read_to_string(path.join("device")).unwrap_or_default();
        return Some(Hardware {
            gpu: format!("pci:{}:{}", vendor.trim(), device.trim()),
            source: "sysfs".into(),
            ..Hardware::default()
        });
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_keeps_every_field() {
        let hw = Hardware {
            gpu: "NVIDIA GB10".into(),
            driver: "580.126.09".into(),
            sm_clock_mhz: Some(208.0),
            gpu_count: Some(1),
            source: "nvidia-smi".into(),
        };
        let back: Hardware = serde_json::from_str(&serde_json::to_string(&hw).unwrap()).unwrap();
        assert_eq!(hw, back);
    }

    #[test]
    fn missing_fields_default_to_unknown() {
        let hw: Hardware = serde_json::from_str("{}").unwrap();
        assert_eq!(hw, Hardware::unknown());
        assert_eq!(hw.one_line(), "unknown hardware");
    }

    #[test]
    fn one_line_lists_each_reported_measurement() {
        let hw = Hardware {
            gpu: "NVIDIA GB10".into(),
            driver: "580.126.09".into(),
            sm_clock_mhz: Some(208.0),
            gpu_count: None,
            source: "nvidia-smi".into(),
        };
        assert_eq!(
            hw.one_line(),
            "NVIDIA GB10 · driver 580.126.09 · sm 208 MHz"
        );
        assert_eq!(
            Hardware {
                gpu: hw.gpu.clone(),
                ..Hardware::default()
            }
            .one_line(),
            "NVIDIA GB10"
        );
        assert_eq!(
            Hardware {
                driver: hw.driver.clone(),
                ..Hardware::default()
            }
            .one_line(),
            "driver 580.126.09"
        );
        assert_eq!(
            Hardware {
                sm_clock_mhz: hw.sm_clock_mhz,
                ..Hardware::default()
            }
            .one_line(),
            "sm 208 MHz"
        );
    }
    /// 2026-09-26: A count above one appears as `×n` after the part; a count
    /// of one and an unmeasured count add nothing.
    #[test]
    fn a_multi_gpu_box_names_its_width_and_a_single_one_stays_quiet() {
        let node = |n| Hardware {
            gpu: "NVIDIA H100 80GB HBM3".into(),
            driver: "580.126.09".into(),
            gpu_count: n,
            ..Hardware::default()
        };
        assert_eq!(
            node(Some(8)).one_line(),
            "NVIDIA H100 80GB HBM3 ×8 · driver 580.126.09"
        );
        assert_eq!(
            node(Some(1)).one_line(),
            "NVIDIA H100 80GB HBM3 · driver 580.126.09"
        );
        assert_eq!(
            node(None).one_line(),
            "NVIDIA H100 80GB HBM3 · driver 580.126.09"
        );
    }

    /// 2026-09-26: `gpu_count` round-trips through JSON, and an unmeasured
    /// count is absent from it.
    #[test]
    fn the_gpu_count_round_trips_and_stays_out_of_the_json_when_unmeasured() {
        let eight = Hardware {
            gpu: "NVIDIA H200".into(),
            gpu_count: Some(8),
            ..Hardware::default()
        };
        let json = serde_json::to_string(&eight).unwrap();
        assert_eq!(
            serde_json::from_str::<Hardware>(&json).unwrap().gpu_count,
            Some(8)
        );
        let unmeasured = Hardware {
            gpu: "NVIDIA GB10".into(),
            ..Hardware::default()
        };
        let json = serde_json::to_string(&unmeasured).unwrap();
        assert!(!json.contains("gpu_count"), "{json}");
        assert_eq!(
            serde_json::from_str::<Hardware>(&json).unwrap().gpu_count,
            None
        );
    }
}

#[cfg(test)]
#[path = "hardware_gate_key_tests.rs"]
mod gate_key_tests;
