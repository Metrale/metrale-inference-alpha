// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Whether two boxes count as one box for Speed-class numbers: same
//! GPU name and driver major, clock ceiling and memory within the class's spreads,
//! no thermal throttle reason asserted, and hottest chassis zones within its delta.
//!
//! Owner: bench hardware.
//! Invariants:
//! - [`equivalent`] returns `Ok` only when every field but `postcheck_valid` is
//!   reported on both sides; a field one side lacks is [`Mismatch::Undecidable`].
//!
//! `gate::agreement` applies [`equivalent`] to the Speed records of different
//! signers, and that is the verdict. `met bench certify` calls it only to report
//! whether its nodes look alike (`remote/schedule.rs` `speed_mode` runs the Speed
//! class on one node either way).

use super::state::HardwareState;
use super::{Hardware, HardwareStateReport};
use crate::gate::GateRecord;
use serde::{Deserialize, Serialize};

/// 2026-09-26: What decides equivalence, read off a record or a live box.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HardwareFingerprint {
    /// 2026-09-26: The accelerator name as the driver reports it (`NVIDIA GB10`).
    pub gpu: String,
    /// 2026-09-26: The driver's major version (`580` of `580.95.05`).
    pub driver_major: Option<u32>,
    /// 2026-09-26: The box's own SM clock ceiling (`clocks.max.sm`), MHz.
    pub sm_clock_max_mhz: Option<f64>,
    /// 2026-09-26: `MemTotal` from `/proc/meminfo`, kB.
    pub mem_total_kb: Option<u64>,
    /// 2026-09-26: [`super::state::ThrottleActive::thermal`] at capture.
    pub thermal_alert: Option<bool>,
    /// 2026-09-26: The hottest chassis zone at capture, °C.
    pub hottest_chassis_c: Option<f64>,
    /// 2026-09-26: For a record: whether its postcheck was [`super::policy::Validity::Valid`].
    /// `None` for a live box, or a record without a report or postcheck.
    pub postcheck_valid: Option<bool>,
}

impl HardwareFingerprint {
    /// 2026-09-26: From a record: static fields off `hardware`, live fields off the
    /// `before` capture, validity off the postcheck.
    pub fn from_record(record: &GateRecord) -> Self {
        let report: Option<&HardwareStateReport> = record.hardware_state.as_ref();
        let before = report.map(|r| &r.before);
        let mut fp = Self::from_parts(&record.hardware, before);
        fp.postcheck_valid = report.and_then(|r| {
            r.postcheck
                .as_ref()
                .map(|p| p.validity == super::policy::Validity::Valid)
        });
        fp
    }

    /// 2026-09-26: From a live box, before anything runs.
    pub fn from_live(hardware: &Hardware, state: &HardwareState) -> Self {
        Self::from_parts(hardware, Some(state))
    }

    fn from_parts(hardware: &Hardware, state: Option<&HardwareState>) -> Self {
        Self {
            gpu: hardware.gpu.clone(),
            driver_major: driver_major(&hardware.driver),
            sm_clock_max_mhz: state.and_then(|s| s.sm_clock_max_mhz),
            mem_total_kb: state.and_then(|s| s.mem_total_kb),
            thermal_alert: state.and_then(|s| s.throttle_active.thermal()),
            hottest_chassis_c: state.and_then(HardwareState::hottest_chassis_c),
            postcheck_valid: None,
        }
    }
}

/// 2026-09-26: `580.95.05` → `580`. Empty or non-numeric → `None`.
pub fn driver_major(driver: &str) -> Option<u32> {
    driver.split('.').next()?.trim().parse().ok()
}

/// 2026-09-26: The tolerances, read from the class's `HARDWARE.toml` by [`EquivalencePolicy::speed`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EquivalencePolicy {
    /// 2026-09-26: Largest `|a - b| / max(a, b)` on the clock ceiling.
    pub clock_spread: f64,
    /// 2026-09-26: Largest `|a - b| / max(a, b)` on memory.
    pub mem_spread: f64,
    /// 2026-09-26: Largest `|a - b|` on the hottest chassis zone, °C.
    pub chassis_delta_c: f64,
}

impl EquivalencePolicy {
    /// 2026-09-26: The policy for one hardware class, from its declared limits
    /// (`kernels/<hw>/HARDWARE.toml` `[benchmarks.limits.equivalence]` and
    /// `.thermal`): clock and memory spreads and the chassis delta, on GB10 1 %,
    /// 5 % and 15 °C.
    #[must_use]
    pub fn speed(limits: &super::limits::Limits) -> Self {
        Self {
            clock_spread: limits.equivalence.clock_spread,
            mem_spread: limits.equivalence.mem_spread,
            chassis_delta_c: limits.thermal.chassis_equivalence_delta_c,
        }
    }

    /// 2026-09-26: The policy for `hardware`; `None` when it declares no limits.
    ///
    /// # Errors
    /// Whatever [`super::limits::limits`] refuses: a missing, malformed or
    /// self-contradicting `HARDWARE.toml`.
    pub fn speed_for(root: &std::path::Path, hardware: &str) -> anyhow::Result<Option<Self>> {
        Ok(super::limits::limits(root, hardware)?.map(|l| Self::speed(&l)))
    }
}

/// 2026-09-26: Why two fingerprints are not one box.
#[derive(Clone, Debug, PartialEq)]
pub enum Mismatch {
    Gpu(String, String),
    DriverMajor(u32, u32),
    ClockSpread {
        a: f64,
        b: f64,
        limit: f64,
    },
    MemSpread {
        a: u64,
        b: u64,
        limit: f64,
    },
    ChassisDelta {
        a: f64,
        b: f64,
        limit: f64,
    },
    /// 2026-09-26: One side (or both) had a thermal reason asserted.
    ThermalAlert {
        a: bool,
        b: bool,
    },
    /// 2026-09-26: A record's postcheck was not valid.
    PostcheckInvalid,
    /// 2026-09-26: A field one side did not report, named so the operator can fix
    /// the capture rather than guess.
    Undecidable(&'static str),
}

impl std::fmt::Display for Mismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Gpu(a, b) => write!(f, "gpu {a:?} vs {b:?}"),
            Self::DriverMajor(a, b) => write!(f, "driver major {a} vs {b}"),
            Self::ClockSpread { a, b, limit } => write!(
                f,
                "clock ceiling {a:.0} vs {b:.0} MHz (limit {:.1} %)",
                limit * 100.0
            ),
            Self::MemSpread { a, b, limit } => write!(
                f,
                "memory {} vs {} MiB (limit {:.0} %)",
                a >> 10,
                b >> 10,
                limit * 100.0
            ),
            Self::ChassisDelta { a, b, limit } => {
                write!(f, "chassis {a:.0} vs {b:.0} °C (limit {limit:.0} °C)")
            }
            Self::ThermalAlert { a, b } => write!(
                f,
                "thermal throttle asserted ({})",
                match (a, b) {
                    (true, true) => "both",
                    (true, false) => "first",
                    _ => "second",
                }
            ),
            Self::PostcheckInvalid => write!(f, "a post-run hardware check was not valid"),
            Self::Undecidable(field) => write!(f, "{field} not reported on one side"),
        }
    }
}

fn spread(a: f64, b: f64) -> f64 {
    let m = a.max(b);
    if m <= 0.0 { 0.0 } else { (a - b).abs() / m }
}

/// 2026-09-26: Are `a` and `b` one box for Speed-class numbers?
///
/// Every mismatch is returned, not the first. A postcheck that is present and
/// not valid on either side is a mismatch; an absent one (a live box) is not.
pub fn equivalent(
    a: &HardwareFingerprint,
    b: &HardwareFingerprint,
    p: &EquivalencePolicy,
) -> Result<(), Vec<Mismatch>> {
    let mut out = Vec::new();
    if a.gpu.is_empty() || b.gpu.is_empty() {
        out.push(Mismatch::Undecidable("gpu"));
    } else if a.gpu != b.gpu {
        out.push(Mismatch::Gpu(a.gpu.clone(), b.gpu.clone()));
    }
    match (a.driver_major, b.driver_major) {
        (Some(x), Some(y)) if x != y => out.push(Mismatch::DriverMajor(x, y)),
        (Some(_), Some(_)) => {}
        _ => out.push(Mismatch::Undecidable("driver")),
    }
    match (a.sm_clock_max_mhz, b.sm_clock_max_mhz) {
        (Some(x), Some(y)) => {
            if spread(x, y) > p.clock_spread {
                out.push(Mismatch::ClockSpread {
                    a: x,
                    b: y,
                    limit: p.clock_spread,
                });
            }
        }
        _ => out.push(Mismatch::Undecidable("sm_clock_max_mhz")),
    }
    match (a.mem_total_kb, b.mem_total_kb) {
        (Some(x), Some(y)) => {
            if spread(x as f64, y as f64) > p.mem_spread {
                out.push(Mismatch::MemSpread {
                    a: x,
                    b: y,
                    limit: p.mem_spread,
                });
            }
        }
        _ => out.push(Mismatch::Undecidable("mem_total_kb")),
    }
    match (a.thermal_alert, b.thermal_alert) {
        (Some(x), Some(y)) => {
            if x || y {
                out.push(Mismatch::ThermalAlert { a: x, b: y });
            }
        }
        _ => out.push(Mismatch::Undecidable("throttle reasons")),
    }
    match (a.hottest_chassis_c, b.hottest_chassis_c) {
        (Some(x), Some(y)) => {
            if (x - y).abs() > p.chassis_delta_c {
                out.push(Mismatch::ChassisDelta {
                    a: x,
                    b: y,
                    limit: p.chassis_delta_c,
                });
            }
        }
        _ => out.push(Mismatch::Undecidable("chassis temperature")),
    }
    if a.postcheck_valid == Some(false) || b.postcheck_valid == Some(false) {
        out.push(Mismatch::PostcheckInvalid);
    }
    if out.is_empty() { Ok(()) } else { Err(out) }
}

#[cfg(test)]
#[path = "equivalence_tests.rs"]
mod equivalence_tests;
