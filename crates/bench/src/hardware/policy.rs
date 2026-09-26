// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The hardware verdicts: may a benchmark start ([`precheck`]), and may
//! the numbers of a finished run be believed ([`postcheck`]).
//!
//! Owner: bench hardware.
//! Invariants:
//! - No I/O; the only environment read is [`PolicyOptions::from_env`].
//! - A Correctness run is never refused, and a Speed run is refused only while the
//!   kill switch is off.
//! - [`Validity::Valid`] only when all three thermal counters were read on both
//!   captures and none advanced.
//!
//! # Record first, gate second
//!
//! The absolute-temperature ceilings warn by default and refuse only under
//! `METRALE_HW_TEMP_GATE=1`: they are calibrations, and a ceiling that refuses a
//! healthy box teaches operators to set the kill switch. The two checks that need
//! no threshold refuse a Speed run by default: a thermal throttle reason asserted
//! before the run, and more than [`MAX_FOREIGN_COMPUTE_APPS`] foreign GPU compute
//! processes. Everything is recorded either way.
//!
//! # Sensitivity, not benchmark id
//!
//! The policy keys on [`Sensitivity`], which every benchmark declares on its
//! [`crate::BenchmarkDescriptor`]. There is no list of benchmark ids here: the
//! registry is the one source of which benchmarks exist.

use serde::{Deserialize, Serialize};

use super::state::{HardwareState, HardwareStateDelta};

/// 2026-09-26: Whether this benchmark's number is a speed number.
///
/// The split is about what thermal state can corrupt: a throttled box produces a
/// slower wall time and the same tool call. So a hot box may refuse a Speed run
/// and never refuses a Correctness run.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Sensitivity {
    /// 2026-09-26: Wall time, TTFT, TPOT, tok/s, or a Σwall bound.
    Speed,
    /// 2026-09-26: Accuracy, fidelity, state integrity: hardware state is recorded,
    /// never gated on.
    Correctness,
}

/// 2026-09-26: The pre-run decision.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Decision {
    /// 2026-09-26: No concern was raised. An unreadable throttle-reason list, an
    /// unlistable compute-process list or undeclared ceilings yield
    /// [`Decision::Warn`] instead; an unreadable temperature raises no concern.
    Proceed,
    /// 2026-09-26: Recorded, not blocking.
    Warn,
    /// 2026-09-26: The run must not start.
    Refuse,
}

/// 2026-09-26: Whether the numbers a completed run produced may be believed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Validity {
    /// 2026-09-26: Speed-sensitive; all three thermal counters were read on both
    /// captures and none advanced.
    Valid,
    /// 2026-09-26: Speed-sensitive; no readable counter advanced, but at least one
    /// could not be read on both captures. Not a pass.
    Unknown,
    /// 2026-09-26: Speed-sensitive, and a thermal counter advanced during the run.
    Invalid,
    /// 2026-09-26: Correctness run: thermal state does not bear on the number.
    NotApplicable,
}

/// 2026-09-26: A verdict plus every concern that fed it, gating or not.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Precheck {
    pub decision: Decision,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub concerns: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Postcheck {
    pub validity: Validity,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub concerns: Vec<String>,
}

/// 2026-09-26: The two operator switches, read from the environment only in
/// [`PolicyOptions::from_env`], and the box class's temperature ceilings.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PolicyOptions {
    /// 2026-09-26: `METRALE_NO_HW_PRECHECK=1`: never refuse. Every concern is still
    /// recorded, a line saying the refusal was suppressed is added, and
    /// [`postcheck`], which ignores the options, still marks a throttled run invalid.
    pub kill_switch: bool,
    /// 2026-09-26: `METRALE_HW_TEMP_GATE=1`: an exceeded temperature ceiling refuses
    /// instead of warning.
    pub absolute_temp_gate: bool,
    /// 2026-09-26: The box class's temperature ceilings (`kernels/<hw>/HARDWARE.toml`
    /// `[benchmarks.limits.thermal]`: `gpu_ceiling_c`, `chassis_park_c`). `None`
    /// (a class that declares none, or a caller without a repository) adds a
    /// concern that the temperatures were not judged.
    pub ceilings: Option<TempCeilings>,
}

/// 2026-09-26: The two absolute-temperature lines a pre-run capture is judged against.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TempCeilings {
    /// 2026-09-26: GPU die (`temperature.gpu`), °C.
    pub gpu_c: f64,
    /// 2026-09-26: Hottest chassis zone, °C.
    pub chassis_c: f64,
}

impl TempCeilings {
    /// 2026-09-26: The ceilings a class declares.
    #[must_use]
    pub fn of(thermal: &super::limits::ThermalEnvelope) -> Self {
        Self {
            gpu_c: thermal.gpu_ceiling_c,
            chassis_c: thermal.chassis_park_c,
        }
    }
}

/// 2026-09-26: Env var that suppresses refusal. Read only by [`PolicyOptions::from_env`].
pub const KILL_SWITCH_ENV: &str = "METRALE_NO_HW_PRECHECK";
/// 2026-09-26: Env var that makes the absolute-temperature ceilings refuse.
pub const TEMP_GATE_ENV: &str = "METRALE_HW_TEMP_GATE";

/// 2026-09-26: GPU compute processes, other than this one, that a Speed run
/// tolerates: one, the server of the model under test. A second means something
/// else is resident.
pub const MAX_FOREIGN_COMPUTE_APPS: usize = 1;

impl PolicyOptions {
    pub fn from_env() -> Self {
        Self {
            kill_switch: flag(KILL_SWITCH_ENV),
            absolute_temp_gate: flag(TEMP_GATE_ENV),
            ceilings: None,
        }
    }
}

/// 2026-09-26: `1`/`true`/`yes`, trimmed and case-insensitive. Anything else,
/// including an empty value, is off.
fn flag(name: &str) -> bool {
    std::env::var(name)
        .is_ok_and(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes"))
}

/// 2026-09-26: Worst wins, so a Refuse cannot be argued down by a later Proceed.
fn worst(a: Decision, b: Decision) -> Decision {
    match (a, b) {
        (Decision::Refuse, _) | (_, Decision::Refuse) => Decision::Refuse,
        (Decision::Warn, _) | (_, Decision::Warn) => Decision::Warn,
        _ => Decision::Proceed,
    }
}

/// 2026-09-26: May this benchmark start?
///
/// A Correctness run always may: a Refuse is downgraded to [`Decision::Warn`], with
/// every concern kept and one more that says so.
pub fn precheck(
    sensitivity: Sensitivity,
    before: &HardwareState,
    options: PolicyOptions,
) -> Precheck {
    let mut concerns = Vec::new();
    let mut decision = Decision::Proceed;
    let mut gate = |d: Decision, why: String| {
        concerns.push(why);
        decision = worst(decision, d);
    };

    // 2026-09-26: Refuses by default: a thermal reason is asserted now.
    match before.throttle_active.thermal() {
        Some(true) => gate(
            Decision::Refuse,
            "a thermal throttle reason is ACTIVE before the run started".to_string(),
        ),
        Some(false) => {}
        None => gate(
            Decision::Warn,
            "throttle reasons could not be read — the box is not known to be unthrottled"
                .to_string(),
        ),
    }

    // 2026-09-26: Refuses by default: another resident compute process.
    match before.foreign_compute_apps() {
        Some(n) if n > MAX_FOREIGN_COMPUTE_APPS => gate(
            Decision::Refuse,
            format!(
                "{n} GPU compute processes other than this one (at most \
                 {MAX_FOREIGN_COMPUTE_APPS} expected — the model under test)"
            ),
        ),
        Some(_) => {}
        None => gate(
            Decision::Warn,
            "GPU compute processes could not be listed — contention is unknown".to_string(),
        ),
    }

    // 2026-09-26: Warns by default; refuses only under METRALE_HW_TEMP_GATE=1.
    let temp_level = if options.absolute_temp_gate {
        Decision::Refuse
    } else {
        Decision::Warn
    };
    match options.ceilings {
        Some(c) => {
            if let Some(t) = before.gpu_temp_c.filter(|t| *t > c.gpu_c) {
                gate(
                    temp_level,
                    format!("GPU die {t:.0} °C is above the {:.0} °C ceiling", c.gpu_c),
                );
            }
            if let Some(t) = before.hottest_chassis_c().filter(|t| *t > c.chassis_c) {
                gate(
                    temp_level,
                    format!(
                        "hottest chassis zone {t:.0} °C is above the {:.0} °C ceiling",
                        c.chassis_c
                    ),
                );
            }
        }
        // 2026-09-26: Warns, never refuses: a missing ceiling is not a cool box.
        None => gate(
            Decision::Warn,
            "no temperature ceilings are declared for this box class, so the capture's \
             temperatures were recorded and not judged"
                .to_string(),
        ),
    }

    // 2026-09-26: A Correctness run records everything above and blocks on none of it.
    if sensitivity == Sensitivity::Correctness && decision == Decision::Refuse {
        decision = Decision::Warn;
        concerns.push(
            "correctness gate — recorded and proceeding; accuracy is not thermally sensitive"
                .to_string(),
        );
    }
    if options.kill_switch && decision == Decision::Refuse {
        decision = Decision::Warn;
        concerns.push(format!(
            "{KILL_SWITCH_ENV}=1 — REFUSAL SUPPRESSED BY OPERATOR. The concerns above stand and \
             are recorded with the run; the kill switch only lets it start."
        ));
    }
    Precheck { decision, concerns }
}

/// 2026-09-26: May a finished run's numbers be believed? Reads the delta, not the
/// absolute state, and ignores `_options`.
pub fn postcheck(
    sensitivity: Sensitivity,
    delta: &HardwareStateDelta,
    _options: PolicyOptions,
) -> Postcheck {
    let mut concerns = Vec::new();
    if let Some(f) = delta.thermal_throttle_fraction().filter(|f| *f > 0.0) {
        // 2026-09-26: A partial sum is a floor, so it is printed as one.
        concerns.push(if delta.thermal_counters_complete() {
            format!("thermally throttled for {:.2}% of the run", f * 100.0)
        } else {
            format!(
                "thermally throttled for AT LEAST {:.2}% of the run — one or more throttle \
                 counters were unreadable, so the true figure can only be higher",
                f * 100.0
            )
        });
    }
    if let Some(d) = delta.hottest_chassis_delta_c {
        concerns.push(format!("hottest chassis zone moved {d:+.0} °C"));
    }

    if sensitivity == Sensitivity::Correctness {
        return Postcheck {
            validity: Validity::NotApplicable,
            concerns,
        };
    }
    let validity = match delta.thermal_throttle_advanced() {
        Some(true) => {
            // 2026-09-26: An unread counter prints as `unreadable`, never as a
            // measured 0 µs.
            fn counter(v: Option<u64>) -> String {
                v.map_or_else(|| "unreadable".to_string(), |us| format!("{us} µs"))
            }
            concerns.push(format!(
                "a thermal throttle counter advanced during the run (sw {}, hw {}, \
                 brake {}) — this speed number is not comparable",
                counter(delta.sw_thermal_us),
                counter(delta.hw_thermal_us),
                counter(delta.hw_power_brake_us),
            ));
            Validity::Invalid
        }
        Some(false) => Validity::Valid,
        None => {
            concerns.push(
                "throttle counters were unreadable on at least one capture — this run is not \
                 known to have been unthrottled"
                    .to_string(),
            );
            Validity::Unknown
        }
    };
    Postcheck { validity, concerns }
}

#[cfg(test)]
#[path = "policy_tests.rs"]
mod tests;
