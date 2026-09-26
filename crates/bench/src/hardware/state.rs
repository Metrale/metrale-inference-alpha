// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: [`HardwareState`], what the box was doing at one instant, and
//! [`HardwareStateDelta`], what changed between the captures before and after a run.
//!
//! Owner: bench hardware.
//! Invariants:
//! - A reading that does not parse is `None`, never a default number (the parsers
//!   in [`super::parse`]).
//! - A throttle counter that went backwards between captures yields `None` in the
//!   delta, never zero.
//!
//! [`super::Hardware`] answers "which box"; this answers "what state was that box in".
//! Two boxes with identical fingerprints can differ in chassis temperature, throttle
//! time and clock under load, and only the state shows it.
//!
//! Memory comes from `/proc/meminfo`: nvidia-smi reports GB10's framebuffer usage as
//! `N/A` (`fixtures/gb10_performance.txt`). `Cached` is kept beside `MemAvailable`
//! rather than folded into it.

use serde::{Deserialize, Serialize};

/// 2026-09-26: One `/sys/class/thermal/thermal_zone*` reading.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ThermalZone {
    /// 2026-09-26: The zone's `type` file, trimmed. Empty when unreadable.
    pub name: String,
    pub temp_c: f64,
}

/// 2026-09-26: Cumulative throttle time per reason, in the microseconds
/// `nvidia-smi -q -d PERFORMANCE` reports under `Clocks Event Reasons Counters`.
///
/// These are counters, not flags: the useful reading is the difference across a
/// run (see [`HardwareStateDelta`]), because a box can start cool and throttle in
/// the middle. `None` means the reason was not reported; `Some(0)` means it was
/// reported as never having fired.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ThrottleCounters {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sw_power_cap_us: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sw_thermal_us: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hw_thermal_us: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hw_power_brake_us: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sync_boost_us: Option<u64>,
}

/// 2026-09-26: Throttle reasons asserted at the instant of capture.
///
/// The pre-run check ([`super::policy::precheck`]) reads these; the post-run
/// check reads the counters' delta.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ThrottleActive {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sw_power_cap: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sw_thermal: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hw_thermal: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hw_power_brake: Option<bool>,
}

impl ThrottleActive {
    /// 2026-09-26: True when SW thermal, HW thermal or the HW power brake is
    /// asserted. `None` when none of the three was reported, so the caller can
    /// tell "not throttling" from "no idea".
    ///
    /// SW power cap is excluded: a power-limited part accrues it in normal
    /// running (16,130 s on the cool box's counter in
    /// `fixtures/gb10_performance.txt`), so it says nothing about a thermal fault.
    pub fn thermal(&self) -> Option<bool> {
        match (self.sw_thermal, self.hw_thermal, self.hw_power_brake) {
            (None, None, None) => None,
            (sw, hw, brake) => {
                Some(sw.unwrap_or(false) || hw.unwrap_or(false) || brake.unwrap_or(false))
            }
        }
    }
}

/// 2026-09-26: One GPU-resident compute process, as `--query-compute-apps` reports it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GpuComputeApp {
    pub pid: u32,
    pub name: String,
    /// 2026-09-26: `None` when the memory cell does not parse, such as `[N/A]`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub used_mib: Option<u64>,
}

/// 2026-09-26: Which individual box this is. [`super::Hardware::gate_key`] gives
/// the silicon class (`"gb10"`) that baselines are keyed by; this tells two boxes
/// of one class apart.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct MachineIdentity {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hostname: Option<String>,
    /// 2026-09-26: `/etc/machine-id`, which a hostname change does not alter.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub machine_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub driver: Option<String>,
}

impl MachineIdentity {
    /// 2026-09-26: `"gb10@dgx1"`: silicon class, then hostname.
    ///
    /// An absent half reads `"unknown"` rather than being dropped, so two records
    /// with different missing halves never collapse to the same string.
    pub fn perf_class(&self) -> String {
        let class = self
            .gpu
            .as_deref()
            .map(|g| super::Hardware {
                gpu: g.to_string(),
                ..super::Hardware::default()
            })
            .map(|h| h.gate_key())
            .unwrap_or_else(|| "unknown".to_string());
        let host = self.hostname.as_deref().unwrap_or("unknown");
        format!("{class}@{host}")
    }
}

/// 2026-09-26: Everything the collector could read about the box at one instant.
/// The executor captures it before a run and again when the run's terminal frame
/// is ready; [`HardwareStateDelta`] is the difference.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct HardwareState {
    /// 2026-09-26: Unix seconds; `0` when the system clock reads before the epoch.
    #[serde(default)]
    pub captured_at: u64,
    #[serde(default)]
    pub machine: MachineIdentity,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_temp_c: Option<f64>,
    /// 2026-09-26: Every `/sys/class/thermal` zone whose temperature parsed, in
    /// zone-number order. `None` means the directory could not be read, not
    /// "this box has no zones".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chassis_temps_c: Option<Vec<ThermalZone>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sm_clock_mhz: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sm_clock_max_mhz: Option<f64>,
    #[serde(default)]
    pub throttle_counters: ThrottleCounters,
    #[serde(default)]
    pub throttle_active: ThrottleActive,
    /// 2026-09-26: `MemAvailable` from `/proc/meminfo`, kB.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mem_available_kb: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mem_total_kb: Option<u64>,
    /// 2026-09-26: `Cached` from `/proc/meminfo`, kB.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub page_cache_kb: Option<u64>,
    /// 2026-09-26: The `--query-compute-apps` rows. When that query prints nothing
    /// or fails, `Some(vec![])` if `--list-gpus` answers, else `None`. The
    /// precheck warns on `None` and accepts an empty list.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_compute_apps: Option<Vec<GpuComputeApp>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cpu_governor: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub persistence_mode: Option<bool>,
    /// 2026-09-26: Which collectors answered, from `"nvidia-smi"`, `"procfs"` and
    /// `"sysfs"`. Empty means none did.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub sources: Vec<String>,
}

impl HardwareState {
    /// 2026-09-26: Read the local box; see [`super::collect`]. A source that fails
    /// leaves its fields unset rather than failing the capture.
    pub fn collect() -> Self {
        super::collect::collect()
    }

    /// 2026-09-26: GPU compute processes other than this process, so a server
    /// running in this process is not counted and a separate server process is.
    /// `None` when the list could not be read. The precheck refuses a Speed run
    /// above [`super::policy::MAX_FOREIGN_COMPUTE_APPS`].
    pub fn foreign_compute_apps(&self) -> Option<usize> {
        let own = std::process::id();
        Some(
            self.gpu_compute_apps
                .as_ref()?
                .iter()
                .filter(|a| a.pid != own)
                .count(),
        )
    }

    /// 2026-09-26: The hottest chassis zone; `None` when no zone was read.
    pub fn hottest_chassis_c(&self) -> Option<f64> {
        self.chassis_temps_c
            .as_ref()?
            .iter()
            .map(|z| z.temp_c)
            .fold(None::<f64>, |acc, t| Some(acc.map_or(t, |a| a.max(t))))
    }

    /// 2026-09-26: Current SM clock as a fraction of the box's own maximum;
    /// `None` when either is missing or the maximum is not positive.
    ///
    /// Meaningful only under load: the idle GB10 fixture reads 208 of 3003 MHz
    /// (`fixtures/gb10_query_gpu.csv`).
    pub fn clock_headroom(&self) -> Option<f64> {
        let max = self.sm_clock_max_mhz?;
        (max > 0.0).then(|| self.sm_clock_mhz.map(|c| c / max))?
    }

    /// 2026-09-26: One line for a run log.
    pub fn one_line(&self) -> String {
        let mut parts = vec![self.machine.perf_class()];
        if let Some(t) = self.gpu_temp_c {
            parts.push(format!("gpu {t:.0} °C"));
        }
        if let Some(t) = self.hottest_chassis_c() {
            parts.push(format!("chassis max {t:.0} °C"));
        }
        match (self.sm_clock_mhz, self.sm_clock_max_mhz) {
            (Some(c), Some(m)) => parts.push(format!("sm {c:.0}/{m:.0} MHz")),
            (Some(c), None) => parts.push(format!("sm {c:.0} MHz")),
            _ => {}
        }
        match self.foreign_compute_apps() {
            Some(n) => parts.push(format!("{n} foreign gpu proc")),
            None => parts.push("foreign gpu proc unknown".to_string()),
        }
        if let Some(kb) = self.mem_available_kb {
            parts.push(format!("{:.1} GiB avail", kb as f64 / 1_048_576.0));
        }
        parts.join(" · ")
    }
}

/// 2026-09-26: What changed across the run.
///
/// A box that starts cool and throttles midway has a healthy pre-state; whether a
/// throttle counter advanced while the benchmark ran is visible only in the
/// difference of two captures.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct HardwareStateDelta {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub elapsed_s: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sw_power_cap_us: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sw_thermal_us: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hw_thermal_us: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hw_power_brake_us: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_temp_delta_c: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hottest_chassis_delta_c: Option<f64>,
}

/// 2026-09-26: `after - before`; `None` when either is missing or the counter went
/// backwards (a driver reload mid-run), which is an unusable reading, not zero.
fn advance(before: Option<u64>, after: Option<u64>) -> Option<u64> {
    let (b, a) = (before?, after?);
    // 2026-09-26: `checked_sub`, not `(a >= b).then_some(a - b)`: the argument to
    // `then_some` is evaluated eagerly, so the backwards case would panic in a
    // debug build before the guard runs.
    a.checked_sub(b)
}

impl HardwareStateDelta {
    pub fn between(before: &HardwareState, after: &HardwareState) -> Self {
        let (b, a) = (&before.throttle_counters, &after.throttle_counters);
        Self {
            elapsed_s: after.captured_at.checked_sub(before.captured_at),
            sw_power_cap_us: advance(b.sw_power_cap_us, a.sw_power_cap_us),
            sw_thermal_us: advance(b.sw_thermal_us, a.sw_thermal_us),
            hw_thermal_us: advance(b.hw_thermal_us, a.hw_thermal_us),
            hw_power_brake_us: advance(b.hw_power_brake_us, a.hw_power_brake_us),
            gpu_temp_delta_c: Option::zip(before.gpu_temp_c, after.gpu_temp_c).map(|(b, a)| a - b),
            hottest_chassis_delta_c: Option::zip(
                before.hottest_chassis_c(),
                after.hottest_chassis_c(),
            )
            .map(|(b, a)| a - b),
        }
    }

    /// 2026-09-26: Did a thermal throttle counter (SW thermal, HW thermal, HW power
    /// brake) accumulate time during the run?
    ///
    /// `Some(true)` when any readable one advanced; `Some(false)` only when all
    /// three were readable and none advanced; otherwise `None`, which the policy
    /// renders as unknown. SW power cap is excluded for the reason given on
    /// [`ThrottleActive::thermal`].
    pub fn thermal_throttle_advanced(&self) -> Option<bool> {
        let counters = [
            self.sw_thermal_us,
            self.hw_thermal_us,
            self.hw_power_brake_us,
        ];
        if counters.into_iter().flatten().any(|value| value > 0) {
            return Some(true);
        }
        counters
            .into_iter()
            .all(|value| value == Some(0))
            .then_some(false)
    }

    /// 2026-09-26: True when every counter [`Self::thermal_throttle_fraction`] sums
    /// was readable on both captures.
    ///
    /// When false the fraction is a lower bound, and the caller must say so: an
    /// unreadable counter contributes nothing to the sum.
    pub fn thermal_counters_complete(&self) -> bool {
        self.sw_thermal_us.is_some()
            && self.hw_thermal_us.is_some()
            && self.hw_power_brake_us.is_some()
    }

    /// 2026-09-26: The throttled fraction of the run: the readable thermal
    /// counters' sum over the elapsed time.
    ///
    /// `None` when the run has no positive duration, or when no thermal counter
    /// was readable on both captures. A lower bound when only some were; see
    /// [`Self::thermal_counters_complete`].
    pub fn thermal_throttle_fraction(&self) -> Option<f64> {
        let secs = self.elapsed_s.filter(|s| *s > 0)? as f64;
        let counters = [
            self.sw_thermal_us,
            self.hw_thermal_us,
            self.hw_power_brake_us,
        ];
        if counters.iter().all(Option::is_none) {
            return None;
        }
        let us: u64 = counters.into_iter().flatten().sum();
        Some(us as f64 / 1e6 / secs)
    }
}

#[cfg(test)]
#[path = "state_tests.rs"]
mod tests;
