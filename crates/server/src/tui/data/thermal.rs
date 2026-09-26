// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Thermal and throttle sampling for the Stats section. A detached
//! thread runs `metrale_bench::hardware::collect::collect` (which spawns
//! `nvidia-smi -q -d PERFORMANCE` for the throttle counters) every
//! `SAMPLE_EVERY`, differences the counters with a `ThrottleMonitor`, and
//! stores the result in a mutex the render thread copies from.
//!
//! Owner: server tui.
//! Invariants:
//! - `ThermalAlert::Ok` is returned only when the latest window has a
//!   `thermal_frac`; no window or no fraction is `Unknown`.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use metrale_bench::hardware::throttle_monitor::{ThrottleMonitor, ThrottleWindow};

/// 2026-09-26: How often the background thread re-reads the counters. A
/// collection that takes longer delays the next one.
const SAMPLE_EVERY: Duration = Duration::from_secs(2);

/// 2026-09-26: What the renderer reads. `None` fields mean "not known".
#[derive(Debug, Clone, Copy, Default)]
pub struct ThermalSnapshot {
    pub window: Option<ThrottleWindow>,
    /// 2026-09-26: Flips of the thermal state across the monitor's retained
    /// history (`ThrottleMonitor::transitions`).
    pub transitions: usize,
    /// 2026-09-26: Windows in that history (`ThrottleMonitor::samples`).
    pub samples: usize,
    pub gpu_temp_c: Option<f64>,
    pub sm_clock_mhz: Option<f64>,
    pub sm_clock_max_mhz: Option<f64>,
    /// 2026-09-26: Set once the first collection has been stored, whatever it
    /// found.
    pub have_data: bool,
}

impl ThermalSnapshot {
    /// 2026-09-26: SM clock as a fraction of its maximum, clamped to 0..=1,
    /// when both are known and the maximum is above zero.
    pub fn clock_frac(&self) -> Option<f64> {
        match (self.sm_clock_mhz, self.sm_clock_max_mhz) {
            (Some(now), Some(max)) if max > 0.0 => Some((now / max).clamp(0.0, 1.0)),
            _ => None,
        }
    }
}

/// 2026-09-26: Shared handle. Cloning shares the same snapshot.
#[derive(Debug, Clone, Default)]
pub struct ThermalProbe {
    inner: Arc<Mutex<ThermalSnapshot>>,
}

impl ThermalProbe {
    /// 2026-09-26: Start the sampler thread and return at once. If the thread
    /// cannot be spawned, the snapshot stays at its default.
    pub fn spawn() -> Self {
        let probe = Self::default();
        let sink = Arc::clone(&probe.inner);
        std::thread::Builder::new()
            .name("metrale-thermal".into())
            .spawn(move || {
                let mut monitor = ThrottleMonitor::new();
                loop {
                    let started = Instant::now();
                    let state = metrale_bench::hardware::collect::collect();
                    let now_ms = SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .map(|d| d.as_millis() as u64)
                        .unwrap_or(0);
                    let window = monitor.observe(now_ms, state.throttle_counters);
                    if let Ok(mut slot) = sink.lock() {
                        // 2026-09-26: Keep the previous window when `observe`
                        // returned none: the first sample, or a clock that did
                        // not advance. A counter reset returns a window whose
                        // `thermal_frac` is `None`, which replaces it.
                        if window.is_some() {
                            slot.window = window;
                        }
                        slot.transitions = monitor.transitions();
                        slot.samples = monitor.samples();
                        slot.gpu_temp_c = state.gpu_temp_c;
                        slot.sm_clock_mhz = state.sm_clock_mhz;
                        slot.sm_clock_max_mhz = state.sm_clock_max_mhz;
                        slot.have_data = true;
                    }
                    std::thread::sleep(SAMPLE_EVERY.saturating_sub(started.elapsed()));
                }
            })
            .ok();
        probe
    }

    /// 2026-09-26: Latest snapshot. A poisoned mutex still yields the value it
    /// holds.
    pub fn snapshot(&self) -> ThermalSnapshot {
        self.inner
            .lock()
            .map(|s| *s)
            .unwrap_or_else(|p| *p.into_inner())
    }
}

/// 2026-09-26: Severity the header indicator reflects, ordered by urgency.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ThermalAlert {
    /// 2026-09-26: No window yet, or the window has no `thermal_frac` (no
    /// thermal counters reported, or a counter went backwards).
    Unknown,
    Ok,
    /// 2026-09-26: `thermal_frac` is at least `THROTTLE_WARN_FRAC`.
    Throttling,
    /// 2026-09-26: At least `THRASH_TRANSITIONS` flips over at least
    /// `THRASH_MIN_SAMPLES` samples; checked before `Throttling`.
    Thrashing,
}

/// 2026-09-26: Fraction of a window under thermal throttle that reports
/// `Throttling`.
pub const THROTTLE_WARN_FRAC: f64 = 0.20;

/// 2026-09-26: Thermal-state flips, within the retained history, that count as
/// thrashing. The monitor keeps 32 windows, at `SAMPLE_EVERY` (2 s) apart when
/// collection is fast, so about 64 s; one excursion in and out is two flips.
pub const THRASH_TRANSITIONS: usize = 6;

/// 2026-09-26: Minimum `samples` before `alert` can report `Thrashing`.
pub const THRASH_MIN_SAMPLES: usize = 8;

impl ThermalSnapshot {
    /// 2026-09-26: What the header should show.
    pub fn alert(&self) -> ThermalAlert {
        let Some(w) = self.window else {
            return ThermalAlert::Unknown;
        };
        let Some(frac) = w.thermal_frac else {
            return ThermalAlert::Unknown;
        };
        if self.samples >= THRASH_MIN_SAMPLES && self.transitions >= THRASH_TRANSITIONS {
            return ThermalAlert::Thrashing;
        }
        if frac >= THROTTLE_WARN_FRAC {
            return ThermalAlert::Throttling;
        }
        ThermalAlert::Ok
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snap(frac: Option<f64>, transitions: usize, samples: usize) -> ThermalSnapshot {
        ThermalSnapshot {
            window: Some(ThrottleWindow {
                window_ms: 2_000,
                thermal_frac: frac,
                power_cap_frac: Some(1.0),
                thermal_active: frac.is_some_and(|f| f > 0.0),
            }),
            transitions,
            samples,
            have_data: true,
            ..Default::default()
        }
    }

    #[test]
    fn absent_data_is_unknown_not_ok() {
        assert_eq!(ThermalSnapshot::default().alert(), ThermalAlert::Unknown);
        assert_eq!(snap(None, 0, 20).alert(), ThermalAlert::Unknown);
    }

    #[test]
    fn a_quiet_box_is_ok() {
        assert_eq!(snap(Some(0.01), 0, 20).alert(), ThermalAlert::Ok);
    }

    #[test]
    fn sustained_throttling_warns() {
        assert_eq!(snap(Some(0.55), 0, 20).alert(), ThermalAlert::Throttling);
    }

    /// 2026-09-26: `Thrashing` is checked before the fraction, and orders above
    /// `Throttling`.
    #[test]
    fn flapping_outranks_a_steady_hold() {
        let a = snap(Some(0.05), THRASH_TRANSITIONS, 20).alert();
        assert_eq!(a, ThermalAlert::Thrashing);
        assert!(ThermalAlert::Thrashing > ThermalAlert::Throttling);
    }

    #[test]
    fn thrashing_needs_enough_history_to_be_believable() {
        assert_ne!(
            snap(Some(0.05), THRASH_TRANSITIONS, THRASH_MIN_SAMPLES - 1).alert(),
            ThermalAlert::Thrashing
        );
    }

    /// 2026-09-26: `alert` does not read `power_cap_frac`; `snap` sets it to 1.0.
    #[test]
    fn a_pinned_power_cap_alone_is_still_ok() {
        assert_eq!(snap(Some(0.0), 0, 20).alert(), ThermalAlert::Ok);
    }
}
