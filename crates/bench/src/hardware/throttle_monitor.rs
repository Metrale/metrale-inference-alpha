// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: How much of each recent interval the clocks spent throttled,
//! from successive [`ThrottleCounters`] reads (used by the TUI, `tui/data/thermal.rs`).
//!
//! Owner: bench hardware.
//! Invariants:
//! - A window is never reported for the first read or for a clock that did not advance.
//! - A thermal counter that went backwards makes the window's `thermal_frac` `None`
//!   and `thermal_active` false, never a number.
//!
//! The counters are cumulative, so one read says how much the box has ever
//! throttled; the difference of two says what fraction of the interval the clocks
//! were held. Sustained throttling and flapping in and out of it are different
//! failures, so the monitor reports the fraction per window and the number of
//! on/off transitions across its history ([`ThrottleMonitor::transitions`]).
//!
//! SW power cap is not counted as throttling here, for the reason given on
//! [`super::state::ThrottleActive::thermal`]; it has its own field, `power_cap_frac`.

use super::state::ThrottleCounters;

/// 2026-09-26: One differenced observation of the throttle counters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThrottleWindow {
    /// 2026-09-26: Time between the two reads, ms.
    pub window_ms: u64,
    /// 2026-09-26: Fraction of the window (clamped to 0.0..=1.0) the clocks were
    /// held by a thermal reason: SW thermal, HW thermal or HW power brake. `None`
    /// when none of the three was readable or one went backwards.
    ///
    /// A lower bound: the three counters can overlap, so the union lies between
    /// the largest single counter and their sum, and the largest is reported, so
    /// the figure never overstates throttling.
    pub thermal_frac: Option<f64>,
    /// 2026-09-26: Fraction of the window spent under the SW power cap, kept apart
    /// from `thermal_frac` (see the module docs). `None` when unreadable or when
    /// its counter went backwards.
    pub power_cap_frac: Option<f64>,
    /// 2026-09-26: True when a thermal counter advanced in this window and none
    /// went backwards.
    pub thermal_active: bool,
}

/// 2026-09-26: Rolling detector over successive counter reads.
///
/// Holds the previous read; each [`observe`](Self::observe) returns the window
/// between the two. The first call after construction returns `None`: one read
/// cannot be differenced, and a zero would read as "no throttling" rather than
/// "not known yet".
#[derive(Debug, Default, Clone)]
pub struct ThrottleMonitor {
    prev: Option<(u64, ThrottleCounters)>,
    /// 2026-09-26: Recent `thermal_active` flags, oldest first, for the transition count.
    history: Vec<bool>,
}

/// 2026-09-26: How many windows of `thermal_active` history to keep.
const HISTORY: usize = 32;

impl ThrottleMonitor {
    pub fn new() -> Self {
        Self::default()
    }

    /// 2026-09-26: Difference `counters` (read at `now_ms`) against the previous read.
    ///
    /// Returns `None` on the first read and when `now_ms` is not after the previous
    /// read; in the second case the previous read is kept. A counter that went
    /// backwards (a driver reload or GPU reset) still yields a window, with the
    /// fraction it feeds `None` rather than a negative or saturated number.
    pub fn observe(&mut self, now_ms: u64, counters: ThrottleCounters) -> Option<ThrottleWindow> {
        let out = match &self.prev {
            None => None,
            Some((prev_ms, prev)) => {
                let window_ms = now_ms.checked_sub(*prev_ms).filter(|d| *d > 0)?;
                // 2026-09-26: `Ok(None)`: the counter was not reported on one read.
                // `Ok(Some)`: a usable delta. `Err(())`: it went backwards.
                //
                // Kept distinct: folding "went backwards" into `None` and taking
                // the max over the rest would report Some(0.0), "not throttling",
                // from a reset.
                type Delta = Result<Option<u64>, ()>;
                let delta = |now: Option<u64>, before: Option<u64>| -> Delta {
                    match (now, before) {
                        (Some(n), Some(b)) => n.checked_sub(b).map(Some).ok_or(()),
                        _ => Ok(None),
                    }
                };
                let window_us = (window_ms as f64) * 1000.0;
                let frac = |us: Option<u64>| us.map(|us| (us as f64 / window_us).clamp(0.0, 1.0));

                let sw_t = delta(counters.sw_thermal_us, prev.sw_thermal_us);
                let hw_t = delta(counters.hw_thermal_us, prev.hw_thermal_us);
                let brake = delta(counters.hw_power_brake_us, prev.hw_power_brake_us);
                let cap = delta(counters.sw_power_cap_us, prev.sw_power_cap_us);

                // 2026-09-26: Any thermal counter going backwards makes the whole
                // thermal reading unknown.
                let thermal_us: Option<u64> = match (sw_t, hw_t, brake) {
                    (Err(()), _, _) | (_, Err(()), _) | (_, _, Err(())) => None,
                    (Ok(a), Ok(b), Ok(c)) => {
                        // 2026-09-26: Max, not sum: the three can overlap.
                        [a, b, c].into_iter().flatten().max()
                    }
                };
                let thermal_reset = matches!((sw_t, hw_t, brake), (Err(()), _, _))
                    || matches!((sw_t, hw_t, brake), (_, Err(()), _))
                    || matches!((sw_t, hw_t, brake), (_, _, Err(())));
                Some(ThrottleWindow {
                    window_ms,
                    thermal_frac: if thermal_reset {
                        None
                    } else {
                        frac(thermal_us)
                    },
                    power_cap_frac: frac(cap.unwrap_or(None)),
                    thermal_active: !thermal_reset && thermal_us.is_some_and(|us| us > 0),
                })
            }
        };
        if let Some(w) = out {
            self.history.push(w.thermal_active);
            if self.history.len() > HISTORY {
                self.history.remove(0);
            }
        }
        self.prev = Some((now_ms, counters));
        out
    }

    /// 2026-09-26: Times `thermal_active` flipped across the retained history.
    ///
    /// Separate from the fraction: a part pinned at full throttle has a high
    /// fraction and zero transitions, unlike one that flaps every window.
    pub fn transitions(&self) -> usize {
        self.history.windows(2).filter(|w| w[0] != w[1]).count()
    }

    /// 2026-09-26: Windows retained for [`transitions`](Self::transitions), so a
    /// caller can tell "0 transitions over 30 windows" from "0 over 1".
    pub fn samples(&self) -> usize {
        self.history.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn counters(sw_t: u64, hw_t: u64, brake: u64, cap: u64) -> ThrottleCounters {
        ThrottleCounters {
            sw_thermal_us: Some(sw_t),
            hw_thermal_us: Some(hw_t),
            hw_power_brake_us: Some(brake),
            sw_power_cap_us: Some(cap),
            sync_boost_us: None,
        }
    }

    /// 2026-09-26: One read cannot be differenced, so the first observation is
    /// `None`, not 0.0.
    #[test]
    fn the_first_sample_is_unknown_not_zero() {
        let mut m = ThrottleMonitor::new();
        assert_eq!(m.observe(1_000, counters(0, 0, 0, 0)), None);
    }

    #[test]
    fn half_a_window_of_hw_thermal_reads_as_half() {
        let mut m = ThrottleMonitor::new();
        m.observe(1_000, counters(0, 0, 0, 0));
        let w = m.observe(2_000, counters(0, 500_000, 0, 0)).unwrap();
        assert_eq!(w.window_ms, 1_000);
        assert_eq!(w.thermal_frac, Some(0.5));
        assert!(w.thermal_active);
    }

    /// 2026-09-26: The three thermal counters can overlap, so the union lies
    /// between max and sum; the max is reported.
    #[test]
    fn overlapping_thermal_reasons_take_the_max_not_the_sum() {
        let mut m = ThrottleMonitor::new();
        m.observe(0, counters(0, 0, 0, 0));
        let w = m
            .observe(1_000, counters(400_000, 300_000, 200_000, 0))
            .unwrap();
        // 2026-09-26: The sum would be 0.9; the max is 0.4.
        assert_eq!(w.thermal_frac, Some(0.4));
    }

    /// 2026-09-26: SW power cap time goes to `power_cap_frac`, never to the
    /// thermal fraction.
    #[test]
    fn sw_power_cap_never_counts_as_thermal() {
        let mut m = ThrottleMonitor::new();
        m.observe(0, counters(0, 0, 0, 0));
        let w = m.observe(1_000, counters(0, 0, 0, 1_000_000)).unwrap();
        assert_eq!(w.thermal_frac, Some(0.0));
        assert!(!w.thermal_active);
        assert_eq!(w.power_cap_frac, Some(1.0));
    }

    /// 2026-09-26: A counter that went backwards (a driver reload or GPU reset)
    /// makes the thermal fraction unknown rather than saturated.
    #[test]
    fn counters_going_backwards_report_unknown_rather_than_garbage() {
        let mut m = ThrottleMonitor::new();
        m.observe(0, counters(0, 900_000, 0, 0));
        let w = m.observe(1_000, counters(0, 10_000, 0, 0)).unwrap();
        assert_eq!(w.thermal_frac, None, "must not saturate on a counter reset");
        assert!(!w.thermal_active);
    }

    #[test]
    fn a_non_advancing_clock_yields_no_window() {
        let mut m = ThrottleMonitor::new();
        m.observe(5_000, counters(0, 0, 0, 0));
        assert_eq!(m.observe(5_000, counters(0, 100, 0, 0)), None);
    }

    /// 2026-09-26: Pinned throttling and flapping both look bad by fraction; only
    /// flapping registers transitions.
    #[test]
    fn pinned_throttling_and_flapping_are_told_apart() {
        let mut pinned = ThrottleMonitor::new();
        let mut flapping = ThrottleMonitor::new();
        let (mut p_us, mut f_us) = (0u64, 0u64);
        for i in 0..10u64 {
            p_us += 1_000_000;
            pinned.observe(i * 1_000, counters(0, p_us, 0, 0));
            if i % 2 == 0 {
                f_us += 1_000_000;
            }
            flapping.observe(i * 1_000, counters(0, f_us, 0, 0));
        }
        assert_eq!(pinned.transitions(), 0, "pinned throttling does not flap");
        assert!(
            flapping.transitions() >= 6,
            "flapping must register transitions, got {}",
            flapping.transitions()
        );
    }

    /// 2026-09-26: `transitions()` needs its denominator: zero over one window
    /// means nothing, zero over thirty means stable.
    #[test]
    fn sample_count_is_exposed_so_zero_transitions_is_interpretable() {
        let mut m = ThrottleMonitor::new();
        assert_eq!(m.samples(), 0);
        m.observe(0, counters(0, 0, 0, 0));
        assert_eq!(
            m.samples(),
            0,
            "the undifferenced first sample is not history"
        );
        m.observe(1_000, counters(0, 0, 0, 0));
        assert_eq!(m.samples(), 1);
    }
}
