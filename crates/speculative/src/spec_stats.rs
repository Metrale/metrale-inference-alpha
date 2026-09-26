// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Per-run speculation counters: draft acceptance and per-position
//! draft matches for the K=2, K=3 and K=4 verify arms, the B1 low-margin
//! gauge, decode timing, and one-shot log latches. Nothing here changes
//! generation.
//!
//! Owner: speculative.
//! Invariants: none beyond the types.

use std::sync::atomic::{AtomicU64, Ordering};

pub const SUMMARY_PERIOD: u64 = 512;

/// 2026-09-25: The counters of one run, held by the scheduler's telemetry
/// (`SysTelemetry::stats`). K=2 counts accepts and rejects; K=3 and K=4 also
/// bucket steps by drafts accepted and count per-position draft matches.
#[derive(Debug, Default)]
pub struct SpecStats {
    /// 2026-09-25: One-shot log latches, keyed by a `&'static str`.
    fired: std::sync::Mutex<std::collections::BTreeSet<&'static str>>,
    pub k2_accepts: AtomicU64,
    pub k2_rejects: AtomicU64,

    pub k3_accept: [AtomicU64; 3],
    pub k3_steps: AtomicU64,
    pub k3_d1_match: AtomicU64,
    pub k3_d2_match_uncond: AtomicU64,
    pub k3_d2_match_cond: AtomicU64,

    pub k4_accept: [AtomicU64; 4],
    pub k4_steps: AtomicU64,
    pub k4_d1: AtomicU64,
    pub k4_d2_uncond: AtomicU64,
    pub k4_d3_uncond: AtomicU64,
    pub k4_d2_cond: AtomicU64,
    pub k4_d3_cond: AtomicU64,

    /// 2026-09-25: B1 gauge: decode positions inside a tool parameter body
    /// whose top-1/top-2 logit gap is below the low-margin threshold
    /// (`logit_processors::b1_margin`).
    pub b1_low_margin: AtomicU64,

    // 2026-09-25: Decode timing, recorded under `METRALE_DECODE_TIMING`.
    pub decode_copy_us: AtomicU64,
    pub decode_sample_us: AtomicU64,
    pub decode_count: AtomicU64,
}

/// 2026-09-25: Adds one to `counter` (relaxed) and returns the previous
/// value.
#[inline]
pub fn bump(counter: &AtomicU64) -> u64 {
    counter.fetch_add(1, Ordering::Relaxed)
}

#[inline]
pub fn get(counter: &AtomicU64) -> u64 {
    counter.load(Ordering::Relaxed)
}

#[inline]
pub fn reset(counter: &AtomicU64) {
    counter.store(0, Ordering::Relaxed);
}

impl SpecStats {
    pub fn new() -> Self {
        Self::default()
    }

    /// 2026-09-25: `true` the first time this `SpecStats` sees `key`, `false`
    /// after. The scheduler's counterpart to `ModelStats::once`.
    pub fn once(&self, key: &'static str) -> bool {
        self.fired.lock().expect("run latches poisoned").insert(key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_fresh_run_starts_at_zero() {
        let s = SpecStats::new();
        assert_eq!(get(&s.k4_steps), 0);
        assert_eq!(get(&s.k2_accepts), 0);
        assert!(s.k3_accept.iter().all(|c| get(c) == 0));
    }

    #[test]
    fn two_runs_count_independently() {
        let a = SpecStats::new();
        let b = SpecStats::new();
        for _ in 0..7 {
            bump(&a.k4_steps);
        }
        assert_eq!(get(&a.k4_steps), 7);
        assert_eq!(get(&b.k4_steps), 0, "a second run starts clean");
    }

    #[test]
    fn reset_clears_a_counter_for_the_next_summary_window() {
        let s = SpecStats::new();
        bump(&s.k3_steps);
        bump(&s.k3_steps);
        assert_eq!(get(&s.k3_steps), 2);
        reset(&s.k3_steps);
        assert_eq!(get(&s.k3_steps), 0);
    }
}
