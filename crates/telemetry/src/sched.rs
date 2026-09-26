// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: L2 scheduler and L3 cache instruments (queue depths, batch
//! shape, loop phase timings, blocking syncs and D2H copies, CUDA-graph
//! replays and captures, KV-block and SSM-slot occupancy) and the L4
//! [`SpecMatrix`].
//!
//! Fed through the scheduler's `TelemetryIo` router and the GPU backend's
//! copy, sync and graph entry points. The prefix-cache hit and miss counters
//! live in [`crate::run_metrics`], not here.
//!
//! Owner: telemetry.
//! Invariants: none beyond the types.

use crate::instrument::{CountHistogram, Counter, Gauge, NanosHistogram};

/// 2026-09-26: The most loop phases `Telemetry::set_phase_names` keeps.
pub const MAX_PHASES: usize = 32;

/// 2026-09-26: One tick's scheduler shape, as the scheduler publishes it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub struct SchedShape {
    pub pending: u64,
    pub active: u64,
    pub prefilling: u64,
    pub swapped: u64,
    pub kv_blocks_free: u64,
    pub kv_blocks_total: u64,
    pub ssm_slots_used: u64,
    pub ssm_slots_total: u64,
    /// 2026-09-26: Ticks the publishing scheduler has counted since it was built.
    pub ticks: u64,
}

#[derive(Debug)]
pub struct SchedInstruments {
    pub phase: [NanosHistogram; MAX_PHASES],
    pub pending: Gauge,
    pub active: Gauge,
    pub prefilling: Gauge,
    pub swapped: Gauge,
    pub kv_blocks_free: Gauge,
    pub kv_blocks_total: Gauge,
    pub ssm_slots_used: Gauge,
    pub ssm_slots_total: Gauge,
    pub ticks: Gauge,
    /// 2026-09-26: Active sequences, on ticks that have any.
    pub decode_rows: CountHistogram,
    /// 2026-09-26: Prefilling sequences, on ticks that have any.
    pub prefill_seqs: CountHistogram,
    pub stream_syncs: Counter,
    pub blocking_d2h: Counter,
    pub graph_replays: Counter,
    pub graph_captures: Counter,
}

impl Default for SchedInstruments {
    fn default() -> Self {
        Self::new()
    }
}

impl SchedInstruments {
    pub const fn new() -> Self {
        Self {
            phase: [const { NanosHistogram::new() }; MAX_PHASES],
            pending: Gauge::new(),
            active: Gauge::new(),
            prefilling: Gauge::new(),
            swapped: Gauge::new(),
            kv_blocks_free: Gauge::new(),
            kv_blocks_total: Gauge::new(),
            ssm_slots_used: Gauge::new(),
            ssm_slots_total: Gauge::new(),
            ticks: Gauge::new(),
            decode_rows: CountHistogram::new(),
            prefill_seqs: CountHistogram::new(),
            stream_syncs: Counter::new(),
            blocking_d2h: Counter::new(),
            graph_replays: Counter::new(),
            graph_captures: Counter::new(),
        }
    }

    pub(crate) fn record(&self, s: &SchedShape) {
        self.pending.set(s.pending);
        self.active.set(s.active);
        self.prefilling.set(s.prefilling);
        self.swapped.set(s.swapped);
        self.kv_blocks_free.set(s.kv_blocks_free);
        self.kv_blocks_total.set(s.kv_blocks_total);
        self.ssm_slots_used.set(s.ssm_slots_used);
        self.ssm_slots_total.set(s.ssm_slots_total);
        self.ticks.set(s.ticks);
        if s.active > 0 {
            self.decode_rows.observe(s.active);
        }
        if s.prefilling > 0 {
            self.prefill_seqs.observe(s.prefilling);
        }
    }

    pub(crate) fn shape(&self) -> SchedShape {
        SchedShape {
            pending: self.pending.get(),
            active: self.active.get(),
            prefilling: self.prefilling.get(),
            swapped: self.swapped.get(),
            kv_blocks_free: self.kv_blocks_free.get(),
            kv_blocks_total: self.kv_blocks_total.get(),
            ssm_slots_used: self.ssm_slots_used.get(),
            ssm_slots_total: self.ssm_slots_total.get(),
            ticks: self.ticks.get(),
        }
    }
}

/// 2026-09-26: The widest verify step [`SpecMatrix`] places.
pub const MAX_DRAFTS: usize = 16;

#[derive(Debug)]
pub struct SpecMatrix {
    /// 2026-09-26: `cells[d][a]`: verify steps that checked `d` drafts and accepted `a`.
    cells: [[Counter; MAX_DRAFTS + 1]; MAX_DRAFTS + 1],
    /// 2026-09-26: Steps not placed: zero drafts, more than [`MAX_DRAFTS`], or
    /// more accepted than checked.
    pub overflow: Counter,
}

impl Default for SpecMatrix {
    fn default() -> Self {
        Self::new()
    }
}

impl SpecMatrix {
    pub const fn new() -> Self {
        Self {
            cells: [const { [const { Counter::new() }; MAX_DRAFTS + 1] }; MAX_DRAFTS + 1],
            overflow: Counter::new(),
        }
    }

    pub(crate) fn record(&self, drafts: usize, accepted: usize) {
        if drafts == 0 || drafts > MAX_DRAFTS || accepted > drafts {
            self.overflow.add(1);
            return;
        }
        self.cells[drafts][accepted].add(1);
    }

    pub fn steps(&self, drafts: usize, accepted: usize) -> u64 {
        self.cells
            .get(drafts)
            .and_then(|r| r.get(accepted))
            .map_or(0, Counter::get)
    }

    /// 2026-09-26: For each depth `1..=drafts`, the fraction of `drafts`-wide
    /// verify steps that accepted at least that deep. Empty when no such step
    /// ran.
    pub fn acceptance_by_depth(&self, drafts: usize) -> Vec<f64> {
        if drafts == 0 || drafts > MAX_DRAFTS {
            return Vec::new();
        }
        let row: Vec<u64> = (0..=drafts).map(|a| self.steps(drafts, a)).collect();
        let total: u64 = row.iter().sum();
        if total == 0 {
            return Vec::new();
        }
        (1..=drafts)
            .map(|depth| row[depth..].iter().sum::<u64>() as f64 / total as f64)
            .collect()
    }

    /// 2026-09-26: Draft widths that have recorded any step.
    pub fn widths(&self) -> Vec<usize> {
        (1..=MAX_DRAFTS)
            .filter(|d| (0..=*d).any(|a| self.steps(*d, a) > 0))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn acceptance_by_depth_is_the_survival_curve_of_accepted_depth() {
        let m = SpecMatrix::new();
        // 2026-09-26: Width 3: 4 steps accepting 0, 1, 3, 3.
        for a in [0, 1, 3, 3] {
            m.record(3, a);
        }
        assert_eq!(m.acceptance_by_depth(3), vec![0.75, 0.5, 0.5]);
        assert_eq!(m.widths(), vec![3]);
        assert!(m.acceptance_by_depth(2).is_empty(), "no width-2 step ran");
    }

    #[test]
    fn impossible_outcomes_are_counted_as_overflow_not_placed() {
        let m = SpecMatrix::new();
        m.record(2, 3);
        m.record(0, 0);
        m.record(MAX_DRAFTS + 1, 1);
        assert_eq!(m.overflow.get(), 3);
        assert!(m.widths().is_empty());
    }

    #[test]
    fn a_tick_with_no_decode_rows_is_not_a_batch_shape_sample() {
        let s = SchedInstruments::new();
        s.record(&SchedShape {
            pending: 2,
            ..Default::default()
        });
        assert_eq!(s.decode_rows.count(), 0);
        assert_eq!(s.pending.get(), 2);
        s.record(&SchedShape {
            active: 5,
            ..Default::default()
        });
        assert_eq!(s.decode_rows.count(), 1);
    }
}
