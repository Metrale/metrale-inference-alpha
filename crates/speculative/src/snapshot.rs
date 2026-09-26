// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The scheduler's observability snapshot: one `Copy` struct the
//! scheduler publishes each loop tick into the run's `SnapshotCell`, which the
//! TUI reads.
//!
//! Owner: speculative.
//! Invariants:
//! - A `SnapshotCell` holds at most the latest snapshot; `publish` replaces
//!   it whole.

use std::time::Instant;

use parking_lot::Mutex;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MtpModeSnap {
    Mtp,
    Serial,
    Probing,
    /// 2026-09-25: No MTP gate in this run.
    Off,
}

#[derive(Clone, Copy, Debug)]
pub struct SchedulerSnapshot {
    pub active_seqs: u32,
    pub prefilling_seqs: u32,
    pub swapped_seqs: u32,
    pub pending_len: u32,
    pub kv_blocks_free: u32,
    pub kv_blocks_total: u32,
    pub ssm_slots_used: u32,
    pub ssm_slots_total: u32,
    pub mtp_mode: MtpModeSnap,
    /// 2026-09-25: The MTP gate's delivered-throughput EWMA for the current
    /// mode (tok/s); 0.0 when unmeasured or without a gate.
    pub delivered_tps: f32,
    pub steps_total: u64,
    /// 2026-09-25: When the scheduler published this snapshot, on its clock.
    pub published_at: Instant,
}

/// 2026-09-25: The run's snapshot cell: the scheduler writes it, the TUI
/// reads it. `serve` creates one per run and shares it as an `Arc`, by the
/// same route as `SchedLevers`.
#[derive(Default)]
pub struct SnapshotCell(Mutex<Option<SchedulerSnapshot>>);

impl SnapshotCell {
    /// 2026-09-25: Replaces the held snapshot.
    pub fn publish(&self, s: SchedulerSnapshot) {
        *self.0.lock() = Some(s);
    }

    /// 2026-09-25: The latest snapshot, if one has been published.
    pub fn read(&self) -> Option<SchedulerSnapshot> {
        *self.0.lock()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn publish_read_roundtrip() {
        let cell = SnapshotCell::default();
        let s = SchedulerSnapshot {
            active_seqs: 3,
            prefilling_seqs: 1,
            swapped_seqs: 0,
            pending_len: 2,
            kv_blocks_free: 100,
            kv_blocks_total: 200,
            ssm_slots_used: 4,
            ssm_slots_total: 128,
            mtp_mode: MtpModeSnap::Mtp,
            delivered_tps: 42.5,
            steps_total: 7,
            published_at: Instant::now(),
        };
        cell.publish(s);
        let r = cell.read().expect("published");
        assert_eq!(r.active_seqs, 3);
        assert_eq!(r.mtp_mode, MtpModeSnap::Mtp);
    }
}
