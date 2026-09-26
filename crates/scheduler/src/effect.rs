// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: [`Effect`], the device-side bookkeeping the core asks a router for
//! between steps, and [`EffectOutcome`], the router's answer. Generic over the
//! sequence state `S` and the logits handle `P`.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use crate::plan::Label;

pub enum Effect<'a, S, P> {
    /// 2026-09-25: Release a sequence's device resources: optionally offer its KV to the
    /// prefix cache, free it, then tell the EP worker.
    ReleaseSeq {
        seq: &'a mut S,
        cache: bool,
        /// 2026-09-25: The caller's label, for the failure log lines.
        what: &'static str,
    },
    /// 2026-09-25: Migrate a live sequence onto slot `target`.
    CompactSlot {
        seq: &'a mut S,
        target: usize,
    },
    /// 2026-09-25: Disown a sequence's slot after it was migrated to another
    /// sequence, so a later free or drop cannot release it twice.
    DetachSlot {
        seq: &'a mut S,
    },
    /// 2026-09-25: Stream a sequence's device state into `writer` (spill).
    SaveSequenceState {
        seq: &'a S,
        writer: &'a mut dyn std::io::Write,
    },
    /// 2026-09-25: Copy the logits block `logits` (`rows` rows) to the host.
    ReadLogits {
        logits: P,
        rows: usize,
        into: &'a mut Vec<u8>,
    },
    /// 2026-09-25: Block-aligned Marconi SSM checkpoint after a decode step.
    MarconiCheckpoint {
        seq: &'a mut S,
    },
    SsmSnapshotSave {
        seq: &'a S,
        slot: usize,
    },
    SsmSnapshotRestore {
        seq: &'a S,
        slot: usize,
    },
    /// 2026-09-25: `save_hidden_for_catchup(row, pos)`: copy row `row`'s hidden into the
    /// MTP catch-up ring at label `pos`.
    SaveHiddenCatchup {
        row: usize,
        pos: usize,
    },
    /// 2026-09-25: Make the default stream wait for the secondary stream.
    SyncSecondary,
    /// 2026-09-25: Every worker slot down.
    EpShutdown,
    /// 2026-09-25: Synchronise `stream` before teardown.
    Quiesce {
        stream: u64,
    },
    /// 2026-09-25: Make sure the KV block for the sequence's next decode position exists
    /// before a step is launched ahead of the host; answers `Reserved` with
    /// the blocks it added (0 when the block was already there) or
    /// `Exhausted` when the allocation fails.
    ReserveKv {
        seq: &'a mut S,
    },
    /// 2026-09-25: The over-run of a fed step was discarded for this sequence: release
    /// the blocks the latest `ReserveKv` added for it. Only a router that
    /// reserves ahead supports it; the synchronous router returns an error.
    Rollback {
        seq: &'a mut S,
    },
}

impl<S, P> Label for Effect<'_, S, P> {
    fn label(&self) -> String {
        match self {
            Self::ReleaseSeq { cache, what, .. } => format!("ReleaseSeq{{cache={cache}, {what}}}"),
            Self::CompactSlot { target, .. } => format!("CompactSlot{{target={target}}}"),
            Self::DetachSlot { .. } => "DetachSlot".into(),
            Self::SaveSequenceState { .. } => "SaveSequenceState".into(),
            Self::ReadLogits { rows, .. } => format!("ReadLogits{{rows={rows}}}"),
            Self::MarconiCheckpoint { .. } => "MarconiCheckpoint".into(),
            Self::SsmSnapshotSave { slot, .. } => format!("SsmSnapshotSave{{slot={slot}}}"),
            Self::SsmSnapshotRestore { slot, .. } => format!("SsmSnapshotRestore{{slot={slot}}}"),
            Self::SaveHiddenCatchup { row, pos } => {
                format!("SaveHiddenCatchup{{row={row}, pos={pos}}}")
            }
            Self::SyncSecondary => "SyncSecondary".into(),
            Self::EpShutdown => "EpShutdown".into(),
            Self::Quiesce { stream } => format!("Quiesce{{stream={stream}}}"),
            Self::ReserveKv { .. } => "ReserveKv".into(),
            Self::Rollback { .. } => "Rollback".into(),
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum EffectOutcome {
    Unit,
    /// 2026-09-25: `ReadLogits`: the element width the model writes.
    HostLogits {
        elem_bytes: usize,
    },
    /// 2026-09-25: `ReserveKv`: the blocks added for the next position.
    Reserved {
        blocks: usize,
    },
    /// 2026-09-25: `ReserveKv`: no block could be added; the core does not launch ahead.
    Exhausted,
}
