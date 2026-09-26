// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The decode-time SSM-snapshot ring: a per-sequence bounded set of boundary-token snapshots that boundary rollback restores from.
//!
//! Owner: scheduler.
//! Invariants:
//! - At most `capacity` entries are live, and no two live entries share a `snapshot_slot`.
//!
//! For hybrid (attention + SSM) models, boundary rollback
//! ([`super::rollback::rollback_to_boundary`]) must also rewind the
//! recurrent SSM `h_state` and `conv_state`, which only a snapshot taken at
//! the boundary can do. `snapshot_boundary_if_ssm` records one here each
//! time the plain decode path commits a boundary token outside thinking,
//! and rollback restores the one at the chosen boundary ([`super::rollback::find_last_boundary_with_snapshot`]).
//! This struct tracks slot indices only; the GPU memory belongs to the
//! model's snapshot pool, which allocates `capacity` slots per decode
//! sequence at init.
//!
//! `capacity` is the model's `decode_rollback_ring_slots()`, and the depth
//! policy (the maximum, the free-memory fit, the flag and env overrides)
//! lives in `metrale_model_layers::ssm_reserve`. A smaller ring retains
//! fewer boundaries; 0 declines every rollback on a hybrid model.

/// 2026-09-25: One recorded boundary snapshot: the generated-token position it was
/// taken at, and the model-side snapshot slot holding the GPU state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SsmRingEntry {
    /// 2026-09-25: Length of `output_tokens` when the snapshot was taken, so the
    /// snapshot is the SSM state after exactly this many generated tokens. A
    /// rollback that keeps `keep_len` tokens needs the entry whose
    /// `token_position == keep_len`.
    pub token_position: usize,
    /// 2026-09-25: Decode-rollback snapshot slot index in `[0, capacity)` that the
    /// model wrote this sequence's SSM state into.
    pub snapshot_slot: usize,
}

/// 2026-09-25: Bounded ring of decode-time SSM snapshots for one active sequence.
///
/// No GPU work or I/O: the scheduler issues the save and restore copies
/// (`Effect::SsmSnapshotSave` / `SsmSnapshotRestore`); this struct only
/// decides which slot to use and which positions a rollback can reach.
#[derive(Debug, Clone)]
pub struct SsmDecodeRing {
    /// 2026-09-25: Live entries, oldest first. Length never exceeds `capacity`.
    entries: Vec<SsmRingEntry>,
    /// 2026-09-25: Maximum live entries, equal to the decode-rollback snapshot slots
    /// reserved for this sequence. `0` disables the ring (pure-attention
    /// models, or SSM models with no reserved region).
    capacity: usize,
    /// 2026-09-25: Round-robin cursor over `[0, capacity)` for slot assignment.
    next_slot: usize,
}

impl SsmDecodeRing {
    /// 2026-09-25: Create a ring with room for `capacity` snapshots. `capacity == 0`
    /// gives a disabled ring: every `record` returns `None` and every
    /// lookup finds nothing.
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: Vec::with_capacity(capacity),
            capacity,
            next_slot: 0,
        }
    }

    #[inline]
    pub fn is_enabled(&self) -> bool {
        self.capacity > 0
    }

    /// 2026-09-25: Reserve the snapshot slot the next boundary should be written
    /// into, registering the `(token_position, slot)` entry.
    ///
    /// The slot is the one at the cursor; whichever live entry held it is
    /// evicted. Returns `None` only when the ring is disabled
    /// (`capacity == 0`).
    ///
    /// On a `Some` return the caller must save the SSM state into the slot,
    /// or remove the entry if the save fails; otherwise the entry points at
    /// stale GPU state.
    pub fn record(&mut self, token_position: usize) -> Option<usize> {
        if self.capacity == 0 {
            return None;
        }
        let slot = self.next_slot;
        self.next_slot = (self.next_slot + 1) % self.capacity;

        // 2026-09-25: evict by slot ownership, not by age: the entry
        // holding the cursor's slot is not always the oldest (a
        // `truncate_after` moves the cursor). Two entries sharing a slot
        // would let a rollback restore state a later save overwrote.
        self.entries.retain(|e| e.snapshot_slot != slot);
        self.entries.push(SsmRingEntry {
            token_position,
            snapshot_slot: slot,
        });
        Some(slot)
    }

    /// 2026-09-25: The snapshot slot of the entry whose `token_position` equals
    /// `keep_len`, or `None` when no live snapshot matches.
    pub fn slot_for_position(&self, keep_len: usize) -> Option<usize> {
        self.entries
            .iter()
            .find(|e| e.token_position == keep_len)
            .map(|e| e.snapshot_slot)
    }

    /// 2026-09-25: The token positions that have a live snapshot, in recording order.
    pub fn snapshot_positions(&self) -> impl Iterator<Item = usize> + '_ {
        self.entries.iter().map(|e| e.token_position)
    }

    /// 2026-09-25: Drop every entry whose `token_position` is greater than
    /// `keep_len`, keeping the one at exactly `keep_len`. Called after a
    /// rollback to `keep_len`, and to discard an entry whose save failed.
    pub fn truncate_after(&mut self, keep_len: usize) {
        self.entries.retain(|e| e.token_position <= keep_len);
        // 2026-09-25: resume slot assignment right after the newest
        // surviving entry. The dropped entries were recorded after it, in
        // cursor order, so the next records reuse their slots before
        // evicting a survivor.
        if let Some(newest) = self.entries.last() {
            self.next_slot = (newest.snapshot_slot + 1) % self.capacity;
        } else if self.capacity > 0 {
            self.next_slot = 0;
        }
    }

    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.entries.len()
    }
}

#[cfg(test)]
#[path = "ssm_decode_ring_tests.rs"]
mod ssm_decode_ring_tests;
