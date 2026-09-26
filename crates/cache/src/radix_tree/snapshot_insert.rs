// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The `SsmSnapshotIndex` insert paths: plain, tail and tail
//! sibling.
//!
//! Owner: cache.
//! Invariants: an overwritten or swept entry hands back its slot only if it
//! is resident (`freeable_slot`), and every overwrite leaves the entry
//! resident.

use super::snapshot::{SnapshotEntry, SsmSnapshotIndex};

/// 2026-09-25: The slot a displaced entry hands back for the caller to free,
/// or `None` for a tiered entry.
///
/// A tiered entry's `snapshot_id` was already returned by `evict_to_tier`
/// (`TierEvict::Spill { slot, .. }`). Returning it again would free it twice:
/// the model engine's `SsmSnapshotPool::free` pushes onto its free list
/// without a membership check, so two sequences could get the same slot.
fn freeable_slot(entry: &SnapshotEntry) -> Option<usize> {
    (!entry.tiered).then_some(entry.snapshot_id)
}

impl SsmSnapshotIndex {
    pub(super) fn insert(
        &mut self,
        prefix_hash: u64,
        snapshot_id: usize,
        session_hash: u64,
        token_count: usize,
    ) -> Option<usize> {
        for entry in &mut self.entries {
            if entry.prefix_hash == prefix_hash {
                let old = freeable_slot(entry);
                entry.snapshot_id = snapshot_id;
                entry.session_hash = session_hash;
                entry.token_count = token_count;
                // 2026-09-25: The new save makes the entry resident. If it was
                // tiered, its blob is not removed: this index holds no store
                // handle, and the model engine removes blobs only on its reap
                // paths (`store.remove` in `ssm_snapshot_spill.rs` and
                // `ssm_snapshot_faultin.rs`).
                entry.tiered = false;
                // 2026-09-25: A plain save is not a tail or sibling. Keeping
                // the flag would leave another session's tail flag on an entry
                // now owned by this session, outside `insert_tail`'s sweep.
                entry.is_tail = false;
                entry.is_tail_sibling = false;
                self.access_counter += 1;
                entry.last_access = self.access_counter;
                return old;
            }
        }
        self.access_counter += 1;
        self.stats.saves += 1;
        self.entries.push(SnapshotEntry {
            snapshot_id,
            session_hash,
            token_count,
            prefix_hash,
            last_access: self.access_counter,
            tiered: false,
            is_tail: false,
            is_tail_sibling: false,
        });
        None
    }

    /// 2026-09-25: Insert the session's tail entry, first removing the
    /// session's previous tail and sibling (for a nonzero session). Returns
    /// the resident slots displaced, for the caller to free.
    pub(super) fn insert_tail(
        &mut self,
        prefix_hash: u64,
        snapshot_id: usize,
        session_hash: u64,
        token_count: usize,
    ) -> Vec<usize> {
        let mut displaced = Vec::new();
        if session_hash != 0 {
            let mut i = 0;
            while i < self.entries.len() {
                if (self.entries[i].is_tail || self.entries[i].is_tail_sibling)
                    && self.entries[i].session_hash == session_hash
                {
                    displaced.extend(freeable_slot(&self.entries.swap_remove(i)));
                } else {
                    i += 1;
                }
            }
        }
        for entry in &mut self.entries {
            if entry.prefix_hash == prefix_hash {
                displaced.extend(freeable_slot(entry));
                entry.snapshot_id = snapshot_id;
                entry.session_hash = session_hash;
                entry.token_count = token_count;
                // 2026-09-25: Resident again. An entry left `tiered` while
                // holding a live slot would be skipped by `lookup` and both
                // victim scans, so the slot could never be freed.
                entry.tiered = false;
                entry.is_tail = true;
                entry.is_tail_sibling = false;
                self.access_counter += 1;
                entry.last_access = self.access_counter;
                return displaced;
            }
        }
        self.access_counter += 1;
        self.entries.push(SnapshotEntry {
            snapshot_id,
            session_hash,
            token_count,
            prefix_hash,
            last_access: self.access_counter,
            tiered: false,
            is_tail: true,
            is_tail_sibling: false,
        });
        displaced
    }

    /// 2026-09-25: Insert the tail's sibling, one block below the tail.
    /// Call it after [`Self::insert_tail`] for the same capture: that call's
    /// sweep removed the session's previous sibling, and this one does not
    /// sweep.
    pub(super) fn insert_tail_sibling(
        &mut self,
        prefix_hash: u64,
        snapshot_id: usize,
        session_hash: u64,
        token_count: usize,
    ) -> Option<usize> {
        for entry in &mut self.entries {
            if entry.prefix_hash == prefix_hash {
                let old = freeable_slot(entry);
                entry.snapshot_id = snapshot_id;
                entry.session_hash = session_hash;
                entry.token_count = token_count;
                entry.tiered = false;
                entry.is_tail = false;
                entry.is_tail_sibling = true;
                self.access_counter += 1;
                entry.last_access = self.access_counter;
                return old;
            }
        }
        self.access_counter += 1;
        self.stats.saves += 1;
        self.entries.push(SnapshotEntry {
            snapshot_id,
            session_hash,
            token_count,
            prefix_hash,
            last_access: self.access_counter,
            tiered: false,
            is_tail: false,
            is_tail_sibling: true,
        });
        None
    }
}
