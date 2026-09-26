// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The spill tier side of `SsmSnapshotIndex`: spilling a victim,
//! the lookup over resident and spilled entries, and moving an entry back
//! (`promote`) or retiring it (`forget_tiered`).
//!
//! Owner: cache.
//! Invariants: an entry is marked tiered only by `evict_to_tier`, and made
//! resident again only by `promote` or an insert overwrite.

use metrale_telemetry::prefix_cache::TierEvict;

use super::hash_token_prefix;
use super::snapshot::{SnapLoc, SnapMatch, SsmSnapshotIndex};

impl SsmSnapshotIndex {
    /// 2026-09-25: Pick a resident victim (`session_aware_victim`). One with
    /// `token_count >= min_tokens` is marked spilled and stays findable by
    /// `lookup_tiered` ([`TierEvict::Spill`]); a shallower one is removed
    /// ([`TierEvict::Drop`]). `min_tokens == 0` always spills. Either way the
    /// returned slot is the caller's to free.
    pub(super) fn evict_to_tier(&mut self, min_tokens: usize) -> Option<TierEvict> {
        if self.entries.is_empty() {
            return None;
        }
        let tail_protect = self.tail_lease_active();
        let idx = self.session_aware_victim(tail_protect, true)?;
        self.evictions_since_lookup = self.evictions_since_lookup.saturating_add(1);
        let depth = self.entries[idx].token_count;
        if min_tokens > 0 && depth < min_tokens {
            // 2026-09-25: Same bookkeeping as `evict_lru`'s session-aware
            // path: `swap_remove`, `stats.evictions` and (above)
            // `evictions_since_lookup`.
            let e = self.entries.swap_remove(idx);
            self.stats.evictions += 1;
            return Some(TierEvict::Drop {
                slot: e.snapshot_id,
                depth,
            });
        }
        let e = &mut self.entries[idx];
        e.tiered = true;
        let freed_slot = e.snapshot_id;
        let key = e.prefix_hash;
        self.stats.tier_spills += 1;
        Some(TierEvict::Spill {
            slot: freed_slot,
            key,
            depth,
        })
    }

    /// 2026-09-25: The lookup `RadixTree::lookup` uses: `lookup` over resident
    /// and spilled entries alike, returning the deepest match and where its
    /// state is. Updates the same counters as `lookup`, plus `tier_hits`.
    pub(super) fn lookup_tiered(
        &mut self,
        tokens: &[u32],
        matched_tokens: usize,
        session_hash: u64,
        adapter_id: u64,
    ) -> Option<SnapMatch> {
        if session_hash != 0 {
            self.last_lookup_session = session_hash;
            self.evictions_since_lookup = 0;
        }
        let hermetic = metrale_gpu_runtime::hermetic_enabled();
        let mut best: Option<usize> = None;
        let mut best_depth = 0usize;
        for (i, entry) in self.entries.iter().enumerate() {
            if entry.token_count > matched_tokens {
                continue;
            }
            if super::snapshot_session::session_gate_blocks(entry, session_hash, hermetic) {
                continue;
            }
            if hash_token_prefix(tokens, entry.token_count, adapter_id) != entry.prefix_hash {
                continue;
            }
            tracing::debug!(
                "snapshot candidate: id={} tokens={} tail={} sibling={} tiered={} (matched {matched_tokens})",
                entry.snapshot_id,
                entry.token_count,
                entry.is_tail,
                entry.is_tail_sibling,
                entry.tiered
            );
            if best.is_none() || entry.token_count > best_depth {
                best = Some(i);
                best_depth = entry.token_count;
            }
        }
        self.stats.lookups += 1;
        let result = if let Some(i) = best {
            self.access_counter += 1;
            let ac = self.access_counter;
            let e = &mut self.entries[i];
            e.last_access = ac;
            let tiered = e.tiered;
            let depth = e.token_count;
            let is_tail = e.is_tail;
            let loc = if tiered {
                SnapLoc::Tier(e.prefix_hash)
            } else {
                SnapLoc::Hbm(e.snapshot_id)
            };
            self.stats.hits += 1;
            self.stats.anchor_depth_sum += depth as u64;
            self.stats.recompute_tokens_on_hit += matched_tokens.saturating_sub(depth) as u64;
            if tiered {
                self.stats.tier_hits += 1;
            }
            Some(SnapMatch {
                token_count: depth,
                loc,
                is_tail,
            })
        } else {
            self.stats.recompute_tokens_on_miss += matched_tokens as u64;
            None
        };
        self.log_stats_if_due();
        result
    }

    /// 2026-09-25: Make the entry for `prefix_hash` resident at `new_slot`,
    /// after the caller faulted its state in. Returns `false` if no entry has
    /// that hash.
    pub(super) fn promote(&mut self, prefix_hash: u64, new_slot: usize) -> bool {
        for e in &mut self.entries {
            if e.prefix_hash == prefix_hash {
                e.tiered = false;
                e.snapshot_id = new_slot;
                self.stats.tier_fault_ins += 1;
                return true;
            }
        }
        false
    }

    /// 2026-09-25: Remove the spilled entry for `prefix_hash`, for a caller
    /// whose blob for it is gone; otherwise every warm turn would try to fault
    /// in the dead key again.
    ///
    /// Returns `false` and changes nothing for an unknown key or a resident
    /// entry: a resident entry's `snapshot_id` is a live slot that this path
    /// could not hand back. So a `promote` that won a race makes this a no-op.
    ///
    /// Leaves `evictions_since_lookup` and `stats.evictions` alone: no slot is
    /// freed, so the tail lease is not shortened.
    pub(super) fn forget_tiered(&mut self, prefix_hash: u64) -> bool {
        let Some(idx) = self
            .entries
            .iter()
            .position(|e| e.prefix_hash == prefix_hash && e.tiered)
        else {
            return false;
        };
        self.entries.swap_remove(idx);
        self.stats.tier_reaps += 1;
        true
    }
}
