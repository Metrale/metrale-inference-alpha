// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: SSM snapshot index counters, logged by
//! `SsmSnapshotIndex::log_stats_if_due` when `METRALE_SSM_SNAP_STATS` is set.
//!
//! Owner: cache.
//! Invariants: counters only grow; outside tests only `log_stats_if_due`
//! reads them.

#[derive(Default, Clone, Copy)]
pub(super) struct SnapshotStats {
    /// 2026-09-25: New entries added by `insert` and `insert_tail_sibling`.
    /// Overwrites and `insert_tail` do not count.
    pub saves: u64,
    /// 2026-09-25: Calls to `lookup` and `lookup_tiered`.
    pub lookups: u64,
    /// 2026-09-25: Lookups that found an entry.
    pub hits: u64,
    /// 2026-09-25: Sum of the found entry's `token_count` over hits.
    pub anchor_depth_sum: u64,
    /// 2026-09-25: Sum of `matched_tokens` minus the found entry's
    /// `token_count` over hits: matched tokens whose SSM state is recomputed.
    pub recompute_tokens_on_hit: u64,
    /// 2026-09-25: Sum of `matched_tokens` over misses.
    pub recompute_tokens_on_miss: u64,
    /// 2026-09-25: Entries dropped with their state: by `evict_lru`, and by
    /// the drop arm of `evict_to_tier`.
    pub evictions: u64,
    /// 2026-09-25: Entries `evict_to_tier` marked spilled.
    pub tier_spills: u64,
    /// 2026-09-25: `lookup_tiered` hits on a spilled entry.
    pub tier_hits: u64,
    /// 2026-09-25: Successful `promote` calls.
    pub tier_fault_ins: u64,
    /// 2026-09-25: Spilled entries removed by `forget_tiered`, which the
    /// model engine calls when an entry's blob is gone.
    pub tier_reaps: u64,
}
