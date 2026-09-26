// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The no-op [`PrefixCache`] used when prefix caching is disabled.
//!
//! Owner: telemetry.
//! Invariants: every lookup is empty and nothing is ever stored.

use super::{EvictedBlocks, InsertAcquired, PrefixCache, PrefixMatch};

/// 2026-09-26: No-op prefix cache: `is_active` is `false`, every lookup is
/// empty, and every insert and eviction reports nothing.
pub struct NoPrefixCaching;

impl PrefixCache for NoPrefixCaching {
    fn is_active(&self) -> bool {
        false
    }

    fn lookup(
        &self,
        _tokens: &[u32],
        _block_size: usize,
        _session_hash: u64,
        _adapter_id: u64,
    ) -> PrefixMatch {
        PrefixMatch::empty()
    }

    fn insert(
        &self,
        _tokens: &[u32],
        _block_table: &[u32],
        _disk_block_ids: &[u32],
        _block_size: usize,
        _matched_tokens: usize,
        _adapter_id: u64,
    ) -> InsertAcquired {
        InsertAcquired::default()
    }

    fn insert_with_snapshot(
        &self,
        _tokens: &[u32],
        _block_table: &[u32],
        _disk_block_ids: &[u32],
        _block_size: usize,
        _snapshot_id: usize,
        _session_hash: u64,
        _matched_tokens: usize,
        _adapter_id: u64,
    ) -> (Option<usize>, InsertAcquired) {
        (None, InsertAcquired::default())
    }

    fn insert_intermediate_snapshot(
        &self,
        _tokens: &[u32],
        _block_table: &[u32],
        _disk_block_ids: &[u32],
        _block_size: usize,
        _snapshot_id: usize,
        _session_hash: u64,
        _matched_tokens: usize,
        _adapter_id: u64,
    ) -> Option<usize> {
        None
    }

    fn insert_tail_snapshot(
        &self,
        _tokens: &[u32],
        _snapshot_id: usize,
        _session_hash: u64,
        _adapter_id: u64,
    ) -> Vec<usize> {
        Vec::new()
    }

    fn insert_tail_sibling_snapshot(
        &self,
        _tokens: &[u32],
        _snapshot_id: usize,
        _session_hash: u64,
        _adapter_id: u64,
    ) -> Option<usize> {
        None
    }

    fn release(&self, _tokens: &[u32], _block_size: usize, _adapter_id: u64) {}

    fn release_matched(
        &self,
        _tokens: &[u32],
        _block_size: usize,
        _matched_tokens: usize,
        _adapter_id: u64,
    ) {
    }

    fn evict(&self, _num_blocks: usize) -> EvictedBlocks {
        EvictedBlocks::default()
    }

    fn evict_snapshot_lru(&self) -> Option<usize> {
        None
    }

    fn snapshot_count(&self) -> usize {
        0
    }

    fn stats(&self) -> (usize, usize) {
        (0, 0)
    }
}
