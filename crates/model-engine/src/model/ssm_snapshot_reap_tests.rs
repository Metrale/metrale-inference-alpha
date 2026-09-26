// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests for retiring tier keys: a fault-in miss or a refused spill
//! retires the index entry; a fault-in error keeps it.
//!
//! Owner: model-engine SSM snapshot pool.
//! Invariants: none beyond the types.

use super::*;
use crate::model::ssm_tier::{MemBlobStore, SnapshotBlobStore};
use metrale_gpu_runtime::gpu::mock::MockGpuBackend;

/// 2026-09-25: A small Marconi-only pool, with no decode-rollback ring. The same helper
/// is in `ssm_snapshot_spill_tests.rs`; neither `#[path]` test module can see the
/// other's private items.
fn pool(gpu: &dyn GpuBackend, slots: usize, layers: usize) -> SsmSnapshotPool {
    SsmSnapshotPool::new(
        slots, 32, 16, layers, 0,
        // 2026-09-25: `spill_blob_bytes` counts only h and conv, so the hidden size is
        // not part of a blob.
        0, 8, gpu,
    )
    .unwrap()
}

/// 2026-09-25: A blob the store dropped without telling the prefix cache costs one
/// failed fault-in. After that miss the index entry offers no tier key, so later warm
/// turns recompute instead of spilling a live victim for another miss. Drives
/// `SsmSnapshotPool::fault_in_for_key`, which `try_fault_in_ssm_snapshot` calls, on a
/// `MockGpuBackend`, a `RadixTree` and a one-blob `MemBlobStore`.
#[test]
fn tier_miss_retires_the_key_instead_of_thrashing() {
    use std::sync::atomic::Ordering;

    use metrale_cache::radix_tree::RadixTree;
    use metrale_telemetry::prefix_cache::{PrefixCache, TierEvict};

    const BLK: usize = 16;
    /// 2026-09-25: Above the default spill gate of 1024, so victims are spilled; a
    /// dropped victim's entry is removed and would leave no stale key.
    const DEEP: u32 = 2048;

    /// 2026-09-25: A deep prefix and a disjoint block table, so each `base` is its own
    /// radix branch.
    fn seq(base: u32) -> (Vec<u32>, Vec<u32>) {
        let toks: Vec<u32> = (base..base + DEEP).collect();
        let first_blk = base / BLK as u32;
        let blocks: Vec<u32> = (first_blk..first_blk + DEEP / BLK as u32).collect();
        (toks, blocks)
    }

    let gpu = MockGpuBackend::new();
    let p = pool(&gpu, 2, 2);
    let blob = p.spill_blob_bytes();
    // 2026-09-25: A cap of one blob, so each new record drops the oldest (FIFO).
    let store = MemBlobStore::new(blob);
    let tree = RadixTree::new();

    let (warm, warm_blocks) = seq(0);
    tree.insert_with_snapshot(
        // 2026-09-25: The warm session's anchor, resident in slot 0.
        &warm,
        &warm_blocks,
        &[],
        BLK,
        0,
        7,
        0,
        0,
    );
    assert_eq!(p.try_pop_free_slot(), Some(0));

    // 2026-09-25: Evict it to the tier: the index entry stays findable and the bytes
    // go to the store.
    let TierEvict::Spill { slot, key, .. } = tree.evict_snapshot_to_tier(1024).unwrap() else {
        panic!("a {DEEP}-token victim must spill, not drop");
    };
    assert!(p.spill_slot(slot, key, &store, &gpu, 0).unwrap());
    p.free(slot);

    // 2026-09-25: One more record makes the cap drop the anchor's blob, while the
    // index entry stays tiered with `key`.
    store.put(0xDEAD_BEEF, &vec![0u8; blob]).unwrap();
    let mut probe = vec![0u8; blob];
    assert!(
        !store.get(key, &mut probe).unwrap(),
        "the cap must have dropped the warm anchor's blob for this test to bite"
    );

    // 2026-09-25: Fill the pool with other sessions' snapshots, so each fault-in has
    // to spill a live victim to get a slot.
    for (i, base) in [200_000u32, 300_000].into_iter().enumerate() {
        let (t, b) = seq(base);
        let s = p.try_pop_free_slot().expect("pool has 2 slots");
        tree.insert_with_snapshot(&t, &b, &[], BLK, s, 100 + i as u64, 0, 0);
    }
    assert_eq!(p.try_pop_free_slot(), None, "the pool must be full");

    // 2026-09-25: Read the counters after the setup: the `store.get` probe above is a
    // miss, and the spill and the extra record are puts.
    let puts_before = store.stats.puts.load(Ordering::Relaxed);
    let misses_before = store.stats.get_misses.load(Ordering::Relaxed);

    // 2026-09-25: Four warm turns on the same prefix. Only turn 0 may try the tier;
    // its miss retires the key.
    let mut tier_attempts = 0usize;
    for turn in 0..4u32 {
        let m = tree.lookup(&warm, BLK, 7, 0);
        tree.release(&warm, BLK, 0);
        let Some(k) = m.ssm_snapshot_tier_key else {
            continue;
        };
        tier_attempts += 1;
        assert!(
            p.fault_in_for_key(
                &tree,
                &store,
                &gpu,
                k,
                // 2026-09-25: No slot gets this session: `tag_session` runs only
                // after a successful read.
                7,
                m.ssm_snapshot_tier_tokens,
                0
            )
            .is_none(),
            "the blob is gone — every fault-in attempt must miss"
        );
        // 2026-09-25: The turn saves a snapshot into the slot the failed cycle
        // freed, so the pool is full again and a later fault-in would have to spill.
        let s = p
            .try_pop_free_slot()
            .expect("the failed fault-in returned its slot");
        let (t, b) = seq(500_000 + turn * DEEP);
        tree.insert_with_snapshot(&t, &b, &[], BLK, s, 7, 0, 0);
    }

    assert_eq!(
        tier_attempts, 1,
        "a dropped blob must cost ONE failed fault-in and then degrade to plain \
         recompute — re-offering the same dead key on every warm turn IS the thrash"
    );
    assert_eq!(
        store.stats.get_misses.load(Ordering::Relaxed) - misses_before,
        1,
        "one miss proves the blob is gone; every further miss is re-discovering it"
    );
    assert_eq!(
        store.stats.puts.load(Ordering::Relaxed) - puts_before,
        1,
        "each doomed retry spills a LIVE snapshot D2H to free a slot it then throws \
         away — and under the cap that spill evicts yet another tier record"
    );
    assert_eq!(
        tree.lookup(&warm, BLK, 7, 0).ssm_snapshot_tier_key,
        None,
        "after the miss the anchor must stop advertising a tier key"
    );
}

/// 2026-09-25: A store whose `get` always errors while its blobs stay intact.
struct ErrOnGetStore {
    inner: MemBlobStore,
    gets: std::sync::atomic::AtomicUsize,
    removes: std::sync::atomic::AtomicUsize,
}

impl ErrOnGetStore {
    fn new() -> Self {
        Self {
            inner: MemBlobStore::new(0),
            gets: std::sync::atomic::AtomicUsize::new(0),
            removes: std::sync::atomic::AtomicUsize::new(0),
        }
    }
}

impl SnapshotBlobStore for ErrOnGetStore {
    fn put(&self, key: u64, bytes: &[u8]) -> anyhow::Result<bool> {
        self.inner.put(key, bytes)
    }
    fn get(&self, _key: u64, _out: &mut [u8]) -> anyhow::Result<bool> {
        self.gets.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        anyhow::bail!("simulated tier read failure — the bytes are still there")
    }
    fn remove(&self, key: u64) {
        self.removes
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.inner.remove(key);
    }
    fn len(&self) -> usize {
        self.inner.len()
    }
    fn bytes_resident(&self) -> usize {
        self.inner.bytes_resident()
    }
}

/// 2026-09-25: A `get` error is not evidence the blob is gone: the fault-in returns its
/// slot to the free list and keeps the key, removing nothing from the store.
#[test]
fn tier_error_retains_the_key() {
    use std::sync::atomic::Ordering;

    use metrale_cache::radix_tree::RadixTree;
    use metrale_telemetry::prefix_cache::{PrefixCache, TierEvict};

    const BLK: usize = 16;
    const DEEP: u32 = 2048;

    let gpu = MockGpuBackend::new();
    let p = pool(&gpu, 2, 2);
    let store = ErrOnGetStore::new();
    let tree = RadixTree::new();

    let warm: Vec<u32> = (0..DEEP).collect();
    let warm_blocks: Vec<u32> = (0..DEEP / BLK as u32).collect();
    tree.insert_with_snapshot(
        // 2026-09-25: A deep anchor in slot 0, spilled to the store: its bytes are
        // present, only the reads fail.
        &warm,
        &warm_blocks,
        &[],
        BLK,
        0,
        7,
        0,
        0,
    );
    assert_eq!(p.try_pop_free_slot(), Some(0));
    let TierEvict::Spill { slot, key, .. } = tree.evict_snapshot_to_tier(1024).unwrap() else {
        panic!("a {DEEP}-token victim must spill, not drop");
    };
    assert!(p.spill_slot(slot, key, &store, &gpu, 0).unwrap());
    p.free(slot);
    assert_eq!(store.len(), 1, "the blob is present throughout this test");

    let m = tree.lookup(&warm, BLK, 7, 0);
    tree.release(&warm, BLK, 0);
    let k = m.ssm_snapshot_tier_key.expect("the anchor is tiered");
    let free_before = p.free_slots.lock().len();

    assert!(
        p.fault_in_for_key(
            &tree,
            &store,
            &gpu,
            k,
            // 2026-09-25: No slot gets this session: `tag_session` runs only after a
            // successful read.
            7,
            m.ssm_snapshot_tier_tokens,
            0
        )
        .is_none(),
        "a failed read restores nothing this turn"
    );

    assert_eq!(store.gets.load(Ordering::Relaxed), 1, "exactly one attempt");
    assert_eq!(
        p.free_slots.lock().len(),
        free_before,
        "the slot the failed fault-in took must go back on the free list"
    );
    assert_eq!(
        store.removes.load(Ordering::Relaxed),
        0,
        "reaping on an error would delete a live 66MB snapshot to save one retry"
    );
    assert_eq!(
        tree.lookup(&warm, BLK, 7, 0).ssm_snapshot_tier_key,
        Some(k),
        "an error is not evidence of absence — the key must survive to be retried"
    );
}

/// 2026-09-25: When the store refuses a spill, the entry `evict_to_tier` marked tiered
/// is retired at once, so the next lookup offers no tier key and no resident slot.
#[test]
fn spill_refusal_retires_the_entry_immediately() {
    use metrale_cache::radix_tree::RadixTree;
    use metrale_telemetry::prefix_cache::PrefixCache;

    const BLK: usize = 16;
    const DEEP: u32 = 2048;

    let gpu = MockGpuBackend::new();
    let p = pool(&gpu, 1, 2);
    // 2026-09-25: A cap smaller than one blob: `MemBlobStore::put` refuses it.
    let store = MemBlobStore::new(p.spill_blob_bytes() - 1);
    let tree = RadixTree::new();

    let warm: Vec<u32> = (0..DEEP).collect();
    let warm_blocks: Vec<u32> = (0..DEEP / BLK as u32).collect();
    tree.insert_with_snapshot(
        &warm,
        &warm_blocks,
        &[],
        BLK,
        // 2026-09-25: The only resident snapshot: the pool has one slot.
        0,
        7,
        0,
        0,
    );
    assert_eq!(p.try_pop_free_slot(), Some(0));
    assert_eq!(p.try_pop_free_slot(), None, "the pool is full");

    // 2026-09-25: With a full pool, the acquire path spills this victim and the store
    // refuses it.
    let slot = p
        .acquire_or_spill_slot(&tree, &store, &gpu)
        .expect("the victim's slot is freed regardless");
    assert_eq!(slot, 0);
    assert_eq!(store.len(), 0, "the tier took no bytes");

    let m = tree.lookup(&warm, BLK, 7, 0);
    tree.release(&warm, BLK, 0);
    assert_eq!(
        m.ssm_snapshot_tier_key, None,
        "a refused spill must not leave a findable-but-empty entry"
    );
    assert_eq!(m.ssm_snapshot, None, "and no resident slot either");
}
