// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Spill-tier tests through `RadixTree`, and tests for reaping a tiered entry whose blob is gone (`forget_tiered` / `forget_snapshot_tier_key`).
//!
//! Owner: cache.
//! Invariants: none beyond the types.

use super::super::*;
use metrale_telemetry::prefix_cache::PrefixCache;

/// 2026-09-25: Through the `PrefixCache` API a snapshot goes resident,
/// spilled, resident again, then spilled and reaped, and `lookup` reports each
/// state.
#[test]
fn test_spill_tier_lookup_transitions() {
    let tree = RadixTree::new();
    let tokens: Vec<u32> = (0..64).collect();
    tree.insert_with_snapshot(&tokens, &[10, 20, 30, 40], &[], 16, 99, 7, 0, 0);

    let m = tree.lookup(&tokens, 16, 7, 0);
    assert_eq!(m.ssm_snapshot, Some(99));
    assert_eq!(m.ssm_snapshot_tokens, 64);
    assert_eq!(m.ssm_snapshot_tier_key, None);

    let ev = tree.evict_snapshot_to_tier(0).expect("resident victim");
    let (freed, key) = match ev {
        metrale_telemetry::prefix_cache::TierEvict::Spill { slot, key, .. } => (slot, key),
        other => panic!("an ungated evict must SPILL, not drop: {other:?}"),
    };
    assert_eq!(freed, 99, "the resident slot is freed for reuse");

    let m = tree.lookup(&tokens, 16, 7, 0);
    assert_eq!(m.ssm_snapshot, None, "no resident slot while spilled");
    assert_eq!(m.ssm_snapshot_tier_key, Some(key));
    assert_eq!(m.ssm_snapshot_tier_tokens, 64);

    assert!(tree.promote_snapshot(key, 123));

    let m = tree.lookup(&tokens, 16, 7, 0);
    assert_eq!(m.ssm_snapshot, Some(123));
    assert_eq!(m.ssm_snapshot_tier_key, None);

    // 2026-09-25: Spill again, then reap the entry as a fault-in miss does:
    // the lookup then offers neither the tier key nor a resident slot.
    let ev = tree.evict_snapshot_to_tier(0).expect("resident victim");
    let key = match ev {
        metrale_telemetry::prefix_cache::TierEvict::Spill { key, .. } => key,
        other => panic!("an ungated evict must SPILL, not drop: {other:?}"),
    };
    assert!(
        tree.forget_snapshot_tier_key(key),
        "a tiered entry is reapable"
    );
    let m = tree.lookup(&tokens, 16, 7, 0);
    assert_eq!(
        m.ssm_snapshot_tier_key, None,
        "the dead key is not re-offered"
    );
    assert_eq!(m.ssm_snapshot, None, "and it has no resident slot either");
    tree.release(&tokens, 16, 0);
}

/// 2026-09-25: `NoPrefixCaching` uses the trait defaults of
/// `evict_snapshot_to_tier`, `promote_snapshot` and `forget_snapshot_tier_key`,
/// which do nothing.
#[test]
fn test_no_tier_default_impl() {
    use metrale_telemetry::prefix_cache::NoPrefixCaching;
    let c = NoPrefixCaching;
    assert_eq!(c.evict_snapshot_to_tier(0), None);
    assert!(!c.promote_snapshot(123, 0));
    assert!(!c.forget_snapshot_tier_key(123));
}

/// 2026-09-25: An index with one entry for `tokens`, spilled by
/// `evict_to_tier`. Returns the index and the tier key.
fn spilled_index(tokens: &[u32], slot: usize) -> (SsmSnapshotIndex, u64) {
    let mut idx = SsmSnapshotIndex::new();
    let ph = hash_token_prefix(tokens, tokens.len(), 0);
    idx.insert(ph, slot, 7, tokens.len());
    let ev = idx
        .evict_to_tier(/* 2026-09-25: 0 disables the spill gate */ 0)
        .expect("resident victim");
    match ev {
        metrale_telemetry::prefix_cache::TierEvict::Spill { key, .. } => (idx, key),
        other => panic!("an ungated evict must SPILL: {other:?}"),
    }
}

#[test]
fn forget_tiered_removes_a_tiered_entry() {
    let tokens: Vec<u32> = (0..32).collect();
    let (mut idx, key) = spilled_index(&tokens, 42);
    assert!(idx.forget_tiered(key));
    assert_eq!(idx.len(), 0);
    assert_eq!(idx.lookup_tiered(&tokens, 32, 7, 0), None);
}

/// 2026-09-25: `forget_tiered` refuses a resident entry, such as one promoted
/// back to HBM between a miss and the reap: it returns no slot to its caller,
/// so removing a resident entry would leak that slot.
#[test]
fn forget_tiered_refuses_a_resident_entry() {
    let tokens: Vec<u32> = (0..32).collect();
    let (mut idx, key) = spilled_index(&tokens, 42);
    assert!(idx.promote(key, 123));

    assert!(!idx.forget_tiered(key), "a resident entry must survive");
    assert_eq!(idx.len(), 1);
    let m = idx
        .lookup_tiered(&tokens, 32, 7, 0)
        .expect("still findable");
    assert_eq!(
        m.loc,
        super::super::snapshot::SnapLoc::Hbm(123),
        "its live slot must be untouched — reaping it would leak the slot"
    );
}

/// 2026-09-25: An unknown key and a second reap of the same key both return
/// `false`. Both model-engine callers remove the blob only on `true`
/// (`ssm_snapshot_faultin.rs`, `ssm_snapshot_spill.rs`).
#[test]
fn forget_tiered_unknown_key_is_false() {
    let tokens: Vec<u32> = (0..32).collect();
    let (mut idx, key) = spilled_index(&tokens, 42);
    assert!(!idx.forget_tiered(0xDEAD_BEEF), "unknown key");
    assert!(idx.forget_tiered(key));
    assert!(!idx.forget_tiered(key), "second reap is a no-op");
}

/// 2026-09-25: A reap frees no slot, so it is counted in `tier_reaps` and
/// leaves `evictions` and the lease's eviction count unchanged.
#[test]
fn forget_tiered_does_not_count_as_an_eviction() {
    let tokens: Vec<u32> = (0..32).collect();
    let (mut idx, key) = spilled_index(&tokens, 42);
    let evictions = idx.stats.evictions;
    let since_lookup = idx.evictions_since_lookup;

    assert!(idx.forget_tiered(key));
    assert_eq!(idx.stats.evictions, evictions, "a reap is not an eviction");
    assert_eq!(idx.evictions_since_lookup, since_lookup);
    assert_eq!(idx.stats.tier_reaps, 1, "it has its own counter");
}
