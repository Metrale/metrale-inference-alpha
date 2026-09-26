// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `SsmSnapshotIndex` unit tests: victim choice, recency, lookup, stats and the spill-tier state machine, plus the entry and index builders the `lease` and `insert_tier` submodules share.
//!
//! Owner: cache.
//! Invariants: none beyond the types.

use super::*;
use metrale_telemetry::prefix_cache::TierEvict;

#[path = "snapshot_lease.rs"]
mod lease;

#[path = "snapshot_insert_tier.rs"]
mod insert_tier;

/// 2026-09-25: Build a resident, non-tail entry with an explicit recency.
/// Tests identify entries by `snapshot_id`, which also serves as the prefix
/// hash, because `swap_remove` reorders the Vec.
fn entry(
    snapshot_id: usize,
    session_hash: u64,
    token_count: usize,
    last_access: u64,
) -> SnapshotEntry {
    SnapshotEntry {
        snapshot_id,
        session_hash,
        token_count,
        prefix_hash: snapshot_id as u64,
        last_access,
        tiered: false,
        is_tail: false,
        is_tail_sibling: false,
    }
}

/// 2026-09-25: Like `entry`, with `is_tail` set.
fn tail_entry(
    snapshot_id: usize,
    session_hash: u64,
    token_count: usize,
    last_access: u64,
) -> SnapshotEntry {
    SnapshotEntry {
        is_tail: true,
        ..entry(snapshot_id, session_hash, token_count, last_access)
    }
}

fn index(entries: Vec<SnapshotEntry>, live: u64) -> SsmSnapshotIndex {
    SsmSnapshotIndex {
        entries,
        access_counter: 1000,
        last_lookup_session: live,
        evictions_since_lookup: 0,
        stats: SnapshotStats::default(),
    }
}

/// 2026-09-25: With `tail_protect` false the older entry is the victim even
/// when it is the live session's tail; with it true the tail is spared. Id 9
/// is a tail because on plain entries `tail_protect` changes nothing.
#[test]
fn deep_tail_evicted_without_tail_protect() {
    let idx = index(
        vec![
            entry(
                // 2026-09-25: The newer entry, not a tail.
                7, 1, 8192, 100,
            ),
            tail_entry(
                // 2026-09-25: The older, deeper tail of the same session.
                9, 1, 16000, 50,
            ),
        ],
        1,
    );
    let v = idx.session_aware_victim(false, false).unwrap();
    assert_eq!(idx.entries[v].snapshot_id, 9);
    // 2026-09-25: Same pool, lease armed: without this pair a `leased`
    // predicate that ignored `tail_protect` would also pass.
    let armed = idx.session_aware_victim(true, false).unwrap();
    assert_eq!(idx.entries[armed].snapshot_id, 7);
}

/// 2026-09-25: Eviction ranks by recency only: an entry hit many times but not
/// recently loses to entries saved after its last hit. Uses the serving
/// lookup, `lookup_tiered`.
#[test]
fn eviction_ignores_hit_history() {
    let mut idx = SsmSnapshotIndex::new();
    let toks: Vec<u32> = (0..100).collect();
    let ph = super::hash_token_prefix(&toks, 40, 0);
    idx.insert(ph, 1, 7, 40);
    let ph50 = super::hash_token_prefix(&toks, 50, 0);
    idx.insert(ph50, 4, 7, 50);
    // 2026-09-25: `matched_tokens = 40` keeps the depth-50 entry out of range,
    // so only slot 1 is hit.
    for _ in 0..5 {
        assert!(idx.lookup_tiered(&toks, 40, 7, 0).is_some());
    }
    // 2026-09-25: The hits made slot 1 fresher than the later-saved slot 4.
    // Without this check the last assertion would hold whether or not a hit
    // bumps `last_access`.
    assert_eq!(idx.evict_lru(), Some(4), "a hit must refresh recency");
    let ph80 = super::hash_token_prefix(&toks, 80, 0);
    let ph90 = super::hash_token_prefix(&toks, 90, 0);
    idx.insert(ph80, 2, 7, 80);
    idx.insert(ph90, 3, 7, 90);
    assert_eq!(idx.evict_lru(), Some(1), "hit history must not pin fossils");
}

/// 2026-09-25: A lookup moves only the winner's recency, never a losing
/// candidate's, in both `lookup_tiered` and `lookup`.
#[test]
fn lookup_bumps_winner_only() {
    let mut idx = SsmSnapshotIndex::new();
    let toks: Vec<u32> = (0..100).collect();
    let ph40 = super::hash_token_prefix(&toks, 40, 0);
    let ph80 = super::hash_token_prefix(&toks, 80, 0);
    idx.insert(ph40, 1, 7, 40);
    idx.insert(ph80, 2, 7, 80);
    let shallow_before = idx
        .entries
        .iter()
        .find(|e| e.snapshot_id == 1)
        .unwrap()
        .last_access;
    let m = idx.lookup_tiered(&toks, 100, 7, 0).expect("hit");
    assert_eq!(m.token_count, 80, "deep entry wins");
    let shallow_after = idx
        .entries
        .iter()
        .find(|e| e.snapshot_id == 1)
        .unwrap()
        .last_access;
    assert_eq!(
        shallow_before, shallow_after,
        "losing candidate's recency must not move"
    );
    let m2 = idx.lookup(&toks, 100, 7, 0).expect("hit");
    assert_eq!(m2.1, 80);
    let shallow_final = idx
        .entries
        .iter()
        .find(|e| e.snapshot_id == 1)
        .unwrap()
        .last_access;
    assert_eq!(shallow_before, shallow_final);
}

/// 2026-09-25: A lookup with a non-zero session latches it as the live
/// session and resets the lease's eviction count; session 0 does not latch.
#[test]
fn lookup_tracks_live_session() {
    let mut idx = SsmSnapshotIndex::new();
    assert_eq!(idx.last_lookup_session, 0);
    // 2026-09-25: The index is empty; the lookup misses and still latches.
    let _ = idx.lookup(&[1, 2, 3], 3, 42, 0);
    assert_eq!(idx.last_lookup_session, 42);
    let _ = idx.lookup(&[1, 2, 3], 3, 0, 0);
    assert_eq!(idx.last_lookup_session, 42, "session 0 must not latch");
    // 2026-09-25: `RadixTree::lookup` calls `lookup_tiered`, so the latch
    // and the lease renewal are checked there too.
    idx.evictions_since_lookup = 7;
    let _ = idx.lookup_tiered(&[1, 2, 3], 3, 99, 0);
    assert_eq!(idx.last_lookup_session, 99);
    assert_eq!(
        idx.evictions_since_lookup, 0,
        "a live lookup renews the lease"
    );
}

/// 2026-09-25: Stats: a miss adds all matched tokens to
/// `recompute_tokens_on_miss`; a hit adds the distance from the anchor to the
/// match point to `recompute_tokens_on_hit`.
#[test]
fn stats_track_hits_and_recompute() {
    let mut idx = SsmSnapshotIndex::new();
    let toks: Vec<u32> = (0..100).collect();

    assert!(
        idx.lookup(&toks, 100, 7, 0)
            // 2026-09-25: A cold miss: nothing is registered yet.
            .is_none()
    );
    let ph = super::hash_token_prefix(&toks, 40, 0);
    assert!(
        idx.insert(
            // 2026-09-25: An anchor at depth 40 in slot 3.
            ph, 3, 7, 40
        )
        .is_none()
    );
    let hit = idx.lookup(&toks, 100, 7, 0);
    assert_eq!(hit, Some((3, 40)));

    let s = idx.stats;
    assert_eq!(s.lookups, 2);
    assert_eq!(s.hits, 1);
    assert_eq!(s.saves, 1);
    assert_eq!(s.anchor_depth_sum, 40);
    assert_eq!(s.recompute_tokens_on_hit, 60, "matched(100) - anchor(40)");
    assert_eq!(
        s.recompute_tokens_on_miss, 100,
        "cold miss = full recompute"
    );
}

/// 2026-09-25: An ungated `evict_to_tier` keeps the victim in the index,
/// marks it spilled and hands back its HBM slot.
#[test]
fn evict_to_tier_spills_not_removes() {
    let mut idx = index(vec![entry(3, 1, 8192, 100), entry(9, 1, 16000, 50)], 1);
    let before = idx.len();
    let TierEvict::Spill {
        slot: freed_slot,
        key,
        ..
    } = idx
        .evict_to_tier(/* 2026-09-25: 0 disables the spill gate */ 0)
        .expect("a resident victim exists")
    else {
        panic!("an ungated evict must SPILL, not drop");
    };
    // 2026-09-25: Neither entry is a tail, so the lease spares nothing and
    // the older entry (id 9) is the victim.
    assert_eq!(freed_slot, 9);
    assert_eq!(key, 9, "key is the victim's prefix_hash");
    assert_eq!(idx.len(), before, "entry kept, not removed");
    assert_eq!(idx.stats.tier_spills, 1);
    // 2026-09-25: `evict_lru` skips the spilled entry and frees the resident
    // one.
    assert_eq!(idx.evict_lru(), Some(3));
}

/// 2026-09-25: A victim shallower than `min_tokens` is dropped: the entry is
/// removed, counted in `evictions`, not in `tier_spills`, and nothing tiered is
/// left behind.
#[test]
fn shallow_victim_is_dropped_not_spilled() {
    let mut idx = index(vec![entry(9, 1, 100, 50)], 1);
    let before = idx.len();
    let ev = idx
        .evict_to_tier(/* 2026-09-25: drop victims below 1024 tokens */ 1024)
        .expect("a victim exists");
    assert_eq!(
        ev,
        TierEvict::Drop {
            slot: 9,
            depth: 100
        }
    );
    assert_eq!(idx.len(), before - 1, "entry REMOVED, not kept findable");
    assert_eq!(idx.stats.tier_spills, 0, "a dropped victim is not a spill");
    assert_eq!(idx.stats.evictions, 1, "it is a plain eviction");
    assert_eq!(idx.evict_to_tier(1024), None);
}

#[test]
fn deep_victim_is_spilled_under_the_gate() {
    let mut idx = index(vec![entry(9, 1, 16000, 50)], 1);
    let ev = idx
        .evict_to_tier(/* 2026-09-25: drop victims below 1024 tokens */ 1024)
        .expect("a victim exists");
    assert_eq!(
        ev,
        TierEvict::Spill {
            slot: 9,
            key: 9,
            depth: 16000
        }
    );
    assert_eq!(idx.len(), 1, "entry kept, findable for fault-in");
    assert_eq!(idx.stats.tier_spills, 1);
}

/// 2026-09-25: `lookup` skips a spilled entry, so it never returns a freed
/// slot; `lookup_tiered` finds it as `Tier(key)`.
#[test]
fn spilled_entry_lookup_semantics() {
    let mut idx = SsmSnapshotIndex::new();
    let toks: Vec<u32> = (0..50).collect();
    let ph = super::hash_token_prefix(&toks, 50, 0);
    idx.insert(ph, 4, 7, 50);
    let TierEvict::Spill {
        slot: freed, key, ..
    } = idx.evict_to_tier(0).unwrap()
    else {
        panic!("an ungated evict must SPILL, not drop");
    };
    assert_eq!((freed, key), (4, ph));

    assert!(idx.lookup(&toks, 50, 7, 0).is_none());
    let m = idx.lookup_tiered(&toks, 50, 7, 0).expect("tiered hit");
    assert_eq!(m.token_count, 50);
    assert_eq!(m.loc, SnapLoc::Tier(ph));
    assert_eq!(idx.stats.tier_hits, 1);
}

#[test]
fn promote_rehomes_to_hbm() {
    let mut idx = SsmSnapshotIndex::new();
    let toks: Vec<u32> = (0..30).collect();
    let ph = super::hash_token_prefix(&toks, 30, 0);
    idx.insert(ph, 1, 7, 30);
    idx.evict_to_tier(0).unwrap();

    assert!(idx.promote(ph, 12));
    assert_eq!(idx.stats.tier_fault_ins, 1);
    assert_eq!(idx.lookup(&toks, 30, 7, 0), Some((12, 30)));
    let m = idx.lookup_tiered(&toks, 30, 7, 0).unwrap();
    assert_eq!(m.loc, SnapLoc::Hbm(12));
}

#[test]
fn evict_to_tier_none_when_all_spilled() {
    let mut idx = SsmSnapshotIndex::new();
    idx.insert(10, 0, 7, 5);
    idx.insert(20, 1, 7, 6);
    assert!(idx.evict_to_tier(0).is_some());
    assert!(idx.evict_to_tier(0).is_some());
    assert_eq!(idx.evict_to_tier(0), None, "nothing resident left to spill");
    assert_eq!(idx.evict_lru(), None, "nothing resident left to drop");
}

#[test]
fn reinsert_unspills() {
    let mut idx = SsmSnapshotIndex::new();
    idx.insert(0xAA, 1, 7, 40);
    idx.evict_to_tier(0).unwrap();
    idx.insert(0xAA, 5, 7, 40);
    assert_eq!(idx.evict_lru(), Some(5));
}

#[test]
fn test_snapshot_index_insert_lookup_roundtrip() {
    let mut idx = SsmSnapshotIndex::new();
    let tokens: Vec<u32> = (0..48).collect();
    let prefix_hash = super::hash_token_prefix(&tokens, 32, 0);

    assert!(idx.insert(prefix_hash, 42, 100, 32).is_none());
    // 2026-09-25: Match more tokens than the anchor covers, so the returned
    // depth must be the entry's own `token_count` (32), not the caller's
    // `matched_tokens` (48).
    let result = idx.lookup(&tokens, 48, 100, 0);
    assert_eq!(result, Some((42, 32)));
    assert_eq!(idx.lookup(&tokens, 31, 100, 0), None);
}

#[test]
fn test_snapshot_index_lru_eviction() {
    let mut idx = SsmSnapshotIndex::new();
    let tokens_a: Vec<u32> = (0..16).collect();
    let tokens_b: Vec<u32> = (100..116).collect();
    let ha = super::hash_token_prefix(&tokens_a, 16, 0);
    let hb = super::hash_token_prefix(&tokens_b, 16, 0);

    idx.insert(ha, 1, 0, 16);
    idx.insert(hb, 2, 0, 16);

    let evicted = idx.evict_lru();
    assert_eq!(evicted, Some(1));
    assert_eq!(idx.len(), 1);

    let evicted = idx.evict_lru();
    assert_eq!(evicted, Some(2));
    assert_eq!(idx.len(), 0);

    assert_eq!(idx.evict_lru(), None);
}

#[test]
fn test_snapshot_index_session_isolation() {
    // 2026-09-25: Outside `--hermetic`, `session_gate_blocks` gates only tail
    // entries: a plain entry matches any session, including 0, and a tail
    // matches only its own non-zero session.
    let mut idx = SsmSnapshotIndex::new();
    let tokens: Vec<u32> = (0..16).collect();
    let prefix_hash = super::hash_token_prefix(&tokens, 16, 0);

    idx.insert(prefix_hash, 42, 100, 16);
    assert_eq!(idx.lookup(&tokens, 16, 200, 0), Some((42, 16)));
    assert_eq!(idx.lookup(&tokens, 16, 100, 0), Some((42, 16)));
    assert_eq!(idx.lookup(&tokens, 16, 0, 0), Some((42, 16)));

    idx.insert_tail(prefix_hash, 43, 100, 16);
    assert_eq!(idx.lookup(&tokens, 16, 200, 0), None);
    assert_eq!(idx.lookup(&tokens, 16, 0, 0), None);
    let result = idx.lookup(&tokens, 16, 100, 0);
    assert_eq!(result, Some((43, 16)));
}

#[test]
fn test_snapshot_index_overwrite_existing() {
    let mut idx = SsmSnapshotIndex::new();
    let tokens: Vec<u32> = (0..16).collect();
    let prefix_hash = super::hash_token_prefix(&tokens, 16, 0);

    assert!(idx.insert(prefix_hash, 5, 0, 16).is_none());
    assert_eq!(idx.len(), 1);

    let old = idx.insert(prefix_hash, 8, 0, 16);
    assert_eq!(old, Some(5));
    assert_eq!(idx.len(), 1);

    let result = idx.lookup(&tokens, 16, 0, 0);
    assert_eq!(result, Some((8, 16)));
}

#[test]
fn test_snapshot_index_deepest_match() {
    let mut idx = SsmSnapshotIndex::new();
    let tokens: Vec<u32> = (0..64).collect();

    let h16 = super::hash_token_prefix(&tokens, 16, 0);
    idx.insert(h16, 10, 0, 16);

    let h32 = super::hash_token_prefix(&tokens, 32, 0);
    idx.insert(h32, 20, 0, 32);

    let result = idx.lookup(&tokens, 48, 0, 0);
    assert_eq!(result, Some((20, 32)));

    let result = idx.lookup(&tokens, 20, 0, 0);
    assert_eq!(result, Some((10, 16)));
}
