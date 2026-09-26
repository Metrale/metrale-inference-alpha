// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `SsmSnapshotIndex` tests for the tail lease, the tail sibling, and the depth weight α of the eviction score.
//!
//! Owner: cache.
//! Invariants: none beyond the types.

use super::super::*;
use super::{entry, index, tail_entry};

#[test]
fn tail_lease_protects_live_session_is_tail() {
    let idx = index(vec![entry(7, 1, 8192, 100), tail_entry(9, 1, 6064, 50)], 1);
    let v = idx.session_aware_victim(true, false).unwrap();
    assert_eq!(idx.entries[v].snapshot_id, 7);
}

/// 2026-09-25: The lease protects the tail, not the deepest entry: the live
/// session's deepest non-tail entry is an ordinary candidate.
///
/// The pool holds three entries so that the answer shows both the lease and
/// the recency order among the unleased pair. With two, the lease would leave
/// one candidate, and "no lease" and "lease the tail" would pick the same
/// victim.
#[test]
fn finish_leaf_not_protected() {
    let idx = index(
        vec![
            entry(6, 1, 100, 95),
            entry(7, 1, 6080, 50),
            tail_entry(9, 1, 6064, 40),
        ],
        1,
    );
    let v = idx.session_aware_victim(true, false).unwrap();
    assert_eq!(
        idx.entries[v].snapshot_id, 7,
        "the deepest non-tail entry must be the victim: the tail is leased \
         despite being the oldest, and recency still orders the unleased pair"
    );
}

/// 2026-09-25: The lease covers only the live session's tail. A dormant
/// session's tail is evictable, and the stalest session goes first.
#[test]
fn dormant_session_tail_evictable() {
    // 2026-09-25: Session 2 is live; session 1 is older.
    let idx = index(
        vec![
            tail_entry(1, 1, 20000, 10),
            entry(2, 2, 4000, 90),
            tail_entry(3, 2, 12000, 95),
        ],
        2,
    );
    let v = idx.session_aware_victim(true, false).unwrap();
    assert_eq!(
        idx.entries[v].snapshot_id, 1,
        "stalest (dormant) session evicted first"
    );
}

/// 2026-09-25: A pool whose only entry is the leased tail still yields it as
/// the victim: the lease binds only while an unleased candidate exists.
#[test]
fn single_leased_entry_still_evictable() {
    let idx = index(vec![tail_entry(5, 1, 16000, 50)], 1);
    let v = idx.session_aware_victim(true, false).unwrap();
    assert_eq!(idx.entries[v].snapshot_id, 5);
}

/// 2026-09-25: The lease lapses after `tail_lease_ttl()` evictions with no
/// lookup from the live session; its tail is then an ordinary candidate.
#[test]
fn lease_expires_without_live_lookups() {
    let mut idx = index(vec![tail_entry(9, 1, 6064, 50), entry(7, 2, 4000, 100)], 1);
    // 2026-09-25: With the lease in force the tail (9) is skipped, so the
    // only unleased entry (7) is the victim.
    assert_eq!(idx.evict_lru(), Some(7), "leased tail survives while fresh");
    idx.entries.push(entry(8, 2, 4000, 101));
    // 2026-09-25: Expire the lease; session 1 is now the stalest session.
    idx.evictions_since_lookup = super::tail_lease_ttl();
    assert_eq!(
        idx.evict_lru(),
        Some(9),
        "expired lease: the dead session's tail is evictable (stalest session)"
    );
}

/// 2026-09-25: A plain `insert` over a tail clears `is_tail`; otherwise
/// another session's plain save would carry the tail flag, and with it the
/// session gate and the lease.
#[test]
fn insert_overwrite_clears_is_tail() {
    let mut idx = SsmSnapshotIndex::new();
    let displaced = idx.insert_tail(0xAB, 1, 7, 500);
    assert!(displaced.is_empty());
    assert!(idx.entries[0].is_tail);
    let old = idx.insert(0xAB, 2, 8, 500);
    assert_eq!(old, Some(1));
    assert!(
        !idx.entries[0].is_tail,
        "plain insert must clear is_tail on overwrite"
    );
}

#[test]
fn lookup_tiered_tail_session_gate() {
    let mut idx = SsmSnapshotIndex::new();
    let toks: Vec<u32> = (0..64).collect();
    let ph = super::hash_token_prefix(&toks, 64, 0);
    idx.insert_tail(ph, 4, 7, 64);
    assert!(idx.lookup_tiered(&toks, 64, 7, 0).is_some());
    assert!(idx.lookup_tiered(&toks, 64, 8, 0).is_none());
    assert!(idx.lookup_tiered(&toks, 64, 0, 0).is_none());
}

/// 2026-09-25: With α = 0 (the value when `METRALE_SNAP_EVICT_ALPHA` is
/// unset) the rank within a session is recency only.
#[test]
fn alpha_zero_is_pure_lru_ordering() {
    let idx = index(vec![entry(1, 7, 20000, 10), entry(2, 7, 100, 90)], 7);
    let v = idx.session_aware_victim(false, false).unwrap();
    assert_eq!(idx.entries[v].snapshot_id, 1, "α=0 ranks purely by recency");
}

/// 2026-09-25: With α = 2, depth outweighs a small recency gap within a
/// session, while session staleness stays the first key. α is passed to
/// `session_aware_victim_with_alpha` rather than set through the environment,
/// which the parallel tests share.
#[test]
fn alpha_prefers_depth_within_session_staleness_still_primary() {
    let idx = index(vec![entry(1, 7, 20000, 50), entry(2, 7, 100, 60)], 7);
    let v = idx
        .session_aware_victim_with_alpha(false, false, 2.0)
        .unwrap();
    assert_eq!(
        idx.entries[v].snapshot_id, 2,
        "α=2: the shallow entry is the victim despite being fresher"
    );
    let idx2 = index(vec![entry(1, 1, 20000, 10), entry(2, 2, 100, 90)], 2);
    let v2 = idx2
        .session_aware_victim_with_alpha(false, false, 2.0)
        .unwrap();
    assert_eq!(
        idx2.entries[v2].snapshot_id, 1,
        "session staleness remains the primary key at any α"
    );
}

#[test]
fn sibling_leased_with_live_session() {
    let mut idx = index(vec![entry(7, 1, 8192, 100)], 1);
    idx.insert_tail_sibling(0xE1, 9, 1, 6048);
    // 2026-09-25: Make the sibling the oldest entry, so recency alone would
    // pick it.
    idx.entries
        .iter_mut()
        .find(|e| e.snapshot_id == 9)
        .unwrap()
        .last_access = 10;
    let v = idx.session_aware_victim(true, false).unwrap();
    assert_eq!(idx.entries[v].snapshot_id, 7, "sibling must be leased");
}

#[test]
fn new_tail_sweeps_old_tail_and_sibling() {
    let mut idx = SsmSnapshotIndex::new();
    idx.insert_tail(0xA1, 1, 7, 500);
    idx.insert_tail_sibling(0xA2, 2, 7, 484);
    let displaced = idx.insert_tail(0xB1, 3, 7, 1000);
    let mut d = displaced.clone();
    d.sort_unstable();
    assert_eq!(d, vec![1, 2], "old tail AND sibling displaced together");
    assert_eq!(idx.len(), 1);
}

/// 2026-09-25: A pool of only leased entries (a tail and its sibling) still
/// yields a victim, oldest first. The test first asserts that the lease is
/// in force, since with the lease off any pool yields a victim.
#[test]
fn all_leased_pool_still_evicts() {
    let mut idx = SsmSnapshotIndex::new();
    idx.insert_tail(0xA1, 1, 7, 500);
    idx.insert_tail_sibling(0xA2, 2, 7, 484);
    // 2026-09-25: `matched_tokens = 0` matches no entry but latches session 7.
    let toks: Vec<u32> = (0..8).collect();
    let _ = idx.lookup_tiered(&toks, 0, 7, 0);
    assert!(
        idx.tail_lease_active(),
        "the lease must actually be in force"
    );
    assert!(
        idx.entries
            .iter()
            .all(|e| (e.is_tail || e.is_tail_sibling) && e.session_hash == 7),
        "every entry must be a leased restore point of the live session"
    );
    assert_eq!(
        idx.evict_lru(),
        Some(1),
        "only-leased pool must still yield a victim, oldest-first"
    );
    assert_eq!(idx.len(), 1, "and the reclaim must actually have happened");
}

/// 2026-09-25: Outside `--hermetic`, a sibling entry is not session-gated in
/// `lookup_tiered`: the same session, session 0 and a different session all
/// match it (`session_gate_blocks` gates only `is_tail`).
#[test]
fn sibling_not_session_gated_in_lookup() {
    let mut idx = SsmSnapshotIndex::new();
    let toks: Vec<u32> = (0..64).collect();
    let ph = super::hash_token_prefix(&toks, 64, 0);
    idx.insert_tail_sibling(ph, 4, 7, 64);
    assert!(idx.lookup_tiered(&toks, 64, 7, 0).is_some());
    assert!(
        idx.lookup_tiered(&toks, 64, 0, 0).is_some(),
        "sibling must not carry the is_tail session gate"
    );
    assert!(
        idx.lookup_tiered(&toks, 64, 8, 0).is_some(),
        "a sibling is content-addressed: another session may restore from it"
    );
}

#[test]
fn insert_overwrite_clears_sibling() {
    let mut idx = SsmSnapshotIndex::new();
    idx.insert_tail_sibling(0xC1, 1, 7, 500);
    assert!(idx.entries[0].is_tail_sibling);
    idx.insert(0xC1, 2, 8, 500);
    assert!(!idx.entries[0].is_tail_sibling);
}
