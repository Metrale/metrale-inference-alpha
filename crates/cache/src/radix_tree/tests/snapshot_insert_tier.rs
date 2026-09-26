// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The three snapshot insert paths over spilled (tiered) entries: none hands back a slot the spill already freed, and each makes the entry resident again.
//!
//! A tiered entry's `snapshot_id` was already returned by `evict_to_tier`
//! (`TierEvict::Spill { slot, .. }`), and model-engine frees it
//! (`acquire_or_spill_slot` calls `self.free(evict.slot())`).
//! `SsmSnapshotPool::free` pushes onto the free list with no membership check,
//! so returning the same id again would put one slot on the free list twice
//! and hand it to two sequences.
//!
//! Owner: cache.
//! Invariants: none beyond the types.

use super::super::*;
use super::index;
use metrale_telemetry::prefix_cache::TierEvict;

/// 2026-09-25: An index holding one spilled, non-tail entry of session 7.
/// Returns `(index, prefix_hash, the slot the spill freed)`.
fn spilled(prefix_hash: u64, slot: usize) -> (SsmSnapshotIndex, u64, usize) {
    let mut idx = index(
        vec![SnapshotEntry {
            snapshot_id: slot,
            session_hash: 7,
            token_count: 16000,
            prefix_hash,
            last_access: 50,
            tiered: false,
            is_tail: false,
            is_tail_sibling: false,
        }],
        7,
    );
    let TierEvict::Spill { slot: freed, .. } = idx.evict_to_tier(0).expect("a victim exists")
    else {
        panic!("an ungated evict must SPILL, not drop");
    };
    assert_eq!(freed, slot, "the spill freed exactly this slot");
    (idx, prefix_hash, freed)
}

#[test]
fn insert_over_a_spilled_entry_hands_back_no_slot() {
    let (mut idx, ph, freed) = spilled(0xA1, 4);
    let displaced = idx.insert(
        // 2026-09-25: A fresh save of the same prefix into slot 11.
        ph, 11, 7, 16000,
    );
    assert_eq!(
        displaced, None,
        "slot {freed} was already freed at spill time; returning it again \
         double-frees it into the snapshot pool"
    );
}

#[test]
fn insert_tail_over_a_spilled_entry_hands_back_no_slot() {
    let (mut idx, ph, freed) = spilled(0xB2, 5);
    let displaced = idx.insert_tail(
        // 2026-09-25: A fresh tail save of the same prefix into slot 12.
        ph, 12, 7, 16000,
    );
    assert!(
        displaced.is_empty(),
        "expected no slot to free, got {displaced:?} (slot {freed} was freed at spill time)"
    );
}

/// 2026-09-25: `insert_tail` over a spilled entry clears `tiered`. An entry
/// left `tiered` while holding a live slot is skipped by `lookup` and by both
/// victim scans, so its slot could never be freed.
#[test]
fn insert_tail_rehomes_a_spilled_entry_to_hbm() {
    let (mut idx, ph, _) = spilled(0xC3, 6);
    idx.insert_tail(
        // 2026-09-25: A fresh tail save of the same prefix into slot 13.
        ph, 13, 7, 16000,
    );

    assert!(
        !idx.entries[0].tiered,
        "a fresh HBM save re-homes the prefix; leaving it `tiered` strands slot 13"
    );
    assert_eq!(
        idx.evict_lru(),
        Some(13),
        "a re-homed entry must be evictable — otherwise its slot leaks for the process lifetime"
    );
}

/// 2026-09-25: `insert_tail_sibling` over a spilled entry hands back no slot
/// and clears `tiered`, as the two paths above do. `reinsert_unspills` covers
/// the re-home for plain `insert`.
#[test]
fn insert_tail_sibling_over_a_spilled_entry_hands_back_no_slot() {
    let (mut idx, ph, freed) = spilled(0xD4, 8);
    let displaced = idx.insert_tail_sibling(
        // 2026-09-25: A fresh sibling save of the same prefix into slot 14.
        ph, 14, 7, 16000,
    );
    assert_eq!(
        displaced, None,
        "slot {freed} was already freed at spill time"
    );
    assert!(
        !idx.entries[0].tiered,
        "a fresh HBM save re-homes the prefix; leaving it `tiered` strands slot 14"
    );
    assert_eq!(
        idx.evict_lru(),
        Some(14),
        "a re-homed sibling must be evictable — otherwise its slot leaks"
    );
}

/// 2026-09-25: `insert_tail` displaces entries in two places: the sweep that
/// removes the session's previous tail and sibling, and the overwrite of a
/// matching `prefix_hash`. `spilled()` builds a non-tail entry, so the tests
/// above reach only the overwrite. This one spills a tail and supersedes it
/// at a different prefix: the sweep must not hand back its slot either.
#[test]
fn the_tail_supersede_sweep_does_not_hand_back_a_spilled_tail() {
    let mut idx = index(
        vec![SnapshotEntry {
            snapshot_id: 41,
            session_hash: 7,
            token_count: 16000,
            prefix_hash: 0xF1,
            last_access: 50,
            tiered: false,
            is_tail: true,
            is_tail_sibling: false,
        }],
        7,
    );
    let TierEvict::Spill { slot: freed, .. } = idx.evict_to_tier(0).expect("a victim exists")
    else {
        panic!("an ungated evict must SPILL, not drop");
    };
    assert_eq!(freed, 41);
    assert!(idx.entries[0].tiered, "the victim is now tiered");

    let displaced = idx.insert_tail(
        // 2026-09-25: A new tail for session 7 at a different prefix.
        0xF2, 42, 7, 17000,
    );
    assert!(
        displaced.is_empty(),
        "the sweep must not hand back slot {freed} — it was freed at spill time, and by now \
         the pool has handed it to a live fault-in target; got {displaced:?}"
    );
}

/// 2026-09-25: Control: overwriting a resident entry still hands back its
/// live slot. An insert that never returned a slot would pass the tests above.
#[test]
fn insert_over_a_resident_entry_still_hands_back_its_slot() {
    let mut idx = SsmSnapshotIndex::new();
    let toks: Vec<u32> = (0..40).collect();
    let ph = super::super::hash_token_prefix(&toks, 40, 0);
    idx.insert(ph, 21, 7, 40);

    assert_eq!(
        idx.insert(ph, 22, 7, 40),
        Some(21),
        "the displaced LIVE slot must still be returned for the caller to free"
    );
}

/// 2026-09-25: Control for the sweep: superseding a resident tail still
/// hands back its slot.
#[test]
fn insert_tail_still_displaces_a_resident_tail() {
    let mut idx = SsmSnapshotIndex::new();
    idx.insert_tail(
        // 2026-09-25: Session 7's resident tail in slot 31.
        0xE5, 31, 7, 100,
    );
    let displaced = idx.insert_tail(
        // 2026-09-25: A new tail for session 7 at a different prefix.
        0xE6, 32, 7, 200,
    );
    assert_eq!(displaced, vec![31]);
}
