// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Unit tests for [`SsmDecodeRing`]: insertion, eviction, boundary selection, and the disabled-ring path.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use super::SsmDecodeRing;

#[test]
fn disabled_ring_records_nothing() {
    let mut ring = SsmDecodeRing::new(0);
    assert!(!ring.is_enabled());
    assert_eq!(ring.record(10), None);
    assert_eq!(ring.slot_for_position(10), None);
    assert_eq!(ring.len(), 0);
}

#[test]
fn record_assigns_distinct_slots_until_full() {
    let mut ring = SsmDecodeRing::new(3);
    assert!(ring.is_enabled());
    assert_eq!(ring.record(5), Some(0));
    assert_eq!(ring.record(12), Some(1));
    assert_eq!(ring.record(20), Some(2));
    assert_eq!(ring.len(), 3);
    assert_eq!(ring.slot_for_position(5), Some(0));
    assert_eq!(ring.slot_for_position(12), Some(1));
    assert_eq!(ring.slot_for_position(20), Some(2));
}

#[test]
fn record_evicts_oldest_when_full_and_reuses_its_slot() {
    let mut ring = SsmDecodeRing::new(3);
    ring.record(5);
    ring.record(12);
    ring.record(20);
    assert_eq!(ring.record(31), Some(0));
    assert_eq!(ring.len(), 3);
    assert_eq!(ring.slot_for_position(5), None);
    assert_eq!(ring.slot_for_position(12), Some(1));
    assert_eq!(ring.slot_for_position(20), Some(2));
    assert_eq!(ring.slot_for_position(31), Some(0));
}

#[test]
fn record_wraps_round_robin_over_capacity() {
    let mut ring = SsmDecodeRing::new(2);
    assert_eq!(ring.record(1), Some(0));
    assert_eq!(ring.record(2), Some(1));
    assert_eq!(ring.record(3), Some(0));
    assert_eq!(ring.record(4), Some(1));
    assert_eq!(ring.record(5), Some(0));
    assert_eq!(ring.slot_for_position(4), Some(1));
    assert_eq!(ring.slot_for_position(5), Some(0));
    assert_eq!(ring.slot_for_position(3), None);
}

#[test]
fn slot_for_position_requires_exact_match() {
    let mut ring = SsmDecodeRing::new(3);
    ring.record(10);
    ring.record(20);
    assert_eq!(ring.slot_for_position(15), None);
    assert_eq!(ring.slot_for_position(10), Some(0));
    assert_eq!(ring.slot_for_position(20), Some(1));
}

#[test]
fn snapshot_positions_lists_live_entries_oldest_first() {
    let mut ring = SsmDecodeRing::new(3);
    ring.record(7);
    ring.record(14);
    ring.record(21);
    let positions: Vec<usize> = ring.snapshot_positions().collect();
    assert_eq!(positions, vec![7, 14, 21]);
    ring.record(28);
    let positions: Vec<usize> = ring.snapshot_positions().collect();
    assert_eq!(positions, vec![14, 21, 28]);
}

#[test]
fn truncate_after_drops_entries_past_keep_len() {
    let mut ring = SsmDecodeRing::new(3);
    ring.record(10);
    ring.record(20);
    ring.record(30);
    ring.truncate_after(20);
    assert_eq!(ring.len(), 2);
    assert_eq!(ring.slot_for_position(30), None);
    assert_eq!(ring.slot_for_position(20), Some(1));
    assert_eq!(ring.slot_for_position(10), Some(0));
}

#[test]
fn truncate_after_keeps_exact_boundary_snapshot() {
    let mut ring = SsmDecodeRing::new(3);
    ring.record(10);
    ring.record(25);
    ring.truncate_after(25);
    assert_eq!(ring.slot_for_position(25), Some(1));
    assert_eq!(ring.len(), 2);
}

#[test]
fn boundary_with_snapshot_selection_picks_latest_eligible() {
    // 2026-09-25: simulates the snapshot-aware boundary search: of the
    // candidate boundary positions, the latest one with a live snapshot
    // is chosen.
    let mut ring = SsmDecodeRing::new(3);
    ring.record(8);
    ring.record(16);
    ring.record(40);
    let candidate_boundaries = [40usize, 32, 16, 8];
    let chosen = candidate_boundaries
        .iter()
        .copied()
        .find(|&b| ring.slot_for_position(b).is_some());
    assert_eq!(chosen, Some(40));
}

#[test]
fn decline_when_no_boundary_has_snapshot() {
    // 2026-09-25: no candidate boundary has a live snapshot, so the
    // snapshot-aware search finds nothing and the rollback is declined.
    let mut ring = SsmDecodeRing::new(3);
    ring.record(5);
    ring.record(11);
    let candidate_boundaries = [48usize, 36, 24];
    let chosen = candidate_boundaries
        .iter()
        .copied()
        .find(|&b| ring.slot_for_position(b).is_some());
    assert_eq!(chosen, None);
}

#[test]
fn record_after_truncate_does_not_panic_or_share_slots() {
    // 2026-09-25: after a truncation, refilling past capacity twice over
    // must not panic, must never hold two entries at one token position,
    // and must end with the ring full.
    let mut ring = SsmDecodeRing::new(5);
    for (i, pos) in [10, 20, 30, 40, 50].iter().enumerate() {
        assert_eq!(ring.record(*pos), Some(i));
    }
    ring.truncate_after(10);
    assert_eq!(ring.len(), 1);
    let mut pos = 60;
    for _ in 0..12 {
        ring.record(pos);
        let mut slots: Vec<usize> = ring.snapshot_positions().collect();
        slots.sort_unstable();
        slots.dedup();
        assert_eq!(slots.len(), ring.len(), "duplicate token_position");
        pos += 10;
    }
    assert_eq!(ring.len(), 5);
}

#[test]
fn truncate_rewinds_cursor_to_preserve_survivors() {
    // 2026-09-25: after truncation the cursor resumes after the newest
    // survivor, so the freed tail slot is reused before any surviving
    // snapshot is evicted.
    let mut ring = SsmDecodeRing::new(3);
    ring.record(10);
    ring.record(20);
    ring.record(30);
    ring.truncate_after(20);
    ring.record(40);
    assert_eq!(ring.slot_for_position(10), Some(0));
    assert_eq!(ring.slot_for_position(20), Some(1));
    assert_eq!(ring.slot_for_position(40), Some(2));
    ring.record(50);
    assert_eq!(ring.slot_for_position(10), None);
    assert_eq!(ring.slot_for_position(50), Some(0));
}

#[test]
fn truncate_to_empty_resets_cursor() {
    let mut ring = SsmDecodeRing::new(3);
    ring.record(10);
    ring.record(20);
    ring.truncate_after(0);
    assert_eq!(ring.len(), 0);
    assert_eq!(ring.record(30), Some(0));
}
