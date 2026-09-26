// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Unit tests for `rollback.rs` that need no `ActiveSeq`.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use super::super::ssm_decode_ring::SsmDecodeRing;
use super::{
    RollbackFallback, RollbackOutcome, find_last_boundary, find_last_boundary_with_snapshot,
    rewind_buffers,
};

/// 2026-09-25: `mask` where token id `id` is a boundary iff `id` is in
/// `boundary_ids`.
fn mask_of(boundary_ids: &[u32], vocab: usize) -> Vec<bool> {
    let mut m = vec![false; vocab];
    for &b in boundary_ids {
        m[b as usize] = true;
    }
    m
}

#[test]
fn finds_last_boundary_skipping_min_keep() {
    let tokens = [10, 99, 11, 12, 99, 13, 14, 15];
    let mask = mask_of(&[99], 100);
    // 2026-09-25: min_keep = 2: the search region is indices 0..=5, whose
    // last boundary is index 4.
    assert_eq!(find_last_boundary(&tokens, &mask, 2), Some(4));
}

#[test]
fn boundary_inside_min_keep_window_is_ignored() {
    // 2026-09-25: the only boundary (id 99) is at index 6; with min_keep=3
    // the search region is 0..=4, so it must not be found.
    let tokens = [10, 11, 12, 13, 14, 15, 99, 16];
    let mask = mask_of(&[99], 100);
    assert_eq!(find_last_boundary(&tokens, &mask, 3), None);
}

#[test]
fn no_boundary_in_buffer_returns_none() {
    let tokens: Vec<u32> = (0..40).collect();
    let mask = mask_of(&[200, 201], 256);
    assert_eq!(find_last_boundary(&tokens, &mask, 4), None);
}

#[test]
fn buffer_shorter_than_min_keep_returns_none() {
    let tokens = [99, 99, 99];
    let mask = mask_of(&[99], 100);
    assert_eq!(find_last_boundary(&tokens, &mask, 5), None);
}

#[test]
fn picks_latest_of_several_boundaries() {
    // 2026-09-25: Boundaries at 2, 5 and 8; min_keep = 1 leaves the region
    // 0..=7, so the latest is 5.
    let tokens = [0, 1, 99, 3, 4, 99, 6, 7, 99];
    let mask = mask_of(&[99], 100);
    assert_eq!(find_last_boundary(&tokens, &mask, 1), Some(5));
}

#[test]
fn snapshot_aware_search_picks_latest_boundary_with_a_snapshot() {
    // 2026-09-25: Boundaries at indices 2, 5 and 8; min_keep = 1 leaves
    // the region 0..=8.
    let tokens = [0, 1, 99, 3, 4, 99, 6, 7, 99, 9];
    let mask = mask_of(&[99], 100);
    assert_eq!(find_last_boundary(&tokens, &mask, 1), Some(8));
    // 2026-09-25: Snapshots only at keep_len 3 (index 2) and 6 (index 5).
    let mut ring = SsmDecodeRing::new(3);
    ring.record(3);
    ring.record(6);
    // 2026-09-25: Index 8 has no snapshot (keep_len 9), so index 5 wins.
    assert_eq!(
        find_last_boundary_with_snapshot(&tokens, &mask, 1, &ring),
        Some(5),
    );
}

#[test]
fn snapshot_aware_search_declines_when_no_boundary_has_a_snapshot() {
    // 2026-09-25: Boundaries at indices 2 and 5; the snapshots are at
    // keep_len 4 and 7, which are indices 3 and 6.
    let tokens = [0, 1, 99, 3, 4, 99, 6, 7];
    let mask = mask_of(&[99], 100);
    let mut ring = SsmDecodeRing::new(3);
    ring.record(4);
    ring.record(7);
    assert!(find_last_boundary(&tokens, &mask, 1).is_some());
    assert_eq!(
        find_last_boundary_with_snapshot(&tokens, &mask, 1, &ring),
        None,
    );
}

#[test]
fn snapshot_aware_search_respects_min_keep_window() {
    // 2026-09-25: The only boundary, index 6, has a snapshot. With n = 10
    // and min_keep = 4 the region is 0..=5, which excludes it.
    let tokens = [0, 1, 2, 3, 4, 5, 99, 7, 8, 9];
    let mask = mask_of(&[99], 100);
    let mut ring = SsmDecodeRing::new(3);
    ring.record(7);
    assert_eq!(
        find_last_boundary_with_snapshot(&tokens, &mask, 4, &ring),
        None,
    );
    // 2026-09-25: With min_keep = 3 the region is 0..=6.
    assert_eq!(
        find_last_boundary_with_snapshot(&tokens, &mask, 3, &ring),
        Some(6),
    );
}

#[test]
fn snapshot_aware_search_empty_ring_declines() {
    let tokens = [0, 99, 2, 3, 4, 5];
    let mask = mask_of(&[99], 100);
    let ring = SsmDecodeRing::new(3);
    assert_eq!(
        find_last_boundary_with_snapshot(&tokens, &mask, 1, &ring),
        None,
    );
}

#[test]
fn snapshot_aware_search_after_eviction_only_sees_live_snapshots() {
    // 2026-09-25: Boundaries at indices 1, 4, 7 and 10.
    let tokens = [0, 99, 2, 3, 99, 5, 6, 99, 8, 9, 99, 11];
    let mask = mask_of(&[99], 100);
    // 2026-09-25: A capacity-2 ring keeps the last two of four records.
    let mut ring = SsmDecodeRing::new(2);
    ring.record(2);
    ring.record(5);
    ring.record(8);
    ring.record(11);
    assert_eq!(
        find_last_boundary_with_snapshot(&tokens, &mask, 1, &ring),
        Some(10),
    );
    // 2026-09-25: `truncate_after(8)` drops the keep_len-11 snapshot and
    // keeps keep_len 8 (index 7).
    ring.truncate_after(8);
    assert_eq!(
        find_last_boundary_with_snapshot(&tokens, &mask, 1, &ring),
        Some(7),
    );
}

#[test]
fn rewind_truncates_output_and_seq_and_lowers_seq_len() {
    let mut output = vec![50, 51, 52, 53, 54, 55];
    let mut seq = vec![1, 2, 3, 50, 51, 52, 53, 54, 55];
    let seq_len = 9;
    let new_len = rewind_buffers(&mut output, &mut seq, seq_len, 4);
    assert_eq!(output, vec![50, 51, 52, 53]);
    assert_eq!(seq, vec![1, 2, 3, 50, 51, 52, 53]);
    assert_eq!(new_len, 7, "seq_len must drop by the 2 rewound tokens");
}

#[test]
fn rewind_keeping_all_is_a_noop() {
    let mut output = vec![50, 51, 52];
    let mut seq = vec![1, 50, 51, 52];
    let new_len = rewind_buffers(&mut output, &mut seq, 4, 3);
    assert_eq!(output, vec![50, 51, 52]);
    assert_eq!(seq, vec![1, 50, 51, 52]);
    assert_eq!(new_len, 4);
}

#[test]
fn rewind_seq_len_saturates_at_zero() {
    let mut output = vec![1, 2, 3, 4, 5];
    let mut seq = vec![1, 2, 3, 4, 5];
    let new_len = rewind_buffers(&mut output, &mut seq, 2, 1);
    assert_eq!(output, vec![1]);
    assert_eq!(new_len, 0, "saturating, never underflow");
}

#[test]
fn rollback_cap_is_two() {
    assert_eq!(metrale_kernels::ROLLBACK_RESTEER_CAP, 2);
}

#[test]
fn fallback_variants_are_distinct() {
    assert_ne!(
        RollbackOutcome::Fallback(RollbackFallback::Disabled),
        RollbackOutcome::Fallback(RollbackFallback::CapReached),
    );
    assert_ne!(
        RollbackOutcome::Fallback(RollbackFallback::NoBoundary),
        RollbackOutcome::Fallback(RollbackFallback::CapReached),
    );
    assert_ne!(
        RollbackOutcome::Fallback(RollbackFallback::LayerStateNotRewindable),
        RollbackOutcome::Fallback(RollbackFallback::NoSsmSnapshot),
    );
    assert_eq!(
        RollbackOutcome::RolledBack { dropped: 7 },
        RollbackOutcome::RolledBack { dropped: 7 },
    );
}

struct StubRomHead;
impl super::RomHead for StubRomHead {
    fn repetition_onset_score(&self, recent: &[u32]) -> f32 {
        (recent.len() as f32 / 100.0).min(1.0)
    }
}

#[test]
fn rom_head_trait_seam_is_callable() {
    let head: std::sync::Arc<dyn super::RomHead> = std::sync::Arc::new(StubRomHead);
    assert!((head.repetition_onset_score(&[1, 2, 3]) - 0.03).abs() < 1e-6);
    assert_eq!(head.repetition_onset_score(&vec![0u32; 500]), 1.0);
}

#[test]
fn rom_head_absent_by_default() {
    assert!(
        crate::scheduler::sched_ctx::SchedCtx::for_test()
            .rom_head
            .is_none()
    );
}

/// 2026-09-25: The watchdog path has a token count, but the matcher
/// records steps only for tokens that advanced it. The matcher's
/// `rollback` asserts `n <= history`, so a larger count must be refused
/// rather than passed on.
#[test]
fn the_reported_mismatch_refuses_to_rewind_instead_of_panicking() {
    assert_eq!(super::grammar_rewind(96, 1), None);
    assert_eq!(super::grammar_rewind(88, 1), None);
}

/// 2026-09-25: A count within the history still rewinds; refusing it
/// would leave the matcher out of step with the tokens.
#[test]
fn an_accountable_span_still_rewinds_exactly() {
    assert_eq!(super::grammar_rewind(4, 4), Some(4));
    assert_eq!(super::grammar_rewind(4, 9), Some(4));
    assert_eq!(super::grammar_rewind(0, 0), Some(0));
}

/// 2026-09-25: `dropped == history` rewinds and one more is refused: an
/// off-by-one here would reach the matcher's `assert!(n <= history)`.
#[test]
fn the_boundary_is_inclusive_on_the_safe_side() {
    assert_eq!(super::grammar_rewind(5, 5), Some(5));
    assert_eq!(super::grammar_rewind(6, 5), None);
}

/// 2026-09-25: No clamp to `min(dropped, steps)`. When the dropped tokens
/// recorded no steps, for example after the matcher terminated, every
/// recorded step belongs to a kept token, and a clamp would rewind those.
#[test]
fn it_does_not_silently_clamp_to_the_available_history() {
    assert_ne!(super::grammar_rewind(96, 1), Some(1));
    assert_eq!(super::grammar_rewind(96, 1), None);
}
