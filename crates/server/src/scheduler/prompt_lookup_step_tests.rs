// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-02: Unit tests for the scheduler side of prompt lookup: the drafter
//! trim rule and settling a copy from the verdict's `seq_len` advance. The
//! end-to-end behaviour is covered by `trace_harness::prompt_lookup_tests`.

use super::*;
use crate::scheduler::test_support::active_seq;
use metrale_speculative::prompt_lookup::{PromptLookupConfig, PromptLookupSeq};

const CFG: PromptLookupConfig = PromptLookupConfig {
    ngram: 2,
    max_drafts: 3,
    max_seqs: 8,
    min_match: 2,
    miss_backoff: 0,
};

/// 2026-10-02: A sequence with a 3-token copy in flight.
fn with_copy() -> ActiveSeq {
    let (mut a, _rx) = active_seq(0, 1);
    let mut pl = PromptLookupSeq::new(&CFG);
    assert!(pl.propose(&[1, 2, 3, 4, 5, 1, 2], 3).is_some());
    a.prompt_lookup = Some(Box::new(pl));
    a
}

#[test]
fn the_drafter_keeps_its_accepts_only_for_its_own_drafts() {
    let (mut a, _rx) = active_seq(0, 1);
    assert_eq!(drafter_accepted(&a, 2), 2);
    a.prompt_lookup = Some(Box::new(PromptLookupSeq::new(&CFG)));
    assert_eq!(drafter_accepted(&a, 2), 2, "state without a copy in flight");
    let a = with_copy();
    assert_eq!(copy_in_flight(&a), 3);
    assert_eq!(
        drafter_accepted(&a, 2),
        0,
        "a copy is a full reject for the drafter"
    );
}

#[test]
fn settling_reads_the_accepts_from_the_seq_len_advance() {
    let sched = crate::scheduler::sched_ctx::SchedCtx::for_test();
    let mut active = vec![with_copy(), with_copy()];
    for a in active.iter_mut() {
        a.seq.seq_len = 100;
    }
    // 2026-10-02: Sequence 0 accepted 2 of 3 (+ the bonus row); sequence 1
    // accepted all 3.
    active[0].seq.seq_len = 103;
    active[1].seq.seq_len = 104;
    settle_copies(&mut active, &sched, &[(0, 100), (1, 100)]);
    assert_eq!(sched.prompt_lookup_stats.get(), [2, 6, 5]);
    for a in &active {
        assert_eq!(copy_in_flight(a), 0, "no mark outlives its verify");
    }
    // 2026-10-02: The partial accept halved the window, the full accept kept
    // it at its ceiling.
    assert_eq!(
        active[0].prompt_lookup.as_ref().map(|p| p.window()),
        Some(1)
    );
    assert_eq!(
        active[1].prompt_lookup.as_ref().map(|p| p.window()),
        Some(3)
    );
}

#[test]
fn a_sequence_that_did_not_advance_settles_as_a_miss() {
    let sched = crate::scheduler::sched_ctx::SchedCtx::for_test();
    let mut active = vec![with_copy()];
    active[0].seq.seq_len = 50;
    settle_copies(&mut active, &sched, &[(0, 50)]);
    assert_eq!(sched.prompt_lookup_stats.get(), [1, 3, 0]);
}

#[test]
fn long_copies_are_bucketed_to_powers_of_two() {
    let mk = |n: u32| (0..n).collect::<Vec<u32>>();
    for (n, want) in [
        (1, 1),
        (2, 2),
        (3, 3),
        (4, 4),
        (5, 4),
        (7, 4),
        (8, 8),
        (15, 8),
        (16, 16),
    ] {
        assert_eq!(bucket_copy(mk(n)).len(), want, "copy of {n}");
    }
    assert_eq!(
        bucket_copy(mk(6)),
        vec![0, 1, 2, 3],
        "a bucket keeps the copy's prefix"
    );
}
