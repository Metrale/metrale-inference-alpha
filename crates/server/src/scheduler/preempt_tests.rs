// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests for decode-time KV preemption and resume
//! (`preempt.rs`, `choose_decode_victim`), run against `PreemptStubModel`.
//! They check each request's response channel as the API layer sees it.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use super::io::SchedIo;
use super::preempt::{
    PREEMPT_IMMUNITY_TOKENS, choose_decode_victim, decode_batch_with_preemption, preempt_requeue,
    resume_preempted_seq, resume_preempted_seqs,
};
use super::sched_ctx::SchedCtx;
use super::test_support::{PreemptStubModel, active_seq, streaming_seq};
use std::sync::atomic::{AtomicUsize, Ordering};

#[test]
fn kv_exhaustion_requeues_least_progress_victim_and_sends_nothing() {
    let model = std::sync::Arc::new(PreemptStubModel::failing(1));
    let (a0, _rx0) = active_seq(0, 5);
    let (victim, mut victim_rx) = streaming_seq(1, 2);
    let (a2, _rx2) = active_seq(2, 9);
    let mut active = vec![a0, victim, a2];
    let mut swapped = Vec::new();
    let mut preempted = Vec::new();

    let logits = decode_batch_with_preemption(
        &SchedCtx::for_test_with(model.clone()),
        &mut active,
        None,
        &mut swapped,
        &mut preempted,
        &mut Vec::new(),
    );

    assert!(logits.is_some());
    assert_eq!(model.decode_calls.load(Ordering::SeqCst), 2);
    assert_eq!(active.len(), 2);
    assert_eq!(preempted.len(), 1);
    assert!(swapped.is_empty());
    assert_eq!(preempted[0].a.output_tokens.len(), 2);
    assert_eq!(*model.freed_slots.lock().unwrap(), vec![1]);
    assert_eq!(model.cached_seqs.load(Ordering::SeqCst), 1);
    assert!(matches!(
        victim_rx.try_recv(),
        Err(tokio::sync::mpsc::error::TryRecvError::Empty)
    ));
    assert_eq!(preempted[0].tokens, vec![1, 2, 3, 4, 100]);
    assert_eq!(preempted[0].a.last_token, 101);
}

#[test]
fn non_kv_error_still_fails_the_whole_batch() {
    let model = std::sync::Arc::new(PreemptStubModel {
        hard_error: Some("CUDA error 700: illegal memory access"),
        ..Default::default()
    });
    let (a0, mut rx0) = active_seq(0, 3);
    let (a1, mut rx1) = active_seq(1, 4);
    let mut active = vec![a0, a1];
    let (mut swapped, mut preempted) = (Vec::new(), Vec::new());
    let logits = decode_batch_with_preemption(
        &SchedCtx::for_test_with(model.clone()),
        &mut active,
        None,
        &mut swapped,
        &mut preempted,
        &mut Vec::new(),
    );
    assert!(logits.is_none());
    assert!(active.is_empty() && preempted.is_empty() && swapped.is_empty());
    assert!(rx0.try_recv().expect("response sent").is_err());
    assert!(rx1.try_recv().expect("response sent").is_err());
}

#[test]
fn single_sequence_exhaustion_is_not_preemptible() {
    // 2026-09-25: The retry loop preempts only when more than one sequence
    // is active, so a lone sequence's exhaustion is terminal.
    let model = std::sync::Arc::new(PreemptStubModel {
        hard_error: Some("KV cache exhausted: no free blocks"),
        ..Default::default()
    });
    let (a0, mut rx0) = active_seq(0, 3);
    let mut active = vec![a0];
    let (mut swapped, mut preempted) = (Vec::new(), Vec::new());
    let logits = decode_batch_with_preemption(
        &SchedCtx::for_test_with(model.clone()),
        &mut active,
        None,
        &mut swapped,
        &mut preempted,
        &mut Vec::new(),
    );
    assert!(logits.is_none());
    assert!(preempted.is_empty());
    assert!(rx0.try_recv().expect("response sent").is_err());
}

#[test]
fn victim_policy_least_progress_wins() {
    let model = std::sync::Arc::new(PreemptStubModel::default());
    let (a0, _r0) = active_seq(0, 7);
    let (a1, _r1) = active_seq(1, 3);
    let (a2, _r2) = active_seq(2, 12);
    let active = vec![a0, a1, a2];
    assert_eq!(choose_decode_victim(&*model, &active, false), Some(1));
}

#[test]
fn victim_policy_starvation_guard_skips_resumed_until_progress() {
    let model = std::sync::Arc::new(PreemptStubModel::default());
    let (mut a0, _r0) = active_seq(0, 3);
    a0.preempt_immune_until_tokens = a0.output_tokens.len() + PREEMPT_IMMUNITY_TOKENS;
    let (a1, _r1) = active_seq(1, 9);
    let active = vec![a0, a1];
    assert_eq!(choose_decode_victim(&*model, &active, false), Some(1));

    let (mut a0, _r0) = active_seq(0, 3 + PREEMPT_IMMUNITY_TOKENS);
    a0.preempt_immune_until_tokens = 3 + PREEMPT_IMMUNITY_TOKENS;
    let (a1, _r1) = active_seq(1, 200);
    let active = vec![a0, a1];
    assert_eq!(choose_decode_victim(&*model, &active, false), Some(0));
}

#[test]
fn victim_policy_all_immune_still_yields_a_victim() {
    // 2026-09-25: With every candidate immune, one is still chosen, so
    // immunity never turns a recoverable exhaustion into a batch-wide error.
    let model = std::sync::Arc::new(PreemptStubModel::default());
    let (mut a0, _r0) = active_seq(0, 4);
    a0.preempt_immune_until_tokens = usize::MAX;
    let (mut a1, _r1) = active_seq(1, 2);
    a1.preempt_immune_until_tokens = usize::MAX;
    let active = vec![a0, a1];
    assert_eq!(choose_decode_victim(&*model, &active, false), Some(1));
}

#[test]
fn victim_policy_vision_requeue_excluded_but_spill_allowed() {
    const PAD: u32 = 999;
    let model = std::sync::Arc::new(PreemptStubModel {
        vision_pad: Some(PAD),
        ..Default::default()
    });
    let (mut a0, _r0) = active_seq(0, 2);
    a0.seq.tokens.insert(2, PAD);
    let (a1, _r1) = active_seq(1, 8);
    let active = vec![a0, a1];
    assert_eq!(choose_decode_victim(&*model, &active, false), Some(1));
    assert_eq!(choose_decode_victim(&*model, &active, true), Some(0));
}

#[test]
fn resume_reprefills_exact_history_and_preserves_stream_state() {
    let model = std::sync::Arc::new(PreemptStubModel::default());
    let (a, _rx) = active_seq(3, 6);
    let last_token = a.last_token;
    let out_before = a.output_tokens.clone();
    let remaining_before = a.remaining;
    let history = a.seq.tokens.clone();

    let p = preempt_requeue(&SchedIo::for_test_with(model.clone()), a);
    assert_eq!(p.tokens, history);
    assert_eq!(*model.freed_slots.lock().unwrap(), vec![3]);

    let resumed = resume_preempted_seq(&*model, &SchedIo::for_test_with(model.clone()), p)
        .expect("resume succeeds");
    assert_eq!(*model.prefilled.lock().unwrap(), vec![history.clone()]);
    assert_eq!(resumed.seq.tokens, history);
    assert_eq!(resumed.seq.seq_len, history.len());
    assert_eq!(resumed.last_token, last_token);
    assert_eq!(resumed.output_tokens, out_before);
    assert_eq!(resumed.remaining, remaining_before);
    assert!(!resumed.finished);
    assert_eq!(
        resumed.preempt_immune_until_tokens,
        out_before.len() + PREEMPT_IMMUNITY_TOKENS
    );
}

#[test]
fn resume_loop_gates_on_blocks_and_reclaims_from_prefix_cache() {
    let model = std::sync::Arc::new(PreemptStubModel {
        total_blocks: 100,
        free_blocks: AtomicUsize::new(0),
        reclaimable: AtomicUsize::new(50),
        ..Default::default()
    });
    let (a, _rx) = active_seq(0, 4);
    let p = {
        let mut history_seq = a;
        history_seq.seq.tokens = (0..32).collect();
        preempt_requeue(&SchedIo::for_test_with(model.clone()), history_seq)
    };
    let mut preempted = vec![p];
    let mut active = Vec::new();
    // 2026-09-25: 32 tokens at block size 16 need 32 / 16 + 1 = 3 blocks
    // plus 1 of growth. None are free, so the loop must reclaim.
    resume_preempted_seqs(
        &*model,
        &SchedIo::for_test_with(model.clone()),
        &mut active,
        &mut preempted,
        8,
        16,
    );
    assert_eq!(active.len(), 1);
    assert!(preempted.is_empty());

    let model2 = std::sync::Arc::new(PreemptStubModel {
        total_blocks: 100,
        ..Default::default()
    });
    let (a2, mut rx2) = active_seq(0, 4);
    let p2 = preempt_requeue(&SchedIo::for_test_with(model2.clone()), a2);
    let mut preempted2 = vec![p2];
    let mut active2 = Vec::new();
    resume_preempted_seqs(
        &*model2,
        &SchedIo::for_test_with(model2.clone()),
        &mut active2,
        &mut preempted2,
        8,
        16,
    );
    assert!(active2.is_empty());
    assert_eq!(preempted2.len(), 1);
    assert!(matches!(
        rx2.try_recv(),
        Err(tokio::sync::oneshot::error::TryRecvError::Empty)
    ));
}

#[test]
fn resume_loop_errors_out_a_sequence_that_can_never_fit() {
    let model = std::sync::Arc::new(PreemptStubModel {
        total_blocks: 2,
        ..Default::default()
    });
    let (a, mut rx) = active_seq(0, 4);
    let p = {
        let mut s = a;
        s.seq.tokens = (0..64).collect();
        preempt_requeue(&SchedIo::for_test_with(model.clone()), s)
    };
    let mut preempted = vec![p];
    let mut active = Vec::new();
    resume_preempted_seqs(
        &*model,
        &SchedIo::for_test_with(model.clone()),
        &mut active,
        &mut preempted,
        8,
        16,
    );
    assert!(preempted.is_empty() && active.is_empty());
    assert!(rx.try_recv().expect("error delivered").is_err());
}
