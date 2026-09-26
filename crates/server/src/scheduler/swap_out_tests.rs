// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests that swap-out (`lifecycle::swap_out_sequence`) and swap-in (`lifecycle::resume_swapped_seq`) failures reach the client.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.
//!
//! `swap_out_sequence` `swap_remove`s the victim out of `active` first, so
//! from then on it is the request's only owner, and an early return would
//! drop its `ResponseSink`. A dropped blocking sink reaches
//! `api::chat_blocking` as a closed oneshot, which it renders as
//! `500 "Inference cancelled"`. The tests therefore assert on what the
//! victim's receiver gets, not only on the `Err` return.

use super::io::{FileSpill, SchedIo};
use super::lifecycle::{resume_swapped_seq, swap_out_sequence};
use super::test_support::{PreemptStubModel, active_seq, streaming_seq};
use super::types::ActiveSeq;
use metrale_cache::kv_spill::KvSpillManager;
use std::sync::atomic::Ordering;

/// 2026-09-25: A spill pool in a per-test directory, emptied first.
fn spill(name: &str) -> FileSpill {
    let dir = std::env::temp_dir().join(format!(
        "metrale-swap-out-tests-{}-{name}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    FileSpill::new(KvSpillManager::new(dir, 1 << 20).expect("spill dir"))
}

/// 2026-09-25: The survivors at slots 0 and 2 around a victim placed at index 1.
/// `swap_remove(1)` moves slot 2 into index 1, and a slot that no longer
/// matches its index is the condition under which `swap_out_sequence`
/// compacts.
fn three_actives() -> (ActiveSeq, ActiveSeq) {
    let (a0, _rx0) = active_seq(0, 5);
    let (a2, _rx2) = active_seq(2, 9);
    (a0, a2)
}

#[test]
fn compaction_failure_reaches_the_blocking_client_instead_of_dropping_it() {
    let model = std::sync::Arc::new(PreemptStubModel::failing_compact(
        "compact_sequence: slot 1 still owned",
    ));
    let (a0, a2) = three_actives();
    let (victim, mut victim_rx) = active_seq(1, 2);
    let mut active = vec![a0, victim, a2];

    let r = swap_out_sequence(
        &SchedIo::for_test_with(model.clone()),
        &mut active,
        1,
        &spill("blocking"),
    );
    assert!(r.is_err(), "compaction failed, so the swap-out must fail");

    // 2026-09-25: a dropped sink would close the oneshot, which
    // `api::chat_blocking` renders as `500 "Inference cancelled"`; the
    // client must instead receive the swap-out error.
    let sent = victim_rx
        .try_recv()
        .expect("victim's sink was DROPPED — the client cannot tell this from its own abort");
    let Err(err) = sent else {
        panic!("a failed swap-out must not report success to the client");
    };
    let msg = format!("{err:#}");
    assert!(
        msg.contains("swap-out failed") && msg.contains("compact_sequence"),
        "the client must be told what actually failed, got: {msg}"
    );

    // 2026-09-25: the victim's sequence must also be freed on the way out.
    assert!(
        model.freed_slots.lock().unwrap().contains(&1),
        "victim slot never freed: {:?}",
        model.freed_slots.lock().unwrap()
    );
}

#[test]
fn compaction_failure_sends_a_terminal_frame_to_a_streaming_client() {
    let model = std::sync::Arc::new(PreemptStubModel::failing_compact(
        "compact_sequence: slot 1 still owned",
    ));
    let (a0, a2) = three_actives();
    let (victim, mut victim_rx) = streaming_seq(1, 2);
    let mut active = vec![a0, victim, a2];

    assert!(
        swap_out_sequence(
            &SchedIo::for_test_with(model.clone()),
            &mut active,
            1,
            &spill("streaming")
        )
        .is_err()
    );

    match victim_rx.try_recv() {
        Ok(crate::api::StreamEvent::Error(msg)) => assert!(
            msg.contains("swap-out failed"),
            "terminal frame must name the failure, got: {msg}"
        ),
        Ok(_) => panic!("expected an Error frame, got a different StreamEvent"),
        Err(e) => panic!(
            "no terminal frame on the victim's stream ({e:?}) — the body is \
             truncated under an already-committed HTTP 200"
        ),
    }
}

#[test]
fn a_successful_swap_out_still_hands_back_the_sink_untouched() {
    // 2026-09-25: on success nothing is sent on the sink: the request is
    // parked and resumes later.
    let model = std::sync::Arc::new(PreemptStubModel::default());
    let (a0, a2) = three_actives();
    let (victim, mut victim_rx) = active_seq(1, 2);
    let mut active = vec![a0, victim, a2];

    let s = swap_out_sequence(
        &SchedIo::for_test_with(model.clone()),
        &mut active,
        1,
        &spill("ok"),
    )
    .expect("swap-out");
    assert_eq!(s.output_tokens.len(), 2);
    assert_eq!(
        model.compact_calls.load(Ordering::SeqCst),
        1,
        "the migrated survivor must still be compacted"
    );
    assert!(
        matches!(
            victim_rx.try_recv(),
            Err(tokio::sync::oneshot::error::TryRecvError::Empty)
        ),
        "a parked request's client must hear nothing yet"
    );
}

// 2026-09-25: swap-in (`lifecycle::resume_swapped_seq`). The scheduler's
// swap-in loop does `swapped.remove(idx)`, hands the `SwappedSeq` to
// `resume_swapped_seq` by value and only logs its `Err`, so a failure
// inside that function must answer the client itself.

/// 2026-09-25: Park `victim` on disk through `swap_out_sequence`, so the `SwappedSeq` has
/// the shape the scheduler produces, and return the spill pool holding its
/// image.
fn park(
    model: &std::sync::Arc<PreemptStubModel>,
    victim: ActiveSeq,
    name: &str,
) -> (super::types::SwappedSeq, FileSpill) {
    let sp = spill(name);
    let (a0, a2) = three_actives();
    let mut active = vec![a0, victim, a2];
    let s = swap_out_sequence(&SchedIo::for_test_with(model.clone()), &mut active, 1, &sp)
        .expect("park");
    (s, sp)
}

#[test]
fn swap_in_failure_reaches_the_blocking_client_instead_of_dropping_it() {
    let model = std::sync::Arc::new(PreemptStubModel::default());
    let (victim, mut victim_rx) = active_seq(1, 2);
    let (s, sp) = park(&model, victim, "swapin-blocking");

    // 2026-09-25: the stub leaves `restore_sequence_state` at the trait
    // default, which fails, so the resume fails.
    let r = resume_swapped_seq(
        None,
        None,
        &*model,
        &SchedIo::for_test_with(model.clone()),
        s,
        &sp,
    );
    assert!(r.is_err(), "restore failed, so the resume must fail");

    let sent = victim_rx.try_recv().expect(
        "the parked request's sink was DROPPED — its client reads a server-side \
         swap-in failure as its own abort",
    );
    let Err(err) = sent else {
        panic!("a failed swap-in must not report success to the client");
    };
    assert!(
        format!("{err:#}").contains("swap-in failed"),
        "got: {err:#}"
    );
}

#[test]
fn swap_in_failure_sends_a_terminal_frame_to_a_streaming_client() {
    let model = std::sync::Arc::new(PreemptStubModel::default());
    let (victim, mut victim_rx) = streaming_seq(1, 2);
    let (s, sp) = park(&model, victim, "swapin-streaming");

    assert!(
        resume_swapped_seq(
            None,
            None,
            &*model,
            &SchedIo::for_test_with(model.clone()),
            s,
            &sp
        )
        .is_err()
    );

    match victim_rx.try_recv() {
        Ok(crate::api::StreamEvent::Error(msg)) => assert!(
            msg.contains("swap-in failed"),
            "terminal frame must name the failure, got: {msg}"
        ),
        Ok(_) => panic!("expected an Error frame, got a different StreamEvent"),
        Err(e) => panic!(
            "no terminal frame on the parked request's stream ({e:?}) — the body \
             is truncated under an already-committed HTTP 200"
        ),
    }
}
