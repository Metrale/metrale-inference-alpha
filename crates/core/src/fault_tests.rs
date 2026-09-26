// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests for `classify`, `FaultLatch` and `exit_code`.
//!
//! Each behaviour has a fatal case and a non-fatal case: a missed fault leaves
//! a dead server reporting healthy, and a false one stops a healthy server.
//!
//! Owner: core.
//! Invariants: only `global_starts_healthy` touches the process-wide latch.

use super::*;

/// 2026-09-25: A fatal-looking error text, used to show that classification
/// does not key off the error text.
const STICKY_716: &str = "CUDA_ERROR_MISALIGNED_ADDRESS (716): misaligned address";

#[test]
fn failed_probe_means_context_lost() {
    let v = classify(
        "w4a16_gemm_t launch",
        STICKY_716,
        Err("cuStreamSynchronize returned 716".into()),
    );
    match v {
        Fatality::ContextLost(reason) => {
            assert!(reason.contains("w4a16_gemm_t launch"), "reason: {reason}");
            assert!(reason.contains(STICKY_716), "reason: {reason}");
            assert!(
                reason.contains("cuStreamSynchronize returned 716"),
                "reason: {reason}"
            );
        }
        Fatality::Isolated => panic!("a failed probe must be fatal"),
    }
}

/// 2026-09-25: The same fatal-looking text with a healthy probe is not fatal,
/// so classification by error code or text would fail here.
#[test]
fn scary_error_text_with_a_healthy_probe_is_not_fatal() {
    assert_eq!(
        classify("some launch", STICKY_716, Ok(())),
        Fatality::Isolated,
    );
}

#[test]
fn isolated_failure_with_healthy_probe_is_not_fatal() {
    assert_eq!(
        classify("cuMemAlloc", "CUDA_ERROR_OUT_OF_MEMORY (2)", Ok(())),
        Fatality::Isolated,
    );
}

#[test]
fn fresh_latch_is_healthy() {
    let l = FaultLatch::new();
    assert!(!l.is_faulted());
    assert_eq!(l.fault(), None);
}

#[test]
fn latching_records_the_reason() {
    let l = FaultLatch::new();
    assert!(l.latch("context destroyed by 716"));
    assert!(l.is_faulted());
    assert_eq!(l.fault(), Some("context destroyed by 716"));
}

#[test]
fn latch_is_first_writer_wins() {
    let l = FaultLatch::new();
    assert!(l.latch("first: the launch that poisoned the context"));
    assert!(
        !l.latch("second: a downstream cuMemsetD8Async echo"),
        "a second latch must report that it was not first"
    );
    assert_eq!(
        l.fault(),
        Some("first: the launch that poisoned the context"),
        "the diagnostic (first) fault must survive the echoes"
    );
}

/// 2026-09-25: Eight threads latch at once and exactly one is told it was
/// first; the fault probe logs only on that `true`.
#[test]
fn exactly_one_caller_wins_the_race() {
    use std::sync::{Arc, Barrier};
    let l = Arc::new(FaultLatch::new());
    let start = Arc::new(Barrier::new(9));
    let winners: usize = std::thread::scope(|s| {
        let handles: Vec<_> = (0..8)
            .map(|i| {
                let l = Arc::clone(&l);
                let start = Arc::clone(&start);
                s.spawn(move || {
                    start.wait();
                    usize::from(l.latch(format!("thread {i}")))
                })
            })
            .collect();
        start.wait();
        handles.into_iter().map(|h| h.join().unwrap()).sum()
    });
    assert_eq!(winners, 1, "exactly one latch call may report first");
    assert!(l.fault().is_some(), "the winner's reason must be present");
}

/// 2026-09-25: The process-wide latch starts healthy. This is the only test
/// that touches it, because a latched global cannot be reset for other tests.
#[test]
fn global_starts_healthy() {
    assert!(!global().is_faulted());
    assert_eq!(global().fault(), None);
}

#[test]
fn a_faulted_shutdown_exits_nonzero() {
    assert_ne!(exit_code(true, Some("context destroyed")), 0);
    assert_eq!(exit_code(true, Some("context destroyed")), EXIT_GPU_FAULT);
}

#[test]
fn a_fault_outranks_a_failed_run() {
    assert_eq!(exit_code(false, Some("context destroyed")), EXIT_GPU_FAULT);
}

#[test]
fn a_clean_shutdown_without_a_fault_still_exits_zero() {
    assert_eq!(exit_code(true, None), 0);
}

#[test]
fn an_ordinary_failure_is_not_reported_as_a_gpu_fault() {
    assert_eq!(exit_code(false, None), 1);
    assert_ne!(exit_code(false, None), EXIT_GPU_FAULT);
}
