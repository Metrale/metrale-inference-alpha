// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests of the shutdown latch, the startup escape and the drain.
//! `PHASE` is process-global and the request flag has no reset, and other
//! tests in this binary trip it (Ctrl+C in `app_keys_tests`, `/quit` in
//! `commands_tests`) on threads of the same process, so every case asserts
//! from whatever state it finds.
//!
//! Owner: server tui.
//! Invariants: none beyond the types.

use super::*;

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()
        .expect("a current-thread runtime")
}

/// 2026-09-26: The latch goes up once and never comes down.
/// `bench_selfstart::claim_start_slot` and `model_swap` both refuse to start
/// once it is set.
#[test]
fn a_shutdown_request_is_a_one_way_latch_with_no_reset() {
    let rt = runtime();

    // 2026-09-26: The transition is observable only from an untripped
    // process, so it is asserted only when this test gets there first. The
    // idempotency below is asserted either way.
    if !requested() {
        rt.block_on(async {
            assert!(
                tokio::time::timeout(std::time::Duration::from_millis(50), wait())
                    .await
                    .is_err(),
                "wait() must not resolve before anything has been requested"
            );
        });
        request("SIGINT");
        assert!(requested(), "a request trips the latch");
    }

    // 2026-09-26: Further requests neither panic on the spent escape sender
    // nor clear anything.
    for _ in 0..3 {
        request("Ctrl+C");
        assert!(requested(), "the latch has no reset");
    }

    // 2026-09-26: A waiter created after the request resolves at once.
    rt.block_on(async {
        tokio::time::timeout(std::time::Duration::from_millis(500), wait())
            .await
            .expect("wait() resolves immediately once requested");
    });
}

/// 2026-09-26: The startup escape is taken while in startup and left parked
/// once disarmed. `request` sends on an armed escape whether or not the latch
/// was already tripped, so this holds wherever it is scheduled.
#[test]
fn the_startup_escape_is_taken_while_loading_and_parked_once_disarmed() {
    let (tx, mut rx) = oneshot::channel();
    arm_startup_escape(tx);
    request("escape-armed");
    // 2026-09-26: Only receipt is asserted, not the reason: a concurrent
    // test's request can be the one that takes the sender.
    let reason = rx
        .try_recv()
        .expect("a request while in startup hands its reason to main's one-shot");
    assert!(!reason.is_empty());

    disarm_startup_escape();

    // 2026-09-26: From here the escape is neither taken nor dropped, so the
    // receiver stays pending. Retried because a request that read
    // `in_startup` before the disarm can still take a sender armed after it;
    // the assertion needs one quiet window.
    let mut parked = false;
    for _ in 0..10 {
        let (tx, mut rx) = oneshot::channel();
        arm_startup_escape(tx);
        request("post-disarm");
        if matches!(rx.try_recv(), Err(oneshot::error::TryRecvError::Empty)) {
            parked = true;
            break;
        }
    }
    assert!(
        parked,
        "the parked sender must stay parked and its channel must stay open"
    );
    assert!(requested(), "and shutdown is still requested");
}

/// 2026-09-26: `REQUESTS_ACTIVE` is a process-global gauge, so the tests in
/// this file that read or move it are serialised on this lock.
static DRAIN: parking_lot::Mutex<()> = parking_lot::Mutex::new(());

/// 2026-09-26: Restores the gauge on drop, even if an assertion unwinds.
struct GaugeGuard(i64);

impl Drop for GaugeGuard {
    fn drop(&mut self) {
        crate::metrics::REQUESTS_ACTIVE.sub(self.0);
    }
}

fn hold_requests(n: i64) -> GaugeGuard {
    crate::metrics::REQUESTS_ACTIVE.add(n);
    GaugeGuard(n)
}

#[test]
fn an_idle_server_drains_immediately() {
    let _serial = DRAIN.lock();
    assert_eq!(
        crate::metrics::REQUESTS_ACTIVE.get(),
        0,
        "nothing in flight"
    );
    let start = std::time::Instant::now();
    runtime().block_on(drain_in_flight(std::time::Duration::from_secs(30)));
    assert!(
        start.elapsed() < std::time::Duration::from_secs(5),
        "a 30s grace must not be waited out when there is nothing to wait for"
    );
}

#[test]
fn a_stuck_request_gives_up_at_the_grace_window_rather_than_hanging() {
    // 2026-09-26: An expired grace returns with the request still counted.
    let _serial = DRAIN.lock();
    let _held = hold_requests(1);
    let start = std::time::Instant::now();
    runtime().block_on(drain_in_flight(std::time::Duration::ZERO));
    assert!(
        start.elapsed() < std::time::Duration::from_secs(5),
        "an expired grace returns instead of waiting on the request"
    );
    assert_eq!(
        crate::metrics::REQUESTS_ACTIVE.get(),
        1,
        "draining does not touch the gauge"
    );
}

/// 2026-09-26: A negative gauge, as a double decrement would leave, reads as
/// drained (`active <= 0`).
#[test]
fn a_gauge_below_zero_still_drains() {
    let _serial = DRAIN.lock();
    let _held = hold_requests(-1);
    let start = std::time::Instant::now();
    runtime().block_on(drain_in_flight(std::time::Duration::from_secs(30)));
    assert!(start.elapsed() < std::time::Duration::from_secs(5));
}

/// 2026-09-26: Pins what `drain_in_flight` assumes about the gauge it reads:
/// `metrics::ActiveRequestGuard` moves it.
#[test]
fn the_gauge_the_drain_reads_is_the_one_requests_move() {
    let _serial = DRAIN.lock();
    let before = crate::metrics::REQUESTS_ACTIVE.get();
    let guard = crate::metrics::ActiveRequestGuard::new();
    assert_eq!(crate::metrics::REQUESTS_ACTIVE.get(), before + 1);
    drop(guard);
    assert_eq!(crate::metrics::REQUESTS_ACTIVE.get(), before);
}
