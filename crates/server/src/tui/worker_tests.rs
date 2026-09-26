// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests of `worker::spawn`: the result arrives, the thread is
//! named, a dropped receiver is harmless, and a panic disconnects.
//!
//! Owner: server tui.
//! Invariants: none beyond the types.

use super::*;

#[test]
fn the_result_arrives_on_the_channel() {
    let rx = spawn("metrale-test", || 42u32, |_| 0);
    assert_eq!(rx.recv().expect("a result"), 42);
}

#[test]
fn the_thread_is_named_so_it_is_identifiable_in_a_dump() {
    let rx = spawn(
        "metrale-named",
        || std::thread::current().name().map(str::to_string),
        |_| None,
    );
    assert_eq!(rx.recv().unwrap().as_deref(), Some("metrale-named"));
}

#[test]
fn a_dropped_receiver_does_not_panic_the_worker() {
    // 2026-09-26: A send to a dropped receiver is ignored, not a panic on the
    // worker thread.
    let rx = spawn("metrale-dropped", || 1u8, |_| 0);
    drop(rx);
    // 2026-09-26: Give the worker time to run and attempt its send.
    std::thread::sleep(std::time::Duration::from_millis(50));
}

#[test]
fn work_that_panics_disconnects_rather_than_hanging() {
    // 2026-09-26: A panicking `work` leaves the receiver disconnected rather
    // than empty forever.
    let rx = spawn("metrale-panicky", || -> u8 { panic!("boom") }, |_| 0);
    match rx.recv() {
        Err(_) => {}
        Ok(v) => panic!("a panicking worker must not produce a value: {v}"),
    }
}
