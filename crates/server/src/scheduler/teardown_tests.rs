// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests for `quiesce_streams`: the model-release wait, run without a real `Model`.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use super::quiesce_streams;
use std::cell::RefCell;

/// 2026-09-25: Every stream is waited on, in order.
#[test]
fn every_stream_is_waited_on() {
    let synced = RefCell::new(Vec::<u64>::new());
    let failed = quiesce_streams(&[("default", 1), ("prefill", 2)], |s| {
        synced.borrow_mut().push(s);
        Ok(())
    });
    assert!(failed.is_empty());
    assert_eq!(synced.into_inner(), [1, 2]);
}

/// 2026-09-25: The second stream is waited on too; `core/finish.rs` passes the default
/// stream and the prefill stream.
#[test]
fn the_prefill_stream_is_not_forgotten() {
    let synced = RefCell::new(Vec::<u64>::new());
    quiesce_streams(&[("default", 7), ("prefill", 9)], |s| {
        synced.borrow_mut().push(s);
        Ok(())
    });
    assert_eq!(
        synced.into_inner(),
        [7, 9],
        "the prefill stream must be waited on too"
    );
}

/// 2026-09-25: A stream that will not synchronise is reported by name, and the sweep keeps
/// going so the remaining streams are still waited on.
#[test]
fn a_failed_sync_is_named_and_does_not_stop_the_sweep() {
    let synced = RefCell::new(Vec::<u64>::new());
    let failed = quiesce_streams(&[("default", 1), ("prefill", 2), ("extra", 3)], |s| {
        synced.borrow_mut().push(s);
        if s == 2 {
            anyhow::bail!("stream 2 is wedged")
        } else {
            Ok(())
        }
    });
    assert_eq!(failed, ["prefill"], "the failing stream is named");
    assert_eq!(
        synced.into_inner(),
        [1, 2, 3],
        "a failure must not skip the streams after it"
    );
}

/// 2026-09-25: Every stream failing names every stream; the caller logs one line each and
/// still releases.
#[test]
fn all_failures_are_reported() {
    let failed = quiesce_streams(&[("default", 1), ("prefill", 2)], |_| {
        anyhow::bail!("device is gone")
    });
    assert_eq!(failed, ["default", "prefill"]);
}

#[test]
fn no_streams_is_not_a_failure() {
    let called = RefCell::new(false);
    let failed = quiesce_streams(&[], |_| {
        *called.borrow_mut() = true;
        Ok(())
    });
    assert!(failed.is_empty());
    assert!(!called.into_inner(), "nothing to wait on, nothing called");
}
