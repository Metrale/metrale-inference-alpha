// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for `misc_handlers::readiness`, the `/health` verdict. A fault from
//! the GPU fault latch outranks a loaded model: the answer is 503 `faulted` with the
//! reason, never 200 `ready`. Each test names a mutation of `readiness` that fails it.
//!
//! Owner: server API.
//! Invariants: none beyond the types.

use crate::api::misc_handlers::readiness;
use axum::http::StatusCode;

const REASON: &str = "kernel launch failed (716), and a no-op synchronize also failed";

/// 2026-09-26: A loaded model and no fault: 200 `ready`, with the model named. Fails if
/// `readiness` always returns the fault branch.
#[test]
fn a_loaded_model_with_no_fault_is_ready() {
    let (code, body) = readiness(Some("Qwen/Qwen3.6-27B-NVFP4"), None);
    assert_eq!(code, StatusCode::OK);
    assert_eq!(body["status"], "ready");
    assert_eq!(body["model"], "Qwen/Qwen3.6-27B-NVFP4");
}

/// 2026-09-26: No model and no fault: 503 `loading`. Fails if both `match model` arms
/// return OK.
#[test]
fn no_model_and_no_fault_is_still_loading() {
    let (code, body) = readiness(None, None);
    assert_eq!(code, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(body["status"], "loading");
}

/// 2026-09-26: A loaded model and a fault: 503 `faulted`. Fails with `200 ready` if the
/// `if let Some(reason) = fault` block is deleted.
#[test]
fn a_fault_outranks_a_published_model() {
    let (code, body) = readiness(Some("Qwen/Qwen3.6-27B-NVFP4"), Some(REASON));
    assert_eq!(
        code,
        StatusCode::SERVICE_UNAVAILABLE,
        "a published model on a dead context is NOT ready"
    );
    assert_eq!(body["status"], "faulted");
}

/// 2026-09-26: The fault body carries the reason. Fails if `reason` is dropped from the
/// fault JSON.
#[test]
fn the_fault_body_carries_the_reason() {
    let (_, body) = readiness(None, Some(REASON));
    assert_eq!(body["reason"], REASON);
}

/// 2026-09-26: A fault with no model loaded is `faulted`, not `loading`. Fails with
/// `loading` if `readiness` checks `model` before `fault`.
#[test]
fn a_fault_before_any_model_loaded_is_not_reported_as_loading() {
    let (code, body) = readiness(None, Some(REASON));
    assert_eq!(code, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(
        body["status"], "faulted",
        "a permanently dead server must not look like a starting one"
    );
}
