// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests that the streaming `/v1/completions` request carries `min_tokens`.
//!
//! Owner: server completions API.
//! Invariants: none beyond the types.

use super::*;

#[test]
fn streaming_path_wires_min_tokens() {
    let req: CompletionRequest = serde_json::from_value(serde_json::json!({
        "model": "m",
        "prompt": "hi",
        "min_tokens": 2048,
    }))
    .unwrap();
    let (token_tx, _token_rx) = tokio::sync::mpsc::channel::<StreamEvent>(16);
    let request = build_streaming_request(
        &req,
        crate::api::completions_exec::CompletionParams::test(),
        std::sync::Arc::new(vec![1, 2, 3]),
        42,
        None,
        token_tx,
    );
    assert_eq!(request.min_tokens(), 2048);
}

#[test]
fn streaming_path_defaults_min_tokens_to_zero() {
    let req: CompletionRequest = serde_json::from_value(serde_json::json!({
        "model": "m",
        "prompt": "hi",
    }))
    .unwrap();
    let (token_tx, _token_rx) = tokio::sync::mpsc::channel::<StreamEvent>(16);
    let request = build_streaming_request(
        &req,
        crate::api::completions_exec::CompletionParams::test(),
        std::sync::Arc::new(vec![1]),
        42,
        None,
        token_tx,
    );
    assert_eq!(request.min_tokens(), 0);
}
