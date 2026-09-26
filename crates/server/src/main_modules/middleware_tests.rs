// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for the middleware's pure parts: `apply_compat_stubs` and
//! `fault_rejection`.
//!
//! Owner: server (HTTP layer).
//! Invariants: none beyond the types.

#[test]
fn the_compat_stubs_do_not_overwrite_real_rate_limit_headers() {
    use axum::http::{HeaderMap, HeaderName, HeaderValue};
    let mut headers = HeaderMap::new();
    headers.insert(
        HeaderName::from_static("x-ratelimit-limit-requests"),
        HeaderValue::from_static("3"),
    );
    super::apply_compat_stubs(&mut headers);

    assert_eq!(
        headers
            .get("x-ratelimit-limit-requests")
            .and_then(|v| v.to_str().ok()),
        Some("3"),
        "the limiter's real value survives"
    );
    assert_eq!(
        headers
            .get("x-ratelimit-limit-tokens")
            .and_then(|v| v.to_str().ok()),
        Some("1000000000"),
        "a field the limiter did not set still gets its stub"
    );
}

const GPU_FAULT: &str = "the CUDA context is destroyed";

#[test]
fn inference_is_refused_once_the_gpu_has_faulted() {
    let got = super::fault_rejection("/v1/chat/completions", Some(GPU_FAULT));
    let (code, body) = got.expect("a faulted server must refuse inference");
    assert_eq!(code, axum::http::StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(body["error"]["code"], "gpu_fault");
    assert_eq!(body["error"]["type"], "server_error");
    assert!(
        body["error"]["message"]
            .as_str()
            .is_some_and(|m| m.contains(GPU_FAULT)),
        "the reason must reach the client: {body}"
    );
}

#[test]
fn a_healthy_server_admits_inference() {
    assert!(super::fault_rejection("/v1/chat/completions", None).is_none());
}

#[test]
fn health_endpoints_still_answer_while_faulted() {
    for path in ["/health", "/health/live", "/metrics", "/hardware"] {
        assert!(
            super::fault_rejection(path, Some(GPU_FAULT)).is_none(),
            "{path} must remain reachable during a fault"
        );
    }
}
