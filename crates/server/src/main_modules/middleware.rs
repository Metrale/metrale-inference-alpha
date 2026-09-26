// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Axum middleware for the HTTP router: OpenAI-style response
//! headers, bearer-token auth, rate limiting and the GPU-fault gate.
//!
//! Owner: server (HTTP layer).
//! Invariants: none beyond the types.

use std::sync::Arc;

use crate::rate_limiter;

/// 2026-09-26: Add OpenAI-style headers to every `/v1/*` response:
/// `x-request-id` (the client's, else a new `req_<uuid>`),
/// `openai-processing-ms` (milliseconds until the inner layers returned a
/// response), and the fallbacks of [`apply_compat_stubs`].
pub(crate) async fn openai_observability_middleware(
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    use axum::http::{HeaderName, HeaderValue};
    let start = std::time::Instant::now();
    let is_v1 = req.uri().path().starts_with("/v1/");
    let incoming_req_id = req
        .headers()
        .get("x-request-id")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());
    let mut resp = next.run(req).await;
    if !is_v1 {
        return resp;
    }
    let headers = resp.headers_mut();
    let rid = incoming_req_id.unwrap_or_else(|| format!("req_{}", crate::ids::uuid_v4()));
    if let Ok(v) = HeaderValue::from_str(&rid) {
        headers.insert(HeaderName::from_static("x-request-id"), v);
    }
    let elapsed_ms = start.elapsed().as_millis();
    if let Ok(v) = HeaderValue::from_str(&elapsed_ms.to_string()) {
        headers.insert(HeaderName::from_static("openai-processing-ms"), v);
    }
    apply_compat_stubs(headers);
    resp
}

/// 2026-09-26: Bearer-token gate, active only with `--require-auth`. Then
/// `/v1/*`, `/tokenize` and `/detokenize` need `Authorization: Bearer <token>`
/// matching a loaded token, else 401 with code `missing_api_key` or
/// `invalid_api_key`; every other path passes.
///
/// Each candidate is compared in constant time (`AuthConfig::validate`).
pub(crate) async fn require_auth_middleware(
    // 2026-09-26: The host, not an `Arc<AppState>`: a clone bound for the
    // router's lifetime would keep `request_tx` open and block a swap's
    // scheduler join.
    axum::extract::State(host): axum::extract::State<Arc<super::model_host::ModelHost>>,
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    // 2026-09-26: The policy comes from the host, so it applies with no model
    // loaded. The path is checked before the policy is read.
    let path = req.uri().path();
    let needs_auth = path.starts_with("/v1/") || path == "/tokenize" || path == "/detokenize";
    if !needs_auth {
        return next.run(req).await;
    }
    let Some(auth_cfg) = host.auth() else {
        return next.run(req).await;
    };
    let presented_token = req
        .headers()
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.strip_prefix("Bearer "))
        .map(str::trim);
    let (status, code, message) = match presented_token {
        None => (
            axum::http::StatusCode::UNAUTHORIZED,
            "missing_api_key",
            "Missing Authorization: Bearer header",
        ),
        Some(t) if !auth_cfg.validate(t.as_bytes()) => (
            axum::http::StatusCode::UNAUTHORIZED,
            "invalid_api_key",
            "Invalid bearer token",
        ),
        Some(_) => return next.run(req).await,
    };
    let body = serde_json::json!({
        "error": {
            "message": message,
            "type": "invalid_request_error",
            "param": null,
            "code": code,
        }
    });
    (status, axum::Json(body)).into_response()
}

/// 2026-09-26: Per-client rate limiting of `/v1/*`, when
/// `METRALE_RATE_LIMIT_RPM` or `METRALE_RATE_LIMIT_TPM` is above 0; otherwise
/// a pass-through. A denied request gets 429 with an OpenAI error body and
/// `retry-after`. Allowed and denied responses carry the `x-ratelimit-*`
/// headers from [`apply_rate_headers`].
///
/// The client is the bearer token, else the first `X-Forwarded-For` entry,
/// else the peer address (`rate_limiter::extract_identity`).
pub(crate) async fn rate_limit_middleware(
    axum::extract::State(host): axum::extract::State<Arc<super::model_host::ModelHost>>,
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    use axum::http::{HeaderName, HeaderValue, StatusCode};
    use axum::response::IntoResponse;

    // 2026-09-26: The path is checked before any lock is taken.
    if !req.uri().path().starts_with("/v1/") {
        return next.run(req).await;
    }
    let Some(rate_limiter) = host.rate_limiter() else {
        return next.run(req).await;
    };
    if !rate_limiter.config().is_enabled() {
        return next.run(req).await;
    }
    // 2026-09-26: With no model loaded a request reserves no tokens but still
    // counts as a request.
    let max_seq_len = host.current().map(|state| state.max_seq_len).unwrap_or(0);

    // 2026-09-26: Set by `into_make_service_with_connect_info`
    // (`serve_router.rs`); `None` where the router runs without it.
    let peer = req
        .extensions()
        .get::<axum::extract::ConnectInfo<std::net::SocketAddr>>()
        .map(|ci| ci.0);
    let identity = rate_limiter::extract_identity(req.headers(), peer);

    // 2026-09-26: Reserve `max_seq_len` tokens. The chat handlers refund the
    // unused part through the `RequestContext` stored below (`refund_tokens`).
    let estimated = max_seq_len as u64;

    let decision = rate_limiter.admit(&identity, estimated);
    if !decision.allowed {
        let (param, code) = match decision.denied_by {
            Some(rate_limiter::DenialReason::Requests) => ("requests", "rate_limit_exceeded"),
            Some(rate_limiter::DenialReason::Tokens) => ("tokens", "rate_limit_exceeded"),
            None => ("", "rate_limit_exceeded"),
        };
        let body = serde_json::json!({
            "error": {
                "message": format!("Rate limit exceeded for {param}. Retry after {}s.", decision.retry_after_secs),
                "type": "rate_limit_exceeded",
                "param": param,
                "code": code,
            }
        });
        let mut resp = (StatusCode::TOO_MANY_REQUESTS, axum::Json(body)).into_response();
        apply_rate_headers(resp.headers_mut(), &decision);
        if let Ok(v) = HeaderValue::from_str(&decision.retry_after_secs.to_string()) {
            resp.headers_mut()
                .insert(HeaderName::from_static("retry-after"), v);
        }
        return resp;
    }

    let mut req = req;
    req.extensions_mut().insert(rate_limiter::RequestContext {
        identity: identity.clone(),
        reserved_tokens: estimated,
    });
    let mut resp = next.run(req).await;
    apply_rate_headers(resp.headers_mut(), &decision);
    resp
}

/// 2026-09-26: Add the fixed OpenAI-style headers a response lacks. A header
/// already set is kept: this layer runs outside `rate_limit_middleware`
/// (`serve_router.rs`), whose `x-ratelimit-*` values must reach the client.
fn apply_compat_stubs(headers: &mut axum::http::HeaderMap) {
    use axum::http::{HeaderName, HeaderValue};
    for (k, v) in [
        ("x-ratelimit-limit-requests", "1000000"),
        ("x-ratelimit-remaining-requests", "999999"),
        ("x-ratelimit-reset-requests", "0s"),
        ("x-ratelimit-limit-tokens", "1000000000"),
        ("x-ratelimit-remaining-tokens", "999999999"),
        ("x-ratelimit-reset-tokens", "0s"),
        ("openai-organization", "metrale-local"),
        ("openai-version", "2026-01-01"),
    ] {
        let name = HeaderName::from_static(k);
        if headers.contains_key(&name) {
            continue;
        }
        if let Ok(val) = HeaderValue::from_str(v) {
            headers.insert(name, val);
        }
    }
}

pub(crate) fn apply_rate_headers(
    headers: &mut axum::http::HeaderMap,
    d: &rate_limiter::RateDecision,
) {
    use axum::http::{HeaderName, HeaderValue};
    let set = |h: &mut axum::http::HeaderMap, k: &'static str, v: String| {
        if let Ok(val) = HeaderValue::from_str(&v) {
            h.insert(HeaderName::from_static(k), val);
        }
    };
    set(
        headers,
        "x-ratelimit-limit-requests",
        d.requests.limit.to_string(),
    );
    set(
        headers,
        "x-ratelimit-remaining-requests",
        d.requests.remaining.to_string(),
    );
    set(
        headers,
        "x-ratelimit-reset-requests",
        format!("{}s", d.requests.reset_secs),
    );
    set(
        headers,
        "x-ratelimit-limit-tokens",
        d.tokens.limit.to_string(),
    );
    set(
        headers,
        "x-ratelimit-remaining-tokens",
        d.tokens.remaining.to_string(),
    );
    set(
        headers,
        "x-ratelimit-reset-tokens",
        format!("{}s", d.tokens.reset_secs),
    );
}

/// 2026-09-26: The 503 response, in the OpenAI error envelope, for a `/v1/*`
/// request once the GPU fault latch is set; `None` for any other path or while
/// healthy.
pub(crate) fn fault_rejection(
    path: &str,
    fault: Option<&str>,
) -> Option<(axum::http::StatusCode, serde_json::Value)> {
    // 2026-09-26: `/health*` keeps answering, and reports the fault.
    if !path.starts_with("/v1/") {
        return None;
    }
    let reason = fault?;
    Some((
        axum::http::StatusCode::SERVICE_UNAVAILABLE,
        serde_json::json!({
            "error": {
                "message": format!(
                    "The server's GPU context is unrecoverable and it is shutting down: \
                     {reason}"
                ),
                "type": "server_error",
                "code": "gpu_fault",
            }
        }),
    ))
}

/// 2026-09-26: Refuse `/v1/*` requests with 503 once the GPU fault latch
/// (`metrale_core::fault`) is set.
pub(crate) async fn gpu_fault_middleware(
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    match fault_rejection(req.uri().path(), metrale_core::fault::global().fault()) {
        None => next.run(req).await,
        Some((code, body)) => (code, axum::Json(body)).into_response(),
    }
}

#[cfg(test)]
#[path = "middleware_tests.rs"]
mod tests;
