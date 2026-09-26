// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The rate-limiter key of a request: who is asking. How fast
//! they may go is the parent module's job.
//!
//! Owner: server rate limiter.
//! Invariants: every key starts with `bearer:`, `xff:` or `peer:`, and a
//! bearer token appears in a key only as its FNV-1a hash.

/// 2026-09-26: The rate-limit key, first match wins: `bearer:<hash>` for a
/// non-empty `Authorization: Bearer` token, `xff:<ip>` for the first
/// non-empty `X-Forwarded-For` entry (taken as sent, not validated), then
/// `peer:<ip>`, or `peer:unknown` without a peer address. `rate_limit_middleware`
/// (`main_modules/middleware.rs`) keys the limiter with it.
pub fn extract_identity(
    headers: &axum::http::HeaderMap,
    peer: Option<std::net::SocketAddr>,
) -> String {
    use axum::http::header;
    if let Some(v) = headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        && let Some(tok) = v.strip_prefix("Bearer ")
    {
        let tok = tok.trim();
        if !tok.is_empty() {
            // 2026-09-26: Hashed so the limiter map never holds the raw token.
            return format!("bearer:{}", hash_token(tok));
        }
    }
    if let Some(xff) = headers.get("x-forwarded-for").and_then(|v| v.to_str().ok())
        && let Some(first) = xff.split(',').next()
    {
        let first = first.trim();
        if !first.is_empty() {
            return format!("xff:{first}");
        }
    }
    match peer {
        Some(addr) => format!("peer:{}", addr.ip()),
        None => "peer:unknown".to_string(),
    }
}

/// 2026-09-26: FNV-1a 64-bit hash as 16 hex digits: a stable opaque key, not
/// a cryptographic hash.
fn hash_token(tok: &str) -> String {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in tok.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    format!("{h:016x}")
}
