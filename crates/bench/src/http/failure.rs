// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Why a streamed chat request failed, as data a caller can classify.
//!
//! Owner: bench (HTTP client).
//! Invariants:
//! - `Display` prints `message` and nothing else.
//! - A `body` set through `with_body` holds at most `BODY_EXCERPT` bytes and is
//!   cut on a char boundary.

use std::fmt;

/// 2026-09-25: The cap on the response bytes a failure carries.
pub const BODY_EXCERPT: usize = 320;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FailureKind {
    /// 2026-09-25: The client's deadline passed before the stream ended.
    Timeout,
    /// 2026-09-25: The status line carried this non-200 code.
    Status(u16),
    /// 2026-09-25: Connecting, writing or reading the socket failed.
    Transport,
    /// 2026-09-25: The response could not be decoded: bad framing, a status
    /// line without a code, or `data:` frames that are not JSON.
    Malformed,
    /// 2026-09-25: A 200 stream closed with no `[DONE]` and no finish_reason.
    Truncated,
    /// 2026-09-25: The server sent an in-band error frame under a 200.
    ServerError,
    /// 2026-09-25: A failure the client did not classify.
    Other,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RequestFailure {
    pub kind: FailureKind,
    pub message: String,
    /// 2026-09-25: The finish_reason streamed before the failure, if any.
    pub finish_reason: Option<String>,
    /// 2026-09-25: What the server sent that explains the failure: the error
    /// body, the error frame, or the first undecodable frame. Empty when
    /// nothing arrived.
    pub body: String,
}

impl RequestFailure {
    pub fn new(kind: FailureKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
            finish_reason: None,
            body: String::new(),
        }
    }

    pub fn with_body(mut self, bytes: &[u8]) -> Self {
        self.body = excerpt(bytes);
        self
    }

    pub fn with_finish_reason(mut self, finish_reason: Option<String>) -> Self {
        self.finish_reason = finish_reason;
        self
    }
}

impl fmt::Display for RequestFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for RequestFailure {}

/// 2026-09-25: At most `BODY_EXCERPT` bytes of `bytes`, lossily decoded,
/// trimmed, and never split inside a character.
pub fn excerpt(bytes: &[u8]) -> String {
    let text = String::from_utf8_lossy(bytes);
    let text = text.trim();
    let mut end = text.len().min(BODY_EXCERPT);
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    text[..end].to_string()
}

/// 2026-09-25: The code in a status line such as `HTTP/1.1 503 Service
/// Unavailable`, or `Malformed` when it has none.
pub fn status_kind(status_line: &str) -> FailureKind {
    status_line
        .split_whitespace()
        .nth(1)
        .and_then(|c| c.parse().ok())
        .map_or(FailureKind::Malformed, FailureKind::Status)
}

/// 2026-09-25: The failure an error from `chat_stream` describes. A typed
/// failure anywhere in the chain wins; an I/O error is `Transport`; anything
/// else is `Other`. The message is always the whole chain, as `{:#}` prints it.
pub fn classify(e: &anyhow::Error) -> RequestFailure {
    let message = format!("{e:#}");
    if let Some(f) = e.chain().find_map(|c| c.downcast_ref::<RequestFailure>()) {
        return RequestFailure {
            message,
            ..f.clone()
        };
    }
    let kind = if e.chain().any(|c| c.is::<std::io::Error>()) {
        FailureKind::Transport
    } else {
        FailureKind::Other
    };
    RequestFailure::new(kind, message)
}
