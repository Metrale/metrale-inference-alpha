// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for `compact::completion_error_frame`, the frame a
//! `StreamEvent::Error` becomes on the `/v1/completions` stream (`completions.rs`). The
//! stream has already answered HTTP 200, so this frame is where the client reads the
//! reason: it must be valid JSON that carries the message unchanged.
//!
//! Owner: server API.
//! Invariants: none beyond the types.

use crate::api::compact::completion_error_frame;

/// 2026-09-26: Error texts shaped like the scheduler's: `send_error_to_sink` callers pass
/// `{e:#}` anyhow chains (for example `scheduler/lifecycle.rs` "swap-in failed"), which
/// can hold quotes, newlines and backslashes.
const HOSTILE: &[&str] = &[
    r#"swap-in failed: open "/var/spill/swap_7.bin": No such file"#,
    "prefill failed: CUDA error\nlaunch failed",
    r"grammar compile failed near \x00",
    r#"tool "bash" rejected: unbalanced """#,
];

#[test]
fn an_error_message_survives_the_frame_as_parseable_json() {
    for msg in HOSTILE {
        let frame = completion_error_frame(msg);
        let v: serde_json::Value = serde_json::from_str(&frame).unwrap_or_else(|e| {
            panic!("frame for {msg:?} must be JSON a client can parse, got {frame:?}: {e}")
        });
        assert_eq!(
            v.get("error").and_then(|e| e.as_str()),
            Some(*msg),
            "frame must round-trip the reason verbatim: {frame}"
        );
    }
}

#[test]
fn the_wire_shape_is_unchanged_for_an_ordinary_message() {
    assert_eq!(
        completion_error_frame("Scheduler queue closed"),
        r#"{"error":"Scheduler queue closed"}"#
    );
}
