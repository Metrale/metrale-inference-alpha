// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The action a client can take about an HTTP error, keyed by its
//! `error.type`.
//!
//! Owner: server API.
//! Invariants: `message_with_hint` returns `message` unchanged for a type
//! that has no hint.
//!
//! The hint goes into `message` as well as into its own `hint` field. The
//! error-body readers, `metrale_bench::http::message_from_body` and
//! `error_message_from_response` (used by `tui/chat_stream.rs`), return only
//! `message`, and `metrale-bench` cannot import this table because
//! `metrale-server` depends on `metrale-bench`, not the reverse.

/// 2026-09-26: The hint for an `error.type`, or `None` for a type without one.
/// Keyed on the type, not the status: `model_not_loaded` and `not_ready` are
/// both 503 (`main_modules/model_host.rs`) and call for different actions.
pub fn hint_for(error_type: &str) -> Option<&'static str> {
    match error_type {
        "model_not_loaded" => Some(
            "open the Library (press 4 in the dashboard), choose a model and a \
             recipe, and start it; then retry this request",
        ),
        "not_ready" => Some("the socket binds before the model finishes loading, so retry shortly"),
        "shutting_down" => Some("the server is draining and will not accept new work"),
        _ => None,
    }
}

/// 2026-09-26: `message`, then ` — ` and the hint for `error_type` when there
/// is one. Every error body that carries a `hint` builds its `message` here.
pub fn message_with_hint(message: &str, error_type: &str) -> String {
    match hint_for(error_type) {
        Some(h) => format!("{message} — {h}"),
        None => message.to_string(),
    }
}

#[cfg(test)]
#[path = "error_hints_tests.rs"]
mod tests;
