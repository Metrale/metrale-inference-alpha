// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for [`super`], the error hints.
//!
//! Owner: server API.
//! Invariants: none beyond the types.

use super::*;

#[test]
fn the_no_model_hint_names_the_way_out() {
    let h = hint_for("model_not_loaded").expect("the observed case must have a hint");
    assert!(h.contains("Library"), "{h}");
    assert!(h.contains("4"), "and how to get there: {h}");
    assert!(
        !h.to_lowercase().contains("no model is loaded"),
        "hint must not restate the message it is appended to: {h}"
    );
}

#[test]
fn unknown_types_get_no_hint_rather_than_a_guess() {
    assert_eq!(hint_for("invalid_request_error"), None);
    assert_eq!(hint_for(""), None);
    assert_eq!(hint_for("some_future_type"), None);
}

#[test]
fn message_with_hint_leaves_unhinted_messages_exactly_alone() {
    let m = "context length exceeded";
    assert_eq!(message_with_hint(m, "invalid_request_error"), m);
}

#[test]
fn message_with_hint_appends_for_known_types() {
    let out = message_with_hint("no model is loaded", "model_not_loaded");
    assert!(out.starts_with("no model is loaded"), "{out}");
    assert!(out.contains("Library"), "{out}");
}

#[test]
fn the_hint_survives_a_round_trip_through_a_real_response_body() {
    // 2026-09-26: The server puts the hint in `message` and
    // `metrale_bench::http::message_from_body` reads `message` back; this
    // checks both halves together.
    let body = serde_json::json!({
        "error": {
            "message": message_with_hint("no model is loaded", "model_not_loaded"),
            "type": "model_not_loaded",
            "hint": hint_for("model_not_loaded"),
        }
    })
    .to_string();

    let seen = metrale_bench::http::message_from_body(&body).expect("a well-formed body parses");
    assert!(
        seen.contains("Library"),
        "hint must survive the round trip: {seen}"
    );
}
