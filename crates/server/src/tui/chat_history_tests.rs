// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests of which transcript entries `ChatState::send` sends as
//! history.
//!
//! Owner: server tui.
//! Invariants: none beyond the types.

use super::turn_tests::{body_of, pump_until_settled, runtime};
use super::*;
use crate::tui::chat_stream::fake::{serve, sse};

/// 2026-09-26: An earlier empty model turn is sent as an empty `assistant`
/// message; only the last transcript entry, the placeholder for this send, is
/// left out.
#[test]
fn a_cancelled_earlier_turn_stays_in_history_so_roles_keep_alternating() {
    let f = serve(|s| sse(s, &[b"data: [DONE]\n\n"]));
    let rt = runtime();
    let mut s = ChatState::default();
    s.set_runtime(rt.handle().clone());
    s.transcript
        .push(ChatMessage::new(Role::User, "first".into()));
    s.transcript.push(ChatMessage::new(Role::Model, "".into()));
    s.input = "second".into();
    s.send(f.port);
    pump_until_settled(&mut s);

    let sent = &body_of(&f)["messages"];
    assert_eq!(
        sent,
        &serde_json::json!([
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": ""},
            {"role": "user", "content": "second"},
        ]),
        "the answerless turn must be preserved, not filtered out"
    );

    let roles: Vec<&str> = sent
        .as_array()
        .expect("messages is an array")
        .iter()
        .map(|m| m["role"].as_str().expect("role is a string"))
        .collect();
    assert!(
        roles.windows(2).all(|w| w[0] != w[1]),
        "no two consecutive turns may share a role: {roles:?}"
    );
}

/// 2026-09-26: The empty model placeholder `send` pushes last is not sent.
#[test]
fn the_placeholder_for_the_current_send_is_still_excluded() {
    let f = serve(|s| sse(s, &[b"data: [DONE]\n\n"]));
    let rt = runtime();
    let mut s = ChatState::default();
    s.set_runtime(rt.handle().clone());
    s.input = "only".into();
    s.send(f.port);
    pump_until_settled(&mut s);
    assert_eq!(
        body_of(&f)["messages"],
        serde_json::json!([{"role": "user", "content": "only"}]),
        "the just-pushed empty model placeholder must not be sent"
    );
}
