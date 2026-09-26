// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for how a turn with no parsed tool call is classified
//! (cut off, unparsed call, or finished), and for reasoning compaction.
//!
//! Owner: bench, agentic.
//! Invariants: none beyond the types.

use super::*;
use crate::http;

#[test]
fn a_turn_cut_off_at_the_token_cap_is_not_a_turn_that_finished() {
    // 2026-09-26: `length` with no tool call is a cut-off turn.
    let cut_off = http::ChatOutcome {
        text: "fn main() {".into(),
        finish_reason: Some("length".into()),
        ..Default::default()
    };
    assert!(was_cut_off(&cut_off), "length + no tool call is resumable");

    // 2026-09-26: A natural stop with no tool call is the agent finishing.
    let done = http::ChatOutcome {
        text: "All six steps pass.".into(),
        finish_reason: Some("stop".into()),
        ..Default::default()
    };
    assert!(!was_cut_off(&done), "a natural stop ends the run");

    // 2026-09-26: `length` with a tool call is not a cut-off turn; the call runs.
    let cut_off_with_call = http::ChatOutcome {
        tool_calls: vec![http::ToolCall {
            id: String::new(),
            name: "write".into(),
            arguments: "{}".into(),
        }],
        finish_reason: Some("length".into()),
        ..Default::default()
    };
    assert!(!was_cut_off(&cut_off_with_call));

    // 2026-09-26: No finish_reason is not a cut-off turn.
    let unknown = http::ChatOutcome::default();
    assert!(!was_cut_off(&unknown));
}

#[test]
fn a_tool_call_left_in_the_message_body_is_not_a_turn_that_finished() {
    // 2026-09-26: Tool-call syntax in the content, no parsed call, and
    // finish_reason `stop`.
    let degenerate = http::ChatOutcome {
        text: "Now let me test the other endpoints:\n\
               Now let me test the other endpoints:\n\
               <tool_call>\n<function=bash>\n<parameter=command>\n\
               timeout 15 curl -s http://localhost:3001/pong\n\
               </parameter>\n</function>\n</tool_call>oints:"
            .into(),
        finish_reason: Some("stop".into()),
        ..Default::default()
    };
    assert!(
        tools::emitted_unparsed_call(&degenerate),
        "tool-call syntax in the body with nothing parsed must be re-asked"
    );
    assert!(!was_cut_off(&degenerate));

    // 2026-09-26: Plain prose still ends the run.
    let done = http::ChatOutcome {
        text: "All six steps pass. The server is stopped.".into(),
        finish_reason: Some("stop".into()),
        ..Default::default()
    };
    assert!(!tools::emitted_unparsed_call(&done));

    // 2026-09-26: Prose about tool calls, without the markers, is plain text.
    let talks_about_it = http::ChatOutcome {
        text: "I would normally call the bash function to curl the endpoint.".into(),
        finish_reason: Some("stop".into()),
        ..Default::default()
    };
    assert!(!tools::emitted_unparsed_call(&talks_about_it));

    // 2026-09-26: A turn whose call parsed is not flagged, whatever its text
    // quotes.
    let parsed = http::ChatOutcome {
        text: "running <tool_call> now".into(),
        tool_calls: vec![http::ToolCall {
            id: String::new(),
            name: "bash".into(),
            arguments: "{}".into(),
        }],
        finish_reason: Some("stop".into()),
        ..Default::default()
    };
    assert!(!tools::emitted_unparsed_call(&parsed));
}

/// 2026-09-26: Old reasoning is replaced by a marker, never removed, and the
/// last `LIVE_REASONING` turns keep theirs in full.
#[test]
fn compaction_elides_old_reasoning_to_a_marker_and_keeps_the_recent() {
    let big = "r".repeat(20_000);
    let mut msgs = vec![json!({"role": "user", "content": "task"})];
    for i in 0..10 {
        msgs.push(json!({"role": "assistant", "content": Value::Null,
            "reasoning_content": big,
            "tool_calls": [{"id": format!("c{i}")}]}));
        msgs.push(json!({"role": "tool", "tool_call_id": format!("c{i}"), "content": "ok"}));
    }
    compact(&mut msgs);

    let think: Vec<&str> = msgs
        .iter()
        .filter_map(|m| m["reasoning_content"].as_str())
        .collect();
    assert_eq!(think.len(), 10, "reasoning must never be removed outright");
    assert!(
        think[0].contains("elided"),
        "the oldest reasoning should be elided: {}",
        &think[0][..think[0].len().min(60)]
    );
    for kept in think.iter().rev().take(LIVE_REASONING) {
        assert_eq!(*kept, big, "the live window keeps full reasoning");
    }
    let total: usize = msgs
        .iter()
        .map(|m| {
            m["content"].as_str().map_or(64, str::len)
                + m["reasoning_content"].as_str().map_or(0, str::len)
        })
        .sum();
    assert!(total <= HISTORY_BUDGET, "{total}");
}
