// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for the chat-completions wire format: `token_ids` appears only when
//! set, and reasoning is serialized only as `reasoning_content`.
//!
//! Owner: server (OpenAI API layer) tests.
//! Invariants: none beyond the types.

use crate::openai::*;

#[test]
fn token_ids_absent_by_default_keeps_wire_byte_identical() {
    let chunk = ChatCompletionChunk::content_chunk("m", "id", "hi".into());
    let json = serde_json::to_string(&chunk).unwrap();
    assert!(!json.contains("token_ids"), "default wire changed: {json}");
    let chunk =
        ChatCompletionChunk::content_chunk("m", "id", "hi".into()).with_token_ids(Vec::new());
    let json = serde_json::to_string(&chunk).unwrap();
    assert!(!json.contains("token_ids"));
}

#[test]
fn with_token_ids_stamps_first_choice() {
    let chunk =
        ChatCompletionChunk::content_chunk("m", "id", "hi".into()).with_token_ids(vec![10, 20, 30]);
    assert_eq!(chunk.choices[0].token_ids, vec![10, 20, 30]);
    let json = serde_json::to_string(&chunk).unwrap();
    assert!(json.contains("\"token_ids\":[10,20,30]"), "{json}");
    let usage = Usage {
        prompt_tokens: 1,
        completion_tokens: 1,
        total_tokens: 2,
        prompt_tokens_details: None,
        completion_tokens_details: None,
        time_to_first_token_ms: 0.0,
        response_tokens_per_second: 0.0,
        decode_time_ms: 0.0,
        total_time_ms: 0.0,
    };
    let chunk = ChatCompletionChunk::usage_only_chunk("m", "id", usage).with_token_ids(vec![1, 2]);
    assert!(chunk.choices.is_empty());
}

#[test]
fn reasoning_delta_emits_only_reasoning_content() {
    let chunk = ChatCompletionChunk::reasoning_chunk("m", "id", "thinking".into());
    let json = serde_json::to_string(&chunk).unwrap();
    assert!(
        json.contains("\"reasoning_content\":\"thinking\""),
        "reasoning_content missing: {json}"
    );
    assert!(
        !json.contains("\"reasoning\":"),
        "mirror `reasoning` field leaked into stream delta: {json}"
    );
}

#[test]
fn blocking_message_emits_only_reasoning_content() {
    let msg = ChatMessage {
        role: "assistant".into(),
        reasoning_content: Some("thinking".into()),
        content: Some("hi".into()),
        tool_calls: None,
        annotations: None,
        refusal: None,
    };
    let json = serde_json::to_string(&msg).unwrap();
    assert!(
        json.contains("\"reasoning_content\":\"thinking\""),
        "reasoning_content missing: {json}"
    );
    assert!(
        !json.contains("\"reasoning\":"),
        "mirror `reasoning` field leaked into message: {json}"
    );
}

fn test_usage() -> Usage {
    Usage {
        prompt_tokens: 1,
        completion_tokens: 1,
        total_tokens: 2,
        prompt_tokens_details: None,
        completion_tokens_details: None,
        time_to_first_token_ms: 0.0,
        response_tokens_per_second: 0.0,
        decode_time_ms: 0.0,
        total_time_ms: 0.0,
    }
}

#[test]
fn new_response_carries_reasoning_content() {
    let resp = ChatCompletionResponse::new(
        "m",
        "hi".into(),
        Some("thinking".into()),
        test_usage(),
        "stop",
    );
    assert_eq!(
        resp.choices[0].message.reasoning_content.as_deref(),
        Some("thinking")
    );
    let json = serde_json::to_string(&resp).unwrap();
    assert!(
        json.contains("\"reasoning_content\":\"thinking\""),
        "reasoning_content missing: {json}"
    );
    let resp = ChatCompletionResponse::new("m", "hi".into(), None, test_usage(), "stop");
    let json = serde_json::to_string(&resp).unwrap();
    assert!(!json.contains("reasoning_content"), "{json}");
}

#[test]
fn tool_call_response_carries_reasoning_content() {
    let resp = ChatCompletionResponse::with_tool_calls(
        "m",
        None,
        Some("need the tool".into()),
        vec![crate::tool_parser::ToolCall {
            id: "call_1".into(),
            call_type: "function".into(),
            function: crate::tool_parser::FunctionCall {
                name: "get_weather".into(),
                arguments: "{}".into(),
            },
        }],
        test_usage(),
    );
    assert_eq!(
        resp.choices[0].message.reasoning_content.as_deref(),
        Some("need the tool")
    );
    assert_eq!(resp.choices[0].finish_reason, "tool_calls");
    let json = serde_json::to_string(&resp).unwrap();
    assert!(
        json.contains("\"reasoning_content\":\"need the tool\""),
        "reasoning_content missing: {json}"
    );
}
