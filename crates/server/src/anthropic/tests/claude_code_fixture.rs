// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Lowering tests on a captured Claude Code request: its system
//! prompt (`scripts/fixtures/claude_code_system_prompt.txt`) and its tool list
//! (`scripts/fixtures/claude_code_tools.json`, 70 tools).
//!
//! Owner: server (Anthropic adapter) tests.
//! Invariants: none beyond the types.

use super::super::types::MessagesRequest;
use crate::ir::{ChatRequest, ContentPart, Role};

/// 2026-09-26: The captured system prompt. A `--dump` run records a new one
/// as the `system` field of a `/v1/messages` request entry.
pub(super) fn system_prompt() -> String {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../scripts/fixtures/claude_code_system_prompt.txt"
    );
    std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("claude_code_system_prompt.txt fixture missing at {path}: {e}"))
}

/// 2026-09-26: The captured tool array.
pub(super) fn tools() -> serde_json::Value {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../scripts/fixtures/claude_code_tools.json"
    );
    let raw = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("claude_code_tools.json fixture missing at {path}: {e}"));
    serde_json::from_str(&raw).expect("claude_code_tools.json must be valid JSON")
}

fn text_of(parts: &[ContentPart]) -> String {
    parts
        .iter()
        .filter_map(|p| match p {
            ContentPart::Text(t) => Some(t.as_str()),
            _ => None,
        })
        .collect()
}

#[test]
fn fixtures_are_the_real_capture_not_a_stub() {
    let prompt = system_prompt();
    assert!(
        prompt.len() > 20_000,
        "real Claude Code prompt is >= 20 KB; got {} bytes — fixture may \
         have been replaced by the old short stub",
        prompt.len()
    );
    assert!(
        prompt.contains("Claude Code"),
        "real prompt mentions 'Claude Code'"
    );
    let arr = tools();
    let arr = arr.as_array().expect("tools is a JSON array");
    assert_eq!(arr.len(), 70, "real Claude Code session declares 70 tools");
}

#[test]
fn a_full_session_request_lowers_with_the_prompt_intact() {
    let sys = system_prompt();
    let user = "Build a Rust axum server with a /echo endpoint.";
    let req: MessagesRequest = serde_json::from_value(serde_json::json!({
        "model": "claude-sonnet-4-6",
        "max_tokens": 4096,
        "stream": true,
        "system": sys,
        "messages": [{"role": "user", "content": user}],
    }))
    .expect("MessagesRequest");
    let ir = ChatRequest::from(req);

    assert_eq!(ir.messages.len(), 2, "system + user");
    assert_eq!(ir.messages[0].role, Role::System);
    assert_eq!(text_of(&ir.messages[0].content), sys);
    assert_eq!(ir.messages[1].role, Role::User);
    assert_eq!(text_of(&ir.messages[1].content), user);
    assert_eq!(ir.model, "claude-sonnet-4-6");
    assert_eq!(ir.max_tokens, 4096);
    assert!(ir.stream);
}

#[test]
fn all_seventy_real_tool_schemas_survive_lowering() {
    // 2026-09-26: Each schema must lower unchanged: the prompt's tool
    // section is rendered from these definitions.
    let raw = tools();
    let names: Vec<String> = raw
        .as_array()
        .expect("tools array")
        .iter()
        .map(|t| t["name"].as_str().expect("tool name").to_string())
        .collect();

    let req: MessagesRequest = serde_json::from_value(serde_json::json!({
        "model": "claude-sonnet-4-6",
        "max_tokens": 4096,
        "system": system_prompt(),
        "tools": raw,
        "messages": [{"role": "user", "content": "go"}],
    }))
    .expect("MessagesRequest");
    let ir = ChatRequest::from(req);

    assert_eq!(ir.tools.len(), 70);
    let lowered: Vec<&str> = ir.tools.iter().map(|t| t.function.name.as_str()).collect();
    assert_eq!(lowered, names, "tool order and names must be preserved");
    assert!(
        ir.tools.iter().all(|t| t.tool_type == "function"),
        "every Anthropic tool lowers to an OpenAI function tool"
    );
    assert!(
        ir.tools.iter().all(|t| t.function.parameters.is_some()),
        "every real tool carries an input_schema"
    );

    let raw_again = tools();
    let first_schema = &raw_again.as_array().unwrap()[0]["input_schema"];
    assert_eq!(ir.tools[0].function.parameters.as_ref(), Some(first_schema));
}
