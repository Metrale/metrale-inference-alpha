// SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(unused_imports, dead_code)]

//! 2026-09-26: Tool-parser tests: `subagent_type` backfill for delegation
//! tools, and tool names that keep a `:` after normalisation.
//!
//! Owner: server (tool parser) tests.
//! Invariants: none beyond the types.

use super::super::*;

#[test]
fn backfill_fills_omitted_subagent_type_with_valid_agent() {
    // 2026-09-26: An omitted `subagent_type` is inserted as `""` and then
    // filled with an agent name listed in the tool description.
    let input = "<tool_call>\n\
        <function=task>\n\
        <parameter=description>\nFind hot spots\n</parameter>\n\
        <parameter=prompt>\nexplore the repo\n</parameter>\n\
        </function>\n\
        </tool_call>";
    let (_c, mut calls) = parse_tool_calls(input);
    assert_eq!(calls.len(), 1);

    let tool = ToolDefinition {
        tool_type: "function".to_string(),
        function: FunctionDefinition {
            name: "task".to_string(),
            description: Some(
                "Launch a new agent to handle complex tasks.\n\
                 Available agent types and the tools they have access to:\n\
                 - explore: Fast agent specialized for exploring codebases.\n\
                 - general: General-purpose agent for researching questions."
                    .to_string(),
            ),
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "prompt": {"type": "string"},
                    "subagent_type": {"type": "string"}
                },
                "required": ["description", "prompt", "subagent_type"]
            })),
        },
    };
    backfill_required_params(&mut calls, std::slice::from_ref(&tool));
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(
        args["subagent_type"], "general",
        "omitted subagent_type must be filled with a valid agent, not empty"
    );
    assert!(
        validate_single_tool_call(&calls[0], std::slice::from_ref(&tool)).is_ok(),
        "the repaired call must validate"
    );
}

#[test]
fn backfill_subagent_type_prefers_general_purpose_variant() {
    // 2026-09-26: The first listed name containing `general` wins over the
    // agents listed before it.
    let input = "<tool_call>\n\
        <function=Task>\n\
        </function>\n\
        </tool_call>";
    let (_c, mut calls) = parse_tool_calls(input);
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.arguments, "{}");

    let tool = ToolDefinition {
        tool_type: "function".to_string(),
        function: FunctionDefinition {
            name: "Task".to_string(),
            description: Some(
                "Launch a new agent.\n\
                 - claude-code-guide: Use this agent for Claude Code questions.\n\
                 - Explore: Fast codebase exploration agent.\n\
                 - general-purpose: General-purpose agent for complex questions.\n\
                 - statusline-setup: Configure the status line."
                    .to_string(),
            ),
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {"subagent_type": {"type": "string"}},
                "required": ["subagent_type"]
            })),
        },
    };
    backfill_required_params(&mut calls, std::slice::from_ref(&tool));
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["subagent_type"], "general-purpose");
}

// 2026-09-26: The bare-identifier scanners accept `:` so `ns:tool{...}`
// parses. `normalize_tool_name` strips a namespace only when a tool name
// follows it, so prose like `json:{...}` keeps its `:`; such a candidate is
// refused without consuming the text (`is_normalized_tool_name`).

#[test]
fn phantom_json_colon_prose_is_not_a_tool_call() {
    let input = r#"Here is the payload as json:{"a":1}"#;
    let (content, calls) = parse_tool_calls(input);
    assert!(
        calls.is_empty(),
        "prose `json:{{...}}` must not become a phantom call — got {calls:#?}"
    );
    let content = content.expect("original text must be preserved as content");
    assert!(
        content.contains(r#"json:{"a":1}"#),
        "content must keep the unconsumed text, got: {content}"
    );
}

#[test]
fn namespaced_bare_identifier_still_parses_as_tool() {
    let input = r#"ns:tool{"query":"rust"}"#;
    let (_, calls) = parse_tool_calls(input);
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "tool");
}

#[test]
fn tool_call_colon_prefix_falls_through_to_embedded_json_name() {
    // 2026-09-26: `tool_call:` is refused as a name, and the JSON fallback
    // then takes the embedded `"name"`.
    let input = r#"tool_call:{"name":"get_weather","arguments":{"city":"Paris"}}"#;
    let (_, calls) = parse_tool_calls(input);
    assert_eq!(calls.len(), 1, "expected one call — got {calls:#?}");
    assert_eq!(calls[0].function.name, "get_weather");
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["city"], "Paris");
}
