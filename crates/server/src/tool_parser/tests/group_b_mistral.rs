// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tool-parser tests for Mistral `[TOOL_CALLS]name[ARGS]{…}` calls.
//!
//! Owner: server (tool parser) tests.
//! Invariants: none beyond the types.

use super::super::*;

#[test]
fn parse_mistral_single_call() {
    let input = "[TOOL_CALLS]get_weather[ARGS]{\"location\":\"Paris\"}";
    let (c, calls) = parse_tool_calls(input);
    assert!(c.is_none());
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "get_weather");
    let args: serde_json::Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
    assert_eq!(args["location"], "Paris");
}

#[test]
fn parse_mistral_multiple_calls() {
    let input = "[TOOL_CALLS]search[ARGS]{\"q\":\"rust\"}[TOOL_CALLS]summarize[ARGS]{\"text\":\"found it\"}";
    let (_, calls) = parse_tool_calls(input);
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].function.name, "search");
    assert_eq!(calls[1].function.name, "summarize");
    let a1: serde_json::Value = serde_json::from_str(&calls[1].function.arguments).unwrap();
    assert_eq!(a1["text"], "found it");
}

#[test]
fn parse_mistral_with_leading_content() {
    let input = "Let me check.[TOOL_CALLS]get_weather[ARGS]{\"city\":\"Tokyo\"}";
    let (c, calls) = parse_tool_calls(input);
    assert_eq!(c.unwrap(), "Let me check.");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "get_weather");
}
