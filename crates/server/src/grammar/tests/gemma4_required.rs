// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The Gemma-4 tool grammar in auto and required mode, checked
//! on byte strings.
//!
//! Auto mode (`use_triggers=true`) triggers on `<|tool_call>call:` and lets
//! the model write prose instead of a call. Required mode triggers on
//! `<|tool_call>` with `at_least_one` and `stop_after_first`. The tests pin:
//! the canonical call is accepted in both modes, required mode rejects
//! leading prose, and auto mode accepts prose alone.
//!
//! Owner: server (grammar) tests.
//! Invariants: none beyond the types.

use super::*;
use metrale_grammar::{CompiledGrammar, GrammarMatcher};

fn read_file_tool() -> Vec<ToolDefinition> {
    vec![ToolDefinition {
        tool_type: "function".to_string(),
        function: crate::tool_parser::FunctionDefinition {
            name: "read_file".to_string(),
            description: Some("Read a file from disk".to_string()),
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": { "path": {"type": "string"} },
                "required": ["path"]
            })),
        },
    }]
}

/// 2026-09-26: Whether a fresh matcher accepts every byte of `input` and
/// ends complete (`terminate_without_stop_token` is `true`, so completion
/// needs no stop token).
fn grammar_accepts(compiled: &CompiledGrammar, input: &str) -> bool {
    let mut matcher =
        GrammarMatcher::new(compiled, None, true, -1).expect("GrammarMatcher::new failed");
    if !matcher.accept_string(input, false) {
        return false;
    }
    matcher.is_terminated()
}

const CANONICAL: &str = "<|tool_call>call:read_file{\"path\": \"./Cargo.toml\"}<tool_call|>";

/// 2026-09-26: Required mode accepts the canonical Gemma-4 call.
#[test]
fn gemma4_required_accepts_canonical_call() {
    let vocab = test_vocab();
    let stop_ids = vec![130i32];
    let mut engine = GrammarEngine::new(&vocab, &stop_ids).unwrap();
    let compiled = engine
        .compile_gemma4_tool_grammar(&read_file_tool(), false)
        .expect("required-mode gemma4 grammar must compile");
    assert!(
        grammar_accepts(&compiled, CANONICAL),
        "required-mode grammar must accept the canonical call; input: {CANONICAL:?}"
    );
}

/// 2026-09-26: Auto mode accepts the canonical call.
#[test]
fn gemma4_auto_accepts_canonical_call() {
    let vocab = test_vocab();
    let stop_ids = vec![130i32];
    let mut engine = GrammarEngine::new(&vocab, &stop_ids).unwrap();
    let compiled = engine
        .compile_gemma4_tool_grammar(&read_file_tool(), true)
        .expect("auto-mode gemma4 grammar must compile");
    assert!(
        grammar_accepts(&compiled, CANONICAL),
        "auto-mode grammar must accept the canonical call; input: {CANONICAL:?}"
    );
}

/// 2026-09-26: Required mode rejects prose before the call.
#[test]
fn gemma4_required_rejects_leading_prose() {
    let vocab = test_vocab();
    let stop_ids = vec![130i32];
    let mut engine = GrammarEngine::new(&vocab, &stop_ids).unwrap();
    let compiled = engine
        .compile_gemma4_tool_grammar(&read_file_tool(), false)
        .expect("required-mode gemma4 grammar must compile");
    assert!(
        !grammar_accepts(&compiled, "Sure, let me read that file for you."),
        "required mode must REJECT leading prose (constrained from token 1)"
    );
}

/// 2026-09-26: Auto mode accepts prose with no call; with the test above it
/// shows the verdict follows `use_triggers`.
#[test]
fn gemma4_auto_allows_free_prose() {
    let vocab = test_vocab();
    let stop_ids = vec![130i32];
    let mut engine = GrammarEngine::new(&vocab, &stop_ids).unwrap();
    let compiled = engine
        .compile_gemma4_tool_grammar(&read_file_tool(), true)
        .expect("auto-mode gemma4 grammar must compile");
    assert!(
        grammar_accepts(&compiled, "Just a plain text answer, no tool needed."),
        "auto mode must ALLOW free prose (no forced tool call)"
    );
}
