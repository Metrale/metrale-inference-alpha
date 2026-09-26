// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Compile tests of the Hermes, qwen3_coder and MiniMax XML tool
//! grammars, including the empty tool list.
//!
//! Owner: server (grammar).
//! Invariants: none beyond the types.

use super::*;

#[test]
fn test_hermes_tool_grammar_compilation() {
    let vocab = test_vocab();
    let stop_ids = vec![130i32];
    let mut engine = GrammarEngine::new(&vocab, &stop_ids).unwrap();

    let tools = test_tool_defs();
    let result = engine.compile_hermes_tool_grammar(&tools, false);
    assert!(
        result.is_ok(),
        "Hermes grammar compilation failed: {}",
        result.as_ref().err().unwrap()
    );
}

#[test]
fn test_qwen3_coder_tool_grammar_compilation() {
    let vocab = test_vocab();
    let stop_ids = vec![130i32];
    let mut engine = GrammarEngine::new(&vocab, &stop_ids).unwrap();

    let tools = test_tool_defs();
    let result = engine.compile_qwen3_coder_tool_grammar(&tools, false, "</parameter>");
    assert!(
        result.is_ok(),
        "Qwen3 coder grammar compilation failed: {}",
        result.as_ref().err().unwrap()
    );
}

#[test]
fn test_no_tools_error() {
    let vocab = test_vocab();
    let stop_ids = vec![130i32];
    let mut engine = GrammarEngine::new(&vocab, &stop_ids).unwrap();

    let result = engine.compile_hermes_tool_grammar(&[], false);
    assert!(matches!(result, Err(GrammarError::NoTools)));
}

#[test]
fn test_minimax_xml_tool_grammar_compilation() {
    let vocab = test_vocab();
    let stop_ids = vec![130i32];
    let mut engine = GrammarEngine::new(&vocab, &stop_ids).unwrap();

    let tools = test_tool_defs();
    let result = engine.compile_minimax_xml_tool_grammar(&tools, false);
    assert!(
        result.is_ok(),
        "MiniMax XML grammar compilation failed: {}",
        result.as_ref().err().unwrap()
    );
}

#[test]
fn test_minimax_xml_tool_grammar_no_tools() {
    let vocab = test_vocab();
    let stop_ids = vec![130i32];
    let mut engine = GrammarEngine::new(&vocab, &stop_ids).unwrap();

    let result = engine.compile_minimax_xml_tool_grammar(&[], false);
    assert!(matches!(result, Err(GrammarError::NoTools)));
}
