// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The MiniMax XML tool grammar: accepted and rejected
//! envelopes, and the token-level mask around its multi-byte tag tokens.
//!
//! Owner: server (grammar) tests.
//! Invariants: none beyond the types.

use super::*;
use metrale_grammar::{CompiledGrammar, GrammarMatcher};

fn minimax_test_tool_defs() -> Vec<ToolDefinition> {
    vec![
        ToolDefinition {
            tool_type: "function".to_string(),
            function: crate::tool_parser::FunctionDefinition {
                name: "bash".to_string(),
                description: Some("Run a shell command".to_string()),
                parameters: Some(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"}
                    },
                    "required": ["command"]
                })),
            },
        },
        ToolDefinition {
            tool_type: "function".to_string(),
            function: crate::tool_parser::FunctionDefinition {
                name: "ls".to_string(),
                description: Some("List a directory".to_string()),
                parameters: Some(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"}
                    },
                    "required": ["path"]
                })),
            },
        },
    ]
}

/// 2026-09-26: Whether a fresh matcher accepts every byte of `input` and
/// ends complete. `terminate_without_stop_token` is `true`, so completion
/// needs no stop token.
fn grammar_accepts(compiled: &CompiledGrammar, input: &str) -> bool {
    let mut matcher =
        GrammarMatcher::new(compiled, None, true, -1).expect("GrammarMatcher::new failed");
    if !matcher.accept_string(input, false) {
        return false;
    }
    matcher.is_terminated()
}

/// 2026-09-26: Auto mode accepts a canonical call, with and without prose
/// before it. `test_minimax_xml_grammar_rejects_degenerate` covers the
/// strings it must reject.
#[test]
fn test_minimax_xml_grammar_accepts_canonical() {
    let vocab = test_vocab();
    let stop_ids = vec![130i32];
    let mut engine = GrammarEngine::new(&vocab, &stop_ids).unwrap();
    let tools = minimax_test_tool_defs();
    let compiled = engine
        .compile_minimax_xml_tool_grammar(&tools, true)
        .expect("compile must succeed");

    let canonical = "<minimax:tool_call>\n<invoke name=\"bash\">\n<parameter name=\"command\">uname -r</parameter>\n</invoke>\n</minimax:tool_call>";
    assert!(
        grammar_accepts(&compiled, canonical),
        "canonical bash invocation should be accepted, got reject. \
         input was: {canonical:?}"
    );

    let with_prelude = "Sure, let me check.\n<minimax:tool_call>\n<invoke name=\"ls\">\n<parameter name=\"path\">/etc</parameter>\n</invoke>\n</minimax:tool_call>";
    assert!(
        grammar_accepts(&compiled, with_prelude),
        "leading content before tool envelope should be accepted \
         (triggered_tags model). input was: {with_prelude:?}"
    );
}

/// 2026-09-26: The open-then-close rejection, on tokens. With
/// `<minimax:tool_call>` and `</minimax:tool_call>` as single tokens, the
/// open tag is allowed at the start, and after it `\n` is allowed while the
/// close-tag token and `<` are masked.
#[test]
fn test_minimax_xml_grammar_token_level_close_after_open_rejected() {
    // 2026-09-26: ASCII 0..=127, then the open tag (128), close tag (129)
    // and `<eos>` (130).
    let mut vocab: Vec<String> = (0u8..128).map(|i| (i as char).to_string()).collect();
    vocab.push("<minimax:tool_call>".to_string());
    vocab.push("</minimax:tool_call>".to_string());
    vocab.push("<eos>".to_string());
    let stop_ids = vec![130i32];
    let mut engine = GrammarEngine::new(&vocab, &stop_ids).unwrap();
    let tools = minimax_test_tool_defs();
    let compiled = engine
        .compile_minimax_xml_tool_grammar(&tools, true)
        .expect("compile must succeed");
    let mut state = GrammarState::new(&compiled, engine.vocab_size()).unwrap();

    let _ = state.fill_bitmask();
    assert!(
        state.is_token_allowed(128),
        "open token must be allowed in initial state"
    );

    assert!(
        state.accept_token(128),
        "accept_token for `<minimax:tool_call>` must succeed in initial state"
    );

    let constrained = state.fill_bitmask();
    assert!(
        constrained,
        "fill_bitmask must report a non-trivial constraint after the trigger fires"
    );

    assert!(
        state.is_token_allowed(10),
        "byte-token `\\n` (id 10) must be allowed as the first post-trigger byte"
    );

    assert!(
        !state.is_token_allowed(129),
        "post-trigger bitmask must MASK the close token `</minimax:tool_call>` \
         (id 129) — it starts with `<` which is invalid after the open"
    );

    assert!(
        !state.is_token_allowed(60),
        "byte-token `<` (id 60) must be masked post-trigger"
    );
}

/// 2026-09-26: Pins that, after the partial trigger `<minimax`, the
/// auto-mode grammar still allows a token (`:_`) whose bytes leave the
/// trigger. Outside a tag the auto-mode grammar accepts any text (the prose
/// case in `test_minimax_xml_grammar_accepts_canonical`). The test panics
/// if that token becomes masked.
#[test]
fn test_minimax_xml_grammar_masks_trigger_breaking_multibyte_token() {
    // 2026-09-26: ASCII, the two tags, `<eos>`, then `:_` (131), `min`
    // (132) and `imax` (133).
    let mut vocab: Vec<String> = (0u8..128).map(|i| (i as char).to_string()).collect();
    vocab.push("<minimax:tool_call>".to_string());
    vocab.push("</minimax:tool_call>".to_string());
    vocab.push("<eos>".to_string());
    vocab.push(":_".to_string());
    vocab.push("min".to_string());
    vocab.push("imax".to_string());
    let stop_ids = vec![130i32];
    let mut engine = GrammarEngine::new(&vocab, &stop_ids).unwrap();
    let tools = minimax_test_tool_defs();
    let compiled = engine
        .compile_minimax_xml_tool_grammar(&tools, true)
        .expect("compile must succeed");
    let mut state = GrammarState::new(&compiled, engine.vocab_size()).unwrap();

    assert!(state.accept_token(b'<' as u32), "accept '<'");
    assert!(state.accept_token(132), "accept 'min' (id 132)");
    assert!(state.accept_token(133), "accept 'imax' (id 133)");

    let _constrained = state.fill_bitmask();

    if state.is_token_allowed(131) {
        eprintln!(
            "F70 NOTE: xgrammar TagDispatch allows trigger-breaking \
             multi-byte token `:_` (id 131) after partial `<minimax` \
             match. Metrale Engine adds a runtime backstop because the matcher \
             alone can't anchor partial triggers across BPE merges."
        );
    } else {
        panic!(
            "MUST FAIL — xgrammar appears to now mask trigger-breaking \
             multi-byte tokens. Update the test to assert this strict \
             behavior and remove the F70 runtime backstop."
        );
    }
}

/// 2026-09-26: The grammar rejects an open tag closed at once, a re-open
/// before the close, and a tool name not in the list.
#[test]
fn test_minimax_xml_grammar_rejects_degenerate() {
    let vocab = test_vocab();
    let stop_ids = vec![130i32];
    let mut engine = GrammarEngine::new(&vocab, &stop_ids).unwrap();
    let tools = minimax_test_tool_defs();
    let compiled = engine
        .compile_minimax_xml_tool_grammar(&tools, true)
        .expect("compile must succeed");

    let close_immediate = "<minimax:tool_call></minimax:tool_call>";
    assert!(
        !grammar_accepts(&compiled, close_immediate),
        "degenerate close-immediate envelope must be rejected: {close_immediate:?}"
    );

    let reopen = "<minimax:tool_call><minimax:tool_call>";
    assert!(
        !grammar_accepts(&compiled, reopen),
        "re-open before close must be rejected: {reopen:?}"
    );

    let unknown_tool = "<minimax:tool_call>\n<invoke name=\"ghost\">\n<parameter name=\"x\">y</parameter>\n</invoke>\n</minimax:tool_call>";
    assert!(
        !grammar_accepts(&compiled, unknown_tool),
        "unknown tool name must be rejected: {unknown_tool:?}"
    );
}
