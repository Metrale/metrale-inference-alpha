// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: `GrammarState` at and after termination, the stop-token
//! exemption, and EBNF and multi-tool compilation.
//!
//! Owner: server (grammar) tests.
//! Invariants: none beyond the types.

use super::*;

#[test]
fn test_accept_token_after_termination_short_circuits() {
    let vocab = test_vocab();
    let stop_ids = vec![130i32];
    let mut engine = GrammarEngine::new(&vocab, &stop_ids).unwrap();

    let compiled = engine.compile_json_grammar().unwrap();
    let mut state = GrammarState::new(&compiled, engine.vocab_size()).unwrap();
    assert!(state.accept_token(b'{' as u32));
    assert!(state.accept_token(b'}' as u32));
    let _ = state.accept_token(130);
    assert!(
        state.is_terminated(),
        "grammar must reach terminated state for this test to be meaningful"
    );

    // 2026-09-26: A terminated matcher refuses every token, so
    // `accept_token` returns `true` without feeding it; draft truncation
    // then does not count tokens after the stop as grammar rejections.
    assert!(
        state.accept_token(198),
        "post-termination accept_token must short-circuit to true (token id 198 \
         was the specific id from the user report; any id should behave the same)"
    );
    assert!(
        state.accept_token(0),
        "post-termination accept_token must remain a no-op for any token id"
    );
}

#[test]
fn test_stop_token_exempt_from_grammar_refusal() {
    // 2026-09-26: A token registered with `with_stop_tokens` is accepted
    // mid-structure, where the matcher refuses the same token.
    let vocab = test_vocab();
    let mut engine = GrammarEngine::new(&vocab, &[130i32]).unwrap();
    let compiled = engine.compile_json_grammar().unwrap();
    let invalid = b'Z' as u32;

    let mut bare = GrammarState::new(&compiled, engine.vocab_size()).unwrap();
    assert!(bare.accept_token(b'{' as u32), "accept '{{'");
    assert!(!bare.is_terminated(), "matcher is mid-structure after '{{'");
    assert!(
        !bare.accept_token(invalid),
        "baseline: matcher refuses a non-grammar token mid-object"
    );

    let mut exempt = GrammarState::new(&compiled, engine.vocab_size())
        .unwrap()
        .with_stop_tokens(&[invalid]);
    assert!(exempt.accept_token(b'{' as u32), "accept '{{'");
    assert!(!exempt.is_terminated(), "still mid-structure before stop");
    assert!(
        exempt.accept_token(invalid),
        "stop/EOS token must be accepted unconditionally, exempt from grammar refusal"
    );
}

#[test]
fn test_fill_bitmask_after_stop_token() {
    let vocab = test_vocab();
    let stop_ids = vec![130i32];
    let mut engine = GrammarEngine::new(&vocab, &stop_ids).unwrap();

    // 2026-09-26: `{}` then the stop token terminates the matcher, which
    // `GrammarState` builds to terminate only on an accepted stop token.
    let compiled = engine.compile_json_grammar().unwrap();
    let mut state = GrammarState::new(&compiled, engine.vocab_size()).unwrap();
    assert!(state.accept_token(b'{' as u32), "accept '{{'");
    assert!(state.accept_token(b'}' as u32), "accept '}}'");
    let _ = state.accept_token(130);
    if state.is_terminated() {
        // 2026-09-26: The matcher's own fill panics once terminated; the
        // guard in `fill_bitmask` returns `false` first.
        let has_constraint = state.fill_bitmask();
        assert!(
            !has_constraint,
            "terminated grammar should report no constraint"
        );
        let _ = state.fill_bitmask();
    } else {
        panic!("grammar did not terminate; update the test to force termination");
    }
}

#[test]
fn test_ebnf_compilation() {
    let vocab = test_vocab();
    let stop_ids = vec![130i32];
    let mut engine = GrammarEngine::new(&vocab, &stop_ids).unwrap();

    let ebnf = r#"root ::= "hello""#;
    let result = engine.compile_ebnf(ebnf, "root");
    assert!(
        result.is_ok(),
        "EBNF compilation failed: {}",
        result.as_ref().err().unwrap()
    );
}

#[test]
fn test_extract_ordered_vocab() {
    // 2026-09-26: Empty: this test calls nothing.
}

#[test]
fn test_multiple_tools_hermes() {
    let vocab = test_vocab();
    let stop_ids = vec![130i32];
    let mut engine = GrammarEngine::new(&vocab, &stop_ids).unwrap();

    let tools = vec![
        ToolDefinition {
            tool_type: "function".to_string(),
            function: crate::tool_parser::FunctionDefinition {
                name: "get_weather".to_string(),
                description: Some("Get weather".to_string()),
                parameters: Some(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    },
                    "required": ["location"]
                })),
            },
        },
        ToolDefinition {
            tool_type: "function".to_string(),
            function: crate::tool_parser::FunctionDefinition {
                name: "search".to_string(),
                description: Some("Search the web".to_string()),
                parameters: Some(serde_json::json!({
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"}
                    },
                    "required": ["query"]
                })),
            },
        },
    ];

    let result = engine.compile_hermes_tool_grammar(&tools, false);
    assert!(
        result.is_ok(),
        "Multi-tool Hermes compilation failed: {}",
        result.as_ref().err().unwrap()
    );
}
