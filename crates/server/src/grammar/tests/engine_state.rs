// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: `GrammarEngine` compilation and `GrammarState` behaviour:
//! masks, accept, rollback, forced tokens and the grammar close.
//!
//! Owner: server (grammar) tests.
//! Invariants: none beyond the types.

use super::*;

#[test]
fn test_grammar_engine_creation() {
    let vocab = test_vocab();
    let stop_ids = vec![130i32];
    let engine = GrammarEngine::new(&vocab, &stop_ids);
    assert!(engine.is_ok());
    let engine = engine.unwrap();
    assert_eq!(engine.vocab_size(), vocab.len());
}

#[test]
fn test_json_schema_compilation() {
    let vocab = test_vocab();
    let stop_ids = vec![130i32];
    let mut engine = GrammarEngine::new(&vocab, &stop_ids).unwrap();

    let schema = r#"{
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name", "age"]
    }"#;

    let result = engine.compile_json_schema(schema);
    assert!(
        result.is_ok(),
        "JSON schema compilation failed: {}",
        result.as_ref().err().unwrap()
    );
}

#[test]
fn test_builtin_json_compilation() {
    let vocab = test_vocab();
    let stop_ids = vec![130i32];
    let mut engine = GrammarEngine::new(&vocab, &stop_ids).unwrap();

    let result = engine.compile_json_grammar();
    assert!(
        result.is_ok(),
        "Builtin JSON compilation failed: {}",
        result.as_ref().err().unwrap()
    );
}

#[test]
fn test_grammar_state_basic_json() {
    let vocab = test_vocab();
    let stop_ids = vec![130i32];
    let mut engine = GrammarEngine::new(&vocab, &stop_ids).unwrap();

    let compiled = engine.compile_json_grammar().unwrap();
    let mut state = GrammarState::new(&compiled, engine.vocab_size()).unwrap();

    assert!(!state.is_terminated());

    let has_constraint = state.fill_bitmask();
    assert!(has_constraint);

    assert!(state.is_token_allowed(b'{' as u32));
}

/// 2026-09-26: A `json_schema` grammar constrains the very first token:
/// `{` is legal, and the letters `H` and `s` are masked.
/// `sample_first_token` applies this initial-state mask to the first
/// sampled token.
#[test]
fn test_json_schema_masks_leading_prose_token_at_start() {
    let vocab = test_vocab();
    let stop_ids = vec![130i32];
    let mut engine = GrammarEngine::new(&vocab, &stop_ids).unwrap();

    let schema = r#"{
        "type": "object",
        "properties": {
            "answer": {"type": "string"}
        },
        "required": ["answer"]
    }"#;
    let compiled = engine.compile_json_schema(schema).unwrap();
    let mut state = GrammarState::new(&compiled, engine.vocab_size()).unwrap();

    assert!(
        state.fill_bitmask(),
        "json_schema grammar must constrain the first token"
    );

    assert!(
        state.is_token_allowed(b'{' as u32),
        "'{{' must be grammar-legal as the first token"
    );

    assert!(
        !state.is_token_allowed(b'H' as u32),
        "leading prose 'H' must be masked at generation-start (#131)"
    );
    assert!(
        !state.is_token_allowed(b's' as u32),
        "schema-name leak 's' (\"suggest\") must be masked at start (#131)"
    );
}

#[test]
fn test_grammar_state_accept_and_terminate() {
    let vocab = test_vocab();
    let stop_ids = vec![130i32];
    let mut engine = GrammarEngine::new(&vocab, &stop_ids).unwrap();

    let compiled = engine.compile_json_grammar().unwrap();
    let mut state = GrammarState::new(&compiled, engine.vocab_size()).unwrap();

    assert!(state.accept_token(b'{' as u32));
    assert!(state.accept_token(b'}' as u32));

    // 2026-09-26: After the complete value `{}` the stop token is legal.
    state.fill_bitmask();
    assert!(state.is_token_allowed(130));
}

#[test]
fn test_grammar_state_rollback() {
    let vocab = test_vocab();
    let stop_ids = vec![130i32];
    let mut engine = GrammarEngine::new(&vocab, &stop_ids).unwrap();

    let compiled = engine.compile_json_grammar().unwrap();
    let mut state = GrammarState::new(&compiled, engine.vocab_size()).unwrap();

    assert!(state.accept_token(b'{' as u32));
    assert!(state.accept_token(b'"' as u32));

    state.rollback(1);

    // 2026-09-26: Back to the position after `{`, where `}` is legal.
    state.fill_bitmask();
    assert!(state.is_token_allowed(b'}' as u32));
}

#[test]
fn test_apply_bitmask_to_logits() {
    let vocab = test_vocab();
    let stop_ids = vec![130i32];
    let mut engine = GrammarEngine::new(&vocab, &stop_ids).unwrap();

    let compiled = engine.compile_json_grammar().unwrap();
    let mut state = GrammarState::new(&compiled, engine.vocab_size()).unwrap();

    state.fill_bitmask();

    let mut logits = vec![1.0f32; engine.vocab_size()];
    state.apply_bitmask_to_logits(&mut logits);

    assert!(logits[b'{' as usize].is_finite());
    assert!(logits[1].is_infinite() && logits[1].is_sign_negative());
}

/// 2026-09-26: The only allowed token in `state`'s current bitmask, or
/// `None` when zero or two or more are allowed: the token any sampler must
/// pick once every other logit is `-inf`.
fn single_allowed_token(state: &GrammarState, vocab_size: usize) -> Option<u32> {
    let mut found: Option<u32> = None;
    for id in 0..vocab_size as u32 {
        if state.is_token_allowed(id) {
            if found.is_some() {
                return None;
            }
            found = Some(id);
        }
    }
    found
}

/// 2026-09-26: At every step of a walk, `forced_token()` equals the
/// single-allowed-token reference: `Some` exactly when one token is legal.
#[test]
fn test_forced_token_matches_masked_sample_path() {
    let vocab = test_vocab();
    let stop_ids = vec![130i32];
    let mut engine = GrammarEngine::new(&vocab, &stop_ids).unwrap();
    let vocab_size = engine.vocab_size();

    // 2026-09-26: The required key gives the walk forced steps; the test
    // asserts at least one.
    let schema = r#"{
        "type": "object",
        "properties": { "name": { "type": "string" } },
        "required": ["name"]
    }"#;
    let compiled = engine.compile_json_schema(schema).unwrap();
    let mut state = GrammarState::new(&compiled, vocab_size).unwrap();

    let mut saw_forced = false;
    for _ in 0..64 {
        if state.is_terminated() {
            break;
        }
        let has_constraint = state.fill_bitmask();
        assert!(has_constraint, "tool grammar always constrains");
        let reference = single_allowed_token(&state, vocab_size);
        let forced = state.forced_token().map(|t| t as u32);
        assert_eq!(
            forced, reference,
            "forced_token() must equal the single-allowed-token reference",
        );
        // 2026-09-26: Take the forced token, else the lowest allowed id.
        let next = match reference {
            Some(t) => {
                saw_forced = true;
                t
            }
            None => (0..vocab_size as u32)
                .find(|&id| state.is_token_allowed(id))
                .expect("non-terminated state has at least one allowed token"),
        };
        assert!(state.accept_token(next), "allowed token must be accepted");
    }
    assert!(saw_forced, "expected at least one grammar-forced token");
}

/// 2026-09-26: Accepting the forced token leaves the matcher with the same
/// next mask and termination state as accepting the single token left by
/// `fill_bitmask`, step by step until the first genuine choice.
#[test]
fn test_forced_token_accept_advances_identically() {
    let vocab = test_vocab();
    let stop_ids = vec![130i32];
    let mut engine = GrammarEngine::new(&vocab, &stop_ids).unwrap();
    let vocab_size = engine.vocab_size();

    let schema = r#"{
        "type": "object",
        "properties": { "name": { "type": "string" } },
        "required": ["name"]
    }"#;
    let compiled = engine.compile_json_schema(schema).unwrap();

    let mut fast = GrammarState::new(&compiled, vocab_size).unwrap();
    let mut slow = GrammarState::new(&compiled, vocab_size).unwrap();

    for step in 0..32 {
        if fast.is_terminated() || slow.is_terminated() {
            break;
        }
        let Some(forced) = fast.forced_token() else {
            break;
        };
        let forced = forced as u32;

        slow.fill_bitmask();
        let slow_tok = single_allowed_token(&slow, vocab_size)
            .expect("forced step on `fast` implies forced step on `slow`");
        assert_eq!(forced, slow_tok, "step {step}: same forced token");

        assert!(fast.accept_token(forced));
        assert!(slow.accept_token(slow_tok));
        let f_constraint = fast.fill_bitmask();
        let s_constraint = slow.fill_bitmask();
        assert_eq!(f_constraint, s_constraint, "step {step}: mask parity");
        for id in 0..vocab_size as u32 {
            assert_eq!(
                fast.is_token_allowed(id),
                slow.is_token_allowed(id),
                "step {step}: token {id} allowed-bit parity",
            );
        }
        assert_eq!(fast.is_terminated(), slow.is_terminated());
    }
}

/// 2026-09-26: `forced_token()` is `None` on a terminated matcher.
#[test]
fn test_forced_token_none_after_termination() {
    let vocab = test_vocab();
    let stop_ids = vec![130i32];
    let mut engine = GrammarEngine::new(&vocab, &stop_ids).unwrap();

    let compiled = engine.compile_json_grammar().unwrap();
    let mut state = GrammarState::new(&compiled, engine.vocab_size()).unwrap();

    assert!(state.accept_token(b'{' as u32));
    assert!(state.accept_token(b'}' as u32));
    assert!(state.accept_token(130));
    assert!(state.is_terminated());

    assert_eq!(
        state.forced_token(),
        None,
        "forced_token must decline on a terminated matcher",
    );
}

/// 2026-09-26: After `{`, both `"` and `}` are legal, so `forced_token()`
/// is `None`.
#[test]
fn test_forced_token_none_on_genuine_choice() {
    let vocab = test_vocab();
    let stop_ids = vec![130i32];
    let mut engine = GrammarEngine::new(&vocab, &stop_ids).unwrap();

    let compiled = engine.compile_json_grammar().unwrap();
    let mut state = GrammarState::new(&compiled, engine.vocab_size()).unwrap();

    assert!(state.accept_token(b'{' as u32));
    state.fill_bitmask();
    assert!(state.is_token_allowed(b'"' as u32));
    assert!(state.is_token_allowed(b'}' as u32));

    assert_eq!(
        state.forced_token(),
        None,
        "a two-way choice must not be reported as forced",
    );
}

/// 2026-09-26: Inside an open JSON string the stop token is illegal;
/// `completion_token_ids` returns a non-empty close whose tokens are all
/// accepted and after which the stop token is legal.
#[test]
fn test_budget_close_completes_open_json_string() {
    let vocab = test_vocab();
    let stop_ids = vec![130i32];
    let mut engine = GrammarEngine::new(&vocab, &stop_ids).unwrap();

    let schema = r#"{
        "type": "object",
        "properties": { "name": {"type": "string"} },
        "required": ["name"]
    }"#;
    let compiled = engine.compile_json_schema(schema).unwrap();
    let mut state = GrammarState::new(&compiled, engine.vocab_size()).unwrap();

    for ch in br#"{"name":"ab"# {
        assert!(
            state.accept_token(u32::from(*ch)),
            "byte {ch} must be grammar-legal while building the object",
        );
    }
    assert!(
        !state.stop_legal(&[130]),
        "EOS must be illegal inside an open JSON string",
    );

    let close = state
        .completion_token_ids(32)
        .expect("a bounded close exists for an open string");
    assert!(!close.is_empty(), "a non-empty close is required here");

    for tok in &close {
        assert!(
            state.accept_token(*tok as u32),
            "every close token must be grammar-legal",
        );
    }
    assert!(
        state.stop_legal(&[130]),
        "after the close, EOS is legal (output is parseable)",
    );
}

/// 2026-09-26: After the complete value `{}`, `stop_legal` is true and
/// the close is empty.
#[test]
fn test_budget_close_noop_when_already_stop_legal() {
    let vocab = test_vocab();
    let stop_ids = vec![130i32];
    let mut engine = GrammarEngine::new(&vocab, &stop_ids).unwrap();

    let compiled = engine.compile_json_grammar().unwrap();
    let mut state = GrammarState::new(&compiled, engine.vocab_size()).unwrap();

    assert!(state.accept_token(b'{' as u32));
    assert!(state.accept_token(b'}' as u32));

    assert!(state.stop_legal(&[130]), "a complete value may stop");
    assert_eq!(
        state.completion_token_ids(32),
        Some(Vec::new()),
        "already stop-legal: the close is empty",
    );
}
