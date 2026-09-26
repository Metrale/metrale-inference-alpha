// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Several calls in one turn with the Hermes grammar, and
//! `grammar_blocks_stop` around them.
//!
//! In auto mode (`tool_choice="auto"`) the matcher accepts a second
//! `<tool_call>…</tool_call>` block after the first closes. That needs the
//! single-token `</tool_call>` (id 129 here) to reach the matcher, so it
//! must not be in the `with_stop_tokens` set; the last test shows the
//! matcher stuck when it is.
//!
//! Owner: server (grammar) tests.
//! Invariants: none beyond the types.

use super::*;

const TOOL_CALL_OPEN: u32 = 128;
const TOOL_CALL_CLOSE: u32 = 129;
const EOS: u32 = 130;

/// 2026-09-26: Feed one complete Hermes call: the open and close tags as
/// single tokens, the body between them one byte per token.
fn drive_one_call(state: &mut GrammarState, city: &str) {
    assert!(
        state.accept_token(TOOL_CALL_OPEN),
        "<tool_call> open must be accepted"
    );
    // 2026-09-26: The tag's begin literal is
    // `<tool_call>\n{"name": "get_weather", "arguments": ` and its end
    // literal `}\n</tool_call>`; the arguments schema fills the middle.
    let body =
        format!("\n{{\"name\": \"get_weather\", \"arguments\": {{\"location\":\"{city}\"}}}}\n");
    for (i, b) in body.bytes().enumerate() {
        assert!(
            state.accept_token(u32::from(b)),
            "body byte {i} ({:?}) must be grammar-legal",
            b as char,
        );
    }
    assert!(
        state.accept_token(TOOL_CALL_CLOSE),
        "single-token </tool_call> must ADVANCE the matcher (not exempt)"
    );
}

#[test]
fn hermes_auto_grammar_accepts_two_sequential_calls() {
    let vocab = test_vocab();
    let stop_ids = vec![EOS as i32];
    let mut engine = GrammarEngine::new(&vocab, &stop_ids).unwrap();

    let compiled = engine
        .compile_hermes_tool_grammar(&test_tool_defs(), true)
        .unwrap();
    let mut state = GrammarState::new(&compiled, engine.vocab_size())
        .unwrap()
        .with_stop_tokens(&[EOS]);

    drive_one_call(&mut state, "Paris");
    assert!(
        !state.is_terminated(),
        "auto mode: grammar must NOT terminate after the first call"
    );
    assert!(
        state.stop_legal(&[EOS]),
        "EOS must be legal between completed calls"
    );
    drive_one_call(&mut state, "Berlin");
    assert!(
        state.stop_legal(&[EOS]),
        "EOS must be legal after the second call"
    );
}

/// 2026-09-26: With tools armed and no call made, the auto grammar is not
/// terminated, yet [`grammar_blocks_stop`] lets EOS through, before and
/// after some prose.
#[test]
fn armed_no_call_auto_grammar_never_blocks_eos() {
    let vocab = test_vocab();
    let stop_ids = vec![EOS as i32];
    let mut engine = GrammarEngine::new(&vocab, &stop_ids).unwrap();
    let compiled = engine
        .compile_hermes_tool_grammar(&test_tool_defs(), true)
        .unwrap();
    let mut state = GrammarState::new(&compiled, engine.vocab_size())
        .unwrap()
        .with_stop_tokens(&[EOS]);

    assert!(
        !state.is_terminated(),
        "auto grammar never terminates (this is WHY !is_terminated() was the wrong gate)"
    );
    assert!(
        !grammar_blocks_stop(Some(&mut state), &[EOS]),
        "armed-but-unused tools must not suppress EOS (preamble state permits end-of-turn)"
    );
    for b in b"hi there" {
        assert!(state.accept_token(u32::from(*b)));
    }
    assert!(!grammar_blocks_stop(Some(&mut state), &[EOS]));
    assert!(!grammar_blocks_stop(None, &[EOS]));
}

/// 2026-09-26: EOS is allowed after each completed call and blocked inside
/// an open JSON string of the next one.
#[test]
fn eos_reachable_after_each_close_literal_but_blocked_mid_call() {
    let vocab = test_vocab();
    let stop_ids = vec![EOS as i32];
    let mut engine = GrammarEngine::new(&vocab, &stop_ids).unwrap();
    let compiled = engine
        .compile_hermes_tool_grammar(&test_tool_defs(), true)
        .unwrap();
    let mut state = GrammarState::new(&compiled, engine.vocab_size())
        .unwrap()
        .with_stop_tokens(&[EOS]);

    drive_one_call(&mut state, "Paris");
    assert!(
        !grammar_blocks_stop(Some(&mut state), &[EOS]),
        "a completed call must terminate cleanly (EOS legal after close literal)"
    );

    assert!(state.accept_token(TOOL_CALL_OPEN));
    for b in b"\n{\"name\": \"get_weather\", \"arguments\": {\"location\":\"To" {
        assert!(state.accept_token(u32::from(*b)));
    }
    assert!(
        grammar_blocks_stop(Some(&mut state), &[EOS]),
        "EOS must stay suppressed mid-structure (open JSON string)"
    );

    for b in b"kyo\"}}\n" {
        assert!(state.accept_token(u32::from(*b)));
    }
    assert!(state.accept_token(TOOL_CALL_CLOSE));
    assert!(
        !grammar_blocks_stop(Some(&mut state), &[EOS]),
        "EOS legal again after the SECOND close literal"
    );
}

/// 2026-09-26: In required mode (`at_least_one` and `stop_after_first`),
/// EOS is blocked until the call completes.
#[test]
fn required_mode_blocks_eos_until_call_completes() {
    let vocab = test_vocab();
    let stop_ids = vec![EOS as i32];
    let mut engine = GrammarEngine::new(&vocab, &stop_ids).unwrap();
    let compiled = engine
        .compile_hermes_tool_grammar(&test_tool_defs(), false)
        .unwrap();
    let mut state = GrammarState::new(&compiled, engine.vocab_size())
        .unwrap()
        .with_stop_tokens(&[EOS]);

    assert!(
        grammar_blocks_stop(Some(&mut state), &[EOS]),
        "required mode: EOS suppressed before the mandatory call"
    );
    drive_one_call(&mut state, "Paris");
    assert!(
        !grammar_blocks_stop(Some(&mut state), &[EOS]),
        "required mode: EOS legal once the mandatory call completed"
    );
}

/// 2026-09-26: With `</tool_call>` in the stop-token set, `accept_token`
/// returns `true` without moving the matcher, which stays inside the end
/// literal: the mask still constrains, and a second `<tool_call>` is
/// masked.
#[test]
fn hermes_grammar_wedges_if_tool_call_close_is_stop_exempt() {
    let vocab = test_vocab();
    let stop_ids = vec![EOS as i32];
    let mut engine = GrammarEngine::new(&vocab, &stop_ids).unwrap();
    let compiled = engine
        .compile_hermes_tool_grammar(&test_tool_defs(), true)
        .unwrap();
    let mut state = GrammarState::new(&compiled, engine.vocab_size())
        .unwrap()
        .with_stop_tokens(&[EOS, TOOL_CALL_CLOSE]);

    assert!(state.accept_token(TOOL_CALL_OPEN));
    let body = "\n{\"name\": \"get_weather\", \"arguments\": {\"location\":\"Paris\"}}\n";
    for b in body.bytes() {
        assert!(state.accept_token(u32::from(b)));
    }
    assert!(state.accept_token(TOOL_CALL_CLOSE));
    assert!(
        state.fill_bitmask(),
        "wedged matcher still constrains decoding"
    );
    assert!(
        !state.is_token_allowed(TOOL_CALL_OPEN),
        "second <tool_call> is grammar-illegal while wedged mid-end-literal"
    );
}
