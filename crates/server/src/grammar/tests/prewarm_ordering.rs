// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Ordering of the background mask prewarm: building a
//! `GrammarState` does not wait for it, and the first `fill_bitmask` does.
//!
//! The worker parks on a two-party `Barrier` the test releases, so both
//! orderings are checked without racing the worker.
//!
//! Owner: server (grammar) tests.
//! Invariants: none beyond the types.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Barrier};

use super::{test_tool_defs, test_vocab};
use crate::grammar::{GrammarEngine, GrammarState};
use crate::tool_parser::ToolDefinition;

fn engine_and_grammar() -> (GrammarEngine, metrale_grammar::CompiledGrammar) {
    let mut engine = GrammarEngine::new(&test_vocab(), &[130]).expect("engine");
    let tools: Vec<ToolDefinition> = test_tool_defs();
    let compiled = engine
        .compile_hermes_tool_grammar(&tools, false)
        .expect("tool grammar compiles");
    (engine, compiled)
}

#[test]
fn construction_does_not_block_on_the_prewarm_but_the_first_fill_does() {
    let (engine, compiled) = engine_and_grammar();
    let gate = Arc::new(Barrier::new(2));

    let mut state =
        GrammarState::build_gated(&compiled, engine.vocab_size(), Some(Arc::clone(&gate)))
            .expect("grammar state");
    assert!(
        !state.prewarm_finished(),
        "construction returned only after the prewarm ran — the compile is \
         still on the request's critical path"
    );

    // 2026-09-26: Another thread sets `released`, then releases the worker.
    let released = Arc::new(AtomicBool::new(false));
    let releaser = {
        let (gate, released) = (Arc::clone(&gate), Arc::clone(&released));
        std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(50));
            released.store(true, Ordering::Release);
            gate.wait();
        })
    };

    state.fill_bitmask();
    assert!(
        released.load(Ordering::Acquire),
        "the first constrained fill did not wait for the prewarm"
    );
    assert!(
        state.prewarm_finished(),
        "the first constrained fill returned with the prewarm still running"
    );
    releaser.join().unwrap();
}

#[test]
fn awaiting_the_prewarm_is_idempotent_across_every_mask_reading_path() {
    let (engine, compiled) = engine_and_grammar();
    let mut state = GrammarState::new(&compiled, engine.vocab_size()).expect("grammar state");
    state.fill_bitmask();
    assert!(state.prewarm_finished());
    // 2026-09-26: Later joins return at once. `await_masks` runs in
    // `fill_bitmask` (which `forced_token` and `stop_legal` call) and in
    // `completion_token_ids`.
    state.fill_bitmask();
    state.forced_token();
    state.stop_legal(&[130]);
    state.completion_token_ids(64);
    assert!(state.prewarm_finished());
}

#[test]
fn the_kill_switch_restores_the_synchronous_prewarm() {
    use crate::grammar::prewarm::overlap_from_env;
    assert!(overlap_from_env(None), "unset must overlap");
    assert!(overlap_from_env(Some("1")));
    assert!(overlap_from_env(Some("")), "empty must overlap");
    for off in ["0", "false", "OFF", " no "] {
        assert!(!overlap_from_env(Some(off)), "{off} must disable overlap");
    }

    // 2026-09-26: The inline path is built directly, without setting the
    // process-wide env var.
    let (engine, compiled) = engine_and_grammar();
    let state = GrammarState::build_with(&compiled, engine.vocab_size(), None, false, None)
        .expect("grammar state");
    assert!(
        state.prewarm_finished(),
        "with overlap disabled, construction must warm inline"
    );
}
