// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Earley parser for grammar matching. `EarleyParser` holds the Earley
//! items, advances them one byte at a time (`advance`), expands rule references
//! (`predict`, `predict_fsm`), completes finished rules (`complete`), reports the
//! acceptable next bytes (`accept`), and pushes or rolls back steps for the matcher.
//!
//! Owner: grammar.
//! Invariants: `completable`, `is_completed` and `scanable_history` hold one entry per
//! step. `new`, an accepted `advance`, `push_state_and_expand` and
//! `push_one_state_to_check` add one to each; `pop_last_states(count)` removes `count`
//! from each and panics rather than remove the first step; `reset` clears all three.

mod accept;
mod complete;
mod fsm_view;
mod parser;
mod parser_api;
mod predict;
mod predict_fsm;
mod prune;
mod queue;
mod scan;
mod scan_charclass;
mod state;

pub use parser::{CompletableEntry, EarleyParser};
pub use queue::ProcessQueue;
pub use state::{NO_PREV_INPUT_POS, ParserState, UNEXPANDED_RULE_START_SEQUENCE_ID, cache_key};

#[cfg(test)]
mod tests;
