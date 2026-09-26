// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Declares the `emit_step` submodules (`grammar_close`, `token`,
//! `tool_param`) and re-exports their entry points.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use super::*;

mod grammar_close;
mod token;
mod tool_param;
pub(crate) use grammar_close::emit_grammar_close;
pub use grammar_close::{StartPrefillResult, compile_grammar_state};
pub use token::emit_token;
pub use tool_param::update_tool_param_state;

#[cfg(test)]
mod cc6_envelope_streak_tests;
