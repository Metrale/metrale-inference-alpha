// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for the Anthropic adapter, one module per area.
//!
//! - `convert`: wire deserialization and the stop-reason, tool-choice and
//!   tool-definition conversions.
//! - `to_ir_blocks`: how one wire message splits into IR messages.
//! - `ir_carry`: what each block carries into the IR, and the IR → response
//!   direction.
//! - `claude_code_fixture`: the captured Claude Code system prompt and tool
//!   list in `scripts/fixtures/`.
//! - `translator_stream`: SSE event framing for `/v1/messages`.
//!
//! Owner: server (Anthropic adapter) tests.
//! Invariants: none beyond the types.

mod claude_code_fixture;
mod convert;
mod ir_carry;
mod to_ir_blocks;
mod translator_stream;
