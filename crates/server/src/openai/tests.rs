// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for the OpenAI wire types, one child module per API surface.
//!
//! Owner: server (OpenAI API layer) tests.
//! Invariants: none beyond the types.

mod annotations;
mod chat_wire;
mod completions;
mod responses;
mod thinking;
mod usage_timing;
