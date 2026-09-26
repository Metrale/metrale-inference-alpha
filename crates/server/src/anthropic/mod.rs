// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Anthropic Messages API surface: `POST /v1/messages` and
//! `POST /v1/messages/count_tokens`.
//!
//! `messages` lowers the wire request into the chat IR, runs it through
//! `api::chat_completions_inner`, and encodes the result as an Anthropic JSON
//! body or SSE stream. `count_tokens` renders the same prompt through
//! `prepare_chat_prompt` and returns its length.
//!
//! Owner: server (Anthropic adapter).
//! Invariants: none beyond the types.

mod handlers;
mod handlers_stream;
mod helpers;
mod to_ir;
mod translate;
mod translator;
mod types;

#[cfg(test)]
mod tests;

pub use handlers::{count_tokens, messages};
