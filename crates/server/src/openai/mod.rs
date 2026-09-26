// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The OpenAI wire types (chat completions, `/v1/completions`,
//! Responses) and the encoders that write the IR in those shapes.
//!
//! Owner: server (OpenAI adapter).
//! Invariants: none beyond the types.

mod annotations;
mod chat_message;
mod chat_request;
mod chat_response;
mod completions;
mod encode;
mod encode_stream;
mod responses;
mod responses_lowering;
mod stream_chunk;
mod to_ir;

#[cfg(test)]
mod tests;

pub use annotations::*;
pub use chat_message::*;
pub use chat_request::*;
pub use chat_response::*;
pub use completions::*;
pub(crate) use encode::encode_chat_response;
pub(crate) use encode_stream::encode_sse_response;
pub use responses::*;
pub use responses_lowering::*;
pub use stream_chunk::*;

// 2026-09-26: Imported so the child modules can call `uuid_v4()` and
// `unix_timestamp()` through `use super::*`.
use crate::ids::{unix_timestamp, uuid_v4};

/// 2026-09-26: A `cmpl-` id, used for the streamed `/v1/completions` chunks.
pub fn new_completion_id() -> String {
    format!("cmpl-{}", uuid_v4())
}

/// 2026-09-26: A `chatcmpl-` id, shared by every chunk of one streamed chat
/// response.
pub fn new_chunk_id() -> String {
    format!("chatcmpl-{}", uuid_v4())
}
