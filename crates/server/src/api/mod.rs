// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: HTTP API handlers (axum), one submodule per surface, and the
//! `crate::api::*` re-exports that the router and other modules use.
//!
//! Owner: server API.
//! Invariants: none beyond the types.

pub mod chat;
pub mod chat_blocking;
mod chat_blocking_choice;
pub mod chat_phases;
pub mod chat_stream;
pub mod chat_stream_dispatch;
pub mod compact;
pub mod completions;
pub mod completions_exec;
pub mod completions_logprobs;
pub mod conversations;
pub mod inference_impl;
pub mod inference_types;
pub mod lora_control;
pub mod misc_handlers;
pub mod models;
pub mod responses;
pub mod responses_stream;
pub mod responses_stream_finalize;
pub mod responses_translate;
pub mod sanitizer;
pub mod scrub;
pub mod stored;
pub mod stream_guards;
pub(crate) mod stream_terminal;
pub mod strip;
pub mod stubs;
pub mod telemetry_events;

#[cfg(test)]
mod tests;

pub use chat::chat_completions;
pub(crate) use chat::chat_completions_inner;
pub(crate) use chat::{ChatOutcome, ResponseEcho};
#[allow(unused_imports)]
pub use compact::compact_messages;
pub use completions::completions;
#[allow(unused_imports)]
pub use conversations::{
    AddItemsRequest, CreateConversationRequest, UpdateConversationRequest, add_conversation_items,
    create_conversation, delete_conversation, delete_conversation_item, get_conversation,
    get_conversation_item, list_conversation_items, update_conversation,
};
pub use inference_types::{
    GrammarSpec, InferenceRequest, InferenceResponse, StreamEvent, TokenLogprobs,
};
pub use lora_control::{load_lora_into_slot, set_active_lora};
#[allow(unused_imports)]
pub use misc_handlers::{
    DetokenizeRequest, cancel_response, detokenize, hardware, health, health_live, metrics_handler,
    serve_config, tokenize,
};
pub use models::{embeddings_stub, get_model, list_models};
pub use responses::responses_endpoint;
pub use stored::{
    delete_stored_response, get_stored_completion, get_stored_response, list_response_input_items,
};
pub use stubs::{
    audio_stub, batch_get_stub, batch_list_stub, batches_stub, files_stub, images_stub,
    moderations_stub,
};
pub use telemetry_events::events as telemetry_events;
