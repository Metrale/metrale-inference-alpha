// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Handlers for OpenAI endpoints this server does not implement (batches,
//! files, audio, images, moderations). Routes: `main_modules/serve_router.rs`.
//!
//! Owner: server API.
//! Invariants:
//! - Every handler answers 501 with an OpenAI-shaped error (`completions::not_supported`).

#![allow(unused_imports, dead_code)]

use axum::extract::State;
use axum::extract::rejection::JsonRejection;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive};
use axum::response::{IntoResponse, Json, Response, Sse};
use futures::StreamExt;
use std::sync::Arc;
use tokio_stream::wrappers::ReceiverStream;

use crate::AppState;
use crate::openai::{
    ChatCompletionChunk, ChatCompletionRequest, ChatCompletionResponse, CompletionChunk,
    CompletionRequest, CompletionResponse, ModelInfo, ModelListResponse, Usage,
};
use crate::tool_parser;

use super::chat::chat_completions_inner;
use super::compact::{compact_messages, openai_error_response, openai_error_response_with_param};
use super::completions::not_supported;
use super::inference_impl::{extract_thinking, strip_stop_sequences, tokenize_stop_sequences};
use super::inference_types::{
    GrammarSpec, InferenceRequest, InferenceResponse, StreamEvent, TokenLogprobs,
};
use super::sanitizer::{
    F7_STALL_REFUSE_THRESHOLD, F7_STALL_WARN_THRESHOLD, F7StallBuckets, ToolKind, classify_tool,
    extract_bash_final_action, primary_arg_for_tool, sanitize_content_chunk,
};
use super::strip::strip_thinking_tags;

use super::inference_types::*;
use super::sanitizer::*;

pub async fn batches_stub() -> Response {
    not_supported(
        "Batch API is not supported. Submit requests directly to /v1/chat/completions; Metrale Engine serves them synchronously.",
    )
}

/// 2026-09-26: `GET` and `DELETE /v1/batches/{id}`, and `POST /v1/batches/{id}/cancel`.
pub async fn batch_get_stub() -> Response {
    not_supported("Batch API is not supported. No batches are tracked on this server.")
}

/// 2026-09-26: `GET /v1/batches`.
pub async fn batch_list_stub() -> Response {
    not_supported("Batch API is not supported. No batches are tracked on this server.")
}

/// 2026-09-26: `/v1/files`, `/v1/files/{id}` and `/v1/files/{id}/content`.
pub async fn files_stub() -> Response {
    not_supported(
        "File storage API is not supported. Metrale Engine is an inference-only server; upload-then-reference workflows (batches, vision by file_id) are not available.",
    )
}

/// 2026-09-26: `POST /v1/audio/transcriptions`, `/translations` and `/speech`.
pub async fn audio_stub() -> Response {
    not_supported(
        "Audio API is not supported. Metrale Engine serves text chat/completion models only.",
    )
}

/// 2026-09-26: `POST /v1/images/generations`, `/edits` and `/variations`.
pub async fn images_stub() -> Response {
    not_supported(
        "Image API is not supported. Metrale Engine serves text chat/completion models only.",
    )
}

/// 2026-09-26: `POST /v1/moderations`.
pub async fn moderations_stub() -> Response {
    not_supported(
        "Moderations API is not supported. Metrale Engine does not classify inputs for safety; run your own moderation pass upstream if needed.",
    )
}
