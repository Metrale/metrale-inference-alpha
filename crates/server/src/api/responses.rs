// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: `POST /v1/responses`: lowers a Responses request to a chat
//! request, runs it through the chat pipeline, and encodes the result
//! (streaming in `responses_stream`).
//!
//! Owner: server responses API.
//! Invariants: none beyond the types.

#![allow(unused_imports, dead_code)]

use crate::main_modules::model_host::CurrentModel;
use axum::extract::State;
use axum::extract::rejection::JsonRejection;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive};
use axum::response::{IntoResponse, Json, Response, Sse};
use futures::StreamExt;
use std::sync::Arc;
use tokio_stream::wrappers::ReceiverStream;

use super::chat_stream::run_chat_stream;
use super::responses_stream::responses_endpoint_stream;
use super::responses_translate::{
    build_responses_usage, emit, find_frame_end, translate_chat_response_to_responses,
};
use super::stored::assistant_incoming_from_ir;
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

pub async fn responses_endpoint(
    CurrentModel(state): CurrentModel,
    req: Result<Json<crate::openai::ResponsesRequest>, JsonRejection>,
) -> Response {
    let Json(r) = match req {
        Ok(r) => r,
        Err(e) => {
            return openai_error_response(
                StatusCode::BAD_REQUEST,
                format!("Invalid request JSON: {e}"),
            );
        }
    };
    let metadata = r.metadata.clone();
    let store_flag = r.store.unwrap_or(true);
    let streaming = r.stream;

    // 2026-09-26: `conversation` is a string id or `{"id": ...}`. Its stored
    // items go before the turn's input, and the new items are appended to it
    // after completion (`translate_chat_response_to_responses`, and
    // `finalize_responses_stream` when streaming).
    let conversation_id: Option<String> = match &r.conversation {
        None => None,
        Some(serde_json::Value::String(s)) => Some(s.clone()),
        Some(serde_json::Value::Object(o)) => {
            o.get("id").and_then(|v| v.as_str()).map(|s| s.to_string())
        }
        Some(_) => {
            return openai_error_response_with_param(
                StatusCode::BAD_REQUEST,
                "`conversation` must be a string id or an object with an `id` field.".into(),
                Some("conversation"),
                None,
            );
        }
    };
    let conversation_prefix: Vec<crate::openai::IncomingMessage> = match &conversation_id {
        None => Vec::new(),
        Some(cid) => match state.conversation_store.get(cid) {
            Some(snap) => snap
                .items
                .iter()
                .filter_map(crate::openai::IncomingMessage::from_conversation_item)
                .collect(),
            None => {
                return openai_error_response_with_param(
                    StatusCode::NOT_FOUND,
                    format!("Conversation '{cid}' not found."),
                    Some("conversation"),
                    Some("conversation_not_found"),
                );
            }
        },
    };

    // 2026-09-26: `previous_response_id` is resolved against the response store
    // through this closure, so `lower_responses_to_chat` does not depend on the
    // store; an unknown id is a 400.
    let store = state.response_store.clone();
    let resolve = |prior_id: &str| -> Option<Vec<crate::openai::IncomingMessage>> {
        store
            .get(prior_id, crate::response_store::StoredKind::Response)
            .map(|e| e.messages)
    };

    let mut chat_req = match crate::openai::lower_responses_to_chat(r, resolve) {
        Ok(c) => c,
        Err(crate::openai::LowerResponsesError::BadRequest(m)) => {
            return openai_error_response(StatusCode::BAD_REQUEST, m);
        }
        Err(crate::openai::LowerResponsesError::PriorNotFound(m)) => {
            return openai_error_response_with_param(
                StatusCode::BAD_REQUEST,
                m,
                Some("previous_response_id"),
                Some("response_not_found"),
            );
        }
    };

    if !conversation_prefix.is_empty() {
        let mut combined = conversation_prefix;
        combined.append(&mut chat_req.messages);
        chat_req.messages = combined;
    }

    if streaming {
        return responses_endpoint_stream(
            CurrentModel(state),
            chat_req,
            metadata,
            store_flag,
            conversation_id,
        )
        .await;
    }

    // 2026-09-26: Copied before `chat_req` moves: the stored turn's messages.
    let input_messages = chat_req.messages.clone();

    // 2026-09-26: The lowered request goes through `chat_completions_inner`
    // with no request context and no `--dump` sequence.
    let resp = chat_completions_inner(state.clone(), None, chat_req.into(), None).await;
    let conv_pair = conversation_id.map(|cid| (state.conversation_store.clone(), cid));
    translate_chat_response_to_responses(
        resp,
        metadata,
        Some(state.response_store.clone()),
        input_messages,
        store_flag,
        conv_pair,
    )
    .await
}
