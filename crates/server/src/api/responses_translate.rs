// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Encoding of a blocking chat result as a Responses API
//! response, and the SSE helpers used by the streaming path.
//!
//! Owner: server responses API.
//! Invariants: none beyond the types.

#![allow(unused_imports, dead_code)]

use axum::extract::State;
use axum::extract::rejection::JsonRejection;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive};
use axum::response::{IntoResponse, Json, Response, Sse};
use futures::StreamExt;
use std::sync::Arc;
use tokio_stream::wrappers::ReceiverStream;

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

pub(super) fn find_frame_end(buf: &[u8]) -> Option<usize> {
    buf.windows(2).position(|w| w == b"\n\n")
}

pub(super) async fn emit(
    tx: &tokio::sync::mpsc::Sender<Result<axum::response::sse::Event, std::convert::Infallible>>,
    ev: &crate::openai::ResponsesStreamEvent,
) {
    use axum::response::sse::Event;
    if let Ok(json) = serde_json::to_string(ev)
        && let Err(e) = tx
            .send(Ok(Event::default()
                .event(crate::openai::responses_event_name(ev))
                .data(json)))
            .await
    {
        tracing::warn!("responses_translate::emit: SSE send failed (receiver dropped): {e}");
    }
}

pub(super) fn build_responses_usage(u: &serde_json::Value) -> crate::openai::ResponsesUsage {
    crate::openai::ResponsesUsage {
        input_tokens: u.get("prompt_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as usize,
        input_tokens_details: u
            .get("prompt_tokens_details")
            .and_then(|v| serde_json::from_value(v.clone()).ok()),
        output_tokens: u
            .get("completion_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize,
        output_tokens_details: u
            .get("completion_tokens_details")
            .and_then(|v| serde_json::from_value(v.clone()).ok()),
        total_tokens: u.get("total_tokens").and_then(|v| v.as_u64()).unwrap_or(0) as usize,
    }
}

/// 2026-09-26: Encode a blocking chat outcome as a Responses API response; an
/// error response passes through unchanged.
///
/// When `store` is given and `store_flag` is true, the input messages plus the
/// assistant turn are stored under `resp_<id>` for `previous_response_id`.
/// With `conversation`, the turn's new items and the reply are appended to it.
pub(super) async fn translate_chat_response_to_responses(
    outcome: super::chat::ChatOutcome,
    req_metadata: Option<std::collections::HashMap<String, String>>,
    store: Option<Arc<crate::response_store::ResponseStore>>,
    input_messages: Vec<crate::openai::IncomingMessage>,
    store_flag: bool,
    conversation: Option<(Arc<crate::conversation_store::ConversationStore>, String)>,
) -> Response {
    let chat = match outcome {
        super::chat::ChatOutcome::Http(r) => return r,
        // 2026-09-26: Not reached: the only caller, `responses_endpoint`, uses
        // it for non-streaming requests.
        super::chat::ChatOutcome::Streaming(_) => {
            return openai_error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "internal: expected blocking outcome".to_string(),
            );
        }
        super::chat::ChatOutcome::Blocking(ir) => *ir,
    };
    let full_id = format!("chatcmpl-{}", chat.id);
    let mut output: Vec<crate::openai::ResponsesOutputItem> = Vec::new();
    if let Some(choice) = chat.choices.first() {
        // 2026-09-26: The reasoning item comes first, then function calls,
        // then the message.
        if let Some(reasoning) = choice.reasoning.as_deref().filter(|r| !r.is_empty()) {
            output.push(crate::openai::ResponsesOutputItem::Reasoning {
                id: format!("rs_{}", full_id),
                summary: vec![crate::openai::ResponsesSummaryPart::SummaryText {
                    text: reasoning.to_string(),
                }],
            });
        }
        for (i, tc) in choice.tool_calls.iter().enumerate() {
            output.push(crate::openai::ResponsesOutputItem::FunctionCall {
                id: format!("fc_{}_{}", full_id, i),
                call_id: tc.id.clone(),
                name: tc.name.clone(),
                arguments: tc.arguments.to_string(),
                status: "completed",
            });
        }
        if let Some(text) = choice.content.as_deref() {
            // 2026-09-26: URL-citation annotations, as the chat encoder adds
            // them (`openai/encode.rs`).
            let annotations = crate::openai::merged_annotations(text);
            output.push(crate::openai::ResponsesOutputItem::Message {
                id: format!("msg_{}", full_id),
                status: "completed",
                role: "assistant",
                content: vec![crate::openai::ResponsesContentPart::OutputText {
                    annotations,
                    text: text.to_string(),
                }],
            });
        }
    }
    let usage = crate::openai::ResponsesUsage {
        input_tokens: chat.usage.prompt_tokens,
        input_tokens_details: Some(crate::openai::PromptTokensDetails {
            cached_tokens: chat.usage.cached_prompt_tokens,
            audio_tokens: 0,
        }),
        output_tokens: chat.usage.completion_tokens,
        output_tokens_details: Some(crate::openai::CompletionTokensDetails {
            reasoning_tokens: chat.usage.reasoning_tokens,
            audio_tokens: 0,
            accepted_prediction_tokens: chat.usage.accepted_prediction_tokens,
            rejected_prediction_tokens: 0,
        }),
        total_tokens: chat.usage.prompt_tokens + chat.usage.completion_tokens,
    };
    let resp_id = format!("resp_{}", chat.id);
    let resp = crate::openai::ResponsesResponse {
        id: resp_id.clone(),
        object: "response",
        created_at: chat.created,
        model: chat.model.clone(),
        status: "completed",
        error: None,
        output,
        reasoning: None,
        usage,
        metadata: req_metadata,
    };

    // 2026-09-26: Serialised once, so the stored body is the returned body.
    let body = match serde_json::to_value(&resp) {
        Ok(v) => v,
        Err(e) => {
            return openai_error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("response serialization failed: {e}"),
            );
        }
    };
    if store_flag && let Some(store) = store {
        let mut transcript = input_messages.clone();
        if let Some(assistant_msg) = assistant_incoming_from_ir(&chat) {
            transcript.push(assistant_msg);
        }
        store.insert(crate::response_store::StoredEntry {
            id: resp_id,
            kind: crate::response_store::StoredKind::Response,
            model: chat.model.clone(),
            created_at: chat.created,
            messages: transcript,
            body: body.clone(),
            last_access: std::time::Instant::now(),
        });
    }

    // 2026-09-26: Conversation append: the input messages after the
    // conversation's stored item count, then the assistant reply.
    if let Some((conv_store, conv_id)) = conversation {
        let prior = conv_store.get(&conv_id).map(|s| s.items.len()).unwrap_or(0);
        let mut batch: Vec<serde_json::Value> = input_messages
            .iter()
            .skip(prior)
            .map(|m| {
                serde_json::json!({
                    "type": "message",
                    "role": m.role,
                    "content": [{"type": "input_text", "text": m.content.text}],
                })
            })
            .collect();
        let assistant_text = chat
            .choices
            .first()
            .and_then(|c| c.content.as_deref())
            .unwrap_or("");
        let assistant_reasoning = chat
            .choices
            .first()
            .and_then(|c| c.reasoning.as_deref())
            .unwrap_or("");
        if !assistant_text.is_empty() || !assistant_reasoning.is_empty() {
            let mut item = serde_json::json!({
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": assistant_text}],
            });
            if !assistant_reasoning.is_empty() {
                item["reasoning_content"] = serde_json::json!(assistant_reasoning);
            }
            batch.push(item);
        }
        if !batch.is_empty()
            && let Err(e) = conv_store.add_items(&conv_id, batch)
        {
            tracing::warn!(
                "responses_translate: conversation_store.add_items failed for {conv_id}: {e:?}"
            );
        }
    }

    Json(body).into_response()
}
