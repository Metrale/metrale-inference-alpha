// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Streaming `POST /v1/responses`: turns the chat pipeline's
//! `StreamDelta`s into Responses SSE events (reasoning, message and
//! function-call output items), then hands off to `finalize_responses_stream`.
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
use super::responses_stream_finalize::{
    CloseOpenCtx, FinalizeCtx, ReasoningState, close_open_fc, close_open_items, close_open_message,
    close_open_reasoning, emit_responses_prologue, finalize_responses_stream,
};
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

pub(super) async fn responses_endpoint_stream(
    CurrentModel(state): CurrentModel,
    mut chat_req: ChatCompletionRequest,
    metadata: Option<std::collections::HashMap<String, String>>,
    store_flag: bool,
    conversation_id: Option<String>,
) -> Response {
    use axum::response::sse::Event;
    use tokio::sync::mpsc;
    chat_req.stream = true;
    chat_req.stream_options = None;

    let model = chat_req.model.clone();
    let input_messages = chat_req.messages.clone();
    let state_arc = state.clone();

    // 2026-09-26: The input messages after the conversation's stored item
    // count, appended to the conversation by `finalize_responses_stream`.
    let conv_new_user_items: Vec<serde_json::Value> = if let Some(cid) = conversation_id.as_ref() {
        let prior = state_arc
            .conversation_store
            .get(cid)
            .map(|s| s.items.len())
            .unwrap_or(0);
        input_messages
            .iter()
            .skip(prior)
            .map(|m| {
                serde_json::json!({
                    "type": "message",
                    "role": m.role,
                    "content": [{"type": "input_text", "text": m.content.text}],
                })
            })
            .collect()
    } else {
        Vec::new()
    };

    let deltas = match chat_completions_inner(state, None, chat_req.into(), None).await {
        super::chat::ChatOutcome::Streaming(d) => d,
        super::chat::ChatOutcome::Http(r) => return r,
        // 2026-09-26: Not reached: `chat_req.stream` is true, so a success is
        // `Streaming` (`chat/mod.rs` dispatch).
        super::chat::ChatOutcome::Blocking(_) => {
            return openai_error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "internal: expected streaming outcome".to_string(),
            );
        }
    };

    // 2026-09-26: Same capacity as the chat stream's channel (`chat_stream/mod.rs`).
    let (tx, rx) = mpsc::channel::<Result<Event, std::convert::Infallible>>(1024);
    let created_at = crate::ids::unix_timestamp();
    let resp_id = format!("resp_{}", crate::ids::uuid_v4());
    let metadata_for_done = metadata.clone();

    tokio::spawn(async move {
        let mut seq: u64 = 0;
        let mut content_text = String::new();
        let mut tool_args = String::new();
        let mut current_tool_name: Option<String> = None;
        let mut current_tool_call_id: Option<String> = None;
        let mut output_index: usize = 0;
        // 2026-09-26: Message item ids carry the output index, so a
        // text→function_call→text sequence gets a new id per message.
        let mut message_item_id = format!("msg_{}_{}", resp_id, output_index);
        let mut fc_item_id: Option<String> = None;
        let mut message_started = false;
        let mut fc_started = false;
        let mut fc_done = false;
        // 2026-09-26: Every closed output item, so `response.completed` and the
        // stored response list all of them.
        let mut completed_items: Vec<crate::openai::ResponsesOutputItem> = Vec::new();
        let mut final_usage: Option<serde_json::Value> = None;
        let mut finish_reason = "stop".to_string();
        let mut refusal_text: Option<String> = None;
        // 2026-09-26: The reasoning item opens on a Reasoning delta, streams as
        // `response.reasoning_summary_text.delta` events, and closes at the
        // next content or tool-call delta or at stream end. A later Reasoning
        // delta opens a new reasoning item.
        let mut reasoning = ReasoningState {
            open: false,
            item_id: String::new(),
            text: String::new(),
            output_index: 0,
        };

        seq = emit_responses_prologue(&tx, seq, &resp_id, created_at, &model, &metadata).await;

        let mut deltas = deltas;
        while let Some(delta) = deltas.next().await {
            use crate::ir::StreamDelta;
            match delta {
                StreamDelta::Reasoning { text, .. } if !text.is_empty() => {
                    if !reasoning.open {
                        // 2026-09-26: The chat stream re-enters thinking on a
                        // `<think>` in content (`chat_stream/handle_token.rs`),
                        // so a message or function call may be open here:
                        // close it and open the reasoning item at the next
                        // output index. The content and tool-call arms close
                        // one before opening the other, so at most one is open.
                        if fc_started && !fc_done {
                            if let Some(fcid) = fc_item_id.clone() {
                                seq = close_open_fc(
                                    &tx,
                                    &mut completed_items,
                                    seq,
                                    &fcid,
                                    current_tool_call_id.as_deref().unwrap_or_default(),
                                    current_tool_name.as_deref().unwrap_or_default(),
                                    &tool_args,
                                    output_index,
                                )
                                .await;
                            }
                            output_index += 1;
                            fc_done = true;
                        } else if message_started {
                            seq = close_open_message(
                                &tx,
                                &mut completed_items,
                                seq,
                                &message_item_id,
                                &content_text,
                                output_index,
                            )
                            .await;
                            output_index += 1;
                            message_started = false;
                            content_text.clear();
                        }
                        reasoning.open = true;
                        reasoning.output_index = output_index;
                        reasoning.item_id = format!("rs_{}_{}", resp_id, output_index);
                        // 2026-09-26: A new reasoning item starts with empty text.
                        reasoning.text.clear();
                        let item = crate::openai::ResponsesOutputItem::Reasoning {
                            id: reasoning.item_id.clone(),
                            summary: vec![],
                        };
                        let ev = crate::openai::ResponsesStreamEvent::OutputItemAdded {
                            sequence_number: seq,
                            output_index,
                            item,
                        };
                        emit(&tx, &ev).await;
                        seq += 1;
                        let ev = crate::openai::ResponsesStreamEvent::ReasoningSummaryPartAdded {
                            sequence_number: seq,
                            item_id: reasoning.item_id.clone(),
                            output_index,
                            summary_index: 0,
                            part: crate::openai::ResponsesSummaryPart::SummaryText {
                                text: String::new(),
                            },
                        };
                        emit(&tx, &ev).await;
                        seq += 1;
                    }
                    reasoning.text.push_str(&text);
                    let ev = crate::openai::ResponsesStreamEvent::ReasoningSummaryTextDelta {
                        sequence_number: seq,
                        item_id: reasoning.item_id.clone(),
                        output_index,
                        summary_index: 0,
                        delta: text,
                    };
                    emit(&tx, &ev).await;
                    seq += 1;
                }
                StreamDelta::Reasoning { .. } => {}
                // 2026-09-26: The chat stream sends at most one Refusal delta,
                // with the whole text, at the end (`chat_stream/handle_done.rs`).
                StreamDelta::Refusal { text } if !text.is_empty() => {
                    refusal_text = Some(text.clone());
                    let ev = crate::openai::ResponsesStreamEvent::RefusalDelta {
                        sequence_number: seq,
                        item_id: message_item_id.clone(),
                        output_index,
                        content_index: 0,
                        delta: text,
                    };
                    emit(&tx, &ev).await;
                    seq += 1;
                }
                StreamDelta::Refusal { .. } => {}
                StreamDelta::Content { text, .. } if !text.is_empty() => {
                    // 2026-09-26: Content closes an open reasoning item, and the
                    // message opens at the next output index with an id for it.
                    let had_reasoning = reasoning.open;
                    seq = close_open_reasoning(
                        &tx,
                        &mut completed_items,
                        seq,
                        &mut reasoning,
                        &mut output_index,
                    )
                    .await;
                    if had_reasoning {
                        message_item_id = format!("msg_{}_{}", resp_id, output_index);
                    }
                    // 2026-09-26: Close an open function call first, so the
                    // message does not share its output index.
                    if fc_started && !fc_done {
                        if let Some(fcid) = fc_item_id.clone() {
                            seq = close_open_fc(
                                &tx,
                                &mut completed_items,
                                seq,
                                &fcid,
                                current_tool_call_id.as_deref().unwrap_or_default(),
                                current_tool_name.as_deref().unwrap_or_default(),
                                &tool_args,
                                output_index,
                            )
                            .await;
                        }
                        output_index += 1;
                        message_item_id = format!("msg_{}_{}", resp_id, output_index);
                        fc_done = true;
                        // 2026-09-26: The new message's text starts empty.
                        content_text.clear();
                    }
                    if !message_started {
                        message_started = true;
                        let item = crate::openai::ResponsesOutputItem::Message {
                            id: message_item_id.clone(),
                            status: "in_progress",
                            role: "assistant",
                            content: vec![],
                        };
                        let ev = crate::openai::ResponsesStreamEvent::OutputItemAdded {
                            sequence_number: seq,
                            output_index,
                            item,
                        };
                        emit(&tx, &ev).await;
                        seq += 1;
                        let cp = crate::openai::ResponsesContentPart::OutputText {
                            text: String::new(),
                            annotations: None,
                        };
                        let ev = crate::openai::ResponsesStreamEvent::ContentPartAdded {
                            sequence_number: seq,
                            item_id: message_item_id.clone(),
                            output_index,
                            content_index: 0,
                            part: cp,
                        };
                        emit(&tx, &ev).await;
                        seq += 1;
                    }
                    content_text.push_str(&text);
                    let ev = crate::openai::ResponsesStreamEvent::OutputTextDelta {
                        sequence_number: seq,
                        item_id: message_item_id.clone(),
                        output_index,
                        content_index: 0,
                        delta: text,
                    };
                    emit(&tx, &ev).await;
                    seq += 1;
                }
                StreamDelta::Content { .. } => {}
                StreamDelta::ToolCallStart { id, name, .. } => {
                    // 2026-09-26: A tool call closes an open reasoning item; the
                    // function call opens at the next output index.
                    seq = close_open_reasoning(
                        &tx,
                        &mut completed_items,
                        seq,
                        &mut reasoning,
                        &mut output_index,
                    )
                    .await;
                    current_tool_name = Some(name.clone());
                    current_tool_call_id = Some(id);
                    if !fc_started {
                        if message_started {
                            seq = close_open_message(
                                &tx,
                                &mut completed_items,
                                seq,
                                &message_item_id,
                                &content_text,
                                output_index,
                            )
                            .await;
                            output_index += 1;
                            message_started = false;
                            content_text.clear();
                        }
                        let fcid = format!("fc_{}_{}", resp_id, output_index);
                        fc_item_id = Some(fcid.clone());
                        let item = crate::openai::ResponsesOutputItem::FunctionCall {
                            id: fcid,
                            call_id: current_tool_call_id.clone().unwrap_or_default(),
                            name,
                            arguments: String::new(),
                            status: "in_progress",
                        };
                        let ev = crate::openai::ResponsesStreamEvent::OutputItemAdded {
                            sequence_number: seq,
                            output_index,
                            item,
                        };
                        emit(&tx, &ev).await;
                        seq += 1;
                        fc_started = true;
                    }
                }
                StreamDelta::ToolCallArgs { fragment, .. } if !fragment.is_empty() => {
                    tool_args.push_str(&fragment);
                    if let Some(fcid) = fc_item_id.clone() {
                        let ev = crate::openai::ResponsesStreamEvent::FunctionCallArgumentsDelta {
                            sequence_number: seq,
                            item_id: fcid,
                            output_index,
                            delta: fragment,
                        };
                        emit(&tx, &ev).await;
                        seq += 1;
                    }
                }
                StreamDelta::ToolCallArgs { .. } => {}
                StreamDelta::Finish { reason, usage, .. } => {
                    finish_reason = reason.as_wire().to_string();
                    // 2026-09-26: Usage in the chat wire shape, which
                    // `build_responses_usage` reads.
                    final_usage = Some(serde_json::json!({
                        "prompt_tokens": usage.prompt_tokens,
                        "completion_tokens": usage.completion_tokens,
                        "total_tokens": usage.prompt_tokens + usage.completion_tokens,
                        "prompt_tokens_details": {
                            "cached_tokens": usage.cached_prompt_tokens,
                            "audio_tokens": 0,
                        },
                        "completion_tokens_details": {
                            "reasoning_tokens": usage.reasoning_tokens,
                            "audio_tokens": 0,
                            "accepted_prediction_tokens": usage.accepted_prediction_tokens,
                            "rejected_prediction_tokens": 0,
                        },
                    }));
                }
                // 2026-09-26: An error delta is only logged; the finalizer
                // still closes the response when the deltas end.
                StreamDelta::Error { message } => {
                    tracing::warn!("responses stream: upstream error delta: {message}");
                }
            }
        }

        // 2026-09-26: A turn that ends in reasoning still has it open; close
        // it before the other items. No-op when it is closed.
        seq = close_open_reasoning(
            &tx,
            &mut completed_items,
            seq,
            &mut reasoning,
            &mut output_index,
        )
        .await;

        seq = close_open_items(
            &tx,
            &mut completed_items,
            CloseOpenCtx {
                seq,
                message_started,
                message_item_id: &message_item_id,
                content_text: &content_text,
                fc_started,
                fc_done,
                fc_item_id: fc_item_id.clone(),
                current_tool_call_id: &current_tool_call_id,
                current_tool_name: &current_tool_name,
                tool_args: &tool_args,
                output_index,
            },
        )
        .await;

        finalize_responses_stream(
            &tx,
            state_arc.clone(),
            FinalizeCtx {
                seq,
                completed_items,
                final_usage,
                finish_reason,
                refusal_text,
                message_item_id,
                output_index,
                resp_id,
                created_at,
                model,
                metadata_for_done,
                store_flag,
                input_messages,
                conversation_id,
                conv_new_user_items,
            },
        )
        .await;
    });

    let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
    Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}
