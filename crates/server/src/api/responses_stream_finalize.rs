// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(unused_imports, dead_code)]

//! 2026-09-26: Helpers for the streaming `/v1/responses` endpoint: the opening
//! events, the closing of open output items, and [`finalize_responses_stream`],
//! which runs after the chat stream ends: it emits `response.completed`, stores
//! the response for `previous_response_id` and appends to a linked
//! conversation.
//!
//! Owner: server responses API.
//! Invariants: none beyond the types.

use std::sync::Arc;

use axum::response::sse::Event;
use tokio::sync::mpsc;

use super::responses_translate::{build_responses_usage, emit};
use crate::AppState;

/// 2026-09-26: Emit the opening `response.created` and `response.in_progress`
/// SSE events. Returns the next free sequence number.
pub(super) async fn emit_responses_prologue(
    tx: &mpsc::Sender<Result<Event, std::convert::Infallible>>,
    seq_start: u64,
    resp_id: &str,
    created_at: u64,
    model: &str,
    metadata: &Option<std::collections::HashMap<String, String>>,
) -> u64 {
    let mut seq = seq_start;
    let created = crate::openai::ResponsesStreamEvent::Created {
        sequence_number: seq,
        response: crate::openai::ResponsesStreamEnvelope {
            id: resp_id.to_string(),
            object: "response",
            created_at,
            model: model.to_string(),
            status: "in_progress",
            metadata: metadata.clone(),
        },
    };
    if let Ok(j) = serde_json::to_string(&created)
        && let Err(e) = tx
            .send(Ok(Event::default()
                .event(crate::openai::responses_event_name(&created))
                .data(j)))
            .await
    {
        tracing::warn!("responses_stream: response.created send failed (receiver dropped): {e}");
    }
    seq += 1;

    let in_progress = crate::openai::ResponsesStreamEvent::InProgress {
        sequence_number: seq,
        response: crate::openai::ResponsesStreamEnvelope {
            id: resp_id.to_string(),
            object: "response",
            created_at,
            model: model.to_string(),
            status: "in_progress",
            metadata: metadata.clone(),
        },
    };
    if let Ok(j) = serde_json::to_string(&in_progress)
        && let Err(e) = tx
            .send(Ok(Event::default()
                .event(crate::openai::responses_event_name(&in_progress))
                .data(j)))
            .await
    {
        tracing::warn!(
            "responses_stream: response.in_progress send failed (receiver dropped): {e}"
        );
    }
    seq += 1;
    seq
}

/// 2026-09-26: State of the open `reasoning` output item in the stream loop.
pub(super) struct ReasoningState {
    pub open: bool,
    pub item_id: String,
    pub text: String,
    /// 2026-09-26: The open item's output index.
    pub output_index: usize,
}

/// 2026-09-26: Close the open `reasoning` output item, if any: emit the summary
/// done events and `output_item.done`, record the item in `completed_items`,
/// and advance `output_index` past it. Returns the next free sequence number.
pub(super) async fn close_open_reasoning(
    tx: &mpsc::Sender<Result<Event, std::convert::Infallible>>,
    completed_items: &mut Vec<crate::openai::ResponsesOutputItem>,
    seq: u64,
    reasoning: &mut ReasoningState,
    output_index: &mut usize,
) -> u64 {
    if !reasoning.open {
        return seq;
    }
    reasoning.open = false;
    let mut seq = seq;
    let ev = crate::openai::ResponsesStreamEvent::ReasoningSummaryTextDone {
        sequence_number: seq,
        item_id: reasoning.item_id.clone(),
        output_index: reasoning.output_index,
        summary_index: 0,
        text: reasoning.text.clone(),
    };
    emit(tx, &ev).await;
    seq += 1;
    let ev = crate::openai::ResponsesStreamEvent::ReasoningSummaryPartDone {
        sequence_number: seq,
        item_id: reasoning.item_id.clone(),
        output_index: reasoning.output_index,
        summary_index: 0,
        part: crate::openai::ResponsesSummaryPart::SummaryText {
            text: reasoning.text.clone(),
        },
    };
    emit(tx, &ev).await;
    seq += 1;
    let done = crate::openai::ResponsesOutputItem::Reasoning {
        id: reasoning.item_id.clone(),
        summary: vec![crate::openai::ResponsesSummaryPart::SummaryText {
            text: reasoning.text.clone(),
        }],
    };
    completed_items.push(done.clone());
    let ev = crate::openai::ResponsesStreamEvent::OutputItemDone {
        sequence_number: seq,
        output_index: reasoning.output_index,
        item: done,
    };
    emit(tx, &ev).await;
    seq += 1;
    *output_index += 1;
    seq
}

/// 2026-09-26: The stream loop's open message and function-call state, for
/// closing them after the chat stream ends.
pub(super) struct CloseOpenCtx<'a> {
    pub seq: u64,
    pub message_started: bool,
    pub message_item_id: &'a str,
    pub content_text: &'a str,
    pub fc_started: bool,
    pub fc_done: bool,
    pub fc_item_id: Option<String>,
    pub current_tool_call_id: &'a Option<String>,
    pub current_tool_name: &'a Option<String>,
    pub tool_args: &'a str,
    pub output_index: usize,
}

/// 2026-09-26: Close the open message item: `output_text.done`, then
/// `output_item.done`, recorded in `completed_items`. Returns the next free
/// sequence number. Called when a tool call or a reasoning delta follows
/// content, and at stream end.
pub(super) async fn close_open_message(
    tx: &mpsc::Sender<Result<Event, std::convert::Infallible>>,
    completed_items: &mut Vec<crate::openai::ResponsesOutputItem>,
    seq: u64,
    message_item_id: &str,
    content_text: &str,
    output_index: usize,
) -> u64 {
    let mut seq = seq;
    let ev = crate::openai::ResponsesStreamEvent::OutputTextDone {
        sequence_number: seq,
        item_id: message_item_id.to_string(),
        output_index,
        content_index: 0,
        text: content_text.to_string(),
    };
    emit(tx, &ev).await;
    seq += 1;
    let done = crate::openai::ResponsesOutputItem::Message {
        id: message_item_id.to_string(),
        status: "completed",
        role: "assistant",
        content: vec![crate::openai::ResponsesContentPart::OutputText {
            text: content_text.to_string(),
            annotations: crate::openai::merged_annotations(content_text),
        }],
    };
    completed_items.push(done.clone());
    let ev = crate::openai::ResponsesStreamEvent::OutputItemDone {
        sequence_number: seq,
        output_index,
        item: done,
    };
    emit(tx, &ev).await;
    seq + 1
}

/// 2026-09-26: Close the open function-call item:
/// `function_call_arguments.done`, then `output_item.done`, recorded in
/// `completed_items`. Returns the next free sequence number.
pub(super) async fn close_open_fc(
    tx: &mpsc::Sender<Result<Event, std::convert::Infallible>>,
    completed_items: &mut Vec<crate::openai::ResponsesOutputItem>,
    seq: u64,
    fc_item_id: &str,
    tool_call_id: &str,
    tool_name: &str,
    tool_args: &str,
    output_index: usize,
) -> u64 {
    let mut seq = seq;
    let ev = crate::openai::ResponsesStreamEvent::FunctionCallArgumentsDone {
        sequence_number: seq,
        item_id: fc_item_id.to_string(),
        output_index,
        arguments: tool_args.to_string(),
    };
    emit(tx, &ev).await;
    seq += 1;
    let done = crate::openai::ResponsesOutputItem::FunctionCall {
        id: fc_item_id.to_string(),
        call_id: tool_call_id.to_string(),
        name: tool_name.to_string(),
        arguments: tool_args.to_string(),
        status: "completed",
    };
    completed_items.push(done.clone());
    let ev = crate::openai::ResponsesStreamEvent::OutputItemDone {
        sequence_number: seq,
        output_index,
        item: done,
    };
    emit(tx, &ev).await;
    seq + 1
}

/// 2026-09-26: Close the message or function-call item still open when the
/// chat stream ended, recording it in `completed_items`. Returns the next free
/// sequence number.
pub(super) async fn close_open_items(
    tx: &mpsc::Sender<Result<Event, std::convert::Infallible>>,
    completed_items: &mut Vec<crate::openai::ResponsesOutputItem>,
    ctx: CloseOpenCtx<'_>,
) -> u64 {
    let mut seq = ctx.seq;
    if ctx.message_started {
        seq = close_open_message(
            tx,
            completed_items,
            seq,
            ctx.message_item_id,
            ctx.content_text,
            ctx.output_index,
        )
        .await;
    }
    if ctx.fc_started
        && !ctx.fc_done
        && let Some(fcid) = ctx.fc_item_id.clone()
    {
        seq = close_open_fc(
            tx,
            completed_items,
            seq,
            &fcid,
            ctx.current_tool_call_id.as_deref().unwrap_or_default(),
            ctx.current_tool_name.as_deref().unwrap_or_default(),
            ctx.tool_args,
            ctx.output_index,
        )
        .await;
    }
    seq
}

/// 2026-09-26: What the stream loop hands to [`finalize_responses_stream`].
pub(super) struct FinalizeCtx {
    pub seq: u64,
    pub completed_items: Vec<crate::openai::ResponsesOutputItem>,
    pub final_usage: Option<serde_json::Value>,
    pub finish_reason: String,
    pub refusal_text: Option<String>,
    pub message_item_id: String,
    pub output_index: usize,
    pub resp_id: String,
    pub created_at: u64,
    pub model: String,
    pub metadata_for_done: Option<std::collections::HashMap<String, String>>,
    pub store_flag: bool,
    pub input_messages: Vec<crate::openai::IncomingMessage>,
    pub conversation_id: Option<String>,
    pub conv_new_user_items: Vec<serde_json::Value>,
}

/// 2026-09-26: Emit `response.completed` (status `failed` when the finish
/// reason is `error`), preceded by `response.refusal.done` when there was a
/// refusal; store the response when `store_flag` is set; append to the linked
/// conversation when `conversation_id` is set.
pub(super) async fn finalize_responses_stream(
    tx: &mpsc::Sender<Result<Event, std::convert::Infallible>>,
    state_arc: Arc<AppState>,
    ctx: FinalizeCtx,
) {
    let FinalizeCtx {
        mut seq,
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
    } = ctx;

    let store_ref = state_arc.response_store.clone();

    // 2026-09-26: The output is every item closed during the stream.
    let final_output = completed_items;
    let usage =
        final_usage
            .as_ref()
            .map(build_responses_usage)
            .unwrap_or(crate::openai::ResponsesUsage {
                input_tokens: 0,
                input_tokens_details: None,
                output_tokens: 0,
                output_tokens_details: None,
                total_tokens: 0,
            });
    let final_status: &'static str = if finish_reason == "error" {
        "failed"
    } else {
        "completed"
    };
    let final_resp = crate::openai::ResponsesResponse {
        id: resp_id.clone(),
        object: "response",
        created_at,
        model: model.clone(),
        status: final_status,
        error: None,
        output: final_output,
        reasoning: None,
        usage,
        metadata: metadata_for_done,
    };
    // 2026-09-26: The text of every message item and the reasoning of every
    // reasoning item, for the stored transcript and the conversation append.
    let mut transcript_text = String::new();
    let mut transcript_reasoning = String::new();
    for item in &final_resp.output {
        match item {
            crate::openai::ResponsesOutputItem::Message { content, .. } => {
                for part in content {
                    let crate::openai::ResponsesContentPart::OutputText { text, .. } = part;
                    transcript_text.push_str(text);
                }
            }
            crate::openai::ResponsesOutputItem::Reasoning { summary, .. } => {
                for part in summary {
                    let crate::openai::ResponsesSummaryPart::SummaryText { text } = part;
                    transcript_reasoning.push_str(text);
                }
            }
            crate::openai::ResponsesOutputItem::FunctionCall { .. } => {}
        }
    }
    // 2026-09-26: `store_flag` is the request's `store`, true when absent
    // (`responses_endpoint`).
    if store_flag && let Ok(body) = serde_json::to_value(&final_resp) {
        let mut transcript = input_messages;
        // 2026-09-26: The function-call items become the stored assistant
        // turn's `tool_calls`, so a `previous_response_id` resume includes them.
        let stored_tool_calls: Vec<crate::tool_parser::IncomingToolCall> = final_resp
            .output
            .iter()
            .filter_map(|item| match item {
                crate::openai::ResponsesOutputItem::FunctionCall {
                    call_id,
                    name,
                    arguments,
                    ..
                } => Some(crate::tool_parser::IncomingToolCall {
                    id: Some(call_id.clone()),
                    function: crate::tool_parser::IncomingFunction {
                        name: name.clone(),
                        arguments: arguments.clone(),
                    },
                }),
                _ => None,
            })
            .collect();
        if !transcript_text.is_empty()
            || !stored_tool_calls.is_empty()
            || !transcript_reasoning.is_empty()
        {
            transcript.push(crate::openai::IncomingMessage {
                role: "assistant".to_string(),
                content: crate::openai::ParsedContent::text_only(transcript_text.clone()),
                tool_calls: if stored_tool_calls.is_empty() {
                    None
                } else {
                    Some(stored_tool_calls)
                },
                tool_call_id: None,
                name: None,
                reasoning_content: if transcript_reasoning.is_empty() {
                    None
                } else {
                    Some(transcript_reasoning.clone())
                },
            });
        }
        store_ref.insert(crate::response_store::StoredEntry {
            id: resp_id.clone(),
            kind: crate::response_store::StoredKind::Response,
            model: model.clone(),
            created_at,
            messages: transcript,
            body,
            last_access: std::time::Instant::now(),
        });
    }
    // 2026-09-26: Append the new input items and the assistant output to the
    // linked conversation. A failure is logged and does not affect the stream.
    if let Some(cid) = conversation_id.as_ref() {
        let mut batch = conv_new_user_items.clone();
        if !transcript_text.is_empty() || !transcript_reasoning.is_empty() {
            let mut item = serde_json::json!({
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": transcript_text}],
            });
            if !transcript_reasoning.is_empty() {
                item["reasoning_content"] = serde_json::json!(transcript_reasoning);
            }
            batch.push(item);
        }
        if !batch.is_empty()
            && let Err(e) = state_arc.conversation_store.add_items(cid, batch)
        {
            tracing::warn!(
                "responses_stream_finalize: conversation_store.add_items failed for {cid}: {e:?}"
            );
        }
    }
    // 2026-09-26: `response.refusal.done` precedes `response.completed`.
    if let Some(ref r) = refusal_text {
        let ev = crate::openai::ResponsesStreamEvent::RefusalDone {
            sequence_number: seq,
            item_id: message_item_id.clone(),
            output_index,
            content_index: 0,
            refusal: r.clone(),
        };
        emit(tx, &ev).await;
        seq += 1;
    }
    let completed = crate::openai::ResponsesStreamEvent::Completed {
        sequence_number: seq,
        response: final_resp,
    };
    emit(tx, &completed).await;
}
