// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The `/v1/responses` wire types. A request is lowered into a
//! `ChatCompletionRequest` (`lower_responses_to_chat`), run by the chat
//! pipeline, and the result is written back in these shapes.
//! `previous_response_id` resumes from the transcript kept in
//! `crate::response_store`. Built-in tool types are refused with a 400.
//!
//! Owner: server (OpenAI adapter).
//! Invariants: none beyond the types.

use serde::{Deserialize, Serialize};

use super::*;

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct ResponsesRequest {
    pub model: String,
    /// 2026-09-26: A string or an array of input items; anything else is a
    /// 400.
    pub input: serde_json::Value,
    #[serde(default)]
    pub instructions: Option<String>,
    #[serde(default)]
    pub max_output_tokens: Option<usize>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub stream: bool,
    /// 2026-09-26: Raw tool list, parsed in `lower_responses_to_chat` so a
    /// built-in tool type gets a 400 that names it.
    #[serde(default)]
    pub tools: Option<Vec<serde_json::Value>>,
    /// 2026-09-26: Raw `tool_choice`, parsed in `lower_responses_to_chat`,
    /// which rewrites the flat `{"type":"function","name":"X"}` into the
    /// chat shape `ToolChoice` reads.
    #[serde(default)]
    pub tool_choice: Option<serde_json::Value>,
    #[serde(default)]
    pub metadata: Option<std::collections::HashMap<String, String>>,
    #[serde(default)]
    pub reasoning: Option<ReasoningConfig>,
    #[serde(default)]
    pub service_tier: Option<String>,
    /// 2026-09-26: A prior response id. Its stored transcript goes before
    /// the current `input`; an unknown or expired id is a 400.
    #[serde(default)]
    pub previous_response_id: Option<String>,
    /// 2026-09-26: Store the response for later retrieval. `None` = true
    /// (`api/responses.rs`).
    #[serde(default)]
    pub store: Option<bool>,
    /// 2026-09-26: Accepted and not read, like `include`, `truncation`,
    /// `parallel_tool_calls`, `max_tool_calls` and `text` below. The response
    /// is always finished before it is returned, and
    /// `POST /v1/responses/{id}/cancel` answers 400
    /// `response_not_cancellable`.
    #[serde(default)]
    pub background: Option<bool>,
    #[serde(default)]
    pub include: Option<Vec<String>>,
    #[serde(default)]
    pub truncation: Option<String>,
    /// 2026-09-26: A conversation id, as a string or `{"id": ...}`. Its
    /// stored items go before `input`, and the turn's items are then
    /// appended to it (a failed append is logged). An unknown id is a 404.
    #[serde(default)]
    pub conversation: Option<serde_json::Value>,
    #[serde(default)]
    pub parallel_tool_calls: Option<bool>,
    #[serde(default)]
    pub max_tool_calls: Option<u32>,
    #[serde(default)]
    pub text: Option<serde_json::Value>,
}

#[derive(Debug, Serialize)]
pub struct ResponsesResponse {
    pub id: String,
    pub object: &'static str,
    pub created_at: u64,
    pub model: String,
    pub status: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<serde_json::Value>,
    pub output: Vec<ResponsesOutputItem>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<serde_json::Value>,
    pub usage: ResponsesUsage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<std::collections::HashMap<String, String>>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponsesOutputItem {
    Message {
        id: String,
        status: &'static str,
        role: &'static str,
        content: Vec<ResponsesContentPart>,
    },
    FunctionCall {
        id: String,
        call_id: String,
        name: String,
        arguments: String,
        status: &'static str,
    },
    /// 2026-09-26: The reasoning trace, sent whole as the item's one
    /// `summary_text` part; no summarizer runs.
    Reasoning {
        id: String,
        summary: Vec<ResponsesSummaryPart>,
    },
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponsesSummaryPart {
    SummaryText { text: String },
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponsesContentPart {
    OutputText {
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        annotations: Option<Vec<Annotation>>,
    },
}

#[derive(Debug, Serialize)]
pub struct ResponsesUsage {
    pub input_tokens: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_tokens_details: Option<PromptTokensDetails>,
    pub output_tokens: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_tokens_details: Option<CompletionTokensDetails>,
    pub total_tokens: usize,
}

/// 2026-09-26: One streamed `/v1/responses` event. The handler sends each
/// as an SSE frame named by [`responses_event_name`], opens every stream
/// with `response.created` and `response.in_progress`, and adds one to
/// `sequence_number` after each event it sends.
#[derive(Debug, Serialize)]
#[serde(tag = "type")]
pub enum ResponsesStreamEvent {
    #[serde(rename = "response.created")]
    Created {
        sequence_number: u64,
        response: ResponsesStreamEnvelope,
    },
    #[serde(rename = "response.in_progress")]
    InProgress {
        sequence_number: u64,
        response: ResponsesStreamEnvelope,
    },
    #[serde(rename = "response.output_item.added")]
    OutputItemAdded {
        sequence_number: u64,
        output_index: usize,
        item: ResponsesOutputItem,
    },
    #[serde(rename = "response.content_part.added")]
    ContentPartAdded {
        sequence_number: u64,
        item_id: String,
        output_index: usize,
        content_index: usize,
        part: ResponsesContentPart,
    },
    #[serde(rename = "response.output_text.delta")]
    OutputTextDelta {
        sequence_number: u64,
        item_id: String,
        output_index: usize,
        content_index: usize,
        delta: String,
    },
    #[serde(rename = "response.output_text.done")]
    OutputTextDone {
        sequence_number: u64,
        item_id: String,
        output_index: usize,
        content_index: usize,
        text: String,
    },
    #[serde(rename = "response.function_call_arguments.delta")]
    FunctionCallArgumentsDelta {
        sequence_number: u64,
        item_id: String,
        output_index: usize,
        delta: String,
    },
    #[serde(rename = "response.function_call_arguments.done")]
    FunctionCallArgumentsDone {
        sequence_number: u64,
        item_id: String,
        output_index: usize,
        arguments: String,
    },
    #[serde(rename = "response.output_item.done")]
    OutputItemDone {
        sequence_number: u64,
        output_index: usize,
        item: ResponsesOutputItem,
    },
    #[serde(rename = "response.refusal.delta")]
    RefusalDelta {
        sequence_number: u64,
        item_id: String,
        output_index: usize,
        content_index: usize,
        delta: String,
    },
    #[serde(rename = "response.refusal.done")]
    RefusalDone {
        sequence_number: u64,
        item_id: String,
        output_index: usize,
        content_index: usize,
        refusal: String,
    },
    // 2026-09-26: A reasoning item's single `summary_text` part streams
    // through the four events below; `summary_index` is always 0.
    #[serde(rename = "response.reasoning_summary_part.added")]
    ReasoningSummaryPartAdded {
        sequence_number: u64,
        item_id: String,
        output_index: usize,
        summary_index: usize,
        part: ResponsesSummaryPart,
    },
    #[serde(rename = "response.reasoning_summary_text.delta")]
    ReasoningSummaryTextDelta {
        sequence_number: u64,
        item_id: String,
        output_index: usize,
        summary_index: usize,
        delta: String,
    },
    #[serde(rename = "response.reasoning_summary_text.done")]
    ReasoningSummaryTextDone {
        sequence_number: u64,
        item_id: String,
        output_index: usize,
        summary_index: usize,
        text: String,
    },
    #[serde(rename = "response.reasoning_summary_part.done")]
    ReasoningSummaryPartDone {
        sequence_number: u64,
        item_id: String,
        output_index: usize,
        summary_index: usize,
        part: ResponsesSummaryPart,
    },
    #[serde(rename = "response.completed")]
    Completed {
        sequence_number: u64,
        response: ResponsesResponse,
    },
    #[serde(rename = "response.failed")]
    Failed {
        sequence_number: u64,
        response: ResponsesStreamEnvelope,
        error: serde_json::Value,
    },
}

/// 2026-09-26: The response envelope of the `created`, `in_progress` and
/// `failed` events: `ResponsesResponse` without output items or usage.
#[derive(Debug, Serialize)]
pub struct ResponsesStreamEnvelope {
    pub id: String,
    pub object: &'static str,
    pub created_at: u64,
    pub model: String,
    pub status: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<std::collections::HashMap<String, String>>,
}

/// 2026-09-26: The SSE `event:` name of a stream event; it equals the
/// event's serde `type` tag.
pub fn responses_event_name(ev: &ResponsesStreamEvent) -> &'static str {
    match ev {
        ResponsesStreamEvent::Created { .. } => "response.created",
        ResponsesStreamEvent::InProgress { .. } => "response.in_progress",
        ResponsesStreamEvent::OutputItemAdded { .. } => "response.output_item.added",
        ResponsesStreamEvent::ContentPartAdded { .. } => "response.content_part.added",
        ResponsesStreamEvent::OutputTextDelta { .. } => "response.output_text.delta",
        ResponsesStreamEvent::OutputTextDone { .. } => "response.output_text.done",
        ResponsesStreamEvent::FunctionCallArgumentsDelta { .. } => {
            "response.function_call_arguments.delta"
        }
        ResponsesStreamEvent::FunctionCallArgumentsDone { .. } => {
            "response.function_call_arguments.done"
        }
        ResponsesStreamEvent::OutputItemDone { .. } => "response.output_item.done",
        ResponsesStreamEvent::RefusalDelta { .. } => "response.refusal.delta",
        ResponsesStreamEvent::RefusalDone { .. } => "response.refusal.done",
        ResponsesStreamEvent::ReasoningSummaryPartAdded { .. } => {
            "response.reasoning_summary_part.added"
        }
        ResponsesStreamEvent::ReasoningSummaryTextDelta { .. } => {
            "response.reasoning_summary_text.delta"
        }
        ResponsesStreamEvent::ReasoningSummaryTextDone { .. } => {
            "response.reasoning_summary_text.done"
        }
        ResponsesStreamEvent::ReasoningSummaryPartDone { .. } => {
            "response.reasoning_summary_part.done"
        }
        ResponsesStreamEvent::Completed { .. } => "response.completed",
        ResponsesStreamEvent::Failed { .. } => "response.failed",
    }
}
