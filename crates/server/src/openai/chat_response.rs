// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The OpenAI chat completion response, its usage block, and the
//! `/v1/models` entry.
//!
//! Owner: server (OpenAI adapter).
//! Invariants: none beyond the types.

use serde::{Deserialize, Serialize};

use super::*;

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub model: String,
    pub system_fingerprint: Option<String>,
    pub choices: Vec<ChatChoice>,
    pub usage: Usage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_tier: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<std::collections::HashMap<String, String>>,
}

#[derive(Debug, Serialize)]
pub struct ChatChoice {
    pub index: usize,
    pub message: ChatMessage,
    pub finish_reason: String,
    pub logprobs: Option<ChoiceLogprobs>,
}

/// 2026-09-26: The OpenAI usage block plus four timing fields, all from
/// `ir::Usage`.
#[derive(Debug, Clone, Serialize)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
    /// 2026-09-26: Always set by `From<&ir::Usage>`, even when no prompt
    /// token came from the prefix cache.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens_details: Option<PromptTokensDetails>,
    /// 2026-09-26: Always set by `From<&ir::Usage>`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_tokens_details: Option<CompletionTokensDetails>,
    /// 2026-09-26: `ir::Usage::time_to_first_token_ms`.
    #[serde(rename = "time_to_first_token_ms")]
    pub time_to_first_token_ms: f64,
    /// 2026-09-26: `ir::Usage::decode_rate_tok_s`.
    #[serde(rename = "response_token/s")]
    pub response_tokens_per_second: f64,
    /// 2026-09-26: The decode window on the server's clock; a client gets
    /// the inter-token latency as `decode_time_ms / (completion_tokens − 1)`
    /// without timing SSE arrivals. Meaningless below two completion
    /// tokens; the raw window is sent regardless.
    #[serde(rename = "decode_time_ms")]
    pub decode_time_ms: f64,
    /// 2026-09-26: `ir::Usage::total_time_ms`.
    #[serde(rename = "total_time_ms")]
    pub total_time_ms: f64,
}

impl From<&crate::ir::Usage> for Usage {
    /// 2026-09-26: The chat usage mapping, used by the blocking encoder, the
    /// streaming encoder and the streamed `--dump` capture. The audio and
    /// rejected-prediction counters are always 0.
    fn from(u: &crate::ir::Usage) -> Self {
        Self {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
            total_tokens: u.prompt_tokens + u.completion_tokens,
            prompt_tokens_details: Some(PromptTokensDetails {
                cached_tokens: u.cached_prompt_tokens,
                audio_tokens: 0,
            }),
            completion_tokens_details: Some(CompletionTokensDetails {
                reasoning_tokens: u.reasoning_tokens,
                audio_tokens: 0,
                accepted_prediction_tokens: u.accepted_prediction_tokens,
                rejected_prediction_tokens: 0,
            }),
            time_to_first_token_ms: u.time_to_first_token_ms,
            response_tokens_per_second: u.response_tokens_per_second,
            decode_time_ms: u.decode_time_ms,
            total_time_ms: u.total_time_ms(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PromptTokensDetails {
    pub cached_tokens: usize,
    pub audio_tokens: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CompletionTokensDetails {
    pub reasoning_tokens: usize,
    pub audio_tokens: usize,
    /// 2026-09-26: Speculative draft tokens accepted for this request; the
    /// request's `prediction` field is not read.
    pub accepted_prediction_tokens: usize,
    pub rejected_prediction_tokens: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct TopLogprob {
    pub token: String,
    pub logprob: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes: Option<Vec<u8>>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TokenLogprobInfo {
    pub token: String,
    pub logprob: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes: Option<Vec<u8>>,
    pub top_logprobs: Vec<TopLogprob>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChoiceLogprobs {
    pub content: Vec<TokenLogprobInfo>,
}

#[derive(Debug, Serialize)]
pub struct ModelListResponse {
    pub object: String,
    pub data: Vec<ModelInfo>,
}

#[derive(Debug, Serialize)]
pub struct ModelInfo {
    pub id: String,
    pub object: String,
    pub created: u64,
    pub owned_by: String,
    /// 2026-09-26: The context length in tokens: `AppState::max_seq_len`,
    /// the bound at which `/v1/chat/completions` and `/v1/completions`
    /// reject a prompt. [`ModelInfo::advertise`] always sets it; `None` is
    /// omitted from the wire.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_model_len: Option<usize>,
}

impl ModelInfo {
    /// 2026-09-26: Build one `/v1/models` list entry; every entry of
    /// `list_models` comes from here. The retrieve handler
    /// (`api::models::get_model`) writes its own JSON with the same
    /// `max_seq_len`.
    pub fn advertise(id: String, max_seq_len: usize) -> Self {
        Self {
            id,
            object: "model".to_string(),
            created: crate::ids::unix_timestamp(),
            owned_by: crate::identity::OWNED_BY.to_string(),
            max_model_len: Some(max_seq_len),
        }
    }
}

impl ChatCompletionResponse {
    pub fn new(
        model: &str,
        content: String,
        reasoning_content: Option<String>,
        usage: Usage,
        finish_reason: &str,
    ) -> Self {
        Self {
            id: format!("chatcmpl-{}", uuid_v4()),
            object: "chat.completion".to_string(),
            created: unix_timestamp(),
            model: model.to_string(),
            system_fingerprint: Some("fp_metrale".to_string()),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".to_string(),
                    reasoning_content,
                    annotations: extract_url_annotations(&content),
                    refusal: None,
                    content: Some(content),
                    tool_calls: None,
                },
                finish_reason: finish_reason.to_string(),
                logprobs: None,
            }],
            usage,
            service_tier: None,
            metadata: None,
        }
    }

    pub fn with_tool_calls(
        model: &str,
        content: Option<String>,
        reasoning_content: Option<String>,
        tool_calls: Vec<crate::tool_parser::ToolCall>,
        usage: Usage,
    ) -> Self {
        Self {
            id: format!("chatcmpl-{}", uuid_v4()),
            object: "chat.completion".to_string(),
            created: unix_timestamp(),
            model: model.to_string(),
            system_fingerprint: Some("fp_metrale".to_string()),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".to_string(),
                    reasoning_content,
                    annotations: content.as_deref().and_then(extract_url_annotations),
                    refusal: None,
                    content,
                    tool_calls: Some(tool_calls),
                },
                finish_reason: "tool_calls".to_string(),
                logprobs: None,
            }],
            usage,
            service_tier: None,
            metadata: None,
        }
    }
}
