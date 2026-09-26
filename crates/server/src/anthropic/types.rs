// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Anthropic Messages API wire types: the request as deserialized
//! and the non-streaming response as serialized.
//!
//! Owner: server (Anthropic adapter).
//! Invariants: none beyond the types.

use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize)]
pub struct MessagesRequest {
    pub model: String,
    pub max_tokens: usize,
    #[serde(default)]
    pub system: Option<SystemContent>,
    pub messages: Vec<AnthropicMessage>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_k: Option<u32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub tools: Option<Vec<AnthropicTool>>,
    #[serde(default)]
    pub tool_choice: Option<AnthropicToolChoice>,
    #[serde(default)]
    pub stop_sequences: Vec<String>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub thinking: Option<ThinkingConfig>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum SystemContent {
    Text(String),
    Blocks(Vec<SystemBlock>),
}

#[derive(Debug, Deserialize)]
pub struct SystemBlock {
    #[serde(rename = "type")]
    pub block_type: String,
    #[serde(default)]
    pub text: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct AnthropicMessage {
    pub role: String,
    pub content: AnthropicContent,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum AnthropicContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
pub enum ContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        #[serde(default)]
        content: Option<ToolResultContent>,
        /// 2026-09-26: Anthropic's tool-failure flag. `Some(true)` becomes
        /// `Message::tool_error`, which `api::chat::msg_entry` renders as a
        /// `[tool error]` line ahead of the result text.
        #[serde(default)]
        is_error: Option<bool>,
    },
    #[serde(rename = "thinking")]
    Thinking {
        #[serde(default)]
        thinking: Option<String>,
    },
    #[serde(rename = "image")]
    Image { source: ImageSourceBlock },
    #[serde(other)]
    Unknown,
}

/// 2026-09-26: Anthropic image source. A `base64` source uses `media_type`
/// and `data`; a `url` source uses `url`.
#[derive(Debug, Deserialize)]
pub struct ImageSourceBlock {
    #[serde(rename = "type")]
    pub source_type: String,
    #[serde(default)]
    pub media_type: Option<String>,
    #[serde(default)]
    pub data: Option<String>,
    #[serde(default)]
    pub url: Option<String>,
}

impl ImageSourceBlock {
    /// 2026-09-26: The image as a URI string: `data:<media_type>;base64,<data>`
    /// for a `base64` source (`media_type` defaults to `image/png`), the URL
    /// for a `url` source. `None` when that field is missing or the source
    /// type is neither.
    pub(super) fn maybe_get_image_uri(&self) -> Option<String> {
        match self.source_type.as_str() {
            "base64" => {
                let data = self.data.as_ref()?;
                let mt = self.media_type.as_deref().unwrap_or("image/png");
                Some(format!("data:{mt};base64,{data}"))
            }
            "url" => self.url.clone(),
            _ => None,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum ToolResultContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

impl ToolResultContent {
    pub(super) fn to_text(&self) -> String {
        match self {
            ToolResultContent::Text(s) => s.clone(),
            ToolResultContent::Blocks(blocks) => blocks
                .iter()
                .filter_map(|b| match b {
                    ContentBlock::Text { text } => Some(text.clone()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n"),
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct AnthropicTool {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    pub input_schema: serde_json::Value,
}

#[derive(Debug, Deserialize)]
pub struct AnthropicToolChoice {
    #[serde(rename = "type")]
    pub choice_type: String,
    /// 2026-09-26: Read only when `type` is `tool`.
    #[serde(default)]
    pub name: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct ThinkingConfig {
    #[serde(rename = "type")]
    pub thinking_type: String,
    #[serde(default)]
    pub budget_tokens: Option<usize>,
}

#[derive(Debug, Serialize)]
pub struct MessagesResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub response_type: String,
    pub role: String,
    pub content: Vec<ResponseBlock>,
    pub model: String,
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
    pub usage: AnthropicUsage,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type")]
pub enum ResponseBlock {
    #[serde(rename = "thinking")]
    Thinking { thinking: String },
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
}

#[derive(Debug, Serialize)]
pub struct AnthropicUsage {
    pub input_tokens: usize,
    pub output_tokens: usize,
    /// 2026-09-26: Prompt tokens served from the prefix cache
    /// (`ir::Usage::cached_prompt_tokens`). Always serialized; 0 on a miss.
    pub cache_read_input_tokens: usize,
}
