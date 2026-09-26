// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The OpenAI assistant message and its URL-citation
//! annotations.
//!
//! Owner: server (OpenAI adapter).
//! Invariants: none beyond the types.

use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ChatMessage {
    pub role: String,
    /// 2026-09-26: The reasoning trace. It is the message's only reasoning
    /// field; there is no `reasoning` copy. Requests may send either name
    /// (`alias = "reasoning"` on `IncomingMessage::reasoning_content`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<crate::tool_parser::ToolCall>>,
    /// 2026-09-26: URL citations found in the final `content`; omitted when
    /// there are none.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub annotations: Option<Vec<Annotation>>,
    /// 2026-09-26: The refusal sentence found by `refusal::detect`; the
    /// blocking path then sets `content` to null.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub refusal: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Annotation {
    UrlCitation { url_citation: UrlCitation },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UrlCitation {
    pub start_index: usize,
    pub end_index: usize,
    pub url: String,
    pub title: String,
}

impl From<crate::citation::Citation> for Annotation {
    fn from(c: crate::citation::Citation) -> Self {
        Annotation::UrlCitation {
            url_citation: UrlCitation {
                start_index: c.start_index,
                end_index: c.end_index,
                url: c.url,
                title: c.title,
            },
        }
    }
}

/// 2026-09-26: [`crate::citation::extract_url_citations`] as annotations;
/// `None` when it finds none.
pub fn extract_url_annotations(content: &str) -> Option<Vec<Annotation>> {
    let cits = crate::citation::extract_url_citations(content);
    if cits.is_empty() {
        None
    } else {
        Some(cits.into_iter().map(Annotation::from).collect())
    }
}

/// 2026-09-26: [`crate::citation::merged_citations`] (URLs plus structured
/// citations) as annotations; `None` when it finds none.
pub fn merged_annotations(content: &str) -> Option<Vec<Annotation>> {
    crate::citation::merged_citations(content)
        .map(|cits| cits.into_iter().map(Annotation::from).collect())
}
