// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The incoming OpenAI chat message, and its builders for the
//! Responses API and for stored conversations.
//!
//! Owner: server (OpenAI adapter).
//! Invariants:
//! - `ParsedContent::media` holds images and videos in the order they
//!   appear in the request.

use serde::Deserialize;

// 2026-09-26: The IR's modality tag, so the wire and the IR share one enum.
pub use crate::ir::MediaKind;

#[derive(Debug, Deserialize, Clone)]
pub struct IncomingMessage {
    pub role: String,
    #[serde(default, deserialize_with = "deserialize_message_content")]
    pub content: ParsedContent,
    #[serde(default)]
    pub tool_calls: Option<Vec<crate::tool_parser::IncomingToolCall>>,
    #[serde(default)]
    pub tool_call_id: Option<String>,
    #[serde(default)]
    pub name: Option<String>,
    /// 2026-09-26: A prior assistant turn's reasoning trace, sent back so the
    /// template can re-render its `<think>` block. The wire name is
    /// `reasoning_content` or `reasoning`.
    #[serde(default, alias = "reasoning")]
    pub reasoning_content: Option<String>,
}

/// 2026-09-26: A message's content: all text parts joined, plus every media
/// item in the order the client sent it.
#[derive(Debug, Clone, Default)]
pub struct ParsedContent {
    pub text: String,
    /// 2026-09-26: Images and videos as one tagged sequence, so their
    /// relative order survives.
    pub media: Vec<MediaRef>,
}

/// 2026-09-26: One media item from the wire: its kind, and its URI (a
/// `data:` URI, raw base64, or a remote URL resolved later).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MediaRef {
    pub kind: MediaKind,
    pub uri: String,
}

impl ParsedContent {
    /// 2026-09-26: Content with text and no media.
    pub fn text_only(text: String) -> Self {
        ParsedContent {
            text,
            media: Vec::new(),
        }
    }

    /// 2026-09-26: The image URIs, in order. `api/stored.rs` lists stored
    /// input items from this view, so videos are left out there.
    pub fn images(&self) -> impl Iterator<Item = &String> {
        self.media
            .iter()
            .filter(|m| m.kind == MediaKind::Image)
            .map(|m| &m.uri)
    }

    pub fn has_images(&self) -> bool {
        self.images().next().is_some()
    }
}

impl IncomingMessage {
    /// 2026-09-26: A text-only system message; the Responses adapter puts
    /// `instructions` in one.
    pub fn synthetic_system(text: String) -> Self {
        Self {
            role: "system".to_string(),
            content: ParsedContent::text_only(text),
            tool_calls: None,
            tool_call_id: None,
            name: None,
            reasoning_content: None,
        }
    }

    /// 2026-09-26: A text-only user message; the Responses adapter puts a
    /// string `input` in one.
    pub fn synthetic_user_text(text: String) -> Self {
        Self {
            role: "user".to_string(),
            content: ParsedContent::text_only(text),
            tool_calls: None,
            tool_call_id: None,
            name: None,
            reasoning_content: None,
        }
    }

    /// 2026-09-26: Rebuild a message from a stored conversation item:
    /// `role`, the text of a string or of the parts' `text` fields, and
    /// `reasoning_content`, which the Responses paths store beside the
    /// assistant text. An item without a string `role` gives `None`.
    pub fn from_conversation_item(item: &serde_json::Value) -> Option<Self> {
        let role = item.get("role").and_then(|v| v.as_str())?;
        let content = item.get("content");
        let text = match content {
            Some(serde_json::Value::String(s)) => s.clone(),
            Some(serde_json::Value::Array(parts)) => parts
                .iter()
                .filter_map(|p| {
                    p.get("text")
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string())
                })
                .collect::<Vec<_>>()
                .join(""),
            _ => String::new(),
        };
        let reasoning_content = item
            .get("reasoning_content")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        Some(Self {
            role: role.to_string(),
            content: ParsedContent::text_only(text),
            tool_calls: None,
            tool_call_id: None,
            name: None,
            reasoning_content,
        })
    }

    /// 2026-09-26: Translate one Responses `input` array item into a chat
    /// message. `None` for a `reasoning` item, an unknown type, or an item
    /// missing a field it needs; `lower_responses_to_chat` skips those.
    pub fn from_responses_input_item(v: &serde_json::Value) -> Option<Self> {
        let obj = v.as_object()?;
        let kind = obj
            .get("type")
            .and_then(|t| t.as_str())
            .unwrap_or("message");
        match kind {
            "message" => {
                let role = match obj.get("role").and_then(|r| r.as_str()).unwrap_or("user") {
                    // 2026-09-26: `developer` becomes `user`, not `system`:
                    // some templates allow a system message only at the start.
                    "developer" => "user",
                    other => other,
                }
                .to_string();
                let content_val = obj.get("content")?;
                Some(Self {
                    role,
                    content: ParsedContent::from_responses_content(content_val),
                    tool_calls: None,
                    tool_call_id: None,
                    name: None,
                    reasoning_content: None,
                })
            }
            // 2026-09-26: A prior `function_call` becomes an assistant message
            // with that tool call, so the template renders it before its
            // `function_call_output`.
            "function_call" => {
                let name = obj.get("name").and_then(|v| v.as_str())?.to_string();
                let arguments = obj
                    .get("arguments")
                    .and_then(|v| v.as_str())
                    .unwrap_or("{}")
                    .to_string();
                let call_id = obj
                    .get("call_id")
                    .and_then(|v| v.as_str())
                    .or_else(|| obj.get("id").and_then(|v| v.as_str()))
                    .unwrap_or("")
                    .to_string();
                Some(Self {
                    role: "assistant".to_string(),
                    content: ParsedContent::default(),
                    tool_calls: Some(vec![crate::tool_parser::IncomingToolCall {
                        id: Some(call_id),
                        function: crate::tool_parser::IncomingFunction { name, arguments },
                    }]),
                    tool_call_id: None,
                    name: None,
                    reasoning_content: None,
                })
            }
            // 2026-09-26: The client's tool result becomes a `tool` message
            // answering `call_id`.
            "function_call_output" => {
                let call_id = obj
                    .get("call_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let output = match obj.get("output") {
                    Some(serde_json::Value::String(s)) => ParsedContent::text_only(s.clone()),
                    // 2026-09-26: Structured output parts keep their text and
                    // media, so an image a tool returned reaches the vision
                    // encoder. An array with no recognised part is passed on
                    // as its JSON text.
                    Some(arr @ serde_json::Value::Array(_)) => {
                        let parsed = ParsedContent::from_responses_content(arr);
                        if parsed.text.is_empty() && parsed.media.is_empty() {
                            ParsedContent::text_only(arr.to_string())
                        } else {
                            parsed
                        }
                    }
                    Some(other) => ParsedContent::text_only(other.to_string()),
                    None => ParsedContent::default(),
                };
                let name = obj
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                Some(Self {
                    role: "tool".to_string(),
                    content: output,
                    tool_calls: None,
                    tool_call_id: Some(call_id),
                    name: if name.is_empty() { None } else { Some(name) },
                    reasoning_content: None,
                })
            }
            // 2026-09-26: Reasoning items are dropped, not fed back to the
            // model.
            "reasoning" => None,
            _ => None,
        }
    }
}

impl ParsedContent {
    /// 2026-09-26: Flatten a Responses content value (a string, or an array
    /// of text, image and video parts) into joined text plus media in part
    /// order. Used for `message` and `function_call_output` items.
    fn from_responses_content(v: &serde_json::Value) -> Self {
        let mut text = String::new();
        let mut media: Vec<MediaRef> = Vec::new();
        match v {
            serde_json::Value::String(s) => text.push_str(s),
            serde_json::Value::Array(parts) => {
                for part in parts {
                    if let Some(po) = part.as_object() {
                        let part_kind = po.get("type").and_then(|t| t.as_str()).unwrap_or("");
                        if matches!(part_kind, "input_text" | "output_text" | "text")
                            && let Some(t) = po.get("text").and_then(|t| t.as_str())
                        {
                            text.push_str(t);
                        } else if matches!(part_kind, "input_image" | "image_url" | "image")
                            && let Some(url) = responses_image_url(po)
                        {
                            media.push(MediaRef {
                                kind: MediaKind::Image,
                                uri: url,
                            });
                        } else if matches!(part_kind, "input_video" | "video_url" | "video")
                            && let Some(url) = responses_video_url(po)
                        {
                            media.push(MediaRef {
                                kind: MediaKind::Video,
                                uri: url,
                            });
                        }
                    }
                }
            }
            _ => {}
        }
        ParsedContent { text, media }
    }
}

/// 2026-09-26: The image URL or data URI of a Responses image part, from
/// `"image_url": "..."` or `"image_url": {"url": "..."}`.
fn responses_image_url(po: &serde_json::Map<String, serde_json::Value>) -> Option<String> {
    match po.get("image_url") {
        Some(serde_json::Value::String(s)) => Some(s.clone()),
        Some(serde_json::Value::Object(o)) => {
            o.get("url").and_then(|v| v.as_str()).map(|s| s.to_string())
        }
        _ => None,
    }
}

/// 2026-09-26: The video URL or data URI of a Responses video part, from a
/// `video_url` or `video` key, each as a string or as `{"url": "..."}`.
fn responses_video_url(po: &serde_json::Map<String, serde_json::Value>) -> Option<String> {
    for key in ["video_url", "video"] {
        match po.get(key) {
            Some(serde_json::Value::String(s)) => return Some(s.clone()),
            Some(serde_json::Value::Object(o)) => {
                if let Some(u) = o.get("url").and_then(|v| v.as_str()) {
                    return Some(u.to_string());
                }
            }
            _ => {}
        }
    }
    None
}

fn deserialize_message_content<'de, D>(d: D) -> Result<ParsedContent, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum RawContent {
        Str(String),
        Parts(Vec<ContentPart>),
        Null(()),
    }

    #[derive(Deserialize)]
    struct ContentPart {
        #[serde(rename = "type")]
        kind: String,
        text: Option<String>,
        image_url: Option<ImageUrl>,
        /// 2026-09-26: `{"type": "video_url", "video_url": {"url": "..."}}`.
        video_url: Option<Url>,
        /// 2026-09-26: `{"type": "video", "video": ...}`, as a string or as
        /// `{"url": "..."}`.
        video: Option<UrlOrString>,
    }

    #[derive(Deserialize)]
    struct ImageUrl {
        url: String,
    }

    #[derive(Deserialize)]
    struct Url {
        url: String,
    }

    #[derive(Deserialize)]
    #[serde(untagged)]
    enum UrlOrString {
        Obj { url: String },
        Str(String),
    }

    let mut out = ParsedContent::default();
    match RawContent::deserialize(d)? {
        RawContent::Str(s) => out.text = s,
        RawContent::Null(()) => {}
        RawContent::Parts(parts) => {
            let mut text_parts = Vec::new();
            for p in parts {
                match p.kind.as_str() {
                    "text" => {
                        if let Some(t) = p.text {
                            text_parts.push(t);
                        }
                    }
                    "image_url" => {
                        if let Some(iu) = p.image_url {
                            out.media.push(MediaRef {
                                kind: MediaKind::Image,
                                uri: iu.url,
                            });
                        }
                    }
                    "video_url" | "video" | "input_video" => {
                        let uri = if let Some(v) = p.video_url {
                            Some(v.url)
                        } else {
                            p.video.map(|v| match v {
                                UrlOrString::Obj { url } => url,
                                UrlOrString::Str(s) => s,
                            })
                        };
                        if let Some(uri) = uri {
                            out.media.push(MediaRef {
                                kind: MediaKind::Video,
                                uri,
                            });
                        }
                    }
                    _ => {}
                }
            }
            out.text = text_parts.join("");
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conversation_item_round_trips_reasoning_content() {
        let item = serde_json::json!({
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "answer"}],
            "reasoning_content": "thought",
        });
        let m = IncomingMessage::from_conversation_item(&item).expect("assistant item");
        assert_eq!(m.content.text, "answer");
        assert_eq!(m.reasoning_content.as_deref(), Some("thought"));
    }

    #[test]
    fn conversation_item_without_reasoning_stays_none() {
        let item = serde_json::json!({
            "type": "message",
            "role": "assistant",
            "content": "answer",
        });
        let m = IncomingMessage::from_conversation_item(&item).expect("assistant item");
        assert_eq!(m.content.text, "answer");
        assert_eq!(m.reasoning_content, None);
    }
}

#[cfg(test)]
#[path = "chat_message_video_tests.rs"]
mod video_wire_tests;
