// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The provider-neutral chat message (request direction). The
//! OpenAI chat, Responses and Anthropic adapters build `Vec<Message>`, and
//! `build_msg_entries` turns it into template entries.
//!
//! Owner: server (chat IR).
//! Invariants: none beyond the types.

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
    /// 2026-09-26: Any other wire role string, kept verbatim. It reaches the
    /// template as that string, except `"developer"`, which
    /// `build_msg_entries` renders as `system` unless the tokenizer uses
    /// the DeepSeek-V4 encoding.
    Other(String),
}

impl Role {
    /// 2026-09-26: The wire role string; [`Role::Other`] returns its own.
    pub fn as_wire(&self) -> &str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
            Role::Other(s) => s,
        }
    }

    /// 2026-09-26: Parse one of the four known roles; `None` for anything
    /// else. The `From<&str>` impl keeps an unknown role as [`Role::Other`].
    pub fn from_wire(s: &str) -> Option<Self> {
        match s {
            "system" => Some(Role::System),
            "user" => Some(Role::User),
            "assistant" => Some(Role::Assistant),
            "tool" => Some(Role::Tool),
            _ => None,
        }
    }
}

impl From<&str> for Role {
    /// 2026-09-26: Map any wire role string, keeping an unknown one as
    /// [`Role::Other`].
    fn from(s: &str) -> Self {
        Self::from_wire(s).unwrap_or_else(|| Role::Other(s.to_string()))
    }
}

/// 2026-09-26: One chat message. `content` is a list of parts for every
/// role, so a tool result can carry media like any other message.
#[derive(Debug, Clone, PartialEq)]
pub struct Message {
    pub role: Role,
    pub content: Vec<ContentPart>,
    pub tool_calls: Vec<ToolCall>,
    /// 2026-09-26: Links a `Tool` message to the `ToolCall.id` it answers.
    pub tool_call_id: Option<String>,
    pub name: Option<String>,
    /// 2026-09-26: A prior assistant turn's reasoning trace, which
    /// `build_msg_entries` can hand to the template as `reasoning_content`.
    pub reasoning: Option<Reasoning>,
    /// 2026-09-26: The tool result reported an error. The Anthropic adapter
    /// sets it from `is_error`; `build_msg_entries` then prefixes the text
    /// with `[tool error]\n`.
    pub tool_error: bool,
}

impl Message {
    /// 2026-09-26: A system message holding only `text`. Built for the
    /// Anthropic `system` field and for `inject_tool_system_prompt`.
    pub fn synthetic_system(text: String) -> Self {
        Message {
            role: Role::System,
            content: vec![ContentPart::Text(text)],
            tool_calls: Vec::new(),
            tool_call_id: None,
            name: None,
            reasoning: None,
            tool_error: false,
        }
    }

    /// 2026-09-26: Prepend `prefix` to the first text part, or append a new
    /// text part when there is none. Either way `text()` becomes
    /// `prefix + old text()`.
    pub fn prepend_text(&mut self, prefix: &str) {
        for part in &mut self.content {
            if let ContentPart::Text(t) = part {
                *t = format!("{prefix}{t}");
                return;
            }
        }
        self.content.push(ContentPart::Text(prefix.to_string()));
    }

    /// 2026-09-26: The text parts concatenated in order; media parts are
    /// skipped.
    pub fn text(&self) -> String {
        let mut out = String::new();
        for part in &self.content {
            if let ContentPart::Text(t) = part {
                out.push_str(t);
            }
        }
        out
    }

    /// 2026-09-26: The message's media parts, tagged and in content order.
    /// `build_msg_entries` stores this on every entry, so the template
    /// renders images and videos in the order the client sent them.
    pub fn media_kinds(&self) -> Vec<MediaKind> {
        self.content
            .iter()
            .filter_map(|p| match p {
                ContentPart::Image(_) => Some(MediaKind::Image),
                ContentPart::Video(_) => Some(MediaKind::Video),
                ContentPart::Text(_) => None,
            })
            .collect()
    }
}

/// 2026-09-26: Which vision modality a media item is. The two use different
/// pad tokens, and they travel as one tagged sequence so their relative
/// order survives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MediaKind {
    Image,
    Video,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ContentPart {
    Text(String),
    Image(ImageSource),
    Video(VideoSource),
}

/// 2026-09-26: Where a video comes from. It reuses [`ImageData`]: both
/// modalities are classified by [`ImageData::from_uri`] and resolved under
/// the same remote-fetch policy (`resolve_media_uri` in
/// `api/chat/msg_entry.rs`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VideoSource {
    pub data: ImageData,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageSource {
    pub data: ImageData,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ImageData {
    /// 2026-09-26: Anything that is not an http(s) URL: a `data:` URI or
    /// raw base64. It is passed to the encoder unchanged.
    Base64(String),
    /// 2026-09-26: An http(s) URL. `resolve_media_uri` fetches it into a
    /// `data:` URI when `--vision-allow-remote-images` is set, and answers
    /// 400 otherwise; it is never passed to the encoder as is.
    Url(String),
}

impl ImageData {
    /// 2026-09-26: http(s) URLs become [`ImageData::Url`]; everything else
    /// becomes [`ImageData::Base64`].
    pub fn from_uri(s: String) -> Self {
        if s.starts_with("http://") || s.starts_with("https://") {
            ImageData::Url(s)
        } else {
            ImageData::Base64(s)
        }
    }
}

/// 2026-09-26: A tool call from an assistant turn. `arguments` is parsed
/// JSON, not a string.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Reasoning {
    pub text: String,
}
