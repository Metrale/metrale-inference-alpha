// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The provider-neutral streaming output. The streaming chat path
//! yields a [`DeltaStream`], and each surface encodes it into its own SSE
//! format (OpenAI chat: `openai::encode_sse_response`; the Anthropic and
//! Responses streams translate the same deltas).
//!
//! Owner: server (chat IR).
//! Invariants: none beyond the types.

#[derive(Debug, Clone, PartialEq)]
pub enum StreamDelta {
    /// 2026-09-26: Assistant text. `token_ids` stays empty unless the
    /// request set `return_token_ids`.
    Content { text: String, token_ids: Vec<u32> },
    /// 2026-09-26: Reasoning text; `token_ids` as for `Content`.
    Reasoning { text: String, token_ids: Vec<u32> },
    /// 2026-09-26: A tool call opens at slot `index`.
    ToolCallStart {
        index: usize,
        id: String,
        name: String,
    },
    /// 2026-09-26: A fragment of the argument JSON of the tool call at
    /// `index`.
    ToolCallArgs {
        index: usize,
        fragment: String,
        token_ids: Vec<u32>,
    },
    /// 2026-09-26: The refusal sentence `refusal::detect` found in the
    /// streamed text.
    Refusal { text: String },
    /// 2026-09-26: The last delta: finish reason and usage. `token_ids`
    /// carries the ids that no earlier delta carried. Their sum over the
    /// stream can still fall short of `usage.completion_tokens`: EOS and
    /// hard-stop tokens are counted but never streamed.
    Finish {
        reason: super::response::FinishReason,
        usage: super::response::Usage,
        token_ids: Vec<u32>,
    },
    /// 2026-09-26: A wire-ready error payload; the OpenAI encoder sends it
    /// as SSE data unchanged.
    Error { message: String },
}

/// 2026-09-26: The streaming counterpart of [`super::ChatResponse`].
pub type DeltaStream = std::pin::Pin<Box<dyn futures::Stream<Item = StreamDelta> + Send>>;
