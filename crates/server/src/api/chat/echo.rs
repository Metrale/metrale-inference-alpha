// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: OpenAI request fields that only shape the encoded response
//! (`service_tier` and `metadata` echo, `store: true` persistence,
//! `stream_options.include_usage`). They travel beside the IR request, not in
//! it, and generation never reads them.
//!
//! Owner: server (chat API).
//! Invariants: none beyond the types.

#[derive(Debug, Clone, Default)]
pub(crate) struct ResponseEcho {
    pub(crate) service_tier: Option<String>,
    pub(crate) metadata: Option<std::collections::HashMap<String, String>>,
    /// 2026-09-26: `store: true`: the encoder saves the completion in
    /// `response_store`.
    pub(crate) store: bool,
    /// 2026-09-26: `stream_options.include_usage`, passed to
    /// `encode_sse_response`.
    pub(crate) include_usage: bool,
}

impl<'a> From<&'a crate::openai::ChatCompletionRequest> for ResponseEcho {
    /// 2026-09-26: Absent `store` and `stream_options` read as `false`.
    fn from(req: &'a crate::openai::ChatCompletionRequest) -> Self {
        ResponseEcho {
            service_tier: req.service_tier.clone(),
            metadata: req.metadata.clone(),
            store: req.store.unwrap_or(false),
            include_usage: req.stream_options.map(|o| o.include_usage).unwrap_or(false),
        }
    }
}
