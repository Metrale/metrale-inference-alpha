// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: `ChatTokenizer`: a Hugging Face tokenizer, the chat template prompts are
//! rendered with (minijinja, or the DeepSeek-V4 message encoder), and a streaming decoder.
//!
//! Owner: server (tokenizer).
//! Invariants: none beyond the types.

use anyhow::Result;
use tokenizers::Tokenizer;

/// 2026-09-26: Return a copy of `messages` in which every string
/// `tool_calls[*].function.arguments` that parses as JSON is replaced by the parsed value;
/// anything else is left as it is. Templates iterate the arguments as a map, e.g.
/// `_args.items()` in jinja-templates/openai/minimax_m2.jinja.
fn normalize_tool_call_arguments(messages: &[serde_json::Value]) -> Vec<serde_json::Value> {
    let mut total_parsed = 0usize;
    let mut total_seen = 0usize;
    let out: Vec<_> = messages
        .iter()
        .map(|msg| {
            let mut msg = msg.clone();
            let Some(tool_calls) = msg.get_mut("tool_calls").and_then(|v| v.as_array_mut()) else {
                return msg;
            };
            for tc in tool_calls.iter_mut() {
                let Some(function) = tc.get_mut("function") else {
                    continue;
                };
                let Some(args) = function.get_mut("arguments") else {
                    continue;
                };
                total_seen += 1;
                let parsed_owned = if let Some(s) = args.as_str() {
                    serde_json::from_str::<serde_json::Value>(s).ok()
                } else {
                    None
                };
                if let Some(parsed) = parsed_owned {
                    *args = parsed;
                    total_parsed += 1;
                }
            }
            msg
        })
        .collect();
    if total_seen > 0 {
        tracing::debug!(
            "F76 normalize: {}/{} tool_call arguments parsed string→dict",
            total_parsed,
            total_seen,
        );
    }
    out
}

mod chat_impl;
pub(crate) mod chat_render;
mod deepseek_v4;
pub(crate) mod jinja_helpers;
mod kimi_k3;
mod message_preprocess;

pub(crate) use message_preprocess::{
    autoclose_assistant_think, remap_developer_role, resolve_think_control,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ChatEncoding {
    Jinja,
    DeepseekV4,
    KimiK3XtmlUnsupported,
}

pub struct ChatTokenizer {
    tokenizer: Tokenizer,
    eos_token_id: u32,
    supports_thinking: bool,
    chat_encoding: ChatEncoding,
    /// 2026-09-26: The checkpoint's own Qwen template renders the tool schema, so the parser's
    /// tool prompt is not added (`checkpoint_owns_qwen_tool_prompt` in chat_impl.rs).
    native_qwen_tool_template: bool,
    /// 2026-09-26: Template source `jinja_env` was built from: the override, the model's own
    /// template, or the ChatML default; empty for an official Kimi K3 checkpoint.
    #[allow(dead_code)]
    chat_template: String,
    /// 2026-09-26: Environment holding `chat_template`, compiled once in `from_model_dir`.
    jinja_env: minijinja::Environment<'static>,
    /// 2026-09-26: Environment for `jinja-templates/openai/{model_type}.jinja` when that file
    /// exists and compiles; without it the OpenAI apply paths use `jinja_env`.
    openai_jinja_env: Option<minijinja::Environment<'static>>,
}

/// 2026-09-26: `tokenizers::DecodeStream` with its generic parameters fixed.
pub struct StreamingDecoder<'a> {
    inner: tokenizers::DecodeStream<
        'a,
        tokenizers::models::ModelWrapper,
        tokenizers::normalizers::NormalizerWrapper,
        tokenizers::pre_tokenizers::PreTokenizerWrapper,
        tokenizers::processors::PostProcessorWrapper,
        tokenizers::decoders::DecoderWrapper,
    >,
}

impl StreamingDecoder<'_> {
    /// 2026-09-26: Feed one token. `Ok(Some(text))` returns the new text once the decode has
    /// grown and does not end in U+FFFD; `Ok(None)` otherwise.
    pub fn step(&mut self, id: u32) -> Result<Option<String>> {
        self.inner
            .step(id)
            .map_err(|e| anyhow::anyhow!("Streaming decode error: {e}"))
    }
}

#[cfg(test)]
mod tests;
