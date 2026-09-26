// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Split a model's completed output into reasoning and answer, per reasoning-block format.
//!
//! | Format | Delimiters |
//! |---|---|
//! | [`ReasoningFormat::Qwen`], [`ReasoningFormat::DeepSeekR1`], [`ReasoningFormat::MiniMax`] | `<think>` / `</think>` |
//! | [`ReasoningFormat::Mistral`] | `[THINK]` / `[/THINK]` |
//! | [`ReasoningFormat::Gemma4`] | `<|channel>` / `<channel|>` |
//!
//! The three `<think>` formats share one implementation and differ only in
//! name; they assume the prompt already opened the block. Mistral uses the
//! same implementation but expects the model to emit its own `[THINK]`.
//! Gemma 4 has its own parser. At serve time the format comes from the
//! `[reasoning]` table of `tool_defaults.toml`, keyed by `model_type`, and a
//! thinking-capable model without an entry gets `qwen`
//! (`main_modules/serve_phases/tokenizer_runtime.rs`).
//!
//! Owner: server.
//! Invariants: none beyond the types.

mod parsers;
#[cfg(test)]
mod tests;

use std::str::FromStr;

use crate::tokenizer::ChatTokenizer;

/// 2026-09-26: Splits reasoning blocks out of completed model output.
pub trait ReasoningParser: Send + Sync {
    /// 2026-09-26: Parser name for logging (e.g. `"qwen"`, `"deepseek_r1"`).
    fn name(&self) -> &str;

    /// 2026-09-26: Opening delimiter (e.g. `"<think>"`, `"[THINK]"`, `"<|channel>"`).
    fn start_tag(&self) -> &str;

    /// 2026-09-26: Closing delimiter (e.g. `"</think>"`, `"[/THINK]"`, `"<channel|>"`).
    fn end_tag(&self) -> &str;

    /// 2026-09-26: The end tag's token id, or `None` unless the tag encodes to
    /// exactly one token.
    fn end_token_id(&self, tokenizer: &ChatTokenizer) -> Option<u32> {
        match tokenizer.encode(self.end_tag()) {
            Ok(ids) if ids.len() == 1 => Some(ids[0]),
            _ => None,
        }
    }

    /// 2026-09-26: Split completed text into `(reasoning, content)`.
    ///
    /// `enable_thinking` is the request's resolved thinking state. When it is
    /// `false` the reasoning is `None`; the answer is always in `content`.
    fn extract_thinking(&self, text: &str, enable_thinking: bool) -> (Option<String>, String);
}

/// 2026-09-26: Supported reasoning-block formats; the module doc lists each
/// one's delimiters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReasoningFormat {
    /// 2026-09-26: `<think>...</think>`, parsed as `qwen`.
    Qwen,
    /// 2026-09-26: `<think>...</think>`, parsed as `deepseek_r1`; also named
    /// `deepseek`, `nemotron`, `nemotron_h` and `nano_v3`.
    DeepSeekR1,
    /// 2026-09-26: `<think>...</think>`, parsed as `minimax`.
    MiniMax,
    /// 2026-09-26: `[THINK]...[/THINK]`, opened by the model.
    Mistral,
    /// 2026-09-26: `<|channel>thought ... <channel|> ...`, Gemma 4's channel format.
    Gemma4,
}

impl FromStr for ReasoningFormat {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "qwen" | "qwen3" => Ok(Self::Qwen),
            "deepseek_r1" | "deepseek" | "nemotron" | "nemotron_h" | "nano_v3" => {
                Ok(Self::DeepSeekR1)
            }
            "minimax" | "minimax_m2" => Ok(Self::MiniMax),
            "mistral" => Ok(Self::Mistral),
            "gemma4" | "gemma" => Ok(Self::Gemma4),
            other => Err(format!(
                "Unknown reasoning parser '{other}'. Supported: qwen, \
                 deepseek_r1, minimax, mistral, gemma4"
            )),
        }
    }
}

impl ReasoningFormat {
    /// 2026-09-26: A boxed parser for this format.
    pub fn into_parser(self) -> Box<dyn ReasoningParser> {
        parsers::build(self)
    }
}
