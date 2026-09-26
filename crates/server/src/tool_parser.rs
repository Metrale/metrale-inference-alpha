// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tool calling for the OpenAI-compatible API: the request and
//! response types, the `ToolCallParser` trait, one parser per
//! `ToolCallFormat`, and the shared output parsers in the submodules.
//!
//! Owner: server (tool parsing).
//! Invariants:
//! - `next_tool_call_id` does not repeat an id within a process until its
//!   `u64` counter wraps: the counter is only ever incremented.

use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::grammar::{GrammarEngine, GrammarError};
use metrale_grammar::CompiledGrammar;

/// 2026-09-26: Process-wide source of the `call_*` tool-call ids. It is a
/// static so that ids stay unique across requests and across a model swap in
/// the same process; nothing resets it.
static TOOL_CALL_COUNTER: AtomicU64 = AtomicU64::new(0);

/// 2026-09-26: The next `call_{:016x}` id from `TOOL_CALL_COUNTER`.
fn next_tool_call_id() -> String {
    let id = TOOL_CALL_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("call_{id:016x}")
}

/// 2026-09-26: Characters accepted in a model-emitted tool name before
/// normalization. `:` is accepted so a namespaced name (`ns:tool`) scans as
/// one candidate; `normalize_tool_name` strips the namespace before the name
/// reaches the client.
fn is_tool_name_or_namespace_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.' | ':')
}

fn is_tool_name_component(s: &str) -> bool {
    !s.is_empty()
        && s.chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.'))
}

/// 2026-09-26: Guard for candidate names scanned with
/// `is_tool_name_or_namespace_char`. `normalize_tool_name` strips `ns:` only
/// when a namespace precedes the last colon and a valid name follows it, so a
/// surviving colon means the candidate is not a tool name: prose such as
/// `json:{"a":1}` scans as the name `json:`. The caller then leaves the text
/// for later passes.
fn is_normalized_tool_name(name: &str) -> bool {
    !name.is_empty() && !name.contains(':')
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ToolDefinition {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: FunctionDefinition,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FunctionDefinition {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parameters: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum ToolChoice {
    Mode(String),
    /// 2026-09-26: `{"type": "function", "function": {"name": "X"}}`, or
    /// `{"function": {"name": "X"}}`: only `function` is read.
    Specific {
        function: ToolChoiceFunction,
    },
}

#[derive(Debug, Clone, Deserialize)]
pub struct ToolChoiceFunction {
    pub name: String,
}

impl ToolChoice {
    pub fn is_none(&self) -> bool {
        matches!(self, Self::Mode(s) if s == "none")
    }
}

/// 2026-09-26: Tool call from a previous assistant message.
///
/// `Serialize` is derived for the response store, which writes these calls to
/// disk (`response_store.rs` `messages_to_disk_json`) in the same shape they
/// are parsed from.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct IncomingToolCall {
    #[serde(default)]
    pub id: Option<String>,
    pub function: IncomingFunction,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct IncomingFunction {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: FunctionCall,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct FunctionCall {
    pub name: String,
    pub arguments: String,
}

/// 2026-09-26: Tool call in a streaming chunk. The header chunk carries `id`,
/// `type` and `function.name` with empty `arguments`; an argument chunk
/// carries only `index` and an `arguments` fragment (`openai/stream_chunk.rs`).
#[derive(Debug, Clone, Serialize)]
pub struct ChunkToolCall {
    pub index: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub call_type: Option<String>,
    pub function: ChunkFunction,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChunkFunction {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    pub arguments: String,
}

/// 2026-09-26: Tags the streaming content sanitizer
/// (`api/sanitizer.rs` `sanitize_content_chunk`) uses to suppress tool-call
/// fragments that leak into the assistant content. Each parser declares its
/// own through `ToolCallParser::leak_markers`.
///
/// Semantics:
/// - Outside an envelope, a string from `orphan_open` starts suppression:
///   bytes are dropped up to and including the first string from `close`.
/// - A string from `envelope_open` enters envelope mode and is passed
///   through. In envelope mode `orphan_open` and `close` are not matched; a
///   string from `envelope_close` leaves the mode.
/// - Outside suppression and outside an envelope, a stray `close` string is
///   dropped.
/// - With `orphan_open` and `envelope_open` both empty, the sanitizer passes
///   text through unbuffered.
#[derive(Copy, Clone)]
pub struct LeakMarkers {
    pub orphan_open: &'static [&'static str],
    pub close: &'static [&'static str],
    /// 2026-09-26: Outer-envelope openers (e.g. `<minimax:tool_call>`).
    pub envelope_open: &'static [&'static str],
    /// 2026-09-26: Closers for `envelope_open` (e.g. `</minimax:tool_call>`).
    pub envelope_close: &'static [&'static str],
}

impl LeakMarkers {
    /// 2026-09-26: No markers: the sanitizer passes text through. It is the
    /// `ToolCallParser::leak_markers` default, kept by the Hermes, Mistral
    /// and bare-JSON parsers.
    pub const EMPTY: Self = Self {
        orphan_open: &[],
        close: &[],
        envelope_open: &[],
        envelope_close: &[],
    };
}

pub use prompt_levers::PromptLevers;

/// 2026-09-26: One tool-call format: the system prompt that teaches the
/// model the format, the rendering of earlier tool calls and tool results
/// into the prompt, the tool grammar, and the sanitizer's leak markers.
///
/// Parsing model output is shared by every format (`parse_tool_calls`,
/// `StreamingToolDetector`); the hooks below only adjust it.
pub trait ToolCallParser: Send + Sync {
    /// 2026-09-26: The `--tool-call-parser` name (e.g. "hermes"). Also
    /// matched by name in `api/chat/prepare.rs` and
    /// `api/chat/sampling_setup.rs`.
    fn name(&self) -> &str;

    /// 2026-09-26: The system prompt that teaches the model how to make tool
    /// calls. `levers` carries the model's prompt-rendering settings from its
    /// MODEL.toml `[behavior]` table.
    fn system_prompt(
        &self,
        tools: &[ToolDefinition],
        tool_choice: &ToolChoice,
        levers: &PromptLevers,
    ) -> String;

    /// 2026-09-26: Earlier assistant tool calls, rendered as prompt text.
    fn format_tool_calls(&self, calls: &[IncomingToolCall]) -> String;

    /// 2026-09-26: A tool result, rendered as prompt text. The default wraps
    /// it in `<tool_response>` tags.
    fn format_tool_response(&self, content: &str) -> String {
        format!("<tool_response>\n{content}\n</tool_response>")
    }

    /// 2026-09-26: The tags the streaming content sanitizer suppresses for
    /// this format. The default, `LeakMarkers::EMPTY`, passes text through.
    fn leak_markers(&self) -> LeakMarkers {
        LeakMarkers::EMPTY
    }

    /// 2026-09-26: Compile the XGrammar grammar that constrains this
    /// format's tool calls, through the matching [`GrammarEngine`] entry
    /// point. The prefill steps call it through `compile_grammar_state`
    /// (`scheduler/emit_step/grammar_close.rs`).
    ///
    /// `tools` are the request's tool definitions. `use_triggers` is false
    /// when the request must call a tool (`tool_choice` "required" or a named
    /// function, and every `minimax_xml` request) and true otherwise
    /// (`api/chat/sampling_setup.rs` `tool_choice_required_for_parser`).
    ///
    /// Return value:
    /// - `None` (the default, kept by `MistralNativeParser` and
    ///   `DeepseekV4DsmlParser`): no grammar for this request.
    /// - `Some(Ok(g))`: the grammar the request decodes under.
    /// - `Some(Err(_))`: the failure is logged as a warning and the request
    ///   decodes without a grammar.
    fn compile_tool_grammar(
        &self,
        _engine: &mut GrammarEngine,
        _tools: &[ToolDefinition],
        _use_triggers: bool,
    ) -> Option<Result<CompiledGrammar, GrammarError>> {
        None
    }

    /// 2026-09-26: Whether [`Self::compile_tool_grammar`] returns `Some`; the
    /// two must agree. `ToolCallFormat::has_grammar` reads it for the startup
    /// log (`main_modules/serve_phases/runtime.rs`).
    fn has_tool_grammar(&self) -> bool {
        false
    }

    /// 2026-09-26: The literal that closes a free-text parameter value
    /// (qwen3_coder `</parameter>`, poolside_v1 `</arg_value>`). The parser's
    /// own `compile_tool_grammar` passes it to the grammar builder, which
    /// derives the value rule from it (`grammar/compile_tools.rs`
    /// `ebnf_until_close_ladder_opts`), so a value may contain `<`, `>` and
    /// `</X` text. `None`, the default, for formats without such a value.
    fn param_value_close_delim(&self) -> Option<&'static str> {
        None
    }

    fn broken_opener_stop_strings(&self) -> &'static [&'static str] {
        &[]
    }

    /// 2026-09-26: Whether `coerce_all` converts parsed argument strings to
    /// the types in the tool's schema. The blocking path
    /// (`api/chat_blocking_choice.rs`) and the streaming path
    /// (`api/chat_stream/mod.rs`) both read it.
    fn wants_typed_arguments(&self) -> bool {
        false
    }

    /// 2026-09-26: Whether this format writes a zero-argument call as a bare
    /// tool name inside the `<tool_call>` envelope; only `PoolsideV1Parser`
    /// says yes. The other formats have their own zero-argument form
    /// (`{"name":"f","arguments":{}}`, `<function=f></function>`), so for
    /// them a bare identifier is malformed output, and promoting it would
    /// make up a call the model never made.
    fn promotes_bare_call_names(&self) -> bool {
        false
    }
}

impl std::fmt::Display for dyn ToolCallParser {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// 2026-09-26: A tool-call format, named by `--tool-call-parser`, MODEL.toml
/// `[behavior].tool_call_parser` or `tool_defaults.toml`, in that order
/// (`main_modules/serve_phases/runtime.rs`).
#[derive(Debug, Clone, Copy)]
pub enum ToolCallFormat {
    Hermes,
    Qwen3Coder,
    Qwen3Xml,
    Gemma4,
    Mistral,
    MinimaxXml,
    DeepseekV4,
    BareJson,
    PoolsideV1,
}

impl std::str::FromStr for ToolCallFormat {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "hermes" => Ok(Self::Hermes),
            "qwen3_coder" => Ok(Self::Qwen3Coder),
            "qwen3_xml" => Ok(Self::Qwen3Xml),
            "gemma4" => Ok(Self::Gemma4),
            "mistral" => Ok(Self::Mistral),
            "minimax_xml" => Ok(Self::MinimaxXml),
            "deepseek_v4" | "deepseek_v41" | "dsml" => Ok(Self::DeepseekV4),
            "bare_json" => Ok(Self::BareJson),
            "poolside_v1" => Ok(Self::PoolsideV1),
            other => Err(format!(
                "Unknown tool call parser '{other}'. Supported: hermes, qwen3_coder, qwen3_xml, gemma4, mistral, minimax_xml, deepseek_v4, bare_json, poolside_v1",
            )),
        }
    }
}

impl ToolCallFormat {
    pub fn into_parser(self) -> Box<dyn ToolCallParser> {
        match self {
            Self::Hermes => Box::new(HermesParser),
            Self::Qwen3Coder => Box::new(Qwen3CoderParser),
            Self::Qwen3Xml => Box::new(Qwen3XmlParser),
            Self::Gemma4 => Box::new(Gemma4Parser),
            Self::Mistral => Box::new(MistralNativeParser),
            Self::MinimaxXml => Box::new(MinimaxXmlParser),
            Self::DeepseekV4 => Box::new(DeepseekV4DsmlParser),
            Self::BareJson => Box::new(BareJsonParser),
            Self::PoolsideV1 => Box::new(PoolsideV1Parser),
        }
    }

    /// 2026-09-26: Whether this format's parser has a tool grammar
    /// ([`ToolCallParser::has_tool_grammar`]).
    pub fn has_grammar(self) -> bool {
        self.into_parser().has_tool_grammar()
    }

    /// 2026-09-26: The canonical `--tool-call-parser` name. `from_str` also
    /// accepts `deepseek_v41` and `dsml` for `DeepseekV4`.
    pub fn name(self) -> &'static str {
        match self {
            Self::Hermes => "hermes",
            Self::Qwen3Coder => "qwen3_coder",
            Self::Qwen3Xml => "qwen3_xml",
            Self::Gemma4 => "gemma4",
            Self::Mistral => "mistral",
            Self::MinimaxXml => "minimax_xml",
            Self::DeepseekV4 => "deepseek_v4",
            Self::BareJson => "bare_json",
            Self::PoolsideV1 => "poolside_v1",
        }
    }
}

mod bare_json;
mod deepseek_v4_dsml;
mod fuzzy_match;
mod gemma4;
mod helpers_a;
mod helpers_b;
mod hermes;
mod minimax_xml;
mod mistral;
mod parse_dispatch;
mod parse_single_a;
mod parse_single_b;
mod parse_tools_tag;
mod pipeline;
mod pipeline_helpers;
mod poolside_v1;
mod prompt_levers;
mod qwen3_coder;
mod qwen3_xml;
mod streaming;
mod streaming_emit;
mod streaming_flush;
mod streaming_impl;
mod type_coerce;
pub(crate) mod validation;

pub use bare_json::*;
pub use deepseek_v4_dsml::*;
pub use gemma4::*;
use helpers_a::*;
pub(crate) use helpers_b::append_tool_choice_instruction;
use helpers_b::*;
pub use hermes::*;
pub use minimax_xml::*;
pub use mistral::*;
pub use parse_dispatch::*;
use parse_single_a::*;
use parse_single_b::*;
use parse_tools_tag::*;
pub use pipeline::*;
use pipeline_helpers::*;
pub use poolside_v1::*;
pub use qwen3_coder::*;
pub use qwen3_xml::*;
pub use streaming::*;
pub use type_coerce::coerce_all;
pub use validation::*;

#[cfg(test)]
mod tests;
