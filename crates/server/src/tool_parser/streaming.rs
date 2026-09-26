// SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(unused_imports, dead_code)]

//! 2026-09-26: The streaming tool-call detector's state and output types,
//! and the name and argument locators its `process` uses
//! (streaming_impl.rs).
//!
//! Owner: server (tool parsing).
//! Invariants: none beyond the types.

use super::*;

/// 2026-09-26: Buffers streamed text and splits it into content and tool
/// calls as tokens arrive (`process` in streaming_impl.rs, `flush` in
/// streaming_flush.rs). For a call it emits `ToolCallStart` once the name is
/// known, the arguments as `ToolCallArgsFragment`s (or one `ToolCallDelta`),
/// and `ToolCallEnd` at the close; a call parsed whole is one `ToolCall`.
pub struct StreamingToolDetector {
    pub(super) buffer: String,
    pub(super) inside_tag: bool,
    pub(super) inside_dsml: bool,
    /// 2026-09-26: Whether a bare identifier inside `<tool_call>` is a
    /// zero-argument call; set from `ToolCallParser::promotes_bare_call_names`
    /// (`set_promote_bare_names`). False otherwise, as in the blocking parser.
    pub(super) promote_bare_names: bool,
    pub(super) call_counter: u32,
    /// 2026-09-26: Set once any call is emitted; `flush` then skips its
    /// bare-function and JSON fallbacks.
    pub(super) emitted_tool_calls: bool,
    /// 2026-09-26: Name of the open call, once `ToolCallStart` is emitted.
    pub(super) current_tc_name: Option<String>,
    /// 2026-09-26: Id of the open call, sent in its `ToolCallStart`.
    pub(super) current_tc_id: Option<String>,
    /// 2026-09-26: How far the open call's arguments are streamed. XML: a
    /// byte offset into `buffer` past the last parameter streamed. JSON: the
    /// argument-object bytes streamed. Gemma-4: the bytes of the converted
    /// JSON body streamed.
    pub(super) current_tc_emitted: usize,
    /// 2026-09-26: The request's tool schemas, for per-parameter coercion
    /// while arguments stream. Empty when built with `new()`: values then
    /// stream as JSON strings.
    pub(super) tools: Vec<ToolDefinition>,
    /// 2026-09-26: When true, arguments are not streamed as fragments; they
    /// go out whole at the close. True when `METRALE_BUFFER_TOOL_ARGS` is `1`
    /// or `true` (`new_with_tools`).
    pub(super) buffer_args: bool,
    /// 2026-09-26: Whether the open call's argument `{` has been emitted.
    pub(super) args_open: bool,
    /// 2026-09-26: Keys already streamed for the open XML call, so the
    /// close-time backfill does not repeat them.
    pub(super) emitted_keys: Vec<String>,
    /// 2026-09-26: True once a `ToolCallArgsFragment` was emitted for the
    /// open call; the close then emits only the rest (remaining parameters,
    /// backfill, closing `}` or JSON tail) instead of the full arguments.
    pub(super) incremental_emitted: bool,
}

pub enum DetectorOutput {
    /// 2026-09-26: Text that is not emitted as a tool call.
    Content(String),
    /// 2026-09-26: A call parsed whole, with its index.
    ToolCall(ToolCall, usize),
    /// 2026-09-26: A call's header (id and name), emitted once the name is
    /// known.
    ToolCallStart {
        id: String,
        name: String,
        idx: usize,
    },
    /// 2026-09-26: A call's full arguments, emitted once at its close when
    /// no fragment was streamed. The handler backfills them, coerces them when
    /// the parser asks for it, and checks them
    /// (`api/chat_stream/tool_handlers.rs` `handle_tool_call_delta`).
    ToolCallDelta { args: String, idx: usize },
    /// 2026-09-26: A slice of `function.arguments`, already coerced (XML) or
    /// cut from the model's JSON. The handler forwards it verbatim
    /// (`api/chat_stream/tool_handlers.rs` `handle_tool_call_args_fragment`).
    ToolCallArgsFragment { fragment: String, idx: usize },
    /// 2026-09-26: The call with this index is complete.
    ToolCallEnd { idx: usize },
}

impl StreamingToolDetector {
    pub(super) fn has_partial_tool_opener(&self) -> bool {
        !self.buffer.is_empty()
            && [
                "<tool_call>",
                "<|tool_call>",
                "<minimax:tool_call>",
                "<minimax:_call>",
                DSML_OPEN,
            ]
            .iter()
            .any(|tag| tag.starts_with(&self.buffer))
    }
}

/// 2026-09-26: The function name in a partial call, for an early
/// `ToolCallStart`. Tries Mistral `[TOOL_CALLS]NAME[ARGS]`, Gemma-4
/// `call:NAME{`, qwen3_coder `<function=NAME>`, then Hermes `"name":"NAME"`.
pub(super) fn extract_streaming_name(buffer: &str) -> Option<String> {
    if let Some(start) = buffer.find(MISTRAL_TOOL_CALLS_TAG) {
        let after = &buffer[start + MISTRAL_TOOL_CALLS_TAG.len()..];
        if let Some(end) = after.find(MISTRAL_ARGS_TAG) {
            let name = after[..end].trim();
            if !name.is_empty() {
                return Some(name.to_string());
            }
        }
    }
    if let Some(start) = buffer.find("call:") {
        let after = &buffer[start + 5..];
        if let Some(end) = after.find('{') {
            let name = after[..end].trim();
            if !name.is_empty() {
                return Some(name.to_string());
            }
        }
    }
    if let Some(start) = buffer.find("<function=") {
        let after = &buffer[start + "<function=".len()..];
        if let Some(end) = after.find(['>', '\n', '<']) {
            let mut name = after[..end].trim().to_string();
            // 2026-09-26: A name is cut at its first `=` (`Bash=Bash` gives
            // `Bash`).
            if let Some(eq_pos) = name.find('=') {
                name = name[..eq_pos].trim().to_string();
            }
            if !name.is_empty() {
                return Some(name);
            }
        }
    }
    if let Some(start) = buffer.find("\"name\"") {
        let after = &buffer[start + "\"name\"".len()..];
        let after = after
            .trim_start()
            .strip_prefix(':')
            .unwrap_or(after)
            .trim_start();
        if let Some(after) = after.strip_prefix('"')
            && let Some(end) = after.find('"')
        {
            let name = &after[..end];
            if !name.is_empty() {
                return Some(name.to_string());
            }
        }
    }
    None
}

/// 2026-09-26: Byte offset in `buffer` where a call's arguments start: after
/// Gemma-4 `call:NAME{`, after qwen3_coder `<function=NAME>` and a newline
/// directly after it, or after Hermes `"arguments":`. `buffer.len()` when
/// none is found.
pub(super) fn find_args_start(buffer: &str) -> usize {
    if let Some(pos) = buffer.find("call:")
        && let Some(brace) = buffer[pos..].find('{')
    {
        return pos + brace + 1;
    }
    if let Some(pos) = buffer.find("<function=")
        && let Some(gt) = buffer[pos..].find('>')
    {
        let after_gt = pos + gt + 1;
        if after_gt < buffer.len() && buffer.as_bytes().get(after_gt) == Some(&b'\n') {
            return after_gt + 1;
        }
        return after_gt;
    }
    if let Some(pos) = buffer.find("\"arguments\"") {
        let after = &buffer[pos + "\"arguments\"".len()..];
        let after = after.trim_start();
        if let Some(rest) = after.strip_prefix(':') {
            return buffer.len() - rest.len();
        }
    }
    buffer.len()
}

impl Default for StreamingToolDetector {
    fn default() -> Self {
        Self::new()
    }
}
