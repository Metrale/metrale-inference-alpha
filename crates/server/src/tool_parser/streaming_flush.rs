// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: `StreamingToolDetector` end-of-stream parsing (`flush`) and its
//! buffer queries (`has_tool_calls`, `inside_tool_call`, `safe_emit_len`).
//!
//! Owner: server (tool parser).
//! Invariants:
//! - `flush` returns with the buffer empty.

use super::*;

impl StreamingToolDetector {
    /// 2026-09-26: Drain the buffer at stream end. An open DSML block, or an
    /// unclosed `<tool_call>` body, is parsed as a call when it parses.
    /// Otherwise, if this stream has produced no call yet, bare `<function>`
    /// calls and then unwrapped JSON calls are tried; the rest is content.
    pub fn flush(&mut self) -> Vec<DetectorOutput> {
        if let Some(outputs) = self.flush_dsml() {
            return outputs;
        }
        if self.buffer.is_empty() {
            return vec![];
        }
        let text = std::mem::take(&mut self.buffer);
        let was_inside_tag = self.inside_tag;
        self.inside_tag = false;

        // 2026-09-26: `was_inside_tag` means the stream ended inside a
        // `<tool_call>` body with no close tag. `contain_unterminated_call_tail`
        // cuts the body after its last complete `</parameter>` (or before the
        // first `<parameter=` if none closed), as `parse_tool_calls` does, so an
        // unclosed value cannot run to the end of the text. When `ToolCallStart`
        // already went out (`current_tc_name` is set), the call is finished
        // with `ToolCallDelta` + `ToolCallEnd` under that header: a whole
        // `ToolCall` carries a new id from `parse_one_call`, and
        // `handle_complete_tool_call` would send a second start chunk with it.
        if was_inside_tag
            && !text.contains("<arg_key>")
            && let Some(tc) = parse_one_call(
                contain_unterminated_call_tail(text.trim()),
                self.call_counter,
            )
        {
            let idx = self.call_counter as usize;
            if self.current_tc_name.is_some() {
                // 2026-09-26: Fragments already streamed: emit only the rest
                // (unstreamed complete params, backfill and the closing `}`, or
                // the JSON tail). `stream_ready_fragments` scans `self.buffer`,
                // so the body is put back for it, and it takes the fragment
                // index from `call_counter`, so the counter moves after it.
                if !self.buffer_args && self.incremental_emitted {
                    self.buffer = text;
                    let limit = self.buffer.len();
                    let mut out = self.stream_ready_fragments(limit, true);
                    out.push(DetectorOutput::ToolCallEnd { idx });
                    self.call_counter += 1;
                    self.emitted_tool_calls = true;
                    self.buffer.clear();
                    self.reset_call_state();
                    return out;
                }
                self.call_counter += 1;
                self.emitted_tool_calls = true;
                self.reset_call_state();
                return vec![
                    DetectorOutput::ToolCallDelta {
                        args: tc.function.arguments,
                        idx,
                    },
                    DetectorOutput::ToolCallEnd { idx },
                ];
            }
            self.call_counter += 1;
            self.emitted_tool_calls = true;
            return vec![DetectorOutput::ToolCall(tc, idx)];
        }

        let text = if was_inside_tag {
            format!("<tool_call>{text}")
        } else {
            text
        };

        if !self.has_tool_calls() && !self.emitted_tool_calls {
            let (content, calls) = parse_bare_function_calls(&text);
            if !calls.is_empty() {
                let mut out = Vec::new();
                if let Some(c) = content {
                    out.push(DetectorOutput::Content(c));
                }
                for tc in calls {
                    let idx = self.call_counter as usize;
                    self.call_counter += 1;
                    out.push(DetectorOutput::ToolCall(tc, idx));
                }
                return out;
            }
        }

        // 2026-09-26: JSON calls with no envelope: in a fenced code block, on a
        // line of their own, or as a `{"name"` object inside prose
        // (`parse_json_fallback_calls`).
        if !self.has_tool_calls() && !self.emitted_tool_calls {
            let json_calls = parse_json_fallback_calls(&text);
            if !json_calls.is_empty() {
                let mut out = Vec::new();
                // 2026-09-26: The content is the text minus every fenced code
                // block; JSON outside a fence stays in it.
                let mut clean = text.clone();
                for pattern in extract_json_code_blocks(&text) {
                    clean = clean.replace(&pattern, "");
                }
                let clean = clean.trim().to_string();
                if !clean.is_empty() {
                    out.push(DetectorOutput::Content(clean));
                }
                for tc in json_calls {
                    let idx = self.call_counter as usize;
                    self.call_counter += 1;
                    out.push(DetectorOutput::ToolCall(tc, idx));
                }
                return out;
            }
        }

        vec![DetectorOutput::Content(text)]
    }

    pub fn has_tool_calls(&self) -> bool {
        self.call_counter > 0
    }

    /// 2026-09-26: True while the detector holds tool-call text: inside a
    /// tool-call or DSML envelope, or with a buffered prefix of an opener.
    /// `handle_token` passes it to `strip_bare_role_literal`, which then keeps
    /// a bare `tool` delta: inside a call it can be the first piece of a name
    /// such as `tool_search`.
    pub fn inside_tool_call(&self) -> bool {
        self.inside_tag || self.inside_dsml || self.has_partial_tool_opener()
    }

    /// 2026-09-26: Length of the buffer prefix that can go out as content. It
    /// stops before a trailing partial of any opener listed below, so an
    /// opener split across deltas is still detected.
    pub(super) fn safe_emit_len(&self) -> usize {
        let buf = self.buffer.as_bytes();
        // 2026-09-26: No close tags: `process` matches them only while
        // `inside_tag`, and in that state it buffers the whole body and does
        // not call this.
        for tag in [
            b"<tool_call>" as &[u8],
            b"<|tool_call>",
            b"<minimax:tool_call>",
            b"<minimax:_call>",
            DSML_OPEN.as_bytes(),
            b"<function",
            b"call:",
            MISTRAL_TOOL_CALLS_TAG.as_bytes(),
        ] {
            for i in (buf.len().saturating_sub(tag.len() - 1))..buf.len() {
                if tag.starts_with(&buf[i..]) {
                    return i;
                }
            }
        }
        buf.len()
    }
}
