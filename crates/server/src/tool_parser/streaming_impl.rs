// SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(unused_imports, dead_code)]

//! 2026-09-26: `StreamingToolDetector` construction, reset and `process`, the
//! incremental loop that turns text deltas into content and tool-call events.
//!
//! Owner: server (tool parser).
//! Invariants:
//! - `call_counter` only increases; `reset` keeps it.

use super::*;

impl StreamingToolDetector {
    pub fn new() -> Self {
        Self::new_with_tools(Vec::new())
    }

    /// 2026-09-26: Build a detector that holds the request's tool schemas; live
    /// argument streaming uses them to name and coerce XML parameters. With
    /// `METRALE_BUFFER_TOOL_ARGS` set to `1` or `true`, arguments are not
    /// streamed: one `ToolCallDelta` carries them when the call closes.
    pub fn new_with_tools(tools: Vec<ToolDefinition>) -> Self {
        let buffer_args = matches!(
            std::env::var("METRALE_BUFFER_TOOL_ARGS").as_deref(),
            Ok("1") | Ok("true")
        );
        Self {
            buffer: String::new(),
            inside_tag: false,
            inside_dsml: false,
            promote_bare_names: false,
            call_counter: 0,
            emitted_tool_calls: false,
            current_tc_name: None,
            current_tc_id: None,
            current_tc_emitted: 0,
            tools,
            buffer_args,
            args_open: false,
            emitted_keys: Vec::new(),
            incremental_emitted: false,
        }
    }

    /// 2026-09-26: Treat a bare tool name inside a closed `<tool_call>` as a
    /// zero-argument call (Poolside v1's encoding). `reset` keeps the setting.
    pub fn set_promote_bare_names(&mut self, on: bool) {
        self.promote_bare_names = on;
    }

    /// 2026-09-26: Drop the buffer, envelope state and per-call state.
    /// `handle_token` calls it at `</think>` so tag fragments from the thinking
    /// text cannot start a call. The request-scoped settings (`tools`,
    /// `buffer_args`, `promote_bare_names`) and `call_counter` are kept.
    pub fn reset(&mut self) {
        self.buffer.clear();
        self.inside_tag = false;
        self.inside_dsml = false;
        self.reset_call_state();
    }

    /// 2026-09-26: Clear the per-call streaming state, after a call closes and
    /// on `reset`.
    pub(super) fn reset_call_state(&mut self) {
        self.current_tc_name = None;
        self.current_tc_id = None;
        self.current_tc_emitted = 0;
        self.args_open = false;
        self.emitted_keys.clear();
        self.incremental_emitted = false;
    }

    /// 2026-09-26: Feed a text delta and return the events it completes:
    /// content, `ToolCallStart` once a name is readable, argument fragments as
    /// parameters complete, and `ToolCallEnd` at the close tag.
    pub fn process(&mut self, new_text: &str) -> Vec<DetectorOutput> {
        let mut outputs = Vec::new();
        self.buffer.push_str(new_text);
        loop {
            match self.process_dsml(&mut outputs) {
                DsmlStreamAction::Continue => continue,
                DsmlStreamAction::Wait => break,
                DsmlStreamAction::NotDsml => {}
            }
            if self.inside_tag {
                // 2026-09-26: Close tags: `</tool_call>`, `<tool_call|>` (Gemma
                // 4), and MiniMax's `</minimax:tool_call>` or `</minimax:_call>`.
                let close_pos = self
                    .buffer
                    .find("</tool_call>")
                    .map(|p| (p, 12usize))
                    .or_else(|| self.buffer.find("<tool_call|>").map(|p| (p, 12usize)))
                    .or_else(|| {
                        self.buffer
                            .find("</minimax:tool_call>")
                            .map(|p| (p, "</minimax:tool_call>".len()))
                    })
                    .or_else(|| {
                        self.buffer
                            .find("</minimax:_call>")
                            .map(|p| (p, "</minimax:_call>".len()))
                    });
                if let Some((end, close_len)) = close_pos {
                    let idx = self.call_counter as usize;

                    if self.current_tc_name.is_some() {
                        // 2026-09-26: `ToolCallStart` already went out. If
                        // fragments were streamed, emit only the rest (unstreamed
                        // complete params, backfill and `}` for XML, or the JSON
                        // tail), before the buffer is cut: `stream_ready_fragments`
                        // reads `self.buffer[..end]`.
                        if !self.buffer_args && self.incremental_emitted {
                            let frags = self.stream_ready_fragments(end, true);
                            outputs.extend(frags);
                            outputs.push(DetectorOutput::ToolCallEnd { idx });
                            self.call_counter += 1;
                            self.emitted_tool_calls = true;
                            self.buffer = self.buffer[end + close_len..].to_string();
                            self.inside_tag = false;
                            self.reset_call_state();
                            continue;
                        }
                        let inner = self.buffer[..end].to_string();
                        self.buffer = self.buffer[end + close_len..].to_string();
                        self.inside_tag = false;
                        // 2026-09-26: Buffered mode, or nothing streamed yet: the
                        // whole arguments go out once in a `ToolCallDelta`.
                        if let Some(tc) =
                            parse_complete_call(&inner, self.call_counter, self.promote_bare_names)
                        {
                            // 2026-09-26: Emitted even when the arguments are
                            // `{}`: a tool may take none.
                            outputs.push(DetectorOutput::ToolCallDelta {
                                args: tc.function.arguments,
                                idx,
                            });
                            outputs.push(DetectorOutput::ToolCallEnd { idx });
                            self.call_counter += 1;
                            self.emitted_tool_calls = true;
                        } else {
                            tracing::warn!("Failed to parse tool call body, dropping");
                        }
                        self.reset_call_state();
                        continue;
                    } else {
                        let inner = self.buffer[..end].to_string();
                        self.buffer = self.buffer[end + close_len..].to_string();
                        self.inside_tag = false;
                        // 2026-09-26: No `ToolCallStart` went out, so whole
                        // `ToolCall`s are emitted. A MiniMax envelope can hold
                        // several `<invoke>` blocks and `parse_one_call` returns
                        // one call, so each block is parsed here.
                        let trimmed = inner.trim();
                        if trimmed.contains("<invoke name=") {
                            for tc in parse_minimax_xml_calls_all(trimmed) {
                                let call_idx = self.call_counter as usize;
                                self.call_counter += 1;
                                self.emitted_tool_calls = true;
                                outputs.push(DetectorOutput::ToolCall(tc, call_idx));
                            }
                        } else if let Some(tc) =
                            parse_complete_call(trimmed, self.call_counter, self.promote_bare_names)
                        {
                            self.call_counter += 1;
                            self.emitted_tool_calls = true;
                            outputs.push(DetectorOutput::ToolCall(tc, idx));
                        }
                    }
                    self.reset_call_state();
                    continue;
                }

                // 2026-09-26: No close tag yet. `ToolCallStart` goes out as soon
                // as the name can be read.
                if self.current_tc_name.is_none()
                    && let Some(name) = extract_streaming_name(&self.buffer)
                {
                    let id = next_tool_call_id();
                    let idx = self.call_counter as usize;
                    outputs.push(DetectorOutput::ToolCallStart {
                        id: id.clone(),
                        name: name.clone(),
                        idx,
                    });
                    self.current_tc_name = Some(name);
                    self.current_tc_id = Some(id);
                }
                // 2026-09-26: Unless `buffer_args` is set, stream every argument
                // fragment completed so far.
                if !self.buffer_args && self.current_tc_name.is_some() {
                    let frags = self.stream_ready_fragments(self.buffer.len(), false);
                    outputs.extend(frags);
                }
                break;
            } else if let Some(mistral_start) = self.buffer.find(MISTRAL_TOOL_CALLS_TAG) {
                // 2026-09-26: Mistral `[TOOL_CALLS]name[ARGS]{json}` has no close
                // tag. Text before the tag goes out as content; the call is
                // parsed once `[ARGS]` and a balanced JSON object are buffered.
                if mistral_start > 0 {
                    let before = self.buffer[..mistral_start].to_string();
                    outputs.push(DetectorOutput::Content(before));
                    self.buffer = self.buffer[mistral_start..].to_string();
                }
                let after_tag = &self.buffer[MISTRAL_TOOL_CALLS_TAG.len()..];
                let args_rel = match after_tag.find(MISTRAL_ARGS_TAG) {
                    Some(p) => p,
                    None => break,
                };
                let name = after_tag[..args_rel].trim().to_string();
                let json_abs_start =
                    MISTRAL_TOOL_CALLS_TAG.len() + args_rel + MISTRAL_ARGS_TAG.len();
                let mut json_rel = json_abs_start;
                while json_rel < self.buffer.len()
                    && self.buffer.as_bytes()[json_rel].is_ascii_whitespace()
                {
                    json_rel += 1;
                }
                if json_rel >= self.buffer.len() || self.buffer.as_bytes()[json_rel] != b'{' {
                    break;
                }
                let json_tail = &self.buffer[json_rel..];
                let Some(json_end_rel) = find_balanced_json_end(json_tail) else {
                    break;
                };
                let id = next_tool_call_id();
                let idx = self.call_counter as usize;
                if !name.is_empty() {
                    outputs.push(DetectorOutput::ToolCallStart {
                        id: id.clone(),
                        name: name.clone(),
                        idx,
                    });
                }
                let raw_args = &json_tail[..json_end_rel];
                let canonical = serde_json::from_str::<serde_json::Value>(raw_args)
                    .ok()
                    .and_then(|v| serde_json::to_string(&v).ok())
                    .unwrap_or_else(|| "{}".to_string());
                let args_empty = canonical == "{}" || canonical.is_empty();
                if !name.is_empty() && !args_empty {
                    outputs.push(DetectorOutput::ToolCallDelta {
                        args: canonical,
                        idx,
                    });
                    outputs.push(DetectorOutput::ToolCallEnd { idx });
                    self.call_counter += 1;
                    self.emitted_tool_calls = true;
                } else if !name.is_empty() {
                    tracing::warn!("Dropping empty Mistral tool call '{name}' — args were empty");
                }
                let consumed = json_rel + json_end_rel;
                self.buffer = self.buffer[consumed..].to_string();
                continue;
            } else if let Some((start, tag_len)) = self
                .buffer
                .find("<tool_call>")
                .map(|p| (p, 11usize))
                .or_else(|| self.buffer.find("<|tool_call>").map(|p| (p, 12usize)))
                .or_else(|| {
                    self.buffer
                        .find("<minimax:tool_call>")
                        .map(|p| (p, "<minimax:tool_call>".len()))
                })
                .or_else(|| {
                    self.buffer
                        .find("<minimax:_call>")
                        .map(|p| (p, "<minimax:_call>".len()))
                })
            {
                let before = self.buffer[..start].to_string();
                self.buffer = self.buffer[start + tag_len..].to_string();
                self.inside_tag = true;
                if !before.is_empty() {
                    outputs.push(DetectorOutput::Content(before));
                }
                continue;
            } else if self.buffer.contains("<function") {
                if self.process_bare_function(&mut outputs) {
                    continue;
                }
                break;
            } else {
                if self.buffer.trim().is_empty() {
                    break;
                }
                let safe = self.safe_emit_len();
                if safe > 0 {
                    let content = self.buffer[..safe].to_string();
                    let remainder = self.buffer[safe..].to_string();
                    let dsml_leading_whitespace = !remainder.is_empty()
                        && content.trim().is_empty()
                        && DSML_OPEN.starts_with(&remainder);
                    self.buffer = remainder;
                    if !content.is_empty() && !dsml_leading_whitespace {
                        outputs.push(DetectorOutput::Content(content));
                    }
                }
                break;
            }
        }
        outputs
    }
}
