// SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(unused_imports, dead_code)]

//! 2026-09-26: Argument-fragment streaming for the tool-call detector:
//! newly completed `<parameter>` values, Gemma-4 arguments and JSON argument
//! bytes become `ToolCallArgsFragment`s; also bare `<function…>` blocks.
//! `process` (streaming_impl.rs) and `flush` (streaming_flush.rs) call these.
//!
//! Owner: server (tool parsing).
//! Invariants:
//! - `stream_ready_fragments` never moves `current_tc_emitted` backwards.

use super::*;

impl StreamingToolDetector {
    /// 2026-09-26: Normalize one `<parameter=KEY>VALUE</parameter>` pair for a
    /// fragment: the key through `salvage_echoed_param` and
    /// `normalize_param_name`, the value through `coerce_all` against the
    /// call's schema. Returns `(key, JSON value)`; when coercion yields
    /// nothing, the value is the raw string, JSON-quoted.
    pub(super) fn coerce_kv(&self, raw_key: &str, raw_value: &str) -> (String, String) {
        let name = self.current_tc_name.clone().unwrap_or_default();
        // 2026-09-26: Re-split `parameter=filePath>…`, where the real key
        // leaked into the value, with the helper `backfill_required_params`
        // also uses (validation.rs). A recovered key that was already
        // streamed is not used, so no key appears twice.
        let salvaged =
            crate::tool_parser::salvage_echoed_param(&self.tools, &name, raw_key, raw_value)
                .filter(|(real_key, _)| !self.emitted_keys.contains(real_key));
        let (raw_key, raw_value) = match &salvaged {
            Some((real_key, real_val)) => (real_key.as_str(), real_val.as_str()),
            None => (raw_key, raw_value),
        };
        let norm_key = crate::tool_parser::normalize_param_name(&self.tools, &name, raw_key);
        let fallback = || serde_json::to_string(raw_value).unwrap_or_else(|_| "\"\"".to_string());
        let single = serde_json::to_string(&serde_json::json!({ norm_key.clone(): raw_value }));
        let Ok(single_args) = single else {
            return (norm_key, fallback());
        };
        let mut tc = ToolCall {
            id: String::new(),
            call_type: "function".into(),
            function: FunctionCall {
                name,
                arguments: single_args,
            },
        };
        coerce_all(std::slice::from_mut(&mut tc), &self.tools);
        let json_value_string = serde_json::from_str::<serde_json::Value>(&tc.function.arguments)
            .ok()
            .and_then(|v| v.get(&norm_key).cloned())
            .and_then(|v| serde_json::to_string(&v).ok())
            .unwrap_or_else(fallback);
        (norm_key, json_value_string)
    }

    /// 2026-09-26: Emit the argument fragments of the open call that are newly
    /// complete in `self.buffer[..limit]`, advancing `current_tc_emitted`
    /// past them. Every caller skips it when `buffer_args` is set.
    ///
    /// - XML (`<parameter=K>V</parameter>`): each complete parameter becomes a
    ///   coerced `"key":value` fragment with a leading `{` or `,`. On
    ///   `final_close`, missing required string parameters are added as `""`
    ///   and the closing `}` is emitted.
    /// - Gemma-4 (`call:NAME{`): the body up to the first `}`, converted by
    ///   `gemma4_to_json`; `final_close` adds the closing `}`.
    /// - JSON (`"arguments": {...}`): the model's bytes, forwarded without
    ///   coercion, up to the object's balanced end.
    pub(super) fn stream_ready_fragments(
        &mut self,
        limit: usize,
        final_close: bool,
    ) -> Vec<DetectorOutput> {
        let mut outputs = Vec::new();
        let idx = self.call_counter as usize;
        let scan = &self.buffer[..limit.min(self.buffer.len())];

        if scan.contains("<parameter=") {
            loop {
                let from = self.current_tc_emitted;
                let Some(rel_open) = self.buffer[from..limit].find("<parameter=") else {
                    break;
                };
                let open_at = from + rel_open;
                let key_region = open_at + "<parameter=".len();
                let Some(rel_gt) = self.buffer[key_region..limit].find('>') else {
                    break;
                };
                let gt_at = key_region + rel_gt;
                let value_region = gt_at + 1;
                // 2026-09-26: A close missing its `>` right before the next
                // opener (`</parameter<parameter=KEY>`) also ends the value;
                // scanning resumes at that `<parameter=`.
                let exact = self.buffer[value_region..limit].find("</parameter>");
                let garbled = self.buffer[value_region..limit].find("</parameter<parameter=");
                let (rel_close, advance) = match (exact, garbled) {
                    (Some(e), Some(g)) if g < e => (g, "</parameter".len()),
                    (Some(e), _) => (e, "</parameter>".len()),
                    (None, Some(g)) => (g, "</parameter".len()),
                    (None, None) => break,
                };
                let close_at = value_region + rel_close;
                // 2026-09-26: Key and value are trimmed, as in
                // `parse_qwen3_coder_call`.
                let key = self.buffer[key_region..gt_at].trim().to_string();
                let raw_value = self.buffer[value_region..close_at].trim();
                let (norm_key, json_value_string) = self.coerce_kv(&key, raw_value);
                let prefix = if !self.args_open {
                    self.args_open = true;
                    "{"
                } else {
                    ","
                };
                let quoted_key =
                    serde_json::to_string(&norm_key).unwrap_or_else(|_| "\"\"".to_string());
                let fragment = format!("{prefix}{quoted_key}:{json_value_string}");
                outputs.push(DetectorOutput::ToolCallArgsFragment { fragment, idx });
                self.incremental_emitted = true;
                self.emitted_keys.push(norm_key);
                self.current_tc_emitted = close_at + advance;
            }

            if final_close {
                // 2026-09-26: Required parameters the model never wrote are
                // added as `""` when their schema type is `string` or absent,
                // as `backfill_required_params` does (`validation/backfill.rs`).
                if let Some(name) = self.current_tc_name.clone()
                    && let Some(tool_def) = self.tools.iter().find(|t| t.function.name == name)
                    && let Some(params) = tool_def.function.parameters.as_ref()
                {
                    let properties = params.get("properties").and_then(|p| p.as_object());
                    let required: Vec<String> = params
                        .get("required")
                        .and_then(|r| r.as_array())
                        .map(|a| {
                            a.iter()
                                .filter_map(|v| v.as_str().map(String::from))
                                .collect()
                        })
                        .unwrap_or_default();
                    for req in &required {
                        if self.emitted_keys.iter().any(|k| k == req) {
                            continue;
                        }
                        let is_string = properties
                            .and_then(|p| p.get(req))
                            .and_then(|v| v.get("type"))
                            .and_then(|t| t.as_str())
                            .is_none_or(|t| t == "string");
                        if !is_string {
                            continue;
                        }
                        let prefix = if !self.args_open {
                            self.args_open = true;
                            "{"
                        } else {
                            ","
                        };
                        let quoted_key =
                            serde_json::to_string(req).unwrap_or_else(|_| "\"\"".to_string());
                        outputs.push(DetectorOutput::ToolCallArgsFragment {
                            fragment: format!("{prefix}{quoted_key}:\"\""),
                            idx,
                        });
                        self.incremental_emitted = true;
                        self.emitted_keys.push(req.clone());
                    }
                }
                let closing = if !self.args_open {
                    self.args_open = true;
                    "{}".to_string()
                } else {
                    "}".to_string()
                };
                outputs.push(DetectorOutput::ToolCallArgsFragment {
                    fragment: closing,
                    idx,
                });
                self.incremental_emitted = true;
            }
        } else if scan.contains("call:") {
            let args_start = super::streaming::find_args_start(&self.buffer);
            if args_start >= limit {
                return outputs;
            }
            let body = &self.buffer[args_start..limit];
            let settled = if let Some(e) = body.find('}') {
                e
            } else {
                let mut s = body.len();
                while s > 0 && !body.is_char_boundary(s) {
                    s -= 1;
                }
                s
            };

            let accumulated_raw = &body[..settled];
            let converted_json = gemma4_to_json(&format!("gemma4{{{}}}", accumulated_raw));
            if converted_json.len() > 2 {
                let json_args_body = &converted_json[1..converted_json.len() - 1];
                let already = self.current_tc_emitted;
                if json_args_body.len() > already {
                    let new_frag = &json_args_body[already..];
                    let prefix = if !self.args_open {
                        self.args_open = true;
                        "{"
                    } else {
                        ""
                    };
                    outputs.push(DetectorOutput::ToolCallArgsFragment {
                        fragment: format!("{}{}", prefix, new_frag),
                        idx,
                    });
                    self.current_tc_emitted = json_args_body.len();
                    self.incremental_emitted = true;
                }
            }

            if final_close {
                let closing = if !self.args_open {
                    self.args_open = true;
                    "{}".to_string()
                } else {
                    "}".to_string()
                };
                outputs.push(DetectorOutput::ToolCallArgsFragment {
                    fragment: closing,
                    idx,
                });
                self.incremental_emitted = true;
            }
        } else if scan.contains("\"arguments\"") {
            let args_start = super::streaming::find_args_start(&self.buffer);
            if args_start >= limit {
                return outputs;
            }
            // 2026-09-26: Start at the object's `{`: `find_balanced_json_end`
            // needs a leading `{`, so whitespace after the colon is skipped.
            let mut args_start = args_start;
            while args_start < limit && self.buffer.as_bytes()[args_start].is_ascii_whitespace() {
                args_start += 1;
            }
            let body = &self.buffer[args_start..limit];
            let settled = if let Some(e) = find_balanced_json_end(body) {
                e
            } else {
                // 2026-09-26: No balanced close yet: stream up to a char
                // boundary.
                let mut s = body.len();
                while s > 0 && !body.is_char_boundary(s) {
                    s -= 1;
                }
                s
            };
            let already = self.current_tc_emitted.min(settled);
            let new = &body[already..settled];
            if !new.is_empty() {
                outputs.push(DetectorOutput::ToolCallArgsFragment {
                    fragment: new.to_string(),
                    idx,
                });
                self.current_tc_emitted = settled;
                self.incremental_emitted = true;
            }
        }
        outputs
    }
}

impl StreamingToolDetector {
    /// 2026-09-26: Handle a bare `<function…>` block (no `<tool_call>`
    /// wrapper). While the block is open, its header and each completed
    /// parameter stream as for a `<tool_call>` block.
    ///
    /// Returns `true` when the caller's scan loop should `continue` (a
    /// complete block was consumed), `false` when it should `break` and wait
    /// for more tokens.
    pub(super) fn process_bare_function(&mut self, outputs: &mut Vec<DetectorOutput>) -> bool {
        let Some(func_pos) = self.buffer.find("<function") else {
            return false;
        };
        if func_pos > 0 {
            let before = self.buffer[..func_pos].to_string();
            self.buffer = self.buffer[func_pos..].to_string();
            outputs.push(DetectorOutput::Content(before));
        }

        if let Some(end) = bare_function_end(&self.buffer) {
            // 2026-09-26: A call whose fragments already streamed (below) is
            // closed as the `<tool_call>` arm closes one: remaining fragments
            // and `ToolCallEnd`. A whole `ToolCall` here would deliver it
            // twice, so the two branches must stay exclusive.
            if !self.buffer_args && self.incremental_emitted {
                let idx = self.call_counter as usize;
                let frags = self.stream_ready_fragments(end, true);
                outputs.extend(frags);
                outputs.push(DetectorOutput::ToolCallEnd { idx });
                self.call_counter += 1;
                self.emitted_tool_calls = true;
                self.buffer = self.buffer[end..].to_string();
                self.reset_call_state();
                return true;
            }
            let block = self.buffer[..end].to_string();
            self.buffer = self.buffer[end..].to_string();
            let (_, calls) = parse_bare_function_calls(&block);
            for tc in calls {
                let idx = self.call_counter as usize;
                self.call_counter += 1;
                self.emitted_tool_calls = true;
                outputs.push(DetectorOutput::ToolCall(tc, idx));
            }
            self.reset_call_state();
            return true;
        }

        // 2026-09-26: Block not closed yet: emit the header once the name is
        // known, then any newly completed parameters. `stream_ready_fragments`
        // looks only for `<parameter=`, so the `<function=NAME>` still in the
        // buffer is skipped.
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
        if !self.buffer_args && self.current_tc_name.is_some() {
            let frags = self.stream_ready_fragments(self.buffer.len(), false);
            outputs.extend(frags);
        }
        false
    }
}
