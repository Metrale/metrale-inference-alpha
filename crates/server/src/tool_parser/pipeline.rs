// SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(unused_imports, dead_code)]

//! 2026-09-26: The pass pipeline for tool calls written without a
//! `<tool_call>` wrapper. Primary passes match the call shapes themselves;
//! salvage passes recover malformed output such as `<parameter=X>` written
//! for `<function=X>`. The primary passes run in order until one finds a
//! call; only if none does, the salvage passes run the same way.
//!
//! Owner: server (tool parsing).
//! Invariants:
//! - Within `ToolCallPipeline::run`, a pass clears `remaining` exactly when
//!   it pushes a call.

use super::*;

/// 2026-09-26: When a pass runs: `Primary` passes match call shapes;
/// `Salvage` passes recover malformed output and run only if no primary pass
/// found a call.
pub enum PassKind {
    Primary,
    Salvage,
}

/// 2026-09-26: State shared by the passes of one run.
///
/// A pass that finds a call pushes it, and any content before it, then
/// clears `remaining`; a pass that finds none leaves `remaining` unchanged,
/// so the next pass sees the same text.
pub struct PassState<'a> {
    pub remaining: &'a mut String,
    pub calls: &'a mut Vec<ToolCall>,
    pub content: &'a mut Vec<String>,
    pub call_counter: &'a mut u32,
}

/// 2026-09-26: A recognizer for one call shape (e.g. tag-style
/// `<function>NAME</function>`).
pub trait ToolCallPass: Send + Sync {
    /// 2026-09-26: Name for logs, in `lowercase_snake`.
    fn name(&self) -> &str;
    /// 2026-09-26: Look for this pass's shape in `state.remaining`.
    fn apply(&self, state: &mut PassState<'_>);
}

/// 2026-09-26: The passes, split by `PassKind`, in registration order.
pub struct ToolCallPipeline {
    primary: Vec<Box<dyn ToolCallPass>>,
    salvage: Vec<Box<dyn ToolCallPass>>,
}

impl Default for ToolCallPipeline {
    fn default() -> Self {
        Self::bare_function_default()
    }
}

impl ToolCallPipeline {
    pub fn new() -> Self {
        Self {
            primary: Vec::new(),
            salvage: Vec::new(),
        }
    }

    pub fn register(mut self, pass: Box<dyn ToolCallPass>, kind: PassKind) -> Self {
        match kind {
            PassKind::Primary => self.primary.push(pass),
            PassKind::Salvage => self.salvage.push(pass),
        }
        self
    }

    /// 2026-09-26: The pipeline `parse_bare_function_calls` runs. Primary:
    /// tag-style `<function>NAME</function>`, attribute-style
    /// `<function=NAME>`, and `NAME{json}`. Salvage: `<parameter=NAME>`
    /// written for `<function=NAME>`.
    pub fn bare_function_default() -> Self {
        Self::new()
            .register(Box::new(BareFunctionTagPass), PassKind::Primary)
            .register(Box::new(BareFunctionAttrPass), PassKind::Primary)
            .register(Box::new(BareMistralNamePass), PassKind::Primary)
            .register(Box::new(ParamAsFunctionSalvagePass), PassKind::Salvage)
    }

    /// 2026-09-26: Run the pipeline. Returns `(content, calls)`: content is
    /// text before a call opener; text between and after calls is not
    /// returned.
    pub fn run(&self, text: &str) -> (Option<String>, Vec<ToolCall>) {
        let mut remaining = text.to_string();
        let mut calls = Vec::new();
        let mut content = Vec::new();
        let mut call_counter = 0u32;
        let mut state = PassState {
            remaining: &mut remaining,
            calls: &mut calls,
            content: &mut content,
            call_counter: &mut call_counter,
        };

        for pass in &self.primary {
            pass.apply(&mut state);
            if !state.calls.is_empty() {
                break;
            }
        }

        if state.calls.is_empty() {
            for pass in &self.salvage {
                pass.apply(&mut state);
                if !state.calls.is_empty() {
                    tracing::warn!(
                        pass = pass.name(),
                        "tool-call salvage fired — model emitted malformed output"
                    );
                    break;
                }
            }
        }

        let combined = if content.is_empty() {
            None
        } else {
            Some(content.join("\n"))
        };
        (combined, calls)
    }
}

/// 2026-09-26: Tag-style `<function>NAME</function>` calls, each with an
/// optional `<parameters>…</parameters>` block.
pub struct BareFunctionTagPass;
impl ToolCallPass for BareFunctionTagPass {
    fn name(&self) -> &str {
        "bare_function_tag"
    }
    fn apply(&self, state: &mut PassState<'_>) {
        let text = state.remaining.clone();
        let mut cur = text.as_str();
        let mut first = true;
        while let Some(start) = cur.find("<function>") {
            if first {
                let before = cur[..start].trim();
                if !before.is_empty() {
                    state.content.push(before.to_string());
                }
                first = false;
            }
            if let Some(tc) = parse_tag_style_call(&cur[start..], *state.call_counter) {
                let call_end = cur[start..]
                    .find("</function>")
                    .map(|e| start + e + "</function>".len())
                    .unwrap_or(cur.len());
                cur = &cur[call_end..];
                state.calls.push(tc);
                *state.call_counter += 1;
            } else {
                cur = &cur[start + "<function>".len()..];
            }
        }
        if !state.calls.is_empty() {
            let _ = cur;
            state.remaining.clear();
        }
    }
}

/// 2026-09-26: Attribute-style `<function=NAME>` / `<function NAME>` calls.
pub struct BareFunctionAttrPass;
impl ToolCallPass for BareFunctionAttrPass {
    fn name(&self) -> &str {
        "bare_function_attr"
    }
    fn apply(&self, state: &mut PassState<'_>) {
        let text = state.remaining.clone();
        let mut cur = text.as_str();
        let mut first = true;
        while let Some(start) = cur.find("<function=").or_else(|| cur.find("<function ")) {
            if first {
                let before = cur[..start].trim();
                if !before.is_empty() {
                    state.content.push(before.to_string());
                }
                first = false;
            }
            if let Some(tc) = parse_qwen3_coder_call(&cur[start..], *state.call_counter) {
                let call_end = cur[start..]
                    .find("</function>")
                    .map(|e| start + e + "</function>".len())
                    .unwrap_or(cur.len());
                cur = &cur[call_end..];
                state.calls.push(tc);
                *state.call_counter += 1;
            } else {
                cur = &cur[start + "<function".len()..];
            }
        }
        if !state.calls.is_empty() {
            let _ = cur;
            state.remaining.clear();
        }
    }
}

/// 2026-09-26: The whole trimmed text is `name{…}`, where `{…}` is a JSON
/// object.
pub struct BareMistralNamePass;
impl ToolCallPass for BareMistralNamePass {
    fn name(&self) -> &str {
        "bare_mistral_name"
    }
    fn apply(&self, state: &mut PassState<'_>) {
        let trimmed = state.remaining.trim();
        let Some(brace_pos) = trimmed.find('{') else {
            return;
        };
        let name_part = trimmed[..brace_pos].trim();
        let json_part = &trimmed[brace_pos..];
        if name_part.is_empty()
            || !name_part.chars().all(is_tool_name_or_namespace_char)
            || !json_part.ends_with('}')
        {
            return;
        }
        let func_name = normalize_tool_name(name_part);
        // 2026-09-26: `json:` / `tool_call:` prose keeps its colon through
        // normalization; returning leaves the text for later passes.
        if !is_normalized_tool_name(&func_name) {
            return;
        }
        let Ok(args_obj) = serde_json::from_str::<serde_json::Value>(json_part) else {
            return;
        };
        if !args_obj.is_object() {
            return;
        }
        state.calls.push(ToolCall {
            id: next_tool_call_id(),
            call_type: "function".to_string(),
            function: FunctionCall {
                name: func_name,
                arguments: json_part.to_string(),
            },
        });
        *state.call_counter += 1;
        state.remaining.clear();
    }
}

/// 2026-09-26: Salvage for `<parameter=NAME>` written where `<function=NAME>`
/// belongs. Fires only when the text has a `</function>` and no
/// `<function>`, `<function=`, `<function `, `<|function=` or `<|function `
/// opener. The first `<parameter=NAME>` becomes `<function=NAME>` and the
/// result is parsed by `parse_qwen3_coder_call`, like a well-formed call.
pub struct ParamAsFunctionSalvagePass;
impl ToolCallPass for ParamAsFunctionSalvagePass {
    fn name(&self) -> &str {
        "param_as_function_salvage"
    }
    fn apply(&self, state: &mut PassState<'_>) {
        let text = state.remaining.as_str();
        if !text.contains("</function>") {
            return;
        }
        if text.contains("<function>")
            || text.contains("<function=")
            || text.contains("<function ")
            || text.contains("<|function=")
            || text.contains("<|function ")
        {
            return;
        }
        let Some(first_param) = text.find("<parameter=") else {
            return;
        };
        let after_eq = &text[first_param + "<parameter=".len()..];
        let Some(close_gt) = after_eq.find('>') else {
            return;
        };
        let name = after_eq[..close_gt].trim().to_string();
        if name.is_empty() || !name.chars().all(is_tool_name_or_namespace_char) {
            return;
        }
        // 2026-09-26: Only the first `<parameter=NAME>` is rewritten; later
        // `<parameter=K>V</parameter>` blocks stay arguments.
        let before = &text[..first_param];
        let tail = &text[first_param..];
        let from = format!("<parameter={name}>");
        let func_name = normalize_tool_name(&name);
        if !is_normalized_tool_name(&func_name) {
            return;
        }
        let to = format!("<function={func_name}>");
        let fixed_tail = tail.replacen(&from, &to, 1);
        let reconstructed = format!("{before}{fixed_tail}");
        let func_start = match reconstructed.find("<function=") {
            Some(p) => p,
            None => return,
        };
        let Some(tc) = parse_qwen3_coder_call(&reconstructed[func_start..], *state.call_counter)
        else {
            return;
        };
        let before_trimmed = before.trim();
        if !before_trimmed.is_empty() {
            state.content.push(before_trimmed.to_string());
        }
        state.calls.push(tc);
        *state.call_counter += 1;
        state.remaining.clear();
    }
}
