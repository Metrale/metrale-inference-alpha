// SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(unused_imports, dead_code)]

//! 2026-09-26: The Gemma-4 tool-call format.
//!
//! Owner: server (tool parsing).
//! Invariants: none beyond the types.

use super::*;

/// 2026-09-26: Gemma-4 format: `<|tool_call>call:fn_name{key:val,...}<tool_call|>`.
pub struct Gemma4Parser;

impl ToolCallParser for Gemma4Parser {
    fn name(&self) -> &str {
        "gemma4"
    }

    fn leak_markers(&self) -> LeakMarkers {
        LeakMarkers {
            orphan_open: &[],
            close: &[],
            envelope_open: &["<|tool_call>"],
            envelope_close: &["<tool_call|>"],
        }
    }

    fn compile_tool_grammar(
        &self,
        engine: &mut GrammarEngine,
        tools: &[ToolDefinition],
        use_triggers: bool,
    ) -> Option<Result<CompiledGrammar, GrammarError>> {
        Some(engine.compile_gemma4_tool_grammar(tools, use_triggers))
    }

    fn has_tool_grammar(&self) -> bool {
        true
    }

    fn system_prompt(
        &self,
        _tools: &[ToolDefinition],
        _tool_choice: &ToolChoice,
        _levers: &super::PromptLevers,
    ) -> String {
        // 2026-09-26: `jinja-templates/gemma4.jinja` renders the tool
        // definitions as `<|tool>` blocks.
        String::new()
    }

    fn format_tool_calls(&self, calls: &[IncomingToolCall]) -> String {
        let mut out = String::new();
        for tc in calls {
            out.push_str("<|tool_call>call:");
            out.push_str(&tc.function.name);
            out.push('{');
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&tc.function.arguments)
                && let Some(obj) = v.as_object()
            {
                let mut first = true;
                for (key, val) in obj {
                    if !first {
                        out.push(',');
                    }
                    first = false;
                    out.push_str(key);
                    out.push(':');
                    format_gemma4_value(&mut out, val);
                }
            }
            out.push_str("}<tool_call|>");
        }
        out
    }

    fn format_tool_response(&self, content: &str) -> String {
        format!("<|tool_response>response:result{{value:<|\"|>{content}<|\"|>}}<tool_response|>")
    }
}
