// SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(unused_imports, dead_code)]

//! 2026-09-26: The `qwen3_xml` parser name.
//!
//! Owner: server (tool parsing).
//! Invariants: none beyond the types.

use super::*;

/// 2026-09-26: `qwen3_xml`: the qwen3_coder wire format. Prompt, rendering,
/// leak markers and grammar delegate to [`Qwen3CoderParser`], and arguments
/// are coerced to schema types as for [`Qwen3CoderParser`]
/// (`wants_typed_arguments`).
pub struct Qwen3XmlParser;

impl ToolCallParser for Qwen3XmlParser {
    fn name(&self) -> &str {
        "qwen3_xml"
    }

    fn wants_typed_arguments(&self) -> bool {
        true
    }

    fn system_prompt(
        &self,
        tools: &[ToolDefinition],
        tool_choice: &ToolChoice,
        levers: &super::PromptLevers,
    ) -> String {
        Qwen3CoderParser.system_prompt(tools, tool_choice, levers)
    }

    fn format_tool_calls(&self, calls: &[IncomingToolCall]) -> String {
        Qwen3CoderParser.format_tool_calls(calls)
    }

    fn format_tool_response(&self, content: &str) -> String {
        Qwen3CoderParser.format_tool_response(content)
    }

    fn leak_markers(&self) -> LeakMarkers {
        Qwen3CoderParser.leak_markers()
    }

    fn compile_tool_grammar(
        &self,
        engine: &mut GrammarEngine,
        tools: &[ToolDefinition],
        use_triggers: bool,
    ) -> Option<Result<CompiledGrammar, GrammarError>> {
        Qwen3CoderParser.compile_tool_grammar(engine, tools, use_triggers)
    }

    fn has_tool_grammar(&self) -> bool {
        true
    }
}
