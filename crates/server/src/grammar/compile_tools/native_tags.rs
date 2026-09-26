// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tool-call grammars for the Gemma-4 and MiniMax XML formats, whose
//! tags carry a JSON-schema body or free text.
//!
//! Owner: server grammar.
//! Invariants:
//! - Each `compile_*_tool_grammar` returns `GrammarError::NoTools` when `tools` is empty
//!   or every tool is skipped.

use metrale_grammar::CompiledGrammar;

use crate::tool_parser::ToolDefinition;

use super::super::engine::{GrammarEngine, GrammarError};
use super::super::schema::{enforce_min_length_on_required_strings, sanitize_schema_for_grammar};

impl GrammarEngine {
    /// 2026-09-26: Compile a grammar for Gemma-4 tool calls:
    /// `<|tool_call>call:NAME{"key":"val",...}<tool_call|>`. The argument block is standard
    /// JSON checked against the tool's schema, not Gemma's unquoted-key `<|"|>` form;
    /// `parse_gemma4_native_call` (`tool_parser/helpers_b.rs`) reads it through
    /// `gemma4_to_json`, which copies quoted strings unchanged. A tool whose schema cannot
    /// be sanitized, or has neither `properties` nor `type`, is skipped with a warning.
    pub fn compile_gemma4_tool_grammar(
        &mut self,
        tools: &[ToolDefinition],
        use_triggers: bool,
    ) -> Result<CompiledGrammar, GrammarError> {
        if tools.is_empty() {
            return Err(GrammarError::NoTools);
        }

        let mut tag_entries = Vec::with_capacity(tools.len());

        for tool in tools {
            let name = &tool.function.name;
            let raw_schema = tool
                .function
                .parameters
                .as_ref()
                .cloned()
                .unwrap_or_else(|| serde_json::json!({"type":"object","properties":{}}));
            let raw_schema = match sanitize_schema_for_grammar(&raw_schema) {
                Some(s) => s,
                None => {
                    tracing::warn!(target: "met::grammar::compile_tools", "Skipping tool '{name}' in grammar — schema unsanitizable");
                    continue;
                }
            };
            if raw_schema.get("properties").is_none() && raw_schema.get("type").is_none() {
                tracing::warn!(target: "met::grammar::compile_tools", "Skipping tool '{name}' in grammar — schema has no properties or type"
                );
                continue;
            }
            let schema = enforce_min_length_on_required_strings(&raw_schema);

            let begin = format!("<|tool_call>call:{name}");
            let end = "<tool_call|>";
            tag_entries.push(serde_json::json!({
                "type": "tag",
                "begin": begin,
                "content": {"type": "json_schema", "json_schema": schema},
                "end": end,
            }));
        }

        if tag_entries.is_empty() {
            return Err(GrammarError::NoTools);
        }

        // 2026-09-26: With `use_triggers`, the one trigger is `<|tool_call>call:`, which
        // starts every tag's `begin`. Without it the trigger is `<|tool_call>`: the
        // converter then emits one tag from its full `begin`, but still requires each tag
        // to start with exactly one trigger, so the list cannot be empty.
        let triggers = if use_triggers {
            vec!["<|tool_call>call:".to_string()]
        } else {
            vec!["<|tool_call>".to_string()]
        };

        let at_least_one = !use_triggers;
        let stop_after_first = !use_triggers;

        self.compile_structural_tag_raw(&triggers, &tag_entries, at_least_one, stop_after_first)
    }

    /// 2026-09-26: Compile a grammar for MiniMax XML tool calls.
    ///
    /// Native MiniMax format:
    /// ```xml
    /// <minimax:tool_call>
    /// <invoke name="tool_name">
    /// <parameter name="key1">value1</parameter>
    /// <parameter name="key2">value2</parameter>
    /// </invoke>
    /// </minimax:tool_call>
    /// ```
    ///
    /// Each tool's tag fixes the outer frame, `<minimax:tool_call>\n<invoke name="NAME">`
    /// and `</invoke>\n</minimax:tool_call>`, and leaves the body as `any_text`, so the
    /// `<parameter name="K">V</parameter>` lines are not checked against the schema.
    pub fn compile_minimax_xml_tool_grammar(
        &mut self,
        tools: &[ToolDefinition],
        use_triggers: bool,
    ) -> Result<CompiledGrammar, GrammarError> {
        if tools.is_empty() {
            return Err(GrammarError::NoTools);
        }

        let mut tag_entries = Vec::with_capacity(tools.len());

        for tool in tools {
            let name = &tool.function.name;
            // 2026-09-26: The schema does not shape the body. It is sanitized only so that a
            // tool with an unsanitizable schema is skipped with the same warning as in the
            // other formats.
            let raw_schema = tool
                .function
                .parameters
                .as_ref()
                .cloned()
                .unwrap_or_else(|| serde_json::json!({"type":"object","properties":{}}));
            if sanitize_schema_for_grammar(&raw_schema).is_none() {
                tracing::warn!(target: "met::grammar::compile_tools", "Skipping tool '{name}' in grammar — schema unsanitizable");
                continue;
            }

            let begin = format!("<minimax:tool_call>\n<invoke name=\"{name}\">");
            let end = "</invoke>\n</minimax:tool_call>";
            tag_entries.push(serde_json::json!({
                "type": "tag",
                "begin": begin,
                "content": {"type": "any_text"},
                "end": end,
            }));
        }

        if tag_entries.is_empty() {
            return Err(GrammarError::NoTools);
        }

        // 2026-09-26: One trigger, `<minimax:tool_call>`, which starts every tag's `begin`.
        // With or without `use_triggers`, `<minimax:tool_call>` must then continue with
        // `\n<invoke name="NAME">` for a registered tool (the `use_triggers` case is
        // `test_minimax_xml_grammar_rejects_degenerate`).
        let triggers = vec!["<minimax:tool_call>".to_string()];

        let at_least_one = !use_triggers;
        let stop_after_first = !use_triggers;

        self.compile_structural_tag_raw(&triggers, &tag_entries, at_least_one, stop_after_first)
    }
}
