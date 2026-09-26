// SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(unused_imports, dead_code)]

//! 2026-09-26: The bare-JSON tool-call format.
//!
//! Owner: server (tool parsing).
//! Invariants: none beyond the types.

use super::*;

/// 2026-09-26: Bare-JSON format: a top-level
/// `{"name":"<tool>","arguments":{...}}` object with no `<tool_call>`
/// wrapper.
pub struct BareJsonParser;

impl ToolCallParser for BareJsonParser {
    fn name(&self) -> &str {
        "bare_json"
    }

    fn compile_tool_grammar(
        &self,
        engine: &mut GrammarEngine,
        tools: &[ToolDefinition],
        use_triggers: bool,
    ) -> Option<Result<CompiledGrammar, GrammarError>> {
        Some(engine.compile_bare_json_tool_grammar(tools, use_triggers))
    }

    fn has_tool_grammar(&self) -> bool {
        true
    }

    fn system_prompt(
        &self,
        tools: &[ToolDefinition],
        tool_choice: &ToolChoice,
        levers: &super::PromptLevers,
    ) -> String {
        let tools_json = tool_list_body(tools, levers, || {
            serde_json::to_string(tools).unwrap_or_else(|_| "[]".into())
        });
        let mut prompt = format!(
            "You are a function-calling AI model. You have access to the following tools, \
             provided as JSON schemas inside <tools></tools>:\n<tools>\n{tools_json}\n</tools>\n\n\
             To invoke a tool, output a single top-level JSON object with exactly two fields: \
             \"name\" (one of the tool names above) and \"arguments\" (an object matching that \
             tool's parameter schema). Do not wrap it in any tags. Do not output any other text \
             after the JSON object.\n\n\
             Example: {{\"name\": \"<tool-name>\", \"arguments\": {{...}}}}"
        );
        append_tool_choice_instruction(&mut prompt, tool_choice);
        prompt
    }

    fn format_tool_calls(&self, calls: &[IncomingToolCall]) -> String {
        let mut out = String::new();
        for tc in calls {
            let args: serde_json::Value = serde_json::from_str(&tc.function.arguments)
                .unwrap_or(serde_json::Value::Object(Default::default()));
            out.push_str(&format!(
                "\n{}",
                serde_json::json!({"name": tc.function.name, "arguments": args})
            ));
        }
        out
    }
}
