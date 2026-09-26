// SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(unused_imports, dead_code)]

//! 2026-09-26: The Mistral native tool-call format.
//!
//! Owner: server (tool parsing).
//! Invariants: none beyond the types.

use super::*;

/// 2026-09-26: Mistral native format: `[TOOL_CALLS]name[ARGS]{"key":"val",...}`.
///
/// Arguments are JSON, with no `<tool_call>` wrapper. Calls chain:
/// `[TOOL_CALLS]f1[ARGS]{...}[TOOL_CALLS]f2[ARGS]{...}`; a call ends at the
/// next `[TOOL_CALLS]` or at the end of the text (`parse_mistral_native_calls`).
pub struct MistralNativeParser;

pub(crate) const MISTRAL_TOOL_CALLS_TAG: &str = "[TOOL_CALLS]";
pub(crate) const MISTRAL_ARGS_TAG: &str = "[ARGS]";

impl ToolCallParser for MistralNativeParser {
    fn name(&self) -> &str {
        "mistral"
    }

    fn system_prompt(
        &self,
        tools: &[ToolDefinition],
        tool_choice: &ToolChoice,
        levers: &super::PromptLevers,
    ) -> String {
        // 2026-09-26: `jinja-templates/mistral.jinja` also renders the tools,
        // in an `[AVAILABLE_TOOLS]` block.
        let tools_json = tool_list_body(tools, levers, || {
            serde_json::to_string(tools).unwrap_or_else(|_| "[]".into())
        });
        let mut prompt = format!(
            "You have access to the following tools:\n<tools>\n{tools_json}\n</tools>\n\n\
             When you need to call a tool, respond in Mistral native format:\n\
             [TOOL_CALLS]function_name[ARGS]{{\"arg1\": \"val1\", \"arg2\": \"val2\"}}\n\
             Use JSON for arguments. To call multiple tools, chain them:\n\
             [TOOL_CALLS]f1[ARGS]{{...}}[TOOL_CALLS]f2[ARGS]{{...}}"
        );
        append_tool_choice_instruction(&mut prompt, tool_choice);
        prompt
    }

    fn format_tool_calls(&self, calls: &[IncomingToolCall]) -> String {
        let mut out = String::new();
        for tc in calls {
            let args: serde_json::Value = serde_json::from_str(&tc.function.arguments)
                .unwrap_or(serde_json::Value::Object(Default::default()));
            out.push_str(MISTRAL_TOOL_CALLS_TAG);
            out.push_str(&tc.function.name);
            out.push_str(MISTRAL_ARGS_TAG);
            out.push_str(&serde_json::to_string(&args).unwrap_or_else(|_| "{}".into()));
        }
        out
    }

    fn format_tool_response(&self, content: &str) -> String {
        // 2026-09-26: The same `[TOOL_RESULTS]` wrapper
        // `jinja-templates/mistral.jinja` puts around tool messages.
        format!("[TOOL_RESULTS]{content}[/TOOL_RESULTS]")
    }
}
