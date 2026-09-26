// SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(unused_imports, dead_code)]

//! 2026-09-26: The MiniMax XML tool-call format.
//!
//! Owner: server (tool parsing).
//! Invariants: none beyond the types.

use super::*;

/// 2026-09-26: MiniMax XML format.
///
/// Outer tag: `<minimax:tool_call>...</minimax:tool_call>`, which
/// `parse_tool_calls` rewrites to `<tool_call>` before its scan.
///
/// Inner format:
/// ```xml
/// <invoke name="tool_name">
/// <parameter name="key1">value1</parameter>
/// <parameter name="key2">value2</parameter>
/// </invoke>
/// ```
///
/// Compared with qwen3_coder: `<invoke name="X">` for `<function=X>`,
/// `<parameter name="K">` for `<parameter=K>`, and `</invoke>` for
/// `</function>`.
///
/// Values are parsed as JSON strings, and this parser keeps the
/// `wants_typed_arguments` default (false), so they are not coerced.
pub struct MinimaxXmlParser;

impl ToolCallParser for MinimaxXmlParser {
    fn name(&self) -> &str {
        "minimax_xml"
    }

    fn compile_tool_grammar(
        &self,
        engine: &mut GrammarEngine,
        tools: &[ToolDefinition],
        use_triggers: bool,
    ) -> Option<Result<CompiledGrammar, GrammarError>> {
        Some(engine.compile_minimax_xml_tool_grammar(tools, use_triggers))
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
        let mut prompt =
            String::from("# Tools\n\nYou have access to the following functions:\n\n<tools>\n");
        let body = tool_list_body(tools, levers, || {
            let mut s = String::new();
            for tool in tools {
                let json = serde_json::to_string(tool).unwrap_or_default();
                s.push_str(&format!("<tool>{json}</tool>\n"));
            }
            s
        });
        prompt.push_str(body.trim_end());
        prompt.push('\n');
        prompt.push_str(
            "</tools>\n\n\
             When making tool calls, use XML format to invoke tools and pass parameters:\n\
             \n<minimax:tool_call>\n\
             <invoke name=\"tool-name-1\">\n\
             <parameter name=\"param-key-1\">param-value-1</parameter>\n\
             <parameter name=\"param-key-2\">param-value-2</parameter>\n\
             ...\n\
             </invoke>\n\
             </minimax:tool_call>\n",
        );
        append_tool_choice_instruction(&mut prompt, tool_choice);
        prompt
    }

    fn format_tool_calls(&self, calls: &[IncomingToolCall]) -> String {
        let mut out = String::new();
        for tc in calls {
            let args: serde_json::Value = serde_json::from_str(&tc.function.arguments)
                .unwrap_or(serde_json::Value::Object(Default::default()));
            out.push_str("\n<minimax:tool_call>\n");
            out.push_str(&format!("<invoke name=\"{}\">\n", tc.function.name));
            if let Some(obj) = args.as_object() {
                for (key, val) in obj {
                    let val_str = match val {
                        serde_json::Value::String(s) => s.clone(),
                        other => serde_json::to_string(other).unwrap_or_default(),
                    };
                    out.push_str(&format!(
                        "<parameter name=\"{key}\">{val_str}</parameter>\n"
                    ));
                }
            }
            out.push_str("</invoke>\n</minimax:tool_call>");
        }
        out
    }

    fn leak_markers(&self) -> LeakMarkers {
        // 2026-09-26: The envelope markers cover `<minimax:tool_call>`, the
        // `<minimax:_call>` form and plain `<tool_call>`; inside any of them
        // the sanitizer passes `<invoke>` and `<parameter>` through for the
        // parser. Outside an envelope those inner tags are leaks and are
        // suppressed up to a close tag.
        const MARKERS: LeakMarkers = LeakMarkers {
            orphan_open: &["<parameter name=\"", "<invoke name=\""],
            close: &[
                "</parameter>",
                "</invoke>",
                "</minimax:tool_call>",
                "</tool_call>",
            ],
            envelope_open: &["<minimax:tool_call>", "<minimax:_call>", "<tool_call>"],
            envelope_close: &["</minimax:tool_call>", "</minimax:_call>", "</tool_call>"],
        };
        MARKERS
    }
}
