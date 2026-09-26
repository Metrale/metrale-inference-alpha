// SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(unused_imports, dead_code)]

//! 2026-09-26: The qwen3_coder tool-call format.
//!
//! Owner: server (tool parsing).
//! Invariants: none beyond the types.

use super::*;

/// 2026-09-26: Qwen3-Coder format: XML
/// `<function=name><parameter=key>value</parameter></function>` inside
/// `<tool_call>` tags.
pub struct Qwen3CoderParser;

impl ToolCallParser for Qwen3CoderParser {
    fn name(&self) -> &str {
        "qwen3_coder"
    }

    /// 2026-09-26: True: `parse_qwen3_coder_call` returns every value as a
    /// string, and `coerce_all` converts it to the schema's type, e.g. `"30"`
    /// to `30` for an `integer` parameter.
    fn wants_typed_arguments(&self) -> bool {
        true
    }

    /// 2026-09-26: `</parameter>` ends each `<parameter=NAME>` value; see
    /// [`ToolCallParser::param_value_close_delim`].
    fn param_value_close_delim(&self) -> Option<&'static str> {
        Some("</parameter>")
    }

    fn compile_tool_grammar(
        &self,
        engine: &mut GrammarEngine,
        tools: &[ToolDefinition],
        use_triggers: bool,
    ) -> Option<Result<CompiledGrammar, GrammarError>> {
        let value_close = self
            .param_value_close_delim()
            .expect("qwen3_coder declares a parameter-value close delimiter");
        Some(engine.compile_qwen3_coder_tool_grammar(tools, use_triggers, value_close))
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
        // 2026-09-26: The same layout as the tools block of
        // `jinja-templates/qwen3_5_moe.jinja`: tool definitions as JSON, one
        // per line, then the XML call format.
        let mut prompt =
            String::from("# Tools\n\nYou have access to the following functions:\n\n<tools>\n");
        // 2026-09-26: Append a retry rule, tagged `[metrale-f33]`, to the
        // description of a tool named `Bash` or `bash`. It is applied to a
        // copy of the tools, so it appears in both the JSON and the TSCG
        // rendering; a description that already has the tag is left alone.
        let f33_tools: Vec<ToolDefinition> = tools
            .iter()
            .map(|tool| {
                if !matches!(tool.function.name.as_str(), "Bash" | "bash") {
                    return tool.clone();
                }
                let mut t = tool.clone();
                let suffix = " | After ONE failure with \"command not found\" or exit code 127, do NOT retry the same command — the binary is permanently unavailable in this environment. Choose a different approach or tell the user the dependency is missing.";
                t.function.description = Some(match t.function.description {
                    Some(d) if !d.contains("[metrale-f33]") => format!("{d}\n[metrale-f33]{suffix}"),
                    Some(d) => d,
                    None => format!("[metrale-f33]{suffix}"),
                });
                t
            })
            .collect();
        if levers.tscg {
            prompt.push_str(&crate::tscg::compile_tools(&f33_tools));
            prompt.push('\n');
        } else {
            for t in &f33_tools {
                prompt.push_str(&serde_json::to_string(t).unwrap_or_default());
                prompt.push('\n');
            }
        }
        prompt.push_str(
            "\
</tools>\n\n\
If you choose to call a function ONLY reply in the following format with NO suffix:\n\n\
<tool_call>\n\
<function=example_function_name>\n\
<parameter=example_parameter_1>\n\
value_1\n\
</parameter>\n\
<parameter=example_parameter_2>\n\
This is the value for the second parameter\n\
that can span\n\
multiple lines\n\
</parameter>\n\
</function>\n\
</tool_call>\n\n\
",
        );
        // 2026-09-26: `METRALE_OFFICIAL_TOOL_PROMPT=1` uses the four-bullet
        // `<IMPORTANT>` reminder of `jinja-templates/qwen3_5_moe.jinja`
        // (line 53); otherwise the `<IMMEDIATE_TOOL_USE>` block and the longer
        // `<IMPORTANT>` list below are used.
        if std::env::var("METRALE_OFFICIAL_TOOL_PROMPT").as_deref() == Ok("1") {
            prompt.push_str("\
<IMPORTANT>\n\
Reminder:\n\
- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n\
- Required parameters MUST be specified\n\
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n\
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n\
</IMPORTANT>");
        } else {
            prompt.push_str("\
<IMMEDIATE_TOOL_USE>\n\
Tools are IMMEDIATELY EXECUTABLE. When you decide to use a tool, emit the <tool_call> directly. The tool_call's parameter values are the ONLY place file content, commands, or other tool inputs should appear — do NOT pre-render that content as prose or in markdown fences before the call.\n\
For 'bash'/'Bash' tools specifically: do NOT stage the command in a ```bash``` fence and do NOT prefix with phrases like \"Let me run this command:\" or \"Let me execute:\". The next emission after deciding to run a shell command must be the <tool_call> — the command goes inside <parameter=command>, not in markdown.\n\
For 'Write'/'Edit' tools specifically: do NOT pre-render the file content as a markdown ```toml/```rust/```python fence before the tool call. Do NOT write phrases like \"Let me create the Cargo.toml:\" or \"Now the source file:\" followed by a code fence containing the file body — the file body goes inside <parameter=content>, not in markdown. Pre-rendering the same content twice (once in markdown, once in the parameter) wastes tokens, can cause the tool call to be dropped if the markdown is long enough, and is the documented \"narrate-then-tool\" loop pattern. The next emission after deciding to write a file must be the <tool_call>.\n\
Example:\n\
    <tool_call>\n\
    <function=Write>\n\
    <parameter=file_path>/path/to/Cargo.toml</parameter>\n\
    <parameter=content>[package]\nname = \"x\"</parameter>\n\
    </function>\n\
    </tool_call>\n\
</IMMEDIATE_TOOL_USE>\n\n\
<IMPORTANT>\n\
- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags.\n\
- EVERY required parameter MUST have a non-empty value.\n\
- For 'write'/'Write': ALWAYS provide both file_path AND content. PREFER 'write' over 'edit' — write the complete updated file content instead of find-and-replace. Only use 'edit' for very small single-line changes.\n\
- For 'edit'/'Edit': file_path MUST be non-empty. oldString MUST be copied verbatim from the file.\n\
- To make MULTIPLE tool calls, use SEPARATE <tool_call> blocks — NEVER embed <tool_call> tags inside bash commands or heredocs.\n\
- NEVER simulate a tool response in your content. Do NOT emit <tool_response>, <file>, \"1: ...\\n2: ...\" line-numbered previews, or any other text pretending to show what a tool returned. The real response is provided by the system AFTER you emit the <tool_call>.\n\
- Tool names like 'write', 'read', 'edit', 'bash' are ONLY valid inside the `<function=NAME>` line of a `<tool_call>` block. NEVER emit bare tags like `<write>`, `<filePath>`, or `<command>` at the top level of your content.\n\
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after.\n\
- If there is no function call available, answer the question with your current knowledge and do not tell the user about function calls.\n\
</IMPORTANT>");
        }
        append_tool_choice_instruction(&mut prompt, tool_choice);
        prompt
    }

    fn format_tool_calls(&self, calls: &[IncomingToolCall]) -> String {
        let mut out = String::new();
        for tc in calls {
            let args: serde_json::Value = serde_json::from_str(&tc.function.arguments)
                .unwrap_or(serde_json::Value::Object(Default::default()));
            out.push_str("\n<tool_call>\n");
            out.push_str(&format!("<function={}>\n", tc.function.name));
            if let Some(obj) = args.as_object() {
                for (key, val) in obj {
                    let val_str = match val {
                        serde_json::Value::String(s) => s.clone(),
                        other => serde_json::to_string(other).unwrap_or_default(),
                    };
                    out.push_str(&format!("<parameter={key}>\n{val_str}\n</parameter>\n"));
                }
            }
            out.push_str("</function>\n</tool_call>");
        }
        out
    }

    fn leak_markers(&self) -> LeakMarkers {
        // 2026-09-26: No envelope markers, so each `orphan_open` string in
        // content starts suppression, which ends after the next `close`
        // string.
        //
        // `<tool_response>` is the wrapper the chat template puts around tool
        // messages (`jinja-templates/qwen3_5_moe.jinja`), so in the model's
        // output it is a simulated tool result.
        const MARKERS: LeakMarkers = LeakMarkers {
            orphan_open: &[
                "<parameter=",
                "<tool_response>",
                "<function_results>",
                "<result>",
                "<function=",
                "<tool_call>",
                "<tool_use>",
                "<>",
                "<param=",
                "<response>",
                "<_call>",
                "<_output>",
                "<_use_error>",
            ],
            close: &[
                "</parameter>",
                "</function>",
                "</tool_call>",
                "</tool_response>",
                "</function_results>",
                "</result>",
                "</tool_use>",
                "</response>",
                "</_call>",
                "</_output>",
                "</_use_error>",
            ],
            envelope_open: &[],
            envelope_close: &[],
        };
        MARKERS
    }
}
