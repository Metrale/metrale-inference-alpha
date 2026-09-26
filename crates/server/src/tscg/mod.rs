// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: TSCG: compile JSON tool schemas into compact function signatures for the prompt.
//!
//! Owner: server (tool prompt).
//! Invariants: `compile_tools` reads the tools by shared reference and returns
//! prompt text, so the request's tool list is unchanged.
//!
//! Technique from "TSCG: Deterministic Tool-Schema Compilation for Agentic LLM
//! Deployments" (arXiv:2605.04107). Operators implemented here: SDM in `sdm.rs`;
//! DRO, CFO, CAS and SAD-F in `render.rs`. Enabled per model by MODEL.toml
//! `[behavior].tscg`, false when absent; the value reaches the renderers as
//! `tool_parser::PromptLevers::tscg`.

mod render;
mod sdm;

use crate::tool_parser::ToolDefinition;

/// 2026-09-26: Compile a tool list into the compact TSCG block, one stanza per tool
/// separated by a newline. With `PromptLevers::tscg` set, it replaces the JSON
/// `<tools>` body inside a parser's `system_prompt()`.
///
/// Output shape (one stanza per tool):
///
/// ```text
/// search_files(query:str path?:str)
/// |Search files by content or pattern
///   query: search text
/// ```
///
/// Line 1 is the signature: `name(<params>)` with `?` marking optional
/// params (those absent from the schema's `required` list). Line 2, when the
/// description densifies to something, is the `|`-prefixed description.
/// Indented lines carry per-parameter docs, emitted only when the densified doc
/// is neither empty nor the parameter name. A tool with at least
/// `render::SAD_F_MIN_REQUIRED` required params ends with a `!needs:` line.
pub fn compile_tools(tools: &[ToolDefinition]) -> String {
    let mut out = String::new();
    for (i, tool) in tools.iter().enumerate() {
        if i > 0 {
            out.push('\n');
        }
        out.push_str(&render::compile_one(tool));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool_parser::{FunctionDefinition, ToolDefinition};

    fn tool(name: &str, desc: &str, params: serde_json::Value) -> ToolDefinition {
        ToolDefinition {
            tool_type: "function".to_string(),
            function: FunctionDefinition {
                name: name.to_string(),
                description: Some(desc.to_string()),
                parameters: Some(params),
            },
        }
    }

    #[test]
    fn compiles_basic_signature() {
        let t = tool(
            "search_files",
            "Search project files by content or filename pattern",
            serde_json::json!({
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query string"},
                    "path": {"type": "string", "description": "Optional directory path"}
                },
                "required": ["query"]
            }),
        );
        let out = compile_tools(std::slice::from_ref(&t));
        // 2026-09-26: Signature line: required `query` before optional `path?`.
        assert!(
            out.starts_with("search_files(query:str path?:str)"),
            "got: {out}"
        );
        // 2026-09-26: The description is on a `|` line, and the block is shorter than
        // the JSON.
        assert!(out.contains("\n|"), "got: {out}");
        assert!(out.len() < serde_json::to_string(&t).unwrap().len());
    }

    #[test]
    fn empty_tool_list_is_empty() {
        assert_eq!(compile_tools(&[]), "");
    }

    #[test]
    fn a_default_model_does_not_compile_its_schemas() {
        assert!(!crate::tool_parser::PromptLevers::default().tscg);
    }
}
