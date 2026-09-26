// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Grammar tests: one module per area, plus the shared test
//! vocabulary and tool definition.
//!
//! Owner: server (grammar) tests.
//! Invariants: none beyond the types.

use super::*;
use crate::tool_parser::ToolDefinition;

mod cold_compile;
mod engine_state;
mod gemma4_required;
mod minimax;
mod misc;
mod native_prewarm;
mod parallel_calls;
mod param_key_constraint;
mod poolside;
mod prewarm_ordering;
mod qwen3_coder_required;
mod sanitize;
mod tools_basic;
mod value_entry;

/// 2026-09-26: A 131-token vocabulary: ids 0..=127 are the single ASCII
/// characters, then `<tool_call>` (128), `</tool_call>` (129) and `<eos>`
/// (130).
pub(crate) fn test_vocab() -> Vec<String> {
    let mut vocab = Vec::new();
    for i in 0u8..128 {
        vocab.push(String::from(i as char));
    }
    vocab.push("<tool_call>".to_string());
    vocab.push("</tool_call>".to_string());
    vocab.push("<eos>".to_string());
    vocab
}

/// 2026-09-26: One tool, `get_weather`, with a required string `location`.
pub(crate) fn test_tool_defs() -> Vec<ToolDefinition> {
    vec![ToolDefinition {
        tool_type: "function".to_string(),
        function: crate::tool_parser::FunctionDefinition {
            name: "get_weather".to_string(),
            description: Some("Get weather for a location".to_string()),
            parameters: Some(serde_json::json!({
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City name"
                    }
                },
                "required": ["location"]
            })),
        },
    }]
}
