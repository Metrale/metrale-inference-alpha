// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tool-call grammars whose body is raw EBNF over `<parameter=…>` or
//! `<arg_key>`/`<arg_value>` pairs: qwen3_coder and poolside_v1. The EBNF
//! builders and env switches they use are in the parent `compile_tools`.
//!
//! Owner: server grammar.
//! Invariants:
//! - Each `compile_*_tool_grammar` returns `GrammarError::NoTools` when `tools` is empty
//!   or every tool is skipped.

use std::collections::HashMap;

use metrale_grammar::CompiledGrammar;

use crate::tool_parser::ToolDefinition;

use super::super::engine::{GrammarEngine, GrammarError};
use super::super::schema::{enforce_min_length_on_required_strings, sanitize_schema_for_grammar};

use super::{
    ebnf_until_close_ladder, ebnf_until_close_ladder_no_leading_ws, schema_param_names,
    short_tool_trigger_enabled, xml_param_value_body_ebnf,
};

impl GrammarEngine {
    /// 2026-09-25: Compile a grammar for qwen3_coder XML tool calls:
    /// `<tool_call>\n<function=NAME>\n`, a raw-EBNF body from
    /// `xml_param_value_body_ebnf` with parameter names limited by `schema_param_names`,
    /// then `\n</function>\n</tool_call>`. `value_close` is the parser's
    /// `param_value_close_delim` (`</parameter>` for `Qwen3CoderParser`). A tool whose
    /// schema cannot be sanitized, or has neither `properties` nor `type`, is skipped
    /// with a warning. When compiling fails, it retries once with tag entries built the
    /// same way.
    pub fn compile_qwen3_coder_tool_grammar(
        &mut self,
        tools: &[ToolDefinition],
        use_triggers: bool,
        value_close: &str,
    ) -> Result<CompiledGrammar, GrammarError> {
        if tools.is_empty() {
            return Err(GrammarError::NoTools);
        }

        let mut tag_entries = Vec::with_capacity(tools.len());

        struct SanitizedTool {
            name: String,
            schema: serde_json::Value,
        }
        let mut sanitized_tools = Vec::with_capacity(tools.len());
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
            sanitized_tools.push(SanitizedTool {
                name: name.clone(),
                schema,
            });
        }

        for st in &sanitized_tools {
            let begin = format!("<tool_call>\n<function={}>\n", st.name);
            let end = "\n</function>\n</tool_call>";
            let param_names = schema_param_names(&st.schema);
            let body_ebnf = xml_param_value_body_ebnf(value_close, param_names.as_deref());
            tag_entries.push(serde_json::json!({
                "type": "tag",
                "begin": begin,
                "content": {"type": "grammar", "grammar": body_ebnf},
                "end": end,
            }));
        }

        if tag_entries.is_empty() {
            return Err(GrammarError::NoTools);
        }

        // 2026-09-26: With `use_triggers` (and `METRALE_TOOL_SHORT_TRIGGER` off), one
        // trigger per tool, `<tool_call>\n<function=NAME>`, so text stays free until a
        // whole trigger has been emitted. The converter rejects a grammar in which one tag's
        // `begin` starts with two triggers; the closing `>` keeps the set prefix-free when
        // one tool name is a prefix of another. Otherwise the one trigger is `<tool_call>`.
        let triggers: Vec<String> = if use_triggers && !short_tool_trigger_enabled() {
            // 2026-09-26: Deduplicate: two tools with one name would give two equal
            // triggers, which the converter rejects the same way.
            let mut seen = HashMap::<String, bool>::new();
            sanitized_tools
                .iter()
                .map(|st| format!("<tool_call>\n<function={}>", st.name))
                .filter(|trigger| seen.insert(trigger.clone(), true).is_none())
                .collect()
        } else {
            vec!["<tool_call>".to_string()]
        };

        let at_least_one = !use_triggers;
        let stop_after_first = !use_triggers;

        match self.compile_structural_tag_raw(
            &triggers,
            &tag_entries,
            at_least_one,
            stop_after_first,
        ) {
            Ok(compiled) => Ok(compiled),
            Err(e) => {
                // 2026-09-26: Retry once with tag entries built as above (same `begin`,
                // EBNF body and `end`), after logging the tool names.
                let tool_names: Vec<&str> =
                    sanitized_tools.iter().map(|st| st.name.as_str()).collect();
                tracing::info!(target: "met::grammar::compile_tools", "qwen_xml_parameter grammar fell back to json_schema ({e:?}). \
                     Functional but slightly looser tool-call grammar. Tools in \
                     this batch: [{}]. If you want to help narrow this down, \
                     set RUST_LOG=trace and re-run — the rejected schema is \
                     emitted at trace level by xgrammar.",
                    tool_names.join(", "),
                );
                let tag_entries_fallback: Vec<serde_json::Value> = sanitized_tools
                    .iter()
                    .map(|st| {
                        let param_names = schema_param_names(&st.schema);
                        let body_ebnf =
                            xml_param_value_body_ebnf(value_close, param_names.as_deref());
                        serde_json::json!({
                            "type": "tag",
                            "begin": format!("<tool_call>\n<function={}>\n", st.name),
                            "content": {"type": "grammar", "grammar": body_ebnf},
                            "end": "\n</function>\n</tool_call>",
                        })
                    })
                    .collect();
                self.compile_structural_tag_raw(
                    &triggers,
                    &tag_entries_fallback,
                    at_least_one,
                    stop_after_first,
                )
            }
        }
    }

    /// 2026-09-26: Compile a grammar for poolside_v1 tool calls:
    /// `<tool_call>NAME<arg_key>K</arg_key><arg_value>V{value_close}…</tool_call>`, one or
    /// more pairs, with `value_close` from the parser (`</arg_value>` for
    /// `PoolsideV1Parser`). One trigger, `<tool_call>`. A tool whose `properties` is an
    /// empty object (and `additionalProperties` is not `true`) gets an empty body. A tool
    /// whose schema cannot be sanitized is skipped with a warning.
    pub fn compile_poolside_v1_tool_grammar(
        &mut self,
        tools: &[ToolDefinition],
        use_triggers: bool,
        value_close: &str,
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
            let has_no_parameters = raw_schema
                .get("properties")
                .and_then(serde_json::Value::as_object)
                .is_some_and(serde_json::Map::is_empty)
                && raw_schema
                    .get("additionalProperties")
                    .and_then(serde_json::Value::as_bool)
                    != Some(true);
            let Some(schema) = sanitize_schema_for_grammar(&raw_schema) else {
                tracing::warn!(target: "met::grammar::compile_tools", "Skipping tool '{name}' in grammar — schema unsanitizable");
                continue;
            };

            let param_names = schema_param_names(&schema);
            // 2026-09-26: A required parameter typed `string` gets `req_value`, which needs
            // at least one non-blank byte; other parameters keep `value ::= value_part*`,
            // which may be empty. The grammar does not force a required parameter to
            // appear. `enforce_min_length_on_required_strings` is not used: this body is
            // raw EBNF, and only the schema's property names and `required` list are read.
            let required_strings: std::collections::BTreeSet<&str> = {
                let props = schema
                    .get("properties")
                    .and_then(serde_json::Value::as_object);
                schema
                    .get("required")
                    .and_then(serde_json::Value::as_array)
                    .map(|req| {
                        req.iter()
                            .filter_map(serde_json::Value::as_str)
                            .filter(|key| {
                                props
                                    .and_then(|p| p.get(*key))
                                    .and_then(|prop| prop.get("type"))
                                    .and_then(serde_json::Value::as_str)
                                    == Some("string")
                            })
                            .collect()
                    })
                    .unwrap_or_default()
            };
            let quote =
                |param: &str| serde_json::to_string(param).unwrap_or_else(|_| "\"\"".into());
            let alternation = |names: &[String]| {
                names
                    .iter()
                    .map(|n| quote(n))
                    .collect::<Vec<_>>()
                    .join(" | ")
            };
            let value_ladder = ebnf_until_close_ladder(value_close);
            let first_nonws_ladder = ebnf_until_close_ladder_no_leading_ws(value_close);
            let content = if has_no_parameters {
                serde_json::json!({"type": "const_string", "value": ""})
            } else {
                let body_ebnf = match param_names.as_ref() {
                    // 2026-09-26: Known property set and at least one required string: split
                    // the pair rule so only those parameters need a non-blank value.
                    Some(names) if !names.is_empty() && !required_strings.is_empty() => {
                        let (req, opt): (Vec<String>, Vec<String>) = names
                            .iter()
                            .cloned()
                            .partition(|n| required_strings.contains(n.as_str()));
                        let mut rules = vec![
                            "root ::= pair pair*".to_string(),
                            format!(
                                "reqpair ::= \"<arg_key>\" reqname \"</arg_key><arg_value>\" \
                                 req_value \"{value_close}\""
                            ),
                            format!("reqname ::= {}", alternation(&req)),
                        ];
                        if opt.is_empty() {
                            rules.insert(1, "pair ::= reqpair".to_string());
                        } else {
                            rules.insert(1, "pair ::= reqpair | optpair".to_string());
                            rules.push(format!(
                                "optpair ::= \"<arg_key>\" optname \"</arg_key><arg_value>\" \
                                 value \"{value_close}\""
                            ));
                            rules.push(format!("optname ::= {}", alternation(&opt)));
                            rules.push("value ::= value_part*".to_string());
                        }
                        // 2026-09-26: Leading blanks are legal, but `req_first_nonws` must
                        // then match a non-blank byte (`ebnf_until_close_ladder_no_leading_ws`),
                        // so an all-blank value has no parse.
                        rules.push(
                            "req_value ::= req_leading_ws req_first_nonws value_part*".to_string(),
                        );
                        rules.push("req_leading_ws ::= [ \\t\\r\\n]*".to_string());
                        rules.push(format!("req_first_nonws ::= {first_nonws_ladder}"));
                        rules.push(format!("value_part ::= {value_ladder}"));
                        rules.join("\n")
                    }
                    // 2026-09-26: No required string parameter, or an open or unknown
                    // property set (`param_names` is `None`): one pair rule, and any value
                    // may be empty.
                    other => {
                        let paramname_rule = match other {
                            Some(names) if !names.is_empty() => {
                                format!("paramname ::= {}", alternation(names))
                            }
                            _ => "paramname ::= [a-zA-Z_] [a-zA-Z_0-9]*".to_string(),
                        };
                        format!(
                            "root ::= pair pair*\n\
                             pair ::= \"<arg_key>\" paramname \"</arg_key><arg_value>\" value \"{value_close}\"\n\
                             {paramname_rule}\n\
                             value ::= value_part*\n\
                             value_part ::= {value_ladder}"
                        )
                    }
                };
                serde_json::json!({"type": "grammar", "grammar": body_ebnf})
            };
            tag_entries.push(serde_json::json!({
                "type": "tag",
                "begin": format!("<tool_call>{name}"),
                "content": content,
                "end": "</tool_call>",
            }));
        }

        if tag_entries.is_empty() {
            return Err(GrammarError::NoTools);
        }

        self.compile_structural_tag_raw(
            &["<tool_call>".to_string()],
            &tag_entries,
            !use_triggers,
            !use_triggers,
        )
    }
}
