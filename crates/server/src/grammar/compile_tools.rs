// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tool-call grammar compilation for the Hermes, bare-JSON, qwen3_coder,
//! poolside_v1, Gemma-4 and MiniMax XML formats. Each format is one `triggered_tags`
//! structural tag with one tag per tool.
//!
//! `use_triggers` is false for `tool_choice` `required` or a named function, and always
//! for the minimax_xml parser (`api/chat/sampling_setup.rs`). Every compiler sets
//! `at_least_one = stop_after_first = !use_triggers`. With both set, the structural-tag
//! converter emits one tag, from its full `begin`, and only checks the triggers;
//! otherwise text is free until a trigger, after which the rest of a matching tag's
//! `begin` is forced (`metrale_grammar` `structural_tag/converter_tags.rs`).
//!
//! Owner: server grammar.
//! Invariants:
//! - Each `compile_*_tool_grammar` returns `GrammarError::NoTools` when `tools` is empty
//!   or every tool is skipped.

use std::collections::HashMap;

use metrale_grammar::CompiledGrammar;

use crate::tool_parser::ToolDefinition;

use super::engine::{GrammarEngine, GrammarError};
use super::schema::{enforce_min_length_on_required_strings, sanitize_schema_for_grammar};

mod native_tags;
mod xml_param;

/// 2026-09-26: Escape a single char for use inside an EBNF char class `[^ … ]`.
fn ebnf_class_escape(c: char) -> String {
    match c {
        ']' | '\\' | '^' | '-' => format!("\\{c}"),
        _ => c.to_string(),
    }
}

/// 2026-09-26: Escape a single char for use inside an EBNF double-quoted string literal.
fn ebnf_literal_escape(c: char) -> String {
    match c {
        '"' | '\\' => format!("\\{c}"),
        _ => c.to_string(),
    }
}

/// 2026-09-26: One repetition of a run of text that stops before the literal `close`, as
/// an EBNF alternation (a negative-prefix ladder). For `close = c0 c1 … c{n-1}` it is
///   `[^c0] | "c0" [^c1] | "c0c1" [^c2] | … | "c0…c{n-2}" [^c{n-1}]`:
/// one byte that is not `c0`, or a proper prefix of `close` followed by one byte that
/// does not continue it. The enclosing rule consumes the close itself. The close comes
/// from the format (`ToolCallParser::param_value_close_delim`).
///
/// With `force_close` and a `close` longer than one char, the last arm
/// (`"c0…c{n-2}" [^c{n-1}]`, for qwen3_coder `"</parameter" [^>]`) is left out, so no
/// arm continues the longest proper prefix of `close` with another byte.
/// [`xml_param_value_body_ebnf`] sets `force_close` from `METRALE_GRAMMAR_FORCE_CLOSE=1`.
fn ebnf_until_close_ladder_opts(close: &str, force_close: bool) -> String {
    let chars: Vec<char> = close.chars().collect();
    debug_assert!(!chars.is_empty(), "close delimiter must be non-empty");
    let mut alts: Vec<String> = Vec::with_capacity(chars.len().max(1));
    let depth = if force_close && chars.len() > 1 {
        chars.len() - 1
    } else {
        chars.len()
    };
    for k in 0..depth {
        let neg = ebnf_class_escape(chars[k]);
        if k == 0 {
            alts.push(format!("[^{neg}]"));
        } else {
            let prefix: String = chars[..k]
                .iter()
                .copied()
                .map(ebnf_literal_escape)
                .collect();
            alts.push(format!("\"{prefix}\" [^{neg}]"));
        }
    }
    if alts.is_empty() {
        // 2026-09-26: Empty `close`: any single byte except NUL.
        return "[^\\x00]".to_string();
    }
    alts.join(" | ")
}

fn ebnf_until_close_ladder(close: &str) -> String {
    ebnf_until_close_ladder_opts(close, false)
}

/// 2026-09-26: Like [`ebnf_until_close_ladder`], but the first arm also excludes space,
/// tab, CR and LF: `[^ \t\r\nc0] | "c0" [^c1] | …`. The other arms begin with `c0` and
/// are unchanged. poolside_v1 uses it for the first byte of a required string value. An
/// empty `close` gives `[^ \t\r\n\x00]`.
fn ebnf_until_close_ladder_no_leading_ws(close: &str) -> String {
    let chars: Vec<char> = close.chars().collect();
    debug_assert!(!chars.is_empty(), "close delimiter must be non-empty");
    let mut alts: Vec<String> = Vec::with_capacity(chars.len().max(1));
    for (k, &ch) in chars.iter().enumerate() {
        let neg = ebnf_class_escape(ch);
        if k == 0 {
            alts.push(format!("[^ \\t\\r\\n{neg}]"));
        } else {
            let prefix: String = chars[..k]
                .iter()
                .copied()
                .map(ebnf_literal_escape)
                .collect();
            alts.push(format!("\"{prefix}\" [^{neg}]"));
        }
    }
    if alts.is_empty() {
        return "[^ \\t\\r\\n\\x00]".to_string();
    }
    alts.join(" | ")
}

/// 2026-09-26: `METRALE_GRAMMAR_ALLOW_EMPTY_VALUE` and `METRALE_GRAMMAR_FORCE_CLOSE`; each
/// is on only when exactly `1`, and is read on every call.
fn grammar_allow_empty_value() -> bool {
    std::env::var("METRALE_GRAMMAR_ALLOW_EMPTY_VALUE").as_deref() == Ok("1")
}
fn grammar_force_close() -> bool {
    std::env::var("METRALE_GRAMMAR_FORCE_CLOSE").as_deref() == Ok("1")
}

/// 2026-09-26: Most `rest_part` repetitions in a parameter value when
/// `METRALE_GRAMMAR_VALUE_HARDEN=1`: the rule becomes `rest ::= rest_part{0,6000}`
/// instead of `rest ::= rest_part*`, so a value whose close never matches cannot grow
/// without end. A `rest_part` is one byte or a proper prefix of the close plus one byte.
const VALUE_REST_MAX_REPEAT: u32 = 6000;

/// 2026-09-26: Whether `METRALE_GRAMMAR_VALUE_HARDEN` is exactly `1`; read on every call.
fn value_harden_enabled() -> bool {
    std::env::var("METRALE_GRAMMAR_VALUE_HARDEN").as_deref() == Ok("1")
}

/// 2026-09-26: Whether `METRALE_TOOL_SHORT_TRIGGER` is exactly `1`; read on every call.
/// When it is, qwen3_coder with `use_triggers` has the one trigger `<tool_call>` instead
/// of one `<tool_call>\n<function=NAME>` trigger per tool.
fn short_tool_trigger_enabled() -> bool {
    std::env::var("METRALE_TOOL_SHORT_TRIGGER").as_deref() == Ok("1")
}

/// 2026-09-26: EBNF for an XML-style tool-call body: one or more
/// `<parameter=NAME>VALUE{value_close}` blocks separated by newlines. `value_close`
/// comes from the format's [`crate::tool_parser::ToolCallParser::param_value_close_delim`].
/// NAME is one of `param_names` when that is non-empty (see [`schema_param_names`]), and
/// any identifier otherwise. VALUE is built from [`ebnf_until_close_ladder_opts`]. The
/// opt-ins come from `METRALE_GRAMMAR_ALLOW_EMPTY_VALUE` and `METRALE_GRAMMAR_FORCE_CLOSE`.
pub(crate) fn xml_param_value_body_ebnf(
    value_close: &str,
    param_names: Option<&[String]>,
) -> String {
    xml_param_value_body_ebnf_opts(
        value_close,
        param_names,
        grammar_allow_empty_value(),
        grammar_force_close(),
    )
}

/// 2026-09-26: [`xml_param_value_body_ebnf`] with the two opt-ins as parameters. It still
/// reads `METRALE_GRAMMAR_VALUE_HARDEN` itself (the `rest` rule).
pub(crate) fn xml_param_value_body_ebnf_opts(
    value_close: &str,
    param_names: Option<&[String]>,
    allow_empty_value: bool,
    force_close: bool,
) -> String {
    let ladder = ebnf_until_close_ladder_opts(value_close, force_close);
    let rest_rule = if value_harden_enabled() {
        format!("rest ::= rest_part{{0,{VALUE_REST_MAX_REPEAT}}}")
    } else {
        "rest ::= rest_part*".to_string()
    };
    let paramname_rule = match param_names {
        Some(names) if !names.is_empty() => {
            let alts: Vec<String> = names
                .iter()
                .map(|n| serde_json::to_string(n).unwrap_or_else(|_| "\"\"".into()))
                .collect();
            format!("paramname ::= {}", alts.join(" | "))
        }
        _ => "paramname ::= [a-zA-Z_] [a-zA-Z_0-9]*".to_string(),
    };
    // 2026-09-26: VALUE is a run of blanks (`leading_ws`, which admits `\n`), then
    // `nonempty_value`: a first byte that is not blank, `=`, `>` or `<`, or one of the
    // ladder's quoted arms (a proper prefix of `value_close` and a byte that breaks it,
    // `ladder_lt_arms`), then `rest`. Excluding `=` and `>` stops a token such as `>=`
    // from closing the NAME with its `>` and starting the value with `=`. With
    // `allow_empty_value` (`METRALE_GRAMMAR_ALLOW_EMPTY_VALUE=1`) `nonempty_value` is
    // optional.
    let value_rule = if allow_empty_value {
        "value ::= leading_ws nonempty_value?"
    } else {
        "value ::= leading_ws nonempty_value"
    };
    format!(
        r#"root ::= param ("\n" param)*
param ::= "<parameter=" paramname ">" value "{value_close}"
{paramname_rule}
{value_rule}
nonempty_value ::= first_content rest
leading_ws ::= [ \t\r\n]*
first_content ::= [^ \t\r\n<=>] | {first_lt_arms}
{rest_rule}
rest_part ::= {ladder}
"#,
        first_lt_arms = ladder_lt_arms(&ladder),
    )
}

/// 2026-09-26: The arms of a ladder ([`ebnf_until_close_ladder`]) that begin with a
/// quoted prefix of the close. For close `</parameter>` the ladder is
/// `[^<] | "<" [^/] | "</" [^p] | …` and the result is `"<" [^/] | "</" [^p] | …`.
/// `first_content` uses them so a value may start with a `<` that does not start the
/// close, such as `<script>` or `<!DOCTYPE`.
fn ladder_lt_arms(ladder: &str) -> String {
    let arms: Vec<&str> = ladder
        .split(" | ")
        .filter(|arm| arm.starts_with('"'))
        .collect();
    if arms.is_empty() {
        // 2026-09-26: A one-char close has no quoted arms; return the base class of
        // `first_content` again so the alternation is not empty.
        return "[^ \\t\\r\\n<=>]".to_string();
    }
    arms.join(" | ")
}

/// 2026-09-26: The parameter names a tool call may use for this (sanitized) schema: the
/// keys of `properties`. `None` (any identifier) when `properties` is missing, not an
/// object or empty, or when `additionalProperties` is present and not `false`, since
/// that schema allows other keys. Used by qwen3_coder and poolside_v1.
pub(crate) fn schema_param_names(schema: &serde_json::Value) -> Option<Vec<String>> {
    let props = schema.get("properties")?.as_object()?;
    if props.is_empty() {
        return None;
    }
    match schema.get("additionalProperties") {
        None | Some(serde_json::Value::Bool(false)) => {}
        Some(_) => return None,
    }
    Some(props.keys().cloned().collect())
}

impl GrammarEngine {
    /// 2026-09-26: Compile a grammar for Hermes tool calls:
    /// `<tool_call>\n{"name": "NAME", "arguments": ARGS}\n</tool_call>`, with ARGS checked
    /// against the tool's schema. A tool whose schema cannot be sanitized, or has neither
    /// `properties` nor `type`, is skipped with a warning.
    pub fn compile_hermes_tool_grammar(
        &mut self,
        tools: &[ToolDefinition],
        use_triggers: bool,
    ) -> Result<CompiledGrammar, GrammarError> {
        if tools.is_empty() {
            return Err(GrammarError::NoTools);
        }

        let mut tag_entries = Vec::with_capacity(tools.len());
        let mut triggers = Vec::new();
        let mut seen_triggers = HashMap::<String, bool>::new();

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
                    tracing::warn!("Skipping tool '{name}' in grammar — schema unsanitizable");
                    continue;
                }
            };
            if raw_schema.get("properties").is_none() && raw_schema.get("type").is_none() {
                tracing::warn!(
                    "Skipping tool '{name}' in grammar — schema has no properties or type"
                );
                continue;
            }
            let schema = enforce_min_length_on_required_strings(&raw_schema);

            // 2026-09-26: `begin` and `end` follow the format `HermesParser::system_prompt`
            // shows the model: a newline after `<tool_call>` and a space after each colon
            // (`tool_parser/hermes.rs`).
            let begin = format!("<tool_call>\n{{\"name\": \"{name}\", \"arguments\": ");
            tag_entries.push(serde_json::json!({
                "type": "tag",
                "begin": begin,
                "content": {"type": "json_schema", "json_schema": schema},
                "end": "}\n</tool_call>",
            }));

            // 2026-09-26: One shared trigger, `<tool_call>`; each tag's `begin` then fixes
            // the rest of the opening.
            let trigger = "<tool_call>".to_string();
            if !seen_triggers.contains_key(&trigger) {
                seen_triggers.insert(trigger.clone(), true);
                triggers.push(trigger);
            }
        }

        if tag_entries.is_empty() {
            return Err(GrammarError::NoTools);
        }

        let at_least_one = !use_triggers;
        let stop_after_first = !use_triggers;

        self.compile_structural_tag_raw(&triggers, &tag_entries, at_least_one, stop_after_first)
    }

    /// 2026-09-26: Compile a grammar for bare-JSON tool calls, with no wrapper tag:
    /// `{"name":"NAME","arguments":ARGS}`, with ARGS checked against the tool's schema and
    /// one trigger, `{"name":"`. A tool whose schema cannot be sanitized, or has neither
    /// `properties` nor `type`, is skipped with a warning.
    pub fn compile_bare_json_tool_grammar(
        &mut self,
        tools: &[ToolDefinition],
        use_triggers: bool,
    ) -> Result<CompiledGrammar, GrammarError> {
        if tools.is_empty() {
            return Err(GrammarError::NoTools);
        }

        let mut tag_entries = Vec::with_capacity(tools.len());
        let mut triggers = Vec::new();
        let mut seen_triggers = HashMap::<String, bool>::new();

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
                    tracing::warn!("Skipping tool '{name}' in grammar — schema unsanitizable");
                    continue;
                }
            };
            if raw_schema.get("properties").is_none() && raw_schema.get("type").is_none() {
                tracing::warn!(
                    "Skipping tool '{name}' in grammar — schema has no properties or type"
                );
                continue;
            }
            let schema = enforce_min_length_on_required_strings(&raw_schema);

            let begin = format!(r#"{{"name":"{name}","arguments":"#);
            tag_entries.push(serde_json::json!({
                "type": "tag",
                "begin": begin,
                "content": {"type": "json_schema", "json_schema": schema},
                "end": "}",
            }));
        }

        let trigger = r#"{"name":""#.to_string();
        if !seen_triggers.contains_key(&trigger) {
            seen_triggers.insert(trigger.clone(), true);
            triggers.push(trigger);
        }

        if tag_entries.is_empty() {
            return Err(GrammarError::NoTools);
        }

        let at_least_one = !use_triggers;
        let stop_after_first = !use_triggers;

        self.compile_structural_tag_raw(&triggers, &tag_entries, at_least_one, stop_after_first)
    }
}
