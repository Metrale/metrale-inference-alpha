// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: JSON Schema to EBNF conversion. `json_schema_to_ebnf` parses the schema
//! into a `JsonValue`, resolves it into a `SchemaSpec` (`parser*`), and emits the EBNF
//! rules (`converter`, `gen_*`); `json_schema_to_grammar` parses that EBNF into
//! `GrammarData`.
//!
//! `JsonValue` keeps object entries in document order, and `SchemaSpec::properties` is
//! filled in that order. An `allOf` with more than one schema is generated as `any`.
//!
//! Owner: grammar.
//! Invariants: none beyond the types.

mod api;
mod cache_key;
mod converter;
mod error;
mod float_regex;
mod formats;
mod gen_array;
mod gen_composite;
mod gen_object;
mod gen_object_props;
mod gen_object_props_constrained;
mod gen_scalar;
mod indent;
mod json_value;
mod options;
mod parser;
mod parser_collections;
mod parser_composite;
mod range_regex;
mod script;
mod spec;

#[cfg(test)]
mod tests;

pub use api::{
    builtin_json_grammar, builtin_json_grammar_ebnf, deepseek_xml_tool_calling_to_ebnf,
    json_schema_to_ebnf, json_schema_to_grammar, json_value_to_ebnf,
    minimax_xml_tool_calling_to_ebnf, qwen_xml_tool_calling_to_ebnf,
};
pub use error::{SchemaError, SchemaErrorKind, SchemaResult};
pub use float_regex::generate_float_range_regex;
pub use json_value::JsonValue;
pub use options::{JsonFormat, SchemaConverterOptions};
pub use range_regex::generate_range_regex;
