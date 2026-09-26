// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The qwen3_coder `<parameter=NAME>` key slot: with a closed
//! schema it is an alternation of the schema's property names, so a key
//! such as `parameter` or `write` cannot be generated; otherwise it is any
//! identifier. Also the value rule's close ladder and its two options.
//!
//! Owner: server (grammar) tests.
//! Invariants: none beyond the types.

use super::super::compile_tools::{schema_param_names, xml_param_value_body_ebnf};

#[test]
fn schema_param_names_extracts_properties() {
    let schema = serde_json::json!({
        "type": "object",
        "properties": {"filePath": {"type": "string"}, "content": {"type": "string"}},
        "required": ["content", "filePath"]
    });
    let mut names = schema_param_names(&schema).expect("closed schema must constrain");
    names.sort();
    assert_eq!(names, vec!["content".to_string(), "filePath".to_string()]);
}

#[test]
fn schema_param_names_leaves_open_schemas_unconstrained() {
    // 2026-09-26: `additionalProperties` other than `false` leaves the key
    // unconstrained.
    let open = serde_json::json!({
        "type": "object",
        "properties": {"a": {"type": "string"}},
        "additionalProperties": true
    });
    assert_eq!(schema_param_names(&open), None);
    assert_eq!(
        schema_param_names(&serde_json::json!({"type": "object"})),
        None
    );
    assert_eq!(
        schema_param_names(&serde_json::json!({"type": "object", "properties": {}})),
        None
    );
    let closed = serde_json::json!({
        "type": "object",
        "properties": {"a": {"type": "string"}},
        "additionalProperties": false
    });
    assert_eq!(schema_param_names(&closed), Some(vec!["a".to_string()]));
}

#[test]
fn body_ebnf_constrains_paramname_to_schema_alternation() {
    let names = vec!["filePath".to_string(), "content".to_string()];
    let ebnf = xml_param_value_body_ebnf("</parameter>", Some(&names));
    assert!(
        ebnf.contains(r#"paramname ::= "filePath" | "content""#),
        "constrained rule must be a literal alternation of schema keys:\n{ebnf}"
    );
    assert!(
        !ebnf.contains("[a-zA-Z_] [a-zA-Z_0-9]*"),
        "the permissive identifier rule must be gone when names are known:\n{ebnf}"
    );
}

#[test]
fn body_ebnf_keeps_identifier_rule_without_names() {
    for names in [None, Some(&[][..])] {
        let ebnf = xml_param_value_body_ebnf("</parameter>", names);
        assert!(
            ebnf.contains("paramname ::= [a-zA-Z_] [a-zA-Z_0-9]*"),
            "schema-less path must keep the historical identifier rule:\n{ebnf}"
        );
    }
}

#[test]
fn body_ebnf_first_content_allows_lt_via_close_ladder() {
    let ebnf = xml_param_value_body_ebnf("</parameter>", None);
    assert!(
        ebnf.contains(r#"first_content ::= [^ \t\r\n<=>] | "<" [^/]"#),
        "first_content must allow `<` unless it starts the close tag:\n{ebnf}"
    );
    assert!(
        ebnf.contains(r#""</parameter" [^>]"#),
        "the full close-prefix arm must be present:\n{ebnf}"
    );
}

#[test]
fn p1_opts_empty_value_and_force_close_shapes() {
    use super::super::compile_tools::xml_param_value_body_ebnf_opts;
    let d = xml_param_value_body_ebnf_opts("</parameter>", None, false, false);
    assert!(d.contains("value ::= leading_ws nonempty_value"));
    assert!(d.contains("nonempty_value ::= first_content rest"));
    assert!(d.contains(r#""</parameter" [^>]"#));
    // 2026-09-26: `allow_empty_value` makes the value optional.
    let ev = xml_param_value_body_ebnf_opts("</parameter>", None, true, false);
    assert!(
        ev.contains("value ::= leading_ws nonempty_value?"),
        "empty-value opt-in must make content optional:\n{ev}"
    );
    // 2026-09-26: `force_close` drops the deepest ladder arm, so after
    // `</parameter` only `>` is legal.
    let fc = xml_param_value_body_ebnf_opts("</parameter>", None, false, true);
    assert!(
        !fc.contains(r#""</parameter" [^>]"#),
        "force-close must drop the deepest arm:\n{fc}"
    );
    assert!(
        fc.contains(r#""</paramete" [^r]"#),
        "shallower arms must survive:\n{fc}"
    );
    assert!(!fc.contains(r#"first_content ::= [^ \t\r\n<=>] | "<" [^/] | "</" [^p] | "</p" [^a] | "</pa" [^r] | "</par" [^a] | "</para" [^m] | "</param" [^e] | "</parame" [^t] | "</paramet" [^e] | "</paramete" [^r] | "</parameter" [^>]"#));
}
