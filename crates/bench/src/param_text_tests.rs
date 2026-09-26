// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for the text ↔ [`ParamValues`] conversions.
//!
//! Owner: bench (parameters).
//! Invariants: none beyond the types.

use super::*;
use crate::params::ParamKind;

fn specs() -> Vec<ParamSpec> {
    vec![
        ParamSpec::new(
            "osl",
            "Output tokens",
            "tokens per reply",
            ParamKind::Int { min: 1, max: 8192 },
            ParamValue::Int(128),
        ),
        ParamSpec::new(
            "isls",
            "Input lengths",
            "prompt lengths to sweep",
            ParamKind::IntList { min: 16, max: 4096 },
            ParamValue::IntList(vec![128, 512]),
        ),
        ParamSpec::new(
            "mode",
            "Prompt mode",
            "how prompts are built",
            ParamKind::Choice(&["count", "natural"]),
            ParamValue::Text("count".into()),
        ),
        ParamSpec::new(
            "enabled",
            "Enabled",
            "whether the leg runs",
            ParamKind::Bool,
            ParamValue::Bool(true),
        ),
        ParamSpec::new(
            "ratio",
            "Ratio",
            "fraction to sample",
            ParamKind::Float { min: 0.0, max: 1.0 },
            ParamValue::Float(0.5),
        ),
        ParamSpec::new(
            "label",
            "Label",
            "record label",
            ParamKind::Text,
            ParamValue::Text("baseline".into()),
        ),
    ]
}

#[test]
fn to_strings_writes_every_key_not_just_the_edited_ones() {
    // 2026-09-26: A record describes the whole run, so a reader can tell a
    // default from an omission and a re-run does not pick up today's
    // defaults.
    let s = specs();
    let values = ParamValues::from_overrides(&s, [("osl", "8")]).expect("parses");
    let text = values.to_strings();
    assert_eq!(text.len(), 6, "all six keys, got {text:?}");
    assert_eq!(text["osl"], "8");
    assert_eq!(text["isls"], "128, 512", "untouched default still recorded");
    assert_eq!(text["mode"], "count");
    assert_eq!(text["enabled"], "true");
    assert_eq!(text["ratio"], "0.5");
    assert_eq!(text["label"], "baseline");
}

#[test]
fn text_round_trips_back_to_the_same_values() {
    // 2026-09-26: Render, re-parse, get the same values: the property
    // `RunRecord::values` relies on.
    let s = specs();
    let original = ParamValues::from_overrides(
        &s,
        [
            ("osl", "64"),
            ("isls", "16,32,64"),
            ("enabled", "false"),
            ("ratio", "0.25"),
            ("label", "snapshot-a"),
        ],
    )
    .expect("parses");
    let text = original.to_strings();
    let pairs: Vec<(&str, &str)> = text.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
    let restored = ParamValues::from_overrides(&s, pairs).expect("re-parses");
    assert_eq!(original, restored);
}

#[test]
fn an_unknown_key_is_an_error_that_names_the_valid_ones() {
    let s = specs();
    let err = ParamValues::from_overrides(&s, [("osi", "8")]).expect_err("rejects the typo");
    let msg = err.to_string();
    assert_eq!(
        msg,
        "unknown parameter \"osi\" — this benchmark takes: osl, isls, mode, enabled, ratio, label"
    );
}

#[test]
fn an_out_of_domain_override_fails_before_the_run_starts() {
    let s = specs();
    let err = ParamValues::from_overrides(&s, [("osl", "999999")]).expect_err("out of range");
    assert!(
        err.to_string().contains("Output tokens"),
        "reports against the label: {err}"
    );
}

#[test]
fn iteration_is_deterministic() {
    let s = specs();
    let v = ParamValues::from_overrides(&s, []).expect("defaults");
    let keys: Vec<&str> = v.iter().map(|(k, _)| k).collect();
    let mut sorted = keys.clone();
    sorted.sort_unstable();
    assert_eq!(keys, sorted, "BTreeMap order, so records diff cleanly");
}
