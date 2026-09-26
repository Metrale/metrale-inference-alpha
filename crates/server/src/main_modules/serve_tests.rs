// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Unit tests for `serve.rs`: the vision area bound, the quant
//! label pairs and `--default-chat-template-kwargs` parsing.
//!
//! Owner: server startup (`met serve`).
//! Invariants: none beyond the types.

use super::*;

fn dir_with(files: &[(&str, &str)]) -> tempfile::TempDir {
    let d = tempfile::tempdir().expect("tempdir");
    for (name, body) in files {
        std::fs::write(d.path().join(name), body).expect("write");
    }
    d
}

/// 2026-09-26: Image fields at the top level of `preprocessor_config.json`.
#[test]
fn reads_the_flat_preprocessor_config() {
    let d = dir_with(&[(
        "preprocessor_config.json",
        r#"{"size": {"longest_edge": 16777216, "shortest_edge": 65536}}"#,
    )]);
    assert_eq!(read_preprocessor_max_pixels(d.path()), Some(16_777_216));
}

/// 2026-09-26: Image fields under `image_processor` in
/// `processor_config.json`.
#[test]
fn reads_the_nested_processor_config() {
    let d = dir_with(&[(
        "processor_config.json",
        r#"{"image_processor": {"size": {"longest_edge": 16777216}}}"#,
    )]);
    assert_eq!(read_preprocessor_max_pixels(d.path()), Some(16_777_216));
}

/// 2026-09-26: A larger `longest_edge` under `video_processor` (1.5x the
/// image bound here) does not replace the `image_processor` bound.
#[test]
fn the_video_bound_never_wins_over_the_image_bound() {
    let d = dir_with(&[(
        "processor_config.json",
        r#"{
            "video_processor": {"size": {"longest_edge": 25165824}, "fps": 2},
            "image_processor": {"size": {"longest_edge": 16777216}}
        }"#,
    )]);
    assert_eq!(read_preprocessor_max_pixels(d.path()), Some(16_777_216));
}

/// 2026-09-26: A config with only a video bound gives no image bound.
#[test]
fn a_video_only_config_yields_no_image_bound() {
    let d = dir_with(&[(
        "processor_config.json",
        r#"{"video_processor": {"size": {"longest_edge": 25165824}}}"#,
    )]);
    assert_eq!(read_preprocessor_max_pixels(d.path()), None);
}

/// 2026-09-26: A top-level `max_pixels` is read as the bound.
#[test]
fn accepts_the_direct_max_pixels_spelling() {
    let d = dir_with(&[("preprocessor_config.json", r#"{"max_pixels": 1048576}"#)]);
    assert_eq!(read_preprocessor_max_pixels(d.path()), Some(1_048_576));
}

/// 2026-09-26: With both files present, `preprocessor_config.json` wins.
#[test]
fn the_dedicated_file_outranks_the_combined_one() {
    let d = dir_with(&[
        ("preprocessor_config.json", r#"{"max_pixels": 1048576}"#),
        (
            "processor_config.json",
            r#"{"image_processor": {"size": {"longest_edge": 16777216}}}"#,
        ),
    ]);
    assert_eq!(read_preprocessor_max_pixels(d.path()), Some(1_048_576));
}

/// 2026-09-26: An absent, unparseable or zero bound gives `None`, not an
/// error.
#[test]
fn unreadable_or_absent_config_falls_back_rather_than_failing() {
    let empty = tempfile::tempdir().expect("tempdir");
    assert_eq!(read_preprocessor_max_pixels(empty.path()), None);

    let broken = dir_with(&[("preprocessor_config.json", "{not json")]);
    assert_eq!(read_preprocessor_max_pixels(broken.path()), None);

    let zero = dir_with(&[("preprocessor_config.json", r#"{"max_pixels": 0}"#)]);
    assert_eq!(read_preprocessor_max_pixels(zero.path()), None);
}

/// 2026-09-26: A source without a usable bound falls through to the next.
#[test]
fn a_broken_first_source_does_not_mask_a_good_second() {
    let d = dir_with(&[
        ("preprocessor_config.json", r#"{"size": {}}"#),
        (
            "processor_config.json",
            r#"{"image_processor": {"size": {"longest_edge": 16777216}}}"#,
        ),
    ]);
    assert_eq!(read_preprocessor_max_pixels(d.path()), Some(16_777_216));
}

#[test]
fn compat_self_pair() {
    assert!(quant_pair_compatible("nvfp4", "nvfp4"));
    assert!(quant_pair_compatible("fp8", "fp8"));
    assert!(quant_pair_compatible("bf16", "bf16"));
}

#[test]
fn compat_nvfp4_handles_fp8_and_bf16() {
    assert!(quant_pair_compatible("nvfp4", "fp8"));
    assert!(quant_pair_compatible("nvfp4", "bf16"));
}

#[test]
fn incompat_unknown_rejected() {
    assert!(!quant_pair_compatible("nvfp4", "gptq-4bit"));
    assert!(!quant_pair_compatible("fp8", "nvfp4"));
}

// 2026-09-26: `--default-chat-template-kwargs`: bad JSON, an unknown key or an
// unknown effort value is an error.

#[test]
fn default_kwargs_reasoning_effort_sets_both_halves() {
    use crate::ir::{EffortLevel, ReasoningEffort, ThinkingDirective};
    let kw = parse_default_chat_template_kwargs(r#"{"reasoning_effort":"xhigh"}"#).unwrap();
    // 2026-09-26: One parse sets both the template's `reasoning_effort` and
    // the thinking directive.
    assert_eq!(kw.reasoning_effort, Some(ReasoningEffort::Max));
    assert_eq!(kw.thinking, ThinkingDirective::OnEffort(EffortLevel::XHigh));
    assert_eq!(kw.preserve_thinking, None);

    let kw = parse_default_chat_template_kwargs(r#"{"reasoning_effort":"none"}"#).unwrap();
    assert_eq!(kw.reasoning_effort, None);
    assert_eq!(kw.thinking, ThinkingDirective::Off);
}

#[test]
fn default_kwargs_explicit_thinking_keys_outrank_effort_directive() {
    use crate::ir::{ReasoningEffort, ThinkingDirective};
    let kw =
        parse_default_chat_template_kwargs(r#"{"thinking_budget":512,"reasoning_effort":"low"}"#)
            .unwrap();
    // 2026-09-26: The budget sets the directive; the effort still sets the
    // template's `reasoning_effort`.
    assert_eq!(kw.thinking, ThinkingDirective::On { budget: Some(512) });
    assert_eq!(kw.reasoning_effort, Some(ReasoningEffort::Low));
}

#[test]
fn default_kwargs_legacy_shapes_still_parse() {
    use crate::ir::ThinkingDirective;
    let kw = parse_default_chat_template_kwargs(r#"{"enable_thinking":true}"#).unwrap();
    assert_eq!(kw.thinking, ThinkingDirective::On { budget: None });
    let kw = parse_default_chat_template_kwargs(r#"{"enable_thinking":false}"#).unwrap();
    assert_eq!(kw.thinking, ThinkingDirective::Off);
    let kw = parse_default_chat_template_kwargs("").unwrap();
    assert_eq!(kw, DefaultChatTemplateKwargs::default());
    let kw = parse_default_chat_template_kwargs(r#"{"preserve_thinking":false}"#).unwrap();
    assert_eq!(kw.preserve_thinking, Some(false));
}

#[test]
fn default_kwargs_fail_fast_on_typos() {
    assert!(
        parse_default_chat_template_kwargs(r#"{"reasoning_effort":"hgih"}"#)
            .unwrap_err()
            .to_string()
            .contains("hgih")
    );
    // 2026-09-26: An unknown key (`deny_unknown_fields`). The plural stands
    // in for a misspelling, which the typos check would reject.
    assert!(parse_default_chat_template_kwargs(r#"{"reasoning_efforts":"low"}"#).is_err());
    assert!(parse_default_chat_template_kwargs("not json").is_err());
}

#[test]
fn official_k3_mxfp4_is_distinct_from_nvfp4_and_bf16() {
    let config = metrale_config::parse_config(include_str!(
        "../../../../docs/k3/fixtures/moonshotai-Kimi-K3-config.json"
    ))
    .unwrap();
    assert_eq!(canonicalize_model_quant(&config), "mxfp4");
    assert!(quant_pair_compatible("mxfp4", "mxfp4"));
    assert!(!quant_pair_compatible("bf16", "mxfp4"));
    assert!(!quant_pair_compatible("nvfp4", "mxfp4"));
}
