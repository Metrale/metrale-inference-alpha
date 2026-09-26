// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: `reasoning_effort` renders of jinja-templates/mistral.jinja through
//! `render_chat`. That template accepts only `none` and `high`, and maps `medium`, the
//! `render_chat` default with thinking on, to `high`.
//!
//! Owner: server (tokenizer) tests.
//! Invariants: none beyond the types.

use super::super::chat_render::{RenderFlags, render_chat};
use super::super::jinja_helpers;
use serde_json::json;

fn render_mistral(flags: RenderFlags<'_>) -> anyhow::Result<String> {
    let raw = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../jinja-templates/mistral.jinja"
    ))
    .expect("bundled Mistral override template must be present");
    let converted = jinja_helpers::convert_python_jinja_to_minijinja(&raw);
    let env = jinja_helpers::build_jinja_env(&converted).expect("template compiles");
    let messages = [json!({"role": "user", "content": "Hello"})];
    render_chat(&env, &messages, None, flags)
}

const SETTINGS_HIGH: &str = r#"[MODEL_SETTINGS]{"reasoning_effort": "high"}[/MODEL_SETTINGS]"#;
const SETTINGS_NONE: &str = r#"[MODEL_SETTINGS]{"reasoning_effort": "none"}[/MODEL_SETTINGS]"#;

#[test]
fn unset_effort_thinking_on_renders_high_settings() {
    let r = render_mistral(RenderFlags {
        enable_thinking: true,
        ..Default::default()
    })
    .unwrap();
    assert!(r.contains(SETTINGS_HIGH), "render:\n{r}");
}

#[test]
fn explicit_medium_maps_to_high_settings() {
    let r = render_mistral(RenderFlags {
        enable_thinking: true,
        reasoning_effort: Some("medium"),
        ..Default::default()
    })
    .unwrap();
    assert!(r.contains(SETTINGS_HIGH), "render:\n{r}");
}

#[test]
fn unset_effort_thinking_off_renders_none_settings() {
    let r = render_mistral(RenderFlags::default()).unwrap();
    assert!(r.contains(SETTINGS_NONE), "render:\n{r}");
}

#[test]
fn unsupported_explicit_tiers_still_raise() {
    for effort in ["low", "xhigh"] {
        let err = render_mistral(RenderFlags {
            enable_thinking: true,
            reasoning_effort: Some(effort),
            ..Default::default()
        })
        .unwrap_err();
        assert!(
            format!("{err:#}").contains("reasoning_effort must be either"),
            "effort={effort}: expected the Mistral validator to fire, got: {err:#}"
        );
    }
}
