// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for [`super`] (message preprocessing): unit tests of each rewrite, and
//! renders of a Holo model template fixture after `preprocess_for_render`.
//!
//! Owner: server (tokenizer) tests.
//! Invariants: none beyond the types.

use super::*;
use serde_json::json;

#[test]
fn autoclose_inserts_before_tool_call() {
    let content = "<think>reasoning here<tool_call>\n<function=foo>";
    let fixed = autoclose_think_before_tool_call(content);
    assert_eq!(
        fixed,
        "<think>reasoning here</think><tool_call>\n<function=foo>"
    );
}

#[test]
fn autoclose_noop_when_already_closed() {
    let content = "<think>reasoning</think>\nanswer<tool_call>\n<function=foo>";
    let fixed = autoclose_think_before_tool_call(content);
    assert_eq!(fixed, content, "already-closed think must be untouched");
}

#[test]
fn autoclose_noop_without_tool_call() {
    let content = "<think>still thinking";
    assert_eq!(autoclose_think_before_tool_call(content), content);
}

#[test]
fn autoclose_noop_without_think() {
    let content = "plain answer<tool_call>\n<function=foo>";
    assert_eq!(autoclose_think_before_tool_call(content), content);
}

#[test]
fn autoclose_assistant_only() {
    let mut messages = vec![
        json!({"role": "user", "content": "<think>x<tool_call>"}),
        json!({"role": "assistant", "content": "<think>x<tool_call>\n<function=f>"}),
    ];
    autoclose_assistant_think(&mut messages);
    assert_eq!(messages[0]["content"], "<think>x<tool_call>");
    assert_eq!(
        messages[1]["content"],
        "<think>x</think><tool_call>\n<function=f>"
    );
}

#[test]
fn think_control_off_disables() {
    let messages = vec![json!({"role": "system", "content": "Be terse <|think_off|>"})];
    let (out, effective) = resolve_think_control(&messages);
    assert_eq!(effective, Some(false));
    assert_eq!(out[0]["content"], "Be terse ");
}

#[test]
fn think_control_on_enables() {
    let messages = vec![json!({"role": "user", "content": "<|think_on|>solve this"})];
    let (out, effective) = resolve_think_control(&messages);
    assert_eq!(effective, Some(true));
    assert_eq!(out[0]["content"], "solve this");
}

#[test]
fn think_control_last_wins_across_messages() {
    let messages = vec![
        json!({"role": "system", "content": "<|think_on|>"}),
        json!({"role": "user", "content": "now <|think_off|> please"}),
    ];
    let (_out, effective) = resolve_think_control(&messages);
    assert_eq!(
        effective,
        Some(false),
        "last control token across messages wins"
    );
}

#[test]
fn think_control_absent_returns_none() {
    let messages = vec![json!({"role": "user", "content": "hello"})];
    let (out, effective) = resolve_think_control(&messages);
    assert_eq!(effective, None);
    assert_eq!(out, messages);
}

#[test]
fn think_control_strips_in_array_content() {
    let messages = vec![json!({
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "describe <|think_off|>"}
        ]
    })];
    let (out, effective) = resolve_think_control(&messages);
    assert_eq!(effective, Some(false));
    assert_eq!(out[0]["content"][1]["text"], "describe ");
}

#[test]
fn remap_developer_only_to_system() {
    let out = remap_developer_role(vec![
        json!({"role": "developer", "content": "You are terse."}),
        json!({"role": "user", "content": "hi"}),
    ]);
    assert_eq!(out.len(), 2);
    assert_eq!(out[0]["role"], "system", "developer → system");
    assert_eq!(out[0]["content"], "You are terse.", "content untouched");
    assert_eq!(out[1]["role"], "user");
}

#[test]
fn remap_developer_plus_system_coalesces() {
    let out = remap_developer_role(vec![
        json!({"role": "developer", "content": "Be terse."}),
        json!({"role": "system", "content": "You are helpful."}),
        json!({"role": "user", "content": "hi"}),
    ]);
    let systems = out.iter().filter(|m| m["role"] == "system").count();
    assert_eq!(systems, 1, "exactly one system message after coalesce");
    assert_eq!(out[0]["role"], "system");
    assert_eq!(out[0]["content"], "Be terse.\n\nYou are helpful.");
    assert_eq!(out[1]["role"], "user", "non-system messages preserved");
    assert_eq!(out.len(), 2);
}

#[test]
fn remap_noop_without_developer() {
    let messages = vec![
        json!({"role": "system", "content": "s"}),
        json!({"role": "user", "content": "u"}),
    ];
    assert_eq!(
        remap_developer_role(messages.clone()),
        messages,
        "no developer role → untouched"
    );
}

// 2026-09-26: The fixture below contains none of `<|think_on|>`, `rfind` or
// `reasoning_effort`, so the behaviours the following tests check come from
// `preprocess_for_render`, not from the template.

const HOLO_MODEL_TEMPLATE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/holo3_1_moe.model_template.jinja"
);

/// 2026-09-26: Runs `chat_impl::preprocess_for_render` itself: tool-argument parsing,
/// developer remap, think autoclose, then think control.
fn preprocess(messages: &[Value], enable_thinking: bool) -> (Vec<Value>, bool) {
    super::super::chat_impl::preprocess_for_render(messages, enable_thinking)
}

fn render_holo_model_template(messages: &[Value], enable_thinking: bool) -> String {
    let raw = std::fs::read_to_string(HOLO_MODEL_TEMPLATE)
        .expect("bundled Holo model template fixture must be present");
    // 2026-09-26: The same conversion `load_config_template` applies to a model's own
    // template.
    let converted = super::super::jinja_helpers::convert_python_jinja_to_minijinja(&raw);
    let env = super::super::jinja_helpers::build_jinja_env(&converted).expect("template compiles");
    let tmpl = env.get_template("chat").unwrap();
    let (prepared, effective) = preprocess(messages, enable_thinking);
    let reasoning_effort: minijinja::Value = if effective { "high" } else { "none" }.into();
    let ctx = minijinja::context! {
        messages => minijinja::Value::from_serialize(&prepared),
        tools => minijinja::Value::UNDEFINED,
        add_generation_prompt => true,
        enable_thinking => effective,
        reasoning_effort => reasoning_effort,
        disable_tool_steering => false,
        add_vision_id => false,
    };
    tmpl.render(ctx).expect("Holo model template renders")
}

#[test]
fn holo_renders_off_model_template() {
    let messages = vec![json!({"role": "user", "content": "Hi"})];
    let rendered = render_holo_model_template(&messages, true);
    assert!(
        rendered.contains("<|im_start|>user\nHi<|im_end|>"),
        "expected Holo model-template framing: {rendered}"
    );
    assert!(
        rendered.ends_with("<|im_start|>assistant\n<think>\n"),
        "expected open-think generation prompt: {rendered}"
    );
}

/// 2026-09-26: The fixture raises `Unexpected message role.` on `developer`; after
/// `preprocess_for_render` the developer message renders as a system turn.
#[test]
fn holo_renders_developer_message_as_system() {
    let messages = vec![
        json!({"role": "developer", "content": "You are a terse assistant."}),
        json!({"role": "user", "content": "Hi"}),
    ];
    let rendered = render_holo_model_template(&messages, true);
    assert!(
        rendered.contains("<|im_start|>system\nYou are a terse assistant.<|im_end|>"),
        "developer content must render as a system turn: {rendered}"
    );
    assert!(
        rendered.contains("<|im_start|>user\nHi<|im_end|>"),
        "user turn still renders: {rendered}"
    );
}

/// 2026-09-26: Developer plus system in one request. Renaming alone would leave two system
/// messages, and the fixture raises `System message must be at the beginning.` on the
/// second; the merged system message renders once with both instructions.
#[test]
fn holo_renders_developer_plus_system_coalesced() {
    let messages = vec![
        json!({"role": "developer", "content": "Always end replies with Done."}),
        json!({"role": "system", "content": "You are helpful."}),
        json!({"role": "user", "content": "Hi"}),
    ];
    let rendered = render_holo_model_template(&messages, true);
    assert_eq!(
        rendered.matches("<|im_start|>system\n").count(),
        1,
        "exactly one system block: {rendered}"
    );
    assert!(
        rendered.contains("Always end replies with Done.") && rendered.contains("You are helpful."),
        "both developer and system instructions must survive: {rendered}"
    );
}

/// 2026-09-26: An assistant-history turn that opens `<think>` and emits `<tool_call>`
/// without closing it. The fixture separates reasoning from content only when
/// `'</think>' in content`, and leaves the reasoning of a turn before the last user message
/// out of the prompt. The test renders the turn with and without the autoclose: only the
/// unrepaired render keeps the reasoning text.
#[test]
fn holo_autocloses_think_before_tool_call() {
    let messages = vec![
        json!({"role": "user", "content": "List /tmp"}),
        json!({
            "role": "assistant",
            "content": "<think>I should list it<tool_call>\n<function=bash>\n<parameter=cmd>\nls\n</parameter>\n</function>\n</tool_call>"
        }),
        json!({"role": "user", "content": "thanks"}),
    ];
    let with_fix = render_holo_model_template(&messages, true);

    assert!(
        !with_fix.contains("I should list it"),
        "autoclose must let the template separate reasoning from content: {with_fix}"
    );
    assert!(
        !with_fix.contains("<think>I should list it"),
        "dangling <think> must not leak into the prompt: {with_fix}"
    );
    assert!(
        with_fix.contains("<tool_call>") && with_fix.contains("<function=bash>"),
        "tool call must survive the repair: {with_fix}"
    );

    let raw = std::fs::read_to_string(HOLO_MODEL_TEMPLATE).unwrap();
    let converted = super::super::jinja_helpers::convert_python_jinja_to_minijinja(&raw);
    let env = super::super::jinja_helpers::build_jinja_env(&converted).unwrap();
    let tmpl = env.get_template("chat").unwrap();
    let unrepaired = tmpl
        .render(minijinja::context! {
            messages => minijinja::Value::from_serialize(&messages),
            tools => minijinja::Value::UNDEFINED,
            add_generation_prompt => true,
            enable_thinking => true,
            reasoning_effort => "high",
            disable_tool_steering => false,
            add_vision_id => false,
        })
        .unwrap();
    assert!(
        unrepaired.contains("I should list it"),
        "sanity: without autoclose the reasoning DOES leak: {unrepaired}"
    );
}

#[test]
fn holo_think_off_control_closes_generation_prompt() {
    let messages = vec![json!({"role": "user", "content": "Answer fast <|think_off|>"})];
    let rendered = render_holo_model_template(&messages, true);
    assert!(
        !rendered.contains("<|think_off|>"),
        "control token must be stripped from prompt: {rendered}"
    );
    assert!(
        rendered.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"),
        "think_off must yield a CLOSED-think generation prompt: {rendered}"
    );
}

#[test]
fn holo_think_on_control_opens_generation_prompt() {
    let messages = vec![json!({"role": "user", "content": "<|think_on|>reason it out"})];
    let rendered = render_holo_model_template(&messages, false);
    assert!(!rendered.contains("<|think_on|>"));
    assert!(
        rendered.ends_with("<|im_start|>assistant\n<think>\n"),
        "think_on must yield an OPEN-think generation prompt: {rendered}"
    );
}
