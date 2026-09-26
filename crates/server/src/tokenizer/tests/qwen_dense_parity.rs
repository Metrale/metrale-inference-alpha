// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Render parity between the retired `qwen3_5` override
//! (`retired-qwen3_5-override-2026-04.jinja`, see jinja-templates/README.md) and three dense
//! Qwen checkpoint templates in test_data/chat_templates: `qwen3.6-27b-unsloth.jinja`,
//! `qwen3.6-27b-official.jinja` (md5 52b6d51ae5b203cb67e64b648494dad2) and
//! `qwen3.5-27b-kbenkhaled.jinja` (md5 94f89e03284d911fc65d06422439fd79).
//!
//! The parity tests cover conversation shapes beyond the one fixture conversation of
//! `qwen_dense.rs`; the divergence tests pin the shapes where a checkpoint template renders
//! different bytes than the override, or fails.
//!
//! Owner: server (tokenizer) tests.
//! Invariants: none beyond the types.

use super::qwen_dense::{fixture_messages, fixture_tools, render_fixture, thinking_on};
use crate::tokenizer::chat_render::RenderFlags;
use serde_json::json;

/// 2026-09-26: The retired override first, then the checkpoint templates compared with it.
const ALL_TEMPLATES: [&str; 4] = [
    "retired-qwen3_5-override-2026-04",
    "qwen3.6-27b-unsloth",
    "qwen3.6-27b-official",
    "qwen3.5-27b-kbenkhaled",
];

/// 2026-09-26: Render `messages` with every template, assert each render equals the retired
/// override's, and return that render.
fn assert_fleet_parity(
    label: &str,
    messages: &[serde_json::Value],
    tools: Option<&[serde_json::Value]>,
    flags: RenderFlags<'_>,
) -> String {
    let reference = render_fixture(ALL_TEMPLATES[0], messages, tools, flags)
        .unwrap_or_else(|e| panic!("{label}: retired override failed to render: {e:#}"));
    for fixture in &ALL_TEMPLATES[1..] {
        let got = render_fixture(fixture, messages, tools, flags)
            .unwrap_or_else(|e| panic!("{label}: {fixture} failed to render: {e:#}"));
        assert_eq!(got, reference, "{label}: {fixture} diverged from override");
    }
    reference
}

#[test]
fn fleet_parity_on_the_baseline_fixture_conversation() {
    let msgs = fixture_messages();
    let tools = fixture_tools();
    for (label, tools, flags) in [
        ("tools+thinking", Some(&tools[..]), thinking_on()),
        ("tools+nothink", Some(&tools[..]), RenderFlags::default()),
        ("plain+thinking", None, thinking_on()),
    ] {
        assert_fleet_parity(label, &msgs, tools, flags);
    }
}

#[test]
fn fleet_parity_assistant_content_and_tool_calls_together() {
    // 2026-09-26: Content and tool calls in one assistant turn take the templates'
    // `content|trim` branch, which writes `\n\n` before the first call.
    let mut msgs = fixture_messages();
    msgs[2]["content"] = json!("Let me check the live weather for you.");
    let r = assert_fleet_parity(
        "content+tool_calls",
        &msgs,
        Some(&fixture_tools()),
        thinking_on(),
    );
    assert!(
        r.contains("Let me check the live weather for you.\n\n<tool_call>\n"),
        "content must precede the call with a blank line:\n{r}"
    );
}

#[test]
fn fleet_parity_multiple_tool_calls_and_consecutive_results() {
    // 2026-09-26: Two calls in one turn, answered by two consecutive `tool` messages: the
    // separator for a call that is not `loop.first`, and the `previtem`/`nextitem` checks
    // that put both `<tool_response>` blocks in one user turn.
    let msgs = vec![
        json!({"role": "system", "content": "You are a helpful assistant."}),
        json!({"role": "user", "content": "Weather in Paris and Lyon?"}),
        json!({
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "get_weather", "arguments": {"location": "Paris"}}},
                {"id": "c2", "type": "function",
                 "function": {"name": "get_weather", "arguments": {"location": "Lyon"}}}
            ]
        }),
        json!({"role": "tool", "content": "18C, sunny"}),
        json!({"role": "tool", "content": "16C, cloudy"}),
        json!({"role": "user", "content": "Summarize."}),
    ];
    let r = assert_fleet_parity(
        "two calls, two results",
        &msgs,
        Some(&fixture_tools()),
        thinking_on(),
    );
    assert!(
        r.contains("</tool_call>\n<tool_call>\n<function=get_weather>\n<parameter=location>\nLyon"),
        "second call separated by a single newline:\n{r}"
    );
    assert!(
        r.contains(
            "<|im_start|>user\n<tool_response>\n18C, sunny\n</tool_response>\n<tool_response>\n16C, cloudy\n</tool_response><|im_end|>\n"
        ),
        "consecutive results share one user turn:\n{r}"
    );
}

#[test]
fn fleet_parity_tool_result_as_final_message() {
    // 2026-09-26: The conversation ends on the tool result: the `loop.last` arm of the tool
    // branch, then the generation prompt.
    let msgs = fixture_messages()[..4].to_vec();
    let r = assert_fleet_parity(
        "trailing tool result",
        &msgs,
        Some(&fixture_tools()),
        thinking_on(),
    );
    assert!(
        r.ends_with("</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n"),
        "tool tail closes before the generation prompt:\n{r}"
    );
}

#[test]
fn fleet_parity_inline_think_split_without_reasoning_content() {
    // 2026-09-26: Without `reasoning_content` the templates split the reasoning out of the
    // content with `content.split('</think>')…`, which `convert_python_jinja_to_minijinja`
    // rewrites.
    let msgs = vec![
        json!({"role": "user", "content": "hi"}),
        json!({
            "role": "assistant",
            "content": "<think>\nsome hidden reasoning\n</think>\n\nHello there!"
        }),
        json!({"role": "user", "content": "again?"}),
    ];
    let r = assert_fleet_parity("inline think split", &msgs, None, thinking_on());
    assert!(
        r.contains("<|im_start|>assistant\nHello there!<|im_end|>\n"),
        "historical inline think must be stripped with the content kept:\n{r}"
    );
}

#[test]
fn fleet_parity_null_content_list_content_and_no_system() {
    let msgs = vec![
        json!({"role": "user", "content": [
            {"type": "text", "text": "Check the weather in "},
            {"type": "text", "text": "Paris."}
        ]}),
        json!({
            "role": "assistant",
            "content": serde_json::Value::Null,
            "tool_calls": [{"id": "c1", "type": "function",
                "function": {"name": "get_weather", "arguments": {"location": "Paris"}}}]
        }),
        json!({"role": "tool", "content": "18C"}),
        json!({"role": "user", "content": "  Thanks!  \n\n"}),
    ];
    let r = assert_fleet_parity(
        "null/list content, no system",
        &msgs,
        Some(&fixture_tools()),
        thinking_on(),
    );
    assert!(
        r.contains("<|im_start|>user\nCheck the weather in Paris.<|im_end|>\n"),
        "content parts concatenate:\n{r}"
    );
    assert!(
        r.contains("<|im_start|>user\nThanks!<|im_end|>\n"),
        "user content is trimmed:\n{r}"
    );
}

#[test]
fn fleet_parity_scalar_and_structured_tool_args() {
    // 2026-09-26: The override writes scalar argument values with `|string` and the official
    // template with `|tojson`; for numbers and booleans the two agree.
    let mut msgs = fixture_messages();
    msgs[2]["tool_calls"][0]["function"]["arguments"] = json!({
        "location": "Paris",
        "days": 3,
        "metric": true,
        "filters": {"wind": true},
        "hours": [6, 12, 18]
    });
    let r = assert_fleet_parity(
        "scalar+structured args",
        &msgs,
        Some(&fixture_tools()),
        thinking_on(),
    );
    for expect in [
        "<parameter=days>\n3\n</parameter>\n",
        "<parameter=metric>\ntrue\n</parameter>\n",
        "<parameter=filters>\n{\"wind\":true}\n</parameter>\n",
        "<parameter=hours>\n[6,12,18]\n</parameter>\n",
    ] {
        assert!(r.contains(expect), "missing {expect:?} in:\n{r}");
    }
}

#[test]
fn fleet_parity_empty_tools_slice_matches_absent_tools() {
    // 2026-09-26: `if tools and tools is iterable` treats an empty list as false.
    let msgs = fixture_messages();
    let with_empty = assert_fleet_parity("tools=[]", &msgs, Some(&[]), thinking_on());
    let with_none = assert_fleet_parity("tools absent", &msgs, None, thinking_on());
    assert_eq!(with_empty, with_none, "empty tool list must equal absent");
    assert!(
        !with_empty.contains("# Tools"),
        "no tools header:\n{with_empty}"
    );
}

#[test]
fn fleet_parity_multiple_tools_and_thinking_off() {
    let mut tools = fixture_tools();
    tools.push(json!({
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Current time for a location",
            "parameters": {"type": "object", "properties": {"location": {"type": "string"}},
                           "required": ["location"]}
        }
    }));
    let r = assert_fleet_parity(
        "two tools, thinking off",
        &fixture_messages(),
        Some(&tools),
        RenderFlags::default(),
    );
    assert!(
        r.contains("\"get_weather\"") && r.contains("\"get_time\""),
        "both tools serialized:\n{r}"
    );
    assert!(
        r.ends_with("<think>\n\n</think>\n\n"),
        "closed think tail:\n{r}"
    );
}

#[test]
fn fleet_parity_continue_final_assistant_message() {
    let msgs = vec![
        json!({"role": "user", "content": "Say exactly: banana"}),
        json!({"role": "assistant", "content": "ban"}),
    ];
    let flags = RenderFlags {
        enable_thinking: true,
        allow_continue_final: true,
        ..Default::default()
    };
    let r = assert_fleet_parity("continue-final", &msgs, None, flags);
    // 2026-09-26: The last assistant turn comes after the last user query, so each template
    // writes an empty think block before its content; `render_chat` then strips only the
    // trailing `<|im_end|>`.
    assert!(
        r.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\nban"),
        "prefill tail:\n{r}"
    );
}

// 2026-09-26: Known divergences, where a checkpoint template does not reproduce the retired
// override's bytes: a JSON `null` argument value, tool-call arguments that are still a
// string after `normalize_tool_call_arguments`, and a second system message.

/// 2026-09-26: A JSON `null` argument value: the override, unsloth and Kbenkhaled templates
/// write "none" (`|string`), the official template "null" (`|tojson`).
#[test]
fn divergence_null_arg_value_official_renders_null_not_none() {
    let mut msgs = fixture_messages();
    msgs[2]["tool_calls"][0]["function"]["arguments"] =
        json!({"location": "Paris", "days": serde_json::Value::Null});
    let tools = fixture_tools();
    for (fixture, expect) in [
        (
            "retired-qwen3_5-override-2026-04",
            "<parameter=days>\nnone\n</parameter>\n",
        ),
        (
            "qwen3.6-27b-unsloth",
            "<parameter=days>\nnone\n</parameter>\n",
        ),
        (
            "qwen3.5-27b-kbenkhaled",
            "<parameter=days>\nnone\n</parameter>\n",
        ),
        (
            "qwen3.6-27b-official",
            "<parameter=days>\nnull\n</parameter>\n",
        ),
    ] {
        let r = render_fixture(fixture, &msgs, Some(&tools), thinking_on()).unwrap();
        assert!(r.contains(expect), "{fixture}: wanted {expect:?} in:\n{r}");
    }
}

/// 2026-09-26: Arguments that are not valid JSON stay a string. The override writes the raw
/// string inside the function block, the unsloth template drops it, and the official and
/// Kbenkhaled templates fail the render.
#[test]
fn divergence_unparseable_string_args_raw_vs_dropped_vs_error() {
    let mut msgs = fixture_messages();
    msgs[2]["tool_calls"][0]["function"]["arguments"] = json!("{\"location\": \"Par");
    let tools = fixture_tools();

    let retired = render_fixture(
        "retired-qwen3_5-override-2026-04",
        &msgs,
        Some(&tools),
        thinking_on(),
    )
    .unwrap();
    assert!(
        retired.contains("<function=get_weather>\n{\"location\": \"Par</function>\n"),
        "override emitted the raw string:\n{retired}"
    );

    let unsloth =
        render_fixture("qwen3.6-27b-unsloth", &msgs, Some(&tools), thinking_on()).unwrap();
    assert!(
        unsloth.contains("<function=get_weather>\n</function>\n"),
        "unsloth drops the malformed arguments entirely:\n{unsloth}"
    );

    for fixture in ["qwen3.6-27b-official", "qwen3.5-27b-kbenkhaled"] {
        let err = render_fixture(fixture, &msgs, Some(&tools), thinking_on())
            .expect_err("official-style templates raise on string arguments");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("render"),
            "{fixture}: expected a render error, got: {msg}"
        );
    }
}

/// 2026-09-26: A second system message, with no developer message, so
/// `remap_developer_role` leaves both. The override, official and Kbenkhaled templates
/// fail the render; the unsloth template merges both into one system block.
#[test]
fn divergence_two_system_messages_error_before_merge_on_unsloth_now() {
    let msgs = vec![
        json!({"role": "system", "content": "Rule A."}),
        json!({"role": "system", "content": "Rule B."}),
        json!({"role": "user", "content": "hi"}),
    ];
    for fixture in [
        "retired-qwen3_5-override-2026-04",
        "qwen3.6-27b-official",
        "qwen3.5-27b-kbenkhaled",
    ] {
        render_fixture(fixture, &msgs, None, thinking_on())
            .expect_err("second system message must raise");
    }
    let r = render_fixture("qwen3.6-27b-unsloth", &msgs, None, thinking_on()).unwrap();
    assert!(
        r.contains("<|im_start|>system\nRule A.\nRule B.<|im_end|>\n"),
        "unsloth merges the leading system pair:\n{r}"
    );
}
