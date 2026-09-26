// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Render tests for the dense Qwen 27B templates in test_data/chat_templates,
//! through `render_chat` after `convert_python_jinja_to_minijinja`, as the server renders:
//! - `qwen3.6-27b-unsloth.jinja` (md5 a7f294a5f0be5f1903214304f259f87f);
//! - `qwen3.8-27b-unsloth.jinja` (md5 2a79880b328d0e0387c8ecb62c4c0c80);
//! - `retired-qwen3_5-override-2026-04.jinja`, the retired `qwen3_5` override
//!   (jinja-templates/README.md), kept as a parity reference.
//!
//! Variant goldens are built from [`Q36_GOLDEN`] with byte-delta helpers, so a failure
//! shows which bytes moved.
//!
//! Owner: server (tokenizer) tests.
//! Invariants: none beyond the types.

use super::super::chat_render::{RenderFlags, render_chat};
use super::super::jinja_helpers;
use serde_json::json;

pub(crate) fn render_fixture(
    fixture: &str,
    messages: &[serde_json::Value],
    tools: Option<&[serde_json::Value]>,
    flags: RenderFlags<'_>,
) -> anyhow::Result<String> {
    let path = format!(
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/test_data/chat_templates/{}.jinja"
        ),
        fixture
    );
    let raw = std::fs::read_to_string(&path).expect("checkpoint template fixture present");
    let converted = jinja_helpers::convert_python_jinja_to_minijinja(&raw);
    let env = jinja_helpers::build_jinja_env(&converted).expect("template compiles");
    render_chat(&env, messages, tools, flags)
}

/// 2026-09-26: A system prompt, a tool round-trip with `reasoning_content` on the earlier
/// assistant turns, and a new user query: the Qwen3.6 and Qwen3.8 templates differ on
/// whether that earlier reasoning is rendered.
pub(crate) fn fixture_messages() -> Vec<serde_json::Value> {
    vec![
        json!({"role": "system", "content": "You are a helpful assistant."}),
        json!({"role": "user", "content": "Check the weather in Paris, then summarize."}),
        json!({
            "role": "assistant",
            "content": "",
            "reasoning_content": "User wants live weather. I should call get_weather.",
            "tool_calls": [{
                "id": "c1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": {"location": "Paris"}}
            }]
        }),
        json!({"role": "tool", "content": "18C, sunny"}),
        json!({
            "role": "assistant",
            "content": "It is 18C and sunny in Paris.",
            "reasoning_content": "Tool says 18C sunny; summarize."
        }),
        json!({"role": "user", "content": "Thanks - now in one word?"}),
    ]
}

pub(crate) fn fixture_tools() -> Vec<serde_json::Value> {
    vec![json!({
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"]
            }
        }
    })]
}

pub(crate) fn thinking_on() -> RenderFlags<'static> {
    RenderFlags {
        enable_thinking: true,
        ..Default::default()
    }
}

/// 2026-09-26: Qwen3.6 render with thinking on, tools, and `preserve_thinking` unset. The
/// tool JSON is compact (the default `tojson`), and the earlier turns' reasoning is left
/// out, the template's default while `preserve_thinking` is undefined.
const Q36_GOLDEN: &str = "<|im_start|>system\n# Tools\n\nYou have access to the following functions:\n\n<tools>\n{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"description\":\"Get current weather for a location\",\"parameters\":{\"type\":\"object\",\"properties\":{\"location\":{\"type\":\"string\"}},\"required\":[\"location\"]}}}\n</tools>\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>\n\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nCheck the weather in Paris, then summarize.<|im_end|>\n<|im_start|>assistant\n<tool_call>\n<function=get_weather>\n<parameter=location>\nParis\n</parameter>\n</function>\n</tool_call><|im_end|>\n<|im_start|>user\n<tool_response>\n18C, sunny\n</tool_response><|im_end|>\n<|im_start|>assistant\nIt is 18C and sunny in Paris.<|im_end|>\n<|im_start|>user\nThanks - now in one word?<|im_end|>\n<|im_start|>assistant\n<think>\n";

/// 2026-09-26: The Qwen3.8 template's instruction for `xhigh`, which it also writes for
/// `high` because it maps `high` to `xhigh`. The thinking budget still differs: `high` gets
/// twice and `xhigh` four times the model's `max_thinking_budget` (`effort_budget` in
/// api/chat/thinking.rs). A request without an effort renders `"medium"`, which writes no
/// instruction.
const XHIGH_SENTENCE: &str = "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.";
const LOW_SENTENCE: &str = "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.";

/// 2026-09-26: The reasoning blocks the templates write for the two earlier assistant turns.
const THINK1: &str = "<think>\nUser wants live weather. I should call get_weather.\n</think>\n\n";
const THINK2: &str = "<think>\nTool says 18C sunny; summarize.\n</think>\n\n";

/// 2026-09-26: Byte-delta helpers over the baseline, one template behaviour each.
fn with_history_think(base: &str) -> String {
    base.replacen(
        "assistant\n<tool_call>\n<function=get_weather>",
        &format!("assistant\n{THINK1}<tool_call>\n<function=get_weather>"),
        1,
    )
    .replacen(
        "assistant\nIt is 18C",
        &format!("assistant\n{THINK2}It is 18C"),
        1,
    )
}

fn with_effort_sentence(base: &str, sentence: &str) -> String {
    base.replacen(
        "<|im_start|>system\n# Tools",
        &format!("<|im_start|>system\n{sentence}\n\n# Tools"),
        1,
    )
}

fn with_closed_think_tail(base: &str) -> String {
    let open = "<|im_start|>assistant\n<think>\n";
    assert!(base.ends_with(open));
    format!(
        "{}<|im_start|>assistant\n<think>\n\n</think>\n\n",
        base.strip_suffix(open).unwrap()
    )
}

/// 2026-09-26: Qwen3.8 render for an explicit `xhigh` or `high`: the Qwen3.6 render plus the
/// xhigh instruction and the earlier turns' reasoning, which Qwen3.8 keeps by default.
fn q38_golden() -> String {
    with_history_think(&with_effort_sentence(Q36_GOLDEN, XHIGH_SENTENCE))
}

/// 2026-09-26: Qwen3.8 render for a request without an effort: `"medium"` writes no
/// instruction, so it differs from the Qwen3.6 render only by the earlier turns' reasoning.
fn q38_unset_golden() -> String {
    with_history_think(Q36_GOLDEN)
}

#[test]
fn q36_default_strips_history_think() {
    let r = render_fixture(
        "qwen3.6-27b-unsloth",
        &fixture_messages(),
        Some(&fixture_tools()),
        thinking_on(),
    )
    .unwrap();
    assert_eq!(r, Q36_GOLDEN);
}

/// 2026-09-26: The Qwen3.6 checkpoint template and the retired `qwen3_5` override render the
/// same bytes for three flag sets. Both go through the same conversion and preprocessing.
#[test]
fn q36_render_matches_retired_override_bytes() {
    let msgs = fixture_messages();
    let tools = fixture_tools();
    for (name, tools, flags) in [
        ("tools+thinking", Some(&tools[..]), thinking_on()),
        ("tools+nothink", Some(&tools[..]), RenderFlags::default()),
        ("plain+thinking", None, thinking_on()),
    ] {
        let retired =
            render_fixture("retired-qwen3_5-override-2026-04", &msgs, tools, flags).unwrap();
        let own = render_fixture("qwen3.6-27b-unsloth", &msgs, tools, flags).unwrap();
        assert_eq!(own, retired, "parity with retired override broke: {name}");
    }
}

#[test]
fn q36_preserve_thinking_true_rehydrates_history_think() {
    let r = render_fixture(
        "qwen3.6-27b-unsloth",
        &fixture_messages(),
        Some(&fixture_tools()),
        RenderFlags {
            enable_thinking: true,
            preserve_thinking: Some(true),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(r, with_history_think(Q36_GOLDEN));
}

#[test]
fn q36_thinking_off_closes_think_tail_only() {
    let r = render_fixture(
        "qwen3.6-27b-unsloth",
        &fixture_messages(),
        Some(&fixture_tools()),
        RenderFlags::default(),
    )
    .unwrap();
    // 2026-09-26: The Qwen3.6 template does not read `reasoning_effort`, so the `"none"`
    // sent with thinking off changes nothing; only the generation tail differs.
    assert_eq!(r, with_closed_think_tail(Q36_GOLDEN));
}

#[test]
fn q38_default_keeps_history_think_and_injects_no_effort_sentence() {
    let r = render_fixture(
        "qwen3.8-27b-unsloth",
        &fixture_messages(),
        Some(&fixture_tools()),
        thinking_on(),
    )
    .unwrap();
    assert_eq!(r, q38_unset_golden());
}

/// 2026-09-26: A request without an effort renders the same bytes as an explicit
/// `"medium"`. The budget side agrees: both get the model's `max_thinking_budget`.
#[test]
fn q38_unset_equals_explicit_medium() {
    let msgs = fixture_messages();
    let tools = fixture_tools();
    let unset = render_fixture("qwen3.8-27b-unsloth", &msgs, Some(&tools), thinking_on()).unwrap();
    let medium = render_fixture(
        "qwen3.8-27b-unsloth",
        &msgs,
        Some(&tools),
        RenderFlags {
            enable_thinking: true,
            reasoning_effort: Some("medium"),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(unset, medium);
}

#[test]
fn q38_preserve_thinking_false_strips_history_think() {
    let r = render_fixture(
        "qwen3.8-27b-unsloth",
        &fixture_messages(),
        Some(&fixture_tools()),
        RenderFlags {
            enable_thinking: true,
            preserve_thinking: Some(false),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(r, Q36_GOLDEN);
}

#[test]
fn q38_thinking_off_skips_effort_validator_and_sentence() {
    let r = render_fixture(
        "qwen3.8-27b-unsloth",
        &fixture_messages(),
        Some(&fixture_tools()),
        RenderFlags::default(),
    )
    .unwrap();
    // 2026-09-26: With thinking off the Qwen3.8 template skips its whole effort block, so the
    // `"none"` sent then never reaches its validator. The earlier turns' reasoning is still
    // kept: that default does not depend on `enable_thinking`.
    assert_eq!(r, with_closed_think_tail(&with_history_think(Q36_GOLDEN)));
}

#[test]
fn q38_explicit_efforts_render_their_sentences() {
    let msgs = fixture_messages();
    let tools = fixture_tools();
    let flags = |effort: &'static str| RenderFlags {
        enable_thinking: true,
        reasoning_effort: Some(effort),
        ..Default::default()
    };
    // 2026-09-26: "xhigh" is what `ir::ReasoningEffort::Max` renders as.
    let r = render_fixture("qwen3.8-27b-unsloth", &msgs, Some(&tools), flags("xhigh")).unwrap();
    assert_eq!(r, q38_golden());
    let r = render_fixture("qwen3.8-27b-unsloth", &msgs, Some(&tools), flags("high")).unwrap();
    assert_eq!(r, q38_golden());
    let r = render_fixture("qwen3.8-27b-unsloth", &msgs, Some(&tools), flags("low")).unwrap();
    assert_eq!(
        r,
        with_history_think(&with_effort_sentence(Q36_GOLDEN, LOW_SENTENCE))
    );
    let r = render_fixture("qwen3.8-27b-unsloth", &msgs, Some(&tools), flags("medium")).unwrap();
    assert_eq!(r, with_history_think(Q36_GOLDEN));
}

/// 2026-09-26: The Qwen3.8 template raises on the string `"max"`, which is why
/// `ir::ReasoningEffort::Max` renders as `"xhigh"`.
#[test]
fn q38_raw_max_effort_string_raises() {
    let err = render_fixture(
        "qwen3.8-27b-unsloth",
        &fixture_messages(),
        Some(&fixture_tools()),
        RenderFlags {
            enable_thinking: true,
            reasoning_effort: Some("max"),
            ..Default::default()
        },
    )
    .unwrap_err();
    assert!(
        format!("{err:#}").contains("Unexpected reasoning effort"),
        "expected the template's effort validator to fire, got: {err:#}"
    );
}

/// 2026-09-26: The Qwen3.8 template raises on tool-call `arguments` given as a non-empty
/// string. `render_chat` parses them first (`normalize_tool_call_arguments`), so the render
/// equals the one with structured arguments.
#[test]
fn q38_string_tool_args_are_normalized_before_strict_validation() {
    let mut msgs = fixture_messages();
    msgs[2]["tool_calls"][0]["function"]["arguments"] = json!("{\"location\":\"Paris\"}");
    let r = render_fixture(
        "qwen3.8-27b-unsloth",
        &msgs,
        Some(&fixture_tools()),
        thinking_on(),
    )
    .unwrap();
    assert_eq!(r, q38_unset_golden());
}
