// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Jinja chat rendering shared by the `ChatTokenizer` apply paths.
//!
//! [`render_chat`] is a free function so the render tests in `tokenizer/tests/` call the
//! same context-building code the server uses.
//!
//! Owner: server (tokenizer).
//! Invariants: none beyond the types.

use anyhow::{Context, Result};

use super::chat_impl::preprocess_for_render;

/// 2026-09-26: Render-time flags for [`render_chat`].
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct RenderFlags<'a> {
    pub enable_thinking: bool,
    pub disable_tool_steering: bool,
    /// 2026-09-26: Reasoning-effort string from the request or the server default. `None`
    /// renders `"medium"` when thinking is on and `"none"` when it is off. For `"medium"`
    /// the Qwen3.8 template (test_data/chat_templates/qwen3.8-27b-unsloth.jinja) writes no
    /// effort instruction, jinja-templates/mistral.jinja maps it to `"high"`, and the Qwen3.6
    /// templates in test_data/chat_templates do not read the variable.
    pub reasoning_effort: Option<&'a str>,
    /// 2026-09-26: `Some(_)` sets the template's `preserve_thinking`; `None` leaves it
    /// undefined so the template's own default applies. In test_data/chat_templates the
    /// Qwen3.6 templates drop reasoning from earlier turns unless it is true, and the Qwen3.8
    /// template keeps it unless it is defined and not true. A Jinja `none` would therefore
    /// flip the Qwen3.8 default, which is why `None` is not passed as `none`.
    pub preserve_thinking: Option<bool>,
    /// 2026-09-26: "Continue final message": when true and the last message is an assistant
    /// turn, render without a generation prompt and strip trailing whitespace and then a
    /// trailing `<|im_end|>`, so the prompt ends with the assistant content. The Jinja apply
    /// path sets it true, the OpenAI-variant path false.
    pub allow_continue_final: bool,
}

/// 2026-09-26: Run `preprocess_for_render`, then render the `chat` template of `env`. Both
/// `apply_chat_template_jinja_with_effort` and `apply_chat_template_openai_with_effort`
/// render through here; the DeepSeek-V4 encoding does not.
pub(crate) fn render_chat(
    env: &minijinja::Environment<'static>,
    messages: &[serde_json::Value],
    tools: Option<&[serde_json::Value]>,
    flags: RenderFlags<'_>,
) -> Result<String> {
    let tmpl = env
        .get_template("chat")
        .context("Failed to get compiled template")?;

    let (messages_for_render, enable_thinking) =
        preprocess_for_render(messages, flags.enable_thinking);
    let messages_val = minijinja::Value::from_serialize(&messages_for_render);
    let tools_val = tools.map(minijinja::Value::from_serialize);

    let continue_final = flags.allow_continue_final
        && messages
            .last()
            .and_then(|m| m.get("role"))
            .and_then(|r| r.as_str())
            == Some("assistant");

    // 2026-09-26: `reasoning_effort` is never undefined here, so a template's own default for
    // it never applies (`default('xhigh')` in the Qwen3.8 template, `'high'` in
    // jinja-templates/mistral.jinja, where `"none"` is what turns thinking off). A server
    // default from `--default-chat-template-kwargs` is already in `flags.reasoning_effort`
    // (api/chat/prepare.rs).
    let reasoning_effort: minijinja::Value = if let Some(effort) = flags.reasoning_effort {
        effort.into()
    } else if enable_thinking {
        "medium".into()
    } else {
        "none".into()
    };
    let preserve_thinking = flags
        .preserve_thinking
        .map(minijinja::Value::from)
        .unwrap_or(minijinja::Value::UNDEFINED);
    let ctx = minijinja::context! {
        messages => messages_val,
        tools => tools_val.unwrap_or(minijinja::Value::UNDEFINED),
        add_generation_prompt => !continue_final,
        enable_thinking => enable_thinking,
        reasoning_effort => reasoning_effort,
        preserve_thinking => preserve_thinking,
        disable_tool_steering => flags.disable_tool_steering,
        add_vision_id => false,
    };

    let mut rendered = tmpl.render(ctx).map_err(|e| {
        tracing::error!("Jinja template error: {e:#}");
        anyhow::anyhow!("Failed to render Jinja chat template: {e}")
    })?;

    if continue_final {
        let trimmed = rendered.trim_end();
        let stripped = trimmed.strip_suffix("<|im_end|>").unwrap_or(trimmed);
        rendered = stripped.to_string();
        tracing::info!("continue_final_message: stripped trailing EOT for prefill A/B");
    }

    Ok(rendered)
}
