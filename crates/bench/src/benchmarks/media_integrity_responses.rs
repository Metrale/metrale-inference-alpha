// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The Responses-API and thinking-on parity cells of
//! `media_integrity` (its `responses` submodule).
//!
//! Owner: bench, media integrity.
//! Invariants: a cell whose request failed never returns `Cell::Pass`.

use super::*;

/// 2026-09-26: A blocking Responses-API request carrying one image.
pub fn responses_image_request(model: &str, mime: &str, bytes: &[u8], prompt: &str) -> Value {
    use base64::Engine;
    let mut uri = format!("data:{mime};base64,");
    base64::engine::general_purpose::STANDARD.encode_string(bytes, &mut uri);
    json!({
        "model": model,
        "stream": false,
        "temperature": 0.0,
        // 2026-09-26: `reasoning.effort` is this surface's thinking control:
        // `openai/responses_lowering.rs` passes `reasoning` through and sets
        // `chat_template_kwargs: None`. The budget is generous because with
        // thinking on the reply can arrive as reasoning with an empty
        // `output_text`, and a truncated reply would read as a vision failure.
        "max_output_tokens": 600,
        "reasoning": {"effort": "low"},
        "input": [{"role": "user", "content": [
            {"type": "input_image", "image_url": {"url": uri}},
            {"type": "input_text", "text": prompt},
        ]}],
    })
}

/// 2026-09-26: The Responses API sees the image, and sizes it the same.
///
/// `/v1/responses` has its own content vocabulary (`input_image`,
/// `input_text`) and its own lowering into the IR
/// (`openai/responses_lowering.rs`).
///
/// The assertion is a difference inside this one surface, not a comparison
/// with chat completions: the chat-completions legs turn thinking off, this
/// request asks for `reasoning.effort` "low", so their prompts differ in
/// envelope as well as image. Two images whose vision-token counts differ by
/// a known amount both go through Responses; the envelope is the same twice,
/// so it cancels:
///
/// ```text
///   (tokens_b - tokens_a)  ==  (vision_b - vision_a)
/// ```
///
/// Both must also answer correctly, which shows the pixels arrived rather
/// than merely being counted. A 404 on the first request is a skip.
pub async fn responses_parity(
    handle: &PluginHandle,
    smaller: Value,
    larger: Value,
    expect_delta: usize,
    want: &(dyn Fn(&str) -> bool + Sync),
    timeout: Duration,
) -> Cell {
    const ID: &str = "responses-api-parity";
    let a = match http::responses_blocking(handle.target(), &smaller, timeout).await {
        Ok(o) => o,
        Err(e) => {
            let msg = crate::benchmarks::one_line(format!("{e:#}"));
            return if msg.contains("404") {
                Cell::Skipped { id: ID, why: msg }
            } else {
                Cell::Error { id: ID, msg }
            };
        }
    };
    let b = match http::responses_blocking(handle.target(), &larger, timeout).await {
        Ok(o) => o,
        Err(e) => {
            return Cell::Error {
                id: ID,
                msg: crate::benchmarks::one_line(format!("{e:#}")),
            };
        }
    };
    let delta = b.prompt_tokens.abs_diff(a.prompt_tokens);
    if delta != expect_delta {
        return Cell::Fail {
            id: ID,
            detail: format!(
                "via /v1/responses the two images differ by {delta} prompt tokens, expected \
                 {expect_delta} ({} and {}) — the image is sized differently on this surface",
                a.prompt_tokens, b.prompt_tokens
            ),
        };
    }
    let a_text = a.choices.first().cloned().unwrap_or_default();
    let b_text = b.choices.first().cloned().unwrap_or_default();
    if !want(a_text.trim()) || !want(b_text.trim()) {
        return Cell::Fail {
            id: ID,
            detail: format!(
                "sizing is right but the surface did not answer about the image: \"{}\"",
                crate::benchmarks::one_line(a_text.chars().take(70).collect::<String>())
            ),
        };
    }
    Cell::Pass {
        id: ID,
        detail: format!(
            "both images answered, and the image contributes exactly {expect_delta} tokens \
             ({} vs {})",
            a.prompt_tokens, b.prompt_tokens
        ),
    }
}

/// 2026-09-26: The same image request with thinking on.
pub fn thinking_image_request(model: &str, mime: &str, bytes: &[u8], prompt: &str) -> Value {
    let mut v = image_request(model, mime, bytes, prompt, 600);
    v["chat_template_kwargs"] = json!({"enable_thinking": true});
    // 2026-09-26: Generous on purpose: reasoning spends the budget, and a
    // truncated reply reads as a vision failure when it is only a budget one.
    v["max_tokens"] = json!(600);
    v
}

/// 2026-09-26: Vision with thinking on, through chat completions.
///
/// The chat-completions legs built by `image_request` turn thinking off,
/// because a reasoning block can consume the whole of `max_tokens` and leave
/// empty content. This leg covers thinking on. The assertion is a difference,
/// so the unknown thinking overhead cancels: two images whose vision-token
/// counts differ by a known amount are sent, both with thinking on, and the
/// gap between their prompt sizes must equal that amount exactly:
///
/// ```text
///   (tokens_b - tokens_a)  ==  (vision_b - vision_a)
/// ```
///
/// Whatever the thinking envelope costs, it costs the same in both, so it
/// vanishes from the subtraction. A thinking-on path that mis-sized, dropped
/// or double-counted the image moves it.
pub async fn thinking_parity(
    handle: &PluginHandle,
    smaller: Value,
    larger: Value,
    expect_delta: usize,
    want: &(dyn Fn(&str) -> bool + Sync),
    timeout: Duration,
) -> Cell {
    const ID: &str = "thinking-on-vision";
    let a = match http::chat_stream(handle.target(), &smaller, timeout).await {
        Ok(o) => o,
        Err(e) => {
            return Cell::Error {
                id: ID,
                msg: crate::benchmarks::one_line(format!("{e:#}")),
            };
        }
    };
    let b = match http::chat_stream(handle.target(), &larger, timeout).await {
        Ok(o) => o,
        Err(e) => {
            return Cell::Error {
                id: ID,
                msg: crate::benchmarks::one_line(format!("{e:#}")),
            };
        }
    };
    let delta = b.prompt_tokens.abs_diff(a.prompt_tokens);
    if delta != expect_delta {
        return Cell::Fail {
            id: ID,
            detail: format!(
                "with thinking ON the two images differ by {delta} prompt tokens, expected \
                 {expect_delta} ({} and {} total) — the image's contribution changes when \
                 thinking is enabled",
                a.prompt_tokens, b.prompt_tokens
            ),
        };
    }
    // 2026-09-26: The model must still answer. An empty reply is reported as
    // the reasoning block consuming the budget, not as a vision fault.
    let answered = want(a.text.trim()) && want(b.text.trim());
    if !answered {
        let empty = a.text.trim().is_empty() || b.text.trim().is_empty();
        return Cell::Fail {
            id: ID,
            detail: if empty {
                "geometry is right but a reply came back EMPTY with thinking on — the \
                 reasoning block consumed the whole token budget"
                    .to_string()
            } else {
                format!(
                    "geometry is right but the answer is wrong with thinking on: \"{}\"",
                    crate::benchmarks::one_line(a.text.chars().take(60).collect::<String>())
                )
            },
        };
    }
    Cell::Pass {
        id: ID,
        detail: format!(
            "thinking on: both answered, and the image still contributes exactly \
             {expect_delta} tokens ({} vs {})",
            a.prompt_tokens, b.prompt_tokens
        ),
    }
}
