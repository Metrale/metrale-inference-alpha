// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Building an image chat request, and reading the vision-token
//! count back out.
//!
//! Owner: bench, vision.
//! Invariants: none beyond the types.

use anyhow::{Context, Result};
use base64::Engine;
use serde_json::{Value, json};

/// 2026-09-26: A fixture as a base64 `data:` URI, labelled `image/png`
/// whatever its bytes; the server picks the decoder from the bytes. The server
/// fetches `http(s)` image URLs only with `--vision-allow-remote-images`, so a
/// data URI works on every serve and keeps the network out of the run.
pub fn data_uri(png: &[u8]) -> String {
    let mut s = String::from("data:image/png;base64,");
    base64::engine::general_purpose::STANDARD.encode_string(png, &mut s);
    s
}

/// 2026-09-26: One streamed chat request carrying `images`, in order, then
/// `prompt`.
///
/// Temperature 0, since the assertions are about what the model saw. Thinking
/// is off: a reasoning block can use the whole of `max_tokens` and leave empty
/// content.
pub fn body(model: &str, images: &[&[u8]], prompt: &str, max_tokens: usize) -> Value {
    let mut content: Vec<Value> = images
        .iter()
        .map(|png| json!({"type": "image_url", "image_url": {"url": data_uri(png)}}))
        .collect();
    content.push(json!({"type": "text", "text": prompt}));
    json!({
        "model": model,
        "stream": true,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": false},
        "messages": [{"role": "user", "content": content}],
    })
}

/// 2026-09-26: Vision tokens in a request: `prompt_tokens` minus the template
/// `overhead`, or an error when `overhead` is larger.
///
/// The driver measures `overhead` once per run, from a calibration request
/// whose vision-token count is known, because it depends on the checkpoint's
/// chat template.
pub fn vision_tokens(prompt_tokens: usize, overhead: usize) -> Result<usize> {
    prompt_tokens.checked_sub(overhead).with_context(|| {
        format!(
            "prompt_tokens {prompt_tokens} is below the measured template overhead \
             {overhead} — the calibration request and this one did not render the same \
             template, so the subtraction is meaningless"
        )
    })
}

#[cfg(test)]
#[path = "request_tests.rs"]
mod request_tests;
