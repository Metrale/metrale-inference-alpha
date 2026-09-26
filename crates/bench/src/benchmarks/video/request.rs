// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Request bodies for the video legs, and the decoder-unavailable
//! predicate.
//!
//! Owner: bench, video.
//! Invariants: none beyond the types.

use base64::Engine;
use serde_json::{Value, json};

/// 2026-09-26: The question the ordered-color legs ask. It asks only for the
/// color names in order, so `score::order_matches` can score the reply.
pub const ORDER_PROMPT: &str = "This video is a sequence of solid background colors. List the colors in the order they \
     appear, separated by commas. Answer with only the color names.";

/// 2026-09-26: `bytes` as a base64 `data:` URI with type `mime`.
pub fn data_uri(mime: &str, bytes: &[u8]) -> String {
    let mut s = format!("data:{mime};base64,");
    base64::engine::general_purpose::STANDARD.encode_string(bytes, &mut s);
    s
}

/// 2026-09-26: One streamed request carrying a single video, then `prompt`.
///
/// Temperature 0 and thinking off, as in the vision benchmark's
/// `request::body`: the assertions are about what the model saw, and a
/// reasoning block can use the whole token budget and leave empty content.
pub fn video_body(model: &str, mime: &str, bytes: &[u8], prompt: &str, max_tokens: usize) -> Value {
    json!({
        "model": model,
        "stream": true,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": false},
        "messages": [{"role": "user", "content": [
            {"type": "video_url", "video_url": {"url": data_uri(mime, bytes)}},
            {"type": "text", "text": prompt},
        ]}],
    })
}

/// 2026-09-26: One request carrying an image, then a video, then `prompt`. The
/// image part is labelled `image/png` whatever its bytes; the server picks the
/// image decoder from the bytes (`model-layers/src/vision_preprocess.rs`).
pub fn mixed_body(
    model: &str,
    png: &[u8],
    mime: &str,
    video: &[u8],
    prompt: &str,
    max_tokens: usize,
) -> Value {
    json!({
        "model": model,
        "stream": true,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": false},
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": data_uri("image/png", png)}},
            {"type": "video_url", "video_url": {"url": data_uri(mime, video)}},
            {"type": "text", "text": prompt},
        ]}],
    })
}

/// 2026-09-26: The control: `prompt` alone, as a plain string `content`.
pub fn text_only_body(model: &str, prompt: &str, max_tokens: usize) -> Value {
    json!({
        "model": model,
        "stream": true,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": false},
        "messages": [{"role": "user", "content": prompt}],
    })
}

/// 2026-09-26: One request carrying a single image: with [`text_array_body`],
/// the two baselines the pad-arithmetic cell subtracts.
///
/// Both send `content` as an array, like [`video_body`] and [`mixed_body`], so
/// every term of the arithmetic goes through the same content shape.
/// [`text_only_body`] sends a plain string and is not used as a baseline.
pub fn image_body(model: &str, mime: &str, png: &[u8], prompt: &str, max_tokens: usize) -> Value {
    json!({
        "model": model,
        "stream": true,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": false},
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": data_uri(mime, png)}},
            {"type": "text", "text": prompt},
        ]}],
    })
}

/// 2026-09-26: Text only, as a content array. See [`image_body`].
pub fn text_array_body(model: &str, prompt: &str, max_tokens: usize) -> Value {
    json!({
        "model": model,
        "stream": true,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": false},
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
        ]}],
    })
}

/// 2026-09-26: Does this error read as "the server cannot decode that
/// container"? Such a leg is skipped, not failed: ffmpeg decoding is a
/// deployment choice (`--video-allow-ffmpeg`).
///
/// The strings match the server's messages in
/// `model-layers/src/video_decode_ffmpeg.rs` (`decode_frames`,
/// `spawn_failure`, the availability probe). That file's tests pin only the
/// "is ffmpeg installed" hint, so rewording the others there would turn these
/// skips into failures without failing a test.
pub fn is_decoder_unavailable(err: &str) -> bool {
    let e = err.to_lowercase();
    let quoted_binary_unavailable = e
        .split_once(" could not be run:")
        .is_some_and(|(binary, _)| binary.starts_with('"') && binary.ends_with('"'));
    e.contains("subprocess decoding is disabled")
        || (e.contains("could not run ") && e.contains("is ffmpeg installed"))
        || quoted_binary_unavailable
}

#[cfg(test)]
#[path = "request_tests.rs"]
mod request_tests;
