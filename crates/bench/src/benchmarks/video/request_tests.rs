// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for request shaping and the skip predicate.
//!
//! Owner: bench, video.
//! Invariants: none beyond the types.

use super::*;

#[test]
fn a_video_request_carries_the_clip_then_the_prompt() {
    let b = video_body("m", "video/mp4", b"\x00\x00\x00\x20ftyp", "go", 32);
    assert_eq!(
        b,
        serde_json::json!({
            "model": "m",
            "stream": true,
            "temperature": 0.0,
            "max_tokens": 32,
            "chat_template_kwargs": {"enable_thinking": false},
            "messages": [{"role": "user", "content": [
                {"type": "video_url", "video_url": {
                    "url": "data:video/mp4;base64,AAAAIGZ0eXA="
                }},
                {"type": "text", "text": "go"}
            ]}]
        })
    );
}

/// 2026-09-26: The image comes before the video, the order the Mixed leg
/// sends.
#[test]
fn a_mixed_request_puts_the_image_before_the_video() {
    let b = mixed_body("m", b"png", "image/gif", b"GIF89a", "go", 32);
    assert_eq!(
        b,
        serde_json::json!({
            "model": "m",
            "stream": true,
            "temperature": 0.0,
            "max_tokens": 32,
            "chat_template_kwargs": {"enable_thinking": false},
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {
                    "url": "data:image/png;base64,cG5n"
                }},
                {"type": "video_url", "video_url": {
                    "url": "data:image/gif;base64,R0lGODlh"
                }},
                {"type": "text", "text": "go"}
            ]}]
        })
    );
}

#[test]
fn the_control_carries_no_media_at_all() {
    let b = text_only_body("m", "go", 32);
    assert_eq!(
        b,
        serde_json::json!({
            "model": "m",
            "stream": true,
            "temperature": 0.0,
            "max_tokens": 32,
            "chat_template_kwargs": {"enable_thinking": false},
            "messages": [{"role": "user", "content": "go"}]
        }),
        "the no-media control must differ only in message content shape"
    );
}

/// 2026-09-26: The server's decoder-unavailable messages, copied from
/// `model-layers/src/video_decode_ffmpeg.rs`, are recognised as skips. The
/// copies are literals, so this test does not fail when the server's wording
/// changes.
#[test]
fn a_missing_decoder_is_recognized_as_a_skip() {
    for msg in [
        "this container needs ffmpeg to decode and subprocess decoding is disabled; \
         pass --video-allow-ffmpeg to enable it, or send an animated GIF",
        "could not run \"ffmpeg\" — is ffmpeg installed and on PATH? \
         (set --video-ffmpeg-path to point at it)",
        "\"/nonexistent/ffmpeg\" could not be run: No such file or directory",
    ] {
        assert!(is_decoder_unavailable(msg), "should skip: {msg}");
    }
}

/// 2026-09-26: A real decode failure is not mistaken for a missing decoder,
/// which would turn a defect into a skip.
#[test]
fn a_real_decode_failure_is_not_a_skip() {
    for msg in [
        "decoder failed: Invalid data found when processing input",
        "the container decoded to zero frames (is there a video stream?)",
        "decoded output exceeded the 1024-byte cap",
        "video has 1 usable frame(s) but temporal_patch_size is 2",
        "request could not be run: endpoint timed out",
        "video failed after --video-allow-ffmpeg was enabled",
    ] {
        assert!(!is_decoder_unavailable(msg), "should NOT skip: {msg}");
    }
}

#[test]
fn the_order_prompt_asks_for_a_scoreable_answer() {
    assert_eq!(
        ORDER_PROMPT,
        "This video is a sequence of solid background colors. List the colors in the order they \
         appear, separated by commas. Answer with only the color names."
    );
}
