// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Wire-parse tests for video content parts, on chat completions
//! and on `/v1/responses`.
//!
//! Owner: server (OpenAI adapter).
//! Invariants: none beyond the types.

use super::*;

fn parse_chat(content: serde_json::Value) -> ParsedContent {
    let msg = serde_json::json!({"role": "user", "content": content});
    let m: IncomingMessage = serde_json::from_value(msg).expect("deserialise");
    m.content
}

/// 2026-09-26: The parsed media as `(kind, uri)` pairs, in order.
fn media(c: &ParsedContent) -> Vec<(MediaKind, &str)> {
    c.media.iter().map(|m| (m.kind, m.uri.as_str())).collect()
}

#[test]
fn chat_completions_carries_a_video_url_object() {
    let c = parse_chat(serde_json::json!([
        {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AAA"}},
        {"type": "text", "text": "what happens?"}
    ]));
    assert_eq!(
        media(&c),
        vec![(MediaKind::Video, "data:video/mp4;base64,AAA")],
        "a video must not be counted as an image"
    );
    assert_eq!(c.text, "what happens?");
}

#[test]
fn chat_completions_carries_the_flat_video_spelling() {
    let c = parse_chat(serde_json::json!([
        {"type": "video", "video": "data:image/gif;base64,BBB"}
    ]));
    assert_eq!(
        media(&c),
        vec![(MediaKind::Video, "data:image/gif;base64,BBB")]
    );
}

#[test]
fn chat_completions_carries_images_and_videos_together() {
    let c = parse_chat(serde_json::json!([
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,III"}},
        {"type": "video_url", "video_url": {"url": "data:image/gif;base64,VVV"}},
        {"type": "text", "text": "compare"}
    ]));
    assert_eq!(
        media(&c),
        vec![
            (MediaKind::Image, "data:image/png;base64,III"),
            (MediaKind::Video, "data:image/gif;base64,VVV"),
        ]
    );
}

#[test]
fn a_video_sent_before_an_image_stays_before_it() {
    let c = parse_chat(serde_json::json!([
        {"type": "video_url", "video_url": {"url": "data:image/gif;base64,VVV"}},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,III"}},
        {"type": "text", "text": "which came first?"}
    ]));
    assert_eq!(
        media(&c),
        vec![
            (MediaKind::Video, "data:image/gif;base64,VVV"),
            (MediaKind::Image, "data:image/png;base64,III"),
        ],
        "the client's order must survive the wire parse"
    );
}

/// 2026-09-26: Three alternating items: grouping by modality, in either
/// order, fails this.
#[test]
fn alternating_media_keeps_every_position() {
    let c = parse_chat(serde_json::json!([
        {"type": "video_url", "video_url": {"url": "v1"}},
        {"type": "image_url", "image_url": {"url": "i1"}},
        {"type": "video_url", "video_url": {"url": "v2"}},
    ]));
    assert_eq!(
        media(&c),
        vec![
            (MediaKind::Video, "v1"),
            (MediaKind::Image, "i1"),
            (MediaKind::Video, "v2"),
        ]
    );
}

#[test]
fn a_video_part_is_no_longer_silently_dropped() {
    let c = parse_chat(serde_json::json!([
        {"type": "video_url", "video_url": {"url": "data:image/gif;base64,ZZZ"}}
    ]));
    assert!(
        c.media.iter().any(|m| m.kind == MediaKind::Video),
        "the video was dropped on the floor again"
    );
}

#[test]
fn responses_api_carries_input_video() {
    let item = serde_json::json!({
        "type": "message",
        "role": "user",
        "content": [
            {"type": "input_video", "video_url": {"url": "data:image/gif;base64,RRR"}},
            {"type": "input_text", "text": "describe"}
        ]
    });
    let m = IncomingMessage::from_responses_input_item(&item).expect("message item");
    assert_eq!(
        media(&m.content),
        vec![(MediaKind::Video, "data:image/gif;base64,RRR")]
    );
    assert_eq!(m.content.text, "describe");
}

#[test]
fn responses_api_accepts_the_flat_video_url_string() {
    let item = serde_json::json!({
        "type": "message",
        "role": "user",
        "content": [{"type": "video_url", "video_url": "data:image/gif;base64,SSS"}]
    });
    let m = IncomingMessage::from_responses_input_item(&item).expect("message item");
    assert_eq!(
        media(&m.content),
        vec![(MediaKind::Video, "data:image/gif;base64,SSS")]
    );
}

/// 2026-09-26: `/v1/responses` parses content in its own function
/// (`from_responses_content`), so the order is checked there too.
#[test]
fn responses_api_preserves_media_order() {
    let item = serde_json::json!({
        "type": "message",
        "role": "user",
        "content": [
            {"type": "input_video", "video_url": {"url": "vvv"}},
            {"type": "input_image", "image_url": "iii"},
        ]
    });
    let m = IncomingMessage::from_responses_input_item(&item).expect("message item");
    assert_eq!(
        media(&m.content),
        vec![(MediaKind::Video, "vvv"), (MediaKind::Image, "iii")]
    );
}

#[test]
fn text_only_content_gains_no_videos() {
    let c = parse_chat(serde_json::json!("just a string"));
    assert_eq!(c.text, "just a string");
    assert!(c.media.is_empty());
}

#[test]
fn an_unknown_part_type_is_still_ignored() {
    let c = parse_chat(serde_json::json!([
        {"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64,AAA"}},
        {"type": "text", "text": "hi"}
    ]));
    assert_eq!(c.text, "hi");
    assert!(c.media.is_empty(), "audio must not be read as video");
}
