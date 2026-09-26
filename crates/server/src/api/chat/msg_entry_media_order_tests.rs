// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Media order tests at `collect_message_media`, where the order
//! of the encoder inputs and pad counts is decided; `MsgEntry::media` must
//! give the rendered markers the same order.
//!
//! A base64 URI is carried unchanged at this stage, so the cases need no
//! vision config, pixels or ffmpeg.
//!
//! Owner: server (chat API) tests.
//! Invariants: none beyond the types.

use super::collect_message_media;
use crate::ir::MediaKind;
use crate::ir::message::{ContentPart, ImageData, ImageSource, Message, Role, VideoSource};

fn img(uri: &str) -> ContentPart {
    ContentPart::Image(ImageSource {
        data: ImageData::Base64(uri.into()),
    })
}

fn vid(uri: &str) -> ContentPart {
    ContentPart::Video(VideoSource {
        data: ImageData::Base64(uri.into()),
    })
}

fn message(parts: Vec<ContentPart>) -> Message {
    Message {
        role: Role::User,
        content: parts,
        tool_calls: Vec::new(),
        tool_call_id: None,
        name: None,
        reasoning: None,
        tool_error: false,
    }
}

/// 2026-09-26: `(kind, uri)` per collected item, and the pad-count length.
fn collect(m: &Message) -> (Vec<(MediaKind, String)>, usize) {
    let mut media = Vec::new();
    let mut pads = Vec::new();
    collect_message_media(
        m,
        &mut media,
        &mut pads,
        &crate::api::chat::remote_image::RemoteImagePolicy::default(),
    )
    .expect("base64 URIs need no fetch policy");
    (
        media.iter().map(|i| (i.kind, i.uri.clone())).collect(),
        pads.len(),
    )
}

#[test]
fn video_before_image_is_collected_in_that_order() {
    let m = message(vec![
        vid("clip"),
        img("still"),
        ContentPart::Text("which came first?".into()),
    ]);
    let (media, pads) = collect(&m);
    assert_eq!(
        media,
        vec![
            (MediaKind::Video, "clip".to_string()),
            (MediaKind::Image, "still".to_string()),
        ]
    );
    assert_eq!(pads, 2, "one pad-count slot per media item, in that order");
}

#[test]
fn alternating_media_keeps_every_position() {
    // 2026-09-26: Alternating kinds: an order grouped by kind, in either
    // kind order, fails this.
    let m = message(vec![vid("v1"), img("i1"), vid("v2"), img("i2")]);
    let (media, _) = collect(&m);
    assert_eq!(
        media.iter().map(|(k, _)| *k).collect::<Vec<_>>(),
        vec![
            MediaKind::Video,
            MediaKind::Image,
            MediaKind::Video,
            MediaKind::Image
        ]
    );
}

/// 2026-09-26: `collect_message_media` and `Message::media_kinds` (which
/// fills `MsgEntry::media`) traverse the content separately; the pad counts
/// only line up with the markers while they agree.
#[test]
fn collected_order_matches_the_rendered_marker_order() {
    let m = message(vec![
        vid("v1"),
        ContentPart::Text("mid".into()),
        img("i1"),
        vid("v2"),
    ]);
    let (media, _) = collect(&m);
    assert_eq!(
        media.iter().map(|(k, _)| *k).collect::<Vec<_>>(),
        m.media_kinds(),
        "the encoder items and the template markers must describe the same sequence"
    );
}

/// 2026-09-26: The pad-count vector spans the whole request, so a later
/// message's items follow an earlier message's.
#[test]
fn media_accumulates_across_messages_in_order() {
    let first = message(vec![vid("v1")]);
    let second = message(vec![img("i1")]);
    let mut media = Vec::new();
    let mut pads = Vec::new();
    let policy = crate::api::chat::remote_image::RemoteImagePolicy::default();
    collect_message_media(&first, &mut media, &mut pads, &policy).expect("first");
    collect_message_media(&second, &mut media, &mut pads, &policy).expect("second");
    assert_eq!(
        media.iter().map(|i| i.kind).collect::<Vec<_>>(),
        vec![MediaKind::Video, MediaKind::Image]
    );
    assert_eq!(pads.len(), 2);
}
