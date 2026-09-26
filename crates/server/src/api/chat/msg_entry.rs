// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: IR messages → `MsgEntry`s for the chat template, plus the
//! request's media: fetched or refused URLs, preprocessed images and videos,
//! and one pad count per item.
//!
//! Owner: server (chat API).
//! Invariants:
//! - On success, `image_pixels[i]` and `image_pad_counts[i]` describe the
//!   i-th media part in conversation and content order.
//! - Media without a vision config, or a media URL while remote fetching is
//!   off, is a 400.

use axum::http::StatusCode;
use axum::response::Response;

use metrale_config::VisionConfig;

use crate::ir::{ContentPart, ImageData, MediaKind, Message, Role};

use super::super::compact::openai_error_response;

/// 2026-09-26: What the request path needs to decode a video: the ffmpeg
/// subprocess policy and the sampling rate.
pub(crate) struct VideoDecode<'a> {
    pub(crate) ffmpeg: &'a metrale_model_layers::video_decode_ffmpeg::FfmpegPolicy,
    pub(crate) fps: f32,
}

/// 2026-09-26: One message as the chat template step sees it.
pub(crate) struct MsgEntry {
    pub(super) role: String,
    pub(super) content: String,
    /// 2026-09-26: OpenAI-shaped tool calls whose `arguments` are JSON values,
    /// not strings.
    pub(super) tool_calls: Option<Vec<serde_json::Value>>,
    pub(super) tool_call_id: Option<String>,
    /// 2026-09-26: This message's media kinds in content order
    /// ([`crate::ir::Message::media_kinds`]). When non-empty,
    /// `build_json_messages` renders the content as an array with one media
    /// entry per item, in this order.
    pub(super) media: Vec<MediaKind>,
    /// 2026-09-26: The trimmed reasoning of an earlier assistant turn, passed
    /// to the template as `reasoning_content`. `None` when it is empty, for
    /// any other role, and under `METRALE_STRIP_REASONING_HISTORY`.
    pub(super) reasoning_content: Option<String>,
}

/// 2026-09-26: Outputs of [`build_msg_entries`].
pub(super) struct BuildOut {
    pub(super) messages: Vec<MsgEntry>,
    pub(super) cwd_hint: Option<String>,
    pub(super) image_pixels: Vec<metrale_model_layers::VisionItem>,
    pub(super) image_pad_counts: Vec<usize>,
}

/// 2026-09-26: One media item for the vision path: its kind and the string
/// the preprocessor decodes. Never an http(s) URL: those are fetched into a
/// `data:` URI, or refused, when collected.
struct MediaInput {
    kind: MediaKind,
    uri: String,
}

/// 2026-09-26: Append every media part of `m` to `media` in content order,
/// pushing a 0 pad count for each (the preprocessing loop fills it in). Both
/// the tool-result branch and the other-role branch call it.
#[allow(clippy::result_large_err)]
fn collect_message_media(
    m: &Message,
    media: &mut Vec<MediaInput>,
    image_pad_counts: &mut Vec<usize>,
    remote: &super::remote_image::RemoteImagePolicy,
) -> Result<(), Response> {
    // 2026-09-26: Order is the client's. The template emits one marker per
    // media item in content order, and `expand_vision_pads` hands out pad
    // counts left to right, so this pass and `MsgEntry::media` must walk the
    // content the same way, and the preprocessing loop must keep the collected
    // order. A modality-grouped order would still pass every count check.
    for part in &m.content {
        let (kind, data) = match part {
            ContentPart::Image(src) => (MediaKind::Image, &src.data),
            ContentPart::Video(src) => (MediaKind::Video, &src.data),
            ContentPart::Text(_) => continue,
        };
        media.push(MediaInput {
            kind,
            uri: resolve_media_uri(kind, data, remote)?,
        });
        image_pad_counts.push(0);
    }
    Ok(())
}

/// 2026-09-26: Resolve one media source to the string the preprocessor
/// decodes, applying the remote-fetch policy. Images and videos share it; only
/// the noun in the error text differs.
#[allow(clippy::result_large_err)]
fn resolve_media_uri(
    kind: MediaKind,
    data: &ImageData,
    remote: &super::remote_image::RemoteImagePolicy,
) -> Result<String, Response> {
    let noun = match kind {
        MediaKind::Image => "image",
        MediaKind::Video => "video",
    };
    match data {
        ImageData::Base64(s) => Ok(s.clone()),
        // 2026-09-26: A URL is fetched into a `data:` URI when
        // `--vision-allow-remote-images` is set, and is a 400 naming that flag
        // otherwise. It is never passed on to the preprocessor as a string.
        ImageData::Url(url) => {
            let shown: String = url.chars().take(120).collect();
            if !remote.enabled {
                return Err(openai_error_response(
                    StatusCode::BAD_REQUEST,
                    format!(
                        "{noun} URLs are not fetched by this server (got '{shown}'); \
                         send the {noun} as a base64 data: URI, or start the server \
                         with --vision-allow-remote-images to enable fetching"
                    ),
                ));
            }
            match super::remote_image::fetch_as_data_uri(url, remote) {
                Ok(data_uri) => Ok(data_uri),
                Err(why) => {
                    // 2026-09-26: The fetch error's reason is passed to the
                    // client and the log.
                    tracing::warn!("remote {noun} fetch refused: {shown}: {why}");
                    Err(openai_error_response(
                        StatusCode::BAD_REQUEST,
                        format!("could not fetch {noun} URL '{shown}': {why}"),
                    ))
                }
            }
        }
    }
}

#[allow(clippy::result_large_err)]
pub(super) fn build_msg_entries(
    vision_config: Option<&VisionConfig>,
    vision_max_pixels: Option<usize>,
    remote_images: &super::remote_image::RemoteImagePolicy,
    video: &VideoDecode<'_>,
    input: &[Message],
    tools_active: bool,
    levers: &super::levers::ChatLevers,
    preserve_developer_role: bool,
) -> Result<BuildOut, Response> {
    let mut messages: Vec<MsgEntry> = Vec::with_capacity(input.len());
    let mut media: Vec<MediaInput> = Vec::new();
    let mut image_pad_counts: Vec<usize> = Vec::new();
    let mut consecutive_tool_errors: u32 = 0;
    // 2026-09-26: Tool-call counts for `hint_injector::bash_wander_hint`.
    let mut total_tool_calls: usize = 0;
    let mut productive_tool_calls: usize = 0;
    // 2026-09-26: (index into `messages`, text before hints) for every
    // tool-result entry, for the duplicate-error masking after the loop. The
    // injected hints vary with `consecutive_tool_errors`, so grouping compares
    // the text without them.
    let mut tool_result_originals: Vec<(usize, String)> = Vec::new();

    for m in input.iter() {
        let mut text = m.text();
        // 2026-09-26: A failed tool result (`Message::tool_error`) gets a
        // `[tool error]` first line, here rather than in the surface adapters,
        // before the error-hint scan below reads the text.
        if m.tool_error {
            text = format!("[tool error]\n{text}");
        }

        // 2026-09-26: Assistant tool calls are kept whether or not this
        // request declares tools: earlier turns may carry calls the template
        // renders.
        let tool_calls_json = if m.role == Role::Assistant && !m.tool_calls.is_empty() {
            let parsed: Vec<serde_json::Value> = m
                .tool_calls
                .iter()
                .map(|tc| {
                    serde_json::json!({
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": tc.arguments
                        }
                    })
                })
                .collect();
            Some(parsed)
        } else {
            None
        };

        if m.role == Role::Assistant && !m.tool_calls.is_empty() {
            for tc in &m.tool_calls {
                total_tool_calls += 1;
                if crate::hint_injector::tool_call_is_productive(&tc.name, &tc.arguments) {
                    productive_tool_calls += 1;
                }
            }
        }

        if tools_active && m.role == Role::Tool {
            let mut text = text;
            tool_result_originals.push((messages.len(), text.clone()));
            if crate::hint_injector::looks_like_error(&text) {
                consecutive_tool_errors += 1;
                crate::hint_injector::inject_hints(&mut text, consecutive_tool_errors);
            } else {
                consecutive_tool_errors = 0;
            }
            messages.push(MsgEntry {
                role: "tool".into(),
                content: text,
                tool_calls: None,
                tool_call_id: m.tool_call_id.clone(),
                media: m.media_kinds(),
                reasoning_content: None,
            });
            collect_message_media(m, &mut media, &mut image_pad_counts, remote_images)?;
            continue;
        }

        // 2026-09-26: `METRALE_STRIP_REASONING_HISTORY` set to `1` or `true`
        // drops the reasoning of every earlier turn.
        let strip_reasoning = std::env::var("METRALE_STRIP_REASONING_HISTORY")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        // 2026-09-26: `developer` becomes `system` unless
        // `preserve_developer_role`, so the cwd-hint scan, the cwd injection,
        // the vacuous-system check and the template all see it as `system`.
        let role = match &m.role {
            Role::Other(r) if r == "developer" && !preserve_developer_role => "system".to_string(),
            r => r.as_wire().to_string(),
        };
        messages.push(MsgEntry {
            role,
            content: text,
            tool_calls: tool_calls_json,
            tool_call_id: m.tool_call_id.clone(),
            media: m.media_kinds(),
            reasoning_content: if m.role == Role::Assistant && !strip_reasoning {
                m.reasoning
                    .as_ref()
                    .map(|r| r.text.trim().to_string())
                    .filter(|s| !s.is_empty())
            } else {
                None
            },
        });
        collect_message_media(m, &mut media, &mut image_pad_counts, remote_images)?;
    }

    // 2026-09-26: Duplicate-error masking: older copies of a repeated
    // error-shaped tool result are replaced by a short marker; the newest
    // keeps its text and hints. Off with `METRALE_NO_ERROR_DEDUP=1`. It runs
    // before the vacuous-system removal below because the recorded indices
    // refer to `messages` before that shift.
    if tools_active && !error_dedup_disabled() {
        for (idx, replacement) in duplicate_error_masks(&tool_result_originals) {
            messages[idx].content = replacement;
        }
    }

    // 2026-09-26: The cwd hint comes from the first line of the first system
    // message that contains `working directory`, `working_directory` or
    // `cwd:` (any case) and has a non-empty value after its first `:`.
    let cwd_hint: Option<String> = messages.iter().find(|m| m.role == "system").and_then(|m| {
        for line in m.content.lines() {
            let lower = line.to_lowercase();
            if (lower.contains("working directory")
                || lower.contains("working_directory")
                || lower.contains("cwd:"))
                && let Some(pos) = line.find(':')
            {
                let path = line[pos + 1..]
                    .trim()
                    .trim_matches(|c| c == '`' || c == '"' || c == '\'');
                if !path.is_empty() {
                    return Some(path.to_string());
                }
            }
        }
        None
    });

    // 2026-09-26: With tools active, the hint is appended to the first
    // message when that is the system message.
    if tools_active
        && !levers.disable_cwd_hint_injection
        && let Some(ref cwd) = cwd_hint
    {
        let hints = format!("\n<environment>\nworking_directory: {cwd}\n</environment>");
        if let Some(first) = messages.first_mut()
            && first.role == "system"
        {
            first.content.push_str(&hints);
        }
    }

    // 2026-09-26: A leading system message with no instruction (see
    // `is_vacuous_system_content`) is removed.
    if messages
        .first()
        .is_some_and(|m| m.role == "system" && is_vacuous_system_content(&m.content))
    {
        let removed = messages.remove(0);
        tracing::info!(
            dropped = %removed.content.trim(),
            "Dropped content-free client system message (would bias the model toward terse output)"
        );
    }

    // 2026-09-26: Media without a vision config is a 400. One loop over the
    // collected items, in order, so `image_pixels[i]`, `image_pad_counts[i]`
    // and the i-th rendered marker are the same item.
    let mut image_pixels: Vec<metrale_model_layers::VisionItem> = Vec::new();
    if !media.is_empty() {
        let Some(vcfg) = vision_config else {
            return Err(openai_error_response(
                StatusCode::BAD_REQUEST,
                "this model does not accept image or video input (no vision config)".to_string(),
            ));
        };
        for (idx, input) in media.iter().enumerate() {
            let item = match input.kind {
                MediaKind::Image => {
                    match metrale_model_layers::vision_preprocess::preprocess_image_with_max_pixels(
                        &input.uri,
                        vcfg,
                        vision_max_pixels,
                    ) {
                        Ok((pixels, grid_h, grid_w)) => {
                            metrale_model_layers::VisionItem::image(pixels, grid_h, grid_w)
                        }
                        Err(e) => {
                            return Err(openai_error_response(
                                StatusCode::BAD_REQUEST,
                                format!("Image decode error: {e}"),
                            ));
                        }
                    }
                }
                MediaKind::Video => {
                    match metrale_model_layers::video_preprocess::preprocess_video(
                        &input.uri,
                        vcfg,
                        vision_max_pixels,
                        video.fps,
                        video.ffmpeg,
                    ) {
                        Ok(v) => metrale_model_layers::VisionItem {
                            groups: v.groups,
                            grid_h: v.grid_h,
                            grid_w: v.grid_w,
                        },
                        Err(e) => {
                            return Err(openai_error_response(
                                StatusCode::BAD_REQUEST,
                                format!("Video decode error: {e:#}"),
                            ));
                        }
                    }
                }
            };
            image_pad_counts[idx] = item.pad_count(vcfg.spatial_merge_size);
            if input.kind == MediaKind::Video {
                tracing::info!(
                    "Video (media item {}): {} temporal groups, {}x{} patches, {} vision tokens",
                    idx,
                    item.t_len(),
                    item.grid_h,
                    item.grid_w,
                    image_pad_counts[idx],
                );
            }
            image_pixels.push(item);
        }
    }

    // 2026-09-26: The bash-wander hint, when `bash_wander_hint` returns one
    // (`METRALE_BASH_WANDER_WATCHDOG=1`), goes on the latest tool result.
    if tools_active
        && let Some(hint) = crate::hint_injector::bash_wander_hint(
            total_tool_calls,
            productive_tool_calls,
            levers.bash_wander,
        )
        && let Some(last_tool) = messages.iter_mut().rev().find(|e| e.role == "tool")
    {
        last_tool.content.push_str(&hint);
    }

    Ok(BuildOut {
        messages,
        cwd_hint,
        image_pixels,
        image_pad_counts,
    })
}

/// 2026-09-26: True when a system message carries no instruction: after
/// trimming it is empty, or it is one line of at most 32 bytes that ends in
/// `:` and has only ASCII letters, spaces, `_` or `-` (at least one) before
/// it, e.g. `User Context:`.
fn is_vacuous_system_content(content: &str) -> bool {
    let t = content.trim();
    if t.is_empty() {
        return true;
    }
    if !t.contains('\n') && t.len() <= 32 && t.ends_with(':') {
        let label = &t[..t.len() - 1];
        return !label.is_empty()
            && label
                .chars()
                .all(|c| c.is_ascii_alphabetic() || c == ' ' || c == '_' || c == '-');
    }
    false
}

/// 2026-09-26: `METRALE_NO_ERROR_DEDUP=1` turns off duplicate-error masking.
fn error_dedup_disabled() -> bool {
    std::env::var("METRALE_NO_ERROR_DEDUP").as_deref() == Ok("1")
}

/// 2026-09-26: Duplicate-error masking.
///
/// Input: `(message_index, text before hints)` for every tool-result entry,
/// in conversation order. Output: `(message_index, replacement)` for every
/// member of a duplicate group except the newest. Two results are duplicates
/// when both are error-shaped (`hint_injector::looks_like_error`) and they
/// are equal after trimming or have a Jaccard similarity of at least 0.9 over
/// `loop_detector::shingle_set`. A result that is not error-shaped is never
/// masked. A group is matched against its first member.
fn duplicate_error_masks(tool_results: &[(usize, String)]) -> Vec<(usize, String)> {
    const NEAR_DUP_JACCARD: f64 = 0.9;
    let errors: Vec<(usize, &str)> = tool_results
        .iter()
        .filter(|(_, t)| crate::hint_injector::looks_like_error(t))
        .map(|(i, t)| (*i, t.trim()))
        .collect();
    if errors.len() < 2 {
        return Vec::new();
    }
    let shingle_sets: Vec<_> = errors
        .iter()
        .map(|(_, t)| crate::loop_detector::shingle_set(t))
        .collect();
    // 2026-09-26: An error with fewer tokens than the shingle order (4) has
    // an empty shingle set, and `jaccard` gives 0.0 for an empty set, so such
    // errors group only when equal.
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for i in 0..errors.len() {
        let group = groups.iter_mut().find(|g| {
            let rep = g[0];
            errors[rep].1 == errors[i].1
                || crate::loop_detector::jaccard(&shingle_sets[rep], &shingle_sets[i])
                    >= NEAR_DUP_JACCARD
        });
        match group {
            Some(g) => g.push(i),
            None => groups.push(vec![i]),
        }
    }
    let mut masks = Vec::new();
    for g in &groups {
        let n = g.len();
        if n < 2 {
            continue;
        }
        for (k, &ei) in g.iter().take(n - 1).enumerate() {
            masks.push((
                errors[ei].0,
                format!("[same error as below, attempt {} of {}]", k + 1, n),
            ));
        }
    }
    masks
}

#[cfg(test)]
#[path = "msg_entry_tests.rs"]
mod msg_entry_tests;

#[cfg(test)]
#[path = "msg_entry_media_order_tests.rs"]
mod msg_entry_media_order_tests;
