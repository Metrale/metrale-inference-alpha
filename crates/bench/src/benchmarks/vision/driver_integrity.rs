// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The vision driver's ten integrity legs, one per `next()`
//! (`cursor` 0 to 9), each scored as a `media_integrity::Cell`.
//!
//! Owner: bench, vision.
//! Invariants: none beyond the types.

use super::{
    BenchmarkResult, FIXTURES, LogLine, Phase, Result, VisionFidelity,
    expected_vision_tokens_bounded, http, one_line,
};

impl VisionFidelity {
    // 2026-09-26: Ten integrity legs (`cursor` 0 to 9), one per
    // `next()`, so progress shows and a hang names its leg.
    pub(super) async fn integrity_leg(&mut self) -> Result<BenchmarkResult> {
        use crate::benchmarks::media_integrity as mi;
        let h = self.handle()?;
        let model = h.target().model.clone();
        let tmo = self.timeout();
        // 2026-09-26: The declared bound, for the two legs that assert
        // a vision-token difference.
        let cap = self.vision_max_pixels;
        // 2026-09-26: The flat red/blue image, the second image of the
        // cache leg. It is in EXIF_PAIR, which `self.fixture` does not
        // search.
        let red = super::provision::EXIF_PAIR
            .iter()
            .find(|(n, _)| *n == "16_exif_none_224.jpg")
            .map(|(_, b)| *b);
        let cell = match self.cursor {
            // 2026-09-26: 1. Different requests in flight, each scored
            //    against its own input. The identical-request sweep in
            //    the next phase cannot see one request answered with
            //    another's input.
            0 => {
                let (_, a, _, _) = FIXTURES[0];
                let (_, b, _, _) = FIXTURES[6];
                let (_, c, _, _) = FIXTURES[1];
                let q = "Reply with exactly one word: the number of white \
                         rectangles you can see, spelled out.";
                let subjects: Vec<mi::Subject> = vec![
                    (
                        mi::image_request(&model, "image/png", a, q, 24),
                        Box::new(|r: &str| !r.trim().is_empty()),
                        "224".to_string(),
                    ),
                    (
                        mi::image_request(&model, "image/png", b, q, 24),
                        Box::new(|r: &str| !r.trim().is_empty()),
                        "1280x720".to_string(),
                    ),
                    (
                        mi::image_request(&model, "image/png", c, q, 24),
                        Box::new(|r: &str| !r.trim().is_empty()),
                        "336".to_string(),
                    ),
                    // 2026-09-26: A text-only request in the same batch,
                    // which owns no image grid.
                    (
                        serde_json::json!({
                            "model": model, "stream": true, "temperature": 0.0,
                            "max_tokens": 8,
                            "chat_template_kwargs": {"enable_thinking": false},
                            "messages": [{"role": "user",
                                "content": "Reply with exactly: BANANA"}],
                        }),
                        Box::new(|r: &str| r.to_uppercase().contains("BANANA")),
                        "text-only".to_string(),
                    ),
                ];
                mi::heterogeneous_concurrency(h, subjects, tmo).await
            }
            // 2026-09-26: 2. Same prompt text, different image, back
            //    to back.
            1 => {
                let (_, first, _, _) = FIXTURES[0];
                let Some(second) = red else {
                    return Ok(self.frame(
                        "integrity",
                        vec![LogLine::warn(
                            "prefix-cache-isolation: fixture missing".to_string(),
                        )],
                    ));
                };
                let q = "What is the dominant colour in this image? One word.";
                // 2026-09-26: The second reply passes by naming red or
                // blue, the second image's colors; "purple" or
                // "gradient" is read as an answer about the first.
                mi::cache_leak(
                    h,
                    mi::image_request(&model, "image/png", first, q, 16),
                    mi::image_request(&model, "image/jpeg", second, q, 16),
                    &|r: &str| {
                        let l = r.to_lowercase();
                        l.contains("red") || l.contains("blue")
                    },
                    &|r: &str| {
                        let l = r.to_lowercase();
                        l.contains("purple") || l.contains("gradient")
                    },
                    tmo,
                )
                .await
            }
            // 2026-09-26: 3. The same image question behind a long
            //    prompt.
            2 => {
                let (_, img, _, _) = FIXTURES[0];
                let q = "Reply with exactly one word: YES if this image contains a \
                         white rectangle, NO otherwise.";
                // 2026-09-26: 700 repeats of one sentence that says
                // nothing about the image.
                let filler = "The quick brown fox jumps over the lazy dog. ".repeat(700);
                let long_q = format!("{filler}\n\n{q}");
                mi::long_prompt_path(
                    h,
                    mi::image_request(&model, "image/png", img, q, 16),
                    mi::image_request(&model, "image/png", img, &long_q, 16),
                    &|r: &str| r.to_uppercase().contains("YES"),
                    tmo,
                )
                .await
            }
            // 2026-09-26: 4. The image in an earlier user turn.
            3 => {
                let (_, img, _, _) = FIXTURES[0];
                use base64::Engine;
                let mut uri = String::from("data:image/png;base64,");
                base64::engine::general_purpose::STANDARD.encode_string(img, &mut uri);
                let body = serde_json::json!({
                    "model": model, "stream": true, "temperature": 0.0,
                    "max_tokens": 24,
                    "chat_template_kwargs": {"enable_thinking": false},
                    "messages": [
                        {"role": "user", "content": [
                            {"type": "image_url", "image_url": {"url": uri}},
                            {"type": "text", "text": "Here is a screenshot."}]},
                        {"role": "assistant", "content": "Understood."},
                        {"role": "user", "content":
                            "Reply with exactly one word: YES if the image I sent \
                             earlier contains a white rectangle, NO otherwise."},
                    ],
                });
                mi::media_in_history(
                    h,
                    body,
                    &|r: &str| r.to_uppercase().contains("YES"),
                    "media-in-history",
                    tmo,
                )
                .await
            }
            // 2026-09-26: 5. The same request streamed and not
            //    streamed.
            4 => {
                let (_, img, _, _) = FIXTURES[0];
                mi::stream_parity(
                    h,
                    mi::image_request(
                        &model,
                        "image/png",
                        img,
                        "Reply with exactly one word: YES if this image contains a \
                         white rectangle, NO otherwise.",
                        16,
                    ),
                    &|r: &str| r.to_uppercase().contains("YES"),
                    tmo,
                )
                .await
            }
            // 2026-09-26: 6. `n` = 2 choices for one image request.
            5 => {
                let (_, img, _, _) = FIXTURES[0];
                mi::multi_choice(
                    h,
                    mi::image_request(
                        &model,
                        "image/png",
                        img,
                        "Reply with exactly one word: YES if this image contains a \
                         white rectangle, NO otherwise.",
                        16,
                    ),
                    2,
                    &|r: &str| r.to_uppercase().contains("YES"),
                    tmo,
                )
                .await
            }
            // 2026-09-26: 7. The image on a tool result, which the
            //    server collects in the tool-result branch
            //    (`api/chat/msg_entry.rs`, `collect_message_media`).
            6 => {
                let (_, img, _, _) = FIXTURES[0];
                use base64::Engine;
                let mut uri = String::from("data:image/png;base64,");
                base64::engine::general_purpose::STANDARD.encode_string(img, &mut uri);
                let body = serde_json::json!({
                    "model": model, "stream": true, "temperature": 0.0,
                    "max_tokens": 16,
                    "chat_template_kwargs": {"enable_thinking": false},
                    "messages": [
                        {"role": "user", "content": "Take a screenshot."},
                        {"role": "assistant", "content": "",
                         "tool_calls": [{"id": "c1", "type": "function",
                           "function": {"name": "screenshot", "arguments": "{}"}}]},
                        {"role": "tool", "tool_call_id": "c1", "content": [
                            {"type": "image_url", "image_url": {"url": uri}}]},
                        {"role": "user", "content":
                            "Reply with exactly one word: YES if the screenshot \
                             contains a white rectangle, NO otherwise."},
                    ],
                });
                mi::media_in_history(
                    h,
                    body,
                    &|r: &str| r.to_uppercase().contains("YES"),
                    "image-on-tool-result",
                    tmo,
                )
                .await
            }
            // 2026-09-26: 8. The Responses API, compared with itself
            //    on two images (`media_integrity::responses_parity`).
            7 => {
                let (_, small, sw, sh) = FIXTURES[0];
                let (_, large, lw, lh) = FIXTURES[1];
                let delta = (expected_vision_tokens_bounded(lw, lh, 16, 2, cap)
                    - expected_vision_tokens_bounded(sw, sh, 16, 2, cap))
                    as usize;
                let q = "Reply with exactly one word: YES if this image contains a \
                         white rectangle, NO otherwise.";
                mi::responses_parity(
                    h,
                    mi::responses_image_request(&model, "image/png", small, q),
                    mi::responses_image_request(&model, "image/png", large, q),
                    delta,
                    &|r: &str| r.to_uppercase().contains("YES"),
                    tmo,
                )
                .await
            }
            // 2026-09-26: 9. Thinking on, which the chat-completions
            //    legs turn off; leg 8 sends `reasoning.effort` "low"
            //    instead (`media_integrity::thinking_parity`).
            8 => {
                let (_, small, sw, sh) = FIXTURES[0];
                let (_, large, lw, lh) = FIXTURES[1];
                let delta = (expected_vision_tokens_bounded(lw, lh, 16, 2, cap)
                    - expected_vision_tokens_bounded(sw, sh, 16, 2, cap))
                    as usize;
                let q = "Reply with exactly one word: YES if this image contains a \
                         white rectangle, NO otherwise.";
                mi::thinking_parity(
                    h,
                    mi::thinking_image_request(&model, "image/png", small, q),
                    mi::thinking_image_request(&model, "image/png", large, q),
                    delta,
                    &|r: &str| r.to_uppercase().contains("YES"),
                    tmo,
                )
                .await
            }
            // 2026-09-26: 10. EXIF orientation, with both expected
            //     answers named.
            _ => {
                let pair: Vec<(&str, &[u8])> = super::provision::EXIF_PAIR.to_vec();
                let q = "The image is split into two halves of solid colour. Is the \
                         RED half on the top, bottom, left, or right? One word.";
                let mut answers = Vec::new();
                for (name, bytes) in &pair {
                    let body = mi::image_request(&model, "image/jpeg", bytes, q, 12);
                    match http::chat_stream(h.target(), &body, tmo).await {
                        Ok(o) => answers.push((*name, o.text.trim().to_lowercase())),
                        Err(e) => {
                            answers.push((*name, format!("error: {}", one_line(format!("{e:#}")))))
                        }
                    }
                }
                // 2026-09-26: Orientation 6 means "rotate 90 CW to
                // display", which carries the stored top edge to the
                // right, so the tagged image must read "right" and the
                // untagged one "top".
                let tagged = answers.first().map(|a| a.1.clone()).unwrap_or_default();
                let untagged = answers.get(1).map(|a| a.1.clone()).unwrap_or_default();
                if tagged.contains("right") && untagged.contains("top") {
                    mi::Cell::Pass {
                        id: "exif-orientation",
                        detail: format!(
                            "tagged -> \"{tagged}\", untagged -> \"{untagged}\": EXIF \
                             orientation is APPLIED, so a rotated photo reaches the \
                             model the way its owner sees it"
                        ),
                    }
                } else if tagged == untagged {
                    mi::Cell::Fail {
                        id: "exif-orientation",
                        detail: format!(
                            "both answered \"{tagged}\" — the EXIF tag is being IGNORED \
                             again. Every rotated phone photo is reaching the model a \
                             quarter turn from how the user saw it, and nothing errors"
                        ),
                    }
                } else {
                    mi::Cell::Fail {
                        id: "exif-orientation",
                        detail: format!(
                            "unexpected orientation: tagged -> \"{tagged}\", untagged -> \
                             \"{untagged}\". Orientation=6 should put the red half on \
                             the RIGHT and the untagged one on TOP"
                        ),
                    }
                }
            }
        };
        let line = if cell.passed() {
            LogLine::info(cell.line())
        } else {
            LogLine::warn(cell.line())
        };
        self.integrity.push(cell);
        self.cursor += 1;
        if self.cursor >= 10 {
            self.cursor = 0;
            self.phase = Phase::Concurrency;
        }
        Ok(self.frame("integrity", vec![line]))
    }
}
