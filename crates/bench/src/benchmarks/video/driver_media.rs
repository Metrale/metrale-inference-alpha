// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The video driver's multi-item legs: an image and a video in
//! one request, and the four integrity legs. Each method runs one `next()`
//! of its `Phase` and sets the phase that follows it.
//!
//! Owner: bench, video.
//! Invariants: none beyond the types.

use super::{
    BenchmarkResult, Context, CountCell, LogLine, PALETTE, Phase, Result, VideoFidelity, clip,
    http, one_line, order_matches, request, skip_if_decoder_unavailable,
};

impl VideoFidelity {
    // 2026-09-26: Two cells for an image and a video in one request.
    //   * `mixed-media` asks for the clip's colors as a list. Measured
    //     2026-08-21: qwen3.8-27B named the still as the clip's first
    //     frame and lost the last color while the clip was served as
    //     4 temporal groups, 196 vision tokens, in every ordering, so a
    //     failure here can be the model rather than the engine.
    //   * `mixed-media-pads` checks
    //     t(image + video) == t(image) + t(video) - t(text) in the
    //     server's `usage.prompt_tokens`, which no model output enters.
    pub(super) async fn mixed_leg(&mut self) -> Result<BenchmarkResult> {
        self.phase = Phase::Integrity;
        let h = self.handle()?;
        let vid = clip("03_colors_fwd.gif").context("fixture 03 missing")?;
        // 2026-09-26: The vision benchmark's grayscale fixture: the
        // question asks for the video's colors, and a gray still
        // cannot supply a palette color.
        let png = crate::benchmarks::vision::provision::FIXTURES
            .iter()
            .find(|(n, _, _, _)| *n == "13_gray_224.jpg")
            .map(|(_, b, _, _)| *b)
            .context("grayscale fixture missing")?;
        let body = request::mixed_body(
            &h.target().model,
            png,
            vid.mime,
            vid.bytes,
            "You were given one image and one video. Reply with only the colors in the \
             VIDEO, in order, separated by commas.",
            self.max_tokens,
        );
        let cell = match http::chat_stream(h.target(), &body, self.timeout()).await {
            Ok(out) => {
                let reply = out.text.trim();
                if order_matches(reply, vid.colors, PALETTE) {
                    CountCell::Match {
                        id: "mixed-media",
                        detail: format!(
                            "image + video in one request, {} prompt tokens, video read \
                             correctly",
                            out.prompt_tokens
                        ),
                    }
                } else {
                    CountCell::Mismatch {
                        id: "mixed-media",
                        detail: format!(
                            "image + video together: wanted [{}], got [{}]. This cell \
                             does NOT localise the cause: see mixed-media-pads, which \
                             asks for the LAST frame directly and is the one that \
                             accuses the pad/marker contract",
                            vid.colors.join(", "),
                            super::score::colors_in_order(reply, PALETTE).join(", ")
                        ),
                    }
                }
            }
            Err(e) => {
                let msg = one_line(format!("{e:#}"));
                if request::is_decoder_unavailable(&msg) {
                    CountCell::Skipped {
                        id: "mixed-media",
                        why: msg,
                    }
                } else {
                    CountCell::Error {
                        id: "mixed-media",
                        msg,
                    }
                }
            }
        };
        // 2026-09-26: The pad-arithmetic cell: four short requests,
        // and the identity
        //
        //     t(image + video) == t(image) + t(video) - t(text)
        //
        // must hold exactly in the server's `usage.prompt_tokens`, so a
        // pad run of the wrong length gives a non-zero difference.
        let model = h.target().model.clone();
        let short = 16;
        let b_text = request::text_array_body(&model, request::ORDER_PROMPT, short);
        let b_img = request::image_body(&model, "image/png", png, request::ORDER_PROMPT, short);
        let b_vid = request::video_body(&model, vid.mime, vid.bytes, request::ORDER_PROMPT, short);
        let b_both = request::mixed_body(
            &model,
            png,
            vid.mime,
            vid.bytes,
            request::ORDER_PROMPT,
            short,
        );
        let t_text = http::chat_stream(h.target(), &b_text, self.timeout()).await;
        let t_img = http::chat_stream(h.target(), &b_img, self.timeout()).await;
        let t_vid = http::chat_stream(h.target(), &b_vid, self.timeout()).await;
        let t_both = http::chat_stream(h.target(), &b_both, self.timeout()).await;
        let tail_cell = match (t_text, t_img, t_vid, t_both) {
            (Ok(tx), Ok(ti), Ok(tv), Ok(tb)) => {
                let predicted =
                    (ti.prompt_tokens + tv.prompt_tokens).saturating_sub(tx.prompt_tokens);
                if tb.prompt_tokens == predicted {
                    CountCell::Match {
                        id: "mixed-media-pads",
                        detail: format!(
                            "pad arithmetic exact: image {} + video {} - text {} = {} \
                             served",
                            ti.prompt_tokens, tv.prompt_tokens, tx.prompt_tokens, tb.prompt_tokens
                        ),
                    }
                } else {
                    CountCell::Mismatch {
                        id: "mixed-media-pads",
                        detail: format!(
                            "pad run is the WRONG LENGTH in a mixed request: image {} + \
                             video {} - text {} predicts {predicted}, server charged {} \
                             (off by {}) — this IS the collection/marker/pad-expansion \
                             contract",
                            ti.prompt_tokens,
                            tv.prompt_tokens,
                            tx.prompt_tokens,
                            tb.prompt_tokens,
                            tb.prompt_tokens as i64 - predicted as i64
                        ),
                    }
                }
            }
            (tx, ti, tv, tb) => {
                let msg = [tx.err(), ti.err(), tv.err(), tb.err()]
                    .into_iter()
                    .flatten()
                    .map(|e| one_line(format!("{e:#}")))
                    .next()
                    .unwrap_or_else(|| "unknown".to_string());
                if request::is_decoder_unavailable(&msg) {
                    CountCell::Skipped {
                        id: "mixed-media-pads",
                        why: msg,
                    }
                } else {
                    CountCell::Error {
                        id: "mixed-media-pads",
                        msg,
                    }
                }
            }
        };
        let describe = |c: &CountCell, id: &str| match c {
            CountCell::Match { detail, .. } => LogLine::info(format!("{id}: {detail}")),
            CountCell::Mismatch { detail, .. } => LogLine::warn(format!("{id}: {detail}")),
            CountCell::Skipped { why, .. } => LogLine::info(format!("{id}: skipped — {why}")),
            CountCell::Error { msg, .. } => LogLine::warn(format!("{id}: {msg}")),
        };
        let line = describe(&cell, "mixed-media");
        let tail_line = describe(&tail_cell, "mixed-media-pads");
        self.counts.push(cell);
        self.counts.push(tail_cell);
        Ok(self.frame("mixed", vec![line, tail_line]))
    }

    // 2026-09-26: Four legs whose requests carry video items, which
    // the vision benchmark's `media_integrity` legs never send: two
    // videos, video before image, two opposite clips in flight, and a
    // video in an earlier turn.
    pub(super) async fn integrity_leg(&mut self) -> Result<BenchmarkResult> {
        use crate::benchmarks::media_integrity as mi;
        let h = self.handle()?;
        let model = h.target().model.clone();
        let tmo = self.timeout();
        let fwd = clip("03_colors_fwd.gif").context("fixture 03 missing")?;
        let rev = clip("02_colors_rev.mp4").context("fixture 02 missing")?;

        let cell = match self.cursor {
            // 2026-09-26: 1. Two videos in one request. The question
            //    asks about the second, and a reply with the first's
            //    colors is reported as a wrong per-item offset.
            0 => {
                let body = serde_json::json!({
                    "model": model, "stream": true, "temperature": 0.0,
                    "max_tokens": self.max_tokens,
                    "chat_template_kwargs": {"enable_thinking": false},
                    "messages": [{"role": "user", "content": [
                        {"type": "video_url", "video_url": {"url":
                            request::data_uri(fwd.mime, fwd.bytes)}},
                        {"type": "video_url", "video_url": {"url":
                            request::data_uri(rev.mime, rev.bytes)}},
                        {"type": "text", "text":
                            "Two videos were provided. List the colors of the SECOND \
                             video in the order they appear, separated by commas. \
                             Answer with only the color names."},
                    ]}],
                });
                match http::chat_stream(h.target(), &body, tmo).await {
                    Ok(o) if order_matches(o.text.trim(), rev.colors, PALETTE) => mi::Cell::Pass {
                        id: "two-videos",
                        detail: format!(
                            "second of two clips read correctly ({} prompt tokens)",
                            o.prompt_tokens
                        ),
                    },
                    Ok(o) => {
                        let got = super::score::colors_in_order(o.text.trim(), PALETTE);
                        let first_instead = got == fwd.colors.to_vec();
                        mi::Cell::Fail {
                            id: "two-videos",
                            detail: if first_instead {
                                "asked for the SECOND clip and got the FIRST — the \
                                 per-item row offset is wrong across two videos"
                                    .to_string()
                            } else {
                                format!(
                                    "wanted [{}], got [{}]",
                                    rev.colors.join(", "),
                                    got.join(", ")
                                )
                            },
                        }
                    }
                    Err(e) => {
                        let msg = one_line(format!("{e:#}"));
                        if request::is_decoder_unavailable(&msg) {
                            mi::Cell::Skipped {
                                id: "two-videos",
                                why: msg,
                            }
                        } else {
                            mi::Cell::Error {
                                id: "two-videos",
                                msg,
                            }
                        }
                    }
                }
            }
            // 2026-09-26: 2. Video before image, the reverse of the
            //    Mixed leg's order (`request::mixed_body`).
            1 => {
                // 2026-09-26: Grayscale, as in the Mixed leg.
                let png = crate::benchmarks::vision::provision::FIXTURES
                    .iter()
                    .find(|(n, _, _, _)| *n == "13_gray_224.jpg")
                    .map(|(_, b, _, _)| *b)
                    .context("grayscale fixture missing")?;
                let body = serde_json::json!({
                    "model": model, "stream": true, "temperature": 0.0,
                    "max_tokens": self.max_tokens,
                    "chat_template_kwargs": {"enable_thinking": false},
                    "messages": [{"role": "user", "content": [
                        {"type": "video_url", "video_url": {"url":
                            request::data_uri(fwd.mime, fwd.bytes)}},
                        {"type": "image_url", "image_url": {"url":
                            request::data_uri("image/jpeg", png)}},
                        {"type": "text", "text":
                            "A video came first, then a still image. List the colors of \
                             the VIDEO in order, separated by commas. Only color names."},
                    ]}],
                });
                match http::chat_stream(h.target(), &body, tmo).await {
                    Ok(o) if order_matches(o.text.trim(), fwd.colors, PALETTE) => mi::Cell::Pass {
                        id: "video-before-image",
                        detail: format!(
                            "video read correctly when it precedes the image ({} \
                                 prompt tokens)",
                            o.prompt_tokens
                        ),
                    },
                    Ok(o) => mi::Cell::Fail {
                        id: "video-before-image",
                        detail: format!(
                            "wanted [{}], got [{}] — the ordering contract holds one way \
                             round but not the other",
                            fwd.colors.join(", "),
                            super::score::colors_in_order(o.text.trim(), PALETTE).join(", ")
                        ),
                    },
                    Err(e) => {
                        let msg = one_line(format!("{e:#}"));
                        if request::is_decoder_unavailable(&msg) {
                            mi::Cell::Skipped {
                                id: "video-before-image",
                                why: msg,
                            }
                        } else {
                            mi::Cell::Error {
                                id: "video-before-image",
                                msg,
                            }
                        }
                    }
                }
            }
            // 2026-09-26: 3. Two clips with opposite color orders in
            //    flight together. The concurrency sweep sends
            //    identical requests and cannot see one request
            //    answered with another's input; here that returns the
            //    reversed sequence.
            2 => {
                let f = fwd.colors.to_vec();
                let r = rev.colors.to_vec();
                let subjects: Vec<mi::Subject> = vec![
                    (
                        request::video_body(
                            &model,
                            fwd.mime,
                            fwd.bytes,
                            request::ORDER_PROMPT,
                            self.max_tokens,
                        ),
                        Box::new(move |t: &str| order_matches(t, &f, PALETTE)),
                        "forward".to_string(),
                    ),
                    (
                        request::video_body(
                            &model,
                            rev.mime,
                            rev.bytes,
                            request::ORDER_PROMPT,
                            self.max_tokens,
                        ),
                        Box::new(move |t: &str| order_matches(t, &r, PALETTE)),
                        "reversed".to_string(),
                    ),
                ];
                skip_if_decoder_unavailable(mi::heterogeneous_concurrency(h, subjects, tmo).await)
            }
            // 2026-09-26: 4. A video in an earlier turn.
            _ => {
                let body = serde_json::json!({
                    "model": model, "stream": true, "temperature": 0.0,
                    "max_tokens": self.max_tokens,
                    "chat_template_kwargs": {"enable_thinking": false},
                    "messages": [
                        {"role": "user", "content": [
                            {"type": "video_url", "video_url": {"url":
                                request::data_uri(fwd.mime, fwd.bytes)}},
                            {"type": "text", "text": "Here is a clip."}]},
                        {"role": "assistant", "content": "Understood."},
                        {"role": "user", "content":
                            "List the colors of the video I sent earlier, in order, \
                             separated by commas. Only color names."},
                    ],
                });
                let want = fwd.colors.to_vec();
                mi::media_in_history(
                    h,
                    body,
                    &move |t: &str| order_matches(t, &want, PALETTE),
                    "video-in-history",
                    tmo,
                )
                .await
            }
        };
        let line = if cell.passed() || matches!(cell, mi::Cell::Skipped { .. }) {
            LogLine::info(cell.line())
        } else {
            LogLine::warn(cell.line())
        };
        // 2026-09-26: Integrity cells join `counts`, so a failure here
        // fails the run.
        self.counts.push(if cell.passed() {
            CountCell::Match {
                id: cell.id(),
                detail: String::new(),
            }
        } else if matches!(cell, mi::Cell::Skipped { .. }) {
            CountCell::Skipped {
                id: cell.id(),
                why: String::new(),
            }
        } else {
            CountCell::Mismatch {
                id: cell.id(),
                detail: cell.line(),
            }
        });
        self.integrity.push(cell);
        self.cursor += 1;
        if self.cursor >= 4 {
            self.cursor = 0;
            self.phase = Phase::Concurrency;
        }
        Ok(self.frame("integrity", vec![line]))
    }
}
