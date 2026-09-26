// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The vision driver's prompt-token legs: template-overhead
//! calibration, the geometry ladder and the concurrency levels. Each method
//! runs one `next()` of its `Phase` and sets the phase that follows it.
//!
//! Owner: bench, vision.
//! Invariants: none beyond the types.

use super::{
    BenchmarkResult, Context, FIXTURES, GeomCell, LogLine, Phase, PluginHandle, Result,
    VisionFidelity, concurrency_probe, expected_vision_tokens_bounded, http, one_line,
    reply_matches, request,
};

impl VisionFidelity {
    // 2026-09-26: Measure the chat template's token cost from the first
    // fixture, whose vision-token count is predicted, since the cost
    // depends on the checkpoint's template.
    pub(super) async fn calibrate_leg(&mut self, handle: &PluginHandle) -> Result<BenchmarkResult> {
        let (name, bytes, w, h) = FIXTURES[0];
        let body = request::body(&handle.target().model, &[bytes], "Colour?", 8);
        let out = http::chat_stream(handle.target(), &body, self.timeout())
            .await
            .context("calibration request failed — is this a vision-capable model?")?;
        let want = expected_vision_tokens_bounded(w, h, 16, 2, self.vision_max_pixels) as usize;
        let overhead = out.prompt_tokens.checked_sub(want).with_context(|| {
            format!(
                "calibration: {name} reported {} prompt tokens but its {want} vision \
                 tokens alone exceed that — the served geometry is not patch 16 / \
                 merge 2, so this benchmark's arithmetic does not apply",
                out.prompt_tokens
            )
        })?;
        self.overhead = Some(overhead);
        self.phase = Phase::Geometry;
        Ok(self.frame(
            "calibrate",
            vec![LogLine::info(format!(
                "template overhead {overhead} tokens (from {name}: {} total − {want} vision)",
                out.prompt_tokens
            ))],
        ))
    }

    pub(super) async fn geometry_leg(&mut self, handle: &PluginHandle) -> Result<BenchmarkResult> {
        let (name, bytes, w, h) = FIXTURES[self.cursor];
        let overhead = self.overhead.context("geometry ran before calibration")?;
        let want = expected_vision_tokens_bounded(w, h, 16, 2, self.vision_max_pixels) as usize;
        let body = request::body(&handle.target().model, &[bytes], "Colour?", 8);
        let cell = match http::chat_stream(handle.target(), &body, self.timeout()).await {
            Ok(o) => match request::vision_tokens(o.prompt_tokens, overhead) {
                Ok(got) if got == want => GeomCell::Match {
                    fixture: name,
                    tokens: got,
                },
                Ok(got) => GeomCell::Mismatch {
                    fixture: name,
                    want,
                    got,
                },
                Err(e) => GeomCell::Error {
                    fixture: name,
                    msg: one_line(format!("{e:#}")),
                },
            },
            // 2026-09-26: An image past the encoder's capacity is
            // Unmeasured, not a failure. The engine's refusal says
            // "this encoder holds" (`vision_encoder/enc_impl/pos_embed.rs`).
            Err(e) if format!("{e:#}").contains("this encoder holds") => GeomCell::Unmeasured {
                fixture: name,
                why: one_line(format!("{e:#}")),
            },
            Err(e) => GeomCell::Error {
                fixture: name,
                msg: one_line(format!("{e:#}")),
            },
        };
        let line = match &cell {
            GeomCell::Match { tokens, .. } => LogLine::info(format!("{name}: {tokens} tokens")),
            GeomCell::Mismatch { want, got, .. } => {
                LogLine::warn(format!("{name}: expected {want}, got {got}"))
            }
            GeomCell::Unmeasured { .. } => {
                LogLine::info(format!("{name}: over encoder capacity — unmeasured"))
            }
            GeomCell::Error { msg, .. } => LogLine::warn(format!("{name}: {msg}")),
        };
        self.geom.push(cell);
        self.cursor += 1;
        if self.cursor >= FIXTURES.len() {
            self.cursor = 0;
            self.phase = Phase::Probes;
        }
        Ok(self.frame("geometry", vec![line]))
    }

    // 2026-09-26: `concurrency::LEVELS` copies of the size-label probe:
    // every reply correct and every level at the C=1 prompt-token
    // count. Wall time goes into the metrics and is not asserted.
    pub(super) async fn concurrency_leg(&mut self) -> Result<BenchmarkResult> {
        use crate::benchmarks::video::concurrency::{LEVELS, run_level};
        let level = LEVELS[self.cursor];
        let probe = concurrency_probe();
        let png = self.fixture(probe.images[0])?;
        // 2026-09-26: The same `self.max_tokens` the capability phase
        // gives this probe: the reply must reach the `1280` label, and
        // a short budget can end it first.
        let body = request::body(
            &self.handle()?.target().model,
            &[png],
            probe.prompt,
            self.max_tokens,
        );
        let is_correct = |reply: &str| reply_matches(reply, probe.want_all, probe.want_none);
        let r = run_level(self.handle()?, &body, level, self.timeout(), &is_correct).await;
        let baseline_prompt_tokens = self
            .conc_results
            .first()
            .and_then(|baseline| baseline.prompt_tokens);
        let clean =
            baseline_prompt_tokens.map_or_else(|| r.ok(), |baseline| r.ok_against(baseline));
        let geometry = r.geometry_detail(baseline_prompt_tokens);
        let line = if clean {
            LogLine::info(format!(
                "C={level}: {}/{} returned, {geometry}, {} ms",
                r.returned, r.conc, r.wall_ms,
            ))
        } else {
            LogLine::warn(format!(
                "C={level}: {}/{} returned, {geometry}, {} ms{}",
                r.returned,
                r.conc,
                r.wall_ms,
                if r.errors.is_empty() {
                    String::new()
                } else {
                    format!(" — {}", r.errors.join("; "))
                }
            ))
        };
        self.conc_results.push(r);
        self.cursor += 1;
        if self.cursor >= LEVELS.len() {
            self.cursor = 0;
            self.phase = Phase::Control;
        }
        Ok(self.frame("concurrency", vec![line]))
    }
}
