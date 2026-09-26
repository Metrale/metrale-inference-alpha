// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The video driver's prompt-token legs: group-count geometry,
//! MP4/GIF backend parity and the concurrency levels. Each method runs one
//! `next()` of its `Phase` and sets the phase that follows it.
//!
//! Owner: bench, video.
//! Invariants: none beyond the types.

use super::{
    BenchmarkResult, Context, CountCell, LEVELS, LogLine, PALETTE, Phase, Result, VideoFidelity,
    check_proportional, clip, order_matches, request, run_level,
};

impl VideoFidelity {
    pub(super) async fn geometry_leg(&mut self) -> Result<BenchmarkResult> {
        self.phase = Phase::Order;
        let unit = clip("05_colors_unit.mp4").context("fixture 05 missing")?;
        let half = clip("04_colors_half.mp4").context("fixture 04 missing")?;
        let full = clip("01_colors_fwd.mp4").context("fixture 01 missing")?;

        let mut totals = Vec::new();
        for c in [unit, half, full] {
            match self.tokens_for(c).await {
                Ok(n) => totals.push(n),
                Err((skip, why)) => {
                    let cell = if skip {
                        CountCell::Skipped {
                            id: "group-proportionality",
                            why: why.clone(),
                        }
                    } else {
                        CountCell::Error {
                            id: "group-proportionality",
                            msg: why.clone(),
                        }
                    };
                    self.counts.push(cell);
                    return Ok(self.frame(
                        "geometry",
                        vec![LogLine::info(format!("group-proportionality: {why}"))],
                    ));
                }
            }
        }
        let (t1, t2, t4) = (totals[0], totals[1], totals[2]);
        self.full_tokens = Some(t4);

        let cell = match check_proportional(t1, t2, t4, self.plane) {
            Some(r) => CountCell::Match {
                id: "group-proportionality",
                detail: format!(
                    "1s={t1}, 2s={t2}, 4s={t4} tok -> {}/{}/{} groups at {} tok/group \
                     (template overhead {})",
                    r.unit_groups,
                    r.unit_groups * 2,
                    r.unit_groups * 4,
                    self.plane,
                    r.overhead
                ),
            },
            None => CountCell::Mismatch {
                id: "group-proportionality",
                detail: format!(
                    "1s={t1}, 2s={t2}, 4s={t4} tok: groups do not scale with duration \
                     ((t4-t2) is {} where 2*(t2-t1) is {}), so sampling or temporal \
                     grouping is wrong",
                    t4.saturating_sub(t2),
                    2 * t2.saturating_sub(t1)
                ),
            },
        };
        let line = match &cell {
            CountCell::Match { detail, .. } => {
                LogLine::info(format!("group-proportionality: {detail}"))
            }
            CountCell::Mismatch { detail, .. } => {
                LogLine::warn(format!("group-proportionality: {detail}"))
            }
            _ => LogLine::info("group-proportionality".to_string()),
        };
        self.counts.push(cell);
        Ok(self.frame("geometry", vec![line]))
    }

    pub(super) async fn parity_leg(&mut self) -> Result<BenchmarkResult> {
        self.phase = Phase::Mixed;
        let gif = clip("03_colors_fwd.gif").context("fixture 03 missing")?;
        let cell = match (self.full_tokens, self.tokens_for(gif).await) {
            (Some(mp4), Ok(g)) if mp4 == g => CountCell::Match {
                id: "backend-parity",
                detail: format!("mp4 and gif both {g} prompt tokens"),
            },
            (Some(mp4), Ok(g)) => CountCell::Mismatch {
                id: "backend-parity",
                detail: format!(
                    "identical content decoded to different geometry: mp4 {mp4} tokens, \
                     gif {g}"
                ),
            },
            (None, Ok(_)) => CountCell::Skipped {
                id: "backend-parity",
                why: "the mp4 side was not measured".to_string(),
            },
            (_, Err((skip, why))) => {
                if skip {
                    CountCell::Skipped {
                        id: "backend-parity",
                        why,
                    }
                } else {
                    CountCell::Error {
                        id: "backend-parity",
                        msg: why,
                    }
                }
            }
        };
        let line = match &cell {
            CountCell::Match { detail, .. } => LogLine::info(format!("backend-parity: {detail}")),
            CountCell::Mismatch { detail, .. } => {
                LogLine::warn(format!("backend-parity: {detail}"))
            }
            CountCell::Skipped { why, .. } => {
                LogLine::info(format!("backend-parity: skipped — {why}"))
            }
            CountCell::Error { msg, .. } => LogLine::warn(format!("backend-parity: {msg}")),
        };
        self.counts.push(cell);
        Ok(self.frame("parity", vec![line]))
    }

    // 2026-09-26: `concurrency::LEVELS` copies of one request: every
    // reply correct and every level at the C=1 prompt-token count.
    // Wall time goes into the metrics and is not asserted.
    pub(super) async fn concurrency_leg(&mut self) -> Result<BenchmarkResult> {
        let level = LEVELS[self.cursor];
        let c = clip("03_colors_fwd.gif").context("fixture 03 missing")?;
        let want: Vec<&str> = c.colors.to_vec();
        let body = request::video_body(
            &self.handle()?.target().model,
            c.mime,
            c.bytes,
            request::ORDER_PROMPT,
            self.max_tokens,
        );
        let is_correct = |reply: &str| order_matches(reply, &want, PALETTE);
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
                "C={level}: {}/{} correct, {geometry}, {} ms",
                r.correct, r.conc, r.wall_ms,
            ))
        } else {
            LogLine::warn(format!(
                "C={level}: {}/{} returned, {}/{} CORRECT, {geometry}, {} ms{}",
                r.returned,
                r.conc,
                r.correct,
                r.conc,
                r.wall_ms,
                if r.errors.is_empty() {
                    String::new()
                } else {
                    format!(" — {}", r.errors.join("; "))
                }
            ))
        };
        let cell = if r.errors.iter().any(|e| request::is_decoder_unavailable(e)) {
            CountCell::Skipped {
                id: "concurrency",
                why: format!("C={level}: no video decoder"),
            }
        } else if clean {
            CountCell::Match {
                id: "concurrency",
                detail: format!("C={level} clean in {} ms", r.wall_ms),
            }
        } else {
            CountCell::Mismatch {
                id: "concurrency",
                detail: format!("C={level}: {}/{} correct, {geometry}", r.correct, r.conc),
            }
        };
        self.counts.push(cell);
        self.conc_results.push(r);
        self.cursor += 1;
        if self.cursor >= LEVELS.len() {
            self.cursor = 0;
            self.phase = Phase::Control;
        }
        Ok(self.frame("concurrency", vec![line]))
    }
}
