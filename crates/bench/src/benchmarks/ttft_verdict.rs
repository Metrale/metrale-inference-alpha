// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The TTFT gates' verdict: compare a finished run's median and p90
//! with the stored baseline, and decide whether the run becomes the new
//! baseline. `ttft.rs` measures; this module judges.
//!
//! Owner: bench, ttft.
//! Invariants:
//! - `should_store` is false for a `Fail` verdict, and `ttft.rs` saves a
//!   baseline only when it is true.
//! - `verdict` returns `Pass` only after comparing with a baseline from the same
//!   host and model that holds usable `median_ms` and `p90_ms`.

use super::super::{baseline, stats};
use super::TtftGate;
use crate::result::{CellStyle, Stat, Verdict, VerdictKind};

impl TtftGate {
    /// 2026-09-26: Whether this run's numbers become the new baseline: only with
    /// `update_baseline` set and a verdict other than `Fail`.
    ///
    /// The stored baseline is what the next run is compared with, so storing a
    /// failing run would let the same build pass when it is run again. An
    /// `Info` run stores; that includes the first run on a host, which has no
    /// baseline yet.
    pub(super) fn should_store(&self, verdict: &Verdict) -> bool {
        self.update_baseline && verdict.kind != VerdictKind::Fail
    }

    /// 2026-09-26: Compare with the stored baseline and decide. No baseline, a
    /// baseline from another host or model, or one without usable `median_ms`
    /// and `p90_ms` gives `Info`, never `Pass`. Missing or invalid current
    /// numbers give `Fail`.
    pub(super) fn verdict(&self, median: Option<f64>, p90: Option<f64>) -> (Verdict, Vec<Stat>) {
        let store = match self.handle() {
            Ok(h) => h.artifacts().clone(),
            Err(_) => return (Verdict::info("no handle"), Vec::new()),
        };
        let id = self.mode.descriptor().id;
        // 2026-09-26: Baselines are stored per model (`baseline::save`), so a
        // gate run against several checkpoints reads the one for the model it
        // is serving.
        let model_now = self.handle().ok().map(|h| h.target().model.clone());
        let stored = baseline::load_for(&store, id, model_now.as_deref());
        let mut summary = vec![
            Stat::new("Median TTFT", stats::fmt_ms(median), "ms").with_style(CellStyle::Accent),
            Stat::new("p90 TTFT", stats::fmt_ms(p90), "ms"),
        ];
        if !median.is_some_and(super::valid_ttft_ms) || !p90.is_some_and(super::valid_ttft_ms) {
            return (
                Verdict::fail("run produced no usable median and p90 TTFT measurements"),
                summary,
            );
        }
        let Some(base) = stored else {
            summary.push(Stat::new("Baseline", "none", "").with_style(CellStyle::Dim));
            return (
                Verdict::info("no baseline on this box yet — this run is recorded as the baseline"),
                summary,
            );
        };
        let target_now = self
            .handle()
            .map(|h| h.target().base_url.clone())
            .unwrap_or_default();
        let model_now = self
            .handle()
            .map(|h| h.target().model.clone())
            .unwrap_or_default();
        if !super::ttft_target::same_box(&base.target, &target_now) || base.model != model_now {
            summary.push(Stat::new("Baseline", "other target", "").with_style(CellStyle::Warn));
            return (
                Verdict::info(format!(
                    "baseline was recorded against {} / {} — not comparable, reporting only",
                    base.target, base.model
                )),
                summary,
            );
        }
        if !base.get("median_ms").is_some_and(super::valid_ttft_ms)
            || !base.get("p90_ms").is_some_and(super::valid_ttft_ms)
        {
            summary.push(Stat::new("Baseline", "incomplete", "").with_style(CellStyle::Warn));
            return (
                Verdict::info(
                    "same-box baseline is missing usable median_ms or p90_ms — not comparable, \
                     reporting only",
                ),
                summary,
            );
        }
        let dm = stats::pct_delta(median, base.get("median_ms"));
        let dp = stats::pct_delta(p90, base.get("p90_ms"));
        summary.push(
            Stat::new(
                "vs baseline",
                dm.map(|d| format!("{d:+.1}")).unwrap_or_else(|| "—".into()),
                format!("% median · {}", base.age_text()),
            )
            .with_style(match dm {
                Some(d) if d > self.median_limit_pct => CellStyle::Bad,
                Some(d) if d < 0.0 => CellStyle::Good,
                _ => CellStyle::Neutral,
            }),
        );
        // 2026-09-26: A metric fails only if it is over both its percentage
        // limit and `NOISE_FLOOR_MS`. At a 30 ms baseline a 3% limit is 0.9 ms,
        // so the percentage alone would fail on sub-millisecond deltas.
        let over_floor = |now: Option<f64>, key: &str| {
            now.zip(base.get(key))
                .is_some_and(|(now, was)| now - was > Self::NOISE_FLOOR_MS)
        };
        let median_bad =
            dm.is_some_and(|d| d > self.median_limit_pct) && over_floor(median, "median_ms");
        let p90_bad = dp.is_some_and(|d| d > self.p90_limit_pct) && over_floor(p90, "p90_ms");
        let detail = format!(
            "median {} (limit +{:.1}%) · p90 {} (limit +{:.1}%)",
            dm.map(|d| format!("{d:+.1}%"))
                .unwrap_or_else(|| "—".into()),
            self.median_limit_pct,
            dp.map(|d| format!("{d:+.1}%"))
                .unwrap_or_else(|| "—".into()),
            self.p90_limit_pct,
        );
        if median_bad || p90_bad {
            (Verdict::fail(format!("REGRESSED — {detail}")), summary)
        } else {
            (Verdict::pass(detail), summary)
        }
    }
}
