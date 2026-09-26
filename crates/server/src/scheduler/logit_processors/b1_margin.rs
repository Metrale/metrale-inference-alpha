// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: B1 margin observer: counts low top-1/top-2 logit gaps inside tool parameter bodies.
//!
//! `super::process_position_logits` is the only caller, and calls it for the
//! final decode position only (`PositionKind::FinalDecode`), so verify
//! positions that may be rejected are not counted.
//!
//! Owner: scheduler.
//! Invariants:
//! - `observe` never writes `logits`.

use crate::scheduler::ActiveSeq;

/// 2026-09-25: One warn line per this many low-margin positions.
const B1_SUMMARY_PERIOD: u64 = 100;
/// 2026-09-25: A top-1 minus top-2 logit gap below this is "low margin". The logit
/// gap equals the logprob gap.
const LOW_MARGIN_THRESHOLD: f32 = 1.5;

fn record_low_margin(
    margin: f32,
    top1: u32,
    top2: u32,
    stats: &metrale_speculative::spec_stats::SpecStats,
) {
    let n = metrale_speculative::spec_stats::bump(&stats.b1_low_margin) + 1;
    // 2026-09-25: One trace line per low-margin position; the tracing target is
    // this module's path, `met::scheduler::logit_processors::b1_margin`.
    tracing::trace!("B1 low margin: gap={margin:.3} top1={top1} top2={top2}");
    if n.is_multiple_of(B1_SUMMARY_PERIOD) {
        tracing::warn!(
            "B1 drift gauge: {n} low-margin (<1.5 logprobs) decode positions \
             observed inside parameter bodies. \
             FP8 numerical noise is in the argmax-flip regime — consider \
             reviewing tool-arg outputs for whitespace / digit-collapse drift."
        );
    }
}

/// 2026-09-25: Find the top-1/top-2 gap of the logits after the pipeline stages, and
/// record a low-margin position when the sequence is inside a parameter body
/// that already has emitted characters. `logits` is read, never written.
pub(super) fn observe(
    logits: &[f32],
    a: &ActiveSeq,
    stats: &metrale_speculative::spec_stats::SpecStats,
) {
    // 2026-09-25: One scan for top-1 and top-2, before penalties and bias.
    let (top1_idx, top1_val, top2_idx, top2_val) = {
        let mut t1_idx = 0u32;
        let mut t1_val = f32::NEG_INFINITY;
        let mut t2_idx = 0u32;
        let mut t2_val = f32::NEG_INFINITY;
        for (idx, &v) in logits.iter().enumerate() {
            if v > t1_val {
                t2_val = t1_val;
                t2_idx = t1_idx;
                t1_val = v;
                t1_idx = idx as u32;
            } else if v > t2_val {
                t2_val = v;
                t2_idx = idx as u32;
            }
        }
        (t1_idx, t1_val, t2_idx, t2_val)
    };
    let margin = top1_val - top2_val;
    let low_margin_in_body =
        a.inside_parameter_body && a.param_body_chars_emitted > 0 && margin < LOW_MARGIN_THRESHOLD;
    if low_margin_in_body {
        record_low_margin(margin, top1_idx, top2_idx, stats);
    }
}
