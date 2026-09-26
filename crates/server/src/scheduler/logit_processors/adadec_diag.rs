// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: AdaDec diagnostic: logs the Shannon entropy of the logits after the pipeline stages.
//!
//! [`super::run_pipeline_with_path`] calls [`log_step`] after its last stage,
//! [`super::grammar_bitmask::GrammarBitmaskApply`], so any tokens the grammar
//! masked are already `-inf`. It is not called when a stage returns `EmitToken`.
//! The sink is `RunDumps::adadec`, opened when the run starts:
//!
//!   * `METRALE_ADADEC_DIAGNOSTIC` unset or empty, or the file cannot be
//!     opened → no sink, and `log_step` returns at once;
//!   * otherwise → one JSONL record per call, appended to
//!     `<dir>/adadec_entropy.jsonl`.
//!
//! Owner: scheduler.
//! Invariants:
//! - `log_step` never writes `logits`.
//!
//! Each record also carries `"p"`, the path label (`"decode"` or `"verify"`);
//! `"t"` is `output_tokens.len()`. Record schema (per token):
//! ```json
//! {
//!   "t": 12345,                   // sequence offset (post-prefill)
//!   "h": 0.823,                   // Shannon entropy (nats) over masked dist
//!   "topk_ids":   [27, 9, 4321],  // top-3 grammar-legal token ids
//!   "topk_logits":[ 8.1, 7.6, 7.2],
//!   "thk": false,                 // inside_thinking
//!   "pb":  true,                  // inside_parameter_body
//!   "pbc": 17                     // param_body_chars_emitted
//! }
//! ```

use super::{LogitsContext, LogitsProcessor, ProcessorOutcome};
use crate::scheduler::ActiveSeq;
use std::io::Write;

/// 2026-09-25: A [`LogitsProcessor`] that calls [`log_step`] with the `"verify"`
/// label. It is not in the stage list of `run_pipeline_with_path`, which
/// calls `log_step` directly.
pub struct AdaDecDiagnostic;

/// 2026-09-25: Shannon entropy (nats) of the softmax over `logits`, in log-sum-exp
/// form. Non-finite logits are skipped: a masked `-inf` token contributes 0,
/// and `-inf * 0` would otherwise make the sum NaN.
fn masked_entropy(logits: &[f32]) -> f32 {
    // 2026-09-25: Pass 1: the largest finite logit, for log-sum-exp stability.
    let mut max = f32::NEG_INFINITY;
    for &l in logits {
        if l.is_finite() && l > max {
            max = l;
        }
    }
    if !max.is_finite() {
        // 2026-09-25: No finite logit: report 0 entropy.
        return 0.0;
    }
    // 2026-09-25: Pass 2: Z = Σ exp(l - max) and Σ l * exp(l - max).
    // Entropy H = ln(Z) + max - (Σ l_i * exp(l_i - max)) / Z   [in nats]
    let mut z: f64 = 0.0;
    let mut weighted_logit_sum: f64 = 0.0;
    for &l in logits {
        if !l.is_finite() {
            continue;
        }
        let e = ((l - max) as f64).exp();
        z += e;
        weighted_logit_sum += (l as f64) * e;
    }
    if z <= 0.0 {
        return 0.0;
    }
    let log_z_plus_max = z.ln() + (max as f64);
    let expectation = weighted_logit_sum / z;
    (log_z_plus_max - expectation) as f32
}

/// 2026-09-25: The `k` largest finite logits with their ids, in descending order.
/// Called with `k = 3`.
fn top_k(logits: &[f32], k: usize) -> Vec<(u32, f32)> {
    let mut out: Vec<(u32, f32)> = Vec::with_capacity(k);
    for (i, &l) in logits.iter().enumerate() {
        if !l.is_finite() {
            continue;
        }
        if out.len() < k {
            out.push((i as u32, l));
            // 2026-09-25: Keep sorted descending by logit.
            out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        } else if l > out[k - 1].1 {
            out[k - 1] = (i as u32, l);
            out.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        }
    }
    out
}

/// 2026-09-25: Append one JSONL record for this position to `sink`; return at once
/// when `sink` is `None`. Write errors are ignored.
pub fn log_step(
    sink: Option<&std::sync::Mutex<std::fs::File>>,
    logits: &[f32],
    seq: &ActiveSeq,
    path: &'static str,
) {
    let Some(mtx) = sink else {
        return;
    };

    let h = masked_entropy(logits);
    let tk = top_k(logits, 3);

    let topk_ids: Vec<u32> = tk.iter().map(|(i, _)| *i).collect();
    let topk_logits: Vec<f32> = tk.iter().map(|(_, l)| *l).collect();

    let record = serde_json::json!({
        "t":    seq.output_tokens.len(),
        "h":    h,
        "topk_ids":    topk_ids,
        "topk_logits": topk_logits,
        "thk":  seq.inside_thinking,
        "pb":   seq.inside_parameter_body,
        "pbc":  seq.param_body_chars_emitted,
        "p":    path,
    });

    if let Ok(mut f) = mtx.lock() {
        let mut line = serde_json::to_string(&record).unwrap_or_default();
        line.push('\n');
        let _ = f.write_all(line.as_bytes());
    }
}

impl LogitsProcessor for AdaDecDiagnostic {
    fn apply(
        &self,
        logits: &mut [f32],
        seq: &mut ActiveSeq,
        ctx: &LogitsContext,
    ) -> ProcessorOutcome {
        log_step(ctx.tel.dumps().adadec.as_ref(), logits, seq, "verify");
        ProcessorOutcome::Continue
    }

    fn name(&self) -> &'static str {
        "adadec_diag"
    }

    fn is_argmax_invariant(&self) -> bool {
        true
    }
}
