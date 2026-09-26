// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Per-step logit dump (`METRALE_LOGIT_DUMP=<file>`): one JSONL line per sampled decode position.
//!
//! `decode_logits_seq::process_seq_logits` calls [`record`] only when the run
//! has this sink, and only for a sampled token (not for the forced-token or
//! `METRALE_FORCE_TEMP_ZERO` returns). It passes the logits as they left
//! `process_position_logits`: already masked, penalised and biased. Fields:
//!   - `step`: `output_tokens.len()` before this token.
//!   - `raw_topk`: the 12 largest (id, logit) pairs of those logits.
//!   - `bias`: the `logit_bias` list of this position's `SamplingParams`.
//!   - `post_argmax`: the argmax after adding `bias` to those logits again.
//!   - `sampled`: the sampled token.
//!   - `in_body` / `chars`: whether the sequence is inside a tool parameter
//!     body, and how many body characters it has emitted.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.

use std::io::Write;

/// 2026-09-25: The top-`k` (index, logit) pairs by logit, descending.
fn top_k(logits: &[f32], k: usize) -> Vec<(u32, f32)> {
    let mut idx: Vec<u32> = (0..logits.len() as u32).collect();
    let kk = k.min(idx.len());
    idx.select_nth_unstable_by(kk.saturating_sub(1).max(0), |&a, &b| {
        logits[b as usize]
            .partial_cmp(&logits[a as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut top: Vec<(u32, f32)> = idx
        .into_iter()
        .take(kk)
        .map(|i| (i, logits[i as usize]))
        .collect();
    top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    top
}

/// 2026-09-25: Append one record to `sink` and flush it. Lock and write errors are
/// ignored.
#[allow(clippy::too_many_arguments)]
pub(crate) fn record(
    sink: &std::sync::Mutex<std::io::BufWriter<std::fs::File>>,
    step: usize,
    in_body: bool,
    chars: usize,
    raw_logits: &[f32],
    bias: &[(u32, f32)],
    sampled: u32,
) {
    let w = sink;
    const K: usize = 12;
    let raw_topk = top_k(raw_logits, K);
    // 2026-09-25: Argmax of `raw_logits` plus `bias`.
    let mut post_best = (0u32, f32::NEG_INFINITY);
    for (i, &l) in raw_logits.iter().enumerate() {
        let mut v = l;
        for &(tok, d) in bias {
            if tok as usize == i {
                v += d;
            }
        }
        if v > post_best.1 {
            post_best = (i as u32, v);
        }
    }
    let mut s = String::with_capacity(256);
    s.push_str(&format!(
        "{{\"step\":{step},\"in_body\":{in_body},\"chars\":{chars},\"sampled\":{sampled},\"post_argmax\":{},\"raw_topk\":[",
        post_best.0
    ));
    for (n, (id, lg)) in raw_topk.iter().enumerate() {
        if n > 0 {
            s.push(',');
        }
        s.push_str(&format!("[{id},{lg:.4}]"));
    }
    s.push_str("],\"bias\":[");
    for (n, (id, d)) in bias.iter().enumerate() {
        if n > 0 {
            s.push(',');
        }
        s.push_str(&format!("[{id},{d:.4}]"));
    }
    s.push_str("]}\n");
    if let Ok(mut guard) = w.lock() {
        let _ = guard.write_all(s.as_bytes());
        let _ = guard.flush();
    }
}
