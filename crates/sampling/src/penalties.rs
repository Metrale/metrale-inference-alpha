// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The host penalty stage: the LZ and DRY penalties and
//! `apply_penalties_and_bias`, which applies every penalty and then the
//! per-token logit bias. Re-exported at the crate root.
//!
//! Owner: metrale-sampling.
//! Invariants:
//! - Every step of `apply_penalties_and_bias` is gated on its own
//!   `SamplingParams` field, so neutral params leave the logits unchanged.

use super::SamplingParams;

/// 2026-09-25: LZ penalty over the last `LZ_WINDOW` (256) tokens of `history`.
///
/// For each token in that window and each n-gram length 3, 4 and 5, count the
/// earlier n-grams that equal the last n-1 tokens followed by that token, and
/// subtract `penalty * (ngram_len - 2) * count` from its logit. Token ids at
/// or above `logits.len()` are skipped.
pub fn apply_lz_penalty(logits: &mut [f32], history: &[u32], penalty: f32) {
    use std::collections::HashSet;
    const LZ_WINDOW: usize = 256;
    let history = if history.len() > LZ_WINDOW {
        &history[history.len() - LZ_WINDOW..]
    } else {
        history
    };
    let n = logits.len();
    // 2026-09-25: A token absent from the window cannot end a matching n-gram.
    let token_set: HashSet<u32> = history.iter().copied().collect();
    for &candidate in &token_set {
        if (candidate as usize) >= n {
            continue;
        }
        for ngram_len in 3..=5usize {
            if history.len() < ngram_len {
                continue;
            }
            // 2026-09-25: The n-gram that would form: the last ngram_len-1 tokens,
            // then `candidate`.
            let suffix = &history[history.len() - (ngram_len - 1)..];
            let count = history
                .windows(ngram_len)
                .filter(|w| w[..ngram_len - 1] == *suffix && w[ngram_len - 1] == candidate)
                .count();
            if count > 0 {
                logits[candidate as usize] -= penalty * (ngram_len as f32 - 2.0) * count as f32;
            }
        }
    }
}

/// 2026-09-25: DRY ("don't repeat yourself") penalty. No-op for an empty
/// history or `multiplier == 0.0`.
///
/// For each earlier position `i`, `match_lengths[i]` is the number of tokens
/// that the history ending at `i` shares with the end of the whole history,
/// counted backwards and stopping after a breaker (0 when `history[i]` is a
/// breaker). When that length `len` exceeds `allowed_length`, the logit of
/// `history[i + len]` is reduced by `multiplier * base^(len - allowed_length)`.
pub fn apply_dry_penalty(
    logits: &mut [f32],
    history: &[u32],
    multiplier: f32,
    base: f32,
    allowed_length: u32,
    breakers: &[u32],
) {
    if history.is_empty() || multiplier == 0.0 {
        return;
    }
    let n = logits.len();
    let hist_len = history.len();
    let allowed = allowed_length as usize;

    let mut match_lengths = vec![0usize; hist_len];
    for i in (0..hist_len.saturating_sub(1)).rev() {
        if breakers.contains(&history[i]) {
            match_lengths[i] = 0;
            continue;
        }
        let mut len = 0;
        let mut j = i;
        let mut k = hist_len - 1;
        while j < k && history[j] == history[k] {
            len += 1;
            if breakers.contains(&history[j]) {
                break;
            }
            if j == 0 {
                break;
            }
            j -= 1;
            k -= 1;
        }
        match_lengths[i] = len;
    }

    #[allow(clippy::needless_range_loop)]
    for i in 0..hist_len.saturating_sub(1) {
        let len = match_lengths[i];
        if len > allowed {
            let extend_pos = i + len;
            if extend_pos < hist_len {
                let token = history[extend_pos] as usize;
                if token < n {
                    let penalty = multiplier * base.powi((len - allowed) as i32);
                    logits[token] -= penalty;
                }
            }
        }
    }
}

/// 2026-09-25: Apply the repetition, presence, frequency, LZ and DRY penalties,
/// then the per-token logit bias, to `logits` in place, using `token_history`.
///
/// Called by `sample_with_params_seeded` and by the server's
/// `process_position_logits` (scheduler/logit_processors/mod.rs). It changes
/// nothing when `repetition_penalty` is 1.0 (or <= 0.0), both additive
/// penalties are 0.0, `lz_penalty` and `dry_multiplier` are <= 0.0, and
/// `logit_bias` is empty: every step is gated on its own parameter.
pub fn apply_penalties_and_bias(
    logits: &mut [f32],
    params: &SamplingParams,
    token_history: &[u32],
) {
    let n = logits.len();

    // 2026-09-25: Repetition penalty over the last `repetition_penalty_window`
    // tokens (all of them when the window is 0 or covers the history). A value
    // <= 0.0 is skipped: dividing by 0.0 would turn positive logits into inf.
    let rep_penalty = params.repetition_penalty;
    if rep_penalty != 1.0 && rep_penalty > 0.0 && !token_history.is_empty() {
        let window = params.repetition_penalty_window as usize;
        let effective = if window > 0 && window < token_history.len() {
            &token_history[token_history.len() - window..]
        } else {
            token_history
        };
        for &tid in effective {
            if (tid as usize) < n {
                let logit = &mut logits[tid as usize];
                if *logit > 0.0 {
                    *logit /= rep_penalty;
                } else {
                    *logit *= rep_penalty;
                }
            }
        }
    }

    // 2026-09-25: Additive penalties over the same window. For a token with
    // count c in the window: z' = z - frequency_penalty * c - presence_penalty.
    let freq_pen = params.frequency_penalty;
    let pres_pen = params.presence_penalty;
    if (freq_pen != 0.0 || pres_pen != 0.0) && !token_history.is_empty() {
        let window = params.repetition_penalty_window as usize;
        let effective = if window > 0 && window < token_history.len() {
            &token_history[token_history.len() - window..]
        } else {
            token_history
        };
        let mut counts = std::collections::HashMap::<u32, u32>::new();
        for &tid in effective {
            *counts.entry(tid).or_insert(0) += 1;
        }
        for (&tid, &count) in &counts {
            if (tid as usize) < n {
                logits[tid as usize] -= freq_pen * count as f32 + pres_pen;
            }
        }
    }

    if params.lz_penalty > 0.0 && token_history.len() >= 4 {
        apply_lz_penalty(logits, token_history, params.lz_penalty);
    }

    if params.dry_multiplier > 0.0 && token_history.len() >= 3 {
        apply_dry_penalty(
            logits,
            token_history,
            params.dry_multiplier,
            params.dry_base,
            params.dry_allowed_length,
            &params.dry_sequence_breakers,
        );
    }

    for &(tid, bias) in &params.logit_bias {
        if (tid as usize) < n {
            logits[tid as usize] += bias;
        }
    }
}
