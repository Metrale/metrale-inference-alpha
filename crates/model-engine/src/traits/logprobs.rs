// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Log-softmax and top-k logprob extraction, used by the scheduler's decode-time
//! logprobs (`logprob_of`) and by the prompt-logprob collection during prefill (`extract_bf16`).
//!
//! Owner: model-engine.
//! Invariants: none beyond the types.

/// 2026-09-25: One scored prompt position: `log P(tokens[i+1] | tokens[..=i])` plus the top-k
/// alternatives under the same distribution.
#[derive(Clone, Debug)]
pub struct PromptTokenLogprob {
    /// 2026-09-25: The next prompt token, the one scored.
    pub token_id: u32,
    pub logprob: f32,
    /// 2026-09-25: Top-k `(token_id, logprob)` alternatives, highest first; empty when `k == 0`.
    pub top: Vec<(u32, f32)>,
}

/// 2026-09-25: Log-softmax over `f32_logits`: the target token's logprob and the top-k
/// alternatives, highest first. `k == 0` returns no alternatives. A target outside the vocab,
/// or empty logits, gives `-inf` rather than a panic.
pub fn logprob_of(f32_logits: &[f32], target: u32, k: usize) -> (f32, Vec<(u32, f32)>) {
    if f32_logits.is_empty() {
        return (f32::NEG_INFINITY, Vec::new());
    }
    let max_logit = f32_logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let log_sum_exp = max_logit
        + f32_logits
            .iter()
            .map(|&l| (l - max_logit).exp())
            .sum::<f32>()
            .ln();
    let target_logprob = if (target as usize) < f32_logits.len() {
        f32_logits[target as usize] - log_sum_exp
    } else {
        f32::NEG_INFINITY
    };
    if k == 0 {
        return (target_logprob, Vec::new());
    }
    let mut indexed: Vec<(u32, f32)> = f32_logits
        .iter()
        .enumerate()
        .map(|(j, &l)| (j as u32, l - log_sum_exp))
        .collect();
    let nth = k.min(indexed.len().saturating_sub(1));
    indexed.select_nth_unstable_by(nth, |a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut top: Vec<(u32, f32)> = indexed[..k.min(indexed.len())].to_vec();
    top.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    (target_logprob, top)
}

/// 2026-09-25: BF16 to FP32: the two little-endian bytes become the upper 16 bits of an f32.
#[inline]
pub fn bf16_to_f32(lo: u8, hi: u8) -> f32 {
    f32::from_bits(((lo as u32) | ((hi as u32) << 8)) << 16)
}

/// 2026-09-25: One position's `PromptTokenLogprob` from a row of `vocab` BF16 logits.
pub fn extract_bf16(bf16: &[u8], target: u32, k: usize, vocab: usize) -> PromptTokenLogprob {
    let f32_logits: Vec<f32> = (0..vocab)
        .map(|j| bf16_to_f32(bf16[j * 2], bf16[j * 2 + 1]))
        .collect();
    let (logprob, top) = logprob_of(&f32_logits, target, k);
    PromptTokenLogprob {
        token_id: target,
        logprob,
        top,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logprob_matches_manual_log_softmax() {
        let logits = [1.0f32, 2.0, 3.0];
        let sum: f32 = logits.iter().map(|l| l.exp()).sum();
        let expect = 2.0 - sum.ln();
        let (lp, top) = logprob_of(&logits, 1, 2);
        assert!((lp - expect).abs() < 1e-6, "{lp} vs {expect}");
        assert_eq!(top[0].0, 2);
        assert_eq!(top[1].0, 1);
        assert!(top[0].1 > top[1].1);
    }

    #[test]
    fn k_zero_returns_empty_top() {
        let (lp, top) = logprob_of(&[0.0, 1.0], 0, 0);
        assert!(top.is_empty());
        assert!(lp < 0.0);
    }

    #[test]
    fn out_of_vocab_target_is_neg_inf_not_panic() {
        let (lp, _) = logprob_of(&[0.0, 1.0], 99, 1);
        assert_eq!(lp, f32::NEG_INFINITY);
    }

    #[test]
    fn empty_vocab_is_neg_inf_with_no_alternatives() {
        assert_eq!(logprob_of(&[], 0, 4), (f32::NEG_INFINITY, vec![]));
    }

    #[test]
    fn bf16_slice_roundtrip_extract() {
        let vals = [0.0f32, 1.0, 2.0];
        let mut bytes = Vec::new();
        for v in vals {
            let b = (v.to_bits() >> 16) as u16;
            bytes.push((b & 0xFF) as u8);
            bytes.push((b >> 8) as u8);
        }
        let r = extract_bf16(&bytes, 2, 1, 3);
        let sum: f32 = vals.iter().map(|l| l.exp()).sum();
        let expect = 2.0 - sum.ln();
        assert_eq!(r.token_id, 2);
        assert!((r.logprob - expect).abs() < 1e-3);
        assert_eq!(r.top[0].0, 2);
        assert!((r.top[0].1 - expect).abs() < 1e-3);
    }
}
