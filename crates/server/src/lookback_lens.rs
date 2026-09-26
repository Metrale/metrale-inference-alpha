// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Reranking of candidate tokens by a linear classifier over
//! per-head attention ratios: the share of each head's attention that falls
//! on a lookback span, such as the latest tool result. Nothing outside this
//! module constructs `AttentionSums` or calls `rerank`.
//!
//! Owner: server.
//! Invariants: a `GroundedClassifier` has at least one weight, and `rerank`
//! returns `None` or an index in `1..candidates.len()`.

/// 2026-09-26: For one candidate, per head `h`: `lookback_sums[h]` is the
/// attention mass on the lookback span and `rest_sums[h]` the mass on every
/// other position.
#[derive(Debug, Clone)]
pub struct AttentionSums {
    pub lookback_sums: Vec<f32>,
    pub rest_sums: Vec<f32>,
}

impl AttentionSums {
    /// 2026-09-26: Per head, `lookback / (lookback + rest)`, or 0 when that
    /// sum is at most 1e-10; empty when the two vectors differ in length.
    pub fn ratios(&self) -> Vec<f32> {
        if self.lookback_sums.len() != self.rest_sums.len() {
            return Vec::new();
        }
        self.lookback_sums
            .iter()
            .zip(self.rest_sums.iter())
            .map(|(lb, rest)| {
                let denom = lb + rest;
                if denom > 1e-10 { lb / denom } else { 0.0 }
            })
            .collect()
    }

    pub fn num_heads(&self) -> usize {
        self.lookback_sums.len()
    }
}

/// 2026-09-26: Linear classifier from per-head ratios to a grounded
/// probability.
#[derive(Debug, Clone)]
pub struct GroundedClassifier {
    weights: Vec<f32>,
    bias: f32,
}

impl GroundedClassifier {
    pub fn new(weights: Vec<f32>, bias: f32) -> Option<Self> {
        if weights.is_empty() {
            return None;
        }
        Some(Self { weights, bias })
    }

    /// 2026-09-26: `sigmoid(w · ratios + b)`, or 0.5 when `ratios` and the
    /// weights differ in length.
    pub fn grounded_prob(&self, ratios: &[f32]) -> f32 {
        if ratios.len() != self.weights.len() {
            return 0.5;
        }
        let dot: f32 = ratios
            .iter()
            .zip(self.weights.iter())
            .map(|(r, w)| r * w)
            .sum();
        let z = dot + self.bias;
        1.0 / (1.0 + (-z).exp())
    }
}

/// 2026-09-26: `candidates[0]` is the model's argmax, and
/// `per_candidate_sums[i]` belongs to `candidates[i]`. Returns the index of
/// the candidate with the highest grounded probability (the earliest on a
/// tie) when that is not 0 and beats candidate 0 by at least `min_gap`;
/// `None` otherwise, and for empty or mismatched inputs.
pub fn rerank(
    candidates: &[u32],
    per_candidate_sums: &[AttentionSums],
    classifier: &GroundedClassifier,
    min_gap: f32,
) -> Option<usize> {
    if candidates.is_empty() || per_candidate_sums.is_empty() {
        return None;
    }
    if candidates.len() != per_candidate_sums.len() {
        return None;
    }
    let scores: Vec<f32> = per_candidate_sums
        .iter()
        .map(|s| classifier.grounded_prob(&s.ratios()))
        .collect();
    let mut best = 0usize;
    let mut best_score = scores[0];
    for (i, &s) in scores.iter().enumerate() {
        if s > best_score {
            best_score = s;
            best = i;
        }
    }
    if best == 0 {
        return None;
    }
    let baseline = scores[0];
    if best_score - baseline < min_gap {
        return None;
    }
    Some(best)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ratios_in_unit_interval() {
        let s = AttentionSums {
            lookback_sums: vec![0.3, 0.7, 0.0],
            rest_sums: vec![0.7, 0.3, 1.0],
        };
        let r = s.ratios();
        assert_eq!(r.len(), 3);
        assert!((r[0] - 0.3).abs() < 1e-5);
        assert!((r[1] - 0.7).abs() < 1e-5);
        assert!((r[2] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn ratios_returns_empty_on_dim_mismatch() {
        let s = AttentionSums {
            lookback_sums: vec![0.5, 0.5],
            rest_sums: vec![0.5],
        };
        assert!(s.ratios().is_empty());
    }

    #[test]
    fn grounded_prob_sigmoid_range() {
        let c = GroundedClassifier::new(vec![1.0, 1.0, 1.0], 0.0).unwrap();
        let p = c.grounded_prob(&[0.5, 0.5, 0.5]);
        assert!(p > 0.0 && p < 1.0);
    }

    #[test]
    fn rerank_picks_best_when_above_threshold() {
        let c = GroundedClassifier::new(vec![10.0], 0.0).unwrap();
        let s_lo = AttentionSums {
            lookback_sums: vec![0.1],
            rest_sums: vec![0.9],
        };
        let s_hi = AttentionSums {
            lookback_sums: vec![0.9],
            rest_sums: vec![0.1],
        };
        let candidates = vec![100u32, 200];
        let picked = rerank(&candidates, &[s_lo, s_hi], &c, 0.1);
        assert_eq!(picked, Some(1));
    }

    #[test]
    fn rerank_returns_none_when_argmax_already_best() {
        let c = GroundedClassifier::new(vec![10.0], 0.0).unwrap();
        let s_hi = AttentionSums {
            lookback_sums: vec![0.9],
            rest_sums: vec![0.1],
        };
        let s_lo = AttentionSums {
            lookback_sums: vec![0.1],
            rest_sums: vec![0.9],
        };
        let picked = rerank(&[100, 200], &[s_hi, s_lo], &c, 0.1);
        assert!(picked.is_none(), "no rerank needed when argmax is grounded");
    }
}
