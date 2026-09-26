// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: A linear probe that scores a hidden-state vector as a
//! hallucination probability, `sigmoid(w · h + b)`.
//!
//! Owner: server.
//! Invariants: a `LinearProbe` has at least one weight.
//!
//! Nothing outside this module uses it: no hidden state reaches the host for
//! it, and no response carries its result.

/// 2026-09-26: Probe weights and bias. `logit` expects a hidden vector with
/// one element per weight.
#[derive(Debug, Clone)]
pub struct LinearProbe {
    weights: Vec<f32>,
    bias: f32,
}

impl LinearProbe {
    /// 2026-09-26: `None` when `weights` is empty.
    pub fn new(weights: Vec<f32>, bias: f32) -> Option<Self> {
        if weights.is_empty() {
            return None;
        }
        Some(Self { weights, bias })
    }

    /// 2026-09-26: `w · h + b`. When `hidden` and the weights differ in
    /// length, just `b`.
    pub fn logit(&self, hidden: &[f32]) -> f32 {
        if hidden.len() != self.weights.len() {
            return self.bias;
        }
        let dot: f32 = hidden
            .iter()
            .zip(self.weights.iter())
            .map(|(h, w)| h * w)
            .sum();
        dot + self.bias
    }

    /// 2026-09-26: `sigmoid(logit(hidden))`, in [0, 1].
    pub fn probability(&self, hidden: &[f32]) -> f32 {
        let z = self.logit(hidden);
        1.0 / (1.0 + (-z).exp())
    }

    /// 2026-09-26: The hidden-vector length this probe expects.
    pub fn hidden_dim(&self) -> usize {
        self.weights.len()
    }
}

/// 2026-09-26: One token's probe result. Nothing constructs it.
#[derive(Debug, Clone)]
pub struct ConfidenceSample {
    pub token_id: u32,
    /// 2026-09-26: Hallucination probability in [0, 1]; higher means more
    /// likely fabricated.
    pub p_halluc: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_weights_rejected() {
        let p = LinearProbe::new(Vec::new(), 0.0);
        assert!(p.is_none());
    }

    #[test]
    fn logit_computes_dot_plus_bias() {
        let p = LinearProbe::new(vec![1.0, 2.0, -1.0], 0.5).unwrap();
        let z = p.logit(&[1.0, 0.0, 1.0]);
        assert!((z - 0.5).abs() < 1e-5);
    }

    #[test]
    fn dim_mismatch_returns_bias_only() {
        let p = LinearProbe::new(vec![1.0, 2.0], 0.7).unwrap();
        let z = p.logit(&[1.0, 2.0, 3.0]);
        assert!((z - 0.7).abs() < 1e-5);
    }

    #[test]
    fn probability_in_unit_interval() {
        let p = LinearProbe::new(vec![1.0, -1.0, 2.0], 0.0).unwrap();
        let pr = p.probability(&[0.5, 0.3, 0.7]);
        assert!((0.0..=1.0).contains(&pr));
    }

    #[test]
    fn extreme_logit_saturates() {
        let p = LinearProbe::new(vec![100.0, 100.0], 0.0).unwrap();
        let pr_high = p.probability(&[1.0, 1.0]);
        assert!(pr_high > 0.99);
        let pr_low = p.probability(&[-1.0, -1.0]);
        assert!(pr_low < 0.01);
    }

    #[test]
    fn hidden_dim_reports_weights_length() {
        let p = LinearProbe::new(vec![0.0; 2048], 0.0).unwrap();
        assert_eq!(p.hidden_dim(), 2048);
    }
}
