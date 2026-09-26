// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Host-side MoE routing policies: entropy-gated top-k widening after LASER (arXiv:2510.03293) and a coverage-based expert budget after MoE-Spec (arXiv:2602.16052).
//!
//! Nothing outside this file's tests calls these functions.
//!
//! Owner: server.
//! Invariants: none beyond the types.

/// 2026-09-26: Shannon entropy (natural log) of a router distribution; terms
/// with p <= 1e-10 are skipped. `scores` are softmax probabilities.
pub fn router_entropy(scores: &[f32]) -> f32 {
    let mut h = 0.0f32;
    for &p in scores {
        if p > 1e-10 {
            h -= p * p.ln();
        }
    }
    h
}

/// 2026-09-26: The number of experts to route to: `expanded_top_k` when
/// `router_h` is at least 0.7 * ln(num_experts), otherwise `base_top_k`; both
/// capped at `num_experts`.
pub fn laser_top_k(
    router_h: f32,
    num_experts: usize,
    base_top_k: usize,
    expanded_top_k: usize,
) -> usize {
    let h_max = (num_experts as f32).ln();
    let threshold = 0.7 * h_max;
    if router_h >= threshold {
        expanded_top_k.min(num_experts)
    } else {
        base_top_k.min(num_experts)
    }
}

/// 2026-09-26: The shortest prefix of `weights`, sorted by descending weight,
/// whose sum reaches `coverage` (clamped to 0..=1) of the total. Empty when the
/// input is empty or the total is not positive.
pub fn budget_experts(weights: &[(u32, f32)], coverage: f32) -> Vec<(u32, f32)> {
    if weights.is_empty() {
        return Vec::new();
    }
    let coverage = coverage.clamp(0.0, 1.0);
    let total: f32 = weights.iter().map(|(_, w)| *w).sum();
    if total <= 0.0 {
        return Vec::new();
    }
    let target = total * coverage;
    let mut sorted: Vec<(u32, f32)> = weights.to_vec();
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut cum = 0.0f32;
    let mut out: Vec<(u32, f32)> = Vec::with_capacity(sorted.len());
    for (eid, w) in sorted {
        out.push((eid, w));
        cum += w;
        if cum >= target {
            break;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entropy_uniform_equals_log_n() {
        let n = 8;
        let p = 1.0 / n as f32;
        let probs = vec![p; n];
        let h = router_entropy(&probs);
        assert!((h - (n as f32).ln()).abs() < 1e-3);
    }

    #[test]
    fn entropy_one_hot_is_zero() {
        let probs = vec![1.0, 0.0, 0.0, 0.0];
        let h = router_entropy(&probs);
        assert!(h < 1e-3);
    }

    #[test]
    fn laser_stays_topk_on_sharp_distribution() {
        let probs = vec![0.95, 0.03, 0.01, 0.01];
        let h = router_entropy(&probs);
        let k = laser_top_k(h, 4, 2, 4);
        assert_eq!(k, 2, "sharp distribution → stay top-2");
    }

    #[test]
    fn laser_expands_on_flat_distribution() {
        let probs = vec![0.25, 0.25, 0.25, 0.25];
        let h = router_entropy(&probs);
        let k = laser_top_k(h, 4, 2, 4);
        assert_eq!(k, 4, "flat distribution → expand to 4");
    }

    #[test]
    fn budget_experts_covers_target_mass() {
        let weights = vec![
            (1u32, 0.4),
            (2, 0.3),
            (3, 0.2),
            (4, 0.05),
            (5, 0.03),
            (6, 0.02),
        ];
        let kept = budget_experts(&weights, 0.9);
        // 2026-09-26: The top three weights sum to 0.9.
        assert!(kept.len() <= 3, "got {} experts: {:?}", kept.len(), kept);
        let cum: f32 = kept.iter().map(|(_, w)| *w).sum();
        assert!(cum >= 0.9 - 1e-3);
    }

    #[test]
    fn budget_experts_full_coverage_returns_all() {
        let weights = vec![(1u32, 0.5), (2, 0.3), (3, 0.2)];
        let kept = budget_experts(&weights, 1.0);
        assert_eq!(kept.len(), 3);
    }

    #[test]
    fn budget_experts_empty_input_returns_empty() {
        let kept = budget_experts(&[], 0.95);
        assert!(kept.is_empty());
    }
}
