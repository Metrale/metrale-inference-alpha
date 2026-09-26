// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The high-speed-swap predictor's random Gaussian projection `P`, built
//! on the host from a seed; `Predictor` uploads it once.
//!
//! Owner: storage, high-speed swap.
//! Invariants:
//! - `build_projection` is a pure function of its shape and seed.

use half::bf16;
use rand::SeedableRng;
use rand::distributions::Distribution;
use rand_chacha::ChaCha8Rng;
use rand_distr::StandardNormal;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PredictorShape {
    pub head_dim: usize,
    pub r: usize,
}

impl PredictorShape {
    pub fn new(head_dim: usize, r: usize) -> Self {
        assert!(head_dim > 0 && r > 0, "head_dim/r must be positive");
        assert!(head_dim <= 256, "MAX_HEAD_DIM=256 in kv_lowrank_project.cu");
        assert!(r <= 128, "predictor_score block dim caps r at 128");
        Self { head_dim, r }
    }
}

/// 2026-09-25: Random Gaussian projection `P`, `[head_dim, r]` row-major BF16, from a
/// ChaCha8 stream seeded with `seed`. Entries have variance `1 / head_dim`, so for a
/// key `k` a projected coordinate has variance `|k|^2 / head_dim`.
pub fn build_projection(shape: PredictorShape, seed: u64) -> Vec<bf16> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let inv_sqrt_d = 1.0_f32 / (shape.head_dim as f32).sqrt();
    let dist = StandardNormal;
    let n = shape.head_dim * shape.r;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let v: f32 = dist.sample(&mut rng);
        out.push(bf16::from_f32(v * inv_sqrt_d));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn determinism() {
        let s = PredictorShape::new(128, 32);
        let a = build_projection(s, 0xCAFE_F00D);
        let b = build_projection(s, 0xCAFE_F00D);
        assert_eq!(a, b);
        let c = build_projection(s, 0xDEAD_BEEF);
        assert_ne!(a, c);
    }

    #[test]
    fn shape() {
        let s = PredictorShape::new(128, 32);
        let p = build_projection(s, 1);
        assert_eq!(p.len(), 128 * 32);
    }
}
