// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Host tests of the M16 tier's accumulation floor
//! ([`m16_tc_acc_floor`]): it scales with the reduction depth and the row's
//! RMS, admits the cancelled LM-head logit the fixed `2^-20 * rms` floor
//! rejected, and still rejects structural errors.
//!
//! Measured 2026-09-11 on H100 by `native_bf16_lm_head_m16_microtest` on the
//! `[N=248077, K=5120]` BF16 head: the worst rejected element had
//! `reference = -1.173019409e-4` at 100 ordinal BF16 ULP, in a block of RMS
//! 23.80; that element is reconstructed below.
//!
//! These tests use the two reduction orders over K and a sampled N; the index
//! math at the real geometry is tested in `dense_gemm_m16_bf16_tests.rs`.
//!
//! Owner: model-layers ops.
//! Invariants: none beyond the types.

use crate::layers::dense_ffn::m16_tc::{
    M16_TC_ACC_FLOOR_MARGIN, m16_tc_acc_floor, within_m16_tc_budget,
};
use half::bf16;

/// 2026-09-25: The fixed floor the K-aware one is compared against: `2^-20` of
/// the block RMS, independent of K.
const ROUND6_FIXED_FLOOR: f64 = 9.536_743_164_062_5e-7;

/// 2026-09-25: The LM head's reduction depth.
const K_HEAD: usize = 5120;

/// 2026-09-25: The block RMS measured 2026-09-11 on the real head at M=16.
const HEAD_RMS: f64 = 23.8027;

/// 2026-09-25: `dense_gemv_bf16`'s reduction order, the reference: 64 lanes
/// each walking 8-wide chunks at a stride of 64, then a 32-lane shuffle
/// reduction per warp and one add across the two warps.
fn gemv_order(a: &[f32], b: &[f32]) -> f32 {
    let k = a.len();
    let mut lanes = [0.0_f32; 64];
    for (lane, acc) in lanes.iter_mut().enumerate() {
        let mut kv = lane;
        while kv < k / 8 {
            for i in 0..8 {
                *acc += a[kv * 8 + i] * b[kv * 8 + i];
            }
            kv += 64;
        }
    }
    let mut warps = [0.0_f32; 2];
    for (w, out) in warps.iter_mut().enumerate() {
        let mut v = [0.0_f32; 32];
        v.copy_from_slice(&lanes[w * 32..(w + 1) * 32]);
        let mut off = 16;
        while off > 0 {
            for l in 0..off {
                v[l] += v[l + off];
            }
            off >>= 1;
        }
        *out = v[0];
    }
    warps[0] + warps[1]
}

/// 2026-09-25: `dense_gemm_m16_bf16`'s reduction order: one FP32 accumulator
/// stepped `K/16` times, each step adding one `m16n8k16`'s 16 products, which
/// this sums as a balanced tree.
fn mma_order(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = 0.0_f32;
    for s in 0..a.len() / 16 {
        let mut p = [0.0_f32; 16];
        for (i, slot) in p.iter_mut().enumerate() {
            *slot = a[s * 16 + i] * b[s * 16 + i];
        }
        let mut w = 8;
        while w > 0 {
            for i in 0..w {
                p[i] += p[i + w];
            }
            w >>= 1;
        }
        acc += p[0];
    }
    acc
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        (self.0 >> 32) as u32
    }
    /// 2026-09-25: A BF16-representable value in [-1, 1], drawn as the GPU
    /// oracles draw them.
    fn bf16(&mut self) -> f32 {
        bf16::from_f32(((self.next() % 2049) as f32 - 1024.0) / 1024.0).to_f32()
    }
}

/// 2026-09-25: Weight columns per measurement of the order gap.
const DRAWS: usize = 4;

/// 2026-09-25: RMS of `mma_order - gemv_order` over [`DRAWS`] random weight
/// columns: the absolute FP32 accumulation noise between the two orders.
fn order_gap_rms(rng: &mut Rng, a: &[f32]) -> f64 {
    let mut sum = 0.0_f64;
    for _ in 0..DRAWS {
        let b: Vec<f32> = (0..a.len()).map(|_| rng.bf16()).collect();
        let g = f64::from(mma_order(a, &b)) - f64::from(gemv_order(a, &b));
        sum += g * g;
    }
    (sum / DRAWS as f64).sqrt()
}

/// 2026-09-25: The output row RMS an activation row produces against weights
/// drawn by [`Rng::bf16`]: `o_n = dot(a, b_n)` has `E[o^2] = ||a||^2 * var(b)`,
/// and `var(b)` is about 1/3, so the RMS is `||a|| / sqrt(3)`.
fn row_rms_of(row: &[f32]) -> f64 {
    let sum: f64 = row.iter().map(|v| f64::from(*v) * f64::from(*v)).sum();
    sum.sqrt() / 3.0_f64.sqrt()
}

/// 2026-09-25: The rejected H100 element, put through both floors. One hundred
/// ordinal BF16 ULP from `reference` is an absolute error of 4.8e-5 or 9.1e-5
/// depending on direction, and both directions are checked.
#[test]
fn the_round9_lm_head_outlier_is_a_cancelled_logit() {
    let reference = bf16::from_f32(-1.173_019_4e-4);
    let rb = reference.to_bits();
    assert_eq!(rb, 0xB8F6, "the receipt's reference must be BF16-exact");
    let relative = f64::from(reference.to_f32()).abs() / HEAD_RMS;
    assert!(
        (4.8e-6..5.0e-6).contains(&relative),
        "|ref|/rms is {relative:.3e}; the receipt printed 4.9e-6"
    );

    for direction in [-100_i32, 100] {
        // 2026-09-25: Ordinal ±100 from the reference, back to BF16 bits.
        let ord = -(i32::from(rb & 0x7FFF)) + direction;
        let ab = if ord < 0 {
            ((-ord) as u16) | 0x8000
        } else {
            ord as u16
        };
        let err = f64::from((bf16::from_bits(ab).to_f32() - reference.to_f32()).abs());
        assert!(
            (4.0e-5..1.0e-4).contains(&err),
            "100 ordinal ULP at this magnitude is {err:.3e}, not 4.8e-5..9.1e-5"
        );
        // 2026-09-25: The fixed floor is 2^-20 * HEAD_RMS = 2.27e-5, below both
        // errors.
        assert!(
            err > ROUND6_FIXED_FLOOR * HEAD_RMS,
            "round 6's fixed floor must be the thing that rejected this element"
        );
        // 2026-09-25: The K-aware floor is 8.1e-4, about 9x the larger error.
        assert!(
            within_m16_tc_budget(ab, rb, K_HEAD, HEAD_RMS),
            "a logit cancelled to 4.9e-6 of its row has no relative accuracy to grade"
        );
    }
}

/// 2026-09-25: The order gap grows with K while `2^-20 * row_rms` does not: at
/// K = 4096 the fixed floor is above the gap, at 65536 and 1048576 it is
/// below, and the K-aware floor stays over 50x the gap at every depth.
#[test]
fn the_fixed_floor_does_not_track_the_reduction_depth_and_the_new_one_does() {
    let mut rng = Rng(0x0927_167C_2026);
    let mut previous = 0.0_f64;
    for k in [4096_usize, 65536, 1_048_576] {
        let a: Vec<f32> = (0..k).map(|_| rng.bf16()).collect();
        let row_rms = row_rms_of(&a);
        let gap = order_gap_rms(&mut rng, &a);
        assert!(
            gap > previous,
            "k={k}: the order gap must grow with the reduction depth, \
             {gap:.3e} <= {previous:.3e}"
        );
        previous = gap;
        let floor = m16_tc_acc_floor(k, row_rms);
        assert!(
            floor > 50.0 * gap,
            "k={k}: the K-aware floor {floor:.3e} must keep a real margin over the \
             measured noise {gap:.3e}"
        );
        let fixed = ROUND6_FIXED_FLOOR * row_rms;
        if k <= 4096 {
            assert!(
                gap < fixed,
                "k={k}: {gap:.3e} vs the fixed floor {fixed:.3e}"
            );
        } else {
            assert!(
                gap > fixed,
                "k={k}: the fixed floor {fixed:.3e} was supposed to be under the \
                 noise {gap:.3e} here"
            );
        }
    }
}

/// 2026-09-25: With one quiet and one hot row, each row's floor covers its own
/// noise, while a block-wide RMS would give the quiet row over 8x its floor.
#[test]
fn the_floor_follows_the_row_and_not_the_block() {
    let mut rng = Rng(0x0927_1670_0009);
    let quiet: Vec<f32> = (0..K_HEAD).map(|_| rng.bf16() * 0.0625).collect();
    let hot: Vec<f32> = (0..K_HEAD).map(|_| rng.bf16()).collect();
    let (quiet_rms, hot_rms) = (row_rms_of(&quiet), row_rms_of(&hot));
    assert!(
        hot_rms / quiet_rms > 8.0,
        "the fixture must actually have a hot row"
    );
    let quiet_gap = order_gap_rms(&mut rng, &quiet);
    let hot_gap = order_gap_rms(&mut rng, &hot);

    // 2026-09-25: The noise itself scales with the row.
    assert!(
        hot_gap > 4.0 * quiet_gap,
        "the hot row's own accumulation noise {hot_gap:.3e} must dwarf the quiet \
         row's {quiet_gap:.3e}"
    );
    for (label, gap, rms) in [("quiet", quiet_gap, quiet_rms), ("hot", hot_gap, hot_rms)] {
        assert!(
            gap < m16_tc_acc_floor(K_HEAD, rms),
            "{label}: the row's own floor must cover the row's own noise"
        );
    }

    let block_rms = ((quiet_rms * quiet_rms + hot_rms * hot_rms) / 2.0).sqrt();
    assert!(
        m16_tc_acc_floor(K_HEAD, block_rms) > 8.0 * m16_tc_acc_floor(K_HEAD, quiet_rms),
        "a block RMS dominated by the hot row over-forgives the quiet one"
    );
}

#[test]
fn the_k_aware_floor_still_rejects_every_structural_error() {
    let floor = m16_tc_acc_floor(K_HEAD, HEAD_RMS);
    assert!(
        (8.0e-4..8.3e-4).contains(&floor),
        "the head's floor is 8.1e-4; got {floor:.4e}"
    );
    // 2026-09-25: About 1/29,000 of the block's RMS.
    assert!(HEAD_RMS / floor > 29_000.0, "the floor must stay tiny");
    for (label, r, a) in [
        ("a misplaced output row", 40.0_f32, -12.0_f32),
        ("three BF16 quanta at the top of the range", 96.0, 97.5),
        (
            "a cancelled logit moved by a matrix-scale error",
            1.0e-4,
            0.5,
        ),
        (
            "the microtest's three-ordinal mutation on |value| > 1",
            1.0,
            1.023_437_5,
        ),
    ] {
        let (rb, ab) = (bf16::from_f32(r).to_bits(), bf16::from_f32(a).to_bits());
        assert!(
            !within_m16_tc_budget(ab, rb, K_HEAD, HEAD_RMS),
            "{label}: ref={r} actual={a} must still fail the budget"
        );
    }
    // 2026-09-25: The smallest error above is over 28x the floor.
    assert!(
        (1.0 - 1.023_437_5_f64).abs() > 28.0 * floor,
        "the three-ordinal control must keep a real margin over the floor"
    );
    assert!(
        (M16_TC_ACC_FLOOR_MARGIN - 8.0).abs() < 1e-12,
        "the margin above is stated for M16_TC_ACC_FLOOR_MARGIN = 8"
    );
}
