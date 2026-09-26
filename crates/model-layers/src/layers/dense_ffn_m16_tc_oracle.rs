// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The numerics contract for the tensor-core decode tiers, graded against the scalar GEMV.
//!
//! Owner: model-layers (dense FFN).
//! Invariants: none beyond the types.
//!
//! [`within_m16_tc_budget`] is the one predicate the GPU microtests (for example
//! `examples/native_fp8_ffn_m16_tc_microtest.rs` and
//! `examples/native_bf16_lm_head_m16_microtest.rs`) and the host tests
//! (`dense_ffn_m16_tc_m32_tests.rs`, `ops/dense_gemm_m16_bf16_tests.rs`,
//! `ops/dense_gemm_m16_bf16_floor_tests.rs`) import. An m16n8k16 MMA sums K in
//! a different order from the scalar GEMV, so the contract is a tolerance: an
//! ordinal-ULP budget plus an absolute floor scaled to the reduction
//! ([`m16_tc_acc_floor`]).

/// 2026-09-25: Ordinal BF16 ULP budget against the scalar GEMV reference.
pub const M16_TC_MAX_ULP: i32 = 2;

/// 2026-09-25: FP32 unit roundoff, `2^-24`.
pub const F32_UNIT_ROUNDOFF: f64 = 5.960_464_477_539_063e-8;

/// 2026-09-25: Margin over the `u * sqrt(k) * row_rms` accumulation scale in
/// [`m16_tc_acc_floor`].
///
/// Calibrated, not derived: the MMA's internal summation order is unspecified
/// hardware. Tests pin both ends of the calibration.
/// `the_round9_lm_head_outlier_is_a_cancelled_logit`
/// (`ops/dense_gemm_m16_bf16_floor_tests.rs`) admits an H100 LM-head element
/// 100 ordinal ULP from a reference of -1.173e-4 at an RMS of 23.80, where the
/// floor is 8.1e-4. `the_k_aware_floor_still_rejects_every_structural_error`
/// (same file) and `the_accumulation_floor_still_rejects_a_structural_error`
/// (`dense_ffn_m16_tc_m32_tests.rs`) still reject a misplaced output and a
/// three-ordinal error on a value near 1.
pub const M16_TC_ACC_FLOOR_MARGIN: f64 = 8.0;

/// 2026-09-25: Absolute error below which an ordinal-ULP budget says nothing,
/// for one output of a length-`k` FP32 reduction whose row has RMS `row_rms`.
///
/// An ordinal BF16 ULP is a relative unit, and an output that cancelled to
/// near zero has no relative accuracy left to grade. The floor is
/// `M16_TC_ACC_FLOOR_MARGIN * u * sqrt(k) * row_rms`: two FP32 summation orders
/// over `k` terms drift apart like a random walk of `k` roundings, each about
/// `u * row_rms`. `the_fixed_floor_does_not_track_the_reduction_depth_and_the_new_one_does`
/// (`ops/dense_gemm_m16_bf16_floor_tests.rs`) measures that drift on the host
/// and checks the floor stays more than 50x above it.
///
/// `row_rms` is the RMS of the reference row, not of the whole block: an
/// element's accumulation noise scales with the norm of the activation row
/// that produced it, and `the_floor_follows_the_row_and_not_the_block` pins
/// that a block-wide RMS over-forgives a quiet row.
pub fn m16_tc_acc_floor(k: usize, row_rms: f64) -> f64 {
    M16_TC_ACC_FLOOR_MARGIN * F32_UNIT_ROUNDOFF * (k as f64).sqrt() * row_rms
}

/// 2026-09-25: BF16 bits -> a monotone integer, so `|ord(a) - ord(b)|` is the
/// ULP distance and +0/-0 are the same point.
pub fn bf16_ord(bits: u16) -> i32 {
    if bits & 0x8000 != 0 {
        -((bits & 0x7FFF) as i32)
    } else {
        bits as i32
    }
}

/// 2026-09-25: The tier's numerics contract as one predicate.
///
/// An element passes if it is within [`M16_TC_MAX_ULP`] ordinal BF16 ULP of the
/// reference, or if its absolute error is under [`m16_tc_acc_floor`] for the
/// reduction depth `k` and the reference row's RMS.
///
/// `k` is the reduction depth the kernel ran, not the output width.
pub fn within_m16_tc_budget(actual_bits: u16, reference_bits: u16, k: usize, row_rms: f64) -> bool {
    if (bf16_ord(actual_bits) - bf16_ord(reference_bits)).abs() <= M16_TC_MAX_ULP {
        return true;
    }
    let a = f64::from(half::bf16::from_bits(actual_bits).to_f32());
    let b = f64::from(half::bf16::from_bits(reference_bits).to_f32());
    (a - b).abs() <= m16_tc_acc_floor(k, row_rms)
}

/// 2026-09-25: One element the comparison rejected: where it is, the two
/// values, the ordinal distance and the row RMS the floor was evaluated at.
#[derive(Debug, Clone, Copy)]
pub struct M16TcOutlier {
    pub row: usize,
    pub col: usize,
    pub reference: f32,
    pub actual: f32,
    pub ulp: i32,
    /// 2026-09-25: RMS of the reference row this element sits in, the scale
    /// [`m16_tc_acc_floor`] was evaluated at.
    pub row_rms: f64,
}

/// 2026-09-25: The result of comparing an `m x n` BF16 block against the
/// scalar reference.
#[derive(Debug, Clone, Default)]
pub struct M16TcDiff {
    /// 2026-09-25: Largest ordinal BF16 ULP distance; sign flips are counted in
    /// `sign_flips` and excluded here.
    pub max_ulp: i32,
    /// 2026-09-25: Elements the full criterion rejected (ordinal budget and
    /// accumulation floor both exceeded).
    pub over_budget: Vec<M16TcOutlier>,
    /// 2026-09-25: Elements over the ordinal budget alone, whether or not the
    /// floor admitted them.
    pub over_ulp_only: usize,
    pub sign_flips: usize,
    pub max_abs: f64,
    pub rel_rms: f64,
    /// 2026-09-25: RMS of the whole reference block. The floor uses the per-row
    /// values in `row_rms`, and this one only for a row that has none.
    pub rms: f64,
    /// 2026-09-25: Per-row reference RMS, in row order: the scales the floor was
    /// evaluated at.
    pub row_rms: Vec<f64>,
}

/// 2026-09-25: A sign change on a reference smaller than this in magnitude is
/// counted in `sign_flips` and not graded.
pub const M16_TC_SIGN_FLIP_BAND: f64 = 0.05;

/// 2026-09-25: RMS of a little-endian BF16 slice, in f64; 0 for an empty slice.
fn bf16_rms(block: &[u8]) -> f64 {
    let count = block.len() / 2;
    if count == 0 {
        return 0.0;
    }
    let sum: f64 = block
        .chunks_exact(2)
        .map(|b| {
            let v = f64::from(half::bf16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32());
            v * v
        })
        .sum();
    (sum / count as f64).sqrt()
}

/// 2026-09-25: Compare an `m x n` BF16 block against the scalar reference
/// under the tier's contract. Both slices are `m * n` little-endian BF16
/// elements; `k` is the reduction depth the kernel ran.
///
/// Two passes: the per-row reference RMS is computed first, then each element
/// is graded against its row's floor.
pub fn compare_m16_tc_block(actual: &[u8], reference: &[u8], n: usize, k: usize) -> M16TcDiff {
    let bits = |b: &[u8]| u16::from_le_bytes([b[0], b[1]]);
    let val = |b: u16| f64::from(half::bf16::from_bits(b).to_f32());
    let row_bytes = n * 2;
    let row_rms: Vec<f64> = if row_bytes == 0 {
        Vec::new()
    } else {
        reference.chunks(row_bytes).map(bf16_rms).collect()
    };
    let mut d = M16TcDiff {
        rms: bf16_rms(reference),
        row_rms,
        ..Default::default()
    };
    let (mut err_sq, mut ref_sq) = (0.0_f64, 0.0_f64);
    for (i, (a, b)) in actual
        .chunks_exact(2)
        .zip(reference.chunks_exact(2))
        .enumerate()
    {
        let (ab, bb) = (bits(a), bits(b));
        let (av, bv) = (val(ab), val(bb));
        err_sq += (av - bv) * (av - bv);
        ref_sq += bv * bv;
        d.max_abs = d.max_abs.max((av - bv).abs());
        let ulp = (bf16_ord(ab) - bf16_ord(bb)).abs();
        if av.signum() != bv.signum() && bv.abs() < M16_TC_SIGN_FLIP_BAND {
            d.sign_flips += 1;
            continue;
        }
        d.max_ulp = d.max_ulp.max(ulp);
        if ulp > M16_TC_MAX_ULP {
            d.over_ulp_only += 1;
        }
        let row = if n == 0 { 0 } else { i / n };
        let scale = d.row_rms.get(row).copied().unwrap_or(d.rms);
        if !within_m16_tc_budget(ab, bb, k, scale) {
            d.over_budget.push(M16TcOutlier {
                row,
                col: if n == 0 { 0 } else { i % n },
                reference: bv as f32,
                actual: av as f32,
                ulp,
                row_rms: scale,
            });
        }
    }
    d.rel_rms = if ref_sq > 0.0 {
        (err_sq / ref_sq).sqrt()
    } else {
        err_sq.sqrt()
    };
    d
}
