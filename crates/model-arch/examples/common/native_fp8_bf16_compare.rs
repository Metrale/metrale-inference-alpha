// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The cuBLASLt-against-kernel comparator of
//! `native_fp8_ffn_w8a8_microtest` and its CPU tests.
//!
//! Owner: model-arch examples.
//! Invariants: none beyond the types.

use half::bf16;

/// 2026-09-25: cuBLASLt-against-kernel acceptance. Both read the same FP8 bytes
/// and FP32 scales, so the comparison allows what FP32 accumulation order can
/// change: every element must pass [`within_one_bf16_ulp`], at most one BF16
/// ULP apart at the larger magnitude or both below `CUBLAS_SMALL_MAGNITUDE`,
/// where a value is cancellation residue whose sign follows the accumulation
/// order. The example also requires cosine >= `CUBLAS_COSINE_GATE` and
/// `rel_rms <= CUBLAS_REL_RMS_GATE`; `sign_flips`, `unequal` and `max_ulp` are
/// printed only.
///
/// 0.05 is above the largest sign-flipped magnitude measured on H100 on
/// 2026-09-11 (0.0297; the four values are in `near_zero_sign_flip_passes`).
pub(crate) const CUBLAS_SMALL_MAGNITUDE: f64 = 0.05;
pub(crate) const CUBLAS_COSINE_GATE: f64 = 0.99999;
pub(crate) const CUBLAS_REL_RMS_GATE: f64 = 2e-3;

fn to_f64(bits: &[u16]) -> Vec<f64> {
    bits.iter()
        .map(|b| bf16::from_bits(*b).to_f32() as f64)
        .collect()
}

/// 2026-09-25: BF16 bits to a monotonically ordered integer, so
/// `|ord(a) - ord(b)|` is the ordinal ULP distance; +0 and -0 both map to 0.
fn ord(bits: u16) -> i32 {
    if bits & 0x8000 != 0 {
        -((bits & 0x7fff) as i32)
    } else {
        bits as i32
    }
}

/// 2026-09-25: One BF16 ULP at `bits`' magnitude. BF16 stores 7 mantissa bits,
/// so a normal value with unbiased exponent `e` steps by `2^(e - 7)`; every
/// subnormal steps by `2^-133`, the smallest normal's step. Inf and NaN report
/// an infinite step.
fn bf16_ulp(bits: u16) -> f64 {
    let biased_exp = ((bits >> 7) & 0xff) as i32;
    match biased_exp {
        0xff => f64::INFINITY,
        0 => (-133.0_f64).exp2(),
        e => ((e - 127 - 7) as f64).exp2(),
    }
}

/// 2026-09-25: The per-element acceptance described on
/// [`CUBLAS_SMALL_MAGNITUDE`]: one BF16 step at the larger magnitude, or both
/// values under the small-value escape. A NaN never passes.
fn within_one_bf16_ulp(a_bits: u16, b_bits: u16) -> bool {
    let a = bf16::from_bits(a_bits).to_f32() as f64;
    let b = bf16::from_bits(b_bits).to_f32() as f64;
    if a.abs() < CUBLAS_SMALL_MAGNITUDE && b.abs() < CUBLAS_SMALL_MAGNITUDE {
        return true;
    }
    let ulp = if a.abs() >= b.abs() {
        bf16_ulp(a_bits)
    } else {
        bf16_ulp(b_bits)
    };
    (a - b).abs() <= ulp
}

/// 2026-09-25: Opposite signs with neither value zero; counted, not gated.
fn is_sign_flip(a_bits: u16, b_bits: u16) -> bool {
    let a = bf16::from_bits(a_bits).to_f32();
    let b = bf16::from_bits(b_bits).to_f32();
    a != 0.0 && b != 0.0 && (a < 0.0) != (b < 0.0)
}

pub(crate) struct Compare {
    pub(crate) max_abs: f64,
    pub(crate) cosine: f64,
    pub(crate) rel_rms: f64,
    pub(crate) max_ulp: i32,
    pub(crate) unequal: usize,
    /// 2026-09-25: Elements failing [`within_one_bf16_ulp`].
    pub(crate) over_bound: usize,
    pub(crate) sign_flips: usize,
}

pub(crate) fn compare(a_bits: &[u16], b_bits: &[u16]) -> Compare {
    let (a, b) = (to_f64(a_bits), to_f64(b_bits));
    let (mut dot, mut na, mut nb, mut max_abs, mut sq_diff, mut sq_ref) =
        (0.0, 0.0, 0.0, 0.0_f64, 0.0, 0.0);
    for (x, y) in a.iter().zip(&b) {
        dot += x * y;
        na += x * x;
        nb += y * y;
        max_abs = max_abs.max((x - y).abs());
        sq_diff += (x - y) * (x - y);
        sq_ref += y * y;
    }
    Compare {
        max_abs,
        cosine: dot / (na.sqrt() * nb.sqrt()),
        rel_rms: (sq_diff / sq_ref.max(f64::MIN_POSITIVE)).sqrt(),
        max_ulp: a_bits
            .iter()
            .zip(b_bits)
            .map(|(x, y)| (ord(*x) - ord(*y)).abs())
            .max()
            .unwrap_or(0),
        unequal: a_bits.iter().zip(b_bits).filter(|(x, y)| x != y).count(),
        over_bound: a_bits
            .iter()
            .zip(b_bits)
            .filter(|(x, y)| !within_one_bf16_ulp(**x, **y))
            .count(),
        sign_flips: a_bits
            .iter()
            .zip(b_bits)
            .filter(|(x, y)| is_sign_flip(**x, **y))
            .count(),
    }
}

/// 2026-09-25: CPU tests for the acceptance rule, on synthetic vectors. The
/// example's `[[example]]` stanza sets `test = true`, so
/// `cargo test -p metrale-model-arch --example native_fp8_ffn_w8a8_microtest
/// --features cuda,gpu-examples` runs them.
#[cfg(test)]
mod tests {
    use super::*;

    /// 2026-09-25: The BF16 value one step above positive `x`: the stored bits
    /// plus one.
    fn next_bf16_up(x: f32) -> u16 {
        let b = bf16::from_f32(x).to_bits();
        assert!(b & 0x8000 == 0, "helper is for positive values");
        b + 1
    }

    #[test]
    fn ulp_is_the_gap_to_the_next_bf16() {
        // 2026-09-25: 1.0 is at the bottom of its binade, so its ULP is 2^-7.
        let one = bf16::from_f32(1.0).to_bits();
        assert_eq!(bf16_ulp(one), 2.0_f64.powi(-7));
        let up = bf16::from_bits(next_bf16_up(1.0)).to_f32() as f64;
        assert!((up - 1.0 - bf16_ulp(one)).abs() < 1e-12);
        // 2026-09-25: One ULP is 1.0 in [128, 256) and 2.0 in [256, 512).
        assert_eq!(bf16_ulp(bf16::from_f32(200.0).to_bits()), 1.0);
        assert_eq!(bf16_ulp(bf16::from_f32(400.0).to_bits()), 2.0);
    }

    #[test]
    fn one_ulp_apart_passes() {
        for v in [1.0_f32, 3.5, 128.0, 200.0, 17408.0, 0.0625] {
            let a = bf16::from_f32(v).to_bits();
            let b = next_bf16_up(v);
            assert!(
                within_one_bf16_ulp(a, b),
                "one step at {v} should be accepted"
            );
            assert!(within_one_bf16_ulp(b, a), "the bound is symmetric at {v}");
        }
    }

    #[test]
    fn two_ulps_apart_fails() {
        for v in [1.0_f32, 3.5, 128.0, 200.0, 17408.0, 0.0625] {
            let a = bf16::from_f32(v).to_bits();
            let b = next_bf16_up(v) + 1;
            assert!(
                !within_one_bf16_ulp(a, b),
                "two steps at {v} must be rejected — this is the resolution the \
                 gate exists to have"
            );
        }
    }

    #[test]
    fn near_zero_sign_flip_passes() {
        // 2026-09-25: The four sign-flipped magnitudes measured on H100 on
        // 2026-09-11, and 0.049, just under the escape.
        for v in [0.0148_f32, 0.0167, 0.0237, 0.0297, 0.049] {
            let a = bf16::from_f32(v).to_bits();
            let b = bf16::from_f32(-v).to_bits();
            assert!(
                within_one_bf16_ulp(a, b),
                "a sign flip at |v|={v} is cancellation residue, not a layout bug"
            );
            assert!(is_sign_flip(a, b), "and it is still COUNTED as a flip");
        }
    }

    #[test]
    fn sign_flip_above_the_escape_fails() {
        // 2026-09-25: A flip at 0.06 is 0.12 apart against a one-ULP bound of
        // 2^-12, and a flip at 200.0 is 400 apart against 1.0.
        for v in [0.06_f32, 1.0, 200.0] {
            let a = bf16::from_f32(v).to_bits();
            let b = bf16::from_f32(-v).to_bits();
            assert!(
                !within_one_bf16_ulp(a, b),
                "a sign flip at |v|={v} is outside the small-value escape"
            );
        }
    }

    #[test]
    fn equal_values_and_zero_pass_and_nan_does_not() {
        let x = bf16::from_f32(7.25).to_bits();
        assert!(within_one_bf16_ulp(x, x));
        assert!(!is_sign_flip(x, x));
        let zero = bf16::from_f32(0.0).to_bits();
        let neg_zero = bf16::from_f32(-0.0).to_bits();
        assert!(within_one_bf16_ulp(zero, neg_zero));
        assert!(!is_sign_flip(zero, neg_zero), "+-0 is not a sign flip");
        let nan = bf16::from_f32(f32::NAN).to_bits();
        assert!(!within_one_bf16_ulp(nan, x), "NaN must never be accepted");
        assert!(!within_one_bf16_ulp(nan, nan));
    }

    /// 2026-09-25: A pair straddling a binade boundary is judged at the larger
    /// magnitude's (coarser) grid, so the BF16 neighbours 127.5 and 128.0 pass,
    /// and so does 127.0 against 128.0, two steps of the finer grid.
    #[test]
    fn straddling_a_binade_uses_the_larger_magnitude() {
        let hi = bf16::from_f32(128.0).to_bits();
        let lo = hi - 1;
        assert_eq!(bf16::from_bits(lo).to_f32(), 127.5);
        assert!(within_one_bf16_ulp(lo, hi));
        assert!(within_one_bf16_ulp(hi, lo));
        assert!(
            within_one_bf16_ulp(lo - 1, hi),
            "127.0 vs 128.0 is one step of the coarser grid — accepted by design"
        );
        assert!(
            !within_one_bf16_ulp(lo - 2, hi),
            "126.5 vs 128.0 is 1.5 ULP even at the coarser grid"
        );
    }

    /// 2026-09-25: The whole-vector `compare` must agree with the per-element
    /// rule and count the flips.
    #[test]
    fn compare_counts_over_bound_and_sign_flips() {
        let a: Vec<u16> = vec![
            bf16::from_f32(1.0).to_bits(),
            bf16::from_f32(3.5).to_bits(),
            bf16::from_f32(0.0148).to_bits(),
            bf16::from_f32(64.0).to_bits(),
        ];
        let b: Vec<u16> = vec![
            bf16::from_f32(1.0).to_bits(),
            next_bf16_up(3.5),
            bf16::from_f32(-0.0148).to_bits(),
            next_bf16_up(64.0) + 1,
        ];
        let c = compare(&a, &b);
        assert_eq!(c.over_bound, 1, "only the two-ULP element is out of bound");
        assert_eq!(c.sign_flips, 1);
        assert_eq!(c.unequal, 3);
    }
}
