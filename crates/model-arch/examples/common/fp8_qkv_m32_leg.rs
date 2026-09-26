// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The 17..=32-row leg of `native_fp8_qkv_batch_microtest`:
//! `w8a16_gemm_pipelined_m32_strided` against the per-row scalar loop.
//!
//! The parent's GEMV legs must be byte-exact. This tensor-core M tile reduces
//! K in another order, so its live extents are graded by
//! `compare_m16_tc_block` (`layers::dense_ffn::m16_tc::oracle`): within
//! `M16_TC_MAX_ULP` ordinal BF16 ULP of the scalar result or under the
//! K-scaled accumulation floor, plus the `REL_RMS_GATE` block gate. Every byte
//! outside the live extents, guard bands included, must still be the sentinel
//! on both runs (the parent's `check_untouched`).
//!
//! Owner: model-arch examples.
//! Invariants: none beyond the types.

use anyhow::{Result, ensure};
use metrale_model_layers::layers::dense_ffn::m16_tc::oracle::{
    M16_TC_MAX_ULP, compare_m16_tc_block,
};

use super::{GUARD, check_untouched, values};

/// 2026-09-25: Block relative-RMS gate, equal to `REL_RMS_GATE` in
/// `examples/common/m16_tc_compare.rs`.
const REL_RMS_GATE: f64 = 1e-3;

/// 2026-09-25: Worst-case numbers over the live extents of one case.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct M32Verdict {
    pub(crate) max_ulp: i32,
    pub(crate) over_ulp_only: usize,
    pub(crate) rel_rms: f64,
    pub(crate) max_abs: f64,
}

/// 2026-09-25: Grade one case: every byte outside `live` must equal the
/// sentinel on both runs, and every live extent, reduced over `k`, must be
/// finite and pass `compare_m16_tc_block` and `REL_RMS_GATE` against the scalar
/// baseline.
pub(crate) fn check_m32(
    observed: &[u8],
    baseline: &[u8],
    sentinel: &[u8],
    live: &[(usize, usize)],
    k: usize,
) -> Result<M32Verdict> {
    check_untouched(observed, baseline, sentinel, live)?;
    let mut worst = M32Verdict::default();
    for &(start, len) in live {
        let a = &observed[GUARD + start..GUARD + start + len];
        let b = &baseline[GUARD + start..GUARD + start + len];
        ensure!(
            values(a)
                .iter()
                .chain(values(b).iter())
                .all(|x| x.is_finite()),
            "nonfinite projection output"
        );
        let d = compare_m16_tc_block(a, b, len / 2, k);
        if let Some(o) = d.over_budget.first() {
            anyhow::bail!(
                "M32 tile output exceeds the {M16_TC_MAX_ULP}-ULP / accumulation-floor \
                 budget: {} element(s), first at col {} (scalar {:e}, tile {:e}, {} ULP, \
                 row_rms {:.4})",
                d.over_budget.len(),
                o.col,
                o.reference,
                o.actual,
                o.ulp,
                o.row_rms
            );
        }
        ensure!(
            d.rel_rms <= REL_RMS_GATE,
            "M32 tile rel_rms {:.3e} exceeds the {REL_RMS_GATE:.0e} block gate",
            d.rel_rms
        );
        worst.max_ulp = worst.max_ulp.max(d.max_ulp);
        worst.over_ulp_only += d.over_ulp_only;
        worst.rel_rms = worst.rel_rms.max(d.rel_rms);
        worst.max_abs = worst.max_abs.max(d.max_abs);
    }
    Ok(worst)
}

/// 2026-09-25: Known-bad controls, each of which [`check_m32`] must refuse: a
/// flip of bit 14 (the exponent's top bit, so the biased exponent moves by 128)
/// on the largest live element of the first extent, a flipped bit at
/// `gap_byte`, one in the leading guard band, and a NaN in the first live
/// element.
pub(crate) fn known_bad_controls(
    baseline: &[u8],
    sentinel: &[u8],
    live: &[(usize, usize)],
    gap_byte: usize,
    k: usize,
) -> Result<()> {
    let (start, len) = live[0];
    let row = &baseline[GUARD + start..GUARD + start + len];
    let (idx, _) = values(row)
        .iter()
        .enumerate()
        .fold((0usize, 0.0f64), |acc, (i, v)| {
            if v.abs() > acc.1 { (i, v.abs()) } else { acc }
        });
    for mutation in ["exponent-bit", "gap", "guard", "nonfinite"] {
        let mut bad = baseline.to_vec();
        match mutation {
            "exponent-bit" => bad[GUARD + start + idx * 2 + 1] ^= 0x40,
            "gap" => bad[GUARD + gap_byte] ^= 1,
            "guard" => bad[0] ^= 1,
            _ => bad[GUARD + start..GUARD + start + 2].copy_from_slice(&0x7fc0_u16.to_le_bytes()),
        }
        let err = check_m32(&bad, baseline, sentinel, live, k)
            .err()
            .ok_or_else(|| {
                anyhow::anyhow!("known-bad `{mutation}` was admitted by the M32 oracle")
            })?;
        println!("KNOWN_BAD m32 {mutation}: refused: {err}");
    }
    Ok(())
}
