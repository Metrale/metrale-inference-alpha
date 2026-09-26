// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `f32` to packed NVFP4 on the host, the inverse of [`super::nvfp4_dequant`].
//!
//! Owner: model-arch weight loader (GLM-5.3).
//!
//! `bind_expert` binds a routed-expert projection stored as packed U8 directly. One stored as
//! BF16 is quantised here, once, at load, into the same `Nvfp4Proj` triple, because
//! `Glm5NextExpertWeights` holds only `Nvfp4Proj`.
//!
//! ```text
//! weight_scale_2 = amax(tensor) / (6 * 448)            per tensor, f32
//! block_scale    = amax(block16) / 6 / weight_scale_2  per 16, encoded E4M3
//! code           = rne_e2m1( w / (e4m3(block_scale) * weight_scale_2) )
//! ```
//!
//! Invariants:
//! - `6` (the largest E2M1 magnitude) and `448` (the largest finite E4M3) are read from
//!   `NVFP4_E2M1_LUT` and `e4m3_lut`, the tables `nvfp4_dequant` decodes with.
//! - Codes are chosen against the decoded block scale, not the requested one.
//! - Rounding is to nearest, ties to the even code index; consecutive E2M1 and E4M3 codes
//!   differ in the mantissa's low bit, so this is round-to-nearest-even.
//! - Even flat index is the low nibble, as in the dequantiser.

use anyhow::{Result, bail};
use metrale_cache::kv_dequant::{NVFP4_E2M1_LUT, NVFP4_GROUP_SIZE, e4m3_lut};

/// 2026-09-25: `E4M3` codes `0x00..=0x7E`: every finite non-negative value,
/// ascending. `0x7F` is NaN and `0x80..` are the negatives, neither of which a
/// block scale may be.
const E4M3_FINITE_CODES: usize = 0x7F;

/// 2026-09-25: The largest `E2M1` magnitude (6.0), the top of the codebook's
/// non-negative half.
fn e2m1_max() -> f32 {
    NVFP4_E2M1_LUT[7]
}

/// 2026-09-25: The largest finite `E4M3` value (448.0).
fn e4m3_max() -> f32 {
    e4m3_lut()[E4M3_FINITE_CODES - 1]
}

/// 2026-09-25: One quantised projection, in the three pieces `Nvfp4Proj` is built from.
#[derive(Debug)]
pub(super) struct Nvfp4Blob {
    /// 2026-09-25: `[rows, cols / 2]` U8, two `e2m1` codes per byte.
    pub packed: Vec<u8>,
    /// 2026-09-25: `[rows, cols / 16]` `E4M3` block scales, one byte each.
    pub scales: Vec<u8>,
    /// 2026-09-25: The per-tensor global scale.
    pub scale_2: f32,
}

/// 2026-09-25: Quantise a row-major `f32 [rows, cols]` weight to NVFP4.
///
/// Errors on a length that is not `rows * cols`, on `cols` that is zero or not a
/// multiple of 16, and on any non-finite value.
pub(super) fn quantize_to_nvfp4(
    what: &str,
    values: &[f32],
    rows: usize,
    cols: usize,
) -> Result<Nvfp4Blob> {
    if values.len() != rows * cols {
        bail!(
            "{what}: {} elements, expected [{rows}, {cols}] = {}",
            values.len(),
            rows * cols
        );
    }
    // 2026-09-25: `cols == 0` passes `is_multiple_of` and would make the band
    // width zero, which `chunks` panics on.
    if cols == 0 || !cols.is_multiple_of(NVFP4_GROUP_SIZE) {
        bail!(
            "{what}: {cols} columns is not a whole number of {NVFP4_GROUP_SIZE}-element \
             NVFP4 blocks"
        );
    }
    // 2026-09-25: `f32::max` ignores NaN (it returns the other operand), so
    // testing the folded amax would miss every NaN; the scan rejects per element.
    let mut amax = 0.0f32;
    for &v in values {
        if !v.is_finite() {
            bail!("{what}: weight contains a non-finite value; refusing to quantise it");
        }
        amax = amax.max(v.abs());
    }
    // 2026-09-25: An all-zero tensor has no amax to scale by; with `1.0` every
    // block scale and every code is zero.
    let scale_2 = if amax > 0.0 {
        amax / (e2m1_max() * e4m3_max())
    } else {
        1.0
    };

    let groups_per_row = cols / NVFP4_GROUP_SIZE;
    let mut packed = vec![0u8; rows * cols / 2];
    let mut scales = vec![0u8; rows * groups_per_row];

    // 2026-09-25: Rows are independent once `scale_2` is known (a block never
    // reaches past its own 16 columns), so the sweep runs one scoped thread per
    // row band, over disjoint `chunks_mut` slices.
    let bands = std::thread::available_parallelism().map_or(1, |n| n.get());
    let band_rows = rows.div_ceil(bands).max(1);
    std::thread::scope(|s| {
        for ((v, p), sc) in values
            .chunks(band_rows * cols)
            .zip(packed.chunks_mut(band_rows * cols / 2))
            .zip(scales.chunks_mut(band_rows * groups_per_row))
        {
            s.spawn(move || quantize_rows(v, p, sc, cols, scale_2));
        }
    });
    Ok(Nvfp4Blob {
        packed,
        scales,
        scale_2,
    })
}

/// 2026-09-25: One band of whole rows. `values`, `packed` and `scales` are the
/// band's slices of the three buffers, so every index here is band-local.
/// `packed` must be zeroed: codes are OR-ed in.
fn quantize_rows(values: &[f32], packed: &mut [u8], scales: &mut [u8], cols: usize, scale_2: f32) {
    let groups_per_row = cols / NVFP4_GROUP_SIZE;
    let e4m3 = e4m3_lut();
    for (r, row) in values.chunks(cols).enumerate() {
        for g in 0..groups_per_row {
            let base = g * NVFP4_GROUP_SIZE;
            let block = &row[base..base + NVFP4_GROUP_SIZE];
            let bmax = block.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let sb = encode_e4m3_rne(bmax / e2m1_max() / scale_2);
            scales[r * groups_per_row + g] = sb;
            // 2026-09-25: The decoded scale, not the requested one: the codes
            // are chosen against what the stored byte means.
            let eff = e4m3[sb as usize] * scale_2;
            let inv = if eff > 0.0 { 1.0 / eff } else { 0.0 };
            for (i, &v) in block.iter().enumerate() {
                let code = encode_e2m1_rne(v * inv);
                let flat = r * cols + base + i;
                if flat.is_multiple_of(2) {
                    packed[flat / 2] |= code;
                } else {
                    packed[flat / 2] |= code << 4;
                }
            }
        }
    }
}

/// 2026-09-25: Nearest `E2M1` code to `v`, ties to even, sign preserved.
///
/// The sign survives a magnitude that rounds to zero, so `-0.0` is code `0x8`,
/// which `NVFP4_E2M1_LUT` decodes to `-0.0`.
fn encode_e2m1_rne(v: f32) -> u8 {
    let sign = if v.is_sign_negative() { 0x8u8 } else { 0 };
    let mag = &NVFP4_E2M1_LUT[..8];
    sign | nearest_even_code(mag, v.abs()) as u8
}

/// 2026-09-25: Nearest `E4M3` code to a non-negative `v`, ties to even.
fn encode_e4m3_rne(v: f32) -> u8 {
    nearest_even_code(&e4m3_lut()[..E4M3_FINITE_CODES], v) as u8
}

/// 2026-09-25: Index of the entry of an ascending ladder nearest to `v`, ties
/// to the even index. `v` at or past the top saturates; `v` at or below zero is index 0.
///
/// Ties go to the even index because consecutive codes of both `E2M1` and
/// `E4M3` differ in the mantissa's low bit — so "even index" and "even
/// mantissa" are the same rule, and this is round-to-nearest-even as the
/// format defines it.
fn nearest_even_code(ladder: &[f32], v: f32) -> usize {
    let top = ladder.len() - 1;
    // 2026-09-25: NaN never reaches here (`quantize_to_nvfp4` rejects
    // non-finite input); the explicit test keeps a NaN from being ordered by
    // `partition_point`.
    if v.is_nan() || v <= ladder[0] {
        return 0;
    }
    if v >= ladder[top] {
        return top;
    }
    let hi = ladder.partition_point(|&x| x < v);
    let lo = hi - 1;
    let below = v - ladder[lo];
    let above = ladder[hi] - v;
    if below < above || (below == above && lo.is_multiple_of(2)) {
        lo
    } else {
        hi
    }
}

#[cfg(test)]
#[path = "nvfp4_quant_tests.rs"]
mod tests;
