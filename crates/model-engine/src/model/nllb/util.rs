// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Host-side helpers for the served NLLB runtime: sinusoidal position tables
//! and byte views for H2D copies.
//!
//! Owner: model-engine.
//! Invariants: none beyond the types.

use half::bf16;

/// 2026-09-25: One sinusoidal position row into `out[d]` (bf16): `sin` in the lower
/// half, `cos` in the upper half, `freq_j = exp(-j·ln(10000)/(d/2-1))`.
pub(super) fn sinusoid_row(pos: f32, d: usize, out: &mut [bf16]) {
    let half = d / 2;
    let emb_scale = 10000f32.ln() / (half as f32 - 1.0);
    for j in 0..half {
        let ang = pos * (-(j as f32) * emb_scale).exp();
        out[j] = bf16::from_f32(ang.sin());
        out[half + j] = bf16::from_f32(ang.cos());
    }
}

/// 2026-09-25: Decoder position table `[max_len, d]` (bf16): row `i` holds
/// sinusoid `i + 2` (offset `padding_idx + 1`, with `padding_idx = 1`).
pub(super) fn decoder_pos_table_bf16(max_len: usize, d: usize) -> Vec<bf16> {
    let mut t = vec![bf16::from_f32(0.0); max_len * d];
    for i in 0..max_len {
        sinusoid_row((i + 2) as f32, d, &mut t[i * d..i * d + d]);
    }
    t
}

/// 2026-09-25: Encoder position embeddings `[seq, d]` (bf16) with masked
/// incremental positions: non-pad tokens count from `pad + 1`; pad tokens get a
/// zero row.
pub(super) fn encoder_pos_bf16(ids: &[u32], d: usize, pad: u32) -> Vec<bf16> {
    let seq = ids.len();
    let mut t = vec![bf16::from_f32(0.0); seq * d];
    let mut running = 0u32;
    for (i, &id) in ids.iter().enumerate() {
        let p = if id != pad {
            running += 1;
            running + pad
        } else {
            pad
        };
        if p != pad {
            sinusoid_row(p as f32, d, &mut t[i * d..i * d + d]);
        }
    }
    t
}

/// 2026-09-25: View a `&[u32]` as its native-endian bytes for an H2D copy.
pub(super) fn u32_bytes(v: &[u32]) -> &[u8] {
    // 2026-09-25: SAFETY: `u32` has no padding and every byte pattern is a valid
    // `u8`; the view covers exactly `size_of_val(v)` bytes of the same
    // allocation and borrows `v` for its lifetime.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

/// 2026-09-25: View a `&[bf16]` as its native-endian bytes for an H2D copy.
pub(super) fn bf16_bytes(v: &[bf16]) -> &[u8] {
    // 2026-09-25: SAFETY: `bf16` is a `#[repr(transparent)]` wrapper over `u16`;
    // same reasoning as `u32_bytes`.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

#[cfg(test)]
#[path = "util_tests.rs"]
mod tests;
