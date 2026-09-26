// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The dequantization codebooks and the scale group size.
//!
//! Owner: cache.
//! Invariants: each codebook holds the same values as the `lut_init` array of
//! the decode kernel named on it (the kernel keeps the turbo codebooks as
//! `__half`).

/// 2026-09-25: Elements per scale group in every grouped layout; the
/// `NVFP4_GROUP_SIZE` of `kernels/gb10/common/paged_decode_attn_nvfp4.cu`.
pub const NVFP4_GROUP_SIZE: usize = 16;

/// 2026-09-25: E2M1 4-bit codebook (NVFP4), as in
/// `kernels/gb10/common/paged_decode_attn_nvfp4.cu`.
pub const NVFP4_E2M1_LUT: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// 2026-09-25: Turbo4 16-level Lloyd-Max codebook, as in
/// `kernels/gb10/common/paged_decode_attn_turbo4.cu`.
pub const TURBO4_LUT: [f32; 16] = [
    -2.7326, -2.0690, -1.6180, -1.2562, -0.9423, -0.6568, -0.3880, -0.1284, 0.1284, 0.3880, 0.6568,
    0.9423, 1.2562, 1.6180, 2.0690, 2.7326,
];

/// 2026-09-25: Turbo3 8-level Lloyd-Max codebook, as in
/// `kernels/gb10/common/paged_decode_attn_turbo3.cu`.
pub const TURBO3_LUT: [f32; 8] = [
    -2.1520, -1.3440, -0.7560, -0.2451, 0.2451, 0.7560, 1.3440, 2.1520,
];

/// 2026-09-25: FP8 E4M3 byte to f32, all 256 bytes, computed at compile
/// time. `0x7F` and `0xFF` are NaN.
const E4M3_LUT: [f32; 256] = {
    let mut lut = [0.0f32; 256];
    let mut byte = 0u32;
    while byte < 256 {
        let sign_bit = (byte >> 7) & 1;
        let exp = ((byte >> 3) & 0xF) as i32;
        let mant = byte & 0x7;
        let s: f32 = if sign_bit == 0 { 1.0 } else { -1.0 };
        lut[byte as usize] = if exp == 0 {
            if mant == 0 {
                s * 0.0
            } else {
                s * (mant as f32) * exp2(-9)
            }
        } else if exp == 0xF && mant == 0x7 {
            f32::NAN
        } else {
            s * exp2(exp - 7) * (1.0 + (mant as f32) / 8.0)
        };
        byte += 1;
    }
    lut
};

/// 2026-09-25: `2^e` for the exponents the table uses (`-9..=8`), because
/// `f32::powi` is not a const fn. Each multiply or divide by 2 is exact in
/// that range, so the result is bit-identical to `powi`; a test checks it.
const fn exp2(e: i32) -> f32 {
    let mut v = 1.0f32;
    let mut i = 0i32;
    if e >= 0 {
        while i < e {
            v *= 2.0;
            i += 1;
        }
    } else {
        while i < -e {
            v /= 2.0;
            i += 1;
        }
    }
    v
}

/// 2026-09-25: The E4M3 table.
pub fn e4m3_lut() -> &'static [f32; 256] {
    &E4M3_LUT
}

#[cfg(test)]
mod lut_tests {
    use super::*;

    /// 2026-09-25: Every entry is bit-identical to the same formula computed
    /// with `f32::powi` (NaN entries must both be NaN).
    #[test]
    fn the_const_table_is_bit_identical_to_the_powi_formula() {
        for byte in 0..256u32 {
            let sign_bit = (byte >> 7) & 1;
            let exp = ((byte >> 3) & 0xF) as i32;
            let mant = byte & 0x7;
            let s: f32 = if sign_bit == 0 { 1.0 } else { -1.0 };
            let expected: f32 = if exp == 0 {
                if mant == 0 {
                    s * 0.0
                } else {
                    s * (mant as f32) * 2.0f32.powi(-9)
                }
            } else if exp == 0xF && mant == 0x7 {
                f32::NAN
            } else {
                s * 2.0f32.powi(exp - 7) * (1.0 + (mant as f32) / 8.0)
            };
            let got = E4M3_LUT[byte as usize];
            if expected.is_nan() {
                assert!(got.is_nan(), "byte {byte}: expected NaN, got {got}");
            } else {
                assert_eq!(
                    got.to_bits(),
                    expected.to_bits(),
                    "byte {byte}: {got} != {expected}"
                );
            }
        }
    }

    #[test]
    fn exp2_matches_powi_across_the_domain() {
        for e in -9..=8i32 {
            assert_eq!(exp2(e).to_bits(), 2.0f32.powi(e).to_bits(), "2^{e}");
        }
    }
}
