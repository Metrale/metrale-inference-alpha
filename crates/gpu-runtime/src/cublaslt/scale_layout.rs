// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Index math for the FP32 scale tensors of the block-scaled
//! FP8 GEMM ([`super::fp8_gemm_act_weight_t_blkscaled`]): the activation
//! scales in the layout it passes to cuBLASLt, in the quantizer's layout, and
//! the weight-scale stride check.
//!
//! In cuBLASLt's terms the GEMM is `D[N,M] = opT(weightᶜ[K,N]) ·
//! opN(actᶜ[K,M])`, so the library's M is the weight's N and its N is the
//! token count. The activation (B, `VEC128_32F`) scales are laid out
//! `[L, m_pad]`, token index contiguous, with `L = ceil(K/128)`; the
//! quantizer `per_token_group_quant_fp8` writes `[M, L]`, K group
//! contiguous. The weight (A, `BLK128x128_32F`) scales are the checkpoint's
//! `[ceil(N/128), L]` grid, used as they are when `L` is a multiple of 4.
//! The layouts follow the cuBLAS manual section "128-element 1D and 128x128
//! 2D Block Scaling For FP8 Data Types".
//!
//! Measured 2026-09-11 on 1xH100 (`native_fp8_ffn_w8a8_microtest`, cuBLASLt
//! 13.1): with the quantizer's `[M, L]` scales passed as B, the GEMM
//! disagreed with the in-tree `fp8_gemm_t_blockscaled` on the same inputs,
//! rel_rms 1.1e-2 at M=64 and 7.7e-2 to 8.8e-2 at M=1193.
//!
//! The metal build compiles this file too (`cublaslt_metal_stub.rs`).
//!
//! Owner: gpu-runtime (cuBLASLt wrapper).
//! Invariants: none beyond the types.

/// 2026-09-25: Number of 128-wide K groups in `k`: `L = ceil(k / 128)`.
#[must_use]
pub fn k_groups(k: usize) -> usize {
    k.div_ceil(128)
}

/// 2026-09-25: Offset of the scale for `(token, k_group)` in the B-operand
/// layout `[L, m_pad]`, token index contiguous. The CUDA adapter
/// `fp8_act_scale_to_kmajor` (`kernels/gb10/common/fp8_scale_transpose.cu`)
/// writes the same index.
#[must_use]
pub const fn vec128_b_index(m_pad: usize, token: usize, k_group: usize) -> usize {
    k_group * m_pad + token
}

/// 2026-09-25: Offset of the same `(token, k_group)` scale in the layout
/// `per_token_group_quant_fp8` writes: row-major `[M, L]`, K group
/// contiguous. `fp8_gemm_t_blockscaled` reads this layout.
#[must_use]
pub const fn rowmajor_index(l: usize, token: usize, k_group: usize) -> usize {
    token * l + k_group
}

/// 2026-09-25: FP32 element count of the B-operand scale tensor:
/// `m_pad * L`.
#[must_use]
pub fn vec128_b_elems(m_pad: usize, k: usize) -> usize {
    m_pad * k_groups(k)
}

/// 2026-09-25: Whether `L = ceil(k / 128)` is a multiple of 4, the condition
/// under which the checkpoint's row-major weight-scale grid is passed to the
/// `BLK128x128_32F` operand unchanged. The cuBLASLt dispatch gates check it
/// (`dense_ffn_w8a8_prefill.rs`, `dispatch_proj_decode.rs`).
#[must_use]
pub fn blk128x128_stride_ok(k: usize) -> bool {
    k_groups(k).is_multiple_of(4)
}

/// 2026-09-25: CPU reference for the `fp8_act_scale_to_kmajor` kernel: read
/// row-major `[m, l]` scales and write `[l, m_pad]`, with the pad tokens
/// `m..m_pad` set to 0.0, as the kernel does.
#[must_use]
pub fn act_scale_rowmajor_to_kmajor(src: &[f32], m: usize, m_pad: usize, l: usize) -> Vec<f32> {
    debug_assert!(m_pad >= m, "pad cannot shrink the token extent");
    debug_assert!(src.len() >= m * l, "source holds fewer than m x l scales");
    let mut dst = vec![0.0f32; vec128_b_elems(m_pad, l * 128)];
    for token in 0..m {
        for kg in 0..l {
            dst[vec128_b_index(m_pad, token, kg)] = src[rowmajor_index(l, token, kg)];
        }
    }
    dst
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2026-09-25: Hand-placed values: `src` is row-major `[M=3, L=2]`, and
    /// the output groups tokens by K group, with one zero pad token each.
    #[test]
    fn kmajor_adapter_transposes_and_zero_fills_the_pad() {
        let src = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let dst = act_scale_rowmajor_to_kmajor(&src, 3, 4, 2);
        assert_eq!(dst, vec![1.0, 3.0, 5.0, 0.0, 2.0, 4.0, 6.0, 0.0]);
    }

    #[test]
    fn adapter_is_identity_free_only_when_one_dimension_is_trivial() {
        // 2026-09-25: With one K group, or one token, the two layouts
        // coincide.
        let src = [7.0f32, 8.0, 9.0];
        assert_eq!(
            act_scale_rowmajor_to_kmajor(&src, 3, 3, 1),
            vec![7.0, 8.0, 9.0]
        );
        assert_eq!(
            act_scale_rowmajor_to_kmajor(&src, 1, 1, 3),
            vec![7.0, 8.0, 9.0]
        );
    }

    #[test]
    fn the_two_readings_disagree_at_the_measured_shape() {
        // 2026-09-25: At M=64, K=5120 (L=40) the two offsets agree only where
        // 63*kg == 39*token, i.e. at 4 of the 2560 scales.
        let (m_pad, l) = (64usize, 40usize);
        let same = (0..m_pad)
            .flat_map(|token| (0..l).map(move |kg| (token, kg)))
            .filter(|(token, kg)| {
                vec128_b_index(m_pad, *token, *kg) == rowmajor_index(l, *token, *kg)
            })
            .count();
        assert_eq!(same, 4, "only (0,0), (21,13), (42,26), (63,39) coincide");
        assert_eq!(m_pad * l - same, 2556);
    }

    #[test]
    fn documented_extents_match_the_shapes_we_serve() {
        // 2026-09-25: K = 5120 and 17408, the `H` and `INTER` of
        // `dense_ffn_w8a8_prefill_tests.rs`.
        assert_eq!(k_groups(5120), 40);
        assert_eq!(k_groups(17408), 136);
        assert!(blk128x128_stride_ok(5120));
        assert!(blk128x128_stride_ok(17408));
        assert!(!blk128x128_stride_ok(128 * 5));
        assert_eq!(vec128_b_elems(1200, 5120), 1200 * 40);
    }
}
