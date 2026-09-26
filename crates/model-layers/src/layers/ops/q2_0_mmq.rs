// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Launcher for the packed Q2_0 (ternary) MMQ prefill GEMM,
//! `metrale_q2_0_mmq128_nc/_wc` in `kernels/gb10/qwen3.6-27b/nvfp4/q2_0_mmq.cu`.
//! The 2-bit weight stays packed: the vendored `load_tiles_q2_0` unpacks
//! `code - 1` into the int8 tile, and `vec_dot_q8_0_q8_1_mma` multiplies it by a
//! q8_1 activation and applies both scales. The activation quantizer and scratch
//! size are Q4_K's ([`super::quantize_act_q8_1`], [`super::q8_1_scratch_bytes`]),
//! in the DS4 layout.
//!
//! Owner: model-layers ops.
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::weight_map::PackedQ2Weight;

/// 2026-09-25: Weights per `block_q2_0`.
pub const QK2_0: u32 = 128;
/// 2026-09-25: `sizeof(block_q2_0)`: an fp16 scale `d` and 128 codes at 4 per byte.
pub const Q2_0_BLOCK_BYTES: usize = 34;

/// 2026-09-25: True only when `METRALE_GGUF_NATIVE_Q2_MMQ` is exactly `1`. With
/// it, and with the kernels resolved and group 128 on gate, up and down, the
/// dense FFN prefill runs packed Q2 weights through [`q2_0_mmq_gemm`] instead of
/// dequantizing them to BF16. Whether weights stay packed at all is
/// `METRALE_GGUF_NATIVE_Q2`'s decision.
pub fn native_q2_mmq_enabled() -> bool {
    std::env::var("METRALE_GGUF_NATIVE_Q2_MMQ").ok().as_deref() == Some("1")
}

/// 2026-09-25: Bytes of the packed `block_q2_0` form of an `[n, k]` weight; `k`
/// must be a multiple of 128.
pub fn q2_0_weight_bytes(n: u32, k: u32) -> usize {
    (n as usize) * (k as usize / QK2_0 as usize) * Q2_0_BLOCK_BYTES
}

/// 2026-09-25: `C[m, n] = A[m, k] x W[n, k]` in BF16. `a_q8` is the DS4 q8_1
/// activation from [`super::quantize_act_q8_1`]; `w_q2_0` is the packed weight as
/// loaded. The tile is 128 x 128 like the Q4_K GEMM, and Q2_0's x tile row
/// (`MMQ_MMA_TILE_X_K_Q8_0`) is the same 76 ints as Q4_K's, so it reuses
/// [`Q4K_MMQ_SMEM`](super::Q4K_MMQ_SMEM). `kernel_wc` is used when `n` is not a
/// multiple of 128.
#[allow(clippy::too_many_arguments)]
pub fn q2_0_mmq_gemm(
    gpu: &dyn GpuBackend,
    kernel_nc: KernelHandle,
    kernel_wc: KernelHandle,
    a_q8: DevicePtr,
    w_q2_0: DevicePtr,
    out_bf16: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    let kernel = if !n.is_multiple_of(128) {
        kernel_wc
    } else {
        kernel_nc
    };
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 128), div_ceil(m, 128), 1])
        .block([32, 8, 1])
        .shared_mem(super::q4k_mmq::Q4K_MMQ_SMEM)
        .arg_ptr(w_q2_0)
        .arg_ptr(a_q8)
        .arg_ptr(out_bf16)
        .arg_u32(n)
        .arg_u32(m)
        .arg_u32(k)
        .arg_u32(k / QK2_0)
        .arg_u32(m)
        .arg_u32(n)
        .launch(stream)
}

/// 2026-09-25: [`q2_0_mmq_gemm`] on a [`PackedQ2Weight`]. Returns an error unless
/// `w.group == 128`, the group size of `block_q2_0`.
pub fn q2_0_mmq_gemm_packed(
    gpu: &dyn GpuBackend,
    kernel_nc: KernelHandle,
    kernel_wc: KernelHandle,
    a_q8: DevicePtr,
    w: &PackedQ2Weight,
    out_bf16: DevicePtr,
    m: u32,
    stream: u64,
) -> Result<()> {
    anyhow::ensure!(
        w.group == 128,
        "Q2_0 MMQ requires group 128 (got {}); use the transient-dequant path for group 64",
        w.group
    );
    q2_0_mmq_gemm(
        gpu, kernel_nc, kernel_wc, a_q8, w.weight, out_bf16, m, w.n, w.k, stream,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    use half::f16;

    // 2026-09-25: The block stores the weight scale `d` as fp16, so both models
    // round it through fp16.
    fn f32_to_f16_bits(x: f32) -> u16 {
        f16::from_f32(x).to_bits()
    }
    fn f16_bits_to_f32(bits: u16) -> f32 {
        f16::from_bits(bits).to_f32()
    }

    /// 2026-09-25: Pack an `[n, k]` code matrix (values 0..=2, weight `code - 1`)
    /// and one scale per row and 128 codes into `block_q2_0` bytes: fp16 `d`, then
    /// code `j` at bits `2 * (j % 4)` of byte `2 + j / 4`.
    fn pack_q2_0(codes: &[u8], scales: &[f32], n: usize, k: usize) -> Vec<u8> {
        assert_eq!(k % 128, 0);
        let blocks_per_row = k / 128;
        let mut out = vec![0u8; n * blocks_per_row * Q2_0_BLOCK_BYTES];
        for row in 0..n {
            for b in 0..blocks_per_row {
                let blk = (row * blocks_per_row + b) * Q2_0_BLOCK_BYTES;
                let dbits = f32_to_f16_bits(scales[row * blocks_per_row + b]);
                out[blk] = (dbits & 0xff) as u8;
                out[blk + 1] = (dbits >> 8) as u8;
                for j in 0..128 {
                    let c = codes[row * k + b * 128 + j] & 0x3;
                    let byte = blk + 2 + j / 4;
                    out[byte] |= c << (2 * (j % 4));
                }
            }
        }
        out
    }

    /// 2026-09-25: CPU model of the Q2_0 MMQ arithmetic: quantize the activation
    /// per 32 values (`d_a = absmax / 127`, round to nearest int8), sum int8
    /// products per 32, then scale by `d_w * d_a`. Output is row-major `[m, n]`.
    fn mmq_cpu(
        act: &[f32],
        codes: &[u8],
        scales: &[f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Vec<f32> {
        let bpr = k / 128;
        let ng = k / 32;
        let mut a_q = vec![0i8; m * k];
        let mut a_d = vec![0f32; m * ng];
        for r in 0..m {
            for g in 0..ng {
                let mut amax = 0f32;
                for t in 0..32 {
                    amax = amax.max(act[r * k + g * 32 + t].abs());
                }
                let d = amax / 127.0;
                a_d[r * ng + g] = d;
                for t in 0..32 {
                    let q = if d > 0.0 {
                        (act[r * k + g * 32 + t] / d).round().clamp(-127.0, 127.0)
                    } else {
                        0.0
                    };
                    a_q[r * k + g * 32 + t] = q as i8;
                }
            }
        }
        let mut out = vec![0f32; m * n];
        for r in 0..m {
            for col in 0..n {
                let mut acc = 0f32;
                for g in 0..ng {
                    let b = g / 4;
                    let dw = f16_bits_to_f32(f32_to_f16_bits(scales[col * bpr + b]));
                    let da = a_d[r * ng + g];
                    let mut isum = 0i32;
                    for t in 0..32 {
                        let ki = g * 32 + t;
                        let w = (codes[col * k + ki] & 0x3) as i32 - 1;
                        isum += w * a_q[r * k + ki] as i32;
                    }
                    acc += isum as f32 * dw * da;
                }
                out[r * n + col] = acc;
            }
        }
        out
    }

    /// 2026-09-25: FP oracle: dequantized `(code - 1) * d` times the unquantized
    /// activation.
    fn oracle(act: &[f32], codes: &[u8], scales: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        let bpr = k / 128;
        let mut out = vec![0f32; m * n];
        for r in 0..m {
            for col in 0..n {
                let mut acc = 0f32;
                for ki in 0..k {
                    let b = ki / 128;
                    let dw = f16_bits_to_f32(f32_to_f16_bits(scales[col * bpr + b]));
                    let w = ((codes[col * k + ki] & 0x3) as i32 - 1) as f32 * dw;
                    acc += w * act[r * k + ki];
                }
                out[r * n + col] = acc;
            }
        }
        out
    }

    #[test]
    fn q2_0_block_layout_matches_spec() {
        assert_eq!(Q2_0_BLOCK_BYTES, 2 + 128 / 4);
        assert_eq!(QK2_0, 128);
        assert_eq!(q2_0_weight_bytes(256, 512), 256 * (512 / 128) * 34);
        let codes: Vec<u8> = (0..128).map(|j| (j % 3) as u8).collect();
        let packed = pack_q2_0(&codes, &[0.5], 1, 128);
        assert_eq!(packed.len(), 34);
        for j in 0..128usize {
            let got = (packed[2 + j / 4] >> (2 * (j % 4))) & 0x3;
            assert_eq!(got, (j % 3) as u8, "code {j} mispacked");
        }

        let launcher = include_str!("../../../../../kernels/gb10/qwen3.6-27b/nvfp4/q2_0_mmq.cu");
        let vendor =
            include_str!("../../../../../kernels/gb10/qwen3.6-27b/nvfp4/q4k_vendor/mmq.cuh");
        assert!(launcher.contains("constexpr ggml_type type = GGML_TYPE_Q2_0;"));
        assert!(launcher.contains("metrale_q2_0_tile<128, false>"));
        assert!(launcher.contains("metrale_q2_0_tile<128, true>"));
        assert!(vendor.contains("load_tiles   = load_tiles_q2_0<mmq_y, need_check>;"));
        assert!(vendor.contains(
            "vec_dot_mma  = vec_dot_q8_0_q8_1_mma<mmq_x, mmq_y, MMQ_Q8_1_DS_LAYOUT_DS4>;"
        ));
        assert_eq!(
            vendor.matches("& 0x3) - 1;").count(),
            4,
            "the Q2 unpack must subtract one from all four packed codes"
        );
    }

    #[test]
    fn q2_0_mmq_reference_arithmetic_matches_fp_oracle() {
        // 2026-09-25: K spans two 128-blocks, so the block boundary is crossed.
        let (m, n, k) = (3usize, 5usize, 256usize);
        let bpr = k / 128;
        let mut codes = vec![0u8; n * k];
        for (i, c) in codes.iter_mut().enumerate() {
            *c = (((i * 2654435761usize) >> 5) % 3) as u8;
        }
        let mut scales = vec![0f32; n * bpr];
        for (i, s) in scales.iter_mut().enumerate() {
            *s = 0.015 + 0.01 * ((i % 7) as f32);
        }
        let mut act = vec![0f32; m * k];
        for (i, a) in act.iter_mut().enumerate() {
            let x = (i as f32 * 0.12345).sin();
            *a = x * 1.7;
        }

        let mmq = mmq_cpu(&act, &codes, &scales, m, n, k);
        let orc = oracle(&act, &codes, &scales, m, n, k);

        // 2026-09-25: The two differ only by the int8 activation quantization.
        let mut max_rel = 0f32;
        let mut denom = 0f32;
        let mut num = 0f32;
        for i in 0..m * n {
            let e = (mmq[i] - orc[i]).abs();
            assert!(
                e <= 0.02 * orc[i].abs().max(0.5),
                "idx {i}: mmq {} vs oracle {}",
                mmq[i],
                orc[i]
            );
            num += e * e;
            denom += orc[i] * orc[i];
            let r = e / orc[i].abs().max(1e-3);
            max_rel = max_rel.max(r);
        }
        let l2_rel = (num / denom.max(1e-12)).sqrt();
        assert!(
            l2_rel < 1e-2,
            "Q2_0 MMQ L2 rel_err {l2_rel:.4e} exceeds 1e-2 (max pointwise {max_rel:.4e})"
        );
    }
}
