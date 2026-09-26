// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Launchers for the FP8-weight prefill GEMMs, the NVFP4-to-FP8
//! weight pre-dequant, the BF16-to-FP8 activation cast and the row-scaled
//! BF16-to-FP8 weight quantizer.
//!
//! Owner: model-layers ops.
//! Invariants: none beyond the types.

#![allow(unused_imports)]

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::weight_map::{Fp8DenseWeight, QuantizedWeight};

use super::*;

/// 2026-09-25: FP8-weight prefill GEMM, `C = A @ B_fp8^T`: A `[M, K]` BF16,
/// B_fp8 `[N, K]` FP8 E4M3 (from [`predequant_nvfp4_to_fp8`]), C `[M, N]`
/// BF16.
///
/// When `k` is a multiple of 32 and `METRALE_FP8_LDMAB` is not `0`, it casts A
/// to FP8 into a scratch buffer ([`bf16_to_fp8`]) and launches
/// `fp8_fp8_gemm_ldmab` (module `w4a16_fp8_ldmab`, K step 32) instead of
/// `kernel`.
#[allow(clippy::too_many_arguments)]
pub fn fp8_gemm_n128(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    b_fp8: DevicePtr,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    if k.is_multiple_of(32) && std::env::var("METRALE_FP8_LDMAB").as_deref() != Ok("0") {
        // 2026-09-25: The handles and the scratch come from the backend's op
        // cache, not from statics: they belong to this backend's modules and
        // allocations, which a process-wide cache would outlive.
        let cache = gpu.op_cache();
        let qk = cache.kernel(gpu, "w4a16", "bf16_to_fp8")?;
        let lk = cache.kernel(gpu, "w4a16_fp8_ldmab", "fp8_fp8_gemm_ldmab")?;
        let need = (m as usize) * (k as usize);
        let a8 = cache.scratch(gpu, "fp8_prefill_activation", need)?;
        bf16_to_fp8(gpu, qk, input, a8, m * k, stream)?;
        return KernelLaunch::new(gpu, lk)
            .grid([div_ceil(n, 128), div_ceil(m, 128), 1])
            .block([256, 1, 1])
            .arg_ptr(a8)
            .arg_ptr(b_fp8)
            .arg_ptr(output)
            .arg_u32(m)
            .arg_u32(n)
            .arg_u32(k)
            .launch(stream);
    }
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 128), div_ceil(m, 64), 1])
        .block([128, 1, 1])
        .arg_ptr(input)
        .arg_ptr(b_fp8)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

#[allow(clippy::too_many_arguments)]
/// 2026-09-25: FP8-weight GEMM through the `fp8_gemm_t_mfast` kernel, with M
/// on the fast grid axis (`blockIdx.x` = M block), so consecutive CTAs share a
/// B panel.
pub fn fp8_gemm_n128_mfast(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    b_fp8: DevicePtr,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(m, 64), div_ceil(n, 128), 1])
        .block([128, 1, 1])
        .arg_ptr(input)
        .arg_ptr(b_fp8)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// 2026-09-25: FP8-weight GEMM through `fp8_gemm_t_m128_mfast`: a 128-row M
/// tile (two 64-row chunks per CTA) with M on the fast grid axis.
pub fn fp8_gemm_m128_mfast(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    b_fp8: DevicePtr,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(m, 128), div_ceil(n, 128), 1])
        .block([128, 1, 1])
        .arg_ptr(input)
        .arg_ptr(b_fp8)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// 2026-09-25: FP8 x FP8 GEMM through `fp8_fp8_gemm_t_m128_mfast`: a 128-row
/// M tile with M on the fast grid axis. A must already be FP8 E4M3
/// ([`bf16_to_fp8`]).
pub fn fp8_fp8_gemm_m128_mfast(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    b_fp8: DevicePtr,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(m, 128), div_ceil(n, 128), 1])
        .block([128, 1, 1])
        .arg_ptr(input)
        .arg_ptr(b_fp8)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// 2026-09-25: Dequantize an NVFP4 weight to FP8 E4M3: reads
/// `B_packed[N, K/2]`, `B_scale[N, K/16]` and `scale2`, writes `B_fp8[N, K]`.
/// `QuantizedWeight::predequant_to_fp8` calls it.
pub fn predequant_nvfp4_to_fp8(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    b_packed: DevicePtr,
    b_scale: DevicePtr,
    scale2: f32,
    b_fp8: DevicePtr,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    let total = n * k / 2;
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(total, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(b_packed)
        .arg_ptr(b_scale)
        .arg_f32(scale2)
        .arg_ptr(b_fp8)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// 2026-09-25: Cast `total_elements` BF16 values to FP8 E4M3, two per thread.
/// `total_elements` must be even: the kernel reads and writes pairs.
pub fn bf16_to_fp8(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    src: DevicePtr,
    dst: DevicePtr,
    total_elements: u32,
    stream: u64,
) -> Result<()> {
    let threads_needed = total_elements / 2;
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(threads_needed, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(src)
        .arg_ptr(dst)
        .arg_u32(total_elements)
        .launch(stream)
}

/// 2026-09-25: Quantize a BF16 weight `[N, K]` to FP8 E4M3 `[N, K]` with one
/// f32 scale per row `[N]`, one CTA per row (kernel `quantize_bf16_to_fp8` in
/// `kernels/gb10/common/dense_gemv_fp8w.cu`). `DenseWeight::quantize_to_fp8`
/// calls it; the resulting `Fp8DenseWeight` feeds the row-scaled FP8 kernels
/// such as [`fp8_gemm_n128_row_scaled`].
#[allow(clippy::too_many_arguments)]
pub fn quantize_bf16_to_fp8(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    output: DevicePtr,
    row_scales: DevicePtr,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([n, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(output)
        .arg_ptr(row_scales)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// 2026-09-25: [`fp8_gemm_n128_row_scaled`] through the single-warp kernel
/// `fp8_gemm_t_row_scaled_m16`, whose M tile is 16 rows. The grid has no M
/// dimension, so `m` must be at most 16: rows past the tile are not computed.
#[allow(clippy::too_many_arguments)]
pub fn fp8_gemm_n128_row_scaled_m16(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &Fp8DenseWeight,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 128), 1, 1])
        .block([32, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.row_scale)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// 2026-09-25: Row-scaled FP8 GEMM,
/// `C[M, N] = A[M, K] @ (dequant(B_fp8[N, K]) * row_scale[N])^T`; the scale
/// multiplies each output column before the BF16 store. Takes the
/// `Fp8DenseWeight` from [`crate::weight_map::DenseWeight::quantize_to_fp8`].
/// The DFlash drafter calls it when its quantization is `Fp8Weights`
/// (`dflash_head/forward_block_layer_paged.rs`).
///
/// Kernel: `fp8_gemm_t_row_scaled` (per-model `w4a16_gemm.cu`).
#[allow(clippy::too_many_arguments)]
pub fn fp8_gemm_n128_row_scaled(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &Fp8DenseWeight,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 128), div_ceil(m, 64), 1])
        .block([128, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.row_scale)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}
