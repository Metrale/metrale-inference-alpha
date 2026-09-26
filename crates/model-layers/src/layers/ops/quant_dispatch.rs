// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: GEMV/GEMM dispatch on the weight's quantization format, and the
//! NVFP4 (W4A16) GEMV launchers: single row, fixed 2 and 3 rows, the batched
//! tiers, and the variants that deinterleave Q/Gate or QKVZ as they store.
//!
//! Owner: model-layers ops.
//! Invariants: none beyond the types.

#![allow(unused_imports)]

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::layers::moe;
use crate::weight_map::{DenseWeight, Fp8DenseWeight, Fp8Weight, QuantizedWeight};

use super::*;

/// 2026-09-25: GEMV through the kernel for the weight's format: NVFP4, FP8 or
/// dense BF16. `PackedQ2` weights return an error.
#[allow(clippy::too_many_arguments)]
pub fn quant_gemv(
    gpu: &dyn GpuBackend,
    gemv_nvfp4: KernelHandle,
    gemv_fp8: KernelHandle,
    gemv_dense: KernelHandle,
    input: DevicePtr,
    weight: &crate::weight_map::QuantWeight,
    output: DevicePtr,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    use crate::weight_map::QuantWeight;
    match weight {
        QuantWeight::Nvfp4(w) => w4a16_gemv(gpu, gemv_nvfp4, input, w, output, n, k, stream),
        QuantWeight::Fp8(w) => w8a16_gemv(
            gpu,
            gemv_fp8,
            input,
            w.weight,
            w.row_scale,
            output,
            n,
            k,
            stream,
        ),
        QuantWeight::Dense(w) => dense_gemv(gpu, gemv_dense, input, w, output, n, k, stream),
        // 2026-09-25: No handle here serves PackedQ2; its GEMV is `q2_0_gemv_vec`,
        // launched by the layers themselves.
        QuantWeight::PackedQ2(_) => anyhow::bail!(
            "quant_gemv: PackedQ2 not routed through the generic dispatcher; use q2_0_gemv_vec"
        ),
    }
}

/// 2026-09-25: GEMM (`m` rows) through the kernel for the weight's format:
/// NVFP4, FP8 or dense BF16. `PackedQ2` weights return an error.
#[allow(clippy::too_many_arguments)]
pub fn quant_gemm(
    gpu: &dyn GpuBackend,
    gemm_nvfp4: KernelHandle,
    gemm_fp8: KernelHandle,
    gemm_dense: KernelHandle,
    input: DevicePtr,
    weight: &crate::weight_map::QuantWeight,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    use crate::weight_map::QuantWeight;
    match weight {
        QuantWeight::Nvfp4(w) => w4a16_gemm(gpu, gemm_nvfp4, input, w, output, m, n, k, stream),
        QuantWeight::Fp8(w) => w8a16_gemm(
            gpu,
            gemm_fp8,
            input,
            w.weight,
            w.row_scale,
            output,
            m,
            n,
            k,
            stream,
        ),
        QuantWeight::Dense(w) => dense_gemm(gpu, gemm_dense, input, w, output, m, n, k, stream),
        QuantWeight::PackedQ2(_) => anyhow::bail!(
            "quant_gemm: PackedQ2 not routed through the generic dispatcher; \
             use the layer's transient-dequant prefill path"
        ),
    }
}

/// 2026-09-25: `C[1, N] = A[1, K] @ dequant(B)` with BF16 A and C and an NVFP4
/// weight.
pub fn w4a16_gemv(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([w4a16_gemv_grid_x(n), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_f32(weight.weight_scale_2)
        .arg_ptr(output)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// 2026-09-25: [`w4a16_gemv`] for 2 contiguous rows in one pass over the
/// weight. `gemv_tc::tc_fixed_m` takes the launch to the tensor-core kernel
/// when it routes it (`METRALE_W4A16_TC_WIDE`); otherwise `kernel` runs.
pub fn w4a16_gemv_batch2(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    if super::gemv_tc::tc_fixed_m(gpu, input, weight, output, 2, n, k, stream)? {
        return Ok(());
    }
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_f32(weight.weight_scale_2)
        .arg_ptr(output)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// 2026-09-25: [`w4a16_gemv_batch2`] for 3 rows.
pub fn w4a16_gemv_batch3(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    if super::gemv_tc::tc_fixed_m(gpu, input, weight, output, 3, n, k, stream)? {
        return Ok(());
    }
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_f32(weight.weight_scale_2)
        .arg_ptr(output)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// 2026-09-25: [`w4a16_gemv`] for `m` contiguous rows in one pass over the
/// weight. `kernel` is the CUDA-core `w4a16_gemv_batch{M}` tier the caller
/// picked (`W4a16BatchmTiers::kernel`); `gemv_tc::tc_kernel` replaces it with
/// the tensor-core `w4a16_gemv_tc8`/`tc16` when it routes the shape.
pub fn w4a16_gemv_batchm(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    // 2026-09-25: A tier writes only its first `MAX_M` rows (`w4a16_gemv.cu`), and
    // the widest, `w4a16_gemv_batch32`, stops at 32. Under
    // `--w4a4-downcast-wide`, `W4a16BatchmTiers::kernel` returns that handle
    // for 33..=64 rows as well, rows the W4A4 path is meant to serve, so a
    // launch here above 32 rows is refused instead of partly written.
    anyhow::ensure!(m <= 32, "w4a16_gemv_batchm caps at M=32 (batch32; m={m})");
    // 2026-09-25: The tensor-core kernel takes the same arguments with its own grid
    // and block (`gemv_tc`).
    let (kernel, grid_x, block_x) = match super::gemv_tc::tc_kernel(gpu, m, n, k) {
        Some((tc, grid_x)) => (tc, grid_x, super::gemv_tc::TC_BLOCK),
        None => (kernel, div_ceil(n, 4), 256),
    };
    KernelLaunch::new(gpu, kernel)
        .grid([grid_x, 1, 1])
        .block([block_x, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_f32(weight.weight_scale_2)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// 2026-09-25: [`w4a16_gemv`] over a weight whose rows interleave Q and Gate per
/// head; the output is stored deinterleaved, `[Q of all heads | Gate of all
/// heads]`, with `n = num_heads * head_dim * 2`.
#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemv_qg(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    n: u32,
    k: u32,
    num_heads: u32,
    head_dim: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_f32(weight.weight_scale_2)
        .arg_ptr(output)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(num_heads)
        .arg_u32(head_dim)
        .launch(stream)
}

/// 2026-09-25: [`w4a16_gemv`] over a weight whose rows interleave Q, K, V and Z
/// per group; the output is stored deinterleaved as `[Q | K | V | Z]`.
#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemv_qkvz(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    n: u32,
    k: u32,
    num_groups: u32,
    head_k_dim: u32,
    vheads_per_group: u32,
    head_v_dim: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_f32(weight.weight_scale_2)
        .arg_ptr(output)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(num_groups)
        .arg_u32(head_k_dim)
        .arg_u32(vheads_per_group)
        .arg_u32(head_v_dim)
        .launch(stream)
}

/// 2026-09-25: [`w4a16_gemv_qg`] for 2 rows: A is `[2, K]`, C is `[2, N]`,
/// each row deinterleaved.
#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemv_qg_batch2(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    n: u32,
    k: u32,
    num_heads: u32,
    head_dim: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_f32(weight.weight_scale_2)
        .arg_ptr(output)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(num_heads)
        .arg_u32(head_dim)
        .launch(stream)
}

/// 2026-09-25: [`w4a16_gemv_qg`] for 3 rows: A is `[3, K]`, C is `[3, N]`,
/// each row deinterleaved.
#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemv_qg_batch3(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    n: u32,
    k: u32,
    num_heads: u32,
    head_dim: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_f32(weight.weight_scale_2)
        .arg_ptr(output)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(num_heads)
        .arg_u32(head_dim)
        .launch(stream)
}

/// 2026-09-25: Two NVFP4 GEMVs over the same 3-row input `[3, K]`; `blockIdx.z`
/// picks `weight0` or `weight1`, and each writes `[3, N]`.
#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemv_dual_batch3(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight0: &QuantizedWeight,
    output0: DevicePtr,
    weight1: &QuantizedWeight,
    output1: DevicePtr,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    // 2026-09-25: One tensor-core launch per projection. The route depends only on
    // `(m, n, k)`, the cached handles and switches read once per process, all the
    // same for both calls, so the second launches whenever the first did.
    if super::gemv_tc::tc_fixed_m(gpu, input, weight0, output0, 3, n, k, stream)? {
        super::gemv_tc::tc_fixed_m(gpu, input, weight1, output1, 3, n, k, stream)?;
        return Ok(());
    }
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 2])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight0.weight)
        .arg_ptr(weight0.weight_scale)
        .arg_f32(weight0.weight_scale_2)
        .arg_ptr(output0)
        .arg_ptr(weight1.weight)
        .arg_ptr(weight1.weight_scale)
        .arg_f32(weight1.weight_scale_2)
        .arg_ptr(output1)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// 2026-09-25: [`w4a16_gemv_dual_batch3`] for 2 rows.
#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemv_dual_batch2(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight0: &QuantizedWeight,
    output0: DevicePtr,
    weight1: &QuantizedWeight,
    output1: DevicePtr,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    // 2026-09-25: One tensor-core launch per projection. The route depends only on
    // `(m, n, k)`, the cached handles and switches read once per process, all the
    // same for both calls, so the second launches whenever the first did.
    if super::gemv_tc::tc_fixed_m(gpu, input, weight0, output0, 2, n, k, stream)? {
        super::gemv_tc::tc_fixed_m(gpu, input, weight1, output1, 2, n, k, stream)?;
        return Ok(());
    }
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 2])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight0.weight)
        .arg_ptr(weight0.weight_scale)
        .arg_f32(weight0.weight_scale_2)
        .arg_ptr(output0)
        .arg_ptr(weight1.weight)
        .arg_ptr(weight1.weight_scale)
        .arg_f32(weight1.weight_scale_2)
        .arg_ptr(output1)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}
