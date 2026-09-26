// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Launchers for the FP8 E4M3 weight x BF16 activation (W8A16) GEMV and GEMMs, row-major and transposed.
//!
//! Owner: model-layers ops.
//! Invariants:
//! - Every launcher passes the arguments in the order `(A, B, scale, C, [M,] N, K)` that its
//!   kernel's `extern "C"` signature declares.
//! - Each grid and block is derived from the tile constants of the kernel it launches; the
//!   kernel files are named on each function.

use super::*;

/// 2026-09-25: W8A16 GEMV for M=1 decode: `C[1,N] = A[1,K] @ dequant(B[N,K])^T`.
///
/// `row_scale` is the 2D block scale `[ceil(N/128), ceil(K/128)]` FP32, one scale per
/// 128x128 weight block, despite the parameter name. Each 256-thread block computes 4 outputs
/// with 64 threads per output, so the grid is `ceil(N/4)` (kernels/gb10/common/w8a16_gemv.cu,
/// and the same geometry in kernels/hopper/common/w8a16_gemv.cu).
#[allow(clippy::too_many_arguments)]
pub fn w8a16_gemv(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: DevicePtr,
    row_scale: DevicePtr,
    output: DevicePtr,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight)
        .arg_ptr(row_scale)
        .arg_ptr(output)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// 2026-09-25: W8A16 GEMM for M>1: `C[M,N] = A[M,K] @ dequant(B[N,K])^T`, with the 2D block
/// scale `[ceil(N/128), ceil(K/128)]` FP32.
#[allow(clippy::too_many_arguments)]
pub fn w8a16_gemm(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: DevicePtr,
    block_scale: DevicePtr,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    // 2026-09-25: The geometry is per target because the kernel source is. The native-HIP
    // kernel (kernels/strix-hip/common/w8a16_gemm.cu) is a 256x128 MxN tile with 512 threads;
    // every other target builds kernels/gb10/common/w8a16_gemm.cu, a 64x64 tile with 128
    // threads. Keep both arms equal to their kernel's `M_TILE`/`N_TILE`/`THREADS`.
    #[cfg(metrale_hip)]
    let (grid, block) = ([div_ceil(n, 128), div_ceil(m, 256), 1], [512, 1, 1]);
    #[cfg(not(metrale_hip))]
    let (grid, block) = ([div_ceil(n, 64), div_ceil(m, 64), 1], [128, 1, 1]);
    KernelLaunch::new(gpu, kernel)
        .grid(grid)
        .block(block)
        .arg_ptr(input)
        .arg_ptr(weight)
        .arg_ptr(block_scale)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// 2026-09-25: Pipelined W8A16 GEMM (kernel `w8a16_gemm_pipelined`), with the same arguments
/// and block-scale layout as [`w8a16_gemm`]. A 128x32 MxN tile with 256 threads and a
/// cp.async prefetch (`PM_M_TILE`, `PM_N_TILE`, `PM_THREADS` in
/// kernels/gb10/common/w8a16_gemm_pipelined.cu).
#[allow(clippy::too_many_arguments)]
pub fn w8a16_gemm_pipelined(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: DevicePtr,
    block_scale: DevicePtr,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 32), div_ceil(m, 128), 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight)
        .arg_ptr(block_scale)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// 2026-09-25: Transposed W8A16 GEMM: `C[M,N] = A[M,K] @ dequant(B_t[K,N])`, with the
/// transposed FP32 block scale `block_scale_t[K/128, N/128]`, so weight reads are contiguous
/// along N. Each CTA covers 64 rows and 64 columns with 128 threads
/// (kernels/gb10/common/w8a16_gemm_t.cu, kernels/strix-hip/common/w8a16_gemm_t.cu).
#[allow(clippy::too_many_arguments)]
pub fn w8a16_gemm_t(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight_t: DevicePtr,      // 2026-09-25: [K, N] FP8 transposed
    block_scale_t: DevicePtr, // 2026-09-25: [K/128, N/128] FP32 transposed
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 64), div_ceil(m, 64), 1])
        .block([128, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight_t)
        .arg_ptr(block_scale_t)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// 2026-09-25: Transposed W8A16 GEMM with a 128x128 MxN tile (kernel `w8a16_gemm_t_m128`):
/// two 64-row chunks, 8 warps, `m16n8k16` BF16 MMA, and the block scale folded into an FP32
/// accumulator once per 128-K block (kernels/gb10/common/w8a16_gemm_t_m128.cu). Same
/// arguments and layout as [`w8a16_gemm_t`], so it takes the output of the `transpose_fp8` /
/// `transpose_block_scale` kernels unchanged.
#[allow(clippy::too_many_arguments)]
pub fn w8a16_gemm_n128_m128(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight_t: DevicePtr,      // 2026-09-25: [K, N] FP8 transposed
    block_scale_t: DevicePtr, // 2026-09-25: [K/128, N/128] FP32 transposed
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    super::super::log_gemm_shape(gpu, "w8a16_gemm_t_m128", m, n, k);
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 128), div_ceil(m, 128), 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight_t)
        .arg_ptr(block_scale_t)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// 2026-09-25: Pipelined transposed W8A16 GEMM (kernel `w8a16_gemm_t_pipelined`), with the
/// same arguments as [`w8a16_gemm_t`]: a 128x32 MxN tile with 256 threads (`PT_M_TILE`,
/// `PT_N_TILE`, `PT_THREADS` in kernels/gb10/common/w8a16_gemm_t.cu).
#[allow(clippy::too_many_arguments)]
pub fn w8a16_gemm_t_pipelined(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight_t: DevicePtr,
    block_scale_t: DevicePtr,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    super::super::log_gemm_shape(gpu, "w8a16_gemm_t_pipelined", m, n, k);
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 32), div_ceil(m, 128), 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight_t)
        .arg_ptr(block_scale_t)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}
