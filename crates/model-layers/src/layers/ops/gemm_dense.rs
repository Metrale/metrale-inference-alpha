// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Launchers for the W4A16 GEMMs (NVFP4 weights, BF16
//! activations), and the re-export of the dense BF16 GEMM launchers.
//!
//! Owner: model-layers ops.
//! Invariants: none beyond the types.
//!
//! `gemm_dense_tests.rs` reads this file as text: it counts the `.arg_*` calls
//! of the `_ldb` launchers and reads the body of `w4a16_gemm_n128`.

#![allow(unused_imports)]

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::layers::moe;
use crate::weight_map::{DenseWeight, Fp8DenseWeight, Fp8Weight, QuantizedWeight};

use super::*;
#[path = "gemm_dense_bf16.rs"]
mod bf16;
pub use bf16::{
    dense_gemm, dense_gemm_bf16_pipelined, dense_gemm_prefill, dense_gemm_router,
    dense_gemm_splitk, dense_gemm_tc, dense_gemm_tc_scaled_acc,
};

/// 2026-09-25: W4A16 GEMM, `C = A @ dequant(B)`: A `[M, K]` BF16, B NVFP4
/// (packed E2M1, FP8 block scales, FP32 per-tensor scale), C `[M, N]` BF16.
///
/// Also launches `w4a16_gemm_t_k64_n64_p3`, which takes the same arguments
/// and the same 64-wide N tile.
pub fn w4a16_gemm(
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
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 64), div_ceil(m, 64), 1])
        .block([128, 1, 1])
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

#[allow(clippy::too_many_arguments)]
/// 2026-09-25: [`w4a16_gemm_n128`] with an explicit row stride `ldb` for the
/// transposed B, so row `r` starts at `r * ldb`.
///
/// The tile kernels load B in 16-byte `cp.async` chunks, which need 16-byte
/// aligned sources, so an N that is not a multiple of 16 cannot be the
/// stride. The lm_head's N is the vocab size, which can be odd; its padded
/// twin is built at `align_up(vocab, 128)` with the pad columns zeroed
/// (`impl_a1.rs`, `transpose.rs`), and callers pass that as `ldb`.
pub fn w4a16_gemm_n128_ldb(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    ldb: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 128), div_ceil(m, 64), 1])
        .block([128, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_f32(weight.weight_scale_2)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(ldb)
        .launch(stream)
}

/// 2026-09-25: W4A16 GEMM with a 128-wide N tile over a packed transposed B:
/// it launches through [`w4a16_gemm_n128_ldb`] with `ldb = n`.
pub fn w4a16_gemm_n128(
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
    w4a16_gemm_n128_ldb(gpu, kernel, input, weight, output, m, n, k, n, stream)
}

/// 2026-09-25: W4A16 GEMM v3 (`w4a16_gemm_t_m128_v3`, only in
/// `kernels/gb10/minimax-m2-229b/nvfp4/w4a16_gemm_v3.cu`): a 128x128 CTA tile
/// on 256 threads with a K step of 64.
#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemm_n128_m128_v3(
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
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 128), div_ceil(m, 128), 1])
        .block([256, 1, 1])
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

/// 2026-09-25: W4A16 GEMM v2 (`w4a16_gemm_t_m128_v2`, in the minimax-m2-229b
/// and qwen3.6-27b kernel dirs): the CTA tile of [`w4a16_gemm_n128_m128`]
/// (M=128, N=128, K step 32) on 256 threads instead of 128. Warps 0-3 compute
/// the first 64 rows and warps 4-7 the second, so the two chunks' MMAs run in
/// parallel.
#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemm_n128_m128_v2(
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
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 128), div_ceil(m, 128), 1])
        .block([256, 1, 1])
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

/// 2026-09-25: W4A16 GEMM with a 128-row CTA (two 64-row chunks): each CTA
/// loads a B tile once for both chunks.
///
/// Grid contract: N is the fast axis (`blockIdx.x` = N block, `blockIdx.y` =
/// M block), as every `w4a16_gemm_t_m128` in `kernels/` reads it. Several
/// layers share this launcher (qwen3_attention, dense_ffn, qwen3_ssm,
/// nemotron_mamba2, nemotron_moe), so swapping the axes here would mis-map
/// every CTA of the other kernels with no error. An M-fast order needs its
/// own kernel and launcher, as `w4a4_gemm_mfast` and `fp8_gemm_t_m128_mfast`
/// have.
#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemm_n128_m128(
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
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 128), div_ceil(m, 128), 1])
        .block([128, 1, 1])
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

#[allow(clippy::too_many_arguments)]
/// 2026-09-25: [`w4a16_gemm_n128_m128_bf16`] with an explicit transposed-B
/// row stride, for the same reason as [`w4a16_gemm_n128_ldb`]. This is the
/// launcher for the 9-parameter `w4a16_gemm_t_m128_bf16_v2`.
pub fn w4a16_gemm_n128_m128_bf16_ldb(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    ldb: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 128), div_ceil(m, 128), 1])
        .block([128, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_f32(weight.weight_scale_2)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(ldb)
        .launch(stream)
}

/// 2026-09-25: W4A16 GEMM launching `w4a16_gemm_t_m128_bf16`, which runs BF16
/// m16n8k16 MMAs, on the grid, block and transposed NVFP4 weight layout of
/// [`w4a16_gemm_n128_m128`]. It passes 8 arguments, so it fits only the
/// 8-parameter v1 kernel; `w4a16_gemm_t_m128_bf16_v2` takes a 9th (`ldb`)
/// and goes through [`w4a16_gemm_n128_m128_bf16_ldb`] (`ldb = N` for an
/// unpadded twin).
pub fn w4a16_gemm_n128_m128_bf16(
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
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 128), div_ceil(m, 128), 1])
        .block([128, 1, 1])
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

#[cfg(test)]
#[path = "gemm_dense_tests.rs"]
mod gemm_dense_tests;
