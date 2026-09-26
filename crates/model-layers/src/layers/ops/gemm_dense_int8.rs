// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Launchers for the int8 prefill GEMM: the NVFP4-to-int8 weight
//! requant, and the two-launch prefill (activation requant, then the int8 x
//! int8 block-scaled GEMM `int8_gemm_faith2`).
//!
//! Owner: model-layers ops.
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};

/// 2026-09-25: Requantize an NVFP4 weight (packed E2M1, per-16 E4M3 block
/// scales, per-tensor `scale2`) to int8 with one F32 scale per 32 elements,
/// for `int8_gemm_faith2`. Reads `W_packed[N, K/2]` and `W_e4m3[N, K/16]`;
/// writes `W_i8[N, K]` (signed int8) and `W_scale[N, K/32]` (F32). The dense
/// FFN's `ensure_int8_weight` runs it on a weight's first int8 prefill and
/// keeps the result.
#[allow(clippy::too_many_arguments)]
pub fn requant_w_nvfp4_int8(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    w_packed: DevicePtr,
    w_e4m3: DevicePtr,
    scale2: f32,
    w_i8: DevicePtr,
    w_scale: DevicePtr,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    let blocks = n * (k / 32);
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(blocks, 128), 1, 1])
        .block([128, 1, 1])
        .arg_ptr(w_packed)
        .arg_ptr(w_e4m3)
        .arg_f32(scale2)
        .arg_ptr(w_i8)
        .arg_ptr(w_scale)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// 2026-09-25: int8 prefill GEMM: requantize the BF16 activations to int8
/// with one F32 scale per 32 elements (`requant_a_kernel`), then multiply them
/// with the int8 weight from [`requant_w_nvfp4_int8`] (`faith2_kernel`). Both
/// launches go on `stream`.
///
/// `a_bf16` `[M, K]` BF16, `w_i8` `[N, K]` int8, `w_scale` `[N, K/32]` F32,
/// `out` `[M, N]` BF16. `a_i8_scratch` and `a_scale_scratch` are caller-owned,
/// of at least `M*K` and `M*(K/32)*4` bytes.
#[allow(clippy::too_many_arguments)]
pub fn int8_gemm_faith2_prefill(
    gpu: &dyn GpuBackend,
    faith2_kernel: KernelHandle,
    requant_a_kernel: KernelHandle,
    a_bf16: DevicePtr,
    w_i8: DevicePtr,
    w_scale: DevicePtr,
    a_i8_scratch: DevicePtr,
    a_scale_scratch: DevicePtr,
    out: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    let a_blocks = m * (k / 32);
    KernelLaunch::new(gpu, requant_a_kernel)
        .grid([div_ceil(a_blocks, 128), 1, 1])
        .block([128, 1, 1])
        .arg_ptr(a_bf16)
        .arg_ptr(a_i8_scratch)
        .arg_ptr(a_scale_scratch)
        .arg_u32(m)
        .arg_u32(k)
        .launch(stream)?;
    KernelLaunch::new(gpu, faith2_kernel)
        .grid([div_ceil(n, 128), div_ceil(m, 128), 1])
        .block([256, 1, 1])
        .arg_ptr(a_i8_scratch)
        .arg_ptr(w_i8)
        .arg_ptr(a_scale_scratch)
        .arg_ptr(w_scale)
        .arg_ptr(out)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}
