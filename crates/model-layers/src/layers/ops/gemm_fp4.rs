// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Launchers for the W4A4 prefill path: BF16-to-NVFP4 activation
//! quantization and the NVFP4 x NVFP4 tensor-core GEMMs.
//!
//! Owner: model-layers ops.
//! Invariants: none beyond the types.

#![allow(unused_imports)]

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::weight_map::QuantizedWeight;

use super::*;

/// 2026-09-25: Quantize a BF16 `[M, K]` matrix to NVFP4 with `scale2 = 1.0`:
/// packed E2M1 `[M, K/2]` and one E4M3 scale per 16 elements `[M, K/16]`. One
/// block per row; the kernel's `N` parameter is the row count. Produces the
/// W4A4 prefill activations.
#[allow(clippy::too_many_arguments)]
pub fn quantize_bf16_to_nvfp4(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    packed_out: DevicePtr,
    scale_out: DevicePtr,
    m: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([m, 1, 1])
        .block([128, 1, 1])
        .arg_ptr(input)
        .arg_ptr(packed_out)
        .arg_ptr(scale_out)
        .arg_f32(1.0)
        .arg_u32(m)
        .arg_u32(k)
        .launch(stream)
}

/// 2026-09-25: W4A4 GEMM on FP4 tensor cores (kernel `w4a4_gemm`). The
/// activation comes from [`quantize_bf16_to_nvfp4`] and is launched with
/// `scaleA2 = 1.0`; the weight is an NVFP4 `QuantizedWeight`; the output is
/// BF16 `[M, N]`.
#[allow(clippy::too_many_arguments)]
pub fn w4a4_gemm(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    a_packed: DevicePtr,
    a_scale: DevicePtr,
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
        .arg_ptr(a_packed)
        .arg_ptr(a_scale)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_ptr(output)
        .arg_f32(1.0)
        .arg_f32(weight.weight_scale_2)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

#[allow(clippy::too_many_arguments)]
/// 2026-09-25: [`w4a4_gemm`] through the `w4a4_gemm_mfast` kernel, with M on
/// the fast grid axis (`blockIdx.x` = M block), so consecutive CTAs share a B
/// panel.
pub fn w4a4_gemm_mfast(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    a_packed: DevicePtr,
    a_scale: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(m, 128), div_ceil(n, 128), 1])
        .block([128, 1, 1])
        .arg_ptr(a_packed)
        .arg_ptr(a_scale)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_ptr(output)
        .arg_f32(1.0)
        .arg_f32(weight.weight_scale_2)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}
