// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Launcher for the grouped W4A4 expert up GEMM `moe_w4a4_grouped_gemm_relu2`.
//!
//! Owner: model-layers ops.
//! Invariants: none beyond the types.

#![allow(unused_imports)]

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};

use super::*;

/// 2026-09-25: Grouped W4A4 expert up GEMM on the FP4 block-scale MMA, with
/// relu^2 applied to the accumulator before the BF16 store. A is NVFP4 (packed
/// E2M1 + per-16 E4M3 scales) in token order, gathered through
/// `sorted_token_ids`; B is read from the per-expert NVFP4 pointer tables.
/// `blockIdx.z` is the expert, and `expert_offsets` bounds its rows.
/// The kernel ships only in
/// `kernels/gb10/nemotron-labs-3-puzzle-75b-a9b/nvfp4/moe_w4a4_grouped.cu`.
#[allow(clippy::too_many_arguments)]
pub fn moe_w4a4_grouped_gemm_relu2(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    a_packed: DevicePtr,
    a_sf: DevicePtr,
    b_packed_ptrs: DevicePtr,
    b_scale_ptrs: DevicePtr,
    scale2_vals: DevicePtr,
    output: DevicePtr,
    expert_offsets: DevicePtr,
    sorted_token_ids: DevicePtr,
    num_experts: u32,
    n_out: u32,
    k: u32,
    max_m_tiles: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n_out, 128), max_m_tiles, num_experts])
        .block([128, 1, 1])
        .arg_ptr(a_packed)
        .arg_ptr(a_sf)
        .arg_ptr(b_packed_ptrs)
        .arg_ptr(b_scale_ptrs)
        .arg_ptr(scale2_vals)
        .arg_ptr(output)
        .arg_ptr(expert_offsets)
        .arg_ptr(sorted_token_ids)
        .arg_u32(num_experts)
        .arg_u32(n_out)
        .arg_u32(k)
        .launch(stream)
}
