// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Launchers for the expert-sorted MoE GEMVs `moe_sorted_gate_up` and
//! `moe_sorted_silu_down` (`kernels/gb10/common/moe_sorted_prefill.cu`).
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

/// 2026-09-25: Gate and up GEMV over expert-sorted rows. Block `y` reads token
/// `sorted_token_ids[y]` and expert `sorted_expert_ids[y]`; `blockIdx.z` picks
/// gate (0) or up (1). Outputs stay in sorted order.
#[allow(clippy::too_many_arguments, dead_code)]
pub(crate) fn moe_sorted_gate_up(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    gate_ptrs: &moe::ExpertPtrTable,
    up_ptrs: &moe::ExpertPtrTable,
    gate_out: DevicePtr,
    up_out: DevicePtr,
    sorted_token_ids: DevicePtr,
    sorted_expert_ids: DevicePtr,
    inter: u32,
    hidden: u32,
    total_expanded: u32,
    stream: u64,
) -> Result<()> {
    let grid_x = div_ceil(inter, 8);
    KernelLaunch::new(gpu, kernel)
        .grid([grid_x, total_expanded, 2])
        .block([128, 1, 1])
        .arg_ptr(input)
        .arg_ptr(gate_ptrs.packed_ptrs)
        .arg_ptr(gate_ptrs.scale_ptrs)
        .arg_ptr(gate_ptrs.scale2_vals)
        .arg_ptr(gate_out)
        .arg_ptr(up_ptrs.packed_ptrs)
        .arg_ptr(up_ptrs.scale_ptrs)
        .arg_ptr(up_ptrs.scale2_vals)
        .arg_ptr(up_out)
        .arg_ptr(sorted_token_ids)
        .arg_ptr(sorted_expert_ids)
        .arg_u32(inter)
        .arg_u32(hidden)
        .arg_u32(total_expanded)
        .launch(stream)
}

/// 2026-09-25: SiLU(gate) * up, then the down GEMV, over the sorted rows that
/// `moe_sorted_gate_up` wrote. Output stays in sorted order.
#[allow(clippy::too_many_arguments, dead_code)]
pub(crate) fn moe_sorted_silu_down(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    gate_out: DevicePtr,
    up_out: DevicePtr,
    down_ptrs: &moe::ExpertPtrTable,
    output: DevicePtr,
    sorted_expert_ids: DevicePtr,
    hidden: u32,
    inter: u32,
    total_expanded: u32,
    stream: u64,
) -> Result<()> {
    let grid_x = div_ceil(hidden, 8);
    KernelLaunch::new(gpu, kernel)
        .grid([grid_x, total_expanded, 1])
        .block([128, 1, 1])
        .arg_ptr(gate_out)
        .arg_ptr(up_out)
        .arg_ptr(down_ptrs.packed_ptrs)
        .arg_ptr(down_ptrs.scale_ptrs)
        .arg_ptr(down_ptrs.scale2_vals)
        .arg_ptr(output)
        .arg_ptr(sorted_expert_ids)
        .arg_u32(hidden)
        .arg_u32(inter)
        .arg_u32(total_expanded)
        .launch(stream)
}
