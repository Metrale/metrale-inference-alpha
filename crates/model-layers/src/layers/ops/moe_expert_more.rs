// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Launchers for the MoE blend (`moe_weighted_sum_blend`) and for the two- and three-token routed+shared expert kernels (`moe_shared_expert_fused_batch2.cu`, `_batch3.cu`).
//!
//! Owner: model-layers ops.
//! Invariants: none beyond the types.
//!
//! Row layout of the batch kernels, for `T` tokens: `blockIdx.y` in `[0, T*top_k)` is a routed
//! expert (token `y / top_k`, slot `y % top_k`), and the last `T` values of `blockIdx.y` are the
//! shared expert of token `y - T*top_k`. Routed outputs are rows `token * top_k + slot`; shared
//! outputs are rows `token`.

#![allow(unused_imports)]

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::layers::moe;
use crate::weight_map::{DenseWeight, Fp8DenseWeight, Fp8Weight, QuantizedWeight};

use super::*;

/// 2026-09-25: The MoE combine for one token:
/// `output[j] = sum_e weights[e] * expert_out[e, j] + sigmoid(dot(input, gate_weight)) * shared_out[j]`.
/// Every block computes the gate dot product itself. A null `gate_weight` means an ungated
/// shared expert, blended at weight 1.
#[allow(clippy::too_many_arguments)]
pub fn moe_weighted_sum_blend(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    output: DevicePtr,
    expert_out: DevicePtr,
    expert_weights: DevicePtr,
    shared_out: DevicePtr,
    input: DevicePtr,
    gate_weight: DevicePtr,
    hidden: u32,
    top_k: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(hidden, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(output)
        .arg_ptr(expert_out)
        .arg_ptr(expert_weights)
        .arg_ptr(shared_out)
        .arg_ptr(input)
        .arg_ptr(gate_weight)
        .arg_u32(hidden)
        .arg_u32(top_k)
        .arg_u32(k)
        .launch(stream)
}

/// 2026-09-25: Gate and up GEMVs of the routed and shared experts for two tokens in one launch
/// (row layout in the module doc). The shared expert uses direct weight pointers, the routed
/// experts the pointer tables. A `block_size` of 256 selects the kernel's wide decomposition
/// (`blockDim.x == 256`); any other width runs the 128-thread one.
#[allow(clippy::too_many_arguments)]
pub fn moe_expert_gate_up_shared_batch2(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    gate_packed_ptrs: DevicePtr,
    gate_scale_ptrs: DevicePtr,
    gate_scale2_vals: DevicePtr,
    gate_out: DevicePtr,
    up_packed_ptrs: DevicePtr,
    up_scale_ptrs: DevicePtr,
    up_scale2_vals: DevicePtr,
    up_out: DevicePtr,
    expert_indices: DevicePtr,
    sh_gate: &QuantizedWeight,
    sh_gate_out: DevicePtr,
    sh_up: &QuantizedWeight,
    sh_up_out: DevicePtr,
    n: u32,
    k: u32,
    top_k: u32,
    block_size: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 8), 2 * (top_k + 1), 2])
        .block([block_size, 1, 1])
        .arg_ptr(input)
        .arg_ptr(gate_packed_ptrs)
        .arg_ptr(gate_scale_ptrs)
        .arg_ptr(gate_scale2_vals)
        .arg_ptr(gate_out)
        .arg_ptr(up_packed_ptrs)
        .arg_ptr(up_scale_ptrs)
        .arg_ptr(up_scale2_vals)
        .arg_ptr(up_out)
        .arg_ptr(expert_indices)
        .arg_ptr(sh_gate.weight)
        .arg_ptr(sh_gate.weight_scale)
        .arg_f32(sh_gate.weight_scale_2)
        .arg_ptr(sh_gate_out)
        .arg_ptr(sh_up.weight)
        .arg_ptr(sh_up.weight_scale)
        .arg_f32(sh_up.weight_scale_2)
        .arg_ptr(sh_up_out)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(top_k)
        .launch(stream)
}

/// 2026-09-25: SiLU+down GEMVs of the routed and shared experts for two tokens in one launch.
/// `block_size` as in [`moe_expert_gate_up_shared_batch2`].
#[allow(clippy::too_many_arguments)]
pub fn moe_expert_silu_down_shared_batch2(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    gate_out: DevicePtr,
    up_out: DevicePtr,
    packed_ptrs: DevicePtr,
    scale_ptrs: DevicePtr,
    scale2_vals: DevicePtr,
    output: DevicePtr,
    expert_indices: DevicePtr,
    sh_gate_in: DevicePtr,
    sh_up_in: DevicePtr,
    sh_down: &QuantizedWeight,
    sh_down_out: DevicePtr,
    n: u32,
    k: u32,
    top_k: u32,
    block_size: u32,
    stream: u64,
) -> Result<()> {
    // 2026-09-25: The kernel's `s_act` is dynamic shared memory of K floats.
    let smem_bytes = (k as usize * std::mem::size_of::<f32>()) as u32;
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 8), 2 * (top_k + 1), 1])
        .block([block_size, 1, 1])
        .shared_mem(smem_bytes)
        .arg_ptr(gate_out)
        .arg_ptr(up_out)
        .arg_ptr(packed_ptrs)
        .arg_ptr(scale_ptrs)
        .arg_ptr(scale2_vals)
        .arg_ptr(output)
        .arg_ptr(expert_indices)
        .arg_ptr(sh_gate_in)
        .arg_ptr(sh_up_in)
        .arg_ptr(sh_down.weight)
        .arg_ptr(sh_down.weight_scale)
        .arg_f32(sh_down.weight_scale_2)
        .arg_ptr(sh_down_out)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(top_k)
        .launch(stream)
}

/// 2026-09-25: [`moe_weighted_sum_blend`] for two tokens; `blockIdx.y` is the token.
#[allow(clippy::too_many_arguments)]
pub fn moe_weighted_sum_blend_batch2(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    output: DevicePtr,
    expert_out: DevicePtr,
    expert_weights: DevicePtr,
    shared_out: DevicePtr,
    input: DevicePtr,
    gate_weight: DevicePtr,
    hidden: u32,
    top_k: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(hidden, 256), 2, 1])
        .block([256, 1, 1])
        .arg_ptr(output)
        .arg_ptr(expert_out)
        .arg_ptr(expert_weights)
        .arg_ptr(shared_out)
        .arg_ptr(input)
        .arg_ptr(gate_weight)
        .arg_u32(hidden)
        .arg_u32(top_k)
        .arg_u32(k)
        .launch(stream)
}

/// 2026-09-25: [`moe_expert_gate_up_shared_batch2`] for three tokens.
#[allow(clippy::too_many_arguments)]
pub fn moe_expert_gate_up_shared_batch3(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    gate_packed_ptrs: DevicePtr,
    gate_scale_ptrs: DevicePtr,
    gate_scale2_vals: DevicePtr,
    gate_out: DevicePtr,
    up_packed_ptrs: DevicePtr,
    up_scale_ptrs: DevicePtr,
    up_scale2_vals: DevicePtr,
    up_out: DevicePtr,
    expert_indices: DevicePtr,
    sh_gate: &QuantizedWeight,
    sh_gate_out: DevicePtr,
    sh_up: &QuantizedWeight,
    sh_up_out: DevicePtr,
    n: u32,
    k: u32,
    top_k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 8), 3 * (top_k + 1), 2])
        .block([128, 1, 1])
        .arg_ptr(input)
        .arg_ptr(gate_packed_ptrs)
        .arg_ptr(gate_scale_ptrs)
        .arg_ptr(gate_scale2_vals)
        .arg_ptr(gate_out)
        .arg_ptr(up_packed_ptrs)
        .arg_ptr(up_scale_ptrs)
        .arg_ptr(up_scale2_vals)
        .arg_ptr(up_out)
        .arg_ptr(expert_indices)
        .arg_ptr(sh_gate.weight)
        .arg_ptr(sh_gate.weight_scale)
        .arg_f32(sh_gate.weight_scale_2)
        .arg_ptr(sh_gate_out)
        .arg_ptr(sh_up.weight)
        .arg_ptr(sh_up.weight_scale)
        .arg_f32(sh_up.weight_scale_2)
        .arg_ptr(sh_up_out)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(top_k)
        .launch(stream)
}

/// 2026-09-25: [`moe_expert_silu_down_shared_batch2`] for three tokens.
#[allow(clippy::too_many_arguments)]
pub fn moe_expert_silu_down_shared_batch3(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    gate_out: DevicePtr,
    up_out: DevicePtr,
    packed_ptrs: DevicePtr,
    scale_ptrs: DevicePtr,
    scale2_vals: DevicePtr,
    output: DevicePtr,
    expert_indices: DevicePtr,
    sh_gate_in: DevicePtr,
    sh_up_in: DevicePtr,
    sh_down: &QuantizedWeight,
    sh_down_out: DevicePtr,
    n: u32,
    k: u32,
    top_k: u32,
    stream: u64,
) -> Result<()> {
    // 2026-09-25: The kernel's `s_act` is dynamic shared memory of K floats.
    let smem_bytes = (k as usize * std::mem::size_of::<f32>()) as u32;
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 8), 3 * (top_k + 1), 1])
        .block([128, 1, 1])
        .shared_mem(smem_bytes)
        .arg_ptr(gate_out)
        .arg_ptr(up_out)
        .arg_ptr(packed_ptrs)
        .arg_ptr(scale_ptrs)
        .arg_ptr(scale2_vals)
        .arg_ptr(output)
        .arg_ptr(expert_indices)
        .arg_ptr(sh_gate_in)
        .arg_ptr(sh_up_in)
        .arg_ptr(sh_down.weight)
        .arg_ptr(sh_down.weight_scale)
        .arg_f32(sh_down.weight_scale_2)
        .arg_ptr(sh_down_out)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(top_k)
        .launch(stream)
}

/// 2026-09-25: [`moe_weighted_sum_blend`] for three tokens.
#[allow(clippy::too_many_arguments)]
pub fn moe_weighted_sum_blend_batch3(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    output: DevicePtr,
    expert_out: DevicePtr,
    expert_weights: DevicePtr,
    shared_out: DevicePtr,
    input: DevicePtr,
    gate_weight: DevicePtr,
    hidden: u32,
    top_k: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(hidden, 256), 3, 1])
        .block([256, 1, 1])
        .arg_ptr(output)
        .arg_ptr(expert_out)
        .arg_ptr(expert_weights)
        .arg_ptr(shared_out)
        .arg_ptr(input)
        .arg_ptr(gate_weight)
        .arg_u32(hidden)
        .arg_u32(top_k)
        .arg_u32(k)
        .launch(stream)
}
