// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Launchers for the token-major MoE prefill kernels
//! (`kernels/gb10/common/moe_prefill.cu`), the fused dual and SiLU-input W4A16
//! and W8A16 GEMVs, and the device-gated sigmoid blend.
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

/// 2026-09-25: Gate and up GEMVs for `num_tokens` tokens, routed and shared
/// experts in one launch. Row `y < num_tokens * top_k` is routed (token
/// `y / top_k`, expert `expert_indices[y]`) and writes `gate_out`/`up_out`
/// (`[num_tokens * top_k, n]` BF16); the remaining `num_tokens` rows run the
/// shared expert and write `sh_gate_out`/`sh_up_out` (`[num_tokens, n]` BF16).
/// `blockIdx.z` picks gate (0) or up (1). `input` is `[num_tokens, k]` BF16.
#[allow(clippy::too_many_arguments)]
pub fn moe_expert_gate_up_shared_prefill(
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
    num_tokens: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 8), num_tokens * (top_k + 1), 2])
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
        .arg_u32(num_tokens)
        .launch(stream)
}

/// 2026-09-25: SiLU(gate) * up, then the down GEMV, for the routed and shared
/// rows of [`moe_expert_gate_up_shared_prefill`], with the same row split.
/// Routed rows read `gate_out`/`up_out` and write `output`
/// (`[num_tokens * top_k, n]` BF16); shared rows read `sh_gate_in`/`sh_up_in`
/// and write `sh_down_out` (`[num_tokens, n]` BF16).
#[allow(clippy::too_many_arguments)]
pub fn moe_expert_silu_down_shared_prefill(
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
    num_tokens: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 8), num_tokens * (top_k + 1), 1])
        .block([128, 1, 1])
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
        .arg_u32(num_tokens)
        .launch(stream)
}

/// 2026-09-25: Per token `t`: `output[t] = sum_e expert_weights[t*top_k+e] *
/// expert_out[t*top_k+e] + sigmoid(dot(input[t], gate_weight)) * shared_out[t]`.
/// `expert_out` is `[num_tokens * top_k, hidden]` BF16, `expert_weights`
/// `[num_tokens * top_k]` f32, `input` `[num_tokens, k]` BF16 and `gate_weight`
/// the shared-expert gate `[1, k]` BF16.
#[allow(clippy::too_many_arguments)]
pub fn moe_weighted_sum_blend_prefill(
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
    num_tokens: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(hidden, 256), num_tokens, 1])
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
        .arg_u32(num_tokens)
        .launch(stream)
}

/// 2026-09-25: Two W4A16 GEMVs over the same BF16 input in one launch;
/// `blockIdx.z` picks `weight1` or `weight2`. Both projections have `n` outputs.
#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemv_dual(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight1: &QuantizedWeight,
    output1: DevicePtr,
    weight2: &QuantizedWeight,
    output2: DevicePtr,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([w4a16_gemv_grid_x(n), 1, 2])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight1.weight)
        .arg_ptr(weight1.weight_scale)
        .arg_f32(weight1.weight_scale_2)
        .arg_ptr(output1)
        .arg_ptr(weight2.weight)
        .arg_ptr(weight2.weight_scale)
        .arg_f32(weight2.weight_scale_2)
        .arg_ptr(output2)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// 2026-09-25: One warp per output variant of [`w4a16_gemv_dual`], 8 outputs per
/// block. It shares the per-lane partial `w4a16_dual_partial` with the base
/// kernel (`w4a16_gemv_fused.cu`). Callers pick it through [`use_gemv_sw`] with
/// `ModelLevers::gemv_sw`, which is on unless `METRALE_NO_GEMV_SW=1`.
#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemv_dual_sw(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight1: &QuantizedWeight,
    output1: DevicePtr,
    weight2: &QuantizedWeight,
    output2: DevicePtr,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([w4a16_gemv_sw_grid_x(n), 1, 2])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight1.weight)
        .arg_ptr(weight1.weight_scale)
        .arg_f32(weight1.weight_scale_2)
        .arg_ptr(output1)
        .arg_ptr(weight2.weight)
        .arg_ptr(weight2.weight_scale)
        .arg_f32(weight2.weight_scale_2)
        .arg_ptr(output2)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// 2026-09-25: One warp per output variant of [`w4a16_gemv_silu_input`], 8
/// outputs per block, picked the same way as [`w4a16_gemv_dual_sw`].
#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemv_silu_input_sw(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    gate_out: DevicePtr,
    up_out: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([w4a16_gemv_sw_grid_x(n), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(gate_out)
        .arg_ptr(up_out)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_f32(weight.weight_scale_2)
        .arg_ptr(output)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// 2026-09-25: W4A16 GEMV whose activation is `silu(gate_out) * up_out`,
/// computed per element from the two `[k]` BF16 vectors inside the kernel.
#[allow(clippy::too_many_arguments)]
pub fn w4a16_gemv_silu_input(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    gate_out: DevicePtr,
    up_out: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(gate_out)
        .arg_ptr(up_out)
        .arg_ptr(weight.weight)
        .arg_ptr(weight.weight_scale)
        .arg_f32(weight.weight_scale_2)
        .arg_ptr(output)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// 2026-09-25: Two FP8 E4M3 GEMVs over the same BF16 input in one launch;
/// `blockIdx.z` picks projection 1 or 2. Both have `n` outputs. Each weight is
/// `[n, k]` bytes with an FP32 `[n/128, k/128]` block-scale buffer.
#[allow(clippy::too_many_arguments)]
pub fn w8a16_gemv_dual(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight1: DevicePtr,
    row_scale1: DevicePtr,
    output1: DevicePtr,
    weight2: DevicePtr,
    row_scale2: DevicePtr,
    output2: DevicePtr,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 2])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight1)
        .arg_ptr(row_scale1)
        .arg_ptr(output1)
        .arg_ptr(weight2)
        .arg_ptr(row_scale2)
        .arg_ptr(output2)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// 2026-09-25: FP8 E4M3 GEMV whose activation is `silu(gate_out) * up_out`,
/// computed per element from the two `[k]` BF16 vectors inside the kernel.
/// `block_scale` is FP32 `[n/128, k/128]`.
#[allow(clippy::too_many_arguments)]
pub fn w8a16_gemv_silu_input(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    gate_out: DevicePtr,
    up_out: DevicePtr,
    weight: DevicePtr,
    block_scale: DevicePtr,
    output: DevicePtr,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n, 4), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(gate_out)
        .arg_ptr(up_out)
        .arg_ptr(weight)
        .arg_ptr(block_scale)
        .arg_ptr(output)
        .arg_u32(n)
        .arg_u32(k)
        .launch(stream)
}

/// 2026-09-25: `output[i] += sigmoid(*gate_ptr) * src[i]`, where `gate_ptr` is
/// one BF16 scalar in device memory (`bf16_sigmoid_blend_device` in
/// `kernels/gb10/common/residual_add.cu`).
pub fn sigmoid_blend_device(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    output: DevicePtr,
    src: DevicePtr,
    gate_ptr: DevicePtr,
    num_elements: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(num_elements, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(output)
        .arg_ptr(src)
        .arg_ptr(gate_ptr)
        .arg_u32(num_elements)
        .launch(stream)
}
