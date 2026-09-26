// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Launchers for the two C=4 routed MoE decode kernels of `moe_decode_atomic_c4.cu`: `moe_decode_atomic_c4_silu_down_accum` and `moe_decode_atomic_c4_finalize`.
//!
//! Owner: model-layers ops.
//! Invariants: none beyond the types.
//!
//! The SiLU+down kernel adds each routed expert's weighted output into `routed_accum` with FP32
//! `atomicAdd`. The multi-sequence SSM decode takes this path only for n == 4 with
//! `METRALE_MOE_ATOMIC_C4_DECODE=1` (`qwen3_ssm/trait_decode_multi_seq.rs`).

#![allow(unused_imports)]

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::weight_map::QuantizedWeight;

#[allow(clippy::too_many_arguments)]
pub fn moe_decode_atomic_c4_silu_down_accum(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    gate_out: DevicePtr,
    up_out: DevicePtr,
    packed_ptrs: DevicePtr,
    scale_ptrs: DevicePtr,
    scale2_vals: DevicePtr,
    expert_indices: DevicePtr,
    expert_weights: DevicePtr,
    routed_accum: DevicePtr,
    sh_gate_in: DevicePtr,
    sh_up_in: DevicePtr,
    sh_down: &QuantizedWeight,
    sh_down_out: DevicePtr,
    hidden: u32,
    inter: u32,
    top_k: u32,
    num_tokens: u32,
    stream: u64,
) -> Result<()> {
    let smem_bytes = (inter as usize * std::mem::size_of::<f32>()) as u32;
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(hidden, 8), num_tokens * (top_k + 1), 1])
        .block([128, 1, 1])
        .shared_mem(smem_bytes)
        .arg_ptr(gate_out)
        .arg_ptr(up_out)
        .arg_ptr(packed_ptrs)
        .arg_ptr(scale_ptrs)
        .arg_ptr(scale2_vals)
        .arg_ptr(expert_indices)
        .arg_ptr(expert_weights)
        .arg_ptr(routed_accum)
        .arg_ptr(sh_gate_in)
        .arg_ptr(sh_up_in)
        .arg_ptr(sh_down.weight)
        .arg_ptr(sh_down.weight_scale)
        .arg_f32(sh_down.weight_scale_2)
        .arg_ptr(sh_down_out)
        .arg_u32(hidden)
        .arg_u32(inter)
        .arg_u32(top_k)
        .arg_u32(num_tokens)
        .launch(stream)
}

#[allow(clippy::too_many_arguments)]
pub fn moe_decode_atomic_c4_finalize(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    output: DevicePtr,
    routed_accum: DevicePtr,
    shared_out: DevicePtr,
    input: DevicePtr,
    gate_weight: DevicePtr,
    hidden: u32,
    num_tokens: u32,
    include_shared: bool,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(hidden, 256), num_tokens, 1])
        .block([256, 1, 1])
        .arg_ptr(output)
        .arg_ptr(routed_accum)
        .arg_ptr(shared_out)
        .arg_ptr(input)
        .arg_ptr(gate_weight)
        .arg_u32(hidden)
        .arg_u32(num_tokens)
        .arg_u32(if include_shared { 1 } else { 0 })
        .launch(stream)
}
