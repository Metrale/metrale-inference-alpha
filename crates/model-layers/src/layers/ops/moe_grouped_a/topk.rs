// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Launchers for the batched MoE routers (softmax, sigmoid, sqrt-softplus): one block per token, each choosing that token's experts before the grouped GEMMs run.
//!
//! Owner: model-layers ops.
//! Invariants: none beyond the types.

#![allow(unused_imports)]

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::weight_map::{DenseWeight, Fp8DenseWeight, Fp8Weight, QuantizedWeight};

use super::super::*;

/// 2026-09-25: The top-k softmax router for `num_tokens` tokens.
#[allow(clippy::too_many_arguments)]
pub fn moe_topk_softmax_batched(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    gate_logits: DevicePtr,
    expert_indices: DevicePtr,
    expert_weights: DevicePtr,
    num_experts: u32,
    top_k: u32,
    normalize: bool,
    num_tokens: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(gate_logits)
        .arg_ptr(expert_indices)
        .arg_ptr(expert_weights)
        .arg_u32(num_experts)
        .arg_u32(top_k)
        .arg_u32(if normalize { 1 } else { 0 })
        .launch(stream)
}

/// 2026-09-25: The sigmoid router with a selection-only bias for `num_tokens` tokens.
#[allow(clippy::too_many_arguments)]
pub fn moe_topk_sigmoid_batched(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    gate_logits: DevicePtr,
    bias: DevicePtr,
    expert_indices: DevicePtr,
    expert_weights: DevicePtr,
    num_experts: u32,
    top_k: u32,
    normalize: bool,
    scaling_factor: f32,
    num_tokens: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(gate_logits)
        .arg_ptr(bias)
        .arg_ptr(expert_indices)
        .arg_ptr(expert_weights)
        .arg_u32(num_experts)
        .arg_u32(top_k)
        .arg_u32(if normalize { 1 } else { 0 })
        .arg_f32(scaling_factor)
        .launch(stream)
}

/// 2026-09-25: Same arguments as [`moe_topk_sigmoid_batched`], scoring experts with
/// `sqrt(log(1 + exp(logit)))`, the score of the single-token `moe_topk_sqrtsoftplus`.
#[allow(clippy::too_many_arguments)]
pub fn moe_topk_sqrtsoftplus_batched(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    gate_logits: DevicePtr,
    bias: DevicePtr,
    expert_indices: DevicePtr,
    expert_weights: DevicePtr,
    num_experts: u32,
    top_k: u32,
    normalize: bool,
    scaling_factor: f32,
    num_tokens: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(gate_logits)
        .arg_ptr(bias)
        .arg_ptr(expert_indices)
        .arg_ptr(expert_weights)
        .arg_u32(num_experts)
        .arg_u32(top_k)
        .arg_u32(if normalize { 1 } else { 0 })
        .arg_f32(scaling_factor)
        .launch(stream)
}
