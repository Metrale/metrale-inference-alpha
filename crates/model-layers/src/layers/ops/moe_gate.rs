// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Launchers for the MoE routers (top-k softmax, softmax with correction bias, sigmoid, sqrt-softplus, hash) and the zero-expert blend, plus the sigmoid router's compile-time bounds.
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

/// 2026-09-25: Largest `top_k` the sigmoid routing kernel holds: `#define MAX_TOP_K` in
/// `kernels/gb10/common/moe_topk_sigmoid.cu`, which sizes its shared `s_top_vals` /
/// `s_top_idxs`. The kernel clamps a larger `top_k` to it, and `NemotronMoeLayer` construction
/// refuses one (`nemotron_moe.rs`). `crates/model-engine/tests/moe_topk_sigmoid_bounds.rs` pins
/// this mirror to the define and fails if a directory other than `common/` holds a copy of the
/// kernel.
pub const MOE_TOPK_SIGMOID_MAX_TOP_K: usize = 32;

/// 2026-09-25: Largest `num_experts` the sigmoid routing kernel holds, from `#define MAX_EXPERTS`
/// in the same file. Beyond it the kernel considers only the first `MAX_EXPERTS` experts
/// (`actual_n` is a `min`).
pub const MOE_TOPK_SIGMOID_MAX_EXPERTS: usize = 512;

/// 2026-09-25: Top-k experts of one token's gate logits, with softmax weights, in one block.
/// The kernel handle decides the logit type (`moe_topk_softmax` BF16, `moe_topk_softmax_f32`).
pub fn moe_topk_softmax(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    gate_logits: DevicePtr,
    expert_indices: DevicePtr,
    expert_weights: DevicePtr,
    num_experts: u32,
    top_k: u32,
    normalize: bool,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([1, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(gate_logits)
        .arg_ptr(expert_indices)
        .arg_ptr(expert_weights)
        .arg_u32(num_experts)
        .arg_u32(top_k)
        .arg_u32(if normalize { 1 } else { 0 })
        .launch(stream)
}

/// 2026-09-25: Softmax router with a correction bias and the zero-expert fold, one token. Logits
/// cover `num_logits` experts, of which ids `>= num_routed` are identity experts: a selected one
/// adds its weight to `zero_accum` and its slot becomes expert 0 with weight 0.
#[allow(clippy::too_many_arguments)]
pub fn moe_topk_softmax_bias(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    gate_logits: DevicePtr,
    bias: DevicePtr,
    expert_indices: DevicePtr,
    expert_weights: DevicePtr,
    zero_accum: DevicePtr,
    num_logits: u32,
    num_routed: u32,
    top_k: u32,
    normalize: bool,
    scaling_factor: f32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([1, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(gate_logits)
        .arg_ptr(bias)
        .arg_ptr(expert_indices)
        .arg_ptr(expert_weights)
        .arg_ptr(zero_accum)
        .arg_u32(num_logits)
        .arg_u32(num_routed)
        .arg_u32(top_k)
        .arg_u32(normalize as u32)
        .arg_f32(scaling_factor)
        .launch(stream)
}

/// 2026-09-25: [`moe_topk_softmax_bias`] for `n` tokens, one block per token.
#[allow(clippy::too_many_arguments)]
pub fn moe_topk_softmax_bias_batched(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    gate_logits: DevicePtr,
    bias: DevicePtr,
    expert_indices: DevicePtr,
    expert_weights: DevicePtr,
    zero_accum: DevicePtr,
    num_logits: u32,
    num_routed: u32,
    top_k: u32,
    normalize: bool,
    scaling_factor: f32,
    n: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([n, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(gate_logits)
        .arg_ptr(bias)
        .arg_ptr(expert_indices)
        .arg_ptr(expert_weights)
        .arg_ptr(zero_accum)
        .arg_u32(num_logits)
        .arg_u32(num_routed)
        .arg_u32(top_k)
        .arg_u32(normalize as u32)
        .arg_f32(scaling_factor)
        .launch(stream)
}

/// 2026-09-25: `out[t, :] += zero_accum[t] * x[t, :]`, the identity-expert blend.
pub fn moe_zero_expert_add(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    out: DevicePtr,
    x: DevicePtr,
    zero_accum: DevicePtr,
    n: u32,
    h: u32,
    stream: u64,
) -> Result<()> {
    use metrale_gpu_runtime::kernel_args::div_ceil;
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n * h, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(out)
        .arg_ptr(x)
        .arg_ptr(zero_accum)
        .arg_u32(n)
        .arg_u32(h)
        .launch(stream)
}

/// 2026-09-25: Sigmoid router, one token. The bias is added to the scores for selection only;
/// the weights come from the pre-bias sigmoid scores. `top_k` and `num_experts` are bounded by
/// [`MOE_TOPK_SIGMOID_MAX_TOP_K`] and [`MOE_TOPK_SIGMOID_MAX_EXPERTS`].
pub fn moe_topk_sigmoid(
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
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([1, 1, 1])
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

/// 2026-09-25: Sqrt-softplus router (`sqrt(log(1 + exp(logit)))`), one token. As in
/// [`moe_topk_sigmoid`], the bias steers selection only and the weights come from the pre-bias
/// scores.
#[allow(clippy::too_many_arguments)]
pub fn moe_topk_sqrtsoftplus(
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
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([1, 1, 1])
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

/// 2026-09-25: Hash router, one token: the experts are the static `tid2eid[token_id]` row, and
/// the gate's sqrt-softplus scores weight them. The token id is read on the device from
/// `token_id_ptr`.
#[allow(clippy::too_many_arguments)]
pub fn moe_hash_route(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    gate_logits: DevicePtr,
    tid2eid: DevicePtr,
    token_id_ptr: DevicePtr,
    expert_indices: DevicePtr,
    expert_weights: DevicePtr,
    num_experts: u32,
    top_k: u32,
    normalize: bool,
    scaling_factor: f32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([1, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(gate_logits)
        .arg_ptr(tid2eid)
        .arg_ptr(token_id_ptr)
        .arg_ptr(expert_indices)
        .arg_ptr(expert_weights)
        .arg_u32(num_experts)
        .arg_u32(top_k)
        .arg_u32(if normalize { 1 } else { 0 })
        .arg_f32(scaling_factor)
        .launch(stream)
}

/// 2026-09-25: [`moe_hash_route`] for `n` tokens, one block per token, reading `token_ids[n]`.
#[allow(clippy::too_many_arguments)]
pub fn moe_hash_route_batched(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    gate_logits: DevicePtr,
    tid2eid: DevicePtr,
    token_ids: DevicePtr,
    expert_indices: DevicePtr,
    expert_weights: DevicePtr,
    num_experts: u32,
    top_k: u32,
    normalize: bool,
    scaling_factor: f32,
    n: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([n, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(gate_logits)
        .arg_ptr(tid2eid)
        .arg_ptr(token_ids)
        .arg_ptr(expert_indices)
        .arg_ptr(expert_weights)
        .arg_u32(num_experts)
        .arg_u32(top_k)
        .arg_u32(if normalize { 1 } else { 0 })
        .arg_f32(scaling_factor)
        .launch(stream)
}
