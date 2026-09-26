// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Launchers for the GDN decode kernels (single-token, fused-norm,
//! strided, FP16 h-state, 2-token verify), the fused conv + decode kernel, the
//! register-resident prefill recurrence, and the SSM h-state dtype converters.
//!
//! Owner: model-layers ops (GDN).
//! Invariants: none beyond the types.

#![allow(unused_imports)]

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::layers::moe;
use crate::weight_map::{DenseWeight, Fp8DenseWeight, Fp8Weight, QuantizedWeight};

use super::*;

/// 2026-09-25: FP16 h-state twin of [`gdn_decode_f32_strided_norm`], used when
/// `ssm_h_fp16_enabled()` is set. It adds one argument, `h_seq_stride`: the
/// per-sequence slot pitch of the h-state pool in `__half` elements. The pitch
/// depends on how the pool was sized, not on the head dims, so the caller
/// passes it.
#[allow(clippy::too_many_arguments)]
pub fn gdn_decode_f16_strided_norm(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    h_state: DevicePtr,
    query: DevicePtr,
    key: DevicePtr,
    value: DevicePtr,
    gate: DevicePtr,
    beta: DevicePtr,
    z_gate: DevicePtr,
    norm_weight: DevicePtr,
    output: DevicePtr,
    batch_size: u32,
    num_k_heads: u32,
    num_v_heads: u32,
    k_dim: u32,
    v_dim: u32,
    qk_stride: u32,
    v_stride: u32,
    gb_stride: u32,
    z_stride: u32,
    out_stride: u32,
    h_seq_stride: u64,
    eps: f32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_v_heads, batch_size, 1])
        .block([128, 1, 1])
        .arg_ptr(h_state)
        .arg_ptr(query)
        .arg_ptr(key)
        .arg_ptr(value)
        .arg_ptr(gate)
        .arg_ptr(beta)
        .arg_ptr(z_gate)
        .arg_ptr(norm_weight)
        .arg_ptr(output)
        .arg_u32(batch_size)
        .arg_u32(num_k_heads)
        .arg_u32(num_v_heads)
        .arg_u32(k_dim)
        .arg_u32(v_dim)
        .arg_u32(qk_stride)
        .arg_u32(v_stride)
        .arg_u32(gb_stride)
        .arg_u32(z_stride)
        .arg_u32(out_stride)
        .arg_u64(h_seq_stride)
        .arg_f32(eps)
        .launch(stream)
}

/// 2026-09-25: FP32 -> FP16 conversion of `n` h-state elements
/// (`ssm_h_state_f32_to_f16`, round-to-nearest-even). The kernel is
/// grid-stride, so the clamped grid still covers every element. `src` and
/// `dst` must not alias: an in-place narrowing conversion is a data race.
pub fn ssm_h_state_f32_to_f16(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    src: DevicePtr,
    dst: DevicePtr,
    n: u64,
    stream: u64,
) -> Result<()> {
    const BLOCK: u32 = 256;
    let blocks = div_ceil(n as u32, BLOCK).clamp(1, 4096);
    KernelLaunch::new(gpu, kernel)
        .grid([blocks, 1, 1])
        .block([BLOCK, 1, 1])
        .arg_ptr(src)
        .arg_ptr(dst)
        .arg_u64(n)
        .launch(stream)
}

/// 2026-09-25: FP16 -> FP32 widening of `n` h-state elements
/// (`ssm_h_state_f16_to_f32`). Grid-stride; `src` and `dst` must not alias.
pub fn ssm_h_state_f16_to_f32(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    src: DevicePtr,
    dst: DevicePtr,
    n: u64,
    stream: u64,
) -> Result<()> {
    const BLOCK: u32 = 256;
    let blocks = div_ceil(n as u32, BLOCK).clamp(1, 4096);
    KernelLaunch::new(gpu, kernel)
        .grid([blocks, 1, 1])
        .block([BLOCK, 1, 1])
        .arg_ptr(src)
        .arg_ptr(dst)
        .arg_u64(n)
        .launch(stream)
}

/// 2026-09-25: Single-token GDN decode (`gated_delta_rule_decode`). The
/// `batch_size` h-states are contiguous: `[batch, num_v_heads, k_dim, v_dim]`
/// FP32.
pub fn gdn_decode(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    h_state: DevicePtr,
    query: DevicePtr,
    key: DevicePtr,
    value: DevicePtr,
    gate: DevicePtr,
    beta: DevicePtr,
    output: DevicePtr,
    batch_size: u32,
    num_k_heads: u32,
    num_v_heads: u32,
    k_dim: u32,
    v_dim: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_v_heads, batch_size, 1])
        .block([128, 1, 1])
        .arg_ptr(h_state)
        .arg_ptr(query)
        .arg_ptr(key)
        .arg_ptr(value)
        .arg_ptr(gate)
        .arg_ptr(beta)
        .arg_ptr(output)
        .arg_u32(batch_size)
        .arg_u32(num_k_heads)
        .arg_u32(num_v_heads)
        .arg_u32(k_dim)
        .arg_u32(v_dim)
        .launch(stream)
}

/// 2026-09-25: FP32 GDN decode fused with the gated RMS norm: the kernel
/// normalises each v-head's output in the same CTA and writes BF16, with no
/// FP32 output buffer in global memory.
#[allow(clippy::too_many_arguments)]
pub fn gdn_decode_f32_norm(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    h_state: DevicePtr,
    query: DevicePtr,
    key: DevicePtr,
    value: DevicePtr,
    gate: DevicePtr,
    beta: DevicePtr,
    z_gate: DevicePtr,
    norm_weight: DevicePtr,
    output: DevicePtr,
    batch_size: u32,
    num_k_heads: u32,
    num_v_heads: u32,
    k_dim: u32,
    v_dim: u32,
    eps: f32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_v_heads, batch_size, 1])
        .block([128, 1, 1])
        .arg_ptr(h_state)
        .arg_ptr(query)
        .arg_ptr(key)
        .arg_ptr(value)
        .arg_ptr(gate)
        .arg_ptr(beta)
        .arg_ptr(z_gate)
        .arg_ptr(norm_weight)
        .arg_ptr(output)
        .arg_u32(batch_size)
        .arg_u32(num_k_heads)
        .arg_u32(num_v_heads)
        .arg_u32(k_dim)
        .arg_u32(v_dim)
        .arg_f32(eps)
        .launch(stream)
}

/// 2026-09-25: Token-sequential prefill recurrence with H in registers
/// (`gated_delta_rule_prefill_regresident`). One warp owns one v-column and
/// holds its 128 k-rows (4 per lane); there is no shared-memory H and no block
/// barrier. Requires `k_dim == 128` and `v_dim % 4 == 0` (4 warps per block).
#[allow(clippy::too_many_arguments)]
pub fn gdn_prefill_regresident(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    h_state: DevicePtr,
    query: DevicePtr,
    key: DevicePtr,
    value: DevicePtr,
    gate: DevicePtr,
    beta: DevicePtr,
    output: DevicePtr,
    batch_size: u32,
    seq_len: u32,
    num_k_heads: u32,
    num_v_heads: u32,
    k_dim: u32,
    v_dim: u32,
    qk_stride: u32,
    v_stride: u32,
    gb_stride: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_v_heads, batch_size, v_dim / 4])
        .block([128, 1, 1])
        .arg_ptr(h_state)
        .arg_ptr(query)
        .arg_ptr(key)
        .arg_ptr(value)
        .arg_ptr(gate)
        .arg_ptr(beta)
        .arg_ptr(output)
        .arg_u32(batch_size)
        .arg_u32(seq_len)
        .arg_u32(num_k_heads)
        .arg_u32(num_v_heads)
        .arg_u32(k_dim)
        .arg_u32(v_dim)
        .arg_u32(qk_stride)
        .arg_u32(v_stride)
        .arg_u32(gb_stride)
        .launch(stream)
}

/// 2026-09-25: Conv1d update + L2 norm, the GDN recurrence and the gated RMS
/// norm in one decode launch (`gated_delta_rule_decode_f32_conv_norm`).
///
/// One block per k-head: it owns k-head `kh` and its `head_repeat` v-heads, so
/// it is the only writer of their q/k and v `conv_state` rows. The block is
/// `head_repeat * v_dim` threads, and the kernel also requires
/// `2 * k_dim <= head_repeat * v_dim` and `k_dim == v_dim`.
#[allow(clippy::too_many_arguments)]
pub fn gdn_decode_f32_conv_norm(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    h_state: DevicePtr,
    conv_state: DevicePtr,
    new_input: DevicePtr,
    conv_weight: DevicePtr,
    gate: DevicePtr,
    beta: DevicePtr,
    z_gate: DevicePtr,
    norm_weight: DevicePtr,
    output: DevicePtr,
    batch_size: u32,
    num_k_heads: u32,
    num_v_heads: u32,
    k_dim: u32,
    v_dim: u32,
    conv_dim: u32,
    d_conv: u32,
    l2_eps: f32,
    eps: f32,
    stream: u64,
) -> Result<()> {
    let head_repeat = num_v_heads / num_k_heads;
    KernelLaunch::new(gpu, kernel)
        .grid([num_k_heads, batch_size, 1])
        .block([head_repeat * v_dim, 1, 1])
        .arg_ptr(h_state)
        .arg_ptr(conv_state)
        .arg_ptr(new_input)
        .arg_ptr(conv_weight)
        .arg_ptr(DevicePtr::NULL)
        .arg_ptr(gate)
        .arg_ptr(beta)
        .arg_ptr(z_gate)
        .arg_ptr(norm_weight)
        .arg_ptr(output)
        .arg_u32(batch_size)
        .arg_u32(num_k_heads)
        .arg_u32(num_v_heads)
        .arg_u32(k_dim)
        .arg_u32(v_dim)
        .arg_u32(conv_dim)
        .arg_u32(d_conv)
        .arg_f32(l2_eps)
        .arg_f32(eps)
        .launch(stream)
}

/// 2026-09-25: Strided FP32 GDN decode for several sequences in one launch:
/// Q/K/V, gate/beta and the output are rows `qk_stride`, `v_stride`,
/// `gb_stride` and `out_stride` apart per sequence.
#[allow(clippy::too_many_arguments)]
pub fn gdn_decode_f32_strided(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    h_state: DevicePtr,
    query: DevicePtr,
    key: DevicePtr,
    value: DevicePtr,
    gate: DevicePtr,
    beta: DevicePtr,
    output: DevicePtr,
    batch_size: u32,
    num_k_heads: u32,
    num_v_heads: u32,
    k_dim: u32,
    v_dim: u32,
    qk_stride: u32,
    v_stride: u32,
    gb_stride: u32,
    out_stride: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_v_heads, batch_size, 1])
        .block([128, 1, 1])
        .arg_ptr(h_state)
        .arg_ptr(query)
        .arg_ptr(key)
        .arg_ptr(value)
        .arg_ptr(gate)
        .arg_ptr(beta)
        .arg_ptr(output)
        .arg_u32(batch_size)
        .arg_u32(num_k_heads)
        .arg_u32(num_v_heads)
        .arg_u32(k_dim)
        .arg_u32(v_dim)
        .arg_u32(qk_stride)
        .arg_u32(v_stride)
        .arg_u32(gb_stride)
        .arg_u32(out_stride)
        .launch(stream)
}

/// 2026-09-25: [`gdn_decode_f32_strided`] fused with the gated RMS norm: it
/// writes the BF16 post-norm output directly, `out_stride` apart per sequence.
#[allow(clippy::too_many_arguments)]
pub fn gdn_decode_f32_strided_norm(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    h_state: DevicePtr,
    query: DevicePtr,
    key: DevicePtr,
    value: DevicePtr,
    gate: DevicePtr,
    beta: DevicePtr,
    z_gate: DevicePtr,
    norm_weight: DevicePtr,
    output: DevicePtr,
    batch_size: u32,
    num_k_heads: u32,
    num_v_heads: u32,
    k_dim: u32,
    v_dim: u32,
    qk_stride: u32,
    v_stride: u32,
    gb_stride: u32,
    z_stride: u32,
    out_stride: u32,
    eps: f32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_v_heads, batch_size, 1])
        .block([128, 1, 1])
        .arg_ptr(h_state)
        .arg_ptr(query)
        .arg_ptr(key)
        .arg_ptr(value)
        .arg_ptr(gate)
        .arg_ptr(beta)
        .arg_ptr(z_gate)
        .arg_ptr(norm_weight)
        .arg_ptr(output)
        .arg_u32(batch_size)
        .arg_u32(num_k_heads)
        .arg_u32(num_v_heads)
        .arg_u32(k_dim)
        .arg_u32(v_dim)
        .arg_u32(qk_stride)
        .arg_u32(v_stride)
        .arg_u32(gb_stride)
        .arg_u32(z_stride)
        .arg_u32(out_stride)
        .arg_f32(eps)
        .launch(stream)
}

/// 2026-09-25: Two tokens through the GDN recurrence in one launch
/// (`gated_delta_rule_chunk2`). The state after the first token is written to
/// `h_state_intermediate`, for rollback when the draft is rejected. The
/// strides are in elements, not bytes.
#[allow(clippy::too_many_arguments)]
pub fn gdn_decode_chunk2(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    h_state: DevicePtr,
    query: DevicePtr,
    key: DevicePtr,
    value: DevicePtr,
    gate: DevicePtr,
    beta: DevicePtr,
    output: DevicePtr,
    h_state_intermediate: DevicePtr,
    batch_size: u32,
    num_k_heads: u32,
    num_v_heads: u32,
    k_dim: u32,
    v_dim: u32,
    qk_stride: u32,
    v_stride: u32,
    gb_stride: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_v_heads, batch_size, 1])
        .block([128, 1, 1])
        .arg_ptr(h_state)
        .arg_ptr(query)
        .arg_ptr(key)
        .arg_ptr(value)
        .arg_ptr(gate)
        .arg_ptr(beta)
        .arg_ptr(output)
        .arg_ptr(h_state_intermediate)
        .arg_u32(batch_size)
        .arg_u32(num_k_heads)
        .arg_u32(num_v_heads)
        .arg_u32(k_dim)
        .arg_u32(v_dim)
        .arg_u32(qk_stride)
        .arg_u32(v_stride)
        .arg_u32(gb_stride)
        .launch(stream)
}
