// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Launchers for the exact-verify `_snap` kernels: the fused-norm
//! GDN decodes with an inline per-token h-state snapshot, and the FP32-output
//! fused verify conv.
//!
//! The kernels are optional (`try_kernel`; only the qwen3.6-27b/nvfp4 tree
//! compiles them). Where a handle is 0, the caller runs the parent kernel and
//! copies the snapshots with `copy_d2d_async`.
//!
//! Owner: model-layers ops (GDN).
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::weight_map::DenseWeight;

/// 2026-09-25: [`super::gdn_decode_f32_norm`] plus an inline h-state snapshot.
///
/// `h_inter` receives the updated H after the state-norm clamp, the same
/// values left in `h_state`; NULL skips the snapshot. Same grid, block and
/// argument order as the parent, with `h_inter` appended.
#[allow(clippy::too_many_arguments)]
pub fn gdn_decode_f32_norm_snap(
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
    h_inter: DevicePtr,
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
        .arg_ptr(h_inter)
        .launch(stream)
}

/// 2026-09-25: [`super::gdn_decode_f32_strided_norm`] plus an inline h-state
/// snapshot, for the batched-verify arm over `batch_size` sequences.
///
/// `h_inter` is the snapshot base for this token position, and sequences'
/// snapshots are `h_inter_seq_stride` FP32 elements apart. The caller measures
/// that stride from the pool's pointers; it does not follow from the head
/// dims. NULL skips the snapshot.
#[allow(clippy::too_many_arguments)]
pub fn gdn_decode_f32_strided_norm_snap(
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
    h_inter: DevicePtr,
    h_inter_seq_stride: u64,
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
        .arg_ptr(h_inter)
        .arg_u64(h_inter_seq_stride)
        .launch(stream)
}

/// 2026-09-25: FP32-output twin of [`super::gdn_verify_fused_conv_kn`]: one
/// launch runs conv1d + SiLU + L2 norm for all K verify positions, writes FP32
/// conv rows, and writes each per-token conv-state snapshot inline.
/// `output_stride` is in FP32 elements.
#[allow(clippy::too_many_arguments)]
pub fn gdn_verify_fused_conv_kn_f32(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    conv_state: DevicePtr,
    new_input: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    conv_state_inter: DevicePtr,
    num_tokens: u32,
    dim: u32,
    d_conv: u32,
    qk_channels: u32,
    head_dim: u32,
    input_stride: u32,
    output_stride: u32,
    inter_stride: u32,
    l2_eps: f32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(dim, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(conv_state)
        .arg_ptr(new_input)
        .arg_ptr(weight.weight)
        .arg_ptr(output)
        .arg_ptr(conv_state_inter)
        .arg_u32(num_tokens)
        .arg_u32(dim)
        .arg_u32(d_conv)
        .arg_u32(qk_channels)
        .arg_u32(head_dim)
        .arg_u32(input_stride)
        .arg_u32(output_stride)
        .arg_u32(inter_stride)
        .arg_f32(l2_eps)
        .launch(stream)
}
