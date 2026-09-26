// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Launchers for the fused GDN verify kernels: conv1d + L2 norm over
//! K=2 or K=n draft positions (one sequence or a batch), and the K=2 gated RMS norm.
//!
//! Owner: model-layers ops.
//! Invariants: none beyond the types.

use super::*;

/// 2026-09-25: K=2 MTP-verify conv1d + SiLU + L2 norm: both draft positions in
/// one launch, with the position-0 conv-state snapshot written inline.
///
/// On return `conv_state` holds the committed (post position-1) window and
/// `conv_state_inter` the position-0 rollback snapshot. The
/// `gdn_verify_fused_microtest` example checks both states, and the gated-norm
/// output computed from this kernel's output, against the per-token kernels at
/// cos >= 0.99999.
///
/// Kernel: `gdn_verify_fused_conv_k2(conv_state, new_input, weight, output,
///          conv_state_inter, dim, d_conv, qk_channels, head_dim,
///          input_stride, output_stride, l2_eps)`
/// Grid: (ceil(dim/256), 1, 1)  Block: (256, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn gdn_verify_fused_conv_k2(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    conv_state: DevicePtr,
    new_input: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    conv_state_inter: DevicePtr,
    d_inner: u32,
    d_conv: u32,
    qk_channels: u32,
    head_dim: u32,
    input_stride: u32,
    output_stride: u32,
    l2_eps: f32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(d_inner, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(conv_state)
        .arg_ptr(new_input)
        .arg_ptr(weight.weight)
        .arg_ptr(output)
        .arg_ptr(conv_state_inter)
        .arg_u32(d_inner)
        .arg_u32(d_conv)
        .arg_u32(qk_channels)
        .arg_u32(head_dim)
        .arg_u32(input_stride)
        .arg_u32(output_stride)
        .arg_f32(l2_eps)
        .launch(stream)
}

/// 2026-09-25: [`gdn_verify_fused_conv_kn`] for `n_seq` sequences in one launch,
/// one grid row (`blockIdx.y`) per sequence.
///
/// The kernel offsets `conv_state`, `new_input`, `output` and
/// `conv_state_inter` by `seq` times their `*_seq_stride`, then runs the same
/// per-sequence body as `gdn_verify_fused_conv_kn`, so each sequence gets the
/// result a separate call would give it.
///
/// Grid: (ceil(d_inner/256), n_seq, 1)  Block: (256, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn gdn_verify_fused_conv_kn_batched(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    conv_state: DevicePtr,
    new_input: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    conv_state_inter: DevicePtr,
    num_tokens: u32,
    d_inner: u32,
    d_conv: u32,
    qk_channels: u32,
    head_dim: u32,
    input_stride: u32,
    output_stride: u32,
    inter_stride: u32,
    l2_eps: f32,
    n_seq: u32,
    conv_state_seq_stride: u32,
    input_seq_stride: u32,
    output_seq_stride: u32,
    inter_seq_stride: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(d_inner, 256), n_seq, 1])
        .block([256, 1, 1])
        .arg_ptr(conv_state)
        .arg_ptr(new_input)
        .arg_ptr(weight.weight)
        .arg_ptr(output)
        .arg_ptr(conv_state_inter)
        .arg_u32(num_tokens)
        .arg_u32(d_inner)
        .arg_u32(d_conv)
        .arg_u32(qk_channels)
        .arg_u32(head_dim)
        .arg_u32(input_stride)
        .arg_u32(output_stride)
        .arg_u32(inter_stride)
        .arg_f32(l2_eps)
        .arg_u32(conv_state_seq_stride)
        .arg_u32(input_seq_stride)
        .arg_u32(output_seq_stride)
        .arg_u32(inter_seq_stride)
        .launch(stream)
}

/// 2026-09-25: K-position verify conv1d + SiLU + L2 norm: all `num_tokens`
/// draft positions in one launch. Position `t`'s conv-state snapshot is
/// written to `conv_state_inter + t * inter_stride`.
///
/// On return `conv_state` holds the window after the last position, which is
/// also snapshot `num_tokens - 1`.
///
/// Kernel: `gdn_verify_fused_conv_kn(conv_state, new_input, weight, output,
///          conv_state_inter, num_tokens, dim, d_conv, qk_channels, head_dim,
///          input_stride, output_stride, inter_stride, l2_eps)`
/// Grid: (ceil(dim/256), 1, 1)  Block: (256, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn gdn_verify_fused_conv_kn(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    conv_state: DevicePtr,
    new_input: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    conv_state_inter: DevicePtr,
    num_tokens: u32,
    d_inner: u32,
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
        .grid([div_ceil(d_inner, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(conv_state)
        .arg_ptr(new_input)
        .arg_ptr(weight.weight)
        .arg_ptr(output)
        .arg_ptr(conv_state_inter)
        .arg_u32(num_tokens)
        .arg_u32(d_inner)
        .arg_u32(d_conv)
        .arg_u32(qk_channels)
        .arg_u32(head_dim)
        .arg_u32(input_stride)
        .arg_u32(output_stride)
        .arg_u32(inter_stride)
        .arg_f32(l2_eps)
        .launch(stream)
}

/// 2026-09-25: K=2 MTP-verify gated RMS norm: both draft positions in one
/// launch. The Z gate is read from the deinterleaved [Q|K|V|Z] buffer at
/// `z_offset` within each position. The `gdn_verify_fused_microtest` example
/// checks it against the per-token `gated_rms_norm` at cos >= 0.99999.
///
/// Kernel: `gdn_verify_fused_norm_k2(gdn_out, deint, weight, output,
///          hidden_size, eps, deint_stride, z_offset, out_stride)`
/// Grid: (num_v_heads, 2, 1)  Block: (hidden_size, 1, 1)
#[allow(clippy::too_many_arguments)]
pub fn gdn_verify_fused_norm_k2(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    gdn_out: DevicePtr,
    deint: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    num_v_heads: u32,
    hidden_size: u32,
    eps: f32,
    deint_stride: u32,
    z_offset: u32,
    out_stride: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_v_heads, 2, 1])
        .block([hidden_size, 1, 1])
        .arg_ptr(gdn_out)
        .arg_ptr(deint)
        .arg_ptr(weight.weight)
        .arg_ptr(output)
        .arg_u32(hidden_size)
        .arg_f32(eps)
        .arg_u32(deint_stride)
        .arg_u32(z_offset)
        .arg_u32(out_stride)
        .launch(stream)
}
