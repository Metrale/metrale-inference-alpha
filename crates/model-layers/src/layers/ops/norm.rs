// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Launchers for the RMS norm family: plain, strided, warp-per-row,
//! with residual save, with residual add, and gated.
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

/// 2026-09-25: RMS norm of `num_groups` groups of `rows_per_group` rows in one
/// launch. Groups start `row_stride` elements apart, and rows inside a group are
/// packed at `hidden_size`. The multi-sequence q/k head norms use it because each
/// sequence's heads sit inside that sequence's own interleaved block.
///
/// `rms_norm_strided` in `kernels/gb10/common/rms_norm.cu` has the per-row body
/// of `rms_norm` there; only the row's base address differs.
#[allow(clippy::too_many_arguments)]
pub fn rms_norm_strided(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    rows_per_group: u32,
    num_groups: u32,
    hidden_size: u32,
    eps: f32,
    row_stride: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([rows_per_group, num_groups, 1])
        .block([hidden_size.min(1024), 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(output)
        .arg_u32(hidden_size)
        .arg_f32(eps)
        .arg_u32(row_stride)
        .launch(stream)
}

/// 2026-09-25: RMS norm of `num_tokens` packed rows of `hidden_size`, one block
/// per row. How `weight` scales the result is the kernel's: `rms_norm` in
/// `kernels/gb10/common/rms_norm.cu` multiplies by `1 + weight`, and the
/// `rms_norm_vanilla` kernels by `weight`.
pub fn rms_norm(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    eps: f32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([hidden_size.min(1024), 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(output)
        .arg_u32(hidden_size)
        .arg_f32(eps)
        .launch(stream)
}

/// 2026-09-25: RMS norm with one warp per row and 8 rows per block, so the
/// reduction needs no shared memory or block barrier. The prefill per-head
/// `q_norm`/`k_norm` uses it when [`rms_norm_short_row_eligible`] holds
/// (`qwen3_attention/prefill/cache_skip.rs`).
pub fn rms_norm_warp_row(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    num_rows: u32,
    hidden_size: u32,
    eps: f32,
    stream: u64,
) -> Result<()> {
    const ROWS_PER_BLOCK: u32 = 8;
    KernelLaunch::new(gpu, kernel)
        .grid([num_rows.div_ceil(ROWS_PER_BLOCK), 1, 1])
        .block([32 * ROWS_PER_BLOCK, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(output)
        .arg_u32(num_rows)
        .arg_u32(hidden_size)
        .arg_f32(eps)
        .launch(stream)
}

/// 2026-09-25: Gate for [`rms_norm_warp_row`]: `hidden_size <= 256` and even,
/// and `num_rows >= 1024`. `METRALE_RMS_NORM_WARP_ROW=0` turns it off; the
/// variable is read once per process.
pub fn rms_norm_short_row_eligible(num_rows: u32, hidden_size: u32) -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    let on = *ON.get_or_init(|| std::env::var("METRALE_RMS_NORM_WARP_ROW").as_deref() != Ok("0"));
    on && hidden_size <= 256 && hidden_size.is_multiple_of(2) && num_rows >= 1024
}

/// 2026-09-25: `output = rms_norm(input)` and `residual = input` in one pass,
/// one block per token row.
pub fn rms_norm_residual(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    residual: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    eps: f32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([hidden_size.min(1024), 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(output)
        .arg_ptr(residual)
        .arg_u32(hidden_size)
        .arg_f32(eps)
        .launch(stream)
}

/// 2026-09-25: `hidden += src`, then `output = rms_norm(hidden)` and
/// `residual = hidden`, in one kernel, one block per token row.
#[allow(clippy::too_many_arguments)]
pub fn residual_add_rms_norm(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    hidden: DevicePtr,
    src: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    residual: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    eps: f32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([hidden_size.min(1024), 1, 1])
        .arg_ptr(hidden)
        .arg_ptr(src)
        .arg_ptr(weight.weight)
        .arg_ptr(output)
        .arg_ptr(residual)
        .arg_u32(hidden_size)
        .arg_f32(eps)
        .launch(stream)
}

/// 2026-09-25: [`residual_add_rms_norm`] that also writes the normed row in FP32
/// to `output_f32`, the MoE router input, so routing does not see the BF16
/// rounding of the norm output. Callers use it when `METRALE_FP32_ROUTING` is
/// active for the layer's FFN.
#[allow(clippy::too_many_arguments)]
pub fn residual_add_rms_norm_gatef32(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    hidden: DevicePtr,
    src: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    output_f32: DevicePtr,
    residual: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    eps: f32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([hidden_size.min(1024), 1, 1])
        .arg_ptr(hidden)
        .arg_ptr(src)
        .arg_ptr(weight.weight)
        .arg_ptr(output)
        .arg_ptr(output_f32)
        .arg_ptr(residual)
        .arg_u32(hidden_size)
        .arg_f32(eps)
        .launch(stream)
}

/// 2026-09-25: Gated RMS norm of `num_tokens` rows, one block per row; the gate
/// row of token `t` starts at `t * gate_stride`. The order of norm and gate is
/// the resolved kernel's. `gated_rms_norm` and `gated_rms_norm_f32_input` in
/// `kernels/gb10/common/rms_norm.cu` compute `rms_norm(input) * weight *
/// silu(gate)` and ignore `group_size`. The per-group shadows, e.g.
/// `kernels/gb10/minimax-m2-229b/nvfp4/rms_norm.cu`, gate first and normalise
/// each `group_size` slice.
pub fn gated_rms_norm(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    gate: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    num_tokens: u32,
    hidden_size: u32,
    gate_stride: u32,
    eps: f32,
    group_size: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([hidden_size.min(1024), 1, 1])
        .arg_ptr(input)
        .arg_ptr(gate)
        .arg_ptr(weight.weight)
        .arg_ptr(output)
        .arg_u32(hidden_size)
        .arg_f32(eps)
        .arg_u32(gate_stride)
        .arg_u32(group_size)
        .launch(stream)
}

/// 2026-09-25: Gated RMS norm of every `(head, sequence)` row of a multi-sequence
/// decode in one launch: `heads_per_seq` rows per sequence, `num_seqs`
/// sequences. Sequence strides count elements of each buffer's own type:
/// `input_seq_stride` in f32, `gate_seq_stride` and `output_seq_stride` in BF16.
///
/// The kernel, `gated_rms_norm_f32_input_strided` in
/// `kernels/gb10/common/rms_norm.cu`, has the per-row body of
/// `gated_rms_norm_f32_input`; only the row's base address differs.
#[allow(clippy::too_many_arguments)]
pub fn gated_rms_norm_strided(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    gate: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    heads_per_seq: u32,
    num_seqs: u32,
    hidden_size: u32,
    gate_stride: u32,
    eps: f32,
    group_size: u32,
    input_seq_stride: u32,
    gate_seq_stride: u32,
    output_seq_stride: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([heads_per_seq, num_seqs, 1])
        .block([hidden_size.min(1024), 1, 1])
        .arg_ptr(input)
        .arg_ptr(gate)
        .arg_ptr(weight.weight)
        .arg_ptr(output)
        .arg_u32(hidden_size)
        .arg_f32(eps)
        .arg_u32(gate_stride)
        .arg_u32(group_size)
        .arg_u32(input_seq_stride)
        .arg_u32(gate_seq_stride)
        .arg_u32(output_seq_stride)
        .launch(stream)
}

/// 2026-09-25: Gated RMS norm of every `(head, token)` row of a prefill chunk in
/// one launch. `input` and `output` rows of token `t` start at
/// `t * input_token_stride`, and gate rows at `t * gate_token_stride`, both in
/// BF16 elements.
#[allow(clippy::too_many_arguments)]
pub fn gated_rms_norm_prefill(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    gate: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    heads_per_token: u32,
    head_dim: u32,
    eps: f32,
    num_actual_tokens: u32,
    input_token_stride: u32,
    gate_token_stride: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([heads_per_token, num_actual_tokens, 1])
        .block([head_dim.min(1024), 1, 1])
        .arg_ptr(input)
        .arg_ptr(gate)
        .arg_ptr(weight.weight)
        .arg_ptr(output)
        .arg_u32(head_dim)
        .arg_f32(eps)
        .arg_u32(input_token_stride)
        .arg_u32(gate_token_stride)
        .launch(stream)
}
