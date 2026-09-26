// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Rotary position embedding launchers: plain, strided, proportional,
//! interleaved multimodal (MRoPE) and table-driven (YaRN).
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

/// 2026-09-25: [`rope`] for rows that are not packed: Q rows `q_row_stride` and
/// K rows `k_row_stride` elements apart, all `num_tokens` rows in one launch.
/// `rope_forward_strided` differs from `rope_forward` only in the row address,
/// so the packed strides give the same result as [`rope`].
#[allow(clippy::too_many_arguments)]
pub fn rope_strided(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    q: DevicePtr,
    k: DevicePtr,
    positions: DevicePtr,
    num_tokens: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    rotary_dim: u32,
    theta: f32,
    q_row_stride: u32,
    k_row_stride: u32,
    stream: u64,
) -> Result<()> {
    assert!(
        rotary_dim > 0,
        "rope_strided: rotary_dim=0, nq={num_q_heads} nkv={num_kv_heads} hd={head_dim}"
    );
    let half_rot = (rotary_dim / 2).max(1);
    let pos_per_block = (128 / half_rot).max(1);
    let seq_blocks = div_ceil(num_tokens, pos_per_block);
    KernelLaunch::new(gpu, kernel)
        .grid([num_q_heads + num_kv_heads, seq_blocks, 1])
        .block([128, 1, 1])
        .arg_ptr(q)
        .arg_ptr(k)
        .arg_ptr(positions)
        .arg_u32(num_tokens)
        .arg_u32(num_q_heads)
        .arg_u32(num_kv_heads)
        .arg_u32(head_dim)
        .arg_u32(rotary_dim)
        .arg_f32(theta)
        .arg_u32(q_row_stride)
        .arg_u32(k_row_stride)
        .launch(stream)
}

/// 2026-09-25: Apply rotary position embeddings to Q and K in place, rows packed
/// (`num_q_heads * head_dim` and `num_kv_heads * head_dim` elements apart).
/// `positions` is a device `u32[seq_len]`.
pub fn rope(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    q: DevicePtr,
    k: DevicePtr,
    positions: DevicePtr,
    seq_len: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    rotary_dim: u32,
    theta: f32,
    stream: u64,
) -> Result<()> {
    assert!(
        rotary_dim > 0,
        "rope: rotary_dim=0, nq={num_q_heads} nkv={num_kv_heads} hd={head_dim}"
    );
    let half_rot = (rotary_dim / 2).max(1);
    let pos_per_block = (128 / half_rot).max(1);
    let seq_blocks = div_ceil(seq_len, pos_per_block);
    KernelLaunch::new(gpu, kernel)
        .grid([num_q_heads + num_kv_heads, seq_blocks, 1])
        .block([128, 1, 1])
        .arg_ptr(q)
        .arg_ptr(k)
        .arg_ptr(positions)
        .arg_u32(seq_len)
        .arg_u32(num_q_heads)
        .arg_u32(num_kv_heads)
        .arg_u32(head_dim)
        .arg_u32(rotary_dim)
        .arg_f32(theta)
        .launch(stream)
}

/// 2026-09-25: Proportional RoPE (Gemma-4 full-attention layers).
///
/// Rotation pairs are (i, i + head_dim/2) for i in [0, rope_angles), and the
/// frequency exponent's denominator is `head_dim`.
#[allow(clippy::too_many_arguments)]
pub fn rope_proportional(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    q: DevicePtr,
    k: DevicePtr,
    positions: DevicePtr,
    seq_len: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    rope_angles: u32,
    theta: f32,
    stream: u64,
) -> Result<()> {
    assert!(rope_angles > 0, "rope_proportional: rope_angles=0");
    let pairs_per_pos = rope_angles.max(1);
    let pos_per_block = (128 / pairs_per_pos).max(1);
    let seq_blocks = div_ceil(seq_len, pos_per_block);
    KernelLaunch::new(gpu, kernel)
        .grid([num_q_heads + num_kv_heads, seq_blocks, 1])
        .block([128, 1, 1])
        .arg_ptr(q)
        .arg_ptr(k)
        .arg_ptr(positions)
        .arg_u32(seq_len)
        .arg_u32(num_q_heads)
        .arg_u32(num_kv_heads)
        .arg_u32(head_dim)
        .arg_u32(rope_angles)
        .arg_f32(theta)
        .launch(stream)
}

/// 2026-09-25: Interleaved multimodal RoPE (MRoPE) with three position streams
/// (`pos_t`, `pos_h`, `pos_w`); rotary pair `i` takes its position from stream
/// `i % 3`. With the same pointer for all three, the math is that of `rope`.
#[allow(clippy::too_many_arguments)]
pub fn rope_mrope_interleaved(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    q: DevicePtr,
    k: DevicePtr,
    pos_t: DevicePtr,
    pos_h: DevicePtr,
    pos_w: DevicePtr,
    seq_len: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    rotary_dim: u32,
    theta: f32,
    stream: u64,
) -> Result<()> {
    assert!(rotary_dim > 0, "rope_mrope_interleaved: rotary_dim=0");
    let half_rot = (rotary_dim / 2).max(1);
    let pos_per_block = (128 / half_rot).max(1);
    let seq_blocks = div_ceil(seq_len, pos_per_block);
    KernelLaunch::new(gpu, kernel)
        .grid([num_q_heads + num_kv_heads, seq_blocks, 1])
        .block([128, 1, 1])
        .arg_ptr(q)
        .arg_ptr(k)
        .arg_ptr(pos_t)
        .arg_ptr(pos_h)
        .arg_ptr(pos_w)
        .arg_u32(seq_len)
        .arg_u32(num_q_heads)
        .arg_u32(num_kv_heads)
        .arg_u32(head_dim)
        .arg_u32(rotary_dim)
        .arg_f32(theta)
        .launch(stream)
}

/// 2026-09-25: [`rope_mrope_interleaved`] for K only, for when Q was already
/// rotated by a fused prefill kernel.
#[allow(clippy::too_many_arguments)]
pub fn rope_mrope_interleaved_k_only(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    k: DevicePtr,
    pos_t: DevicePtr,
    pos_h: DevicePtr,
    pos_w: DevicePtr,
    seq_len: u32,
    num_kv_heads: u32,
    head_dim: u32,
    rotary_dim: u32,
    theta: f32,
    stream: u64,
) -> Result<()> {
    assert!(
        rotary_dim > 0,
        "rope_mrope_interleaved_k_only: rotary_dim=0"
    );
    let half_rot = (rotary_dim / 2).max(1);
    let pos_per_block = (128 / half_rot).max(1);
    let seq_blocks = div_ceil(seq_len, pos_per_block);
    KernelLaunch::new(gpu, kernel)
        .grid([num_kv_heads, seq_blocks, 1])
        .block([128, 1, 1])
        .arg_ptr(k)
        .arg_ptr(pos_t)
        .arg_ptr(pos_h)
        .arg_ptr(pos_w)
        .arg_u32(seq_len)
        .arg_u32(num_kv_heads)
        .arg_u32(head_dim)
        .arg_u32(rotary_dim)
        .arg_f32(theta)
        .launch(stream)
}

/// 2026-09-25: RoPE with frequencies from a precomputed `inv_freq` table
/// (`[rotary_dim/2]` FP32); `rope_forward_yarn` ignores `theta`.
#[allow(clippy::too_many_arguments)]
pub fn rope_yarn(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    q: DevicePtr,
    k: DevicePtr,
    positions: DevicePtr,
    seq_len: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    rotary_dim: u32,
    inv_freq: DevicePtr,
    theta: f32,
    stream: u64,
) -> Result<()> {
    assert!(
        rotary_dim > 0,
        "rope: rotary_dim=0, nq={num_q_heads} nkv={num_kv_heads} hd={head_dim}"
    );
    let half_rot = (rotary_dim / 2).max(1);
    let pos_per_block = (128 / half_rot).max(1);
    let seq_blocks = div_ceil(seq_len, pos_per_block);
    KernelLaunch::new(gpu, kernel)
        .grid([num_q_heads + num_kv_heads, seq_blocks, 1])
        .block([128, 1, 1])
        .arg_ptr(q)
        .arg_ptr(k)
        .arg_ptr(positions)
        .arg_u32(seq_len)
        .arg_u32(num_q_heads)
        .arg_u32(num_kv_heads)
        .arg_u32(head_dim)
        .arg_u32(rotary_dim)
        .arg_ptr(inv_freq)
        .arg_f32(theta)
        .launch(stream)
}

/// 2026-09-25: RoPE from an `inv_freq` table with cosine and sine scaled by
/// `attention_factor`.
#[allow(clippy::too_many_arguments)]
pub fn rope_yarn_scaled(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    q: DevicePtr,
    k: DevicePtr,
    positions: DevicePtr,
    seq_len: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    rotary_dim: u32,
    inv_freq: DevicePtr,
    attention_factor: f32,
    stream: u64,
) -> Result<()> {
    assert!(rotary_dim > 0, "rope_yarn_scaled: rotary_dim=0");
    let half_rot = (rotary_dim / 2).max(1);
    let pos_per_block = (128 / half_rot).max(1);
    let seq_blocks = div_ceil(seq_len, pos_per_block);
    KernelLaunch::new(gpu, kernel)
        .grid([num_q_heads + num_kv_heads, seq_blocks, 1])
        .block([128, 1, 1])
        .arg_ptr(q)
        .arg_ptr(k)
        .arg_ptr(positions)
        .arg_u32(seq_len)
        .arg_u32(num_q_heads)
        .arg_u32(num_kv_heads)
        .arg_u32(head_dim)
        .arg_u32(rotary_dim)
        .arg_ptr(inv_freq)
        .arg_f32(attention_factor)
        .launch(stream)
}
