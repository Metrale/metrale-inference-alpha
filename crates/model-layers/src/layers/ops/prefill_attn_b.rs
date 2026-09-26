// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Launchers for the NVFP4 KV cache write, the BF16 absmax used by
//! FP8 KV calibration, and the paged decode attention kernels for NVFP4, FP8
//! and BF16 caches, with their split-K partial and reduce passes.
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

/// 2026-09-25: Write each token's K and V into the paged NVFP4 cache at its
/// `slot_mapping` slot: E2M1 values with one E4M3 scale per 16. One block per
/// token.
pub fn reshape_and_cache_nvfp4(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    key: DevicePtr,
    value: DevicePtr,
    k_cache: DevicePtr,
    v_cache: DevicePtr,
    slot_mapping: DevicePtr,
    num_tokens: u32,
    num_kv_heads: u32,
    head_dim: u32,
    block_size: u32,
    key_stride: u32,
    value_stride: u32,
    block_stride_bytes: u64,
    data_section_bytes: u64,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(key)
        .arg_ptr(value)
        .arg_ptr(k_cache)
        .arg_ptr(v_cache)
        .arg_ptr(slot_mapping)
        .arg_u32(num_kv_heads)
        .arg_u32(head_dim)
        .arg_u32(block_size)
        .arg_u32(key_stride)
        .arg_u32(value_stride)
        .arg_u64(block_stride_bytes)
        .arg_u64(data_section_bytes)
        .launch(stream)
}

/// 2026-09-25: Max absolute value of `n_elems` BF16 values, folded into the f32
/// at `out_max` with an atomic max. The caller zeroes `out_max` first; the FP8
/// KV calibration (`fp8_calibration.rs`) does so before each K/V pair.
pub fn bf16_absmax(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    data: DevicePtr,
    out_max: DevicePtr,
    n_elems: u32,
    stream: u64,
) -> Result<()> {
    // 2026-09-25: At most 256 blocks; the kernel's grid-stride loop covers the rest.
    let grid_x = (n_elems as u64).div_ceil(256 * 2).min(256) as u32;
    KernelLaunch::new(gpu, kernel)
        .grid([grid_x, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(data)
        .arg_ptr(out_max)
        .arg_u32(n_elems)
        .launch(stream)
}

/// 2026-09-25: Paged decode attention over an NVFP4 KV cache, one block per
/// `(q_head, seq)`.
pub fn paged_decode_attn_nvfp4(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    q: DevicePtr,
    k_cache: DevicePtr,
    v_cache: DevicePtr,
    output: DevicePtr,
    block_tables: DevicePtr,
    seq_lens: DevicePtr,
    max_blocks_per_seq: u32,
    num_seqs: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    block_size: u32,
    inv_sqrt_d: f32,
    q_stride: u32,
    block_stride_bytes: u64,
    data_section_bytes: u64,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_q_heads, num_seqs, 1])
        .block([256, 1, 1])
        .arg_ptr(q)
        .arg_ptr(k_cache)
        .arg_ptr(v_cache)
        .arg_ptr(output)
        .arg_ptr(block_tables)
        .arg_ptr(seq_lens)
        .arg_u32(max_blocks_per_seq)
        .arg_u32(num_q_heads)
        .arg_u32(num_kv_heads)
        .arg_u32(head_dim)
        .arg_u32(block_size)
        .arg_f32(inv_sqrt_d)
        .arg_u32(q_stride)
        .arg_u64(block_stride_bytes)
        .arg_u64(data_section_bytes)
        .launch(stream)
}

/// 2026-09-25: Split-K paged decode attention over an NVFP4 KV cache: each
/// `(q_head, seq)` sequence is cut into `num_splits` parts, and each block
/// writes its partial output and softmax state to `workspace`.
#[allow(clippy::too_many_arguments)]
pub fn paged_decode_attn_splitk_nvfp4(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    q: DevicePtr,
    k_cache: DevicePtr,
    v_cache: DevicePtr,
    workspace: DevicePtr,
    block_tables: DevicePtr,
    seq_lens: DevicePtr,
    max_blocks_per_seq: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    block_size: u32,
    inv_sqrt_d: f32,
    num_splits: u32,
    q_stride: u32,
    block_stride_bytes: u64,
    data_section_bytes: u64,
    num_seqs: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_q_heads, num_splits, num_seqs])
        .block([256, 1, 1])
        .arg_ptr(q)
        .arg_ptr(k_cache)
        .arg_ptr(v_cache)
        .arg_ptr(workspace)
        .arg_ptr(block_tables)
        .arg_ptr(seq_lens)
        .arg_u32(max_blocks_per_seq)
        .arg_u32(num_q_heads)
        .arg_u32(num_kv_heads)
        .arg_u32(head_dim)
        .arg_u32(block_size)
        .arg_f32(inv_sqrt_d)
        .arg_u32(num_splits)
        .arg_u32(q_stride)
        .arg_u64(block_stride_bytes)
        .arg_u64(data_section_bytes)
        .launch(stream)
}

/// 2026-09-25: Combine the split-K partials in `workspace` into the BF16 output.
#[allow(clippy::too_many_arguments)]
pub fn paged_decode_attn_reduce_nvfp4(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    workspace: DevicePtr,
    output: DevicePtr,
    seq_lens: DevicePtr,
    num_q_heads: u32,
    head_dim: u32,
    num_splits: u32,
    num_seqs: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_q_heads, num_seqs, 1])
        .block([32, 1, 1])
        .arg_ptr(workspace)
        .arg_ptr(output)
        .arg_ptr(seq_lens)
        .arg_u32(num_q_heads)
        .arg_u32(head_dim)
        .arg_u32(num_splits)
        .launch(stream)
}

/// 2026-09-25: The FP8-KV counterpart of [`paged_decode_attn_splitk_nvfp4`].
#[allow(clippy::too_many_arguments)]
pub fn paged_decode_attn_splitk_fp8(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    q: DevicePtr,
    k_cache: DevicePtr,
    v_cache: DevicePtr,
    workspace: DevicePtr,
    block_tables: DevicePtr,
    seq_lens: DevicePtr,
    max_blocks_per_seq: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    block_size: u32,
    inv_sqrt_d: f32,
    num_splits: u32,
    k_scale: f32,
    v_scale: f32,
    q_stride: u32,
    cache_stride: u64,
    num_seqs: u32,
    sliding_window: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_q_heads, num_splits, num_seqs])
        .block([256, 1, 1])
        .arg_ptr(q)
        .arg_ptr(k_cache)
        .arg_ptr(v_cache)
        .arg_ptr(workspace)
        .arg_ptr(block_tables)
        .arg_ptr(seq_lens)
        .arg_u32(max_blocks_per_seq)
        .arg_u32(num_q_heads)
        .arg_u32(num_kv_heads)
        .arg_u32(head_dim)
        .arg_u32(block_size)
        .arg_f32(inv_sqrt_d)
        .arg_u32(num_splits)
        .arg_f32(k_scale)
        .arg_f32(v_scale)
        .arg_u32(q_stride)
        .arg_u64(cache_stride)
        .arg_u32(sliding_window)
        .launch(stream)
}

/// 2026-09-25: The BF16-KV counterpart of [`paged_decode_attn_splitk_nvfp4`]
/// (`kernels/hopper/common/paged_decode_bf16_splitk_hopper.cu`). It takes no
/// `cache_stride`: the cache is `[blocks, block_size, kv_heads, head_dim]` and
/// the kernel computes the page stride, as [`paged_decode_attn_bf16`] does.
#[allow(clippy::too_many_arguments)]
pub fn paged_decode_attn_splitk_bf16(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    q: DevicePtr,
    k_cache: DevicePtr,
    v_cache: DevicePtr,
    workspace: DevicePtr,
    block_tables: DevicePtr,
    seq_lens: DevicePtr,
    max_blocks_per_seq: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    block_size: u32,
    inv_sqrt_d: f32,
    num_splits: u32,
    q_stride: u32,
    num_seqs: u32,
    sliding_window: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_q_heads, num_splits, num_seqs])
        .block([256, 1, 1])
        .arg_ptr(q)
        .arg_ptr(k_cache)
        .arg_ptr(v_cache)
        .arg_ptr(workspace)
        .arg_ptr(block_tables)
        .arg_ptr(seq_lens)
        .arg_u32(max_blocks_per_seq)
        .arg_u32(num_q_heads)
        .arg_u32(num_kv_heads)
        .arg_u32(head_dim)
        .arg_u32(block_size)
        .arg_f32(inv_sqrt_d)
        .arg_u32(num_splits)
        .arg_u32(q_stride)
        .arg_u32(sliding_window)
        .launch(stream)
}

/// 2026-09-25: Combine split-K partials into the BF16 output. The workspace is F32
/// `[head_dim values, m, l]` per split whatever the cache type, so the BF16
/// split-K's reduce handle is launched through this function too
/// (`qwen3_attention/decode/splitk_dispatch.rs`).
#[allow(clippy::too_many_arguments)]
pub fn paged_decode_attn_reduce_fp8(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    workspace: DevicePtr,
    output: DevicePtr,
    seq_lens: DevicePtr,
    num_q_heads: u32,
    head_dim: u32,
    num_splits: u32,
    num_seqs: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_q_heads, num_seqs, 1])
        .block([32, 1, 1])
        .arg_ptr(workspace)
        .arg_ptr(output)
        .arg_ptr(seq_lens)
        .arg_u32(num_q_heads)
        .arg_u32(head_dim)
        .arg_u32(num_splits)
        .launch(stream)
}
