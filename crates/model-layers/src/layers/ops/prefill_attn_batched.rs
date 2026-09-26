// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Launchers for the batched paged-prefill attention kernels
//! `attn_prefill_paged{,_fp8,_nvfp4}_batched` and their BR=64 siblings, built
//! from `kernels/gb10/common/prefill_paged_compute.cuh` with `PREFILL_BATCHED`.
//! Grid z is the stream index.
//!
//! The caller provides:
//! 1. `block_table_ptrs`: a device array of per-stream block-table pointers.
//! 2. Q and O for all streams in one buffer. With `cu_seqlens` NULL (uniform),
//!    every stream is `q_len` long and stream `b` starts at
//!    `b * q_len * num_q_heads * head_dim`. With `cu_seqlens` set (varlen),
//!    streams are packed: stream `b` starts at
//!    `cu_seqlens[b] * num_q_heads * head_dim` and is
//!    `cu_seqlens[b+1] - cu_seqlens[b]` long; `q_len` must then be at least
//!    the longest stream, and only sizes the grid.
//! 3. The KV extent: `kv_lens[b]` when `kv_lens` is set, else the shared
//!    `kv_len`. `q_offset`, `sliding_window` and the FP8/NVFP4 scales are
//!    shared by every stream.
//!
//! Every launcher passes `causal_mask_enabled = 1`.
//!
//! Owner: model-layers ops.
//! Invariants: none beyond the types.

#![allow(unused_imports, clippy::too_many_arguments)]

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};

/// 2026-09-25: Batched BF16-KV paged prefill attention, 32 query rows per block.
pub fn prefill_attention_paged_batched(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    q: DevicePtr,
    k_cache: DevicePtr,
    v_cache: DevicePtr,
    output: DevicePtr,
    block_table_ptrs: DevicePtr,
    batch_size: u32,
    cu_seqlens: DevicePtr,
    kv_lens: DevicePtr,
    q_len: u32,
    kv_len: u32,
    q_offset: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    cache_block_size: u32,
    sliding_window: u32,
    inv_sqrt_d: f32,
    stream: u64,
) -> Result<()> {
    let br = 32u32;
    KernelLaunch::new(gpu, kernel)
        .grid([num_q_heads, div_ceil(q_len, br), batch_size])
        .block([128, 1, 1])
        .arg_ptr(q)
        .arg_ptr(k_cache)
        .arg_ptr(v_cache)
        .arg_ptr(output)
        .arg_ptr(block_table_ptrs)
        .arg_u32(batch_size)
        .arg_ptr(cu_seqlens)
        .arg_ptr(kv_lens)
        .arg_u32(q_len)
        .arg_u32(kv_len)
        .arg_u32(q_offset)
        .arg_u32(num_q_heads)
        .arg_u32(num_kv_heads)
        .arg_u32(head_dim)
        .arg_u32(cache_block_size)
        .arg_u32(sliding_window)
        .arg_u32(1u32)
        .arg_f32(inv_sqrt_d)
        .launch(stream)
}

/// 2026-09-25: Batched BF16-KV paged prefill attention, 64 query rows per block.
pub fn prefill_attention_paged_batched_64(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    q: DevicePtr,
    k_cache: DevicePtr,
    v_cache: DevicePtr,
    output: DevicePtr,
    block_table_ptrs: DevicePtr,
    batch_size: u32,
    cu_seqlens: DevicePtr,
    kv_lens: DevicePtr,
    q_len: u32,
    kv_len: u32,
    q_offset: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    cache_block_size: u32,
    sliding_window: u32,
    inv_sqrt_d: f32,
    stream: u64,
) -> Result<()> {
    let br = 64u32;
    KernelLaunch::new(gpu, kernel)
        .grid([num_q_heads, div_ceil(q_len, br), batch_size])
        .block([256, 1, 1])
        .arg_ptr(q)
        .arg_ptr(k_cache)
        .arg_ptr(v_cache)
        .arg_ptr(output)
        .arg_ptr(block_table_ptrs)
        .arg_u32(batch_size)
        .arg_ptr(cu_seqlens)
        .arg_ptr(kv_lens)
        .arg_u32(q_len)
        .arg_u32(kv_len)
        .arg_u32(q_offset)
        .arg_u32(num_q_heads)
        .arg_u32(num_kv_heads)
        .arg_u32(head_dim)
        .arg_u32(cache_block_size)
        .arg_u32(sliding_window)
        .arg_u32(1u32)
        .arg_f32(inv_sqrt_d)
        .launch(stream)
}

/// 2026-09-25: Batched FP8-KV paged prefill attention, 32 query rows per block.
pub fn prefill_attention_paged_fp8_batched(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    q: DevicePtr,
    k_cache: DevicePtr,
    v_cache: DevicePtr,
    output: DevicePtr,
    block_table_ptrs: DevicePtr,
    batch_size: u32,
    cu_seqlens: DevicePtr,
    kv_lens: DevicePtr,
    q_len: u32,
    kv_len: u32,
    q_offset: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    cache_block_size: u32,
    sliding_window: u32,
    inv_sqrt_d: f32,
    k_scale: f32,
    v_scale: f32,
    cache_stride: u64,
    stream: u64,
) -> Result<()> {
    let br = 32u32;
    KernelLaunch::new(gpu, kernel)
        .grid([num_q_heads, div_ceil(q_len, br), batch_size])
        .block([128, 1, 1])
        .arg_ptr(q)
        .arg_ptr(k_cache)
        .arg_ptr(v_cache)
        .arg_ptr(output)
        .arg_ptr(block_table_ptrs)
        .arg_u32(batch_size)
        .arg_ptr(cu_seqlens)
        .arg_ptr(kv_lens)
        .arg_u32(q_len)
        .arg_u32(kv_len)
        .arg_u32(q_offset)
        .arg_u32(num_q_heads)
        .arg_u32(num_kv_heads)
        .arg_u32(head_dim)
        .arg_u32(cache_block_size)
        .arg_u32(sliding_window)
        .arg_u32(1u32)
        .arg_f32(inv_sqrt_d)
        .arg_f32(k_scale)
        .arg_f32(v_scale)
        .arg_u64(cache_stride)
        .launch(stream)
}

/// 2026-09-25: Batched FP8-KV paged prefill attention, 64 query rows per block.
pub fn prefill_attention_paged_fp8_batched_64(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    q: DevicePtr,
    k_cache: DevicePtr,
    v_cache: DevicePtr,
    output: DevicePtr,
    block_table_ptrs: DevicePtr,
    batch_size: u32,
    cu_seqlens: DevicePtr,
    kv_lens: DevicePtr,
    q_len: u32,
    kv_len: u32,
    q_offset: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    cache_block_size: u32,
    sliding_window: u32,
    inv_sqrt_d: f32,
    k_scale: f32,
    v_scale: f32,
    cache_stride: u64,
    stream: u64,
) -> Result<()> {
    let br = 64u32;
    KernelLaunch::new(gpu, kernel)
        .grid([num_q_heads, div_ceil(q_len, br), batch_size])
        .block([256, 1, 1])
        .arg_ptr(q)
        .arg_ptr(k_cache)
        .arg_ptr(v_cache)
        .arg_ptr(output)
        .arg_ptr(block_table_ptrs)
        .arg_u32(batch_size)
        .arg_ptr(cu_seqlens)
        .arg_ptr(kv_lens)
        .arg_u32(q_len)
        .arg_u32(kv_len)
        .arg_u32(q_offset)
        .arg_u32(num_q_heads)
        .arg_u32(num_kv_heads)
        .arg_u32(head_dim)
        .arg_u32(cache_block_size)
        .arg_u32(sliding_window)
        .arg_u32(1u32)
        .arg_f32(inv_sqrt_d)
        .arg_f32(k_scale)
        .arg_f32(v_scale)
        .arg_u64(cache_stride)
        .launch(stream)
}

/// 2026-09-25: Batched NVFP4-KV paged prefill attention, 32 query rows per block.
pub fn prefill_attention_paged_nvfp4_batched(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    q: DevicePtr,
    k_cache: DevicePtr,
    v_cache: DevicePtr,
    output: DevicePtr,
    block_table_ptrs: DevicePtr,
    batch_size: u32,
    cu_seqlens: DevicePtr,
    kv_lens: DevicePtr,
    q_len: u32,
    kv_len: u32,
    q_offset: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    cache_block_size: u32,
    sliding_window: u32,
    inv_sqrt_d: f32,
    block_stride_bytes: u64,
    data_section_bytes: u64,
    stream: u64,
) -> Result<()> {
    let br = 32u32;
    KernelLaunch::new(gpu, kernel)
        .grid([num_q_heads, div_ceil(q_len, br), batch_size])
        .block([128, 1, 1])
        .arg_ptr(q)
        .arg_ptr(k_cache)
        .arg_ptr(v_cache)
        .arg_ptr(output)
        .arg_ptr(block_table_ptrs)
        .arg_u32(batch_size)
        .arg_ptr(cu_seqlens)
        .arg_ptr(kv_lens)
        .arg_u32(q_len)
        .arg_u32(kv_len)
        .arg_u32(q_offset)
        .arg_u32(num_q_heads)
        .arg_u32(num_kv_heads)
        .arg_u32(head_dim)
        .arg_u32(cache_block_size)
        .arg_u32(sliding_window)
        .arg_u32(1u32)
        .arg_f32(inv_sqrt_d)
        .arg_u64(block_stride_bytes)
        .arg_u64(data_section_bytes)
        .launch(stream)
}

/// 2026-09-25: Batched NVFP4-KV paged prefill attention, 64 query rows per block,
/// with the arguments of [`prefill_attention_paged_nvfp4_batched`]. The caller
/// (`qwen3_attention/prefill/paged_attn_batched.rs`) picks the 64-row kernels
/// for every KV type when `chunk_len >= 256`.
pub fn prefill_attention_paged_nvfp4_batched_64(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    q: DevicePtr,
    k_cache: DevicePtr,
    v_cache: DevicePtr,
    output: DevicePtr,
    block_table_ptrs: DevicePtr,
    batch_size: u32,
    cu_seqlens: DevicePtr,
    kv_lens: DevicePtr,
    q_len: u32,
    kv_len: u32,
    q_offset: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    cache_block_size: u32,
    sliding_window: u32,
    inv_sqrt_d: f32,
    block_stride_bytes: u64,
    data_section_bytes: u64,
    stream: u64,
) -> Result<()> {
    let br = 64u32;
    KernelLaunch::new(gpu, kernel)
        .grid([num_q_heads, div_ceil(q_len, br), batch_size])
        .block([256, 1, 1])
        .arg_ptr(q)
        .arg_ptr(k_cache)
        .arg_ptr(v_cache)
        .arg_ptr(output)
        .arg_ptr(block_table_ptrs)
        .arg_u32(batch_size)
        .arg_ptr(cu_seqlens)
        .arg_ptr(kv_lens)
        .arg_u32(q_len)
        .arg_u32(kv_len)
        .arg_u32(q_offset)
        .arg_u32(num_q_heads)
        .arg_u32(num_kv_heads)
        .arg_u32(head_dim)
        .arg_u32(cache_block_size)
        .arg_u32(sliding_window)
        .arg_u32(1u32)
        .arg_f32(inv_sqrt_d)
        .arg_u64(block_stride_bytes)
        .arg_u64(data_section_bytes)
        .launch(stream)
}
