// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Cache-write and paged-decode launchers for the KV cache with TurboQuant on both sides: Turbo4K/Turbo3V, Turbo4K/Turbo8V and Turbo3K/Turbo8V.
//!
//! Owner: model-layers ops.
//! Invariants:
//! - Both pools are byte-addressed blocks with a data section followed by a scale section, each
//!   with its own layout, so every launcher takes `k_block_stride_bytes`, `k_data_section_bytes`,
//!   `v_block_stride_bytes` and `v_data_section_bytes`.
//! - Turbo3 and Turbo4 store Lloyd-Max codes with one FP8 scale per 16 elements; Turbo8 stores
//!   FP8 E4M3 with a BF16 group scale (kernels/gb10/common/reshape_and_cache_turbo.cu).
//! - Cache writes run one 256-thread block per token; decode runs one 256-thread block per
//!   (query head, sequence).

#![allow(unused_imports)]

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

/// 2026-09-25: Write K and V to the Turbo4K / Turbo3V cache (kernel
/// `reshape_and_cache_flash_turbo4k_turbo3v`).
#[allow(clippy::too_many_arguments)]
pub fn reshape_and_cache_turbo4k_turbo3v(
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
    k_block_stride_bytes: u64,
    k_data_section_bytes: u64,
    v_block_stride_bytes: u64,
    v_data_section_bytes: u64,
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
        .arg_u64(k_block_stride_bytes)
        .arg_u64(k_data_section_bytes)
        .arg_u64(v_block_stride_bytes)
        .arg_u64(v_data_section_bytes)
        .launch(stream)
}

/// 2026-09-25: Write K and V to the Turbo4K / Turbo8V cache (kernel
/// `reshape_and_cache_flash_turbo4k_turbo8v`).
#[allow(clippy::too_many_arguments)]
pub fn reshape_and_cache_turbo4k_turbo8v(
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
    k_block_stride_bytes: u64,
    k_data_section_bytes: u64,
    v_block_stride_bytes: u64,
    v_data_section_bytes: u64,
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
        .arg_u64(k_block_stride_bytes)
        .arg_u64(k_data_section_bytes)
        .arg_u64(v_block_stride_bytes)
        .arg_u64(v_data_section_bytes)
        .launch(stream)
}

/// 2026-09-25: Write K and V to the Turbo3K / Turbo8V cache (kernel
/// `reshape_and_cache_flash_turbo3k_turbo8v`).
#[allow(clippy::too_many_arguments)]
pub fn reshape_and_cache_turbo3k_turbo8v(
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
    k_block_stride_bytes: u64,
    k_data_section_bytes: u64,
    v_block_stride_bytes: u64,
    v_data_section_bytes: u64,
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
        .arg_u64(k_block_stride_bytes)
        .arg_u64(k_data_section_bytes)
        .arg_u64(v_block_stride_bytes)
        .arg_u64(v_data_section_bytes)
        .launch(stream)
}

/// 2026-09-25: Paged decode attention over the Turbo4K / Turbo3V cache (kernel
/// `paged_decode_attn_turbo4k_turbo3v`).
#[allow(clippy::too_many_arguments)]
pub fn paged_decode_attn_turbo4k_turbo3v(
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
    k_block_stride_bytes: u64,
    k_data_section_bytes: u64,
    v_block_stride_bytes: u64,
    v_data_section_bytes: u64,
    sliding_window: u32,
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
        .arg_u64(k_block_stride_bytes)
        .arg_u64(k_data_section_bytes)
        .arg_u64(v_block_stride_bytes)
        .arg_u64(v_data_section_bytes)
        .arg_u32(sliding_window)
        .launch(stream)
}

/// 2026-09-25: Paged decode attention over the Turbo4K / Turbo8V cache (kernel
/// `paged_decode_attn_turbo4k_turbo8v`).
#[allow(clippy::too_many_arguments)]
pub fn paged_decode_attn_turbo4k_turbo8v(
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
    k_block_stride_bytes: u64,
    k_data_section_bytes: u64,
    v_block_stride_bytes: u64,
    v_data_section_bytes: u64,
    sliding_window: u32,
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
        .arg_u64(k_block_stride_bytes)
        .arg_u64(k_data_section_bytes)
        .arg_u64(v_block_stride_bytes)
        .arg_u64(v_data_section_bytes)
        .arg_u32(sliding_window)
        .launch(stream)
}

/// 2026-09-25: Paged decode attention over the Turbo3K / Turbo8V cache (kernel
/// `paged_decode_attn_turbo3k_turbo8v`).
#[allow(clippy::too_many_arguments)]
pub fn paged_decode_attn_turbo3k_turbo8v(
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
    k_block_stride_bytes: u64,
    k_data_section_bytes: u64,
    v_block_stride_bytes: u64,
    v_data_section_bytes: u64,
    sliding_window: u32,
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
        .arg_u64(k_block_stride_bytes)
        .arg_u64(k_data_section_bytes)
        .arg_u64(v_block_stride_bytes)
        .arg_u64(v_data_section_bytes)
        .arg_u32(sliding_window)
        .launch(stream)
}
