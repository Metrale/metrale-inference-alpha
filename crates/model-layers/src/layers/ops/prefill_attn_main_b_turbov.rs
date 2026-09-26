// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Launchers for the paged prefill attention kernels with a BF16 K
//! cache and a Lloyd-Max packed V cache (`attn_prefill_paged_bf16k_turbo{2,3,4}v`
//! in `kernels/gb10/common/`, built from `prefill_paged_compute_asym.cuh`). V
//! holds 2, 3 or 4 bits per value plus one E4M3 scale per 16 values, laid out
//! as a data section then a scale section per block (`v_block_stride_bytes`,
//! `v_data_section_bytes`). All three launch with `causal_mask_enabled = 1`.
//!
//! Owner: model-layers ops.
//! Invariants: none beyond the types.

use super::*;

/// 2026-09-25: BF16 K (`[block_size, num_kv_heads, head_dim]` per block) with
/// 3-bit V.
#[allow(clippy::too_many_arguments)]
pub fn prefill_attention_paged_bf16k_turbo3v_64(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    q: DevicePtr,
    k_cache: DevicePtr,
    v_cache: DevicePtr,
    output: DevicePtr,
    block_table: DevicePtr,
    q_len: u32,
    kv_len: u32,
    q_offset: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    cache_block_size: u32,
    sliding_window: u32,
    inv_sqrt_d: f32,
    v_block_stride_bytes: u64,
    v_data_section_bytes: u64,
    stream: u64,
) -> Result<()> {
    // 2026-09-25: `prefill_paged_compute_asym.cuh` defines `BR64` as 64 on every
    // target, so under `metrale_scale` this 32-row grid has about twice the
    // blocks the kernel needs; a block whose `q_block * 64` reaches `q_len`
    // returns at once.
    let br = if cfg!(metrale_scale) { 32u32 } else { 64u32 };
    KernelLaunch::new(gpu, kernel)
        .grid([num_q_heads, div_ceil(q_len, br), 1])
        .block([256, 1, 1])
        .arg_ptr(q)
        .arg_ptr(k_cache)
        .arg_ptr(v_cache)
        .arg_ptr(output)
        .arg_ptr(block_table)
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
        .arg_u64(v_block_stride_bytes)
        .arg_u64(v_data_section_bytes)
        .launch(stream)
}

/// 2026-09-25: BF16 K with 4-bit V; the arguments of
/// [`prefill_attention_paged_bf16k_turbo3v_64`].
#[allow(clippy::too_many_arguments)]
pub fn prefill_attention_paged_bf16k_turbo4v_64(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    q: DevicePtr,
    k_cache: DevicePtr,
    v_cache: DevicePtr,
    output: DevicePtr,
    block_table: DevicePtr,
    q_len: u32,
    kv_len: u32,
    q_offset: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    cache_block_size: u32,
    sliding_window: u32,
    inv_sqrt_d: f32,
    v_block_stride_bytes: u64,
    v_data_section_bytes: u64,
    stream: u64,
) -> Result<()> {
    // 2026-09-25: `prefill_paged_compute_asym.cuh` defines `BR64` as 64 on every
    // target, so under `metrale_scale` this 32-row grid has about twice the
    // blocks the kernel needs; a block whose `q_block * 64` reaches `q_len`
    // returns at once.
    let br = if cfg!(metrale_scale) { 32u32 } else { 64u32 };
    KernelLaunch::new(gpu, kernel)
        .grid([num_q_heads, div_ceil(q_len, br), 1])
        .block([256, 1, 1])
        .arg_ptr(q)
        .arg_ptr(k_cache)
        .arg_ptr(v_cache)
        .arg_ptr(output)
        .arg_ptr(block_table)
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
        .arg_u64(v_block_stride_bytes)
        .arg_u64(v_data_section_bytes)
        .launch(stream)
}

/// 2026-09-25: BF16 K with 2-bit V; the arguments of
/// [`prefill_attention_paged_bf16k_turbo3v_64`].
#[allow(clippy::too_many_arguments)]
pub fn prefill_attention_paged_bf16k_turbo2v_64(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    q: DevicePtr,
    k_cache: DevicePtr,
    v_cache: DevicePtr,
    output: DevicePtr,
    block_table: DevicePtr,
    q_len: u32,
    kv_len: u32,
    q_offset: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    cache_block_size: u32,
    sliding_window: u32,
    inv_sqrt_d: f32,
    v_block_stride_bytes: u64,
    v_data_section_bytes: u64,
    stream: u64,
) -> Result<()> {
    // 2026-09-25: `prefill_paged_compute_asym.cuh` defines `BR64` as 64 on every
    // target, so under `metrale_scale` this 32-row grid has about twice the
    // blocks the kernel needs; a block whose `q_block * 64` reaches `q_len`
    // returns at once.
    let br = if cfg!(metrale_scale) { 32u32 } else { 64u32 };
    KernelLaunch::new(gpu, kernel)
        .grid([num_q_heads, div_ceil(q_len, br), 1])
        .block([256, 1, 1])
        .arg_ptr(q)
        .arg_ptr(k_cache)
        .arg_ptr(v_cache)
        .arg_ptr(output)
        .arg_ptr(block_table)
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
        .arg_u64(v_block_stride_bytes)
        .arg_u64(v_data_section_bytes)
        .launch(stream)
}
