// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Cache-write and paged-decode launchers for the asymmetric KV cache with BF16 K and TurboQuant V (Turbo2V, Turbo3V, Turbo4V).
//!
//! Owner: model-layers ops.
//! Invariants:
//! - K is stored as raw BF16 in the `[num_blocks, block_size, num_kv_heads, head_dim]` layout; V
//!   as Lloyd-Max codes (2, 3 or 4 bits per element) followed by one FP8 scale per 16 elements
//!   (kernels/gb10/common/reshape_and_cache_turbo.cu). The two pools therefore have separate
//!   block strides, passed in bytes.
//! - Cache writes run one 256-thread block per token; decode runs one 256-thread block per
//!   (query head, sequence). Each launcher passes its kernel's parameters in declaration order.

use super::*;

/// 2026-09-25: Write K and V to the BF16-K / Turbo3V cache (kernel
/// `reshape_and_cache_flash_bf16k_turbo3v`): K as raw BF16, V as 3-bit Lloyd-Max codes (3/8 byte
/// per element) plus one FP8 scale per 16 elements. `v_data_section_bytes` is the size of a V
/// block's code section, after which its scales start.
#[allow(clippy::too_many_arguments)]
pub fn reshape_and_cache_bf16k_turbo3v(
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
        .arg_u64(v_block_stride_bytes)
        .arg_u64(v_data_section_bytes)
        .launch(stream)
}

/// 2026-09-25: Paged decode attention over the BF16-K / Turbo3V cache (kernel
/// `paged_decode_attn_bf16k_turbo3v`). The kernel skips a position's V load and dequant when its
/// attention weight is at most `TQ_PLUS_SPARSE_V_THRESHOLD` (1e-3 by default).
#[allow(clippy::too_many_arguments)]
pub fn paged_decode_attn_bf16k_turbo3v(
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
        .arg_u64(v_block_stride_bytes)
        .arg_u64(v_data_section_bytes)
        .arg_u32(sliding_window)
        .launch(stream)
}

/// 2026-09-25: Write K and V to the BF16-K / Turbo4V cache (kernel
/// `reshape_and_cache_flash_bf16k_turbo4v`): K as raw BF16, V as 4-bit Lloyd-Max codes plus one
/// FP8 scale per 16 elements.
#[allow(clippy::too_many_arguments)]
pub fn reshape_and_cache_bf16k_turbo4v(
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
        .arg_u64(v_block_stride_bytes)
        .arg_u64(v_data_section_bytes)
        .launch(stream)
}

/// 2026-09-25: Write K and V to the BF16-K / Turbo2V cache (kernel
/// `reshape_and_cache_flash_bf16k_turbo2v`): K as raw BF16, V as 2-bit Lloyd-Max codes plus one
/// FP8 scale per 16 elements, 2.5 bits per V element.
#[allow(clippy::too_many_arguments)]
pub fn reshape_and_cache_bf16k_turbo2v(
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
        .arg_u64(v_block_stride_bytes)
        .arg_u64(v_data_section_bytes)
        .launch(stream)
}

/// 2026-09-25: Paged decode attention over the BF16-K / Turbo4V cache (kernel
/// `paged_decode_attn_bf16k_turbo4v`), with the same sparse-V skip as the Turbo3V kernel.
#[allow(clippy::too_many_arguments)]
pub fn paged_decode_attn_bf16k_turbo4v(
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
        .arg_u64(v_block_stride_bytes)
        .arg_u64(v_data_section_bytes)
        .arg_u32(sliding_window)
        .launch(stream)
}

/// 2026-09-25: Paged decode attention over the BF16-K / Turbo2V cache (kernel
/// `paged_decode_attn_bf16k_turbo2v`), with the same sparse-V skip as the Turbo3V kernel.
#[allow(clippy::too_many_arguments)]
pub fn paged_decode_attn_bf16k_turbo2v(
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
        .arg_u64(v_block_stride_bytes)
        .arg_u64(v_data_section_bytes)
        .arg_u32(sliding_window)
        .launch(stream)
}
