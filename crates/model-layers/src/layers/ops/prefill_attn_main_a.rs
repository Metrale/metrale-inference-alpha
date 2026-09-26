// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Launchers for the prefill flash-attention kernels: contiguous
//! Q/K/V (BF16 and FP8 K/V, 32- and 64-row tiles, the HDIM=512 sink variant)
//! and paged KV (BF16 and FP8, causal and the bidirectional DFlash launches).
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

/// 2026-09-25: Flash-attention prefill on contiguous BF16 tensors: Q and O
/// `[batch, seq_len, num_q_heads, head_dim]`, K and V `[batch, seq_len,
/// num_kv_heads, head_dim]`. `sliding_window == 0` disables the window;
/// otherwise keys with `q - k >= sliding_window` are masked.
#[allow(clippy::too_many_arguments)]
pub fn prefill_attention(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    q: DevicePtr,
    k: DevicePtr,
    v: DevicePtr,
    output: DevicePtr,
    seq_len: u32,
    batch: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    inv_sqrt_d: f32,
    causal: bool,
    sliding_window: u32,
    stream: u64,
) -> Result<()> {
    // 2026-09-25: For head_dim > 256 the handle is resolved in `init.rs` and the
    // grid is built here; both call `wide_prefill_kernel()`, which returns the
    // kernel and its BR together.
    let br = if head_dim > 256 {
        wide_prefill_kernel(gpu).1
    } else {
        32u32
    };
    KernelLaunch::new(gpu, kernel)
        .grid([num_q_heads, div_ceil(seq_len, br), batch])
        .block([128, 1, 1])
        .arg_ptr(q)
        .arg_ptr(k)
        .arg_ptr(v)
        .arg_ptr(output)
        .arg_u32(seq_len)
        .arg_u32(num_q_heads)
        .arg_u32(num_kv_heads)
        .arg_u32(head_dim)
        .arg_f32(inv_sqrt_d)
        .arg_u32(if causal { 1 } else { 0 })
        .arg_u32(sliding_window)
        .launch(stream)
}

/// 2026-09-25: [`prefill_attention`] for the deepseek-v4-flash `attn_prefill_512`
/// kernel (HDIM 512, 16 query rows per block) with a per-head sink logit:
/// `sinks` is FP32 `[num_q_heads]`, added to the softmax denominator only.
/// `DevicePtr::NULL` means no sink.
#[allow(clippy::too_many_arguments)]
pub fn prefill_attention_512_sink(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    q: DevicePtr,
    k: DevicePtr,
    v: DevicePtr,
    output: DevicePtr,
    seq_len: u32,
    batch: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    inv_sqrt_d: f32,
    causal: bool,
    sliding_window: u32,
    sinks: DevicePtr,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_q_heads, div_ceil(seq_len, 16), batch])
        .block([128, 1, 1])
        .arg_ptr(q)
        .arg_ptr(k)
        .arg_ptr(v)
        .arg_ptr(output)
        .arg_u32(seq_len)
        .arg_u32(num_q_heads)
        .arg_u32(num_kv_heads)
        .arg_u32(head_dim)
        .arg_f32(inv_sqrt_d)
        .arg_u32(if causal { 1 } else { 0 })
        .arg_u32(sliding_window)
        .arg_ptr(sinks)
        .launch(stream)
}

/// 2026-09-25: [`prefill_attention`] with 64 query rows per block (32 on the
/// SCALE and HIP targets).
#[allow(clippy::too_many_arguments)]
pub fn prefill_attention_64(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    q: DevicePtr,
    k: DevicePtr,
    v: DevicePtr,
    output: DevicePtr,
    seq_len: u32,
    batch: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    inv_sqrt_d: f32,
    causal: bool,
    sliding_window: u32,
    stream: u64,
) -> Result<()> {
    // 2026-09-25: The grid's rows per block must equal the kernel's `BR64`, or
    // rows 32..63 of every 64-row band are never written. `BR64` is 32 under
    // `__SCALE__` (gb10 `attn_prefill.cu`) and `__SCALE__ || __HIP_PLATFORM_AMD__`
    // (strix-hip), for the 64 KB LDS limit; `metrale_scale` is set for exactly
    // those targets, `strix` and `strix-hip` (`crates/model-layers/build.rs`).
    let br = if cfg!(metrale_scale) { 32u32 } else { 64u32 };
    KernelLaunch::new(gpu, kernel)
        .grid([num_q_heads, div_ceil(seq_len, br), batch])
        .block([256, 1, 1])
        .arg_ptr(q)
        .arg_ptr(k)
        .arg_ptr(v)
        .arg_ptr(output)
        .arg_u32(seq_len)
        .arg_u32(num_q_heads)
        .arg_u32(num_kv_heads)
        .arg_u32(head_dim)
        .arg_f32(inv_sqrt_d)
        .arg_u32(if causal { 1 } else { 0 })
        .arg_u32(sliding_window)
        .launch(stream)
}

/// 2026-09-25: Contiguous prefill with BF16 Q and FP8 E4M3 K and V, which the
/// kernel dequantizes to BF16 in shared memory; 64 query rows per block.
#[allow(clippy::too_many_arguments)]
pub fn prefill_attention_fp8kv(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    q: DevicePtr,
    k_fp8: DevicePtr,
    v_fp8: DevicePtr,
    output: DevicePtr,
    seq_len: u32,
    batch: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    inv_sqrt_d: f32,
    causal: bool,
    stream: u64,
) -> Result<()> {
    let br = 64u32;
    KernelLaunch::new(gpu, kernel)
        .grid([num_q_heads, div_ceil(seq_len, br), batch])
        .block([256, 1, 1])
        .arg_ptr(q)
        .arg_ptr(k_fp8)
        .arg_ptr(v_fp8)
        .arg_ptr(output)
        .arg_u32(seq_len)
        .arg_u32(num_q_heads)
        .arg_u32(num_kv_heads)
        .arg_u32(head_dim)
        .arg_f32(inv_sqrt_d)
        .arg_u32(if causal { 1 } else { 0 })
        .launch(stream)
}

/// 2026-09-25: Causal flash-attention prefill of a contiguous Q chunk against
/// K and V in the paged BF16 cache, found through `block_table`. Query row `i`
/// is at absolute position `q_offset + i`; 32 query rows per block.
#[allow(clippy::too_many_arguments)]
pub fn prefill_attention_paged(
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
    stream: u64,
) -> Result<()> {
    let br = 32u32;
    KernelLaunch::new(gpu, kernel)
        .grid([num_q_heads, div_ceil(q_len, br), 1])
        .block([128, 1, 1])
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
        .launch(stream)
}

/// 2026-09-25: [`prefill_attention_paged`] over an FP8 KV cache.
#[allow(clippy::too_many_arguments)]
pub fn prefill_attention_paged_fp8(
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
    k_scale: f32,
    v_scale: f32,
    cache_stride: u64,
    stream: u64,
) -> Result<()> {
    let br = 32u32;
    KernelLaunch::new(gpu, kernel)
        .grid([num_q_heads, div_ceil(q_len, br), 1])
        .block([128, 1, 1])
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
        .arg_f32(k_scale)
        .arg_f32(v_scale)
        .arg_u64(cache_stride)
        .launch(stream)
}

/// 2026-09-25: [`prefill_attention_paged_fp8`] launched with
/// `causal_mask_enabled = 0`: the causal mask is skipped, so a query of the
/// DFlash draft block also attends to the block's later positions.
#[allow(clippy::too_many_arguments)]
pub fn prefill_attention_paged_fp8_dflash(
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
    k_scale: f32,
    v_scale: f32,
    cache_stride: u64,
    stream: u64,
) -> Result<()> {
    let br = 32u32;
    KernelLaunch::new(gpu, kernel)
        .grid([num_q_heads, div_ceil(q_len, br), 1])
        .block([128, 1, 1])
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
        .arg_u32(0u32)
        .arg_f32(inv_sqrt_d)
        .arg_f32(k_scale)
        .arg_f32(v_scale)
        .arg_u64(cache_stride)
        .launch(stream)
}

/// 2026-09-25: [`prefill_attention_paged`] launched with
/// `causal_mask_enabled = 0`, the BF16-cache counterpart of
/// [`prefill_attention_paged_fp8_dflash`].
#[allow(clippy::too_many_arguments)]
pub fn prefill_attention_paged_dflash(
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
    stream: u64,
) -> Result<()> {
    let br = 32u32;
    KernelLaunch::new(gpu, kernel)
        .grid([num_q_heads, div_ceil(q_len, br), 1])
        .block([128, 1, 1])
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
        .arg_u32(0u32)
        .arg_f32(inv_sqrt_d)
        .launch(stream)
}

/// 2026-09-25: [`prefill_attention_paged_dflash`] with `kv_len`, `q_offset` and
/// `q_rope_pos` read on the device from `kv_len_q_offset_dev` (three u32 at byte
/// offsets 0, 4 and 8) instead of passed as scalars, so a captured CUDA graph
/// can replay the launch with values the host writes before each replay.
/// `q_offset` addresses the cache; `q_rope_pos` is the absolute query position
/// the causal and sliding-window masks compare against. The kernel applies no
/// RoPE. Kernel: `attn_prefill_paged_indirect`.
#[allow(clippy::too_many_arguments)]
pub fn prefill_attention_paged_dflash_bf16_indirect(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    q: DevicePtr,
    k_cache: DevicePtr,
    v_cache: DevicePtr,
    output: DevicePtr,
    block_table: DevicePtr,
    q_len: u32,
    kv_len_q_offset_dev: DevicePtr,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    cache_block_size: u32,
    sliding_window: u32,
    inv_sqrt_d: f32,
    stream: u64,
) -> Result<()> {
    let br = 32u32;
    // 2026-09-25: The kernel's `KERNEL_PREAMBLE` overwrites its scalar `kv_len` and
    // `q_offset` from the device buffer, so zeros are passed for both.
    KernelLaunch::new(gpu, kernel)
        .grid([num_q_heads, div_ceil(q_len, br), 1])
        .block([128, 1, 1])
        .arg_ptr(q)
        .arg_ptr(k_cache)
        .arg_ptr(v_cache)
        .arg_ptr(output)
        .arg_ptr(block_table)
        .arg_u32(q_len)
        .arg_u32(0u32)
        .arg_u32(0u32)
        .arg_u32(num_q_heads)
        .arg_u32(num_kv_heads)
        .arg_u32(head_dim)
        .arg_u32(cache_block_size)
        .arg_u32(sliding_window)
        .arg_u32(0u32)
        .arg_f32(inv_sqrt_d)
        .arg_ptr(kv_len_q_offset_dev)
        .arg_ptr(kv_len_q_offset_dev.offset(4))
        .arg_ptr(kv_len_q_offset_dev.offset(8))
        .launch(stream)
}
