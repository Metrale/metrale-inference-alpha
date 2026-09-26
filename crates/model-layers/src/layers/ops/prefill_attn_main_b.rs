// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Launchers for the paged prefill flash-attention kernels over
//! NVFP4, HDIM=512 BF16 and TurboQuant KV caches, and the 64-row BF16, FP8 and
//! NVFP4 variants. The BF16-K / packed-V launchers live in
//! `prefill_attn_main_b_turbov.rs`.
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
#[path = "prefill_attn_main_b_turbov.rs"]
mod bf16k_turbov;
pub use bf16k_turbov::{
    prefill_attention_paged_bf16k_turbo2v_64, prefill_attention_paged_bf16k_turbo3v_64,
    prefill_attention_paged_bf16k_turbo4v_64,
};

/// 2026-09-25: Causal paged prefill over an NVFP4 KV cache, 32 query rows per
/// block.
#[allow(clippy::too_many_arguments)]
pub fn prefill_attention_paged_nvfp4(
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
    block_stride_bytes: u64,
    data_section_bytes: u64,
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
        .arg_u64(block_stride_bytes)
        .arg_u64(data_section_bytes)
        .launch(stream)
}

/// 2026-09-25: Causal paged prefill for `head_dim == 512` over a BF16 KV cache:
/// 32 query rows and 8 warps per block, with the 101,120 bytes of dynamic shared
/// memory `attn_prefill_paged_512.cu` requires.
#[allow(clippy::too_many_arguments)]
pub fn prefill_attention_paged_512(
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
        .block([256, 1, 1])
        .shared_mem(101_120)
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

/// 2026-09-25: [`prefill_attention_paged`] with 64 query rows per block (32 on
/// the SCALE and HIP targets).
#[allow(clippy::too_many_arguments)]
pub fn prefill_attention_paged_64(
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
    // 2026-09-25: The grid's rows per block must equal the kernel's `BR64`, or rows
    // 32..63 of every 64-row band are never written. `BR64` is 32 in the gb10
    // `prefill_paged_compute.cuh` under `__SCALE__` and always in the strix-hip
    // one; `metrale_scale` is set for exactly `strix` and `strix-hip`.
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
        .launch(stream)
}

/// 2026-09-25: [`prefill_attention_paged_fp8`] with 64 query rows per block (32
/// on the SCALE and HIP targets).
#[allow(clippy::too_many_arguments)]
pub fn prefill_attention_paged_fp8_64(
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
    // 2026-09-25: The grid's rows per block must equal the kernel's `BR64`, or rows
    // 32..63 of every 64-row band are never written. `BR64` is 32 in the gb10
    // `prefill_paged_compute.cuh` under `__SCALE__` and always in the strix-hip
    // one; `metrale_scale` is set for exactly `strix` and `strix-hip`.
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
        .arg_f32(k_scale)
        .arg_f32(v_scale)
        .arg_u64(cache_stride)
        .launch(stream)
}

/// 2026-09-25: Causal paged prefill over a TurboQuant KV cache (same format for K
/// and V), 64 query rows per block (32 on the SCALE and HIP targets). The
/// turbo8, turbo4 and turbo3 `_64` entries share this argument list; the caller
/// picks one through `kernel` and passes that cache's block stride and
/// data-section size.
#[allow(clippy::too_many_arguments)]
pub fn prefill_attention_paged_turbo_64(
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
    block_stride_bytes: u64,
    data_section_bytes: u64,
    stream: u64,
) -> Result<()> {
    // 2026-09-25: The grid's rows per block must equal the kernel's `BR64`, or rows
    // 32..63 of every 64-row band are never written. `BR64` is 32 in the gb10
    // `prefill_paged_compute.cuh` under `__SCALE__` and always in the strix-hip
    // one; `metrale_scale` is set for exactly `strix` and `strix-hip`.
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
        .arg_u64(block_stride_bytes)
        .arg_u64(data_section_bytes)
        .launch(stream)
}

/// 2026-09-25: The turbo2 counterpart of [`prefill_attention_paged_turbo_64`].
pub fn prefill_attention_paged_turbo2_64(
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
    block_stride_bytes: u64,
    data_section_bytes: u64,
    stream: u64,
) -> Result<()> {
    // 2026-09-26: The handle is the 32-row entry `attn_prefill_paged_turbo2`
    // (`qwen3_attention/init_prefill_kernels.rs`), so the grid uses 32 rows and 128
    // threads like the other 32-row paged launchers.
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
        .arg_u64(block_stride_bytes)
        .arg_u64(data_section_bytes)
        .launch(stream)
}

/// 2026-09-25: [`prefill_attention_paged_nvfp4`] with 64 query rows per block
/// (32 on the SCALE and HIP targets).
pub fn prefill_attention_paged_nvfp4_64(
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
    block_stride_bytes: u64,
    data_section_bytes: u64,
    stream: u64,
) -> Result<()> {
    // 2026-09-25: The grid's rows per block must equal the kernel's `BR64`, or rows
    // 32..63 of every 64-row band are never written. `BR64` is 32 in the gb10
    // `prefill_paged_compute.cuh` under `__SCALE__` and always in the strix-hip
    // one; `metrale_scale` is set for exactly `strix` and `strix-hip`.
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
        .arg_u64(block_stride_bytes)
        .arg_u64(data_section_bytes)
        .launch(stream)
}
