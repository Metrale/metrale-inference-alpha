// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Launchers for paged KV-cache writes (BF16 and FP8, and the fused K norm + RoPE writes), plus the BF16-K/Turbo-V and MLA submodules.
//!
//! Owner: model-layers ops.
//! Invariants:
//! - `slot_mapping` is an `i64` array with one entry per token, `block * block_size + offset`;
//!   every cache-write kernel launched here skips a token whose slot is negative
//!   (kernels/gb10/common/reshape_and_cache.cu, fused_k_norm_rope_cache.cu,
//!   reshape_and_cache_fused_k_fp8.cu).
//! - The paged cache layout is `[num_blocks, block_size, num_kv_heads, head_dim]`.

#![allow(unused_imports)]

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::layers::moe;
use crate::weight_map::{DenseWeight, Fp8DenseWeight, Fp8Weight, QuantizedWeight};

use super::*;
#[path = "kv_cache_bf16k_turbov.rs"]
mod bf16k_turbov;
pub use bf16k_turbov::{
    paged_decode_attn_bf16k_turbo2v, paged_decode_attn_bf16k_turbo3v,
    paged_decode_attn_bf16k_turbo4v, reshape_and_cache_bf16k_turbo2v,
    reshape_and_cache_bf16k_turbo3v, reshape_and_cache_bf16k_turbo4v,
};
#[path = "kv_cache_mla.rs"]
mod mla;
pub use mla::{
    mla_batched_gemv, mla_cache_assemble, mla_paged_decode_fp8, mla_paged_decode_nvfp4,
    mla_q_rope_scatter, mla_q_rope_writeback,
};

/// 2026-09-25: Fill `count` slot-mapping entries on the device from a block table:
/// `slots[i] = block_table[(start_pos + i) / block_size] * block_size + (start_pos + i) %
/// block_size`. `count == 0` launches nothing.
pub fn fill_slots_from_block_table(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    slots: DevicePtr,
    block_table: DevicePtr,
    start_pos: u32,
    count: u32,
    block_size: u32,
    stream: u64,
) -> Result<()> {
    if count == 0 {
        return Ok(());
    }
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(count, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(slots)
        .arg_ptr(block_table)
        .arg_u32(start_pos)
        .arg_u32(count)
        .arg_u32(block_size)
        .launch(stream)
}

/// 2026-09-25: Copy BF16 K and V rows into the paged BF16 cache at `slot_mapping` (kernel
/// `reshape_and_cache_flash`), one block per token. `key_stride` / `value_stride` are the row
/// strides in elements. `_cache_stride` is not passed to the kernel, which derives the block
/// stride from `block_size`.
#[allow(clippy::too_many_arguments)]
pub fn reshape_and_cache(
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
    _cache_stride: u64,
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
        .launch(stream)
}

/// 2026-09-25: Copy only V into the paged BF16 cache (kernel `reshape_and_cache_flash_v_only`),
/// for callers whose K is written by a `fused_k_norm_rope_cache_write_*` launch; a
/// [`reshape_and_cache`] there would overwrite that K.
#[allow(clippy::too_many_arguments)]
pub fn reshape_and_cache_flash_v_only(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    value: DevicePtr,
    v_cache: DevicePtr,
    slot_mapping: DevicePtr,
    num_tokens: u32,
    num_kv_heads: u32,
    head_dim: u32,
    block_size: u32,
    value_stride: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(value)
        .arg_ptr(v_cache)
        .arg_ptr(slot_mapping)
        .arg_u32(num_kv_heads)
        .arg_u32(head_dim)
        .arg_u32(block_size)
        .arg_u32(value_stride)
        .launch(stream)
}

/// 2026-09-25: K norm (`x * rsqrt(mean(x^2) + eps) * (1 + w)`), RoPE and the paged BF16 cache
/// write in one kernel. K stays in FP32 from the BF16 load to the single BF16 rounding at the
/// store. One block per (token, KV head) with `head_dim` threads; the kernel supports
/// `head_dim <= 256`.
#[allow(clippy::too_many_arguments)]
pub fn fused_k_norm_rope_cache_write_bf16(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    k_in: DevicePtr,
    k_norm_weight: DevicePtr,
    positions: DevicePtr,
    k_cache: DevicePtr,
    slot_mapping: DevicePtr,
    num_tokens: u32,
    num_kv_heads: u32,
    head_dim: u32,
    rotary_dim: u32,
    block_size: u32,
    rms_eps: f32,
    theta: f32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, num_kv_heads, 1])
        .block([head_dim, 1, 1])
        .arg_ptr(k_in)
        .arg_ptr(k_norm_weight)
        .arg_ptr(positions)
        .arg_ptr(k_cache)
        .arg_ptr(slot_mapping)
        .arg_u32(num_kv_heads)
        .arg_u32(head_dim)
        .arg_u32(rotary_dim)
        .arg_u32(block_size)
        .arg_f32(rms_eps)
        .arg_f32(theta)
        .launch(stream)
}

/// 2026-09-25: The interleaved-MRoPE variant (kernel `fused_k_norm_rope_mrope_cache_write_bf16`):
/// rotation pair `i` takes its position from `pos_t`, `pos_h` or `pos_w` by `i % 3`. The rest of
/// the kernel is the scalar-position one, so equal position streams give its results.
#[allow(clippy::too_many_arguments)]
pub fn fused_k_norm_rope_cache_write_bf16_mrope(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    k_in: DevicePtr,
    k_norm_weight: DevicePtr,
    pos_t: DevicePtr,
    pos_h: DevicePtr,
    pos_w: DevicePtr,
    k_cache: DevicePtr,
    slot_mapping: DevicePtr,
    num_tokens: u32,
    num_kv_heads: u32,
    head_dim: u32,
    rotary_dim: u32,
    block_size: u32,
    rms_eps: f32,
    theta: f32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, num_kv_heads, 1])
        .block([head_dim, 1, 1])
        .arg_ptr(k_in)
        .arg_ptr(k_norm_weight)
        .arg_ptr(pos_t)
        .arg_ptr(pos_h)
        .arg_ptr(pos_w)
        .arg_ptr(k_cache)
        .arg_ptr(slot_mapping)
        .arg_u32(num_kv_heads)
        .arg_u32(head_dim)
        .arg_u32(rotary_dim)
        .arg_u32(block_size)
        .arg_f32(rms_eps)
        .arg_f32(theta)
        .launch(stream)
}

/// 2026-09-25: [`fused_k_norm_rope_cache_write_bf16`] writing an FP8 E4M3 cache. The FP32 result
/// times `inv_scale` (`1 / k_scale`) goes through one saturating FP8 cast, with no BF16 rounding.
#[allow(clippy::too_many_arguments)]
pub fn fused_k_norm_rope_cache_write_fp8(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    k_in: DevicePtr,
    k_norm_weight: DevicePtr,
    positions: DevicePtr,
    k_cache_fp8: DevicePtr,
    slot_mapping: DevicePtr,
    num_tokens: u32,
    num_kv_heads: u32,
    head_dim: u32,
    rotary_dim: u32,
    block_size: u32,
    rms_eps: f32,
    theta: f32,
    inv_scale: f32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, num_kv_heads, 1])
        .block([head_dim, 1, 1])
        .arg_ptr(k_in)
        .arg_ptr(k_norm_weight)
        .arg_ptr(positions)
        .arg_ptr(k_cache_fp8)
        .arg_ptr(slot_mapping)
        .arg_u32(num_kv_heads)
        .arg_u32(head_dim)
        .arg_u32(rotary_dim)
        .arg_u32(block_size)
        .arg_f32(rms_eps)
        .arg_f32(theta)
        .arg_f32(inv_scale)
        .launch(stream)
}

/// 2026-09-25: Decode-path fusion of `rms_norm` + `rope_forward` + `reshape_and_cache_flash_fp8`
/// for an FP8 KV cache: writes K (normed, rotated, quantized) and V (quantized) in one launch,
/// leaving `rope_forward` for Q only.
///
/// The kernel rounds to BF16 at each point where that unfused chain stores BF16, rather than
/// keeping FP32 intermediates, so that it writes the chain's cache bytes (see the header of
/// kernels/gb10/common/reshape_and_cache_fused_k_fp8.cu). Removing a rounding step would change
/// those bytes.
///
/// `k_scale` / `v_scale` are dequant scales; the kernel computes `1.0f / scale` itself, as
/// `reshape_and_cache_flash_fp8` does. `head_dim` threads per block, one block per (token, KV
/// head).
#[allow(clippy::too_many_arguments)]
pub fn fused_k_norm_rope_cache_write_fp8_kv(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    k_in: DevicePtr,
    value: DevicePtr,
    k_norm_weight: DevicePtr,
    positions: DevicePtr,
    k_cache: DevicePtr,
    v_cache: DevicePtr,
    slot_mapping: DevicePtr,
    num_tokens: u32,
    num_kv_heads: u32,
    head_dim: u32,
    rotary_dim: u32,
    block_size: u32,
    k_scale: f32,
    v_scale: f32,
    key_stride: u32,
    value_stride: u32,
    cache_stride: u64,
    rms_eps: f32,
    theta: f32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, num_kv_heads, 1])
        .block([head_dim, 1, 1])
        .arg_ptr(k_in)
        .arg_ptr(value)
        .arg_ptr(k_norm_weight)
        .arg_ptr(positions)
        .arg_ptr(k_cache)
        .arg_ptr(v_cache)
        .arg_ptr(slot_mapping)
        .arg_u32(num_kv_heads)
        .arg_u32(head_dim)
        .arg_u32(rotary_dim)
        .arg_u32(block_size)
        .arg_f32(k_scale)
        .arg_f32(v_scale)
        .arg_u32(key_stride)
        .arg_u32(value_stride)
        .arg_u64(cache_stride)
        .arg_f32(rms_eps)
        .arg_f32(theta)
        .launch(stream)
}

/// 2026-09-25: Quantize BF16 K and V rows into the paged FP8 cache (kernel
/// `reshape_and_cache_flash_fp8`): `fp8 = sat(bf16 / scale)` with the dequant scales
/// `k_scale` / `v_scale`. `k_cache` / `v_cache` are the pool base pointers, and `cache_stride` is
/// the stride between cache blocks in elements (normally `block_size * num_kv_heads * head_dim`).
pub fn reshape_and_cache_fp8(
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
    k_scale: f32,
    v_scale: f32,
    key_stride: u32,
    value_stride: u32,
    cache_stride: u64,
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
        .arg_f32(k_scale)
        .arg_f32(v_scale)
        .arg_u32(key_stride)
        .arg_u32(value_stride)
        .arg_u64(cache_stride)
        .launch(stream)
}
