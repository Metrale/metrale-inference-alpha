// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Launchers for the MLA decode kernels: per-head batched GEMV, Q RoPE scatter and writeback, cache-entry assembly, and the NVFP4 and FP8 paged decode.
//!
//! Owner: model-layers ops.
//! Invariants:
//! - The scatter, writeback and assembly launchers run a single block, so they handle one
//!   token (kernels/gb10/deepseek-v4-flash/nvfp4/mla_absorbed.cu).
//! - The paged decode launchers run one 256-thread block per (query head, sequence) and pass
//!   their kernel's parameters in declaration order.

use super::*;

/// 2026-09-25: MLA batched GEMV over all heads in one launch:
/// `output[head, n] = sum_k(weight[head, n, k] * input[head, k])`, BF16, with the per-head input
/// and output at `input_stride` / `output_stride` and the weight contiguous per head.
pub fn mla_batched_gemv(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weight: DevicePtr,
    output: DevicePtr,
    n_out: u32,
    k: u32,
    num_heads: u32,
    input_stride: u32,
    output_stride: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(n_out, 8), num_heads, 1]) // 2026-09-25: N_PER_BLOCK * 2 = 8 outputs per block.
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight)
        .arg_ptr(output)
        .arg_u32(n_out)
        .arg_u32(k)
        .arg_u32(input_stride)
        .arg_u32(output_stride)
        .launch(stream)
}

/// 2026-09-25: Copy each head's rope part of `q_full` (`[nq, hd]`, at offset `nope`) into
/// `q_absorbed_buf` (`[nq, mla_cache_dim]`, at offset `kv_lora`) and into the contiguous
/// `q_rope_contiguous` (`[nq, rope]`) in one pass.
#[allow(clippy::too_many_arguments)]
pub fn mla_q_rope_scatter(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    q_full: DevicePtr,
    q_absorbed_buf: DevicePtr,
    q_rope_contiguous: DevicePtr,
    nq: u32,
    hd: u32,
    nope: u32,
    rope: u32,
    kv_lora: u32,
    mla_cache_dim: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([1, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(q_full)
        .arg_ptr(q_absorbed_buf)
        .arg_ptr(q_rope_contiguous)
        .arg_u32(nq)
        .arg_u32(hd)
        .arg_u32(nope)
        .arg_u32(rope)
        .arg_u32(kv_lora)
        .arg_u32(mla_cache_dim)
        .launch(stream)
}

/// 2026-09-25: Write the rotated `[nq, rope]` Q rope parts back into `q_absorbed_buf` at offset
/// `kv_lora` of each head's `mla_cache_dim` row.
pub fn mla_q_rope_writeback(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    q_rope_direct: DevicePtr,
    q_absorbed_buf: DevicePtr,
    nq: u32,
    rope: u32,
    kv_lora: u32,
    mla_cache_dim: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([1, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(q_rope_direct)
        .arg_ptr(q_absorbed_buf)
        .arg_u32(nq)
        .arg_u32(rope)
        .arg_u32(kv_lora)
        .arg_u32(mla_cache_dim)
        .launch(stream)
}

/// 2026-09-25: Assemble one token's cache entries: K = `[kv_latent | k_rope]` and
/// V = `[kv_latent | zeros]`, each `mla_cache_dim` wide. The block has one thread per element
/// (`max(mla_cache_dim, 256)` threads).
pub fn mla_cache_assemble(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    kv_latent: DevicePtr,
    k_rope: DevicePtr,
    k_cache: DevicePtr,
    v_cache: DevicePtr,
    kv_lora: u32,
    rope: u32,
    mla_cache_dim: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([1, 1, 1])
        .block([mla_cache_dim.max(256), 1, 1])
        .arg_ptr(kv_latent)
        .arg_ptr(k_rope)
        .arg_ptr(k_cache)
        .arg_ptr(v_cache)
        .arg_u32(kv_lora)
        .arg_u32(rope)
        .arg_u32(mla_cache_dim)
        .launch(stream)
}

/// 2026-09-25: MLA paged decode for DeepSeek-V4-Flash over an NVFP4 cache (kernel
/// `mla_paged_decode_nvfp4`). A cache token is `kv_lora_rank + qk_rope_head_dim` = 512 + 64 = 576
/// values (`MLA_CACHE_DIM` in the kernel); each block holds packed E2M1 data
/// (`data_section_bytes`) followed by one FP8 E4M3 scale per 16 values.
#[allow(clippy::too_many_arguments)]
pub fn mla_paged_decode_nvfp4(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    q: DevicePtr,
    k_cache: DevicePtr,
    v_cache: DevicePtr,
    o: DevicePtr,
    block_tables: DevicePtr,
    seq_lens: DevicePtr,
    max_blocks_per_seq: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    q_head_dim: u32,
    kv_cache_dim: u32,
    block_size: u32,
    inv_sqrt_d: f32,
    block_stride_bytes: u64,
    data_section_bytes: u64,
    num_seqs: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_q_heads, num_seqs, 1])
        .block([256, 1, 1])
        .arg_ptr(q)
        .arg_ptr(k_cache)
        .arg_ptr(v_cache)
        .arg_ptr(o)
        .arg_ptr(block_tables)
        .arg_ptr(seq_lens)
        .arg_u32(max_blocks_per_seq)
        .arg_u32(num_q_heads)
        .arg_u32(num_kv_heads)
        .arg_u32(q_head_dim)
        .arg_u32(kv_cache_dim)
        .arg_u32(block_size)
        .arg_f32(inv_sqrt_d)
        .arg_u64(block_stride_bytes)
        .arg_u64(data_section_bytes)
        .launch(stream)
}

/// 2026-09-25: MLA paged decode for DeepSeek-V4-Flash over an FP8 cache (kernel
/// `mla_paged_decode_fp8`), with scalar dequant scales `k_scale` / `v_scale` and `cache_stride`
/// in bytes. `sinks` (per-head attention sinks, FP32) may be null. The compressed-KV pool is
/// attended only when `comp_pool` is non-null and `comp_block_count > 0`.
#[allow(clippy::too_many_arguments)]
pub fn mla_paged_decode_fp8(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    q: DevicePtr,
    k_cache: DevicePtr,
    v_cache: DevicePtr,
    o: DevicePtr,
    block_tables: DevicePtr,
    seq_lens: DevicePtr,
    max_blocks_per_seq: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    q_head_dim: u32,
    kv_cache_dim: u32,
    block_size: u32,
    inv_sqrt_d: f32,
    k_scale: f32,
    v_scale: f32,
    cache_stride: u64,
    num_seqs: u32,
    sliding_window: u32,
    sinks: DevicePtr,
    comp_pool: DevicePtr,
    comp_block_count: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_q_heads, num_seqs, 1])
        .block([256, 1, 1])
        .arg_ptr(q)
        .arg_ptr(k_cache)
        .arg_ptr(v_cache)
        .arg_ptr(o)
        .arg_ptr(block_tables)
        .arg_ptr(seq_lens)
        .arg_u32(max_blocks_per_seq)
        .arg_u32(num_q_heads)
        .arg_u32(num_kv_heads)
        .arg_u32(q_head_dim)
        .arg_u32(kv_cache_dim)
        .arg_u32(block_size)
        .arg_f32(inv_sqrt_d)
        .arg_f32(k_scale)
        .arg_f32(v_scale)
        .arg_u64(cache_stride)
        .arg_u32(sliding_window)
        .arg_ptr(sinks)
        .arg_ptr(comp_pool)
        .arg_u32(comp_block_count)
        .launch(stream)
}
