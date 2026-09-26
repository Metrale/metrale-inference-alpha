// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Launchers for the MLA prefill helpers (rope extract and writeback,
//! K/V and cache assembly, fused prefill, grouped GEMM, absorbed attention) and
//! the paged decode attention kernels, unpacked and GQA-packed.
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

/// 2026-09-25: Copy each head's rope slice (offset `nope` in the head's `hd`) out
/// of Q, rows `q_dim` apart, into contiguous `[num_tokens, nq, rope]`.
#[allow(clippy::too_many_arguments)]
pub fn mla_q_rope_extract_batched(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    q_full: DevicePtr,
    q_rope_out: DevicePtr,
    num_tokens: u32,
    nq: u32,
    hd: u32,
    nope: u32,
    rope: u32,
    q_dim: u32,
    stream: u64,
) -> Result<()> {
    let total = num_tokens * nq * rope;
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(total, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(q_full)
        .arg_ptr(q_rope_out)
        .arg_u32(num_tokens)
        .arg_u32(nq)
        .arg_u32(hd)
        .arg_u32(nope)
        .arg_u32(rope)
        .arg_u32(q_dim)
        .launch(stream)
}

/// 2026-09-25: Inverse of [`mla_q_rope_extract_batched`]: write `[num_tokens, nq,
/// rope]` back into each head's rope slice of Q.
#[allow(clippy::too_many_arguments)]
pub fn mla_q_rope_writeback_batched(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    q_rope_in: DevicePtr,
    q_full: DevicePtr,
    num_tokens: u32,
    nq: u32,
    hd: u32,
    nope: u32,
    rope: u32,
    q_dim: u32,
    stream: u64,
) -> Result<()> {
    let total = num_tokens * nq * rope;
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(total, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(q_rope_in)
        .arg_ptr(q_full)
        .arg_u32(num_tokens)
        .arg_u32(nq)
        .arg_u32(hd)
        .arg_u32(nope)
        .arg_u32(rope)
        .arg_u32(q_dim)
        .launch(stream)
}

/// 2026-09-25: Per token, build K as `[k_nope | k_rope]` for each KV head (k_rope
/// shared by all heads) and extract V from `kv_expanded`. `blockIdx.y` 0 builds
/// K, 1 extracts V.
#[allow(clippy::too_many_arguments)]
pub fn mla_kv_assemble_batched(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    kv_expanded: DevicePtr,
    k_rope_buf: DevicePtr,
    k_out: DevicePtr,
    v_out: DevicePtr,
    num_tokens: u32,
    nkv: u32,
    nope: u32,
    v_dim: u32,
    rope: u32,
    hd: u32,
    kv_expanded_stride: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 2, 1])
        .block([256, 1, 1])
        .arg_ptr(kv_expanded)
        .arg_ptr(k_rope_buf)
        .arg_ptr(k_out)
        .arg_ptr(v_out)
        .arg_u32(nkv)
        .arg_u32(nope)
        .arg_u32(v_dim)
        .arg_u32(rope)
        .arg_u32(hd)
        .arg_u32(kv_expanded_stride)
        .launch(stream)
}

/// 2026-09-26: Per token, write the compressed MLA cache rows: K = `[kv_latent |
/// k_rope]`, and V = `[kv_latent | zeros]` in the mistral-small-4 kernel or
/// `[kv_latent | k_rope]` in the deepseek-v4-flash kernel.
#[allow(clippy::too_many_arguments)]
pub fn mla_cache_assemble_batched(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    kv_latent: DevicePtr,
    k_rope: DevicePtr,
    k_cache: DevicePtr,
    v_cache: DevicePtr,
    num_tokens: u32,
    kv_lora: u32,
    rope: u32,
    mla_cache_dim: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
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

/// 2026-09-25: MLA prefill in one kernel: Q absorption through `w_uk`, attention
/// over the latent cache, and V extraction through `w_uv`. One block per
/// `(head, query token)`.
#[allow(clippy::too_many_arguments)]
pub fn mla_fused_prefill(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    q_full: DevicePtr,
    q_rope: DevicePtr,
    kv_latent: DevicePtr,
    k_rope: DevicePtr,
    w_uk: DevicePtr,
    w_uv: DevicePtr,
    v_out: DevicePtr,
    k_cache_out: DevicePtr,
    v_cache_out: DevicePtr,
    seq_len: u32,
    nq: u32,
    nope: u32,
    rope: u32,
    kv_lora: u32,
    v_dim: u32,
    hd: u32,
    num_kv_heads: u32,
    inv_sqrt_d: f32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([nq, seq_len, 1])
        .block([256, 1, 1])
        .arg_ptr(q_full)
        .arg_ptr(q_rope)
        .arg_ptr(kv_latent)
        .arg_ptr(k_rope)
        .arg_ptr(w_uk)
        .arg_ptr(w_uv)
        .arg_ptr(v_out)
        .arg_ptr(k_cache_out)
        .arg_ptr(v_cache_out)
        .arg_u32(seq_len)
        .arg_u32(nq)
        .arg_u32(nope)
        .arg_u32(rope)
        .arg_u32(kv_lora)
        .arg_u32(v_dim)
        .arg_u32(hd)
        .arg_u32(num_kv_heads)
        .arg_f32(inv_sqrt_d)
        .launch(stream)
}

/// 2026-09-25: Build Q as `[q_absorbed (kv_lora) | q_rope (rope)]` per head and
/// token, `mla_cache_dim = kv_lora + rope` wide.
#[allow(clippy::too_many_arguments)]
pub fn mla_q_final_assemble_batched(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    q_absorbed: DevicePtr,
    q_rope: DevicePtr,
    q_final: DevicePtr,
    num_tokens: u32,
    nq: u32,
    kv_lora: u32,
    rope: u32,
    mla_cache_dim: u32,
    stream: u64,
) -> Result<()> {
    let total = num_tokens * nq * mla_cache_dim;
    KernelLaunch::new(gpu, kernel)
        .grid([div_ceil(total, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(q_absorbed)
        .arg_ptr(q_rope)
        .arg_ptr(q_final)
        .arg_u32(num_tokens)
        .arg_u32(nq)
        .arg_u32(kv_lora)
        .arg_u32(rope)
        .arg_u32(mla_cache_dim)
        .launch(stream)
}

/// 2026-09-25: `g` independent GEMMs `C_g[m, n_g] = A_g[m, k_g] @ B_g[n_g, k_g]^T`
/// in one launch. `A_g` sits at column `g * k_g` of A, `B_g` at row `g * n_g` of
/// B, and `C_g` at column `g * n_g` of C; one block per `(row, group)` computes
/// 4 outputs.
#[allow(clippy::too_many_arguments)]
pub fn grouped_gemm_mla(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    a: DevicePtr,
    b: DevicePtr,
    c: DevicePtr,
    m: u32,
    g: u32,
    k_g: u32,
    n_g: u32,
    a_stride: u32,
    c_stride: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([m * g, div_ceil(n_g, 4), 1])
        .block([256, 1, 1])
        .arg_ptr(a)
        .arg_ptr(b)
        .arg_ptr(c)
        .arg_u32(m)
        .arg_u32(g)
        .arg_u32(k_g)
        .arg_u32(n_g)
        .arg_u32(a_stride)
        .arg_u32(c_stride)
        .launch(stream)
}

/// 2026-09-25: Absorbed MLA prefill attention with scalar BF16 dot products and a
/// single KV head. The head dim is the kernel's `MLA_HDIM`: 320 in the
/// mistral-small-4 fork, 576 in the deepseek-v4-flash one; both keep the entry
/// name `mla_prefill_attn_320`.
#[allow(clippy::too_many_arguments)]
pub fn mla_prefill_attention_320(
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
    stream: u64,
) -> Result<()> {
    let br = 16u32; // 2026-09-25: `MLA_BR` in both `mla_prefill_attn.cu` forks.
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
        .launch(stream)
}

/// 2026-09-25: Paged BF16-KV decode attention, one block per `(q_head, seq)`.
/// `sliding_window == 0` attends to the whole sequence; otherwise only the last
/// `sliding_window` positions.
pub fn paged_decode_attn_bf16(
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
        .arg_u32(sliding_window)
        .launch(stream)
}

pub fn paged_decode_attn_fp8(
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
    k_scale: f32,
    v_scale: f32,
    q_stride: u32,
    cache_stride: u64,
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
        .arg_f32(k_scale)
        .arg_f32(v_scale)
        .arg_u32(q_stride)
        .arg_u64(cache_stride)
        .arg_u32(sliding_window)
        .launch(stream)
}

/// 2026-09-25: GQA-packed BF16-KV paged decode, one block per `(kv_head, seq)`.
/// Same arguments, argument order and block as [`paged_decode_attn_bf16`]; only
/// the grid's x extent changes, to `num_kv_heads`, because the kernel
/// (`kernels/gb10/common/paged_decode_attn_bf16_gqa.cu`) keeps a KV head's
/// whole query group in registers.
///
/// Preconditions: `num_q_heads == num_kv_heads * DECODE_GQA_PACK_WIDTH` and
/// `head_dim == DECODE_GQA_PACK_HEAD_DIM`. The kernel derives its heads as
/// `kv_head * PD_GQA + h` and sizes its register arrays from `PD_GQA`. The
/// caller checks both with `metrale_kernels::attn_splitk::gqa_pack_shape_ok`
/// and uses the unpacked kernel when they fail.
#[allow(clippy::too_many_arguments)]
pub fn paged_decode_attn_bf16_gqa(
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
    sliding_window: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_kv_heads, num_seqs, 1])
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
        .arg_u32(sliding_window)
        .launch(stream)
}

/// 2026-09-25: GQA-packed FP8-KV paged decode, one block per `(kv_head, seq)`:
/// the arguments and order of [`paged_decode_attn_fp8`], grid x extent
/// `num_kv_heads`, and the preconditions of [`paged_decode_attn_bf16_gqa`].
#[allow(clippy::too_many_arguments)]
pub fn paged_decode_attn_fp8_gqa(
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
    k_scale: f32,
    v_scale: f32,
    q_stride: u32,
    cache_stride: u64,
    sliding_window: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_kv_heads, num_seqs, 1])
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
        .arg_f32(k_scale)
        .arg_f32(v_scale)
        .arg_u32(q_stride)
        .arg_u64(cache_stride)
        .arg_u32(sliding_window)
        .launch(stream)
}
