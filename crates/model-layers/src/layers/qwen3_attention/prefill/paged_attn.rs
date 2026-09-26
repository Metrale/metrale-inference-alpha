// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The single-stream attention dispatch of
//! `prefill_attention_paged`: for head_dim > 256, the contiguous kernel at
//! `seq_len_start == 0` or the paged BF16 kernel after it; otherwise the paged
//! kernel for the cache dtype, using the BR=64 variant when `n >= 256` for
//! BF16, NVFP4 and FP8.
//!
//! Owner: model-layers (attention).
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_cache::kv_cache::{KvCacheDtype, PagedKvCache};
use metrale_gpu_runtime::gpu::DevicePtr;

use super::super::Qwen3AttentionLayer;
use crate::layer::{AttnMetadataDev, ForwardContext};
use crate::layers::ops;

#[allow(clippy::too_many_arguments)]
#[allow(dead_code)]
pub(super) struct PagedAttnArgs<'a> {
    pub q_contiguous: DevicePtr,
    pub k_contiguous: DevicePtr,
    pub v_contiguous: DevicePtr,
    pub attn_out: DevicePtr,
    pub n: u32,
    pub seq_len_start: usize,
    pub num_tokens: usize,
    pub nq: u32,
    pub nkv: u32,
    pub hd: u32,
    pub bs: usize,
    pub bf16: usize,
    pub inv_sqrt_d: f32,
    pub kv_len: u32,
    pub meta: &'a AttnMetadataDev,
    pub block_table: &'a Vec<u32>,
    pub disk_block_ids: &'a mut Vec<u32>,
    pub disk_last_offloaded_per_layer: &'a mut Vec<u32>,
    pub stream: u64,
}

/// 2026-09-25: What the caller does next: on `Continue` it runs the gates and
/// the O projection; on `EarlyReturn` it returns the pointer as the layer
/// output. `prefill_attention_paged_attn` returns only `Continue`.
#[allow(dead_code)]
pub(super) enum PagedAttnOutcome {
    EarlyReturn(DevicePtr),
    Continue,
}

impl Qwen3AttentionLayer {
    /// 2026-09-25: Runs attention into `attn_out` and returns `Continue`.
    pub(super) fn prefill_attention_paged_attn(
        &self,
        kv_cache: &mut PagedKvCache,
        ctx: &ForwardContext,
        args: &mut PagedAttnArgs,
    ) -> Result<PagedAttnOutcome> {
        let PagedAttnArgs {
            q_contiguous,
            k_contiguous: _,
            v_contiguous: _,
            attn_out,
            n,
            seq_len_start,
            num_tokens: _,
            nq,
            nkv,
            hd,
            bs,
            bf16: _,
            inv_sqrt_d,
            kv_len,
            meta,
            block_table: _,
            ref mut disk_block_ids,
            ref mut disk_last_offloaded_per_layer,
            stream,
        } = *args;
        let _ = &disk_block_ids;
        let _ = &disk_last_offloaded_per_layer;

        let bs_u = bs as u32;

        // 2026-09-25: head_dim > 256: contiguous K/V at `seq_len_start == 0`,
        // the paged BF16 kernel after it.
        if hd > 256 && self.prefill_attn_512_k.0 != 0 && seq_len_start == 0 {
            ops::prefill_attention(
                ctx.gpu,
                self.prefill_attn_512_k,
                q_contiguous,
                args.k_contiguous,
                args.v_contiguous,
                attn_out,
                n,
                1,
                nq,
                nkv,
                hd,
                inv_sqrt_d,
                true,
                self.sliding_window.unwrap_or(0),
                stream,
            )?;
        } else if hd > 256 && seq_len_start > 0 {
            if self.kv_dtype != KvCacheDtype::Bf16 {
                anyhow::bail!(
                    "Gemma-4 HDIM=512 chunked prefill only supports BF16 KV cache \
                     (layer {}, seq_len_start={}, kv_dtype={:?}).",
                    self.attn_layer_idx,
                    seq_len_start,
                    self.kv_dtype
                );
            }
            if self.prefill_attn_paged_512_k.0 == 0 {
                anyhow::bail!(
                    "Gemma-4 HDIM=512 paged prefill kernel not loaded \
                     (attn_prefill_paged_512). Rebuild required."
                );
            }
            ops::prefill_attention_paged_512(
                ctx.gpu,
                self.prefill_attn_paged_512_k,
                q_contiguous,
                kv_cache.k_pool_ptr(self.attn_layer_idx),
                kv_cache.v_pool_ptr(self.attn_layer_idx),
                attn_out,
                meta.block_table,
                n,
                kv_len,
                seq_len_start as u32,
                nq,
                nkv,
                hd,
                bs_u,
                self.sliding_window.unwrap_or(0),
                inv_sqrt_d,
                stream,
            )?;
        } else {
            let use_br64 = n >= 256;
            let (fp8_k_scale, fp8_v_scale) = self.effective_fp8_scales();
            match (self.kv_dtype, use_br64) {
                (KvCacheDtype::Nvfp4, true) => ops::prefill_attention_paged_nvfp4_64(
                    ctx.gpu,
                    self.prefill_attn_paged_nvfp4_64_k,
                    q_contiguous,
                    kv_cache.k_pool_ptr(self.attn_layer_idx),
                    kv_cache.v_pool_ptr(self.attn_layer_idx),
                    attn_out,
                    meta.block_table,
                    n,
                    kv_len,
                    seq_len_start as u32,
                    nq,
                    nkv,
                    hd,
                    bs_u,
                    self.sliding_window.unwrap_or(0),
                    inv_sqrt_d,
                    kv_cache.block_stride_bytes_for_layer(self.attn_layer_idx) as u64,
                    kv_cache.nvfp4_data_bytes() as u64,
                    stream,
                )?,
                (KvCacheDtype::Bf16KTurbo3V, _) => {
                    // 2026-09-25: The asymmetric turbo arms match on the dtype
                    // alone, so one kernel serves every chunk length.
                    if self.prefill_attn_paged_bf16k_turbo3v_64_k.0 == 0 {
                        anyhow::bail!(
                            "Bf16KTurbo3V prefill kernel not loaded (layer {}); rebuild kernels.",
                            self.attn_layer_idx
                        );
                    }
                    ops::prefill_attention_paged_bf16k_turbo3v_64(
                        ctx.gpu,
                        self.prefill_attn_paged_bf16k_turbo3v_64_k,
                        q_contiguous,
                        kv_cache.k_pool_ptr(self.attn_layer_idx),
                        kv_cache.v_pool_ptr(self.attn_layer_idx),
                        attn_out,
                        meta.block_table,
                        n,
                        kv_len,
                        seq_len_start as u32,
                        nq,
                        nkv,
                        hd,
                        bs_u,
                        self.sliding_window.unwrap_or(0),
                        inv_sqrt_d,
                        kv_cache.v_block_stride_bytes_for_layer(self.attn_layer_idx) as u64,
                        kv_cache.turbo3_data_bytes() as u64,
                        stream,
                    )?
                }
                (KvCacheDtype::Bf16KTurbo4V, _) => {
                    if self.prefill_attn_paged_bf16k_turbo4v_64_k.0 == 0 {
                        anyhow::bail!(
                            "Bf16KTurbo4V prefill kernel not loaded (layer {}); rebuild kernels.",
                            self.attn_layer_idx
                        );
                    }
                    ops::prefill_attention_paged_bf16k_turbo4v_64(
                        ctx.gpu,
                        self.prefill_attn_paged_bf16k_turbo4v_64_k,
                        q_contiguous,
                        kv_cache.k_pool_ptr(self.attn_layer_idx),
                        kv_cache.v_pool_ptr(self.attn_layer_idx),
                        attn_out,
                        meta.block_table,
                        n,
                        kv_len,
                        seq_len_start as u32,
                        nq,
                        nkv,
                        hd,
                        bs_u,
                        self.sliding_window.unwrap_or(0),
                        inv_sqrt_d,
                        kv_cache.v_block_stride_bytes_for_layer(self.attn_layer_idx) as u64,
                        kv_cache.nvfp4_data_bytes() as u64,
                        stream,
                    )?
                }
                (KvCacheDtype::Bf16KTurbo2V, _) => {
                    if self.prefill_attn_paged_bf16k_turbo2v_64_k.0 == 0 {
                        anyhow::bail!(
                            "Bf16KTurbo2V prefill kernel not loaded (layer {}); rebuild kernels.",
                            self.attn_layer_idx
                        );
                    }
                    ops::prefill_attention_paged_bf16k_turbo2v_64(
                        ctx.gpu,
                        self.prefill_attn_paged_bf16k_turbo2v_64_k,
                        q_contiguous,
                        kv_cache.k_pool_ptr(self.attn_layer_idx),
                        kv_cache.v_pool_ptr(self.attn_layer_idx),
                        attn_out,
                        meta.block_table,
                        n,
                        kv_len,
                        seq_len_start as u32,
                        nq,
                        nkv,
                        hd,
                        bs_u,
                        self.sliding_window.unwrap_or(0),
                        inv_sqrt_d,
                        kv_cache.v_block_stride_bytes_for_layer(self.attn_layer_idx) as u64,
                        kv_cache.turbo2_data_bytes() as u64,
                        stream,
                    )?
                }
                (KvCacheDtype::Turbo4KTurbo3V, _)
                | (KvCacheDtype::Turbo4KTurbo8V, _)
                | (KvCacheDtype::Turbo3KTurbo8V, _) => {
                    // 2026-09-25: Both sides turbo: `prefill_turbok_turbo_v`
                    // (`paged_attn_turbok.rs`) picks the kernel.
                    self.prefill_turbok_turbo_v(
                        ctx,
                        kv_cache,
                        q_contiguous,
                        attn_out,
                        meta.block_table,
                        n,
                        kv_len,
                        seq_len_start,
                        nq,
                        nkv,
                        hd,
                        bs_u,
                        inv_sqrt_d,
                        stream,
                    )?
                }
                (KvCacheDtype::Fp8KTurbo3V, _)
                | (KvCacheDtype::Fp8KTurbo4V, _)
                | (KvCacheDtype::Fp8KTurbo2V, _) => self.prefill_fp8k_turbo_nv(
                    ctx,
                    kv_cache,
                    q_contiguous,
                    attn_out,
                    meta.block_table,
                    n,
                    kv_len,
                    seq_len_start,
                    nq,
                    nkv,
                    hd,
                    bs_u,
                    inv_sqrt_d,
                    fp8_k_scale,
                    stream,
                )?,
                (KvCacheDtype::Turbo8, _)
                | (KvCacheDtype::Turbo4, _)
                | (KvCacheDtype::Turbo3, _)
                | (KvCacheDtype::Turbo2, _) => {
                    // 2026-09-25: Symmetric turbo dtypes: each has its own kernel
                    // and data-section size, and one kernel serves every chunk
                    // length.
                    let (kernel, data_bytes) = match self.kv_dtype {
                        KvCacheDtype::Turbo8 => (
                            self.prefill_attn_paged_turbo8_64_k,
                            kv_cache.turbo8_data_bytes() as u64,
                        ),
                        KvCacheDtype::Turbo4 => (
                            self.prefill_attn_paged_turbo4_64_k,
                            kv_cache.turbo4_data_bytes() as u64,
                        ),
                        KvCacheDtype::Turbo3 => (
                            self.prefill_attn_paged_turbo3_64_k,
                            kv_cache.turbo3_data_bytes() as u64,
                        ),
                        _ => (
                            self.prefill_attn_paged_turbo2_64_k,
                            kv_cache.turbo2_data_bytes() as u64,
                        ),
                    };
                    if kernel.0 == 0 {
                        anyhow::bail!(
                            "{:?} prefill paged-attention kernel not loaded (layer {});                              rebuild kernels.",
                            self.kv_dtype,
                            self.attn_layer_idx
                        );
                    }
                    let launch = if self.kv_dtype == KvCacheDtype::Turbo2 {
                        ops::prefill_attention_paged_turbo2_64
                    } else {
                        ops::prefill_attention_paged_turbo_64
                    };
                    launch(
                        ctx.gpu,
                        kernel,
                        q_contiguous,
                        kv_cache.k_pool_ptr(self.attn_layer_idx),
                        kv_cache.v_pool_ptr(self.attn_layer_idx),
                        attn_out,
                        meta.block_table,
                        n,
                        kv_len,
                        seq_len_start as u32,
                        nq,
                        nkv,
                        hd,
                        bs_u,
                        self.sliding_window.unwrap_or(0),
                        inv_sqrt_d,
                        kv_cache.block_stride_bytes_for_layer(self.attn_layer_idx) as u64,
                        data_bytes,
                        stream,
                    )?
                }
                (KvCacheDtype::Nvfp4, false) => ops::prefill_attention_paged_nvfp4(
                    ctx.gpu,
                    self.prefill_attn_paged_nvfp4_k,
                    q_contiguous,
                    kv_cache.k_pool_ptr(self.attn_layer_idx),
                    kv_cache.v_pool_ptr(self.attn_layer_idx),
                    attn_out,
                    meta.block_table,
                    n,
                    kv_len,
                    seq_len_start as u32,
                    nq,
                    nkv,
                    hd,
                    bs_u,
                    self.sliding_window.unwrap_or(0),
                    inv_sqrt_d,
                    kv_cache.block_stride_bytes_for_layer(self.attn_layer_idx) as u64,
                    kv_cache.nvfp4_data_bytes() as u64,
                    stream,
                )?,
                (KvCacheDtype::Bf16, true) => ops::prefill_attention_paged_64(
                    ctx.gpu,
                    self.prefill_attn_paged_64_k,
                    q_contiguous,
                    kv_cache.k_pool_ptr(self.attn_layer_idx),
                    kv_cache.v_pool_ptr(self.attn_layer_idx),
                    attn_out,
                    meta.block_table,
                    n,
                    kv_len,
                    seq_len_start as u32,
                    nq,
                    nkv,
                    hd,
                    bs_u,
                    self.sliding_window.unwrap_or(0),
                    inv_sqrt_d,
                    stream,
                )?,
                (KvCacheDtype::Bf16, false) => ops::prefill_attention_paged(
                    ctx.gpu,
                    self.prefill_attn_paged_k,
                    q_contiguous,
                    kv_cache.k_pool_ptr(self.attn_layer_idx),
                    kv_cache.v_pool_ptr(self.attn_layer_idx),
                    attn_out,
                    meta.block_table,
                    n,
                    kv_len,
                    seq_len_start as u32,
                    nq,
                    nkv,
                    hd,
                    bs_u,
                    self.sliding_window.unwrap_or(0),
                    inv_sqrt_d,
                    stream,
                )?,
                (_, true) => ops::prefill_attention_paged_fp8_64(
                    ctx.gpu,
                    self.prefill_attn_paged_fp8_64_k,
                    q_contiguous,
                    kv_cache.k_pool_ptr(self.attn_layer_idx),
                    kv_cache.v_pool_ptr(self.attn_layer_idx),
                    attn_out,
                    meta.block_table,
                    n,
                    kv_len,
                    seq_len_start as u32,
                    nq,
                    nkv,
                    hd,
                    bs_u,
                    self.sliding_window.unwrap_or(0),
                    inv_sqrt_d,
                    fp8_k_scale,
                    fp8_v_scale,
                    kv_cache.cache_stride() as u64,
                    stream,
                )?,
                (_, false) => ops::prefill_attention_paged_fp8(
                    ctx.gpu,
                    self.prefill_attn_paged_fp8_k,
                    q_contiguous,
                    kv_cache.k_pool_ptr(self.attn_layer_idx),
                    kv_cache.v_pool_ptr(self.attn_layer_idx),
                    attn_out,
                    meta.block_table,
                    n,
                    kv_len,
                    seq_len_start as u32,
                    nq,
                    nkv,
                    hd,
                    bs_u,
                    self.sliding_window.unwrap_or(0),
                    inv_sqrt_d,
                    fp8_k_scale,
                    fp8_v_scale,
                    kv_cache.cache_stride() as u64,
                    stream,
                )?,
            }
        }

        Ok(PagedAttnOutcome::Continue)
    }
}
