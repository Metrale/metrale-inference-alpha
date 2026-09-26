// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `prefill_attention_paged`: full-attention prefill for chunks
//! with `seq_len_start > 0` and for batched first chunks when those are
//! enabled (`trait_impl/prefill_inner.rs`): Q/K/V projection, norms, RoPE, the
//! KV-cache write for rows from `kv_write_floor` on, attention over the paged
//! cache, gates and O projection. MLA layers return through `paged_mla.rs` or
//! `paged_v4.rs`.
//!
//! Owner: model-layers (attention).
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_cache::kv_cache::PagedKvCache;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::super::Qwen3AttentionLayer;
use crate::layer::{BatchedAttnMetadata, ForwardContext};
use crate::layers::ops;

mod head_gate;
mod norms;
mod rope_cache;

impl Qwen3AttentionLayer {
    pub(in crate::layers::qwen3_attention) fn prefill_attention_paged(
        &self,
        state: &mut dyn crate::layer::LayerState,
        normed: DevicePtr,
        num_tokens: usize,
        seq_len_start: usize,
        kv_cache: &mut PagedKvCache,
        block_table: &Vec<u32>,
        disk_block_ids: &mut Vec<u32>,
        disk_last_offloaded_per_layer: &mut Vec<u32>,
        // 2026-09-25: `Some`: a batched chunk. Attention uses the batched
        // kernel with the metadata's per-stream `block_table_ptrs`, and
        // positions and slots come from its stacked arrays. `None`: one
        // sequence, metadata from `ctx.attn_metadata`.
        batched_meta: Option<&BatchedAttnMetadata>,
        // 2026-09-25: The first `kv_write_floor` rows are positions whose K/V
        // are already in the cache (the caller passes `kv_write_start`); their
        // cache write is skipped.
        kv_write_floor: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        let h = ctx.config.hidden_size as u32;
        let nq = self
            .num_q_heads_override
            .unwrap_or(ctx.config.num_attention_heads) as u32;
        let nkv = self
            .num_kv_heads_override
            .unwrap_or(ctx.config.num_key_value_heads) as u32;
        let hd = self.head_dim_override.unwrap_or(ctx.config.head_dim) as u32;
        let eps = ctx.config.rms_norm_eps as f32;
        let bs = kv_cache.block_size();
        let n = num_tokens as u32;
        let bf16 = 2usize;

        let q_dim = (nq * hd) as usize;
        let q_proj_dim = if self.gated { q_dim * 2 } else { q_dim };
        let kv_dim = (nkv * hd) as usize;

        // 2026-09-25: A batched chunk on an MLA layer is an error; the caller
        // routes MLA layers per stream.
        if batched_meta.is_some() && self.mla.is_some() {
            anyhow::bail!(
                "prefill_attention_paged: batched_meta with MLA layer is not supported \
                 (layer {}). Caller must route MLA layers to per-stream.",
                self.attn_layer_idx
            );
        }

        // 2026-09-25: MLA layers return through `paged_mla.rs`, or `paged_v4.rs`
        // when `o_lora_rank > 0`.
        if let Some(ref mla) = self.mla {
            let args = super::paged_mla::MlaPrefillArgs {
                normed,
                num_tokens,
                n,
                h,
                nq,
                nkv,
                hd,
                seq_len_start,
                kv_dim,
                eps,
                bf16,
                bs: bs as u32,
                stream,
            };
            if mla.o_lora_rank > 0 {
                return self.prefill_attention_paged_v4(kv_cache, ctx, &args, seq_len_start);
            }
            return self.prefill_attention_paged_mla(kv_cache, ctx, &args);
        }

        // 2026-09-25: V sits at `k_contiguous + num_tokens * kv_dim`, where the V
        // projection in `paged_qkv.rs` writes it.
        let qg_out = ctx.buffers.qkv_output();
        let k_contiguous = ctx.buffers.ssm_qkvz();
        let v_contiguous = k_contiguous.offset(num_tokens * kv_dim * bf16);
        let q_contiguous = ctx.buffers.ssm_deinterleaved();
        // 2026-09-25: Zero the V region before the projections run. It must
        // come before `prefill_attention_paged_qkv`, which writes V into exactly
        // this region; any row the V projection does not write reads as zero,
        // not as stale data.
        ctx.gpu
            .memset_async(v_contiguous, 0, num_tokens * kv_dim * bf16, stream)?;

        // 2026-09-25: Q/K/V projection for non-MLA layers.
        if self.mla.is_none() {
            self.prefill_attention_paged_qkv(
                normed, n, h, nkv, hd, q_proj_dim, kv_dim, num_tokens, bf16, ctx, stream,
            )?;
        }
        // 2026-09-25: Fused K path, with `METRALE_FUSED_KV=1` on an
        // mrope-interleaved layer whose fused and V-only kernels resolved: copy
        // the raw projected K into `attn_output` (unused until attention writes
        // it) before k_norm and RoPE change `k_contiguous`. After the normal
        // cache write, a BF16 cache on a layer with a K norm gets its K side
        // rewritten from this copy by one kernel that does the norm and RoPE in
        // FP32 and rounds to BF16 once.
        let fused_kv_enabled = self.mrope_interleaved
            && self.fused_k_norm_rope_mrope_cache_write_bf16_k.0 != 0
            && self.reshape_and_cache_flash_v_only_k.0 != 0
            && std::env::var("METRALE_FUSED_KV").ok().as_deref() == Some("1");
        let raw_k_scratch = if fused_kv_enabled {
            let scratch = ctx.buffers.attn_output();
            ctx.gpu
                .copy_d2d_async(k_contiguous, scratch, num_tokens * kv_dim * bf16, stream)?;
            Some(scratch)
        } else {
            None
        };
        self.prefill_paged_split_and_norm(
            ctx,
            qg_out,
            q_contiguous,
            k_contiguous,
            v_contiguous,
            n,
            nq,
            nkv,
            hd,
            q_proj_dim,
            eps,
            num_tokens,
            q_dim,
            bf16,
            stream,
        )?;

        // 2026-09-25: RoPE. Single-stream mode requires `ctx.attn_metadata`; a
        // batched chunk reads the stacked metadata instead.
        let meta_for_single = match (batched_meta, ctx.attn_metadata) {
            (Some(_), _) => None,
            (None, Some(m)) => Some(m),
            (None, None) => anyhow::bail!(
                "prefill_attention_paged: single-stream mode requires ctx.attn_metadata"
            ),
        };

        // 2026-09-25: Positions and slots: the stacked arrays for a batched
        // chunk, otherwise `meta.positions*` and `meta.slot`.
        let bmeta_positions = batched_meta
            .map(|m| m.positions_stacked)
            .or(meta_for_single.map(|m| m.positions))
            .unwrap();
        let bmeta_positions_h = batched_meta
            .map(|m| m.positions_h_stacked)
            .or(meta_for_single.map(|m| m.positions_h))
            .unwrap();
        let bmeta_positions_w = batched_meta
            .map(|m| m.positions_w_stacked)
            .or(meta_for_single.map(|m| m.positions_w))
            .unwrap();
        let bmeta_slot = batched_meta
            .map(|m| m.slot_stacked)
            .or(meta_for_single.map(|m| m.slot))
            .unwrap();
        self.prefill_paged_rope_cache_write(
            kv_cache,
            q_contiguous,
            k_contiguous,
            v_contiguous,
            raw_k_scratch,
            bmeta_positions,
            bmeta_positions_h,
            bmeta_positions_w,
            bmeta_slot,
            n,
            nq,
            nkv,
            hd,
            bs,
            num_tokens,
            kv_write_floor,
            kv_dim,
            bf16,
            ctx,
            stream,
        )?;

        // 2026-09-25: Attention (`paged_attn.rs`, `paged_attn_batched.rs`).
        let attn_out = ctx.buffers.attn_output();
        let inv_sqrt_d = self.effective_attn_scale(hd);
        // 2026-09-25: In a batched chunk `num_tokens` counts every stream's
        // rows, so the per-stream `kv_len` is `seq_len_start + chunk_len`.
        let kv_len = batched_meta
            .map(|m| seq_len_start + m.chunk_len as usize)
            .unwrap_or(seq_len_start + num_tokens) as u32;

        // 2026-09-25: WHT bookends. For a WHT-rotated side the cache holds
        // rotated K/V (`write_kv_cache` rotates before it writes), so rotate Q
        // when K is rotated (<WHT(Q), WHT(K)> = <Q, K>) and rotate the output
        // back when V is rotated. None of it runs with pre-rotated weights or
        // at a head_dim other than 128, 256 or 512.
        let (wht_k_dtype, wht_v_dtype) = self.kv_dtype.kv_pair();
        let k_is_turbo = wht_k_dtype.is_wht_rotated();
        let v_is_turbo = wht_v_dtype.is_wht_rotated();
        let weight_pre_rotated = crate::layers::ops::ModelLevers::get().weight_pre_rotated;
        let wht_runtime_active = !weight_pre_rotated && (hd == 128 || hd == 256 || hd == 512);
        if k_is_turbo && wht_runtime_active && self.wht_bf16_k.0 != 0 {
            use metrale_gpu_runtime::kernel_args::KernelLaunch;
            KernelLaunch::new(ctx.gpu, self.wht_bf16_k)
                .grid([n * nq, 1, 1]) // 2026-09-25: one warp per (token, q_head)
                .block([32, 1, 1])
                .arg_ptr(q_contiguous)
                .arg_u32(hd)
                .launch(stream)?;
        }
        if let Some(bmeta) = batched_meta {
            // 2026-09-25: FlashInfer ragged (varlen) prefill, for a batched
            // first chunk (`seq_len_start == 0`) with
            // `METRALE_FLASHINFER_PREFILL=1`, FlashInfer available, head_dim 128
            // or 256 and no WHT-rotated side: one launch runs every stream's
            // causal self-attention on the fresh contiguous Q/K/V. Later chunks
            // need the cached prefix, which those buffers do not hold. Only the
            // head_dim-128 entry takes a sliding window, so a windowed layer
            // needs head_dim 128
            // (`crates/gpu-runtime/cuda/flashinfer_ragged_prefill.cu`).
            let windowed_ok = hd == 128 || self.sliding_window.is_none();
            let use_flashinfer = seq_len_start == 0
                && (hd == 256 || hd == 128)
                && windowed_ok
                && !k_is_turbo
                && !v_is_turbo
                && metrale_gpu_runtime::flashinfer::available()
                && std::env::var("METRALE_FLASHINFER_PREFILL").ok().as_deref() == Some("1");
            if use_flashinfer {
                // 2026-09-25: Per-stream `cu_seqlens` from the staged metadata,
                // host and device copies. For first-chunk self-attention the Q
                // and KV index pointers are the same array.
                let batch = bmeta.batch_size as usize;
                let total = bmeta.total_tokens;
                let indptr_h = &bmeta.cu_seqlens_host;
                let indptr_d = bmeta.cu_seqlens.0;
                {
                    if ctx.stats.once("log:flashinfer_prefill_varlen") {
                        tracing::warn!(
                            "FLASHINFER_PREFILL(varlen) batch={batch} total={total} \
                             num_tokens={num_tokens} cu_seqlens={indptr_h:?} \
                             nq={nq} nkv={nkv} hd={hd} sm_scale={inv_sqrt_d} \
                             sliding_window={sw:?}",
                            sw = self.sliding_window
                        );
                    }
                }
                if hd == 128 {
                    metrale_gpu_runtime::flashinfer::ragged_prefill_bf16_hd128(
                        q_contiguous.0,
                        k_contiguous.0,
                        v_contiguous.0,
                        attn_out.0,
                        indptr_h,
                        indptr_h,
                        indptr_d,
                        indptr_d,
                        batch as u32,
                        total,
                        total,
                        nq,
                        nkv,
                        hd,
                        inv_sqrt_d,
                        true,
                        self.sliding_window,
                        stream,
                    )?;
                } else {
                    metrale_gpu_runtime::flashinfer::ragged_prefill_bf16_hd256(
                        q_contiguous.0,
                        k_contiguous.0,
                        v_contiguous.0,
                        attn_out.0,
                        indptr_h,
                        indptr_h,
                        indptr_d,
                        indptr_d,
                        batch as u32,
                        total,
                        total,
                        nq,
                        nkv,
                        hd,
                        inv_sqrt_d,
                        true,
                        stream,
                    )?;
                }
            } else {
                // 2026-09-25: Batched paged attention: each stream's KV pages come
                // from `block_table_ptrs[b]` (`paged_attn_batched.rs`).
                let args = super::paged_attn_batched::PagedAttnBatchedArgs {
                    q_contiguous,
                    attn_out,
                    seq_len_start,
                    nq,
                    nkv,
                    hd,
                    bs,
                    inv_sqrt_d,
                    kv_len,
                    batched_meta: bmeta,
                    stream,
                };
                self.prefill_attention_paged_attn_batched(kv_cache, ctx, &args)?;
            }
        } else {
            // 2026-09-25: Single-stream: `meta_for_single` was checked above.
            let meta = meta_for_single
                .expect("single-stream mode: meta_for_single guaranteed by validation above");
            let mut args = super::paged_attn::PagedAttnArgs {
                q_contiguous,
                k_contiguous,
                v_contiguous,
                attn_out,
                n,
                seq_len_start,
                num_tokens,
                nq,
                nkv,
                hd,
                bs,
                bf16,
                inv_sqrt_d,
                kv_len,
                meta: &meta,
                block_table,
                disk_block_ids,
                disk_last_offloaded_per_layer,
                stream,
            };
            match self.prefill_attention_paged_attn(kv_cache, ctx, &mut args)? {
                super::paged_attn::PagedAttnOutcome::EarlyReturn(out) => return Ok(out),
                super::paged_attn::PagedAttnOutcome::Continue => {}
            }
        }

        // 2026-09-25: Output-side WHT bookend: the output is
        // sum(softmax * WHT(V)), so rotate it back.
        if v_is_turbo && wht_runtime_active && self.wht_bf16_k_inv.0 != 0 {
            use metrale_gpu_runtime::kernel_args::KernelLaunch;
            KernelLaunch::new(ctx.gpu, self.wht_bf16_k_inv)
                .grid([n * nq, 1, 1])
                .block([32, 1, 1])
                .arg_ptr(attn_out)
                .arg_u32(hd)
                .launch(stream)?;
        }

        // 2026-09-25: METRALE_OP_DUMP: the attention output before the gates,
        // the last token's `nq * hd` values.
        if num_tokens > 0 {
            let nq_hd = (nq * hd) as usize;
            super::super::op_dump::dump_bf16(
                ctx.gpu,
                attn_out,
                (num_tokens - 1) * nq_hd * bf16,
                nq_hd,
                self.attn_layer_idx,
                "attn_out_pre_gate",
                stream,
            )?;
        }

        // 2026-09-25: QSA layers whose `seq_len_start + num_tokens` exceeds
        // `inert_bound`: the same `prefill_select` hook as the cache-skip path,
        // single-stream only. The paged cache holds this chunk (written above)
        // and every earlier one. It runs before the gates, while `q_contiguous`
        // still holds Q.
        if let Some(ref qsa) = self.qsa
            && seq_len_start + num_tokens > qsa.inert_bound()
        {
            anyhow::ensure!(
                batched_meta.is_none(),
                "QSA prefill selection is single-stream (batched paged \
                 prefill is refused upstream for this model)"
            );
            let qsa_st =
                crate::layers::qwen3_attention::helpers::qsa_seq_state(qsa, state, ctx.gpu)?;
            qsa.prefill_select(
                qsa_st,
                normed,
                q_contiguous,
                attn_out,
                kv_cache.k_pool_ptr(self.attn_layer_idx),
                kv_cache.v_pool_ptr(self.attn_layer_idx),
                block_table,
                seq_len_start,
                num_tokens,
                nq,
                bs as u32,
                inv_sqrt_d,
                ctx.buffers.qsa_select_scratch(),
                ctx.gpu,
                stream,
            )?;
        }

        // 2026-09-25: Gated layers: multiply the attention output by the
        // sigmoid of the gate.
        if self.gated {
            // 2026-09-25: The gate is in `qg_out` at offset `q_dim`, with stride
            // `q_proj_dim` between tokens.
            let gate_base = qg_out.offset(q_dim * bf16);
            ops::sigmoid_gate_mul_batched(
                ctx.gpu,
                self.sigmoid_gate_mul_batched_k,
                attn_out,
                gate_base,
                attn_out,
                nq * hd,
                q_proj_dim as u32,
                n,
                stream,
            )?;
        }

        // 2026-09-25: Per-head gate (`head_gate_weight`): one scalar per head
        // from the normed input, applied to that head's whole output with the
        // layer's activation (sigmoid or softplus).
        if let Some(ref g_proj) = self.head_gate_weight {
            self.prefill_paged_head_gate(
                g_proj,
                normed,
                q_contiguous,
                attn_out,
                n,
                nq,
                hd,
                h,
                ctx,
                stream,
            )?;
        }

        // 2026-09-25: METRALE_OP_DUMP: the attention output after the gates,
        // which is the O-projection input.
        if num_tokens > 0 {
            let nq_hd = (nq * hd) as usize;
            super::super::op_dump::dump_bf16(
                ctx.gpu,
                attn_out,
                (num_tokens - 1) * nq_hd * bf16,
                nq_hd,
                self.attn_layer_idx,
                "attn_out_post_gate",
                stream,
            )?;
        }

        // 2026-09-25: O projection (`paged_oproj.rs`).
        let o_out = self.prefill_attention_paged_oproj(attn_out, n, h, nq, hd, ctx, stream)?;

        Ok(o_out)
    }
}
