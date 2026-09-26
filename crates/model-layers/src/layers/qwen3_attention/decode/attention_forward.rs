// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Single-token decode for one attention layer: Q/K/V projections, LoRA deltas, q/k
//! norms, RoPE, the KV-cache write, paged decode attention (or the high-speed-swap or QSA route),
//! output gates and the O projection.
//!
//! Owner: model-layers attention decode.
//! Invariants:
//! - On the non-MLA path, K and V are written to the cache after their LoRA deltas, norms and
//!   RoPE, and before the attention that reads them.

#![allow(unused_imports)]

use anyhow::Result;
use metrale_cache::kv_cache::{KvCacheDtype, PagedKvCache};
use metrale_cache::kv_dequant::{
    NVFP4_E2M1_LUT, TURBO4_LUT, dequant_4bit_block_to_bf16, dequant_fp8_to_bf16,
    dequant_turbo3_block_to_bf16, dequant_turbo8_block_to_bf16,
};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

use super::super::Qwen3AttentionLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;

mod lora;
mod q_proj;
mod rope;

impl Qwen3AttentionLayer {
    pub(in super::super) fn attention_forward(
        &self,
        state: &mut dyn crate::layer::LayerState,
        normed: DevicePtr,
        seq_len: usize,
        block_table: &mut Vec<u32>,
        disk_block_ids: &mut Vec<u32>,
        disk_last_offloaded_per_layer: &mut Vec<u32>,
        kv_cache: &mut PagedKvCache,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        let h = ctx.config.hidden_size as u32;
        // 2026-09-25: Per-layer dimension overrides, for models whose layers differ in shape.
        let nq = self
            .num_q_heads_override
            .unwrap_or(ctx.config.num_attention_heads) as u32;
        let nkv = self
            .num_kv_heads_override
            .unwrap_or(ctx.config.num_key_value_heads) as u32;
        let hd = self.head_dim_override.unwrap_or(ctx.config.head_dim) as u32;
        let eps = ctx.config.rms_norm_eps as f32;
        let bs = kv_cache.block_size();

        // 2026-09-25: The caller allocates blocks before this call (`ensure_blocks_through_decode`
        // in model-engine `block_mgmt.rs`); this only checks, in debug builds, that the table is
        // long enough.
        let blocks_needed = (seq_len / bs) + 1;
        let expected_window_size = match kv_cache.config().cache_blocks_per_seq {
            Some(cap) => blocks_needed.min(cap as usize),
            None => blocks_needed,
        };
        debug_assert!(
            block_table.len() >= expected_window_size,
            "Qwen3AttentionLayer::decode entered with under-allocated block_table \
             ({}/{} blocks) — caller must call ensure_blocks_through_decode",
            block_table.len(),
            expected_window_size,
        );

        // 2026-09-25: Q/K/V projections into separate regions of `qkv_output`.
        let q_out = ctx.buffers.qkv_output();
        let q_dim = nq * hd;
        let q_proj_dim = if self.gated { q_dim * 2 } else { q_dim };
        let q_proj_bytes = q_proj_dim as usize * 2;
        let k_out = q_out.offset(q_proj_bytes);
        let v_out = k_out.offset((nkv * hd) as usize * 2);
        let meta = ctx
            .attn_metadata
            .expect("attention layer requires pre-uploaded metadata");

        // 2026-09-25: MLA layers return through `attention_forward_v4` (when `o_lora_rank > 0`) or
        // `attention_forward_mla`.
        if let Some(ref mla) = self.mla {
            let args = super::attention_forward_mla::DecodeMlaArgs {
                normed,
                q_out,
                k_out,
                v_out,
                q_dim,
                h,
                nq,
                hd,
                eps,
                bs,
                stream,
                // 2026-09-25: Passed as `seq_len - 1`; the V4 path uses it for the compressed-pool
                // decode append.
                pos: Some(seq_len.saturating_sub(1) as u32),
            };
            if mla.o_lora_rank > 0 {
                return self.attention_forward_v4(kv_cache, ctx, &args);
            }
            return self.attention_forward_mla(kv_cache, ctx, &args);
        }

        self.attention_forward_q_proj(ctx, normed, q_out, q_dim, q_proj_dim, nq, hd, h, stream)?;

        // 2026-09-25: With profiling on, log the first 8 values of layer 0's input and Q output.
        self.attention_forward_q_diag(ctx, normed, q_out, nq, hd, h, stream)?;

        let k_out = q_out.offset(q_proj_bytes);
        let v_out = k_out.offset((nkv * hd) as usize * 2);

        self.attention_forward_kv(normed, k_out, v_out, nkv, hd, h, ctx, stream)?;

        // 2026-09-25: LoRA deltas on K and V (the Q delta is folded in `apply_q_lora` above). They
        // run before the k norm, RoPE and the cache write below, so the cache stores the adapted K
        // and V. Here rather than inside `attention_forward_kv`, which returns early from several
        // arms.
        self.apply_kv_lora(ctx, normed, k_out, v_out, stream)?;

        // 2026-09-25: Q and K RMS norms, before RoPE, one of three ways:
        // - `q_norm_full`/`k_norm_full`: one norm over the whole `[n*hd]` projection per token.
        //   Only the MiniMax loader sets them.
        // - per-head norm: rows = heads, cols = hd;
        // - neither weight present: no norm.
        if let Some(ref q_norm_full) = self.attn.q_norm_full {
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_w_k,
                q_out,
                q_norm_full,
                q_out,
                1,
                nq * hd,
                eps,
                stream,
            )?;
        } else if !self.attn.q_norm.weight.is_null() {
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_w_k,
                q_out,
                &self.attn.q_norm,
                q_out,
                nq,
                hd,
                eps,
                stream,
            )?;
        }
        // 2026-09-25: When `fused_fp8_kv_decode_eligible` holds, the per-head K `rms_norm` below
        // and the K half of the RoPE launch are skipped and redone inside
        // `write_kv_cache_fp8_fused`, which reads the raw K projection and writes K and V to the
        // FP8 cache in one launch. The kernel reproduces the unfused chain's BF16 roundings
        // (`reshape_and_cache_fused_k_fp8.cu`).
        let rotary_dim = self
            .rotary_dim_override
            .unwrap_or(ctx.config.rotary_dim() as u32);
        let fused_k_fp8 = self.fused_fp8_kv_decode_eligible(hd, rotary_dim);
        if let Some(ref k_norm_full) = self.attn.k_norm_full {
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_w_k,
                k_out,
                k_norm_full,
                k_out,
                1,
                nkv * hd,
                eps,
                stream,
            )?;
        } else if !self.attn.k_norm.weight.is_null() && !fused_k_fp8 {
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_w_k,
                k_out,
                &self.attn.k_norm,
                k_out,
                nkv,
                hd,
                eps,
                stream,
            )?;
        }

        // 2026-09-25: V norm, in place, when the loader set `v_norm_weight` (`set_v_norm` or
        // `set_k_eq_v`). V gets no RoPE.
        if let Some(v_norm_w) = self.v_norm_weight.as_ref() {
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_w_k,
                v_out,
                v_norm_w,
                v_out,
                nkv,
                hd,
                eps,
                stream,
            )?;
        }

        self.attention_forward_rope(
            ctx,
            &meta,
            q_out,
            k_out,
            nq,
            nkv,
            hd,
            rotary_dim,
            fused_k_fp8,
            stream,
        )?;

        let kv_stride = nkv * hd;
        if fused_k_fp8 {
            self.write_kv_cache_fp8_fused(
                ctx.gpu,
                k_out,
                v_out,
                kv_cache,
                meta.slot,
                meta.positions,
                1,
                nkv,
                hd,
                // 2026-09-25: The same `rotary_dim` the eligibility check read.
                rotary_dim,
                bs as u32,
                kv_stride,
                kv_stride,
                eps,
                self.rope_theta_override
                    .unwrap_or(ctx.config.rope_theta as f32),
                stream,
            )?;
        } else {
            self.write_kv_cache(
                ctx.gpu,
                k_out,
                v_out,
                kv_cache,
                meta.slot,
                1,
                nkv,
                hd,
                bs as u32,
                kv_stride,
                kv_stride,
                stream,
                ctx.graph_capture,
            )?;
        }

        // 2026-09-25: A WHT-rotated (turbo) cache stores WHT(K) and WHT(V). The transform is
        // normalized by 1/sqrt(head_dim) (`wht_bf16.cu`), so <WHT(Q), WHT(K)> = <Q, K>. WHT(Q) runs
        // only when K is rotated, and iWHT of the output below only when V is, so K and V may use
        // different dtypes.
        let (k_dtype, v_dtype) = self.kv_dtype.kv_pair();
        let k_is_turbo = k_dtype.is_wht_rotated();
        let v_is_turbo = v_dtype.is_wht_rotated();
        // 2026-09-25: InnerQ scale on Q before the WHT; the kernel returns at once when the device
        // flag `d_innerq_active` is 0. The runtime WHT(Q) is skipped when `TQ_PLUS_WEIGHT_ROTATION`
        // pre-rotated the weights at load.
        let weight_pre_rotated = crate::layers::ops::ModelLevers::get().weight_pre_rotated;
        if k_is_turbo && self.innerq_apply_q_k.0 != 0 && hd == 128 {
            use metrale_gpu_runtime::kernel_args::KernelLaunch;
            KernelLaunch::new(ctx.gpu, self.innerq_apply_q_k)
                .grid([nq, 1, 1])
                .block([32, 1, 1])
                .arg_ptr(q_out)
                .arg_u32(hd)
                .launch(stream)?;
        }
        if k_is_turbo
            && !weight_pre_rotated
            && self.wht_bf16_k.0 != 0
            && (hd == 128 || hd == 256 || hd == 512)
        {
            use metrale_gpu_runtime::kernel_args::KernelLaunch;
            KernelLaunch::new(ctx.gpu, self.wht_bf16_k)
                .grid([nq, 1, 1])
                .block([32, 1, 1])
                .arg_ptr(q_out)
                .arg_u32(hd)
                .launch(stream)?;
        }

        let attn_out = ctx.buffers.attn_output();
        let inv_sqrt_d = self.effective_attn_scale(hd);

        // 2026-09-25: With high-speed swap engaged for this layer (`high_speed_swap_engaged`),
        // attention streams the disk-side history through the thread-local orchestrator instead of
        // the paged kernel. For a turbo cache the WHT(Q) above and the iWHT below bracket it as
        // they do the paged path.
        let use_orchestrator = self.high_speed_swap_engaged(kv_cache);

        // 2026-09-25: QSA indexer: ingest this token's indexer key every step; past the inert
        // bound, select blocks and gather their K/V into contiguous scratch, which the paged decode
        // attention below reads through an identity block table. Runs after the cache write so the
        // current token can be gathered.
        let qsa_sel = if let Some(ref qsa) = self.qsa {
            anyhow::ensure!(
                matches!(self.kv_dtype.kv_pair().0, KvCacheDtype::Bf16)
                    && matches!(self.kv_dtype.kv_pair().1, KvCacheDtype::Bf16),
                "QSA selection requires a plain BF16 KV cache (the gather \
                 copies raw NHD rows); serve with --kv-cache-dtype bf16"
            );
            anyhow::ensure!(
                !use_orchestrator,
                "QSA + --high-speed-swap is not wired (the gather reads the \
                 HBM pool)"
            );
            // 2026-09-25: `seq_len` is the length before this token is appended, so the token being
            // decoded is at position `seq_len`; `decode_select` refuses a position that does not
            // equal the number of keys already ingested.
            let qsa_st =
                crate::layers::qwen3_attention::helpers::qsa_seq_state(qsa, state, ctx.gpu)?;
            qsa.decode_select(
                qsa_st,
                normed,
                seq_len,
                kv_cache.k_pool_ptr(self.attn_layer_idx),
                kv_cache.v_pool_ptr(self.attn_layer_idx),
                meta.block_table,
                bs as u32,
                ctx.gpu,
                stream,
            )?
        } else {
            None
        };

        if use_orchestrator {
            // 2026-09-25: Push this layer's new K/V blocks to disk; `disk_block_ids` was grown by
            // the caller's block allocation.
            self.high_speed_swap_offload_new_blocks(
                kv_cache,
                block_table,
                disk_block_ids,
                disk_last_offloaded_per_layer,
                ctx,
                stream,
                nkv,
                hd,
                bs,
            )?;
            metrale_storage::with_local(|hss| {
                hss.attend_layer_on_stream(
                    stream,
                    self.attn_layer_idx as u32,
                    disk_block_ids,
                    q_out.0,
                    attn_out.0,
                )
            })
            .expect("local installed checked in high_speed_swap_engaged")?;
        } else if let Some(sel) = qsa_sel {
            // 2026-09-25: Attention over only the selected tokens, with the BF16 paged kernel
            // pointed at the gathered scratch. The cached K rows already carry RoPE.
            ops::paged_decode_attn_bf16(
                ctx.gpu,
                self.paged_decode_k,
                q_out,
                sel.k_scratch,
                sel.v_scratch,
                attn_out,
                sel.table_dev,
                sel.seq_len_dev,
                sel.max_blocks,
                1,
                nq,
                nkv,
                hd,
                bs as u32,
                inv_sqrt_d,
                nq * hd,
                0,
                stream,
            )?;
        } else {
            self.run_paged_decode(
                ctx.gpu,
                q_out,
                kv_cache,
                attn_out,
                meta.block_table,
                meta.seq_len,
                meta.max_blocks_per_seq,
                1,
                nq,
                nkv,
                hd,
                bs as u32,
                inv_sqrt_d,
                nq * hd,
                ctx.buffers.splitk_workspace(),
                ctx.levers.max_decode_seqs,
                stream,
            )?;
        }

        // 2026-09-25: For a WHT-rotated V the output is sum(softmax * WHT(V)), so iWHT of it is the
        // real output. The guard reads V's dtype, not K's.
        if v_is_turbo
            && !weight_pre_rotated
            && self.wht_bf16_k_inv.0 != 0
            && (hd == 128 || hd == 256 || hd == 512)
        {
            use metrale_gpu_runtime::kernel_args::KernelLaunch;
            KernelLaunch::new(ctx.gpu, self.wht_bf16_k_inv)
                .grid([nq, 1, 1])
                .block([32, 1, 1])
                .arg_ptr(attn_out)
                .arg_u32(hd)
                .launch(stream)?;
        }

        if self.gated {
            let gate_ptr = q_out.offset(q_dim as usize * 2);
            ops::sigmoid_gate_mul(
                ctx.gpu,
                self.sigmoid_gate_mul_k,
                attn_out,
                gate_ptr,
                attn_out,
                nq * hd,
                stream,
            )?;
        }

        // 2026-09-25: Per-head attention gate: gate[h] = g_proj(normed), then a sigmoid or softplus
        // broadcast over each head.
        if let Some(ref g_proj) = self.head_gate_weight {
            // 2026-09-25: The `[1, nq]` gate reuses the `q_out` scratch.
            let gate_buf = q_out;
            if self.dense_gemv_batchm_k.0 != 0 {
                ops::dense_gemv_batchm(
                    ctx.gpu,
                    self.dense_gemv_batchm_k,
                    normed,
                    g_proj,
                    gate_buf,
                    1,
                    nq,
                    h,
                    nq,
                    stream,
                )?;
            } else {
                ops::dense_gemm_tc(
                    ctx.gpu,
                    self.dense_gemm_tc_k,
                    normed,
                    g_proj,
                    gate_buf,
                    1,
                    nq,
                    h,
                    stream,
                )?;
            }
            match self.head_gate_activation {
                super::super::types::HeadGateActivation::Sigmoid => {
                    ops::sigmoid_gate_mul_head_broadcast(
                        ctx.gpu,
                        self.sigmoid_gate_head_broadcast_k,
                        attn_out,
                        gate_buf,
                        attn_out,
                        nq,
                        hd,
                        1,
                        stream,
                    )?;
                }
                super::super::types::HeadGateActivation::Softplus => {
                    ops::softplus_gate_mul_head_broadcast(
                        ctx.gpu,
                        self.softplus_gate_head_broadcast_k,
                        attn_out,
                        gate_buf,
                        attn_out,
                        nq,
                        hd,
                        1,
                        stream,
                    )?;
                }
            }
        }

        let o_out = self.attention_forward_oproj(attn_out, nq, hd, h, ctx, stream)?;

        Ok(o_out)
    }
}
