// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The attention and gate steps of the cache-skip prefill
//! (`cache_skip.rs`): the WHT bookends, flash attention, the QSA row rewrite
//! and the output and per-head gates.
//!
//! Owner: model-layers (attention).
//! Invariants: `prefill_attention_with_cache_skip` calls these helpers where
//! their statements run, so every launch keeps its order and arguments.

use anyhow::Result;
use metrale_cache::kv_cache::PagedKvCache;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::super::Qwen3AttentionLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;

impl Qwen3AttentionLayer {
    /// 2026-09-26: The WHT bookends around flash attention, and the attention
    /// launch itself, of `prefill_attention_with_cache_skip`. Returns
    /// `wide_head_path`: whether the head_dim > 256 kernel ran.
    pub(super) fn cache_skip_flash(
        &self,
        ctx: &ForwardContext,
        kv_write_start: usize,
        q_contiguous: DevicePtr,
        k_contiguous: DevicePtr,
        v_contiguous: DevicePtr,
        attn_out: DevicePtr,
        flash_seq_len: u32,
        flash_batch: u32,
        n: u32,
        nq: u32,
        nkv: u32,
        hd: u32,
        inv_sqrt_d: f32,
        tp1: Option<std::time::Instant>,
        stream: u64,
    ) -> Result<bool> {
        // 2026-09-25: WHT bookends. For a WHT-rotated side, `write_kv_cache`
        // above rotated rows `[kv_write_start..)` of that side's contiguous
        // buffer in place, so attention reads rotated K/V there. The code below
        // rotates the prefix rows `[0..kv_write_start)` the write skipped,
        // rotates Q when K is rotated (<WHT(Q), WHT(K)> = <Q, K>), and rotates
        // the output back after attention when V is rotated. None of it runs
        // with pre-rotated weights or at a head_dim other than 128, 256 or 512.
        let (wht_k_dtype, wht_v_dtype) = self.kv_dtype.kv_pair();
        let k_is_turbo = wht_k_dtype.is_wht_rotated();
        let v_is_turbo = wht_v_dtype.is_wht_rotated();
        let weight_pre_rotated = crate::layers::ops::ModelLevers::get().weight_pre_rotated;
        let wht_runtime_active = !weight_pre_rotated && (hd == 128 || hd == 256 || hd == 512);
        if wht_runtime_active && kv_write_start > 0 && self.wht_bf16_k.0 != 0 {
            use metrale_gpu_runtime::kernel_args::KernelLaunch;
            let prefix_heads = kv_write_start as u32 * nkv;
            if k_is_turbo {
                KernelLaunch::new(ctx.gpu, self.wht_bf16_k)
                    .grid([prefix_heads, 1, 1]) // 2026-09-25: one warp per (token, kv_head)
                    .block([32, 1, 1])
                    .arg_ptr(k_contiguous)
                    .arg_u32(hd)
                    .launch(stream)?;
            }
            if v_is_turbo {
                KernelLaunch::new(ctx.gpu, self.wht_bf16_k)
                    .grid([prefix_heads, 1, 1])
                    .block([32, 1, 1])
                    .arg_ptr(v_contiguous)
                    .arg_u32(hd)
                    .launch(stream)?;
            }
        }
        if k_is_turbo && wht_runtime_active && self.wht_bf16_k.0 != 0 {
            use metrale_gpu_runtime::kernel_args::KernelLaunch;
            KernelLaunch::new(ctx.gpu, self.wht_bf16_k)
                .grid([n * nq, 1, 1]) // 2026-09-25: one warp per (token, q_head)
                .block([32, 1, 1])
                .arg_ptr(q_contiguous)
                .arg_u32(hd)
                .launch(stream)?;
        }
        let wide_head_path = hd > 256 && self.prefill_attn_512_k.0 != 0;
        if wide_head_path {
            // 2026-09-25: Heads wider than 256: `prefill_attn_512_k`, the
            // tensor-core or scalar kernel `ops::wide_prefill_kernel` resolved,
            // with no sliding window (0).
            ops::prefill_attention(
                ctx.gpu,
                self.prefill_attn_512_k,
                q_contiguous,
                k_contiguous,
                v_contiguous,
                attn_out,
                flash_seq_len,
                flash_batch,
                nq,
                nkv,
                hd,
                inv_sqrt_d,
                true,
                0,
                stream,
            )
            .map_err(|e| {
                anyhow::anyhow!("prefill_512 failed: n={n} nq={nq} nkv={nkv} hd={hd}: {e}")
            })?;
        } else {
            if let Some(t) = tp1 {
                crate::layers::qwen3_attention::add_attn_phase_us(
                    1,
                    t.elapsed().as_micros() as u64,
                );
            }
            let tp2 = std::time::Instant::now();
            ops::prefill_attention_64(
                ctx.gpu,
                self.prefill_attn_64_k,
                q_contiguous,
                k_contiguous,
                v_contiguous,
                attn_out,
                flash_seq_len,
                flash_batch,
                nq,
                nkv,
                hd,
                inv_sqrt_d,
                true,
                self.sliding_window.unwrap_or(0),
                stream,
            )
            .map_err(|e| {
                anyhow::anyhow!("flash_attn_64 failed: n={n} nq={nq} nkv={nkv} hd={hd}: {e}")
            })?;
            crate::layers::qwen3_attention::add_attn_phase_us(2, tp2.elapsed().as_micros() as u64);
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
        Ok(wide_head_path)
    }

    /// 2026-09-26: The `attn_out_pre_gate` op dump, the QSA row rewrite, the
    /// output gate and the per-head gate of `prefill_attention_with_cache_skip`,
    /// in that order.
    pub(super) fn cache_skip_gates(
        &self,
        state: &mut dyn crate::layer::LayerState,
        ctx: &ForwardContext,
        kv_cache: &PagedKvCache,
        seq_block_table: &[u32],
        normed: DevicePtr,
        qg_out: DevicePtr,
        q_contiguous: DevicePtr,
        attn_out: DevicePtr,
        n: u32,
        h: u32,
        nq: u32,
        hd: u32,
        q_proj_dim: usize,
        q_dim: usize,
        num_tokens: usize,
        bs: usize,
        bf16: usize,
        inv_sqrt_d: f32,
        stream: u64,
    ) -> Result<()> {
        // 2026-09-25: METRALE_OP_DUMP: the attention output before the gates.
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

        // 2026-09-25: QSA layers with more rows than `inert_bound`:
        // `prefill_select` rewrites the selected rows' attention output from
        // the paged cache written above (the rule is in `qsa_select.rs`). It
        // runs before the gates, while `q_contiguous` still holds Q.
        if let Some(ref qsa) = self.qsa
            && num_tokens > qsa.inert_bound()
        {
            let qsa_st =
                crate::layers::qwen3_attention::helpers::qsa_seq_state(qsa, state, ctx.gpu)?;
            qsa.prefill_select(
                qsa_st,
                normed,
                q_contiguous,
                attn_out,
                kv_cache.k_pool_ptr(self.attn_layer_idx),
                kv_cache.v_pool_ptr(self.attn_layer_idx),
                seq_block_table,
                0,
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
        // sigmoid of the gate half of `qg_out`.
        if self.gated {
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
            // 2026-09-25: `q_contiguous` is scratch for the `[n, nq]` BF16 gate;
            // nothing reads Q after attention and QSA.
            let gate_buf = q_contiguous;
            // 2026-09-25: `normed [n, H] x g_proj^T [H, nq] -> gate_buf [n, nq]`,
            // through cuBLASLt when the `attn` scope is armed, otherwise
            // `dense_gemm_tc`.
            if ctx.dispatch.cublas.attn {
                ops::cublas_bf16_proj_dense(normed, g_proj.weight, gate_buf, n, nq, h, stream)?;
            } else {
                ops::dense_gemm_tc(
                    ctx.gpu,
                    self.dense_gemm_tc_k,
                    normed,
                    g_proj,
                    gate_buf,
                    n,
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
                        n,
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
                        n,
                        stream,
                    )?;
                }
            }
        }
        Ok(())
    }
}
