// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `forward_batch_position`, one drafter position over n rows,
//! and `mtp_tc_lm_head`, the predicate that picks its LM-head kernel.
//!
//! Owner: model-layers (MTP head).
//! Invariants:
//! - The per-sequence state (`seq_len`, `last_pair_key`) advances only after
//!   the whole position has run and its ids have been read back.

use super::*;

mod propose;

impl MtpHead {
    /// 2026-09-25: One draft position for n sequences, the n-row counterpart
    /// of `forward_one`. Row i embeds `tokens[i]`, takes `hiddens[i]` as its
    /// hidden input at RoPE position `positions[i]`, appends one drafter KV
    /// row, and receives its argmax id in `out_ids[i]`. The ids, and with
    /// `out_lp` the top-1 log-probabilities, come back in one `copy_d2h`.
    /// `target_rows` says whether `hiddens` are target rows
    /// (`target_postnorm_row`). Row i of each arena buffer sits at
    /// `i * row width`.
    #[allow(clippy::too_many_arguments)]
    fn forward_batch_position(
        &self,
        tokens: &[u32],
        hiddens: &[DevicePtr],
        positions: &[usize],
        states: &mut [&mut MtpProposerState],
        ctx: &ForwardContext,
        stream: u64,
        out_ids: &mut [u32],
        out_lp: Option<&mut [f32]>,
        target_rows: bool,
    ) -> Result<()> {
        let n = tokens.len();
        let h = ctx.config.hidden_size;
        let nq = ctx.config.num_attention_heads as u32;
        let nkv = ctx.config.num_key_value_heads as u32;
        let hd = ctx.config.head_dim as u32;
        let eps = ctx.config.rms_norm_eps as f32;
        let bf16 = 2usize;
        let gpu = ctx.gpu;

        let q_dim = (nq * hd) as usize;
        let qg_dim = q_dim * 2;
        let kv_dim = (nkv * hd) as usize;

        // 2026-09-25: 1. Embed the n tokens into `ssm_qkvz`.
        let embeds = ctx.buffers.ssm_qkvz();
        for (i, &t) in tokens.iter().enumerate() {
            let src = self.embed_tokens.weight.offset(t as usize * h * bf16);
            gpu.copy_d2d_async(src, embeds.offset(i * h * bf16), h * bf16, stream)?;
        }

        // 2026-09-25: 2. Pre-fc norms. The embeddings are contiguous and take
        // one n-row launch; `hiddens` are separate pointers (verify-stash rows
        // on the first draft), so they are normed one row at a time into the
        // contiguous `ssm_gates`.
        let normed_embed = ctx.buffers.ssm_deinterleaved();
        ops::rms_norm(
            gpu,
            self.rms_norm_k,
            embeds,
            &self.pre_fc_norm_embedding,
            normed_embed,
            n as u32,
            h as u32,
            eps,
            stream,
        )?;
        let normed_hidden = ctx.buffers.ssm_gates();
        // 2026-09-25: `ssm_ba` is written only in step 3, so row i's first
        // half can hold its target-final-normed hidden until then.
        let concat_scratch = ctx.buffers.ssm_ba();
        for (i, &hp) in hiddens.iter().enumerate() {
            let hp = self.target_postnorm_row(
                ctx,
                target_rows,
                hp,
                concat_scratch.offset(i * 2 * h * bf16),
                stream,
            )?;
            ops::rms_norm(
                gpu,
                self.rms_norm_k,
                hp,
                &self.pre_fc_norm_hidden,
                normed_hidden.offset(i * h * bf16),
                1,
                h as u32,
                eps,
                stream,
            )?;
        }

        // 2026-09-25: 3. Per-row concat [normed_embed_i | normed_hidden_i]
        // into `ssm_ba` as [n, 2h].
        let concat = ctx.buffers.ssm_ba();
        for i in 0..n {
            ops::bf16_concat(
                gpu,
                self.bf16_concat_k,
                normed_embed.offset(i * h * bf16),
                normed_hidden.offset(i * h * bf16),
                concat.offset(i * 2 * h * bf16),
                h as u32,
                stream,
            )?;
        }

        // 2026-09-25: 4. fc: [n, 2h] -> [n, h] into `hidden_states`, copied to
        // `residual`.
        let hidden = ctx.buffers.hidden_states();
        // 2026-09-25: The batched-propose scope admits only BF16 fc/k/v
        // (`batch_caps::batch_weight_layout_ok`); q/o and the dense FFN may be
        // weight-only NVFP4 and go through `proj_rows`.
        let (fc_w, k_w, v_w) = match (&self.fc, &self.k_proj, &self.v_proj) {
            (ProjectionWeight::Bf16(fc), ProjectionWeight::Bf16(k), ProjectionWeight::Bf16(v)) => {
                (fc, k, v)
            }
            _ => anyhow::bail!("propose_batch: non-BF16 fc/k/v (can_propose_batch lied)"),
        };
        let (q_w, o_w) = (&self.q_proj, &self.o_proj);
        self.gemm_rows(
            gpu,
            concat,
            fc_w,
            hidden,
            n,
            h as u32,
            (2 * h) as u32,
            stream,
        )?;
        let residual = ctx.buffers.residual();
        gpu.copy_d2d_async(hidden, residual, n * h * bf16, stream)?;

        // 2026-09-25: 5. Input layernorm [n, h] into `norm_output`.
        let normed = ctx.buffers.norm_output();
        ops::rms_norm(
            gpu,
            self.rms_norm_k,
            hidden,
            &self.input_layernorm,
            normed,
            n as u32,
            h as u32,
            eps,
            stream,
        )?;

        // 2026-09-25: 6. Q+gate, K and V projections: q [n, qg] at
        // `qkv_output`, then k [n, kv], then v [n, kv].
        let q_out = ctx.buffers.qkv_output();
        let k_out = q_out.offset(n * qg_dim * bf16);
        let v_out = k_out.offset(n * kv_dim * bf16);
        self.proj_rows(gpu, normed, q_w, q_out, n, qg_dim as u32, h as u32, stream)?;
        self.gemm_rows(gpu, normed, k_w, k_out, n, kv_dim as u32, h as u32, stream)?;
        self.gemm_rows(gpu, normed, v_w, v_out, n, kv_dim as u32, h as u32, stream)?;

        // 2026-09-25: Per-row Q/gate deinterleave, with `forward_one`'s
        // arguments.
        let deint_k = self.deinterleave_qg_k.unwrap();
        for i in 0..n {
            ops::deinterleave_qg(
                gpu,
                deint_k,
                q_out.offset(i * qg_dim * bf16),
                1,
                nq,
                hd,
                nq * hd * 2,
                stream,
            )?;
        }
        // 2026-09-25: The Q norm runs once per sequence. After the deinterleave
        // each row is [q (q_dim) | gate (q_dim)] at stride qg_dim, so one
        // packed n*nq-row launch would normalize sequence 0's gate as Q heads
        // and never reach the later sequences' Q. K is packed [n, kv_dim] and
        // takes one launch.
        for i in 0..n {
            let q_row = q_out.offset(i * qg_dim * bf16);
            ops::rms_norm(
                gpu,
                self.rms_norm_k,
                q_row,
                &self.q_norm,
                q_row,
                nq,
                hd,
                eps,
                stream,
            )?;
        }
        ops::rms_norm(
            gpu,
            self.rms_norm_k,
            k_out,
            &self.k_norm,
            k_out,
            n as u32 * nkv,
            hd,
            eps,
            stream,
        )?;

        // 2026-09-25: 7. Per sequence: attention metadata, RoPE, KV write,
        // paged attention and the sigmoid gate.
        let mut kv_cache = self.kv_cache.lock();
        let bs = kv_cache.block_size();
        let scratch = ctx.buffers.scratch();
        let attn_out = ctx.buffers.attn_output();
        let inv_sqrt_d = 1.0f32 / (hd as f32).sqrt();
        let kv_stride = nkv * hd;
        for i in 0..n {
            let state = &mut *states[i];
            let blocks_needed = (state.seq_len / bs) + 1;
            while state.block_table.len() < blocks_needed {
                state.block_table.push(kv_cache.alloc_block()?);
            }
            let meta_base = self.propose_meta.offset(i * self.propose_meta_stride);
            let block_idx = state.block_table[state.seq_len / bs];
            let global_slot = (block_idx as i64) * (bs as i64) + ((state.seq_len % bs) as i64);
            // 2026-09-25: The region is one `propose_meta_stride`, sized at
            // construction from `max_seq_len`
            // (`batch_caps::propose_meta_stride_env`). `pack_mtp_attn_meta`
            // refuses a block table that does not fit, with an error text that
            // `mtp_bootstrap_step.rs` matches (see `mtp_meta.rs`).
            let meta_buf = pack_mtp_attn_meta(
                positions[i] as u32,
                global_slot,
                (state.seq_len + 1) as i32,
                &state.block_table,
                self.propose_meta_stride,
            )?;
            gpu.copy_h2d_async(&meta_buf, meta_base, stream)?;

            let q_row = q_out.offset(i * qg_dim * bf16);
            let k_row = k_out.offset(i * kv_dim * bf16);
            let v_row = v_out.offset(i * kv_dim * bf16);
            ops::rope(
                gpu,
                self.rope_k,
                q_row,
                k_row,
                meta_base,
                1,
                nq,
                nkv,
                hd,
                ctx.config.rotary_dim() as u32,
                ctx.config.rope_theta as f32,
                stream,
            )?;
            ops::reshape_and_cache(
                gpu,
                self.reshape_cache_k,
                k_row,
                v_row,
                kv_cache.k_pool_ptr(self.attn_layer_idx),
                kv_cache.v_pool_ptr(self.attn_layer_idx),
                meta_base.offset(8),
                1,
                nkv,
                hd,
                bs as u32,
                kv_stride,
                kv_stride,
                kv_cache.cache_stride() as u64,
                stream,
            )?;
            ops::paged_decode_attn_bf16(
                gpu,
                self.paged_decode_k,
                q_row,
                kv_cache.k_pool_ptr(self.attn_layer_idx),
                kv_cache.v_pool_ptr(self.attn_layer_idx),
                attn_out.offset(i * q_dim * bf16),
                meta_base.offset(256),
                meta_base.offset(16),
                state.block_table.len() as u32,
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
            ops::sigmoid_gate_mul(
                gpu,
                self.sigmoid_gate_mul_k,
                attn_out.offset(i * q_dim * bf16),
                q_row.offset(q_dim * bf16),
                attn_out.offset(i * q_dim * bf16),
                nq * hd,
                stream,
            )?;
        }
        drop(kv_cache);

        // 2026-09-25: 8. O projection [n, q_dim] -> [n, h], then residual add
        // and post-attention norm into `norm_output`.
        let o_out = ctx.buffers.norm_output();
        self.proj_rows(gpu, attn_out, o_w, o_out, n, h as u32, q_dim as u32, stream)?;
        let normed2 = ctx.buffers.norm_output();
        ops::residual_add_rms_norm(
            gpu,
            self.residual_add_rms_norm_k,
            hidden,
            o_out,
            &self.post_attn_layernorm,
            normed2,
            residual,
            n as u32,
            h as u32,
            eps,
            stream,
        )?;

        // 2026-09-25: 9. FFN for the n rows (`forward_batch_ffn`): the dense
        // MLP as n-row projections, or the native-FP8 MoE as one grouped
        // decode; `hidden += ffn(normed2)`.
        self.ffn_rows(normed2, hidden, n, ctx, stream)?;

        // 2026-09-25: 10. Final norm [n, h], batched LM head, per-row argmax.
        let final_normed = ctx.buffers.norm_output();
        ops::rms_norm(
            gpu,
            self.rms_norm_k,
            hidden,
            &self.norm,
            final_normed,
            n as u32,
            h as u32,
            eps,
            stream,
        )?;
        let v = if self.mtp_vocab_size > 0 {
            self.mtp_vocab_size.min(ctx.config.vocab_size as u32)
        } else {
            ctx.config.vocab_size as u32
        };
        let logits = ctx.buffers.logits();
        // 2026-09-25: The tile GEMM on the transposed LM-head twin runs when
        // the tensor-core GEMV is not taken, n >= 5, `w4a16_gemm_t` resolved
        // and the twin is present (`MtpHead::new` and `impl_a1.rs` drop it
        // for a dedicated draft head and under the kill switches). `ldb` is
        // the twin's row stride, padded to a multiple of 128 (`impl_a1.rs`),
        // not `v`. Otherwise `w4a16_gemv_batchm` runs on the narrowest
        // covering tier.
        if !self.mtp_tc_lm_head(gpu, n, v, h as u32)
            && n >= 5
            && self.w4a16_gemm_t_k.0 != 0
            && let Some((ref nvfp4_t, ldb)) = self.lm_head_nvfp4_t
        {
            ops::w4a16_gemm_n128_ldb(
                gpu,
                self.w4a16_gemm_t_k,
                final_normed,
                nvfp4_t,
                logits,
                n as u32,
                v,
                h as u32,
                ldb,
                stream,
            )?;
        } else {
            ops::w4a16_gemv_batchm(
                gpu,
                self.lm_head_batch_kernel(n),
                final_normed,
                &self.lm_head_nvfp4,
                logits,
                n as u32,
                v,
                h as u32,
                stream,
            )?;
        }
        // 2026-09-25: D-Cut confidence. With `out_lp` requested and
        // `argmax_bf16_batch_lp` resolved, that kernel writes the ids and the
        // log-probabilities (at `scratch + LP_SCRATCH_OFF`) and the plain
        // batched argmax does not run.
        let want_lp = out_lp.is_some() && self.argmax_batch_lp_k.0 != 0;
        if want_lp {
            ops::argmax_bf16_batch_lp(
                gpu,
                self.argmax_batch_lp_k,
                logits,
                scratch,
                scratch.offset(LP_SCRATCH_OFF),
                v,
                n as u32,
                v,
                stream,
            )?;
        } else if self.argmax_batch_k.0 != 0 {
            ops::argmax_bf16_batch(
                gpu,
                self.argmax_batch_k,
                logits,
                scratch,
                v,
                n as u32,
                v,
                stream,
            )?;
        } else {
            for i in 0..n {
                ops::argmax_bf16(
                    gpu,
                    self.argmax_k,
                    logits.offset(i * v as usize * bf16),
                    scratch.offset(i * 4),
                    v,
                    stream,
                )?;
            }
        }

        // 2026-09-25: 11. One `copy_d2h` reads the n ids and, when the LP
        // kernel ran, the log-probabilities up to `LP_SCRATCH_OFF + n * 4`.
        let d2h_len = if want_lp {
            LP_SCRATCH_OFF + n * 4
        } else {
            n * 4
        };
        let mut buf = vec![0u8; d2h_len];
        gpu.copy_d2h(scratch, &mut buf)?;
        for (i, id) in out_ids.iter_mut().enumerate() {
            *id = u32::from_le_bytes([buf[i * 4], buf[i * 4 + 1], buf[i * 4 + 2], buf[i * 4 + 3]]);
        }
        if let Some(lp) = out_lp {
            for (i, slot) in lp.iter_mut().enumerate().take(n) {
                *slot = if want_lp {
                    let o = LP_SCRATCH_OFF + i * 4;
                    f32::from_le_bytes([buf[o], buf[o + 1], buf[o + 2], buf[o + 3]])
                } else {
                    // 2026-09-25: 0.0 is log(1): without the LP kernel no
                    // row is reported as uncertain.
                    0.0
                };
            }
        }

        // 2026-09-25: 12. `forward_one`'s tail, per row.
        for (i, state) in states.iter_mut().enumerate() {
            state.seq_len += 1;
            state.last_pair_key = Some(positions[i].saturating_sub(1));
        }
        Ok(())
    }

    /// 2026-09-25: True when the tensor-core drafter path is on
    /// (`ops::dense_gemv_tc::mtp_tc_enabled`) and `gemv_tc::tc_kernel`
    /// resolves a `w4a16_gemv_tc8`/`tc16` entry for this shape; the LM head
    /// then stays on `w4a16_gemv_batchm`, which launches that entry, instead
    /// of the tile twin. The dispatch and the propose log line both call it.
    fn mtp_tc_lm_head(
        &self,
        gpu: &dyn metrale_gpu_runtime::gpu::GpuBackend,
        n: usize,
        v: u32,
        h: u32,
    ) -> bool {
        ops::dense_gemv_tc::mtp_tc_enabled()
            && ops::gemv_tc::tc_kernel(gpu, n as u32, v, h).is_some()
    }
}
