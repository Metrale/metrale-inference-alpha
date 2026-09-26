// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The compressed-attention branch, the output de-rotation and the
//! grouped low-rank O projection of the DeepSeek-V4 cache-skip prefill
//! (`cache_skip_v4.rs`).
//!
//! Owner: model-layers (attention).
//! Invariants: `prefill_attention_cache_skip_v4` calls these helpers where
//! their statements run, so every launch keeps its order and arguments.

use super::cache_skip_v4::V4_WINDOW;
use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;
use metrale_gpu_runtime::kernel_args::KernelLaunch;

use super::super::Qwen3AttentionLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;

impl Qwen3AttentionLayer {
    /// 2026-09-26: The compressed-attention branch of the V4 prefill: compressor
    /// projections, `csa_compress`, RoPE on the compressed rows, the FP8 pool
    /// store and `prefill_attn_compressed` over the windowed raw KV plus the
    /// compressed KV.
    pub(super) fn cache_skip_v4_csa(
        &self,
        ctx: &ForwardContext,
        mla: &crate::layers::qwen3_attention::MlaWeights,
        comp: crate::layers::qwen3_attention::CompressorWeights,
        normed: DevicePtr,
        q_full: DevicePtr,
        k_out: DevicePtr,
        attn_out: DevicePtr,
        n: u32,
        h: u32,
        nq: u32,
        nkv: u32,
        hd_mla: u32,
        nope: u32,
        rope: u32,
        eps: f32,
        stream: u64,
    ) -> Result<()> {
        let ratio = comp.ratio as u32;
        let proj_dim = comp.proj_dim as u32;
        let n_win = n / ratio;
        // 2026-09-25: Compressor projections: kv and gate, `[n, proj_dim]`.
        let kv_comp = ctx.buffers.expert_up_out();
        let gate_comp = ctx.buffers.expert_down_out();
        ops::dense_gemm(
            ctx.gpu,
            self.dense_gemm_k,
            normed,
            &comp.wkv,
            kv_comp,
            n,
            proj_dim,
            h,
            stream,
        )?;
        ops::dense_gemm(
            ctx.gpu,
            self.dense_gemm_k,
            normed,
            &comp.wgate,
            gate_comp,
            n,
            proj_dim,
            h,
            stream,
        )?;
        // 2026-09-25: `csa_compress`: one `hd_mla` row per window (`n_win`
        // rows), then an RMS norm with `comp.norm`.
        let compressed = ctx.buffers.moe_output();
        KernelLaunch::new(ctx.gpu, self.csa_compress_k)
            .grid([n_win, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(kv_comp)
            .arg_ptr(gate_comp)
            .arg_ptr(comp.ape)
            .arg_ptr(compressed)
            .arg_u32(n)
            .arg_u32(ratio)
            .arg_u32(hd_mla)
            .arg_u32(proj_dim)
            .arg_u32(if comp.is_csa { 1 } else { 0 })
            .launch(stream)?;
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_w_k,
            compressed,
            &comp.norm,
            compressed,
            n_win,
            hd_mla,
            eps,
            stream,
        )?;
        // 2026-09-25: `comp_k` = the compressed rows with interleaved RoPE on
        // their last `rope` dims at window position `w * ratio`
        // (`yarn_inv_freq`). The attention launch passes `comp_k` as both the
        // compressed K and V; `comp_v` is read only as the source of the copy below.
        let comp_v = compressed;
        let comp_k = compressed.offset((n_win * hd_mla) as usize * 2);
        ctx.gpu
            .copy_d2d_async(comp_v, comp_k, (n_win * hd_mla) as usize * 2, stream)?;
        let comp_pos: Vec<u8> = (0..n_win).flat_map(|w| (w * ratio).to_le_bytes()).collect();
        let comp_positions = ctx.buffers.ssm_ba();
        ctx.gpu.copy_h2d_async(&comp_pos, comp_positions, stream)?;
        let comp_rope_tmp = ctx.buffers.ssm_conv_out_f32();
        ops::mla_q_rope_extract_batched(
            ctx.gpu,
            self.mla_q_rope_extract_batched_k,
            comp_k,
            comp_rope_tmp,
            n_win,
            1,
            hd_mla,
            nope,
            rope,
            hd_mla,
            stream,
        )?;
        ops::rope_yarn(
            ctx.gpu,
            self.rope_yarn_interleaved_k,
            comp_rope_tmp,
            comp_rope_tmp,
            comp_positions,
            n_win,
            0,
            1,
            rope,
            rope,
            mla.yarn_inv_freq,
            super::super::helpers::yarn_rope_mscale(ctx.config),
            stream,
        )?;
        ops::mla_q_rope_writeback_batched(
            ctx.gpu,
            self.mla_q_rope_writeback_batched_k,
            comp_rope_tmp,
            comp_k,
            n_win,
            1,
            hd_mla,
            nope,
            rope,
            hd_mla,
            stream,
        )?;
        // 2026-09-25: Store the `n_win` compressed K rows as FP8-E4M3 in the
        // layer's pool, blocks [0, n_win). `bf16_to_fp8` is a plain cast; it
        // matches an FP8 cache write only when `k_scale` is 1.0. The
        // `debug_assert!`s check that and the pool size in debug builds only.
        let (k_scale, _v_scale) = self.effective_fp8_scales();
        debug_assert!(
            (k_scale - 1.0).abs() < 1e-6,
            "V4 compressed-pool persist assumes k_scale=1.0 (got {k_scale}); add scale-aware cast"
        );
        let n_elems = (n_win * hd_mla) as usize;
        debug_assert!(
            (n_win as usize) <= comp.pool_blocks,
            "V4 compressed pool overflow: n_win={n_win} > pool_blocks={}",
            comp.pool_blocks
        );
        ops::bf16_to_fp8(
            ctx.gpu,
            self.bf16_to_fp8_k,
            comp_k,
            comp.pool,
            n_elems as u32,
            stream,
        )?;
        // 2026-09-25: The number of compressed blocks written. Decode reads
        // it and appends after it (`decode/attention_forward_v4.rs`). This
        // write starts at pool offset 0 on every prefill call.
        self.v4_comp_pool_filled
            .store(n_win, std::sync::atomic::Ordering::Relaxed);
        KernelLaunch::new(ctx.gpu, self.prefill_attn_compressed_k)
            .grid([nq, n.div_ceil(16), 1])
            .block([128, 1, 1])
            .arg_ptr(q_full)
            .arg_ptr(k_out)
            // 2026-09-25: V is K (rope in the tail), for both the raw and the
            // compressed KV.
            .arg_ptr(k_out)
            .arg_ptr(comp_k)
            .arg_ptr(comp_k)
            .arg_ptr(mla.attn_sink)
            .arg_ptr(attn_out)
            .arg_u32(n)
            .arg_u32(nq)
            .arg_u32(nkv)
            .arg_u32(hd_mla)
            .arg_u32(n_win)
            .arg_u32(ratio)
            .arg_u32(V4_WINDOW)
            .arg_f32(1.0f32 / (hd_mla as f32).sqrt())
            .launch(stream)?;
        Ok(())
    }

    /// 2026-09-26: Inverse interleaved RoPE on the rope dims of the V4 attention
    /// output, at each query position.
    pub(super) fn cache_skip_v4_derotate(
        &self,
        ctx: &ForwardContext,
        mla: &crate::layers::qwen3_attention::MlaWeights,
        meta: &crate::layer::AttnMetadataDev,
        attn_out: DevicePtr,
        n: u32,
        nq: u32,
        hd_mla: u32,
        nope: u32,
        rope: u32,
        stream: u64,
    ) -> Result<()> {
        // 2026-09-25: De-rotate the rope dims of the attention output at each
        // query position (inverse interleaved RoPE) before the O projection.
        {
            let o_rope_tmp = ctx.buffers.ssm_conv_out_f32();
            ops::mla_q_rope_extract_batched(
                ctx.gpu,
                self.mla_q_rope_extract_batched_k,
                attn_out,
                o_rope_tmp,
                n,
                nq,
                hd_mla,
                nope,
                rope,
                nq * hd_mla,
                stream,
            )?;
            ops::rope_yarn(
                ctx.gpu,
                self.rope_yarn_interleaved_inv_k,
                o_rope_tmp,
                o_rope_tmp,
                meta.positions,
                n,
                nq,
                0,
                rope,
                rope,
                // 2026-09-25: The same frequencies and mscale as this layer's Q/K
                // RoPE above.
                if mla.compressor.is_none() {
                    mla.main_inv_freq
                } else {
                    mla.yarn_inv_freq
                },
                if mla.compressor.is_none() {
                    1.0f32
                } else {
                    super::super::helpers::yarn_rope_mscale(ctx.config)
                },
                stream,
            )?;
            ops::mla_q_rope_writeback_batched(
                ctx.gpu,
                self.mla_q_rope_writeback_batched_k,
                o_rope_tmp,
                attn_out,
                n,
                nq,
                hd_mla,
                nope,
                rope,
                nq * hd_mla,
                stream,
            )?;
        }
        Ok(())
    }

    /// 2026-09-26: The V4 grouped low-rank O projection (`wo_a` per group, then
    /// `wo_b`). Returns `o_out`.
    pub(super) fn cache_skip_v4_o_proj(
        &self,
        ctx: &ForwardContext,
        mla: &crate::layers::qwen3_attention::MlaWeights,
        attn_out: DevicePtr,
        n: u32,
        h: u32,
        nq: u32,
        hd_mla: u32,
        o_lora: u32,
        diag_this: bool,
        stream: u64,
    ) -> Result<DevicePtr> {
        // 2026-09-25: Grouped low-rank O projection: `wo_a` is block-diagonal
        // over `o_groups`, run as one GEMV per (token, group); `wo_b` is one GEMM
        // over the `o_groups * o_lora` latent.
        let o_groups = ctx.config.o_groups.max(1) as u32;
        let group_in = (nq * hd_mla) / o_groups;
        let latent_dim = o_groups * o_lora;
        let o_latent = ctx.buffers.o_latent();
        let o_out = ctx.buffers.qkv_output();
        for t in 0..n {
            for g in 0..o_groups {
                let in_g = attn_out.offset(((t * nq * hd_mla) + g * group_in) as usize * 2);
                let w_g = crate::weight_map::DenseWeight {
                    weight: mla
                        .wo_a
                        .weight
                        .offset((g as usize) * (o_lora as usize) * (group_in as usize) * 2),
                };
                let out_g = o_latent.offset(((t * latent_dim) + g * o_lora) as usize * 2);
                ops::dense_gemv(
                    ctx.gpu,
                    self.dense_gemv_k,
                    in_g,
                    &w_g,
                    out_g,
                    o_lora,
                    group_in,
                    stream,
                )?;
            }
        }
        ctx.gpu
            .synchronize(stream)
            .map_err(|e| anyhow::anyhow!("V4 attn: wo_a grouped gemv sync failed: {e}"))?;
        ops::dense_gemm(
            ctx.gpu,
            self.dense_gemm_k,
            o_latent,
            &mla.wo_b,
            o_out,
            n,
            h,
            latent_dim,
            stream,
        )?;
        ctx.gpu
            .synchronize(stream)
            .map_err(|e| anyhow::anyhow!("V4 attn: wo_b gemm sync failed: {e}"))?;
        if diag_this {
            super::super::trait_impl::diag_norm(
                ctx.gpu,
                o_out,
                h as usize,
                stream,
                &format!("V4-prefill L{} o_out token0", self.attn_layer_idx),
            );
            let last_token_offset = ((n - 1) * h * 2) as usize;
            super::super::trait_impl::diag_norm(
                ctx.gpu,
                o_out.offset(last_token_offset),
                h as usize,
                stream,
                &format!("V4-prefill L{} o_out last", self.attn_layer_idx),
            );
        }
        Ok(o_out)
    }
}
