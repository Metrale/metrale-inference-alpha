// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: DeepSeek-V4-Flash single-token decode: the MLA path's low-rank Q projection
//! (wq_a, norm, wq_b), a direct KV projection with K = V, and a grouped low-rank O projection
//! (wo_a, wo_b). On a compressor layer it also appends compressed blocks at window boundaries.
//!
//! Owner: model-layers attention decode.
//! Invariants:
//! - The compressed-pool append runs only on a compressor layer with `pos` set, one sequence and
//!   no graph capture.
//! - This module only raises `v4_comp_pool_filled`, to `w + 1` for a window `w` at or above the
//!   current count.

use anyhow::Result;
use metrale_cache::kv_cache::PagedKvCache;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::super::Qwen3AttentionLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;

mod comp_append;
mod derotate;
mod proj;

impl Qwen3AttentionLayer {
    /// 2026-09-25: Run the DeepSeek-V4-Flash decode chain. Returns the O-projection output
    /// (`ctx.buffers.qkv_output()`).
    ///
    /// Visible to the whole `qwen3_attention` module because the multi-sequence path
    /// (`trait_impl::multi_seq::mla`) runs this single-token chain once per token.
    pub(in crate::layers::qwen3_attention) fn attention_forward_v4(
        &self,
        kv_cache: &mut PagedKvCache,
        ctx: &ForwardContext,
        args: &super::attention_forward_mla::DecodeMlaArgs,
    ) -> Result<DevicePtr> {
        let super::attention_forward_mla::DecodeMlaArgs {
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
            pos,
        } = *args;
        let mla = self
            .mla
            .as_ref()
            .expect("attention_forward_v4 called without MLA config");
        let meta = ctx
            .attn_metadata
            .expect("V4-Flash decode requires pre-uploaded metadata");

        let q_lora = mla.q_lora_rank as u32;
        let mla_rope = mla.rope as u32;
        let o_lora = mla.o_lora_rank as u32;
        let nkv = ctx.config.num_key_value_heads as u32;
        let profile = ctx.profile;
        let diag_all =
            std::env::var("METRALE_DIAG_V4_ALL_LAYERS").is_ok_and(|v| v == "1" || v == "true");
        let diag_this = self.attn_layer_idx == 0 || diag_all;
        macro_rules! prof {
            ($label:expr, $body:expr) => {{
                if profile {
                    let _t = std::time::Instant::now();
                    let _r = $body;
                    ctx.gpu.synchronize(stream)?;
                    tracing::info!("    V4 {}: {:.0}µs", $label, _t.elapsed().as_micros());
                    _r
                } else {
                    $body
                }
            }};
        }

        // 2026-09-25: Decode-time compressed-block append. Copy this token's compressor input
        // (`normed`) into a per-layer BF16 ring, and at each window boundary run the compress
        // pipeline over the ring to append one FP8 pool block. Runs before the Q/K/V compute, which
        // reuses the same scratch buffers. Needs `pos`, one sequence, and no graph capture, because
        // it runs host logic per step.
        if let (Some(pos), Some(comp)) = (pos, mla.compressor.as_ref())
            && meta.num_seqs == 1
            && !ctx.graph_capture
        {
            {
                let ratio = comp.ratio as u32;
                let proj_dim = comp.proj_dim as u32;
                let nope = mla.nope as u32;
                let rope_d = mla_rope;
                let hd_mla = nope + rope_d;
                let hb = h as usize * 2;

                // 2026-09-25: Ring slot `pos % ratio`; the append runs when `pos + 1` completes a
                // window.
                let slot = (pos % ratio) as usize;
                ctx.gpu
                    .copy_d2d_async(normed, comp.ring.offset(slot * hb), hb, stream)?;

                if (pos + 1) % ratio == 0 {
                    self.v4_decode_comp_append(
                        ctx, mla, comp, pos, ratio, proj_dim, nope, rope_d, hd_mla, hb, h, eps,
                        stream,
                    )?;
                }
            }
        }

        let q_latent = ctx.buffers.ssm_ba();
        prof!(
            "wq_a",
            self.v4_wq_a(ctx, mla, normed, q_latent, q_lora, h, stream)
        )?;
        prof!("q_norm", {
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_w_k,
                q_latent,
                &mla.q_a_norm,
                q_latent,
                1,
                q_lora,
                eps,
                stream,
            )
        })?;
        prof!(
            "wq_b",
            self.v4_wq_b(ctx, mla, q_latent, q_out, q_dim, q_lora, stream)
        )?;
        // 2026-09-25: Unweighted per-head RMS norm of Q over head_dim (an all-ones weight), before
        // RoPE.
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_k,
            q_out,
            &crate::weight_map::DenseWeight {
                weight: ctx.buffers.norm_unit_w(),
            },
            q_out,
            nq,
            hd,
            eps,
            stream,
        )?;
        if diag_this {
            super::super::trait_impl::diag_norm(
                ctx.gpu,
                q_out,
                q_dim as usize,
                stream,
                &format!("V4-decode L{} Q after q_b_norm", self.attn_layer_idx),
            );
        }

        let kv_dim = nkv * hd;
        prof!(
            "wkv",
            self.v4_wkv_a(ctx, mla, normed, k_out, kv_dim, h, stream)
        )?;
        // 2026-09-25: Weighted RMS norm of the KV projection before RoPE, over `nkv` rows of
        // `kv_dim / nkv`.
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_w_k,
            k_out,
            &mla.kv_a_norm,
            k_out,
            nkv,
            kv_dim / nkv,
            eps,
            stream,
        )?;
        // 2026-09-25: V = K, copied before RoPE touches K.
        ctx.gpu
            .copy_d2d_async(k_out, v_out, (kv_dim as usize) * 2, stream)?;
        if diag_this {
            super::super::trait_impl::diag_norm(
                ctx.gpu,
                k_out,
                kv_dim as usize,
                stream,
                &format!("V4-decode L{} K after proj", self.attn_layer_idx),
            );
            super::super::trait_impl::diag_norm(
                ctx.gpu,
                v_out,
                kv_dim as usize,
                stream,
                &format!("V4-decode L{} V after copy", self.attn_layer_idx),
            );
        }

        // 2026-09-25: RoPE for Q and K. The rope dims sit at offset `nope` in each head, so each is
        // extracted, rotated and written back.
        let q_rope_tmp = ctx.buffers.ssm_conv_out_f32();
        // 2026-09-25: Reuses `q_latent`, which wq_b has consumed.
        let k_rope_tmp = q_latent;
        prof!("rope_extract", {
            ops::mla_q_rope_extract_batched(
                ctx.gpu,
                self.mla_q_rope_extract_batched_k,
                q_out,
                q_rope_tmp,
                1,
                nq,
                hd,
                mla.nope as u32,
                mla_rope,
                nq * hd,
                stream,
            )
        })?;
        // 2026-09-25: Extract K's rope channels too (one KV head, stride hd).
        prof!("k_rope_extract", {
            ops::mla_q_rope_extract_batched(
                ctx.gpu,
                self.mla_q_rope_extract_batched_k,
                k_out,
                k_rope_tmp,
                1,
                1,
                hd,
                mla.nope as u32,
                mla_rope,
                hd,
                stream,
            )
        })?;
        prof!("rope", {
            ops::rope_yarn(
                ctx.gpu,
                // 2026-09-25: Interleaved RoPE: adjacent channel pairs (2i, 2i+1).
                self.rope_yarn_interleaved_k,
                q_rope_tmp,
                k_rope_tmp,
                meta.positions,
                1,
                nq,
                1,
                mla_rope,
                mla_rope,
                // 2026-09-25: Layers without a compressor use `main_inv_freq` with mscale 1;
                // compressor layers use `yarn_inv_freq` with the YaRN mscale.
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
            )
        })?;
        prof!("rope_writeback", {
            ops::mla_q_rope_writeback_batched(
                ctx.gpu,
                self.mla_q_rope_writeback_batched_k,
                q_rope_tmp,
                q_out,
                1,
                nq,
                hd,
                mla.nope as u32,
                mla_rope,
                nq * hd,
                stream,
            )
        })?;
        prof!("k_rope_writeback", {
            ops::mla_q_rope_writeback_batched(
                ctx.gpu,
                self.mla_q_rope_writeback_batched_k,
                k_rope_tmp,
                k_out,
                1,
                1,
                hd,
                mla.nope as u32,
                mla_rope,
                hd,
                stream,
            )
        })?;
        if diag_this {
            super::super::trait_impl::diag_norm(
                ctx.gpu,
                k_out,
                kv_dim as usize,
                stream,
                &format!("V4-decode L{} K after RoPE", self.attn_layer_idx),
            );
            super::super::trait_impl::diag_norm(
                ctx.gpu,
                k_out.offset(mla.nope * 2),
                (kv_dim - mla.nope as u32) as usize,
                stream,
                &format!("V4-decode L{} K rope after RoPE", self.attn_layer_idx),
            );
            super::super::trait_impl::diag_norm(
                ctx.gpu,
                q_out.offset(mla.nope * 2),
                (hd - mla.nope as u32) as usize,
                stream,
                &format!("V4-decode L{} Q rope after RoPE", self.attn_layer_idx),
            );
        }

        // 2026-09-25: Assemble the `kv_lora + rope`-wide cache rows from the latent (`v_out`,
        // copied before RoPE) and K's rotated rope part (`k_rope_tmp`).
        let k_cache_assembled = ctx.buffers.ssm_deinterleaved();
        let v_cache_assembled = ctx.buffers.ssm_qkvz();
        let kv_lora = mla.kv_lora_rank as u32;
        let mla_cache_dim = kv_lora + mla_rope;
        prof!("cache_assemble", {
            ops::mla_cache_assemble_batched(
                ctx.gpu,
                self.mla_cache_assemble_batched_k,
                v_out,
                k_rope_tmp,
                k_cache_assembled,
                v_cache_assembled,
                1,
                kv_lora,
                mla_rope,
                mla_cache_dim,
                stream,
            )
        })?;

        prof!("write_kv_cache", {
            self.write_kv_cache(
                ctx.gpu,
                k_cache_assembled,
                v_cache_assembled,
                kv_cache,
                meta.slot,
                1,
                1,
                mla_cache_dim,
                bs as u32,
                mla_cache_dim,
                mla_cache_dim,
                stream,
                ctx.graph_capture,
            )
        })?;

        let attn_out = ctx.buffers.attn_output();
        let inv_sqrt_d = self.effective_attn_scale(hd);
        prof!("paged_attn", {
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
            )
        })?;
        if diag_this {
            super::super::trait_impl::diag_norm(
                ctx.gpu,
                attn_out,
                (nq * hd) as usize,
                stream,
                &format!("V4-decode L{} attn_out", self.attn_layer_idx),
            );
        }

        // 2026-09-25: De-rotate the attention output by the query position. V = K carries rotated
        // rope in its trailing `mla_rope` dims, so the output's rope part is rotated back with the
        // conjugate (negated-sin) kernel before o_proj, through the same extract and writeback.
        self.v4_derotate_attn_out(ctx, mla, &meta, attn_out, nq, hd, mla_rope, stream)?;

        // 2026-09-25: Grouped low-rank O projection. wo_a is block-diagonal: the `nq * hd`
        // attention output splits into `o_groups` groups, each projected `group_in -> o_lora` by
        // its own GEMV over weight rows `[g*o_lora, (g+1)*o_lora)` of `[o_groups*o_lora,
        // group_in]`. wo_b then maps the `o_groups * o_lora` latent to `h`.
        let o_groups = ctx.config.o_groups.max(1) as u32;
        let group_in = (nq * hd) / o_groups;
        let latent_dim = o_groups * o_lora;
        let o_latent = ctx.buffers.o_latent();
        let o_out = ctx.buffers.qkv_output();
        prof!("wo_a_grouped", {
            for g in 0..o_groups {
                let in_g = attn_out.offset((g * group_in) as usize * 2);
                let out_g = o_latent.offset((g * o_lora) as usize * 2);
                if let Some(ref woa_fp8) = mla.wo_a_fp8 {
                    // 2026-09-25: FP8 per group: the group's weight rows (one byte per element) and
                    // its `[o_lora/128, group_in/128]` FP32 block-scale sub-tile.
                    let w_off = (g as usize) * (o_lora as usize) * (group_in as usize);
                    let s_off =
                        (g as usize) * (o_lora as usize / 128) * (group_in as usize / 128) * 4;
                    ops::w8a16_gemv(
                        ctx.gpu,
                        self.w8a16_gemv_k,
                        in_g,
                        woa_fp8.weight.offset(w_off),
                        woa_fp8.row_scale.offset(s_off),
                        out_g,
                        o_lora,
                        group_in,
                        stream,
                    )?;
                } else {
                    let w_g = crate::weight_map::DenseWeight {
                        weight: mla
                            .wo_a
                            .weight
                            .offset((g as usize) * (o_lora as usize) * (group_in as usize) * 2),
                    };
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
            Ok::<(), anyhow::Error>(())
        })?;
        prof!(
            "wo_b",
            self.v4_wo_b(ctx, mla, o_latent, o_out, h, latent_dim, stream)
        )?;
        if diag_this {
            super::super::trait_impl::diag_norm(
                ctx.gpu,
                o_out,
                h as usize,
                stream,
                &format!("V4-decode L{} o_out", self.attn_layer_idx),
            );
        }

        Ok(o_out)
    }
}
