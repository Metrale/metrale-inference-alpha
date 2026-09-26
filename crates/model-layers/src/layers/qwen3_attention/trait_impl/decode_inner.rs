// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Single-token decode for `Qwen3AttentionLayer`: input norm,
//! attention, the FFN (single, dual dense-plus-MoE, or with a LongCat shortcut
//! carry) and the residual adds. Hyper-connection layers take `decode_inner_hc`.
//! `TransformerLayer::decode` calls `decode_inner` directly.
//!
//! Owner: model-layers (qwen3 attention).
//! Invariants:
//! - The `diag_norm` diagnostics never run during graph capture: both gates
//!   include `!ctx.graph_capture`.

use anyhow::Result;
use metrale_cache::kv_cache::PagedKvCache;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

use super::super::Qwen3AttentionLayer;
use super::{diag_norm, diag_norm_f32};
use crate::layer::{ForwardContext, LayerState};
use crate::layers::ops;

mod hc;

impl Qwen3AttentionLayer {
    pub(super) fn decode_inner(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len: usize,
        block_table: &mut Vec<u32>,
        disk_block_ids: &mut Vec<u32>,
        disk_last_offloaded_per_layer: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        // 2026-09-25: With hyper-connections the persistent multi-stream state is
        // `hc_streams`, and `hidden` is single-stream scratch.
        if self.hc.is_some() {
            return self.decode_inner_hc(
                hidden,
                residual,
                state,
                kv_cache,
                seq_len,
                block_table,
                disk_block_ids,
                disk_last_offloaded_per_layer,
                ctx,
                stream,
            );
        }

        let h = ctx.config.hidden_size;
        let eps = ctx.config.rms_norm_eps as f32;
        // 2026-09-25: `diag_norm` synchronises and copies to the host, which a
        // stream capture cannot contain, so it is off during capture.
        let gemma4_diag =
            ctx.config.model_type == "gemma4" && ctx.levers.gemma4_diag && !ctx.graph_capture;
        let diag_hidden =
            |gpu: &dyn GpuBackend, ptr: DevicePtr, n: usize, stream: u64, label: &str| {
                diag_norm(gpu, ptr, n, stream, label);
            };

        let normed = ctx.buffers.norm_output();
        if gemma4_diag {
            diag_hidden(
                ctx.gpu,
                hidden,
                h,
                stream,
                &format!("L{:02} hidden_in", self.attn_layer_idx),
            );
        }
        ops::rms_norm_residual(
            ctx.gpu,
            self.rms_norm_residual_k,
            hidden,
            &self.input_norm,
            normed,
            residual,
            1,
            h as u32,
            eps,
            stream,
        )?;
        if gemma4_diag {
            diag_norm(
                ctx.gpu,
                normed,
                h,
                stream,
                &format!("L{:02} normed", self.attn_layer_idx),
            );
        }

        let attn_out = self.attention_forward(
            state,
            normed,
            seq_len,
            block_table,
            disk_block_ids,
            disk_last_offloaded_per_layer,
            kv_cache,
            ctx,
            stream,
        )?;
        // 2026-09-25: Under tensor parallelism each rank's o_proj output is a
        // partial sum over the full hidden size; the all-reduce completes it
        // before the residual add.
        if ctx.config.tp_world_size > 1
            && let Some(comm) = ctx.comm
        {
            let bytes = h * 2;
            comm.all_reduce_async(attn_out.0, bytes, stream)?;
        }
        if gemma4_diag {
            diag_norm(
                ctx.gpu,
                attn_out,
                h,
                stream,
                &format!("L{:02} attn_out", self.attn_layer_idx),
            );
        }

        // 2026-09-25: Post-attention output norm (`post_attn_out_norm`, set by the
        // Gemma-4 loader), before the residual add.
        if let Some(ref post_norm) = self.post_attn_out_norm {
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_w_k,
                attn_out,
                post_norm,
                attn_out,
                1,
                h as u32,
                eps,
                stream,
            )?;
            if gemma4_diag {
                diag_norm(
                    ctx.gpu,
                    attn_out,
                    h,
                    stream,
                    &format!("L{:02} post_attn_normed", self.attn_layer_idx),
                );
            }
        }

        if self.ffn.is_none() {
            ops::residual_add(
                ctx.gpu,
                self.residual_add_k,
                hidden,
                attn_out,
                h as u32,
                stream,
            )?;
            return Ok(());
        }

        // 2026-09-25: `ctx.profile`: synchronise and log the FFN time. This
        // branch runs only `self.ffn`: no FP32-routing norm, no dual MoE and no
        // shortcut carry.
        if ctx.profile {
            use std::time::Instant;
            ctx.gpu.synchronize(stream)?;
            let t0 = Instant::now();

            let normed2 = ctx.buffers.norm_output();
            ops::residual_add_rms_norm(
                ctx.gpu,
                self.residual_add_rms_norm_k,
                hidden,
                attn_out,
                &self.post_attn_norm,
                normed2,
                residual,
                1,
                h as u32,
                eps,
                stream,
            )?;
            let moe_out = self.ffn.forward(normed2, ctx, stream)?;

            if let Some(ref post_norm) = self.post_ffn_out_norm {
                ops::rms_norm(
                    ctx.gpu,
                    self.rms_norm_w_k,
                    moe_out,
                    post_norm,
                    moe_out,
                    1,
                    h as u32,
                    eps,
                    stream,
                )?;
            }

            ctx.gpu.synchronize(stream)?;
            let moe_us = t0.elapsed().as_micros();
            tracing::info!("  Attn-MoE: {:.1}ms", moe_us as f64 / 1000.0);

            ops::residual_add(
                ctx.gpu,
                self.residual_add_k,
                hidden,
                moe_out,
                h as u32,
                stream,
            )?;
            if let Some(scalar) = self.layer_scalar {
                self.apply_layer_scalar(ctx.gpu, hidden, h, scalar, stream)?;
            }
            return Ok(());
        }

        let normed2 = ctx.buffers.norm_output();
        // 2026-09-25: When the MoE's FP32 routing is active (`fp32_routing_active`,
        // lever `fp32_routing`), the norm also writes an FP32 copy of the MoE
        // input (`moe_router_in_f32`) for the router.
        if self.ffn.fp32_routing_active(ctx.levers) && self.residual_add_rms_norm_gatef32_k.0 != 0 {
            ops::residual_add_rms_norm_gatef32(
                ctx.gpu,
                self.residual_add_rms_norm_gatef32_k,
                hidden,
                attn_out,
                &self.post_attn_norm,
                normed2,
                ctx.buffers.moe_router_in_f32(),
                residual,
                1,
                h as u32,
                eps,
                stream,
            )?;
        } else {
            ops::residual_add_rms_norm(
                ctx.gpu,
                self.residual_add_rms_norm_k,
                hidden,
                attn_out,
                &self.post_attn_norm,
                normed2,
                residual,
                1,
                h as u32,
                eps,
                stream,
            )?;
        }

        // 2026-09-25: Dual FFN, a MoE beside the dense FFN (`set_moe_ffn`, Gemma-4
        // loader): hidden += post_ffn_out_norm(post_dense_ffn_norm(dense) +
        // post_moe_out_norm(moe)). Both FFNs return `moe_output`, so the MoE
        // runs first and its normalised output is saved.
        if let (Some(moe_ffn), Some(_pre_norm), Some(post_norm), Some(dense_norm)) = (
            &self.moe_ffn,
            &self.pre_moe_norm,
            &self.post_moe_out_norm,
            &self.post_dense_ffn_norm,
        ) {
            // 2026-09-25: The MoE takes `hidden`: its router reads it as is, and
            // its experts apply the pre-expert norm (`set_pre_expert_norm`, the
            // same weight as `pre_moe_norm`).
            let moe_out = moe_ffn.forward(hidden, ctx, stream)?;
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_w_k,
                moe_out,
                post_norm,
                moe_out,
                1,
                h as u32,
                eps,
                stream,
            )?;
            // 2026-09-25: Saved in the `logits` buffer, because the dense FFN
            // overwrites `moe_output`.
            let moe_saved = ctx.buffers.logits();
            ctx.gpu.copy_d2d_async(moe_out, moe_saved, h * 2, stream)?;

            let dense_out = self.ffn.forward(normed2, ctx, stream)?;
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_w_k,
                dense_out,
                dense_norm,
                dense_out,
                1,
                h as u32,
                eps,
                stream,
            )?;

            ops::residual_add(
                ctx.gpu,
                self.residual_add_k,
                dense_out,
                moe_saved,
                h as u32,
                stream,
            )?;

            if let Some(ref combined_norm) = self.post_ffn_out_norm {
                ops::rms_norm(
                    ctx.gpu,
                    self.rms_norm_w_k,
                    dense_out,
                    combined_norm,
                    dense_out,
                    1,
                    h as u32,
                    eps,
                    stream,
                )?;
            }

            ops::residual_add(
                ctx.gpu,
                self.residual_add_k,
                hidden,
                dense_out,
                h as u32,
                stream,
            )?;
        } else {
            if gemma4_diag {
                diag_norm(
                    ctx.gpu,
                    normed2,
                    h,
                    stream,
                    &format!("L{:02} normed2", self.attn_layer_idx),
                );
            }
            // 2026-09-25: LongCat shortcut MoE, producer side: run on the same
            // normed input as the FFN, add the zero-expert contribution, and copy
            // the result into the carry before `self.ffn` overwrites `moe_output`.
            // The paired sublayer (`shortcut_carry_in`) adds it.
            if let (Some(moe_ffn), Some((carry, cap))) = (&self.moe_ffn, self.shortcut_carry_out) {
                anyhow::ensure!(1 <= cap, "shortcut carry capacity");
                let moe_out = moe_ffn.forward(normed2, ctx, stream)?;
                if let crate::layers::FfnComponent::Moe(m) = moe_ffn {
                    m.apply_zero_expert(moe_out, normed2, 1, ctx, stream)?;
                }
                ctx.gpu.copy_d2d_async(moe_out, carry, h * 2, stream)?;
            }
            let dense_out = self.ffn.forward(normed2, ctx, stream)?;
            if gemma4_diag {
                diag_norm(
                    ctx.gpu,
                    dense_out,
                    h,
                    stream,
                    &format!("L{:02} dense_out", self.attn_layer_idx),
                );
            }
            if let Some(ref post_norm) = self.post_ffn_out_norm {
                ops::rms_norm(
                    ctx.gpu,
                    self.rms_norm_w_k,
                    dense_out,
                    post_norm,
                    dense_out,
                    1,
                    h as u32,
                    eps,
                    stream,
                )?;
                if gemma4_diag {
                    diag_norm(
                        ctx.gpu,
                        dense_out,
                        h,
                        stream,
                        &format!("L{:02} post_ffn_normed", self.attn_layer_idx),
                    );
                }
            }
            ops::residual_add(
                ctx.gpu,
                self.residual_add_k,
                hidden,
                dense_out,
                h as u32,
                stream,
            )?;
            // 2026-09-25: LongCat shortcut, consumer side: add the paired
            // sublayer's carried output, last.
            if let Some((carry, _cap)) = self.shortcut_carry_in {
                ops::residual_add(
                    ctx.gpu,
                    self.residual_add_k,
                    hidden,
                    carry,
                    h as u32,
                    stream,
                )?;
            }
        }

        if gemma4_diag {
            diag_hidden(
                ctx.gpu,
                hidden,
                h,
                stream,
                &format!("L{:02} post_residual", self.attn_layer_idx),
            );
        }

        // 2026-09-25: The Gemma-4 per-layer scalar multiplies the whole hidden
        // state at the end of the layer.
        if let Some(scalar) = self.layer_scalar {
            self.apply_layer_scalar(ctx.gpu, hidden, h, scalar, stream)?;
            if gemma4_diag {
                diag_hidden(
                    ctx.gpu,
                    hidden,
                    h,
                    stream,
                    &format!(
                        "L{:02} post_layer_scalar(scalar={:.4})",
                        self.attn_layer_idx, scalar
                    ),
                );
            }
        }

        Ok(())
    }
}
