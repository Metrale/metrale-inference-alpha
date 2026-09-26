// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Multi-sequence decode for the GDN layer under an mHC highway
//! (`decode_multi_seq_inner_hc`).
//!
//! Owner: model-layers, GDN/SSM layer (`qwen3_ssm`).
//! Invariants: none beyond the types.
//!
//! The highway replaces the layer's residual bookkeeping, so the non-hc
//! path's `rms_norm_residual` and `residual_add_rms_norm` steps do not run.
//! Per layer:
//! - `hc_expand` over the `n` rows, on the first model layer only;
//! - PLE per row, each against its own `PleSeqState`;
//! - `hc_pre`, writing the mixed rows into `norm_output`;
//! - the GDN: the batched-projection mixer (rows in `moe_output`), or
//!   `ssm_forward` per row (rows copied into `hidden`) when it declines;
//! - `hc_post`;
//! - `hc_pre` again, into `norm_output`;
//! - the FFN: `forward_k2`/`forward_k3` at n = 2/3 (rows in `moe_output`),
//!   else `forward` per row (rows copied into `hidden`);
//! - `hc_post`.

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::super::Qwen3SsmLayer;
use crate::layer::{ForwardContext, LayerState, SsmLayerState};
use crate::layers::ops;

impl Qwen3SsmLayer {
    #[allow(clippy::too_many_arguments)]
    pub(in super::super) fn decode_multi_seq_inner_hc<'a, 'b: 'a>(
        &self,
        hidden: DevicePtr,
        num_seqs: usize,
        states: &'a mut [&'b mut (dyn LayerState + 'static)],
        seq_lens: &[usize],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size;
        let eps = ctx.config.rms_norm_eps as f32;
        let n = num_seqs;
        let bf16 = 2usize;
        let hc = self
            .hc
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("decode_multi_seq_inner_hc without mHC weights"))?;
        let hc_mult = hc.hc_mult as u32;
        let streams = ctx.buffers.hc_streams();
        let post = ctx.buffers.hc_post();
        let comb = ctx.buffers.hc_comb();
        let normed = ctx.buffers.norm_output();
        let moe_out = ctx.buffers.moe_output();

        if hc.is_first_model_layer {
            ops::hc_expand(
                ctx.gpu,
                self.hc_expand_k,
                hidden,
                streams,
                n as u32,
                h as u32,
                hc_mult,
                stream,
            )?;
        }

        // 2026-09-25: PLE per row: row `i` takes its token id from
        // `ctx.host_token_ids` and its own carry, and `forward_row` runs the
        // host half inline. The highway rows are FP32, `hc_mult * h` wide.
        if let Some(ple) = self.ple.as_ref() {
            let host = ctx.host_token_ids.ok_or_else(|| {
                anyhow::anyhow!("hc multi-seq decode: PLE needs host_token_ids threaded")
            })?;
            for (i, state) in states.iter_mut().enumerate().take(n) {
                // 2026-09-25: Padding rows (seq_len 0) carry a dummy state
                // without a PLE carry, so they are skipped.
                if seq_lens.get(i).copied() == Some(0) {
                    continue;
                }
                anyhow::ensure!(
                    host.len() > i,
                    "hc multi-seq decode: {} host ids for real seq row {i}",
                    host.len()
                );
                let ssm = state
                    .as_any_mut()
                    .downcast_mut::<SsmLayerState>()
                    .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState for seq {i}"))?;
                let st = ssm.ple.as_mut().ok_or_else(|| {
                    anyhow::anyhow!("PLE multi-seq decode before prefill: no seq state {i}")
                })?;
                ple.forward_row(
                    st,
                    streams.offset(i * (hc_mult as usize) * h * 4),
                    &host[i..i + 1],
                    ctx,
                    stream,
                )?;
            }
        }

        // 2026-09-25: `hc_pre` writes the mixed, normed rows into
        // `norm_output`, where both the batched mixer and the per-row
        // fallback read them.
        ops::hc_pre_site(
            ctx.gpu,
            self.hc_pre_k,
            streams,
            &hc.attn,
            hc,
            normed,
            post,
            comb,
            ctx.buffers.hc_lowrank_scratch(),
            n as u32,
            h as u32,
            eps,
            stream,
        )?;
        // 2026-09-25: The batched-projection mixer; its output rows land in
        // `moe_output[0..n]`.
        let gdn_rows = if self.try_decode_multi_seq_ssm_batched(
            hidden,
            DevicePtr::NULL,
            n,
            states,
            true,
            ctx,
            stream,
        )? {
            moe_out
        } else {
            // 2026-09-25: Per-row fallback. `ssm_forward` returns every row in
            // the same `moe_output` buffer, so each is copied into `hidden`,
            // which the highway does not need.
            for (i, state) in states.iter_mut().enumerate().take(n) {
                let ssm_state = state
                    .as_any_mut()
                    .downcast_mut::<SsmLayerState>()
                    .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState for seq {i}"))?;
                let ssm_out =
                    self.ssm_forward(normed.offset(i * h * bf16), ssm_state, ctx, stream, false)?;
                ctx.gpu
                    .copy_d2d_async(ssm_out, hidden.offset(i * h * bf16), h * bf16, stream)?;
            }
            hidden
        };
        ops::hc_post_site(
            ctx.gpu,
            self.hc_post_k,
            hc,
            gdn_rows,
            streams,
            post,
            comb,
            streams,
            n as u32,
            h as u32,
            stream,
        )?;

        // 2026-09-25: `hc_pre` writes the mixed rows into `norm_output`, where
        // the batched FFN kernels read their input.
        ops::hc_pre_site(
            ctx.gpu,
            self.hc_pre_k,
            streams,
            &hc.ffn,
            hc,
            normed,
            post,
            comb,
            ctx.buffers.hc_lowrank_scratch(),
            n as u32,
            h as u32,
            eps,
            stream,
        )?;
        let moe_rows = match n {
            2 => {
                self.ffn.forward_k2(normed, ctx, stream)?;
                moe_out
            }
            3 => {
                self.ffn.forward_k3(normed, ctx, stream)?;
                moe_out
            }
            _ => {
                // 2026-09-25: Per-row loop, staged into `hidden` rows; any GDN
                // rows there were consumed by the `hc_post_site` above.
                for i in 0..n {
                    let out = self.ffn.forward(normed.offset(i * h * bf16), ctx, stream)?;
                    ctx.gpu
                        .copy_d2d_async(out, hidden.offset(i * h * bf16), h * bf16, stream)?;
                }
                hidden
            }
        };
        ops::hc_post_site(
            ctx.gpu,
            self.hc_post_k,
            hc,
            moe_rows,
            streams,
            post,
            comb,
            streams,
            n as u32,
            h as u32,
            stream,
        )?;
        Ok(())
    }
}
