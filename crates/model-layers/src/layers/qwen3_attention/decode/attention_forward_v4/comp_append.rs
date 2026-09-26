// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The window step of the `attention_forward_v4` decode-time compressed-block
//! append: once a token completes a window of the ring, compress that window into one FP8
//! pool block.
//!
//! Owner: model-layers attention decode.
//! Invariants:
//! - `v4_comp_pool_filled` only rises, to `w + 1` for a window `w` at or above the current
//!   count.

use anyhow::Result;

use super::super::super::{CompressorWeights, MlaWeights, Qwen3AttentionLayer};
use crate::layer::ForwardContext;
use crate::layers::ops;

impl Qwen3AttentionLayer {
    /// 2026-09-26: Called by `attention_forward_v4` when token `pos` completes a window: appends
    /// window `(pos + 1) / ratio - 1` to the pool unless the pool already holds it.
    pub(super) fn v4_decode_comp_append(
        &self,
        ctx: &ForwardContext,
        mla: &MlaWeights,
        comp: &CompressorWeights,
        pos: u32,
        ratio: u32,
        proj_dim: u32,
        nope: u32,
        rope_d: u32,
        hd_mla: u32,
        hb: usize,
        h: u32,
        eps: f32,
        stream: u64,
    ) -> Result<()> {
        use std::sync::atomic::Ordering::Relaxed;
        let w = (pos + 1) / ratio - 1;
        let filled = self.v4_comp_pool_filled.load(Relaxed);
        // 2026-09-25: Append only a window the pool does not hold yet. Prefill
        // (`cache_skip_v4.rs`) sets the pool count, seeds the ring with the prompt's
        // partial window, and for CSA seeds `prev_win` with the last full window.
        if w >= filled {
            use metrale_gpu_runtime::kernel_args::KernelLaunch;
            let prev_valid = self.v4_comp_prev_valid.load(Relaxed);
            // 2026-09-25: CSA with a valid previous window compresses 2 x ratio rows
            // (previous window, then the ring) over two blocks and keeps block 1. HCA,
            // or CSA without one, compresses the ring alone and keeps block 0. The
            // Ca/Cb layout is in `csa_compress.cu`.
            let (comp_in, t_rows, launch_win, tgt) = if comp.is_csa && prev_valid {
                ctx.gpu
                    .copy_d2d_async(comp.prev_win, comp.stage, ratio as usize * hb, stream)?;
                ctx.gpu.copy_d2d_async(
                    comp.ring,
                    comp.stage.offset(ratio as usize * hb),
                    ratio as usize * hb,
                    stream,
                )?;
                (comp.stage, 2 * ratio, 2u32, 1u32)
            } else {
                (comp.ring, ratio, 1u32, 0u32)
            };

            // 2026-09-25: Compressor projections kv, gate = W . comp_in, `[t_rows,
            // proj_dim]`.
            let kv_comp = ctx.buffers.expert_up_out();
            let gate_comp = ctx.buffers.expert_down_out();
            ops::dense_gemm(
                ctx.gpu,
                self.dense_gemm_k,
                comp_in,
                &comp.wkv,
                kv_comp,
                t_rows,
                proj_dim,
                h,
                stream,
            )?;
            ops::dense_gemm(
                ctx.gpu,
                self.dense_gemm_k,
                comp_in,
                &comp.wgate,
                gate_comp,
                t_rows,
                proj_dim,
                h,
                stream,
            )?;
            // 2026-09-25: Softmax-gated window compression into `[launch_win, hd_mla]`.
            let compressed = ctx.buffers.moe_output();
            KernelLaunch::new(ctx.gpu, self.csa_compress_k)
                .grid([launch_win, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(kv_comp)
                .arg_ptr(gate_comp)
                .arg_ptr(comp.ape)
                .arg_ptr(compressed)
                .arg_u32(t_rows)
                .arg_u32(ratio)
                .arg_u32(hd_mla)
                .arg_u32(proj_dim)
                .arg_u32(if comp.is_csa { 1 } else { 0 })
                .launch(stream)?;
            // 2026-09-25: RMS-norm the kept block in place.
            let block = compressed.offset(tgt as usize * hd_mla as usize * 2);
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_w_k,
                block,
                &comp.norm,
                block,
                1,
                hd_mla,
                eps,
                stream,
            )?;
            // 2026-09-25: comp_k = RoPE(block) at the window's position `w * ratio`:
            // copy, extract the rope tail, interleaved YaRN with `yarn_inv_freq`, write
            // back.
            let comp_k = compressed.offset(launch_win as usize * hd_mla as usize * 2);
            ctx.gpu
                .copy_d2d_async(block, comp_k, hd_mla as usize * 2, stream)?;
            let pos_bytes = (w * ratio).to_le_bytes();
            let comp_positions = ctx.buffers.ssm_ba();
            ctx.gpu.copy_h2d_async(&pos_bytes, comp_positions, stream)?;
            let comp_rope_tmp = ctx.buffers.ssm_conv_out_f32();
            ops::mla_q_rope_extract_batched(
                ctx.gpu,
                self.mla_q_rope_extract_batched_k,
                comp_k,
                comp_rope_tmp,
                1,
                1,
                hd_mla,
                nope,
                rope_d,
                hd_mla,
                stream,
            )?;
            ops::rope_yarn(
                ctx.gpu,
                self.rope_yarn_interleaved_k,
                comp_rope_tmp,
                comp_rope_tmp,
                comp_positions,
                1,
                0,
                1,
                rope_d,
                rope_d,
                mla.yarn_inv_freq,
                super::super::super::helpers::yarn_rope_mscale(ctx.config),
                stream,
            )?;
            ops::mla_q_rope_writeback_batched(
                ctx.gpu,
                self.mla_q_rope_writeback_batched_k,
                comp_rope_tmp,
                comp_k,
                1,
                1,
                hd_mla,
                nope,
                rope_d,
                hd_mla,
                stream,
            )?;
            // 2026-09-25: Cast the rotated block to FP8 into `pool[w]`, one byte per
            // element.
            ops::bf16_to_fp8(
                ctx.gpu,
                self.bf16_to_fp8_k,
                comp_k,
                comp.pool.offset(w as usize * hd_mla as usize),
                hd_mla,
                stream,
            )?;
            // 2026-09-25: The compressed pool now holds windows `[0, w + 1)`.
            self.v4_comp_pool_filled.store(w + 1, Relaxed);
            // 2026-09-25: CSA: this window becomes the next window's Ca source.
            if comp.is_csa {
                ctx.gpu
                    .copy_d2d_async(comp.ring, comp.prev_win, ratio as usize * hb, stream)?;
                self.v4_comp_prev_valid.store(true, Relaxed);
            }
        }
        Ok(())
    }
}
