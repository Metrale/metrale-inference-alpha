// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The non-paged drafter layer body, which `forward_block` runs when the
//! paged drafter cache is off (`METRALE_DFLASH_OPTION_B=0`). It covers `n_attn` rows:
//! `eff_ctx` ctx rows, whose K/V are projected from the target hiddens in `fc_proj`,
//! then the block rows. BF16 weights only, and no DFlash2 convs.
//!
//! Owner: model-arch (DFlash drafter).
//! Invariants: none beyond the types.

use anyhow::Result;

use super::{BlockDiffusionDraftHead, DflashLayer};
use metrale_model_layers::layer::ForwardContext;

/// 2026-09-25: Values `forward_block` computes once and passes to every layer.
#[allow(clippy::too_many_arguments)]
pub(super) struct LayerArgs {
    pub layer_idx: usize,
    pub n_attn: u32,
    pub eff_ctx: usize,
    pub h: u32,
    pub q_dim: u32,
    pub kv_dim: u32,
    pub inter: u32,
    pub bf16: usize,
    pub inv_sqrt_d: f32,
    pub stream: u64,
}

impl BlockDiffusionDraftHead {
    /// 2026-09-25: Run one drafter layer over `n_attn` rows. Works in the
    /// `self.scratch` buffers and leaves the layer's output in `stream_buf`.
    pub(super) fn forward_block_layer(
        &self,
        layer: &DflashLayer,
        args: &LayerArgs,
        ctx: &ForwardContext,
        debug_dump: bool,
    ) -> Result<()> {
        use metrale_model_layers::layers::ops;

        let LayerArgs {
            layer_idx,
            n_attn,
            eff_ctx,
            h,
            q_dim,
            kv_dim,
            inter,
            bf16,
            inv_sqrt_d,
            stream,
        } = *args;
        let gpu = ctx.gpu;

        let dump_bf16 =
            |label: &str, ptr: metrale_gpu_runtime::gpu::DevicePtr, n: usize| -> Result<()> {
                if !debug_dump {
                    return Ok(());
                }
                let mut buf = vec![0u8; n * 2];
                gpu.synchronize(stream)?;
                gpu.copy_d2h(ptr, &mut buf)?;
                let vals: Vec<f32> = buf
                    .chunks_exact(2)
                    .map(|c| {
                        let bits = u16::from_le_bytes([c[0], c[1]]);
                        f32::from_bits((bits as u32) << 16)
                    })
                    .collect();
                tracing::info!("DFLASH DUMP {label} [{n}]: {:?}", &vals);
                Ok(())
            };

        ops::rms_norm(
            gpu,
            self.kernels.rms_norm,
            self.scratch.stream_buf,
            &layer.input_layernorm,
            self.scratch.norm_buf,
            n_attn,
            h,
            self.rms_norm_eps,
            stream,
        )?;

        ops::dense_gemm_bf16_pipelined(
            gpu,
            self.kernels.dense_gemm_pipelined,
            self.scratch.norm_buf,
            &layer.q_proj,
            self.scratch.q_buf,
            n_attn,
            q_dim,
            h,
            stream,
        )?;
        ops::dense_gemm_bf16_pipelined(
            gpu,
            self.kernels.dense_gemm_pipelined,
            self.scratch.norm_buf,
            &layer.k_proj,
            self.scratch.k_buf,
            n_attn,
            kv_dim,
            h,
            stream,
        )?;
        ops::dense_gemm_bf16_pipelined(
            gpu,
            self.kernels.dense_gemm_pipelined,
            self.scratch.norm_buf,
            &layer.v_proj,
            self.scratch.v_buf,
            n_attn,
            kv_dim,
            h,
            stream,
        )?;

        // 2026-09-25: The ctx rows' K/V are overwritten with projections of `fc_proj`,
        // which skip input_layernorm.
        if eff_ctx > 0 {
            ops::dense_gemm_bf16_pipelined(
                gpu,
                self.kernels.dense_gemm_pipelined,
                self.scratch.fc_proj,
                &layer.k_proj,
                self.scratch.k_buf,
                eff_ctx as u32,
                kv_dim,
                h,
                stream,
            )?;
            ops::dense_gemm_bf16_pipelined(
                gpu,
                self.kernels.dense_gemm_pipelined,
                self.scratch.fc_proj,
                &layer.v_proj,
                self.scratch.v_buf,
                eff_ctx as u32,
                kv_dim,
                h,
                stream,
            )?;
            // 2026-09-25: Zero the ctx rows' Q; the tail reads only the rows from
            // `eff_ctx` on.
            gpu.memset(self.scratch.q_buf, 0, eff_ctx * q_dim as usize * bf16)?;
        }

        if layer_idx == 0 {
            dump_bf16("layer0.k_buf[ctx0].pre_k_norm", self.scratch.k_buf, 10)?;
            dump_bf16("layer0.v_buf[ctx0]", self.scratch.v_buf, 10)?;
            let noise_q_offset = eff_ctx * q_dim as usize * bf16;
            let noise_k_offset = eff_ctx * kv_dim as usize * bf16;
            dump_bf16(
                "layer0.q_buf[noise0].pre_q_norm",
                self.scratch.q_buf.offset(noise_q_offset),
                10,
            )?;
            dump_bf16(
                "layer0.k_buf[noise0].pre_k_norm",
                self.scratch.k_buf.offset(noise_k_offset),
                10,
            )?;
        }

        // 2026-09-25: q_norm and k_norm are per-head RMSNorms over head_dim slices.
        ops::rms_norm(
            gpu,
            self.kernels.rms_norm,
            self.scratch.q_buf,
            &layer.q_norm,
            self.scratch.q_buf,
            n_attn * self.num_q_heads as u32,
            self.head_dim as u32,
            self.rms_norm_eps,
            stream,
        )?;
        ops::rms_norm(
            gpu,
            self.kernels.rms_norm,
            self.scratch.k_buf,
            &layer.k_norm,
            self.scratch.k_buf,
            n_attn * self.num_kv_heads as u32,
            self.head_dim as u32,
            self.rms_norm_eps,
            stream,
        )?;
        if layer_idx == 0 {
            dump_bf16("layer0.k_buf[ctx0].post_k_norm", self.scratch.k_buf, 10)?;
            let noise_q_offset = eff_ctx * q_dim as usize * bf16;
            let noise_k_offset = eff_ctx * kv_dim as usize * bf16;
            dump_bf16(
                "layer0.q_buf[noise0].post_q_norm",
                self.scratch.q_buf.offset(noise_q_offset),
                10,
            )?;
            dump_bf16(
                "layer0.k_buf[noise0].post_k_norm",
                self.scratch.k_buf.offset(noise_k_offset),
                10,
            )?;
        }

        ops::rope_yarn(
            gpu,
            self.kernels.rope_qwen3,
            self.scratch.q_buf,
            self.scratch.k_buf,
            self.scratch.position_ids,
            n_attn,
            self.num_q_heads as u32,
            self.num_kv_heads as u32,
            self.head_dim as u32,
            self.rotary_dim as u32,
            self.yarn_inv_freq,
            self.rope_theta,
            stream,
        )?;
        if layer_idx == 0 {
            let noise_q_offset = eff_ctx * q_dim as usize * bf16;
            let noise_k_offset = eff_ctx * kv_dim as usize * bf16;
            dump_bf16(
                "layer0.q_buf[noise0].post_rope",
                self.scratch.q_buf.offset(noise_q_offset),
                10,
            )?;
            dump_bf16(
                "layer0.k_buf[noise0].post_rope",
                self.scratch.k_buf.offset(noise_k_offset),
                10,
            )?;
            dump_bf16("layer0.k_buf[ctx0].post_rope", self.scratch.k_buf, 10)?;
        }

        // 2026-09-25: Non-causal attention, q_len = kv_len = n_attn.
        ops::prefill_attention(
            gpu,
            self.kernels.prefill_attn,
            self.scratch.q_buf,
            self.scratch.k_buf,
            self.scratch.v_buf,
            self.scratch.attn_out,
            n_attn,
            1,
            self.num_q_heads as u32,
            self.num_kv_heads as u32,
            self.head_dim as u32,
            inv_sqrt_d,
            false,
            0,
            stream,
        )?;
        if layer_idx == 0 {
            let noise_q_offset = eff_ctx * q_dim as usize * bf16;
            dump_bf16(
                "layer0.attn_out[noise0]",
                self.scratch.attn_out.offset(noise_q_offset),
                10,
            )?;
            dump_bf16(
                "layer0.attn_out[noise0][1000..1010]",
                self.scratch.attn_out.offset(noise_q_offset + 1000 * bf16),
                10,
            )?;
            dump_bf16(
                "layer0.attn_out[noise0][4086..4096]",
                self.scratch.attn_out.offset(noise_q_offset + 4086 * bf16),
                10,
            )?;
            // 2026-09-25: METRALE_DFLASH_DEBUG_DUMP_FULL=1: also write the first block
            // row of attn_out (q_dim BF16 values) to /tmp/metrale_attn_out.bin.
            if self.levers.debug_dump_full {
                let n_bytes = q_dim as usize * bf16;
                let mut buf = vec![0u8; n_bytes];
                gpu.synchronize(stream)?;
                gpu.copy_d2h(self.scratch.attn_out.offset(noise_q_offset), &mut buf)?;
                std::fs::write("/tmp/metrale_attn_out.bin", &buf)
                    .map_err(|e| anyhow::anyhow!("write attn_out dump: {e}"))?;
                tracing::info!(
                    "DFLASH DUMP wrote {} bytes attn_out[noise0] to /tmp/metrale_attn_out.bin",
                    n_bytes
                );
            }
        }

        ops::dense_gemm_bf16_pipelined(
            gpu,
            self.kernels.dense_gemm_pipelined,
            self.scratch.attn_out,
            &layer.o_proj,
            self.scratch.stream_acc,
            n_attn,
            h,
            q_dim,
            stream,
        )?;
        if layer_idx == 0 {
            let noise_offset = eff_ctx * self.hidden_size * bf16;
            dump_bf16(
                "layer0.stream_acc[noise0].post_o_proj",
                self.scratch.stream_acc.offset(noise_offset),
                10,
            )?;
            dump_bf16(
                "layer0.stream_buf[noise0].pre_residual",
                self.scratch.stream_buf.offset(noise_offset),
                10,
            )?;
            if self.levers.debug_dump_full {
                let n_bytes = self.hidden_size * bf16;
                let mut buf = vec![0u8; n_bytes];
                gpu.synchronize(stream)?;
                gpu.copy_d2h(self.scratch.stream_acc.offset(noise_offset), &mut buf)?;
                std::fs::write("/tmp/metrale_o_proj_out.bin", &buf)
                    .map_err(|e| anyhow::anyhow!("write o_proj_out: {e}"))?;
            }
        }

        // 2026-09-25: Residual: stream_buf += stream_acc.
        ops::residual_add(
            gpu,
            self.kernels.residual_add,
            self.scratch.stream_buf,
            self.scratch.stream_acc,
            n_attn * h,
            stream,
        )?;
        if layer_idx == 0 {
            let noise_offset = eff_ctx * self.hidden_size * bf16;
            dump_bf16(
                "layer0.stream_buf[noise0].post_attn_residual",
                self.scratch.stream_buf.offset(noise_offset),
                10,
            )?;
        }

        ops::rms_norm(
            gpu,
            self.kernels.rms_norm,
            self.scratch.stream_buf,
            &layer.post_attention_layernorm,
            self.scratch.norm_buf,
            n_attn,
            h,
            self.rms_norm_eps,
            stream,
        )?;

        ops::dense_gemm_bf16_pipelined(
            gpu,
            self.kernels.dense_gemm_pipelined,
            self.scratch.norm_buf,
            &layer.gate_proj,
            self.scratch.mlp_intermediate,
            n_attn,
            inter,
            h,
            stream,
        )?;
        ops::dense_gemm_bf16_pipelined(
            gpu,
            self.kernels.dense_gemm_pipelined,
            self.scratch.norm_buf,
            &layer.up_proj,
            self.scratch.mlp_up,
            n_attn,
            inter,
            h,
            stream,
        )?;

        ops::silu_mul(
            gpu,
            self.kernels.silu_mul,
            self.scratch.mlp_intermediate,
            self.scratch.mlp_up,
            self.scratch.mlp_intermediate,
            n_attn * inter,
            stream,
        )?;

        ops::dense_gemm_bf16_pipelined(
            gpu,
            self.kernels.dense_gemm_pipelined,
            self.scratch.mlp_intermediate,
            &layer.down_proj,
            self.scratch.stream_acc,
            n_attn,
            h,
            inter,
            stream,
        )?;

        ops::residual_add(
            gpu,
            self.kernels.residual_add,
            self.scratch.stream_buf,
            self.scratch.stream_acc,
            n_attn * h,
            stream,
        )?;
        if layer_idx == 0 {
            let noise_offset = eff_ctx * self.hidden_size * bf16;
            dump_bf16(
                "layer0.stream_buf[noise0].post_layer",
                self.scratch.stream_buf.offset(noise_offset),
                10,
            )?;
        }

        Ok(())
    }
}
