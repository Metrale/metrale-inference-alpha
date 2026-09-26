// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Phase 1 of the two-phase and batched GDN prefill: input norm,
//! QKVZ projection, BA gates, conv1d and the Q/K L2 norm, staged into
//! `GdnPrefillBuffers` for the whole-range recurrence.
//!
//! Owner: model-layers, GDN/SSM layer (`qwen3_ssm`).
//! Invariants: none beyond the types.

use super::*;

impl Qwen3SsmLayer {
    pub(super) fn is_ssm_layer_inner(&self) -> bool {
        true
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn prefill_phase1_inner(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        state: &mut dyn LayerState,
        _kv_cache: &mut PagedKvCache,
        _seq_len_start: usize,
        _block_table: &mut Vec<u32>,
        _disk_block_ids: &mut Vec<u32>,
        _disk_last_offloaded_per_layer: &mut Vec<u32>,
        _kv_write_start: usize,
        gdn_bufs: &GdnPrefillBuffers,
        token_offset: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size;
        let eps = ctx.config.rms_norm_eps as f32;
        let k = num_tokens as u32;
        let bf16 = 2usize;
        let fp32 = 4usize;

        let ssm_state = state
            .as_any_mut()
            .downcast_mut::<SsmLayerState>()
            .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState"))?;

        let nk = ctx.config.linear_num_key_heads;
        let kd = ctx.config.linear_key_head_dim;
        let nv = ctx.config.linear_num_value_heads;
        let vd = ctx.config.linear_value_head_dim;
        let vpg = nv / nk;
        let key_dim = nk * kd;
        let value_dim = nv * vd;
        let conv_dim = key_dim * 2 + value_dim;
        let d_conv = ctx.config.linear_conv_kernel_dim;
        let qkvz_size = ctx.config.ssm_qkvz_size();

        // 2026-09-25: For k > 4096, synchronise at entry and after the norm, the
        // gates and the conv, so a fault is reported at its stage; smaller
        // chunks skip the syncs.
        if k > 4096 {
            tracing::info!("ssm phase1 ENTRY: k={k} h={h} qkvz={qkvz_size}");
            ctx.gpu.synchronize(stream).map_err(|e| {
                anyhow::anyhow!("ssm phase1 ENTRY: stream broken BEFORE we start (M={k}): {e}")
            })?;
        }

        let normed = ctx.buffers.norm_output();
        ops::rms_norm_residual(
            ctx.gpu,
            self.rms_norm_residual_k,
            hidden,
            &self.input_norm,
            normed,
            residual,
            k,
            h as u32,
            eps,
            stream,
        )?;
        if k > 4096 {
            ctx.gpu.synchronize(stream).map_err(|e| {
                anyhow::anyhow!(
                    "ssm phase1 L{}: SYNC after rms_norm (M={k}): {e}",
                    0 /* 2026-09-25: a constant, not the layer index */
                )
            })?;
        }

        // 2026-09-25: The same `prefill_qkvz_proj` dispatch the single-stream
        // prefill uses.
        let deinterleaved = ctx.buffers.ssm_deinterleaved();
        self.prefill_qkvz_proj(
            normed,
            deinterleaved,
            k,
            qkvz_size,
            h,
            nk,
            kd,
            vpg,
            vd,
            ctx,
            stream,
        )?;
        let ba_size = ctx.config.ssm_ba_size();
        let gates_buf = ctx.buffers.ssm_gates();
        let gate_stride = nv * 2;
        ops::dense_gemm_ba_gates_prefill(
            ctx.gpu,
            self.ba_gates_prefill_k,
            self.ba_gates_prefill_hopper_k,
            normed,
            &self.ssm.in_proj_ba,
            self.ssm.a_log.weight,
            self.ssm.dt_bias.weight,
            gates_buf,
            k,
            ba_size as u32,
            h as u32,
            h as u32,
            gate_stride as u32,
            nv as u32,
            vpg as u32,
            stream,
        )?;

        if k > 4096 {
            ctx.gpu
                .synchronize(stream)
                .map_err(|e| anyhow::anyhow!("ssm phase1: SYNC after BA+gates (M={k}): {e}"))?;
        }
        let conv_out_buf = ctx.buffers.ssm_qkvz();
        ops::conv1d_update_prefill(
            ctx.gpu,
            self.conv1d_prefill_k,
            self.conv1d_prefill_tp_k,
            ssm_state.conv_state,
            deinterleaved,
            &self.ssm.conv1d,
            DevicePtr::NULL,
            conv_out_buf,
            conv_dim as u32,
            d_conv as u32,
            k,
            qkvz_size as u32,
            conv_dim as u32,
            stream,
        )?;
        if k > 4096 {
            ctx.gpu
                .synchronize(stream)
                .map_err(|e| anyhow::anyhow!("ssm phase1: SYNC after conv1d (M={k}): {e}"))?;
        }

        ops::l2_norm(
            ctx.gpu,
            self.l2_norm_k,
            conv_out_buf,
            (nk * 2) as u32,
            kd as u32,
            1e-6,
            k,
            conv_dim as u32,
            stream,
        )?;

        // 2026-09-25: Stage the recurrence inputs at `token_offset` in
        // `gdn_bufs`. QKV rows are `conv_dim` wide in both buffers, so one
        // contiguous copy.
        let qkv_dst = gdn_bufs.qkv.offset(token_offset * conv_dim * bf16);
        ctx.gpu
            .copy_d2d_async(conv_out_buf, qkv_dst, num_tokens * conv_dim * bf16, stream)?;

        // 2026-09-25: Gates: `2 * nv` FP32 per token in both buffers, one
        // contiguous copy.
        let gb_dst = gdn_bufs.gate_beta.offset(token_offset * gate_stride * fp32);
        ctx.gpu
            .copy_d2d_async(gates_buf, gb_dst, num_tokens * gate_stride * fp32, stream)?;

        // 2026-09-25: Z: source rows `qkvz_size` apart with Z at
        // `2 * key_dim + value_dim`, destination rows `value_dim` apart. One
        // pitched 2D copy.
        let z_src_base = deinterleaved.offset((key_dim * 2 + value_dim) * bf16);
        let z_dst_base = gdn_bufs.z.offset(token_offset * value_dim * bf16);
        let z_elem_bytes = value_dim * bf16;
        ctx.gpu.copy_d2d_2d_async(
            z_src_base,
            qkvz_size * bf16,
            z_dst_base,
            value_dim * bf16,
            z_elem_bytes,
            num_tokens,
            stream,
        )?;

        Ok(())
    }

    // 2026-09-25: Batched phase 1 for the co-dispatch path. The norm, QKVZ and
    // BA-gate GEMMs hold no recurrent state, so they run once over all stacked
    // tokens. Only conv1d advances per-request state: the caller runs
    // `prefill_phase1_conv1d_one` per request between
    // `prefill_phase1_proj_batched` and `prefill_phase1_l2_batched`.

    /// 2026-09-25: Steps 1-5 and the gate/Z staging over the whole stacked
    /// batch, one launch per stage. Writes `gdn_bufs.gate_beta` and
    /// `gdn_bufs.z`, and leaves the QKVZ result in
    /// `ctx.buffers.ssm_deinterleaved()` for the per-request conv1d.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prefill_phase1_proj_batched_inner(
        &self,
        hidden_stacked: DevicePtr,
        residual_stacked: DevicePtr,
        total_tokens: usize,
        gdn_bufs: &GdnPrefillBuffers,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size;
        let eps = ctx.config.rms_norm_eps as f32;
        let k = total_tokens as u32;
        let bf16 = 2usize;
        let fp32 = 4usize;
        let nk = ctx.config.linear_num_key_heads;
        let kd = ctx.config.linear_key_head_dim;
        let nv = ctx.config.linear_num_value_heads;
        let vd = ctx.config.linear_value_head_dim;
        let vpg = nv / nk;
        let key_dim = nk * kd;
        let value_dim = nv * vd;
        let qkvz_size = ctx.config.ssm_qkvz_size();

        let normed = ctx.buffers.norm_output();
        ops::rms_norm_residual(
            ctx.gpu,
            self.rms_norm_residual_k,
            hidden_stacked,
            &self.input_norm,
            normed,
            residual_stacked,
            k,
            h as u32,
            eps,
            stream,
        )?;

        let deinterleaved = ctx.buffers.ssm_deinterleaved();
        self.prefill_qkvz_proj(
            normed,
            deinterleaved,
            k,
            qkvz_size,
            h,
            nk,
            kd,
            vpg,
            vd,
            ctx,
            stream,
        )?;

        // 2026-09-25: BA GEMM and gates over all tokens into `ssm_gates`, then
        // one contiguous copy to `gdn_bufs.gate_beta`.
        let ba_size = ctx.config.ssm_ba_size();
        let gates_buf = ctx.buffers.ssm_gates();
        let gate_stride = nv * 2;
        ops::dense_gemm_ba_gates_prefill(
            ctx.gpu,
            self.ba_gates_prefill_k,
            self.ba_gates_prefill_hopper_k,
            normed,
            &self.ssm.in_proj_ba,
            self.ssm.a_log.weight,
            self.ssm.dt_bias.weight,
            gates_buf,
            k,
            ba_size as u32,
            h as u32,
            h as u32,
            gate_stride as u32,
            nv as u32,
            vpg as u32,
            stream,
        )?;
        ctx.gpu.copy_d2d_async(
            gates_buf,
            gdn_bufs.gate_beta,
            total_tokens * gate_stride * fp32,
            stream,
        )?;

        // 2026-09-25: Z: one pitched copy from `deinterleaved` (rows
        // `qkvz_size` apart) to `gdn_bufs.z` (rows `value_dim` apart).
        let z_src_base = deinterleaved.offset((key_dim * 2 + value_dim) * bf16);
        let z_elem_bytes = value_dim * bf16;
        ctx.gpu.copy_d2d_2d_async(
            z_src_base,
            qkvz_size * bf16,
            gdn_bufs.z,
            value_dim * bf16,
            z_elem_bytes,
            total_tokens,
            stream,
        )?;
        Ok(())
    }

    /// 2026-09-25: Per-request conv1d: reads the request's rows of the stacked
    /// `ssm_deinterleaved` scratch (filled by `prefill_phase1_proj_batched`),
    /// advances this request's `conv_state`, and writes the conv output into
    /// `gdn_bufs.qkv` at the request's token offset.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prefill_phase1_conv1d_one_inner(
        &self,
        state: &mut dyn LayerState,
        token_offset: usize,
        len: usize,
        gdn_bufs: &GdnPrefillBuffers,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let bf16 = 2usize;
        let nk = ctx.config.linear_num_key_heads;
        let kd = ctx.config.linear_key_head_dim;
        let nv = ctx.config.linear_num_value_heads;
        let vd = ctx.config.linear_value_head_dim;
        let key_dim = nk * kd;
        let value_dim = nv * vd;
        let conv_dim = key_dim * 2 + value_dim;
        let d_conv = ctx.config.linear_conv_kernel_dim;
        let qkvz_size = ctx.config.ssm_qkvz_size();
        let ssm_state = state
            .as_any_mut()
            .downcast_mut::<SsmLayerState>()
            .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState"))?;
        let deinterleaved = ctx.buffers.ssm_deinterleaved();
        let src = deinterleaved.offset(token_offset * qkvz_size * bf16);
        let dst = gdn_bufs.qkv.offset(token_offset * conv_dim * bf16);
        ops::conv1d_update_prefill(
            ctx.gpu,
            self.conv1d_prefill_k,
            self.conv1d_prefill_tp_k,
            ssm_state.conv_state,
            src,
            &self.ssm.conv1d,
            DevicePtr::NULL,
            dst,
            conv_dim as u32,
            d_conv as u32,
            len as u32,
            qkvz_size as u32,
            conv_dim as u32,
            stream,
        )
    }

    /// 2026-09-25: L2 norm on Q and K over the whole stacked `gdn_bufs.qkv`,
    /// after every per-request conv1d has written its rows.
    pub(super) fn prefill_phase1_l2_batched_inner(
        &self,
        total_tokens: usize,
        gdn_bufs: &GdnPrefillBuffers,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let nk = ctx.config.linear_num_key_heads;
        let kd = ctx.config.linear_key_head_dim;
        let nv = ctx.config.linear_num_value_heads;
        let vd = ctx.config.linear_value_head_dim;
        let conv_dim = nk * kd * 2 + nv * vd;
        ops::l2_norm(
            ctx.gpu,
            self.l2_norm_k,
            gdn_bufs.qkv,
            (nk * 2) as u32,
            kd as u32,
            1e-6,
            total_tokens as u32,
            conv_dim as u32,
            stream,
        )
    }
}
