// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The per-row fallback route of `ms_phase_qkv`: the Q projection and the K/V
//! projections of one row, for any weight encoding.
//!
//! Owner: model-layers (qwen3 attention).
//! Invariants:
//! - Each call writes one row's Q at `q_out_i`, or its K and V at `k_out_i` and `v_out_i`.

use anyhow::Result;

use crate::layers::ops;
use crate::layers::qwen3_attention::Qwen3AttentionLayer;

impl Qwen3AttentionLayer {
    /// 2026-09-25: Q projection for one row, any weight encoding, gated or not.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn ms_qkv_seq_q(
        &self,
        fwd: &crate::layer::ForwardContext<'_>,
        normed_i: metrale_gpu_runtime::gpu::DevicePtr,
        q_out_i: metrale_gpu_runtime::gpu::DevicePtr,
        q_proj_dim: u32,
        q_dim: u32,
        nq: u32,
        hd: u32,
        h: usize,
        stream: u64,
    ) -> Result<()> {
        if self.gated {
            if let Some(q2) = self.q_weight.as_ref().and_then(|w| w.as_packed_q2()) {
                ops::q2_0_gemv_vec(fwd.gpu, self.q2_0_gemv_k, normed_i, q2, q_out_i, stream)?;
                ops::deinterleave_qg(
                    fwd.gpu,
                    self.deinterleave_qg_k,
                    q_out_i,
                    1,
                    nq,
                    hd,
                    q_proj_dim,
                    stream,
                )?;
            } else if let Some(fp8) = self.q_weight.as_ref().and_then(|w| w.as_fp8()) {
                ops::w8a16_gemv(
                    fwd.gpu,
                    self.w8a16_gemv_k,
                    normed_i,
                    fp8.weight,
                    fp8.row_scale,
                    q_out_i,
                    q_proj_dim,
                    h as u32,
                    stream,
                )?;
                // 2026-09-25: With a q adapter the split waits for
                // `ms_qkv_deinterleave_q`.
                if !self.q_lora_active() {
                    ops::deinterleave_qg(
                        fwd.gpu,
                        self.deinterleave_qg_k,
                        q_out_i,
                        1,
                        nq,
                        hd,
                        q_proj_dim,
                        stream,
                    )?;
                }
            } else if let Some(nvfp4) = self.q_weight.as_ref().and_then(|w| w.as_nvfp4()) {
                if self.q_lora_active() {
                    // 2026-09-25: With a q adapter: the plain GEMV instead of the
                    // fused GEMV-and-split; `ms_qkv_deinterleave_q` splits later.
                    self.nvfp4_decode_gemv(
                        fwd.gpu,
                        fwd.levers.gemv_sw,
                        normed_i,
                        nvfp4,
                        q_out_i,
                        q_proj_dim,
                        h as u32,
                        stream,
                    )?;
                } else {
                    ops::w4a16_gemv_qg(
                        fwd.gpu,
                        self.w4a16_gemv_qg_k,
                        normed_i,
                        nvfp4,
                        q_out_i,
                        q_proj_dim,
                        h as u32,
                        nq,
                        hd,
                        stream,
                    )?;
                }
            } else {
                ops::dense_gemv(
                    fwd.gpu,
                    self.dense_gemv_k,
                    normed_i,
                    &self.attn.q_proj,
                    q_out_i,
                    q_proj_dim,
                    h as u32,
                    stream,
                )?;
                if !self.q_lora_active() {
                    ops::deinterleave_qg(
                        fwd.gpu,
                        self.deinterleave_qg_k,
                        q_out_i,
                        1,
                        nq,
                        hd,
                        q_proj_dim,
                        stream,
                    )?;
                }
            }
        } else if let Some(q2) = self.q_weight.as_ref().and_then(|w| w.as_packed_q2()) {
            ops::q2_0_gemv_vec(fwd.gpu, self.q2_0_gemv_k, normed_i, q2, q_out_i, stream)?;
        } else if let Some(fp8) = self.q_weight.as_ref().and_then(|w| w.as_fp8()) {
            ops::w8a16_gemv(
                fwd.gpu,
                self.w8a16_gemv_k,
                normed_i,
                fp8.weight,
                fp8.row_scale,
                q_out_i,
                q_dim,
                h as u32,
                stream,
            )?;
        } else if let Some(nvfp4) = self.q_weight.as_ref().and_then(|w| w.as_nvfp4()) {
            self.nvfp4_decode_gemv(
                fwd.gpu,
                fwd.levers.gemv_sw,
                normed_i,
                nvfp4,
                q_out_i,
                q_dim,
                h as u32,
                stream,
            )?;
        } else {
            ops::dense_gemv(
                fwd.gpu,
                self.dense_gemv_k,
                normed_i,
                &self.attn.q_proj,
                q_out_i,
                q_dim,
                h as u32,
                stream,
            )?;
        }
        Ok(())
    }

    /// 2026-09-25: K and V projections for one row, any weight encoding.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn ms_qkv_seq_kv(
        &self,
        fwd: &crate::layer::ForwardContext<'_>,
        normed_i: metrale_gpu_runtime::gpu::DevicePtr,
        k_out_i: metrale_gpu_runtime::gpu::DevicePtr,
        v_out_i: metrale_gpu_runtime::gpu::DevicePtr,
        nkv: u32,
        hd: u32,
        h: usize,
        stream: u64,
    ) -> Result<()> {
        if let (Some(k_q2), Some(v_q2)) = (
            self.k_weight.as_ref().and_then(|w| w.as_packed_q2()),
            self.v_weight.as_ref().and_then(|w| w.as_packed_q2()),
        ) {
            ops::q2_0_gemv_vec(fwd.gpu, self.q2_0_gemv_k, normed_i, k_q2, k_out_i, stream)?;
            ops::q2_0_gemv_vec(fwd.gpu, self.q2_0_gemv_k, normed_i, v_q2, v_out_i, stream)?;
        } else if let (Some(k_fp8), Some(v_fp8)) = (
            self.k_weight.as_ref().and_then(|w| w.as_fp8()),
            self.v_weight.as_ref().and_then(|w| w.as_fp8()),
        ) {
            ops::w8a16_gemv(
                fwd.gpu,
                self.w8a16_gemv_k,
                normed_i,
                k_fp8.weight,
                k_fp8.row_scale,
                k_out_i,
                nkv * hd,
                h as u32,
                stream,
            )?;
            ops::w8a16_gemv(
                fwd.gpu,
                self.w8a16_gemv_k,
                normed_i,
                v_fp8.weight,
                v_fp8.row_scale,
                v_out_i,
                nkv * hd,
                h as u32,
                stream,
            )?;
        } else if let (Some(k_fp4), Some(v_fp4)) = (
            self.k_weight.as_ref().and_then(|w| w.as_nvfp4()),
            self.v_weight.as_ref().and_then(|w| w.as_nvfp4()),
        ) {
            ops::w4a16_gemv_dual(
                fwd.gpu,
                self.w4a16_gemv_dual_k,
                normed_i,
                k_fp4,
                k_out_i,
                v_fp4,
                v_out_i,
                nkv * hd,
                h as u32,
                stream,
            )?;
        } else {
            if let Some(nvfp4) = self.k_weight.as_ref().and_then(|w| w.as_nvfp4()) {
                self.nvfp4_decode_gemv(
                    fwd.gpu,
                    fwd.levers.gemv_sw,
                    normed_i,
                    nvfp4,
                    k_out_i,
                    nkv * hd,
                    h as u32,
                    stream,
                )?;
            } else {
                ops::dense_gemv(
                    fwd.gpu,
                    self.dense_gemv_k,
                    normed_i,
                    &self.attn.k_proj,
                    k_out_i,
                    nkv * hd,
                    h as u32,
                    stream,
                )?;
            }
            if let Some(nvfp4) = self.v_weight.as_ref().and_then(|w| w.as_nvfp4()) {
                self.nvfp4_decode_gemv(
                    fwd.gpu,
                    fwd.levers.gemv_sw,
                    normed_i,
                    nvfp4,
                    v_out_i,
                    nkv * hd,
                    h as u32,
                    stream,
                )?;
            } else {
                ops::dense_gemv(
                    fwd.gpu,
                    self.dense_gemv_k,
                    normed_i,
                    &self.attn.v_proj,
                    v_out_i,
                    nkv * hd,
                    h as u32,
                    stream,
                )?;
            }
        }
        Ok(())
    }
}
