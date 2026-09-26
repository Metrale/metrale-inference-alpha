// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The NVFP4 q/k/v routes of `ms_phase_qkv` that project every row with one
//! launch per weight: n = 3 (`ms_qkv_batch3`), n = 2 (`ms_qkv_batch2`) and n > 3
//! (`ms_qkv_batchn`).
//!
//! Owner: model-layers (qwen3 attention).
//! Invariants:
//! - Each is called only when q, k and v all hold an NVFP4 weight, which the
//!   `unwrap()` of each `as_nvfp4()` relies on.

use anyhow::Result;

use super::super::ctx::MultiSeqCtx;
use super::fused_qkv_enabled;
use crate::layers::ops;
use crate::layers::qwen3_attention::Qwen3AttentionLayer;

impl Qwen3AttentionLayer {
    /// 2026-09-25: n = 3 with NVFP4 q/k/v: batch-3 GEMVs into scratch, then a
    /// copy of each row into `qkv_buf`.
    pub(super) fn ms_qkv_batch3(&self, c: &MultiSeqCtx<'_>) -> Result<()> {
        let MultiSeqCtx {
            fwd,
            stream,
            h,
            nq,
            nkv,
            hd,
            eps,
            bf16,
            q_proj_dim,
            q_proj_bytes,
            per_seq_qkv,
            normed,
            qkv_buf,
            ..
        } = *c;
        let q_nvfp4 = self.q_weight.as_ref().and_then(|w| w.as_nvfp4()).unwrap();
        let k_nvfp4 = self.k_weight.as_ref().and_then(|w| w.as_nvfp4()).unwrap();
        let v_nvfp4 = self.v_weight.as_ref().and_then(|w| w.as_nvfp4()).unwrap();

        let q_scratch = fwd.buffers.ssm_qkvz();
        if self.gated && !self.q_lora_active() {
            ops::w4a16_gemv_qg_batch3(
                fwd.gpu,
                self.w4a16_gemv_qg_batch3_k,
                normed,
                q_nvfp4,
                q_scratch,
                q_proj_dim,
                h as u32,
                nq,
                hd,
                stream,
            )?;
        } else {
            // 2026-09-25: Ungated, or gated with a q adapter: the plain GEMV,
            // leaving a gated Q interleaved for `ms_qkv_deinterleave_q`.
            ops::w4a16_gemv_batch3(
                fwd.gpu,
                self.w4a16_gemv_batch3_k,
                normed,
                q_nvfp4,
                q_scratch,
                q_proj_dim,
                h as u32,
                stream,
            )?;
        }

        let kv_dim = nkv * hd;
        let kv_bytes = kv_dim as usize * bf16;
        let k_scratch = fwd.buffers.attn_output();
        let v_scratch = k_scratch.offset(3 * kv_bytes);
        ops::w4a16_gemv_dual_batch3(
            fwd.gpu,
            self.w4a16_gemv_dual_batch3_k,
            normed,
            k_nvfp4,
            k_scratch,
            v_nvfp4,
            v_scratch,
            kv_dim,
            h as u32,
            stream,
        )?;

        for i in 0..3usize {
            let q_out_i = qkv_buf.offset(i * per_seq_qkv);
            let k_out_i = q_out_i.offset(q_proj_bytes);
            let v_out_i = k_out_i.offset(kv_bytes);
            fwd.gpu.copy_d2d_async(
                q_scratch.offset(i * q_proj_bytes),
                q_out_i,
                q_proj_bytes,
                stream,
            )?;
            fwd.gpu
                .copy_d2d_async(k_scratch.offset(i * kv_bytes), k_out_i, kv_bytes, stream)?;
            fwd.gpu
                .copy_d2d_async(v_scratch.offset(i * kv_bytes), v_out_i, kv_bytes, stream)?;
        }

        // 2026-09-25: The q/k norms run later, in `ms_qkv_norms`.
        let _ = (nq, eps);
        Ok(())
    }

    /// 2026-09-25: n = 2 with NVFP4 q/k/v; as `ms_qkv_batch3` with batch-2 GEMVs.
    pub(super) fn ms_qkv_batch2(&self, c: &MultiSeqCtx<'_>) -> Result<()> {
        let MultiSeqCtx {
            fwd,
            stream,
            h,
            nq,
            nkv,
            hd,
            eps,
            bf16,
            q_proj_dim,
            q_proj_bytes,
            per_seq_qkv,
            normed,
            qkv_buf,
            ..
        } = *c;
        let q_nvfp4 = self.q_weight.as_ref().and_then(|w| w.as_nvfp4()).unwrap();
        let k_nvfp4 = self.k_weight.as_ref().and_then(|w| w.as_nvfp4()).unwrap();
        let v_nvfp4 = self.v_weight.as_ref().and_then(|w| w.as_nvfp4()).unwrap();

        let q_scratch = fwd.buffers.ssm_qkvz();
        if self.gated && !self.q_lora_active() {
            ops::w4a16_gemv_qg_batch2(
                fwd.gpu,
                self.w4a16_gemv_qg_batch2_k,
                normed,
                q_nvfp4,
                q_scratch,
                q_proj_dim,
                h as u32,
                nq,
                hd,
                stream,
            )?;
        } else {
            // 2026-09-25: Ungated, or gated with a q adapter: the plain GEMV,
            // leaving a gated Q interleaved for `ms_qkv_deinterleave_q`.
            ops::w4a16_gemv_batch2(
                fwd.gpu,
                self.w4a16_gemv_batch2_k,
                normed,
                q_nvfp4,
                q_scratch,
                q_proj_dim,
                h as u32,
                stream,
            )?;
        }

        let kv_dim = nkv * hd;
        let kv_bytes = kv_dim as usize * bf16;
        let k_scratch = fwd.buffers.attn_output();
        let v_scratch = k_scratch.offset(2 * kv_bytes);
        ops::w4a16_gemv_dual_batch2(
            fwd.gpu,
            self.w4a16_gemv_dual_batch2_k,
            normed,
            k_nvfp4,
            k_scratch,
            v_nvfp4,
            v_scratch,
            kv_dim,
            h as u32,
            stream,
        )?;

        for i in 0..2usize {
            let q_out_i = qkv_buf.offset(i * per_seq_qkv);
            let k_out_i = q_out_i.offset(q_proj_bytes);
            let v_out_i = k_out_i.offset(kv_bytes);
            fwd.gpu.copy_d2d_async(
                q_scratch.offset(i * q_proj_bytes),
                q_out_i,
                q_proj_bytes,
                stream,
            )?;
            fwd.gpu
                .copy_d2d_async(k_scratch.offset(i * kv_bytes), k_out_i, kv_bytes, stream)?;
            fwd.gpu
                .copy_d2d_async(v_scratch.offset(i * kv_bytes), v_out_i, kv_bytes, stream)?;
        }

        // 2026-09-25: The q/k norms run later, in `ms_qkv_norms`.
        let _ = (nq, eps);
        Ok(())
    }

    /// 2026-09-25: n > 3 with NVFP4 q/k/v: each projection reads its weight
    /// once for all n rows (`wide_verify_gemm`) into scratch, and each row is
    /// copied into `qkv_buf`. The fused `[q | k | v]` route writes `qkv_buf`
    /// directly and copies nothing.
    pub(super) fn ms_qkv_batchn(&self, c: &MultiSeqCtx<'_>) -> Result<()> {
        let MultiSeqCtx {
            fwd,
            n,
            stream,
            h,
            nq,
            nkv,
            hd,
            eps,
            bf16,
            q_proj_dim,
            q_proj_bytes,
            per_seq_qkv,
            normed,
            qkv_buf,
            ..
        } = *c;
        let q_nvfp4 = self.q_weight.as_ref().and_then(|w| w.as_nvfp4()).unwrap();
        let k_nvfp4 = self.k_weight.as_ref().and_then(|w| w.as_nvfp4()).unwrap();
        let v_nvfp4 = self.v_weight.as_ref().and_then(|w| w.as_nvfp4()).unwrap();

        // 2026-09-25: Unfused, Q goes to `q_scratch` as contiguous
        // `[n, q_proj_dim]` rows (interleaved `[Q | gate]` when gated).
        let q_scratch = fwd.buffers.ssm_qkvz();
        let kv_dim_e = (nkv * hd) as usize;
        // 2026-09-25: Fused `[q | k | v]`: one GEMM over the concatenated
        // transposed weight `qkv_nvfp4_t` instead of three. Its only loader
        // (`qwen35_dense.rs`) builds it only when q, k and v have bit-equal
        // `weight_scale_2`, the one scale the GEMM applies.
        let fused_n = q_proj_dim as usize + 2 * kv_dim_e;
        // 2026-09-25: `n > 8` keeps the fused N off the batched-GEMV arm of
        // `wide_verify_gemm` for m <= 8: that arm reads the base q weight, not
        // `w_t`, and a fused N would read past it.
        let use_fused = fused_qkv_enabled() && self.qkv_nvfp4_t.is_some() && n > 8;
        if use_fused {
            // 2026-09-25: `per_seq_qkv == q_proj_bytes + 2 * kv_bytes == fused_n * 2`,
            // so the fused `[n, fused_n]` output is the `qkv_buf` layout, and it
            // is written there directly.
            self.wide_verify_gemm(
                c,
                normed,
                q_nvfp4,
                self.qkv_nvfp4_t.as_ref(),
                qkv_buf,
                n as u32,
                fused_n as u32,
                h as u32,
                false,
            )?;
        } else {
            self.wide_verify_gemm(
                c,
                normed,
                q_nvfp4,
                self.q_nvfp4_t.as_ref(),
                q_scratch,
                n as u32,
                q_proj_dim,
                h as u32,
                false,
            )?;
        }
        if self.gated && !self.q_lora_active() {
            // 2026-09-25: Split `[Q | gate]` in place for all n rows. With a q
            // adapter the split waits for `ms_qkv_deinterleave_q`, after the fold.
            ops::deinterleave_qg(
                fwd.gpu,
                self.deinterleave_qg_k,
                if use_fused { qkv_buf } else { q_scratch },
                n as u32,
                nq,
                hd,
                if use_fused {
                    fused_n as u32
                } else {
                    q_proj_dim
                },
                stream,
            )?;
        }

        // 2026-09-25: Unfused, K and V take one projection each.
        let kv_dim = nkv * hd;
        let kv_bytes = kv_dim as usize * bf16;
        let k_scratch = fwd.buffers.attn_output();
        let v_scratch = k_scratch.offset(n * kv_bytes);
        if !use_fused {
            self.wide_verify_gemm(
                c,
                normed,
                k_nvfp4,
                self.k_nvfp4_t.as_ref(),
                k_scratch,
                n as u32,
                kv_dim,
                h as u32,
                true,
            )?;
            self.wide_verify_gemm(
                c,
                normed,
                v_nvfp4,
                self.v_nvfp4_t.as_ref(),
                v_scratch,
                n as u32,
                kv_dim,
                h as u32,
                true,
            )?;
        }

        // 2026-09-25: Unfused, copy each row's Q, K and V from scratch into `qkv_buf`.
        for i in (0..n).take_while(|_| !use_fused) {
            let q_out_i = qkv_buf.offset(i * per_seq_qkv);
            let k_out_i = q_out_i.offset(q_proj_bytes);
            let v_out_i = k_out_i.offset(kv_bytes);
            fwd.gpu.copy_d2d_async(
                q_scratch.offset(i * q_proj_bytes),
                q_out_i,
                q_proj_bytes,
                stream,
            )?;
            fwd.gpu
                .copy_d2d_async(k_scratch.offset(i * kv_bytes), k_out_i, kv_bytes, stream)?;
            fwd.gpu
                .copy_d2d_async(v_scratch.offset(i * kv_bytes), v_out_i, kv_bytes, stream)?;
        }

        // 2026-09-25: The q/k norms run later, in `ms_qkv_norms`.
        let _ = (nq, eps);
        Ok(())
    }
}
