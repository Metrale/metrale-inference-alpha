// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: W8A8 block-scaled cuBLASLt arm for the attention decode projections at 5..=16 rows: Q/K/V into the strided multi-sequence QKV buffer, and O contiguous.
//!
//! Owner: model-layers (qwen3 attention, multi-sequence decode).
//! Invariants:
//! - Q, K and V take this arm together or not at all.
//! - A projection takes it only when its full padded write extent (`ceil16(n)`
//!   rows at the output row pitch, counted from that projection's own offset)
//!   fits in the output buffer; `ops::decode_w8a8_selected` checks it.
//! - The rows' arithmetic moves from W8A16 to W8A8 (E4M3 activations with
//!   per-token 1x128 scales). `METRALE_NO_W8A8_DECODE_PROJ` keeps the GEMV arms.

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::ctx::MultiSeqCtx;
use crate::layer::ForwardContext;
use crate::layers::ops;
use crate::layers::qwen3_attention::Qwen3AttentionLayer;
use crate::weight_map::Fp8Weight;

/// 2026-09-25: The [`ops::CublasScope`] bit that arms this family (`attn`), in
/// one function so a test and the dispatch site read the same bit.
/// `METRALE_CUBLAS_GEMM=ffn` does not arm it.
pub(super) fn attn_decode_family_armed(scope: ops::CublasScope) -> bool {
    scope.attn
}

impl Qwen3AttentionLayer {
    /// 2026-09-25: The activation-quant scratch for the decode W8A8 arm: the
    /// same arena buffers the attention prefill W8A8 projections use.
    fn decode_w8a8_scratch(&self, fwd: &ForwardContext) -> ops::DecodeW8a8Scratch {
        ops::DecodeW8a8Scratch {
            act_fp8: fwd.buffers.fp8_act(),
            act_fp8_bytes: fwd.buffers.fp8_act_bytes(),
            act_scale: fwd.buffers.fp8_act_scale(),
            act_scale_bytes: fwd.buffers.fp8_act_scale_bytes(),
            act_scale_kmajor: fwd.buffers.fp8_act_scale_kmajor(),
            act_scale_kmajor_bytes: fwd.buffers.fp8_act_scale_kmajor_bytes(),
            quant_k: self.per_token_group_quant_fp8_k,
            scale_kmajor_k: self.fp8_act_scale_kmajor_k,
        }
    }

    fn decode_w8a8_selected(
        &self,
        fwd: &ForwardContext,
        plan: &ops::DecodeW8a8Plan,
        fp8w: &Fp8Weight,
    ) -> bool {
        ops::decode_w8a8_selected(
            attn_decode_family_armed(fwd.dispatch.cublas),
            ops::w8a8_decode_proj_disabled(),
            plan,
            fp8w.scale_format,
            &self.decode_w8a8_scratch(fwd),
        )
    }

    /// 2026-09-25: The Q, K and V plans for this step, each bounded from its own
    /// offset inside `qkv_output` (Q at 0, K at `q_proj_bytes`, V after K), so
    /// the write-extent check shrinks with the offset.
    ///
    /// `qkv_capacity` is the allocated size of `qkv_output`; `ldc` is
    /// `per_seq_qkv` in BF16 elements, the row pitch of the per-row layout.
    pub(super) fn qkv_decode_w8a8_plans(
        &self,
        c: &MultiSeqCtx<'_>,
        kv_dim: u32,
        qkv_capacity: usize,
    ) -> [(usize, ops::DecodeW8a8Plan); 3] {
        let ldc = (c.per_seq_qkv / c.bf16) as u32;
        let kv_bytes = kv_dim as usize * c.bf16;
        let plan = |offset: usize, n_out: u32| {
            (
                offset,
                ops::DecodeW8a8Plan::strided(
                    c.n,
                    n_out,
                    c.h as u32,
                    ldc,
                    qkv_capacity.saturating_sub(offset),
                ),
            )
        };
        [
            plan(0, c.q_proj_dim),
            plan(c.q_proj_bytes, kv_dim),
            plan(c.q_proj_bytes + kv_bytes, kv_dim),
        ]
    }

    /// 2026-09-25: Route Q/K/V through cuBLASLt W8A8 if every projection
    /// qualifies; `Ok(false)` leaves all three to the caller's GEMV arms.
    ///
    /// All or nothing because the three share one activation quantization: a
    /// split route would pay the quantizer and still keep a GEMV weight pass.
    /// They share K, `ldc` and the row count; each plan's output width and
    /// write extent are checked on their own.
    pub(super) fn try_ms_qkv_decode_w8a8(
        &self,
        c: &MultiSeqCtx<'_>,
        q: &Fp8Weight,
        k: &Fp8Weight,
        v: &Fp8Weight,
        kv_dim: u32,
    ) -> Result<bool> {
        let fwd = c.fwd;
        let plans = self.qkv_decode_w8a8_plans(c, kv_dim, fwd.buffers.qkv_output_bytes());
        let weights = [q, k, v];
        if !plans
            .iter()
            .zip(weights)
            .all(|((_, plan), w)| self.decode_w8a8_selected(fwd, plan, w))
        {
            return Ok(false);
        }
        let scratch = self.decode_w8a8_scratch(fwd);
        self.log_decode_w8a8_route(fwd, "q/k/v", c.n, c.h as u32);
        // 2026-09-25: One quantization of `normed` for all three: Q, K and V
        // read the same `[n, h]` rows.
        ops::decode_w8a8_quant_act(
            fwd.gpu, &scratch, c.normed, c.n as u32, c.h as u32, c.stream,
        )?;
        for ((offset, plan), w) in plans.iter().zip(weights) {
            ops::decode_w8a8_gemm(&scratch, w, c.qkv_buf.offset(*offset), plan, c.stream)?;
        }
        Ok(true)
    }

    /// 2026-09-25: Route the O projection through cuBLASLt W8A8; `Ok(false)`
    /// leaves it to the caller's GEMV arms.
    ///
    /// Contiguous on both sides: `attn_out` is `[n, q_dim]` and `o_out` is
    /// `[n, hidden]`, so `ldc` is the output width.
    pub(super) fn try_ms_o_proj_decode_w8a8(
        &self,
        c: &MultiSeqCtx<'_>,
        o_fp8: &Fp8Weight,
        attn_out: DevicePtr,
        o_out: DevicePtr,
    ) -> Result<bool> {
        let fwd = c.fwd;
        let k = c.nq * c.hd;
        let plan =
            ops::DecodeW8a8Plan::contiguous(c.n, c.h as u32, k, fwd.buffers.moe_output_bytes());
        if !self.decode_w8a8_selected(fwd, &plan, o_fp8) {
            return Ok(false);
        }
        let scratch = self.decode_w8a8_scratch(fwd);
        self.log_decode_w8a8_route(fwd, "o_proj", c.n, k);
        ops::decode_w8a8_quant_act(fwd.gpu, &scratch, attn_out, c.n as u32, k, c.stream)?;
        ops::decode_w8a8_gemm(&scratch, o_fp8, o_out, &plan, c.stream)?;
        Ok(true)
    }

    /// 2026-09-25: Log once per family (Q/K/V, O) that these rows took W8A8,
    /// so a quality report can state which arithmetic produced it; the line
    /// names the switch that restores the GEMV arms.
    fn log_decode_w8a8_route(&self, fwd: &ForwardContext, what: &str, rows: usize, k: u32) {
        let key = if what == "o_proj" {
            "log:attn_o_proj_w8a8_decode"
        } else {
            "log:attn_qkv_w8a8_decode"
        };
        if fwd.stats.once(key) {
            tracing::info!(
                "[metrale] attention {what} decode (n={rows} rows, K={k}): W8A8 block-scaled via \
                 cuBLASLt (per-token 1x128 act scales x 128x128 weight scales, FP32 epilogue; \
                 vLLM-equivalent FP8 numerics), replacing w8a16_gemv_batch16. \
                 METRALE_NO_W8A8_DECODE_PROJ restores the GEMV tier; M=1 decode is untouched."
            );
        }
    }
}

#[cfg(test)]
#[path = "w8a8_decode_tests.rs"]
mod tests;
