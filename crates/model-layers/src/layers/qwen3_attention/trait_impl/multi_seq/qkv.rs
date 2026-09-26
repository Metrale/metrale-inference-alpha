// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Multi-sequence decode Q/K/V projections into `qkv_buf`, one
//! `[Q | K | V]` row per sequence, then the per-request LoRA delta, the deferred
//! Q/gate split and the q/k RMS norms.
//!
//! Owner: model-layers (qwen3 attention).
//! Invariants:
//! - Every projection route leaves the rows in the same layout: Q at offset 0
//!   (`q_proj_dim` wide), K at `q_proj_bytes`, V after K, rows `per_seq_qkv`
//!   bytes apart. When a route succeeds, the LoRA and norm passes run after it.
//!
//! `ms_phase_qkv` routes: n = 3, n = 2 or n > 3 with NVFP4 q/k/v (`ms_qkv_batch3`,
//! `ms_qkv_batch2`, `ms_qkv_batchn`); block-scaled FP8 q/k/v
//! (`qkv_fp8_batch.rs`); n in 2..=8 with ungated dense BF16 q/k/v
//! (`ms_qkv_batchm_bf16`); otherwise one GEMV per row per projection, which
//! handles every weight encoding.

use anyhow::Result;

use super::ctx::MultiSeqCtx;
use crate::layers::ops;
use crate::layers::qwen3_attention::Qwen3AttentionLayer;

mod batch;
mod post;
mod rows;

/// 2026-09-25: The batched dense-BF16 decode GEMVs (q/k/v here; the O and head
/// gate projections in `attn/o_proj.rs`) are on unless `METRALE_BF16_QKV_BATCHM=0`.
/// Read once per process, so the route cannot change between graph replays.
pub(super) fn bf16_batchm_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("METRALE_BF16_QKV_BATCHM").ok().as_deref() != Some("0"))
}

/// 2026-09-25: The fused `[q | k | v]` GEMM in `ms_qkv_batchn` (one launch
/// instead of three) is on unless `METRALE_NO_FUSED_QKV=1`, read once per process.
fn fused_qkv_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("METRALE_NO_FUSED_QKV").ok().as_deref() != Some("1"))
}

impl Qwen3AttentionLayer {
    pub(super) fn ms_phase_qkv(&self, c: &MultiSeqCtx<'_>) -> Result<()> {
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
            q_dim,
            q_proj_dim,
            q_proj_bytes,
            per_seq_qkv,
            normed,
            qkv_buf,
            ..
        } = *c;

        if n == 3
            && self.q_weight.as_ref().and_then(|w| w.as_nvfp4()).is_some()
            && self.k_weight.as_ref().and_then(|w| w.as_nvfp4()).is_some()
            && self.v_weight.as_ref().and_then(|w| w.as_nvfp4()).is_some()
        {
            self.ms_qkv_batch3(c)?;
        } else if n == 2
            && self.q_weight.as_ref().and_then(|w| w.as_nvfp4()).is_some()
            && self.k_weight.as_ref().and_then(|w| w.as_nvfp4()).is_some()
            && self.v_weight.as_ref().and_then(|w| w.as_nvfp4()).is_some()
        {
            self.ms_qkv_batch2(c)?;
        } else if n > 3
            && self.q_weight.as_ref().and_then(|w| w.as_nvfp4()).is_some()
            && self.k_weight.as_ref().and_then(|w| w.as_nvfp4()).is_some()
            && self.v_weight.as_ref().and_then(|w| w.as_nvfp4()).is_some()
        {
            // 2026-09-25: One projection launch per weight for all n rows.
            self.ms_qkv_batchn(c)?;
        } else if self.ms_qkv_batchm_fp8_selected(c, super::qkv_fp8_batch::fp8_batchm_enabled()) {
            // 2026-09-25: Batched block-scaled FP8. It needs all three weights FP8
            // (`qkv_fp8_block_scaled`); the NVFP4 routes above need all three
            // NVFP4 and the BF16 route below needs neither, so their order does
            // not matter.
            self.ms_qkv_batchm_fp8(c)?;
        } else if (2..=8).contains(&n)
            && !self.gated
            && self.dense_gemv_batchm_k.0 != 0
            && self.qkv_is_dense_bf16()
            && bf16_batchm_enabled()
        {
            // 2026-09-25: Batched dense BF16: one pass over each weight for all n
            // rows. On the decode path `n` is `padded_batch_n` of the batch
            // (`decode_a2.rs`), which the decode graph key includes, so a branch on
            // it is the same at capture and replay, as for the n = 2 and n = 3
            // branches above.
            self.ms_qkv_batchm_bf16(c)?;
        } else {
            for i in 0..n {
                let normed_i = normed.offset(i * h * bf16);
                let q_out_i = qkv_buf.offset(i * per_seq_qkv);
                let k_out_i = q_out_i.offset(q_proj_bytes);
                let v_out_i = k_out_i.offset((nkv * hd) as usize * bf16);

                self.ms_qkv_seq_q(fwd, normed_i, q_out_i, q_proj_dim, q_dim, nq, hd, h, stream)?;
                self.ms_qkv_seq_kv(fwd, normed_i, k_out_i, v_out_i, nkv, hd, h, stream)?;
            }
        }

        // 2026-09-25: Per-request Q/K/V LoRA delta (batched bgmv), before the
        // norms. On a gated layer with a q adapter the projection routes leave
        // Q as raw interleaved `[Q | gate]` (except the packed Q2_0 row path,
        // which splits it inline), the delta folds onto that, and
        // `ms_qkv_deinterleave_q` splits it afterwards.
        self.ms_qkv_apply_lora(c)?;
        self.ms_qkv_deinterleave_q(c)?;

        // 2026-09-25: The q/k norms run after the LoRA delta, so they normalise
        // the adapted projection.
        let _ = eps;
        self.ms_qkv_norms(c)?;
        Ok(())
    }

    /// 2026-09-25: True when none of q/k/v has an NVFP4 or FP8 weight, so the
    /// BF16 route reads `self.attn.{q,k,v}_proj`. A packed Q2_0 weight is
    /// neither, so it also counts as dense here.
    fn qkv_is_dense_bf16(&self) -> bool {
        let dense = |w: &Option<crate::weight_map::QuantWeight>| {
            w.as_ref()
                .is_none_or(|w| w.as_nvfp4().is_none() && w.as_fp8().is_none())
        };
        dense(&self.q_weight) && dense(&self.k_weight) && dense(&self.v_weight)
    }

    /// 2026-09-25: Batched dense-BF16 q/k/v: one pass over each weight for all
    /// `n` rows, written straight into `qkv_buf` through the kernel's
    /// `out_stride`, with no scratch or scatter.
    ///
    /// Each row is bit-identical to the per-row `dense_gemv` route: the
    /// `dense_gemv_bf16_batchm.cu` header states the per-row order matches
    /// `dense_gemv_bf16`, and the build passes `--fmad=false`. The check is
    /// `examples/dense_gemv_bf16_batchm_microtest`.
    fn ms_qkv_batchm_bf16(&self, c: &MultiSeqCtx<'_>) -> Result<()> {
        let MultiSeqCtx {
            fwd,
            n,
            stream,
            h,
            nkv,
            hd,
            bf16,
            q_proj_dim,
            q_proj_bytes,
            per_seq_qkv,
            normed,
            qkv_buf,
            ..
        } = *c;

        // 2026-09-25: Output rows are `per_seq_qkv` bytes apart; the kernel takes
        // the stride in BF16 elements.
        debug_assert_eq!(per_seq_qkv % bf16, 0);
        let out_stride = (per_seq_qkv / bf16) as u32;
        let kv_dim = nkv * hd;
        let kv_bytes = kv_dim as usize * bf16;

        let gemv = |w, out, n_out| {
            ops::dense_gemv_batchm(
                fwd.gpu,
                self.dense_gemv_batchm_k,
                normed,
                w,
                out,
                n as u32,
                n_out,
                h as u32,
                out_stride,
                stream,
            )
        };

        gemv(&self.attn.q_proj, qkv_buf, q_proj_dim)?;
        gemv(&self.attn.k_proj, qkv_buf.offset(q_proj_bytes), kv_dim)?;
        gemv(
            &self.attn.v_proj,
            qkv_buf.offset(q_proj_bytes + kv_bytes),
            kv_dim,
        )?;
        Ok(())
    }

    /// 2026-09-25: True when the layer's LoRA weights include a q adapter
    /// (`lora.q`). Gated projection routes then write raw interleaved
    /// `[Q | gate]` and leave the split to `ms_qkv_deinterleave_q`.
    pub(super) fn q_lora_active(&self) -> bool {
        self.lora.as_ref().and_then(|lw| lw.q.as_ref()).is_some()
    }

    /// 2026-09-25: The q/k RMS norms over every row's Q and K heads, for all
    /// projection routes, after the LoRA delta. A layer without a q or k norm
    /// weight skips that norm.
    fn ms_qkv_norms(&self, c: &MultiSeqCtx<'_>) -> Result<()> {
        let MultiSeqCtx {
            fwd,
            n,
            stream,
            nq,
            nkv,
            hd,
            eps,
            bf16,
            q_proj_bytes,
            per_seq_qkv,
            qkv_buf,
            ..
        } = *c;
        // 2026-09-25: One launch per norm for all n rows: a row's heads are `hd`
        // apart and rows are `per_seq_qkv` apart, the (rows_per_group,
        // num_groups, row_stride) shape `rms_norm_strided` takes. Its kernel
        // header (`rms_norm.cu`) states it is bit-identical to `rms_norm`, one
        // block per row. Off when `METRALE_NO_QK_NORM_STRIDED=1`, read once.
        fn qk_norm_strided_enabled() -> bool {
            static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *ON.get_or_init(|| {
                std::env::var("METRALE_NO_QK_NORM_STRIDED").ok().as_deref() != Some("1")
            })
        }
        if n > 1
            && self.rms_norm_strided_k.0 != 0
            && qk_norm_strided_enabled()
            && per_seq_qkv.is_multiple_of(bf16)
        {
            let stride_e = (per_seq_qkv / bf16) as u32;
            if !self.attn.q_norm.weight.is_null() {
                ops::rms_norm_strided(
                    fwd.gpu,
                    self.rms_norm_strided_k,
                    qkv_buf,
                    &self.attn.q_norm,
                    qkv_buf,
                    nq,
                    n as u32,
                    hd,
                    eps,
                    stride_e,
                    stream,
                )?;
            }
            if !self.attn.k_norm.weight.is_null() {
                let k0 = qkv_buf.offset(q_proj_bytes);
                ops::rms_norm_strided(
                    fwd.gpu,
                    self.rms_norm_strided_k,
                    k0,
                    &self.attn.k_norm,
                    k0,
                    nkv,
                    n as u32,
                    hd,
                    eps,
                    stride_e,
                    stream,
                )?;
            }
            return Ok(());
        }
        for i in 0..n {
            let q_out_i = qkv_buf.offset(i * per_seq_qkv);
            let k_out_i = q_out_i.offset(q_proj_bytes);
            if !self.attn.q_norm.weight.is_null() {
                ops::rms_norm(
                    fwd.gpu,
                    self.rms_norm_w_k,
                    q_out_i,
                    &self.attn.q_norm,
                    q_out_i,
                    nq,
                    hd,
                    eps,
                    stream,
                )?;
            }
            if !self.attn.k_norm.weight.is_null() {
                ops::rms_norm(
                    fwd.gpu,
                    self.rms_norm_w_k,
                    k_out_i,
                    &self.attn.k_norm,
                    k_out_i,
                    nkv,
                    hd,
                    eps,
                    stream,
                )?;
            }
        }
        Ok(())
    }

    /// 2026-09-25: One NVFP4 projection of `m` rows, `[m, k]` to `[m, n]`. Takes
    /// the first that applies: the batched GEMV `w4a16_batchm.kernel(m)` on the
    /// base weight `w_base`; with a transposed weight `w_t`, the small-M tile
    /// GEMMs (`m <= 64`, `k % 32 == 0`, `ffn_small_m`), then the M128 v2 and
    /// M128 GEMMs; otherwise `w4a16_gemm` on `w_base`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn wide_verify_gemm(
        &self,
        c: &MultiSeqCtx<'_>,
        input: metrale_gpu_runtime::gpu::DevicePtr,
        w_base: &crate::weight_map::QuantizedWeight,
        w_t: Option<&crate::weight_map::QuantizedWeight>,
        output: metrale_gpu_runtime::gpu::DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        same_input_as_previous: bool,
    ) -> Result<()> {
        let gpu = c.fwd.gpu;
        let stream = c.stream;
        // 2026-09-25: The batched GEMV reads the base (non-transposed) weight
        // once for all rows. `w4a16_batchm.kernel(m)` covers m <= 8, and more
        // rows only in the wide modes (`w4a16_gemv_tiers.rs`); otherwise it is
        // zero and the GEMMs below run.
        let batchm = self.w4a16_batchm.kernel(m);
        if batchm.0 != 0 {
            return if same_input_as_previous {
                ops::w4a4_proj::nvfp4_proj_small_m_same_input(
                    gpu, batchm, input, w_base, output, m, n, k, stream,
                )
            } else {
                ops::w4a4_proj::nvfp4_proj_small_m(
                    gpu, batchm, input, w_base, output, m, n, k, stream,
                )
            };
        }
        if let Some(wt) = w_t {
            // 2026-09-25: Small-M routing, the same rule as
            // `dense_ffn::w4a16_prefill_gemm`, under the same lever
            // `ffn_small_m` (off when `METRALE_FFN_SMALLM=0`).
            if m <= 64 && k.is_multiple_of(32) && c.fwd.levers.ffn_small_m {
                if k >= crate::layers::w4a16_k64_min_k()
                    && k.is_multiple_of(64)
                    && self.w4a16_gemm_t_k64_k.0 != 0
                {
                    // 2026-09-25: The N64 twin of the deep-K kernel, when
                    // `k64_n64_wins(m, n)`.
                    if self.w4a16_gemm_t_k64_n64_k.0 != 0 && crate::layers::k64_n64_wins(m, n) {
                        return ops::w4a16_gemm(
                            gpu,
                            self.w4a16_gemm_t_k64_n64_k,
                            input,
                            wt,
                            output,
                            m,
                            n,
                            k,
                            stream,
                        );
                    }
                    return ops::w4a16_gemm_n128(
                        gpu,
                        self.w4a16_gemm_t_k64_k,
                        input,
                        wt,
                        output,
                        m,
                        n,
                        k,
                        stream,
                    );
                }
                if self.w4a16_gemm_t_k.0 != 0 {
                    return ops::w4a16_gemm_n128(
                        gpu,
                        self.w4a16_gemm_t_k,
                        input,
                        wt,
                        output,
                        m,
                        n,
                        k,
                        stream,
                    );
                }
            }
            if self.w4a16_gemm_t_m128_v2_k.0 != 0 {
                return ops::w4a16_gemm_n128_m128_v2(
                    gpu,
                    self.w4a16_gemm_t_m128_v2_k,
                    input,
                    wt,
                    output,
                    m,
                    n,
                    k,
                    stream,
                );
            }
            if self.w4a16_gemm_t_m128_k.0 != 0 {
                return ops::w4a16_gemm_n128_m128(
                    gpu,
                    self.w4a16_gemm_t_m128_k,
                    input,
                    wt,
                    output,
                    m,
                    n,
                    k,
                    stream,
                );
            }
        }
        ops::w4a16_gemm(
            gpu,
            self.w4a16_gemm_k,
            input,
            w_base,
            output,
            m,
            n,
            k,
            stream,
        )
    }
}
