// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The Q/K/V projections of the cache-skip prefill path
//! (`prefill_attention_with_cache_skip`, non-MLA).
//!
//! One W8A8 decision is made for all three projections; each projection then
//! takes the first arm of `cache_skip_one_proj` that applies. The FP8xFP8 arm
//! reads the pre-converted `normed_fp8` activations.
//!
//! Owner: model-layers (attention).
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::super::Qwen3AttentionLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;

pub(super) enum SkipProj {
    Q,
    K,
    V,
}

impl Qwen3AttentionLayer {
    /// 2026-09-25: Runs Q, then K, then V. Q is written to `qkv_output`, K to
    /// `ssm_qkvz`, and V directly after K's `num_tokens` rows.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prefill_attention_cache_skip_qkv(
        &self,
        normed: DevicePtr,
        normed_fp8: DevicePtr,
        n: u32,
        h: u32,
        nkv: u32,
        hd: u32,
        q_proj_dim: usize,
        kv_dim: usize,
        num_tokens: usize,
        bf16: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        macro_rules! prof_step {
            ($label:expr, $t0:expr) => {
                if ctx.profile {
                    ctx.gpu.synchronize(stream)?;
                    tracing::info!(
                        "  ATTN prefill [{}] N={}: {}µs",
                        $label,
                        n,
                        $t0.elapsed().as_micros()
                    );
                }
            };
        }

        // 2026-09-25: One W8A8 decision for all three projections and, when it
        // is taken, one shared activation quantization. `prefill_qkv_w8a8.rs`
        // explains why the decision is all-or-none.
        let w8a8 = self.cache_skip_qkv_w8a8_selected(ctx, n, q_proj_dim as u32, nkv * hd, h);
        self.log_cache_skip_qkv_route(ctx, w8a8);
        if w8a8 {
            self.cache_skip_qkv_w8a8_quant(ctx, normed, n, h, stream)?;
        }
        let qg_out = ctx.buffers.qkv_output();
        let t0 = std::time::Instant::now();
        self.cache_skip_one_proj(
            SkipProj::Q,
            normed,
            normed_fp8,
            qg_out,
            n,
            q_proj_dim as u32,
            h,
            w8a8,
            ctx,
            stream,
        )?;
        prof_step!("q_proj", t0);
        super::super::op_dump::dump_bf16(
            ctx.gpu,
            qg_out,
            (num_tokens - 1) * q_proj_dim * bf16,
            q_proj_dim,
            self.attn_layer_idx,
            "q_proj_full",
            stream,
        )?;
        let k_contiguous = ctx.buffers.ssm_qkvz();
        let t0 = std::time::Instant::now();
        self.cache_skip_one_proj(
            SkipProj::K,
            normed,
            normed_fp8,
            k_contiguous,
            n,
            nkv * hd,
            h,
            w8a8,
            ctx,
            stream,
        )?;
        prof_step!("k_proj", t0);
        super::super::op_dump::dump_bf16(
            ctx.gpu,
            k_contiguous,
            (num_tokens - 1) * kv_dim * bf16,
            kv_dim,
            self.attn_layer_idx,
            "k_proj",
            stream,
        )?;
        let v_contiguous = k_contiguous.offset(num_tokens * kv_dim * bf16);
        let t0 = std::time::Instant::now();
        self.cache_skip_one_proj(
            SkipProj::V,
            normed,
            normed_fp8,
            v_contiguous,
            n,
            nkv * hd,
            h,
            w8a8,
            ctx,
            stream,
        )?;
        prof_step!("v_proj", t0);
        super::super::op_dump::dump_bf16(
            ctx.gpu,
            v_contiguous,
            (num_tokens - 1) * kv_dim * bf16,
            kv_dim,
            self.attn_layer_idx,
            "v_proj",
            stream,
        )?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn cache_skip_one_proj(
        &self,
        proj: SkipProj,
        normed: DevicePtr,
        normed_fp8: DevicePtr,
        out: DevicePtr,
        n: u32,
        out_dim: u32,
        h: u32,
        w8a8: bool,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        // 2026-09-25: Q uses its transposed FP8 copy only with
        // `METRALE_ATTN_PREFILL_Q_T=1`; K and V use theirs whenever present.
        let use_q_t = std::env::var("METRALE_ATTN_PREFILL_Q_T").ok().as_deref() == Some("1");
        let (fp8w_t, weight_opt, fp8, nvfp4_t, dense, label) = match proj {
            SkipProj::Q => (
                use_q_t.then_some(self.q_fp8w_t.as_ref()).flatten(),
                self.q_weight.as_ref(),
                self.q_fp8,
                self.q_nvfp4_t.as_ref(),
                &self.attn.q_proj,
                "q_proj",
            ),
            SkipProj::K => (
                self.k_fp8w_t.as_ref(),
                self.k_weight.as_ref(),
                self.k_fp8,
                self.k_nvfp4_t.as_ref(),
                &self.attn.k_proj,
                "k_proj",
            ),
            SkipProj::V => (
                self.v_fp8w_t.as_ref(),
                self.v_weight.as_ref(),
                self.v_fp8,
                self.v_nvfp4_t.as_ref(),
                &self.attn.v_proj,
                "v_proj",
            ),
        };

        // 2026-09-25: Keep-packed Q2_0 weights go through `try_q2_prefill`
        // first; the arms below would read null pointers for them.
        if let Some(r) = self.try_q2_prefill(ctx, weight_opt, normed, out, n, stream) {
            return r;
        }

        let use_t_pipelined =
            std::env::var("METRALE_ATTN_PREFILL_T_PIPE").ok().as_deref() == Some("1");
        if ctx.dispatch.cutlass_nvfp4_attn_qkv(label)
            && let Some(nvfp4_t) = nvfp4_t
        {
            ops::log_cutlass_nvfp4_route(ctx.gpu, label, n, out_dim, h);
            ops::cutlass_nvfp4_proj(ctx, normed, nvfp4_t, out, n, out_dim, h, stream)?;
        } else if ctx.dispatch.cutlass_nvfp4_attn_qkv(label)
            && let Some(fp8w) = weight_opt.and_then(|w| w.as_fp8())
        {
            ops::log_cutlass_nvfp4_route(ctx.gpu, label, n, out_dim, h);
            ops::cutlass_nvfp4_proj_from_fp8(ctx, normed, fp8w, out, n, out_dim, h, stream)?;
        // 2026-09-25: The cuBLASLt W8A8 arm; the caller decides `w8a8` once
        // per chain. cuBLASLt writes `ceil16(M)` rows, so K's padding rows land
        // in V's region; V is written after K and its real rows cover them
        // because the selector requires `m >= 16` (`prefill_qkv_w8a8.rs`). This
        // chain must not allocate device memory; `alloc_tests.rs` checks it.
        } else if w8a8 && let Some(fp8w) = weight_opt.and_then(|w| w.as_fp8()) {
            self.cache_skip_qkv_w8a8_gemm(ctx, fp8w, out, n, out_dim, h, stream)?;
        } else if let Some(fp8t) = fp8w_t
            && use_t_pipelined
            && self.w8a16_gemm_t_pipelined_k.0 != 0
        {
            ops::w8a16_gemm_t_pipelined(
                ctx.gpu,
                self.w8a16_gemm_t_pipelined_k,
                normed,
                fp8t.weight_t,
                fp8t.scale_t,
                out,
                n,
                out_dim,
                h,
                stream,
            )?;
        } else if let Some(fp8t) = fp8w_t
            && self.w8a16_gemm_t_m128_k.0 != 0
        {
            ops::w8a16_gemm_n128_m128(
                ctx.gpu,
                self.w8a16_gemm_t_m128_k,
                normed,
                fp8t.weight_t,
                fp8t.scale_t,
                out,
                n,
                out_dim,
                h,
                stream,
            )?;
        } else if let Some(fp8t) = fp8w_t {
            ops::w8a16_gemm_t(
                ctx.gpu,
                self.w8a16_gemm_t_k,
                normed,
                fp8t.weight_t,
                fp8t.scale_t,
                out,
                n,
                out_dim,
                h,
                stream,
            )?;
        } else if weight_opt.and_then(|w| w.as_fp8()).is_some()
            && self.w8a16_gemm_pipelined_k.0 != 0
        {
            let fp8w = weight_opt.and_then(|w| w.as_fp8()).unwrap();
            ops::w8a16_gemm_pipelined(
                ctx.gpu,
                self.w8a16_gemm_pipelined_k,
                normed,
                fp8w.weight,
                fp8w.row_scale,
                out,
                n,
                out_dim,
                h,
                stream,
            )?;
        } else if weight_opt.and_then(|w| w.as_fp8()).is_some() && self.w8a16_gemm_k.0 != 0 {
            let fp8w = weight_opt.and_then(|w| w.as_fp8()).unwrap();
            // 2026-09-25: For targets without `w8a16_gemm_pipelined`; strix-hip
            // builds only `w8a16_gemm.cu`.
            ops::w8a16_gemm(
                ctx.gpu,
                self.w8a16_gemm_k,
                normed,
                fp8w.weight,
                fp8w.row_scale,
                out,
                n,
                out_dim,
                h,
                stream,
            )?;
        } else if weight_opt.and_then(|w| w.as_fp8()).is_some() {
            anyhow::bail!("w8a16_gemm kernel not loaded — cannot prefill with FP8 weights");
        } else if let Some(fp8p) = fp8 {
            if n > 128 {
                ops::fp8_fp8_gemm_n128_m128(
                    ctx.gpu,
                    self.fp8_fp8_gemm_t_m128_k,
                    normed_fp8,
                    fp8p,
                    out,
                    n,
                    out_dim,
                    h,
                    stream,
                )?;
            } else {
                ops::fp8_fp8_gemm_n128(
                    ctx.gpu,
                    self.fp8_fp8_gemm_k,
                    normed_fp8,
                    fp8p,
                    out,
                    n,
                    out_dim,
                    h,
                    stream,
                )?;
            }
        } else if let Some(nvfp4_t) = nvfp4_t {
            if n > 128 {
                self.w4a16_gemm_m128_dispatch(
                    ctx.gpu,
                    ctx.dispatch,
                    normed,
                    nvfp4_t,
                    out,
                    n,
                    out_dim,
                    h,
                    stream,
                )?;
            } else {
                ops::w4a16_gemm_n128(
                    ctx.gpu,
                    self.w4a16_gemm_t_k,
                    normed,
                    nvfp4_t,
                    out,
                    n,
                    out_dim,
                    h,
                    stream,
                )?;
            }
        } else if let Some(nvfp4) = weight_opt.and_then(|w| w.as_nvfp4()) {
            ops::w4a16_gemm(
                ctx.gpu,
                self.w4a16_gemm_k,
                normed,
                nvfp4,
                out,
                n,
                out_dim,
                h,
                stream,
            )
            .map_err(|e| {
                anyhow::anyhow!("{label} w4a16_gemm failed: m={n} n={out_dim} k={h}: {e}")
            })?;
        } else if ctx.dispatch.cublas.attn && n > 1 {
            ops::cublas_bf16_proj_dense(normed, dense.weight, out, n, out_dim, h, stream)?;
        } else {
            ops::dense_gemm_prefill(
                ctx.gpu,
                self.dense_gemm_k,
                self.dense_gemm_pipelined_k,
                normed,
                dense,
                out,
                n,
                out_dim,
                h,
                stream,
            )?;
        }
        Ok(())
    }
}
