// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Multi-row forward of a Qwen3 SSM layer (`decode_batched_inner`): K rows of
//! one sequence, or the Σ ks rows of a batched MTP verify, through norm, QKVZ, BA gates,
//! conv + GDN, gated norm, out_proj and FFN.
//!
//! Owner: model-layers (Qwen3 SSM layer).
//! Invariants:
//! - Rows are sequence-major. For `GdnStates::Multi`, `num_tokens == Σ ks` and
//!   `ks.len() == states.len()` are checked before any conv/GDN launch; a mismatch returns
//!   an error.

use super::*;

mod conv_gdn_route;
mod gates_norm;
mod out_proj;
mod qkvz_proj;

/// 2026-09-25: With the `k4_diag` lever (`METRALE_K4_DIAG=1`), synchronize the stream after
/// the named phase and return a device error tagged with that phase. It does nothing under
/// graph capture; the verify entry points (model-engine `verify_c2.rs`, `verify_e.rs`) run
/// without a graph when the lever is set.
fn k4_diag_checkpoint(ctx: &ForwardContext, phase: &str, stream: u64) -> Result<()> {
    let on = ctx.levers.k4_diag;
    if on
        && !ctx.graph_capture
        && let Err(e) = ctx.gpu.synchronize(stream)
    {
        anyhow::bail!("K4_DIAG: CUDA error after GDN phase `{phase}`: {e:#}");
    }
    Ok(())
}

/// 2026-09-25: Kill switch for the single-launch BA projection + GDN gates:
/// `METRALE_NO_BATCHED_BA_GATES` set to any value, `0` included, selects the per-token
/// GEMV + `compute_gdn_gates` pair. Read once per process.
fn batched_ba_gates_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("METRALE_NO_BATCHED_BA_GATES").is_err())
}

/// 2026-09-25: Kill switch for the single-launch gated RMS norm:
/// `METRALE_NO_BATCHED_GDN_NORM` set to any value, `0` included, selects the per-token
/// loop. Read once per process.
fn batched_norm_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("METRALE_NO_BATCHED_GDN_NORM").is_err())
}

/// 2026-09-25: Row count above which the batched QKVZ and out_proj read the NVFP4
/// transposed twin through a tile GEMM (`ms_proj_gemm`) ahead of the single-scale FP8
/// prefill-copy arm (`qkvz_fp8` / `out_proj_fp8`). Both projections use this one value.
/// It equals `ops::gemv_tc::NARROW_MAX_ROWS`, the default row edge of the NVFP4 GEMV arms
/// earlier in the dispatch (`ops::w4a4_proj::proj_max_rows`).
pub(super) const VERIFY_TGEMM_MIN_TOKENS: usize = 8;

/// 2026-09-25: Kill switch for the NVFP4 QKVZ arm above `VERIFY_TGEMM_MIN_TOKENS` rows:
/// `METRALE_NO_QKVZ_NVFP4_DECODE` set to any value, `0` included, keeps the FP8
/// prefill-copy arm. Read once per process.
fn qkvz_nvfp4_decode_off() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| std::env::var("METRALE_NO_QKVZ_NVFP4_DECODE").is_ok())
}

/// 2026-09-25: Whether the batched QKVZ projection reads the NVFP4 transposed twin instead
/// of the single-scale FP8 prefill copy. Pure: the row count, which copies the layer holds
/// and the resolved kill switch are all parameters.
///
/// True only when the FP8 prefill copy exists (the arm replaces that arm and no other),
/// the NVFP4 twin exists, and `has_tile_gemm` (`deep_k_gemm(K).0 != 0`, the handle
/// `ms_proj_gemm` falls back to) is set, so a zero handle is never launched.
pub(super) fn qkvz_verify_nvfp4_wins(
    num_tokens: usize,
    has_fp8_prefill: bool,
    has_nvfp4_t: bool,
    has_tile_gemm: bool,
    kill_switch: bool,
) -> bool {
    !kill_switch
        && has_fp8_prefill
        && has_nvfp4_t
        && has_tile_gemm
        && num_tokens > VERIFY_TGEMM_MIN_TOKENS
}

/// 2026-09-26: Whether the out_proj arm above `VERIFY_TGEMM_MIN_TOKENS` rows that reads the
/// NVFP4 transposed twin through `ms_proj_gemm` may run.
fn verify_outproj_tgemm_enabled() -> bool {
    // 2026-09-25: `METRALE_NO_VERIFY_OUTPROJ_TGEMM` set to any value, `0`
    // included, disables this arm. Read once per process.
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    !*OFF.get_or_init(|| std::env::var("METRALE_NO_VERIFY_OUTPROJ_TGEMM").is_ok())
}

/// 2026-09-25: GDN-state routing for [`Qwen3SsmLayer::decode_batched_inner`].
///
/// `Single`: all `num_tokens` rows belong to one sequence, whose conv and GDN state
/// advance through every row.
///
/// `Multi`: batched MTP verify. `num_tokens = Σ ks` sequence-major rows, `ks[i]` of them
/// for sequence `i` (the counts may differ). Projections and FFN run over all rows; the
/// conv/GDN body runs per run of equal `ks` through `decode_batched_conv_gdn_multi`, or
/// per sequence through `decode_batched_conv_gdn` with row-offset buffer bases.
pub(super) enum GdnStates<'a, 'b> {
    Single(&'a mut dyn LayerState),
    Multi {
        states: &'a mut [&'b mut (dyn LayerState + 'static)],
        ks: &'a [usize],
        /// 2026-09-25: This layer's slice of the model-staged WY pointer tables
        /// (`crate::layer::VERIFY_WY_LAYER_STRIDE_BYTES` apart). NULL makes
        /// `decode_batched_conv_gdn_multi` decline, so every sequence runs alone.
        wy_tables: DevicePtr,
    },
}

/// 2026-09-26: The per-call scalars of [`Qwen3SsmLayer::decode_batched_inner`], computed once
/// at its top from `ctx.config`, the row count and the stream. Each field holds the value of
/// the local of the same name there; the phase helpers in the child modules destructure the
/// fields they read.
struct BatchedDims {
    num_tokens: usize,
    k: u32,
    h: usize,
    eps: f32,
    bf16: usize,
    fp32: usize,
    nk: usize,
    kd: usize,
    nv: usize,
    vd: usize,
    vpg: usize,
    key_dim: usize,
    value_dim: usize,
    conv_dim: usize,
    qk_ch: u32,
    d_conv: usize,
    qkvz_size: usize,
    stream: u64,
}

impl Qwen3SsmLayer {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn decode_batched_inner(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        gdn: GdnStates<'_, '_>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size;
        let eps = ctx.config.rms_norm_eps as f32;
        let k = num_tokens as u32;
        let bf16 = 2usize;
        let fp32 = 4usize;

        let nk = ctx.config.linear_num_key_heads;
        let kd = ctx.config.linear_key_head_dim;
        let nv = ctx.config.linear_num_value_heads;
        let vd = ctx.config.linear_value_head_dim;
        let vpg = nv / nk;
        let key_dim = nk * kd;
        let value_dim = nv * vd;
        let conv_dim = key_dim * 2 + value_dim;
        // 2026-09-25: Q and K channels, the ones the conv kernels L2-normalize.
        let qk_ch = (key_dim * 2) as u32;
        let d_conv = ctx.config.linear_conv_kernel_dim;
        let qkvz_size = ctx.config.ssm_qkvz_size();
        let dims = BatchedDims {
            num_tokens,
            k,
            h,
            eps,
            bf16,
            fp32,
            nk,
            kd,
            nv,
            vd,
            vpg,
            key_dim,
            value_dim,
            conv_dim,
            qk_ch,
            d_conv,
            qkvz_size,
            stream,
        };

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

        k4_diag_checkpoint(ctx, "1:rms_norm_residual", stream)?;

        // 2026-09-25: QKVZ projection. A `sequential_qkvz` layer (weights concatenated
        // [Q|K|V|Z] at load, `new_sequential`) projects straight into the deinterleaved
        // buffer; any other layer projects into `ssm_qkvz` and is deinterleaved per row
        // below.
        let deinterleaved = ctx.buffers.ssm_deinterleaved();
        let proj_dst = if self.sequential_qkvz {
            deinterleaved
        } else {
            ctx.buffers.ssm_qkvz()
        };
        self.batched_qkvz_proj(ctx, &dims, normed, proj_dst)?;
        if !self.sequential_qkvz {
            for t in 0..(num_tokens as u32) {
                let src = proj_dst.offset(t as usize * qkvz_size * bf16);
                let dst = deinterleaved.offset(t as usize * qkvz_size * bf16);
                ops::deinterleave_qkvz(
                    ctx.gpu,
                    self.deinterleave_k,
                    src,
                    dst,
                    1,
                    nk as u32,
                    kd as u32,
                    vpg as u32,
                    vd as u32,
                    stream,
                )?;
            }
        }

        k4_diag_checkpoint(ctx, "2+3:qkvz_proj+deinterleave", stream)?;

        // 2026-09-25: BA projection and GDN gates. `gates_buf` holds one
        // [gate(nv) | beta(nv)] FP32 row per token, `nv * 2` elements apart; the WY
        // launches read it with `gb_stride = nv * 2`.
        let gates_buf = ctx.buffers.ssm_gates();
        let gate_beta_stride = nv * 2 * fp32;
        let ba_size = ctx.config.ssm_ba_size();
        self.batched_ba_gates(ctx, &dims, normed, gates_buf, gate_beta_stride, ba_size)?;

        k4_diag_checkpoint(ctx, "4:ba_proj+gates", stream)?;

        // 2026-09-25: Conv1d + L2 norm + GDN. The conv output reuses `ssm_qkvz`, which the
        // deinterleave above has finished reading.
        let (conv_out_buf, gdn_out_buf) =
            self.batched_conv_gdn_route(gdn, ctx, &dims, deinterleaved, gates_buf)?;

        k4_diag_checkpoint(ctx, "5-7:conv1d+l2norm+gdn_wy", stream)?;

        // 2026-09-25: Gated RMS norm; the Z gate of each row starts after its [Q|K|V].
        let normed_out_buf = conv_out_buf;
        let z_offset = key_dim * 2 + value_dim;
        self.batched_gated_norm(
            ctx,
            &dims,
            gdn_out_buf,
            deinterleaved,
            normed_out_buf,
            z_offset,
        )?;

        k4_diag_checkpoint(ctx, "8:gated_rms_norm", stream)?;

        // 2026-09-25: Output projection into `moe_output`, `[num_tokens, h]` BF16.
        let out_proj_buf = ctx.buffers.moe_output();
        self.batched_out_proj(ctx, &dims, normed_out_buf, out_proj_buf)?;

        // 2026-09-25: With `tp_world_size > 1`, all-reduce the partial out_proj
        // (`num_tokens * h` BF16) across ranks; then add the out_proj LoRA delta, if any.
        self.ssm_tp_all_reduce(out_proj_buf, normed_out_buf, num_tokens, ctx, stream)?;

        k4_diag_checkpoint(ctx, "9:out_proj", stream)?;

        // 2026-09-25: Residual add and post-mixer norm over all rows, then the FFN and
        // its residual add.
        let normed2_base = ctx.buffers.norm_output();
        ops::residual_add_rms_norm(
            ctx.gpu,
            self.residual_add_rms_norm_k,
            hidden,
            out_proj_buf,
            &self.post_attn_norm,
            normed2_base,
            residual,
            num_tokens as u32,
            h as u32,
            eps,
            stream,
        )?;
        if num_tokens == 3 {
            self.ffn.forward_k3(normed2_base, ctx, stream)?;
            let moe_out = ctx.buffers.moe_output();
            ops::residual_add(
                ctx.gpu,
                self.residual_add_k,
                hidden,
                moe_out,
                (3 * h) as u32,
                stream,
            )?;
        } else if num_tokens == 2 {
            self.ffn.forward_k2(normed2_base, ctx, stream)?;
            let moe_out = ctx.buffers.moe_output();
            ops::residual_add(
                ctx.gpu,
                self.residual_add_k,
                hidden,
                moe_out,
                (2 * h) as u32,
                stream,
            )?;
        } else if (4..=ops::w4a4_proj::ffn_proj_max_rows() as usize).contains(&num_tokens)
            && self
                .ffn
                .try_forward_km(normed2_base, num_tokens as u32, ctx, stream)
                .inspect_err(|e| tracing::error!("ffn.try_forward_km: {e:#}"))
                .unwrap_or(false)
        {
            // 2026-09-25: Dense FFN through batched GEMVs (`try_forward_km`). When it
            // returns false (MoE, or no kernel for these rows) or an error, which is
            // logged, the arms below run instead.
            let moe_out = ctx.buffers.moe_output();
            ops::residual_add(
                ctx.gpu,
                self.residual_add_k,
                hidden,
                moe_out,
                (num_tokens * h) as u32,
                stream,
            )?;
        } else if self.ffn.fp8_grouped_decode_ok(num_tokens, ctx) {
            // 2026-09-25: Grouped FP8 MoE (`forward_fp8_grouped_decode`): rows are grouped
            // by expert, so each routed and shared expert streams its weights once for all
            // rows routed to it, instead of the per-row loop below.
            k4_diag_checkpoint(ctx, "10a:residual_add_rms_norm", stream)?;
            self.ffn
                .forward_fp8_grouped_decode(normed2_base, num_tokens, ctx, stream)?;
            k4_diag_checkpoint(ctx, "10b:ffn_fp8_grouped_decode", stream)?;
            let moe_out = ctx.buffers.moe_output();
            ops::residual_add(
                ctx.gpu,
                self.residual_add_k,
                hidden,
                moe_out,
                (num_tokens * h) as u32,
                stream,
            )?;
        } else if self.ffn.is_dense() {
            // 2026-09-25: Dense FFN over all rows with `forward_prefill`; `normed2_base`
            // already holds `[num_tokens, h]`.
            k4_diag_checkpoint(ctx, "10a:residual_add_rms_norm", stream)?;
            self.ffn
                .forward_prefill(normed2_base, num_tokens, ctx, stream)?;
            k4_diag_checkpoint(ctx, "10b:ffn_forward_prefill", stream)?;
            let moe_out = ctx.buffers.moe_output();
            ops::residual_add(
                ctx.gpu,
                self.residual_add_k,
                hidden,
                moe_out,
                (num_tokens * h) as u32,
                stream,
            )?;
        } else {
            // 2026-09-25: Every other FFN runs once per row. `hidden` rows are BF16, `h * 2`
            // bytes apart.
            let residual_elem = 2usize;
            for t in 0..(num_tokens as u32) {
                let normed2 = normed2_base.offset(t as usize * h * bf16);
                let moe_out = self.ffn.forward(normed2, ctx, stream)?;
                let hidden_t = hidden.offset(t as usize * h * residual_elem);
                ops::residual_add(
                    ctx.gpu,
                    self.residual_add_k,
                    hidden_t,
                    moe_out,
                    h as u32,
                    stream,
                )?;
            }
        }

        Ok(())
    }
}
