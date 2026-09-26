// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Single-token dense-FFN decode, `DenseFfnLayer::forward`.
//!
//! Owner: model-layers (dense FFN).
//! Invariants:
//! - On success the `[1, hidden]` output is in `ctx.buffers.moe_output()`, and `forward`
//!   returns that pointer.

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::{DenseFfnLayer, FfnActivation, fp8_down};
use crate::layer::ForwardContext;
use crate::layers::ops;
use crate::weight_map::QuantizedWeight;

impl DenseFfnLayer {
    /// 2026-09-25: Decode for one token; returns `moe_output` holding `[1, hidden]`. Packed-Q2, FP8
    /// and BF16 layers take their own branches. NVFP4 runs the fused gate+up GEMV, then either
    /// `act_mul` plus the down GEMV (split SiLU: the default, and always with a LoRA adapter or
    /// GELU) or the SiLU-fused down GEMV. `METRALE_DECODE_FFN_VIA_GEMM=1` runs NVFP4 through
    /// `w4a16_prefill_gemm` instead.
    pub fn forward(
        &self,
        input: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.intermediate_size as u32;

        let gate_out = ctx.buffers.expert_gate_out();
        let up_out = ctx.buffers.expert_up_out();

        // 2026-09-25: Packed-Q2: gate, up, `act_mul`, down (four launches); SiLU only.
        if let Some(ref q2w) = self.q2_weights {
            if self.q2_0_gemv_k.0 == 0 {
                anyhow::bail!(
                    "q2_0_gemv kernel missing in this target build — packed-Q2 decode \
                     (METRALE_GGUF_NATIVE_Q2) is unavailable"
                );
            }
            if self.activation != FfnActivation::SiLU {
                anyhow::bail!(
                    "packed-Q2 FFN decode supports SiLU only (got {:?})",
                    self.activation
                );
            }
            let output = ctx.buffers.moe_output();
            ops::q2_0_gemv_vec(
                ctx.gpu,
                self.q2_0_gemv_k,
                input,
                &q2w.gate_proj,
                gate_out,
                stream,
            )?;
            ops::q2_0_gemv_vec(
                ctx.gpu,
                self.q2_0_gemv_k,
                input,
                &q2w.up_proj,
                up_out,
                stream,
            )?;
            ops::silu_mul(
                ctx.gpu,
                self.act_mul,
                gate_out,
                up_out,
                gate_out,
                inter,
                stream,
            )?;
            ops::q2_0_gemv_vec(
                ctx.gpu,
                self.q2_0_gemv_k,
                gate_out,
                &q2w.down_proj,
                output,
                stream,
            )?;
            return Ok(output);
        }

        // 2026-09-25: FP8: `fp8_down::fp8_down_arm` picks the arm. The default `SplitSilu` stages
        // `silu(gate)*up` once with `act_mul` before a plain `w8a16_gemv`, because the fused
        // `w8a16_gemv_silu_input` recomputes it for every output. Numerics and the measurement
        // are on `Fp8DownArm`.
        if let Some(ref fp8w) = self.fp8_weights {
            let output = ctx.buffers.moe_output();
            let arm = fp8_down::fp8_down_arm(
                self.activation == FfnActivation::SiLU,
                self.w8a16_gemv_dual_k.0 != 0,
                self.w8a16_gemv_silu_input_k.0 != 0,
                self.act_mul.0 != 0,
                self.w8a16_gemv_k.0 != 0,
                ctx.levers.decode_split_silu,
            );
            if arm != fp8_down::Fp8DownArm::PerProjection {
                ops::w8a16_gemv_dual(
                    ctx.gpu,
                    self.w8a16_gemv_dual_k,
                    input,
                    fp8w.gate_proj.weight,
                    fp8w.gate_proj.row_scale,
                    gate_out,
                    fp8w.up_proj.weight,
                    fp8w.up_proj.row_scale,
                    up_out,
                    inter,
                    h,
                    stream,
                )?;
                if arm == fp8_down::Fp8DownArm::SplitSilu {
                    ops::silu_mul(
                        ctx.gpu,
                        self.act_mul,
                        gate_out,
                        up_out,
                        gate_out,
                        inter,
                        stream,
                    )?;
                    ops::w8a16_gemv(
                        ctx.gpu,
                        self.w8a16_gemv_k,
                        gate_out,
                        fp8w.down_proj.weight,
                        fp8w.down_proj.row_scale,
                        output,
                        h,
                        inter,
                        stream,
                    )?;
                } else {
                    ops::w8a16_gemv_silu_input(
                        ctx.gpu,
                        self.w8a16_gemv_silu_input_k,
                        gate_out,
                        up_out,
                        fp8w.down_proj.weight,
                        fp8w.down_proj.row_scale,
                        output,
                        h,
                        inter,
                        stream,
                    )?;
                }
                return Ok(output);
            }
            ops::w8a16_gemv(
                ctx.gpu,
                self.w8a16_gemv_k,
                input,
                fp8w.gate_proj.weight,
                fp8w.gate_proj.row_scale,
                gate_out,
                inter,
                h,
                stream,
            )?;
            ops::w8a16_gemv(
                ctx.gpu,
                self.w8a16_gemv_k,
                input,
                fp8w.up_proj.weight,
                fp8w.up_proj.row_scale,
                up_out,
                inter,
                h,
                stream,
            )?;
            ops::silu_mul(
                ctx.gpu,
                self.act_mul,
                gate_out,
                up_out,
                gate_out,
                inter,
                stream,
            )?;
            ops::w8a16_gemv(
                ctx.gpu,
                self.w8a16_gemv_k,
                gate_out,
                fp8w.down_proj.weight,
                fp8w.down_proj.row_scale,
                output,
                h,
                inter,
                stream,
            )?;
            return Ok(output);
        }

        if let Some(ref bf16w) = self.bf16_weights {
            ops::dense_gemv(
                ctx.gpu,
                self.dense_gemv_bf16_k,
                input,
                &bf16w.gate_proj,
                gate_out,
                inter,
                h,
                stream,
            )?;
            ops::dense_gemv(
                ctx.gpu,
                self.dense_gemv_bf16_k,
                input,
                &bf16w.up_proj,
                up_out,
                inter,
                h,
                stream,
            )?;
            ops::silu_mul(
                ctx.gpu,
                self.act_mul,
                gate_out,
                up_out,
                gate_out,
                inter,
                stream,
            )?;
            let output = ctx.buffers.moe_output();
            ops::dense_gemv(
                ctx.gpu,
                self.dense_gemv_bf16_k,
                gate_out,
                &bf16w.down_proj,
                output,
                h,
                inter,
                stream,
            )?;
            return Ok(output);
        }

        // 2026-09-25: `METRALE_DECODE_FFN_VIA_GEMM=1` (SiLU layers): run the three NVFP4
        // projections through `w4a16_prefill_gemm` at m=1 instead of the decode GEMVs. It needs the
        // gate and up `_t` copies, which the `finalize_*_load` functions may have freed; without
        // them it logs a warning and uses the GEMVs. This branch applies no LoRA delta.
        if ctx.levers.decode_ffn_via_gemm
            && self.activation == FfnActivation::SiLU
            && self.act_mul.0 != 0
        {
            let wt_alive =
                |w: &Option<QuantizedWeight>| w.as_ref().is_some_and(|w| !w.weight.is_null());
            if wt_alive(&self.weights.gate_proj_t) && wt_alive(&self.weights.up_proj_t) {
                if ctx.stats.once("log:decode_ffn_via_gemm") {
                    tracing::info!(target: "metrale_model_layers::layers::dense_ffn", "decode FFN via verify GEMM path (METRALE_DECODE_FFN_VIA_GEMM=1): \
                         gate/up/down through w4a16_prefill_gemm at M=1"
                    );
                }
                self.w4a16_prefill_gemm(
                    ctx,
                    &self.weights.gate_proj,
                    self.weights.gate_proj_t.as_ref(),
                    input,
                    gate_out,
                    1,
                    inter,
                    h,
                    stream,
                )?;
                self.w4a16_prefill_gemm(
                    ctx,
                    &self.weights.up_proj,
                    self.weights.up_proj_t.as_ref(),
                    input,
                    up_out,
                    1,
                    inter,
                    h,
                    stream,
                )?;
                ops::silu_mul(
                    ctx.gpu,
                    self.act_mul,
                    gate_out,
                    up_out,
                    gate_out,
                    inter,
                    stream,
                )?;
                let output = ctx.buffers.moe_output();
                self.w4a16_prefill_gemm(
                    ctx,
                    &self.weights.down_proj,
                    self.weights.down_proj_t.as_ref(),
                    gate_out,
                    output,
                    1,
                    h,
                    inter,
                    stream,
                )?;
                return Ok(output);
            }
            if ctx.stats.once("log:decode_ffn_no_twins") {
                tracing::warn!(target: "metrale_model_layers::layers::dense_ffn", "METRALE_DECODE_FFN_VIA_GEMM=1 requested but transposed FFN copies \
                     are freed/absent (NVFP4-MMQ prefill arm?) — falling back to GEMV; \
                     the unification experiment is NOT active"
                );
            }
        }

        // 2026-09-25: Fused gate+up GEMV. The single-warp choice is made separately for the dual
        // and the silu-input kernels, so a missing `w4a16_gemv_silu_input_sw` does not disable
        // `w4a16_gemv_dual_sw`.
        let use_dual_sw = ops::use_gemv_sw(ctx.levers.gemv_sw, self.w4a16_gemv_dual_sw);
        let use_silu_sw = ops::use_gemv_sw(ctx.levers.gemv_sw, self.w4a16_gemv_silu_input_sw);
        if use_dual_sw {
            ops::w4a16_gemv_dual_sw(
                ctx.gpu,
                self.w4a16_gemv_dual_sw,
                input,
                &self.weights.gate_proj,
                gate_out,
                &self.weights.up_proj,
                up_out,
                inter,
                h,
                stream,
            )?;
        } else {
            ops::w4a16_gemv_dual(
                ctx.gpu,
                self.w4a16_gemv_dual,
                input,
                &self.weights.gate_proj,
                gate_out,
                &self.weights.up_proj,
                up_out,
                inter,
                h,
                stream,
            )?;
        }

        let output = ctx.buffers.moe_output();
        // 2026-09-25: Split SiLU (default; off with `METRALE_NO_DECODE_SPLIT_SILU`): stage
        // `silu(gate)*up` once with `act_mul`, then run the down GEMV. The fused
        // `w4a16_gemv_silu_input` recomputes the activation for every output row. An installed LoRA
        // adapter forces this path, because the down delta needs the materialised activation;
        // `set_lora_weights` refuses an adapter on a layer that cannot take it.
        let split_silu = self.activation == FfnActivation::SiLU
            && self.act_mul.0 != 0
            && self.w4a16_gemv.0 != 0
            && (ctx.levers.decode_split_silu || self.lora.is_some());
        if split_silu {
            self.apply_lora_gate_up(ctx, input, gate_out, up_out, 1, stream)?;
            ops::silu_mul(
                ctx.gpu,
                self.act_mul,
                gate_out,
                up_out,
                gate_out,
                inter,
                stream,
            )?;
            ops::w4a16_decode_gemv(
                ctx.gpu,
                self.w4a16_gemv,
                self.w4a16_gemv_sw,
                ctx.levers.gemv_sw,
                gate_out,
                &self.weights.down_proj,
                output,
                h,
                inter,
                stream,
            )?;
            self.apply_lora_down(ctx, gate_out, output, 1, stream)?;
            return Ok(output);
        }
        debug_assert!(
            self.lora.is_none(),
            "LoRA installed but decode took the fused silu_input path, which \
             never materialises the activation the down delta contracts over; \
             set_lora_weights is supposed to make this unreachable"
        );
        match self.activation {
            FfnActivation::SiLU => {
                if use_silu_sw {
                    ops::w4a16_gemv_silu_input_sw(
                        ctx.gpu,
                        self.w4a16_gemv_silu_input_sw,
                        gate_out,
                        up_out,
                        &self.weights.down_proj,
                        output,
                        h,
                        inter,
                        stream,
                    )?;
                } else {
                    ops::w4a16_gemv_silu_input(
                        ctx.gpu,
                        self.w4a16_gemv_silu_input,
                        gate_out,
                        up_out,
                        &self.weights.down_proj,
                        output,
                        h,
                        inter,
                        stream,
                    )?;
                }
            }
            FfnActivation::GeLU => {
                ops::silu_mul(
                    ctx.gpu,
                    self.act_mul,
                    gate_out,
                    up_out,
                    gate_out,
                    inter,
                    stream,
                )?;
                ops::w4a16_decode_gemv(
                    ctx.gpu,
                    self.w4a16_gemv,
                    self.w4a16_gemv_sw,
                    ctx.levers.gemv_sw,
                    gate_out,
                    &self.weights.down_proj,
                    output,
                    h,
                    inter,
                    stream,
                )?;
            }
        }

        Ok(output)
    }
}
