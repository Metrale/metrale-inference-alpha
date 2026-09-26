// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The native-FP8 branch of `DenseFfnLayer::forward_prefill_inner`: the `w8_gemm!`
//! ladder for gate, up and down, and the fused gate+up W8A8 GEMM when it is selected.
//!
//! Owner: model-layers (dense FFN).
//! Invariants:
//! - On success the `[m, hidden]` output is in `ctx.buffers.moe_output()`.

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::{DenseFfnLayer, DenseFfnWeightsFp8};
use crate::layer::ForwardContext;
use crate::layers::ops;
use crate::weight_map::Fp8WeightTransposed;

impl DenseFfnLayer {
    /// 2026-09-26: The FP8 branch of `forward_prefill_inner`.
    pub(super) fn prefill_fp8(
        &self,
        fp8w: &DenseFfnWeightsFp8,
        input: DevicePtr,
        ctx: &ForwardContext,
        m: u32,
        h: u32,
        inter: u32,
        gate_out: DevicePtr,
        up_out: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let batch16 = self.ffn_batch16_plan(m);
        // 2026-09-25: Per projection depth: gate/up reduce over `h` and down over `inter`, and
        // the tier declines a K that is not a multiple of 128.
        let m16_tc = |k: u32| self.ffn_m16_tc_plan(m, k);
        macro_rules! w8_gemm {
            ($w:expr, $wt:expr, $in:expr, $out:expr, $n:expr, $k:expr, $a8:expr, $cap:expr) => {
                match $wt {
                    _ if (1..=4).contains(&m) && self.w8a16_gemv_batch4_k.0 != 0 => {
                        ops::w8a16_gemv_batch4(
                            ctx.gpu,
                            self.w8a16_gemv_batch4_k,
                            $in,
                            $w.weight,
                            $w.row_scale,
                            $out,
                            m,
                            $n,
                            $k,
                            stream,
                        )?
                    }
                    _ if m16_tc($k).is_some() => self.w8a16_m16_tc_proj(
                        ctx,
                        m16_tc($k).expect("guarded by is_some"),
                        &$w,
                        $in,
                        $out,
                        m,
                        $n,
                        $k,
                        stream,
                    )?,
                    _ if batch16.is_some() => self.w8a16_batch16_proj(
                        ctx,
                        batch16.expect("guarded by is_some"),
                        &$w,
                        $in,
                        $out,
                        m,
                        $n,
                        $k,
                        stream,
                    )?,
                    // 2026-09-25: `$a8` is `Some` exactly when `prefill_w8a8_selected` held for
                    // this projection.
                    _ if $a8.is_some() => {
                        let (a_fp8, a_scale) = $a8.expect("guarded by is_some");
                        self.w8a8_gemm(ctx, a_fp8, a_scale, &$w, $out, $cap, m, $n, $k, stream)?
                    }
                    Some(wt) if self.w8a16_gemm_t_m128_k.0 != 0 => {
                        let wt: Fp8WeightTransposed = wt;
                        ops::w8a16_gemm_n128_m128(
                            ctx.gpu,
                            self.w8a16_gemm_t_m128_k,
                            $in,
                            wt.weight_t,
                            wt.scale_t,
                            $out,
                            m,
                            $n,
                            $k,
                            stream,
                        )?
                    }
                    _ if self.w8a16_gemm_pipelined_k.0 != 0 => ops::w8a16_gemm_pipelined(
                        ctx.gpu,
                        self.w8a16_gemm_pipelined_k,
                        $in,
                        $w.weight,
                        $w.row_scale,
                        $out,
                        m,
                        $n,
                        $k,
                        stream,
                    )?,
                    _ => ops::w8a16_gemm(
                        ctx.gpu,
                        self.w8a16_gemm_k,
                        $in,
                        $w.weight,
                        $w.row_scale,
                        $out,
                        m,
                        $n,
                        $k,
                        stream,
                    )?,
                }
            };
        }
        let gate_t: Option<Fp8WeightTransposed> = None;
        let up_t: Option<Fp8WeightTransposed> = None;
        let down_t: Option<Fp8WeightTransposed> = None;
        // 2026-09-25: Quantize the activation for W8A8 once per input: gate and up share
        // `input`, and down quantizes the activation later. Nothing is quantized when the
        // batch16 tier, or the tensor-core tier at either reduction depth, claims `m`.
        let w8a8_reachable = batch16.is_none() && m16_tc(h).is_none() && m16_tc(inter).is_none();
        let gate_up_w8a8 = w8a8_reachable
            && self.prefill_w8a8_selected(ctx, m, inter, h, &fp8w.gate_proj)
            && self.prefill_w8a8_selected(ctx, m, inter, h, &fp8w.up_proj);
        let down_w8a8 =
            w8a8_reachable && self.prefill_w8a8_selected(ctx, m, h, inter, &fp8w.down_proj);
        if !gate_up_w8a8 && !down_w8a8 && w8a8_reachable {
            self.log_w8a16_prefill_route(ctx);
        }
        let gu_a8 = if gate_up_w8a8 {
            Some(self.w8a8_quant_act(ctx, input, m, h, stream)?)
        } else {
            None
        };
        let gu_cap = ctx.buffers.expert_gate_out_bytes();
        // 2026-09-25: Fused gate+up: one W8A8 GEMM over the `[2*inter, hidden]` weight, then
        // the strided SiLU·mul into `gate_out`. It requires `gate_up_w8a8`, so it replaces only
        // the W8A8 arm of the pair below. Rule: `gateup_fused::gateup_fused_selected`.
        if let Some((a_fp8, a_scale)) = gu_a8
            && let Some(fused) = self.gateup_fused_plan(ctx, m, inter, gate_up_w8a8)
        {
            self.w8a8_gate_up_fused(ctx, a_fp8, a_scale, fused, gate_out, m, inter, h, stream)?;
        } else {
            w8_gemm!(
                fp8w.gate_proj,
                gate_t,
                input,
                gate_out,
                inter,
                h,
                gu_a8,
                gu_cap
            );
            w8_gemm!(fp8w.up_proj, up_t, input, up_out, inter, h, gu_a8, gu_cap);
            ops::silu_mul(
                ctx.gpu,
                self.act_mul,
                gate_out,
                up_out,
                gate_out,
                m * inter,
                stream,
            )?;
        }
        let output = ctx.buffers.moe_output();
        // 2026-09-25: `gate_out` holds the activation written above. The down quantization
        // reuses the gate/up quantization's scratch, which is safe because every launch is on
        // `stream`.
        let down_a8 = if down_w8a8 {
            Some(self.w8a8_quant_act(ctx, gate_out, m, inter, stream)?)
        } else {
            None
        };
        let down_cap = ctx.buffers.moe_output_bytes();
        w8_gemm!(
            fp8w.down_proj,
            down_t,
            gate_out,
            output,
            h,
            inter,
            down_a8,
            down_cap
        );
        Ok(())
    }
}
