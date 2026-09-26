// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The NVFP4 branch of `DenseFfnLayer::forward_prefill_inner`: the `w4_gemm!`
//! ladder for gate, up and down, the activation quantizations each enabled arm needs,
//! the SiLU·mul, and the LoRA deltas.
//!
//! Owner: model-layers (dense FFN).
//! Invariants:
//! - On success the `[m, hidden]` output is in `ctx.buffers.moe_output()`.
//! - Under the NVFP4 MMQ arm, each projection's `weight_scale_2` is applied after its GEMM.

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::nvfp4_plan::Nvfp4PrefillPlan;
use super::{DenseFfnLayer, mmq_small_tile_enabled, mmq_tile64_enabled};
use crate::layer::ForwardContext;
use crate::layers::ops;

impl DenseFfnLayer {
    /// 2026-09-26: The NVFP4 branch of `forward_prefill_inner`.
    pub(super) fn prefill_nvfp4(
        &self,
        input: DevicePtr,
        num_tokens: usize,
        ctx: &ForwardContext,
        m: u32,
        h: u32,
        inter: u32,
        gate_out: DevicePtr,
        up_out: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        // 2026-09-25: NVFP4 prefill. `w4_gemm!` takes the first arm that matches: NVFP4 MMQ, Q4_K
        // MMQ, W4A4 (`METRALE_FP4_PREFILL`), int8 (`METRALE_INT8_PREFILL`, or down under the Q4_K
        // arm); then, with a transposed copy: `w4a16_gemm_t` (`METRALE_FP8_M64_PREFILL`), the BF16
        // tensor-core kernels (`METRALE_BF16_TC_PREFILL`), `w4a16_prefill_gemm` for m <= 64,
        // `w4a16_gemm_t_m128_v2`, `w4a16_gemm_t_m128`; and the base `w4a16_gemm` last.
        let Nvfp4PrefillPlan {
            fp8_m64_prefill,
            int8_prefill,
            fp4mmq_prefill,
            fp4mmq_down,
            down_faith2,
            int8_a_i8,
            int8_a_scale,
            fp4_prefill,
            nvfp4_a_packed,
            nvfp4_a_scale,
            q4k_prefill,
            q4k_a,
            fp4_y,
            use_v2,
            bf16_kernel,
            bf16_tc_prefill,
        } = self.nvfp4_prefill_plan(ctx);

        macro_rules! w4_gemm {
            ($w:expr, $wt:expr, $cell:expr, $qcell:expr, $fp4cell:expr, $allow_fp4:expr, $in:expr, $out:expr, $n:expr, $k:expr, $allow_q4k:expr) => {
                match $wt {
                    // 2026-09-25: NVFP4 MMQ. `$allow_fp4` is `fp4mmq_prefill` for gate/up and
                    // `fp4mmq_down` for down. The caller has quantized the activation into `fp4_y`
                    // and applies `weight_scale_2` to the output afterwards.
                    _ if $allow_fp4 => {
                        let _ = $in;
                        let qw =
                            self.ensure_nvfp4_mmq_weight($fp4cell, ctx.gpu, $w, $n, $k, stream)?;
                        // 2026-09-25: Size the M tile to the batch: `grid.y = ceil(m / tile)`, and
                        // every extra M tile streams the whole weight again.
                        let (tk_nc, tk_wc, tile) = if m <= 16
                            && self.nvfp4_mmq16_nc_k.0 != 0
                            && mmq_small_tile_enabled()
                        {
                            (self.nvfp4_mmq16_nc_k, self.nvfp4_mmq16_wc_k, 16u32)
                        } else if m <= 32
                            && self.nvfp4_mmq32_nc_k.0 != 0
                            && mmq_small_tile_enabled()
                        {
                            (self.nvfp4_mmq32_nc_k, self.nvfp4_mmq32_wc_k, 32u32)
                        } else if m <= 64
                            && self.nvfp4_mmq64_nc_k.0 != 0
                            && mmq_small_tile_enabled()
                            && mmq_tile64_enabled()
                        {
                            (self.nvfp4_mmq64_nc_k, self.nvfp4_mmq64_wc_k, 64u32)
                        } else {
                            (self.nvfp4_mmq_nc_k, self.nvfp4_mmq_wc_k, 128u32)
                        };
                        ops::nvfp4_mmq_gemm_tiled(
                            ctx.gpu, tk_nc, tk_wc, tile, fp4_y, qw.w, $out, m, $n, $k, stream,
                        )?;
                    }
                    // 2026-09-25: Q4_K MMQ, gate and up only (`$allow_q4k` is false for down). The
                    // caller has quantized the activation into `q4k_a`.
                    _ if q4k_prefill && $allow_q4k => {
                        let qw = self.ensure_q4k_weight($qcell, ctx.gpu, $w, $n, $k, stream)?;
                        ops::q4k_mmq_gemm(
                            ctx.gpu,
                            self.q4k_mmq_nc_k,
                            self.q4k_mmq_wc_k,
                            q4k_a,
                            qw.w_q4k,
                            $out,
                            m,
                            $n,
                            $k,
                            stream,
                        )?;
                    }
                    // 2026-09-25: W4A4: the caller has quantized the activation into
                    // `nvfp4_a_packed` / `nvfp4_a_scale`.
                    _ if fp4_prefill => {
                        let _ = $in;
                        ops::w4a4_gemm(
                            ctx.gpu,
                            self.w4a4_gemm_k,
                            nvfp4_a_packed,
                            nvfp4_a_scale,
                            $w,
                            $out,
                            m,
                            $n,
                            $k,
                            stream,
                        )?;
                    }
                    // 2026-09-25: int8 (`METRALE_INT8_PREFILL`, or down under the Q4_K arm). It
                    // reads the non-transposed NVFP4 weight, so it does not need `$wt`.
                    _ if int8_prefill || (down_faith2 && !$allow_q4k) => {
                        let iw = self.ensure_int8_weight($cell, ctx.gpu, $w, $n, $k, stream)?;
                        // 2026-09-25: `METRALE_INT8_FAITH5` swaps in `int8_gemm_i32acc`, launched
                        // exactly like `int8_gemm_faith2`.
                        let int8_kernel = if self.int8_faith5_k.0 != 0 && ctx.levers.int8_faith5 {
                            self.int8_faith5_k
                        } else {
                            self.int8_faith2_k
                        };
                        ops::int8_gemm_faith2_prefill(
                            ctx.gpu,
                            int8_kernel,
                            self.requant_a_int8_k,
                            $in,
                            iw.w_i8,
                            iw.w_scale,
                            int8_a_i8,
                            int8_a_scale,
                            $out,
                            m,
                            $n,
                            $k,
                            stream,
                        )?;
                    }
                    Some(wt) if fp8_m64_prefill => ops::w4a16_gemm_n128(
                        ctx.gpu,
                        self.w4a16_gemm_t_k,
                        $in,
                        &wt,
                        $out,
                        m,
                        $n,
                        $k,
                        stream,
                    )?,
                    // 2026-09-25: The v2 kernel takes a ninth parameter, `ldb` (the transposed row
                    // stride, `N` for these weights), so it needs the `_ldb` launcher; v1 takes
                    // eight.
                    Some(wt) if bf16_tc_prefill && use_v2 => ops::w4a16_gemm_n128_m128_bf16_ldb(
                        ctx.gpu,
                        bf16_kernel,
                        $in,
                        &wt,
                        $out,
                        m,
                        $n,
                        $k,
                        $n,
                        stream,
                    )?,
                    Some(wt) if bf16_tc_prefill => ops::w4a16_gemm_n128_m128_bf16(
                        ctx.gpu,
                        bf16_kernel,
                        $in,
                        &wt,
                        $out,
                        m,
                        $n,
                        $k,
                        stream,
                    )?,
                    // 2026-09-25: m <= 64: `w4a16_prefill_gemm` picks the small-M kernels (unless
                    // `METRALE_FFN_SMALLM=0`) and otherwise the same v2/m128 kernels as below.
                    Some(wt) if m <= 64 => {
                        self.w4a16_prefill_gemm(ctx, $w, Some(&wt), $in, $out, m, $n, $k, stream)?
                    }
                    Some(wt) if self.w4a16_gemm_t_m128_v2_k.0 != 0 => ops::w4a16_gemm_n128_m128_v2(
                        ctx.gpu,
                        self.w4a16_gemm_t_m128_v2_k,
                        $in,
                        &wt,
                        $out,
                        m,
                        $n,
                        $k,
                        stream,
                    )?,
                    Some(wt) if self.w4a16_gemm_t_m128_k.0 != 0 => ops::w4a16_gemm_n128_m128(
                        ctx.gpu,
                        self.w4a16_gemm_t_m128_k,
                        $in,
                        &wt,
                        $out,
                        m,
                        $n,
                        $k,
                        stream,
                    )?,
                    _ => {
                        ops::w4a16_gemm(ctx.gpu, self.w4a16_gemm, $in, $w, $out, m, $n, $k, stream)?
                    }
                }
            };
        }

        // 2026-09-25: Quantize the shared gate/up input once for each enabled arm: W4A4 (NVFP4),
        // Q4_K (q8_1) and NVFP4 MMQ.
        if fp4_prefill {
            ops::quantize_bf16_to_nvfp4(
                ctx.gpu,
                self.quantize_nvfp4_k,
                input,
                nvfp4_a_packed,
                nvfp4_a_scale,
                m,
                h,
                stream,
            )?;
        }
        if q4k_prefill {
            ops::quantize_act_q8_1(ctx.gpu, self.q4k_quant_act_k, input, q4k_a, m, h, stream)?;
        }
        if fp4mmq_prefill {
            ops::nvfp4_mmq_quantize_act(
                ctx.gpu,
                self.nvfp4_quant_act_k,
                input,
                fp4_y,
                m,
                h,
                stream,
            )?;
        }
        // 2026-09-25: Under `ctx.profile`, log each projection's time and restart the clock, so the
        // three figures do not overlap.
        macro_rules! ffn_step {
            ($label:expr, $t0:expr) => {
                if ctx.profile {
                    ctx.gpu.synchronize(stream)?;
                    tracing::info!(target: "metrale_model_layers::layers::dense_ffn", "  FFN prefill [{}] N={}: {}µs",
                        $label,
                        num_tokens,
                        $t0.elapsed().as_micros()
                    );
                    #[allow(unused_assignments)]
                    {
                        $t0 = std::time::Instant::now();
                    }
                }
            };
        }
        #[allow(unused_mut, unused_assignments)]
        let mut t_ffn = std::time::Instant::now();
        w4_gemm!(
            &self.weights.gate_proj,
            self.weights.gate_proj_t,
            &self.int8_gate,
            &self.q4k_gate,
            &self.fp4mmq_gate,
            fp4mmq_prefill,
            input,
            gate_out,
            inter,
            h,
            true
        );
        ffn_step!("gate_proj", t_ffn);
        w4_gemm!(
            &self.weights.up_proj,
            self.weights.up_proj_t,
            &self.int8_up,
            &self.q4k_up,
            &self.fp4mmq_up,
            fp4mmq_prefill,
            input,
            up_out,
            inter,
            h,
            true
        );
        ffn_step!("up_proj", t_ffn);

        // 2026-09-25: With an adapter installed the NVFP4 MMQ arm is off, so `gate_out`/`up_out`
        // hold fully scaled BF16 here and the delta adds in the same units.
        self.apply_lora_gate_up(ctx, input, gate_out, up_out, m, stream)?;
        let fused_down_quant = fp4mmq_down && self.nvfp4_silu_quant_k.0 != 0;
        if fused_down_quant {
            // 2026-09-25: SiLU·mul, gate's and up's `weight_scale_2`, and the down MMQ quantization
            // in one kernel, writing `fp4_y`; `gate_out` is not written.
            ops::nvfp4_silu_mul_quant(
                ctx.gpu,
                self.nvfp4_silu_quant_k,
                gate_out,
                up_out,
                fp4_y,
                self.weights.gate_proj.weight_scale_2,
                self.weights.up_proj.weight_scale_2,
                m,
                inter,
                stream,
            )?;
        } else if fp4mmq_prefill {
            // 2026-09-25: Apply gate's and up's `weight_scale_2` inside the SiLU·mul, before the
            // nonlinearity.
            ops::nvfp4_silu_mul_scaled(
                ctx.gpu,
                self.nvfp4_silu_scaled_k,
                gate_out,
                up_out,
                gate_out,
                self.weights.gate_proj.weight_scale_2,
                self.weights.up_proj.weight_scale_2,
                m * inter,
                stream,
            )?;
        } else {
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

        // 2026-09-25: Quantize the down input for each enabled arm: W4A4; Q4_K only when down does
        // not run int8; NVFP4 MMQ only when the fused kernel above did not already.
        if fp4_prefill {
            ops::quantize_bf16_to_nvfp4(
                ctx.gpu,
                self.quantize_nvfp4_k,
                gate_out,
                nvfp4_a_packed,
                nvfp4_a_scale,
                m,
                inter,
                stream,
            )?;
        }
        if q4k_prefill && !down_faith2 {
            ops::quantize_act_q8_1(
                ctx.gpu,
                self.q4k_quant_act_k,
                gate_out,
                q4k_a,
                m,
                inter,
                stream,
            )?;
        }
        if fp4mmq_down && !fused_down_quant {
            ops::nvfp4_mmq_quantize_act(
                ctx.gpu,
                self.nvfp4_quant_act_k,
                gate_out,
                fp4_y,
                m,
                inter,
                stream,
            )?;
        }
        let output = ctx.buffers.moe_output();
        w4_gemm!(
            &self.weights.down_proj,
            self.weights.down_proj_t,
            &self.int8_down,
            &self.q4k_down,
            &self.fp4mmq_down,
            fp4mmq_down,
            gate_out,
            output,
            h,
            inter,
            false
        );
        ffn_step!("down_proj", t_ffn);
        // 2026-09-25: Apply down's `weight_scale_2` to the NVFP4 MMQ output.
        if fp4mmq_down {
            ops::nvfp4_scale_bf16(
                ctx.gpu,
                self.nvfp4_scale_k,
                output,
                self.weights.down_proj.weight_scale_2,
                m * h,
                stream,
            )?;
        }
        // 2026-09-25: After the scale fold, which belongs to the base projection's output only.
        // `gate_out` holds `silu(gate)*up`, the activation the down GEMM read: with an adapter
        // installed the fused SiLU·mul + quantize arm is off.
        self.apply_lora_down(ctx, gate_out, output, m, stream)?;

        Ok(())
    }
}
