// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Dense-FFN prefill: `forward_prefill`, its dispatch over the installed weights,
//! the packed-Q2 branch, and the NVFP4 projection GEMM `w4a16_prefill_gemm`.
//!
//! Owner: model-layers (dense FFN).
//! Invariants:
//! - On success the `[num_tokens, hidden]` output is in `ctx.buffers.moe_output()`.
//! - The weights are tried in the order packed Q2, FP8, BF16, NVFP4; the first installed
//!   one serves the call.

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::{DenseFfnLayer, DenseFfnWeightsQ2, FfnActivation};
use crate::layer::ForwardContext;
use crate::layers::ops;
use crate::weight_map::{DenseWeight, PackedQ2Weight, QuantizedWeight};

impl DenseFfnLayer {
    /// 2026-09-25: One NVFP4 projection GEMM for `m` rows. With a transposed copy `wt`, m <= 64,
    /// k % 32 == 0 and `ModelLevers::ffn_small_m` (off with `METRALE_FFN_SMALLM=0`), it runs
    /// the deep-K `w4a16_gemm_t_k64` when k >= `w4a16_k64_min_k()` and k % 64 == 0, else
    /// `w4a16_gemm_t`. Otherwise, with `wt`, it runs `w4a16_gemm_t_m128_v2` or
    /// `w4a16_gemm_t_m128`; without `wt` or those handles, the base `w4a16_gemm`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn w4a16_prefill_gemm(
        &self,
        ctx: &ForwardContext,
        w: &QuantizedWeight,
        wt: Option<&QuantizedWeight>,
        input: DevicePtr,
        output: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<()> {
        if let Some(wt) = wt {
            if m <= 64 && k.is_multiple_of(32) && ctx.levers.ffn_small_m {
                if k >= crate::layers::w4a16_k64_min_k()
                    && k.is_multiple_of(64)
                    && self.w4a16_gemm_t_k64_k.0 != 0
                {
                    return ops::w4a16_gemm_n128(
                        ctx.gpu,
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
                        ctx.gpu,
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
                    ctx.gpu,
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
                    ctx.gpu,
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
        ops::w4a16_gemm(ctx.gpu, self.w4a16_gemm, input, w, output, m, n, k, stream)
    }

    /// 2026-09-25: Dense-FFN prefill of `num_tokens` rows of `input`; the output lands in
    /// `moe_output`. Under `ctx.profile` it synchronizes `stream` and logs
    /// `FFN prefill [dense_total] N=<n>: <us>µs`.
    pub fn forward_prefill(
        &self,
        input: DevicePtr,
        num_tokens: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        if !ctx.profile {
            return self.forward_prefill_inner(input, num_tokens, ctx, stream);
        }
        let t0 = std::time::Instant::now();
        let r = self.forward_prefill_inner(input, num_tokens, ctx, stream);
        // 2026-09-25: Synchronize so the time covers the kernels, not just their launch.
        ctx.gpu.synchronize(stream)?;
        tracing::info!(target: "metrale_model_layers::layers::dense_ffn", "  FFN prefill [dense_total] N={}: {}µs",
            num_tokens,
            t0.elapsed().as_micros()
        );
        r
    }

    fn forward_prefill_inner(
        &self,
        input: DevicePtr,
        num_tokens: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.intermediate_size as u32;
        let m = num_tokens as u32;

        let gate_out = ctx.buffers.expert_gate_out();
        let up_out = ctx.buffers.expert_up_out();

        // 2026-09-25: Packed-Q2 prefill, SiLU only: the Q2_0 MMQ GEMM when it can run, else
        // dequantize each projection to BF16 and run the BF16 GEMM.
        if let Some(ref q2w) = self.q2_weights {
            return self.prefill_q2(q2w, input, ctx, m, h, inter, gate_out, up_out, stream);
        }

        // 2026-09-25: FP8. `w8_gemm!` takes, per projection, the first arm that matches, each
        // only when its kernel resolved:
        //   1. m <= 4: `w8a16_gemv_batch4`.
        //   2. m 5..=32 with `m16_tc` on: tensor-core `w8a16_gemm_m16`, two launches above 16
        //      rows. It reorders the K reduction; its tests compare it with `w8a16_gemv` within
        //      `m16_tc::within_m16_tc_budget`, not bit for bit.
        //   3. m 5..=32 with `METRALE_FFN_BATCH16=1`: `w8a16_gemv_batch16`.
        //   4. The W8A8 block-scaled GEMM, when `prefill_w8a8_selected` held.
        //   5. `w8a16_gemm_pipelined`, else the base `w8a16_gemm`.
        // The transposed `w8a16_gemm_t_m128` arm never matches: no transposed FP8 copy is passed.
        // Every `[defaults]` table in `kernels/*/HARDWARE.toml` sets `ffn_m16_tc = false`, so with
        // no variables set, m 5..=32 reaches arm 4 or 5.
        if let Some(ref fp8w) = self.fp8_weights {
            return self.prefill_fp8(fp8w, input, ctx, m, h, inter, gate_out, up_out, stream);
        }

        if let Some(ref bf16w) = self.bf16_weights {
            return self.prefill_bf16(bf16w, input, ctx, m, h, inter, gate_out, up_out, stream);
        }

        self.prefill_nvfp4(
            input, num_tokens, ctx, m, h, inter, gate_out, up_out, stream,
        )
    }

    /// 2026-09-26: The packed-Q2 branch of `forward_prefill_inner`, SiLU only: the Q2_0 MMQ
    /// GEMM when it can run, else a BF16 dequant of each projection and the BF16 GEMM.
    fn prefill_q2(
        &self,
        q2w: &DenseFfnWeightsQ2,
        input: DevicePtr,
        ctx: &ForwardContext,
        m: u32,
        h: u32,
        inter: u32,
        gate_out: DevicePtr,
        up_out: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        if self.activation != FfnActivation::SiLU {
            anyhow::bail!(
                "packed-Q2 FFN prefill supports SiLU only (got {:?})",
                self.activation
            );
        }

        // 2026-09-25: Q2_0 MMQ, when `METRALE_GGUF_NATIVE_Q2_MMQ=1`, `q2_0_mmq_nc_k` and
        // `q4k_quant_act_k` resolved and all three weights use 128-element groups. Gate and
        // up share one q8_1 quantization of `input`; down quantizes the activation.
        let q2_mmq = self.q2_0_mmq_nc_k.0 != 0
            && self.q4k_quant_act_k.0 != 0
            && ops::native_q2_mmq_enabled()
            && q2w.gate_proj.group == 128
            && q2w.up_proj.group == 128
            && q2w.down_proj.group == 128;
        if q2_mmq {
            static Q2MMQ_LOG: std::sync::Once = std::sync::Once::new();
            Q2MMQ_LOG.call_once(|| {
                eprintln!(
                    "[metrale] METRALE_GGUF_NATIVE_Q2_MMQ=1: dense-FFN prefill via native packed Q2_0 MMQ (W2A8, keep-packed)"
                );
            });
            let a_q8 = ctx.buffers.q2_act_q8();
            let mmq = |w: &PackedQ2Weight, out: DevicePtr| -> Result<()> {
                ops::q2_0_mmq_gemm(
                    ctx.gpu,
                    self.q2_0_mmq_nc_k,
                    self.q2_0_mmq_wc_k,
                    a_q8,
                    w.weight,
                    out,
                    m,
                    w.n,
                    w.k,
                    stream,
                )
            };
            ops::quantize_act_q8_1(ctx.gpu, self.q4k_quant_act_k, input, a_q8, m, h, stream)?;
            mmq(&q2w.gate_proj, gate_out)?;
            mmq(&q2w.up_proj, up_out)?;
            ops::silu_mul(
                ctx.gpu,
                self.act_mul,
                gate_out,
                up_out,
                gate_out,
                m * inter,
                stream,
            )?;
            let output = ctx.buffers.moe_output();
            ops::quantize_act_q8_1(
                ctx.gpu,
                self.q4k_quant_act_k,
                gate_out,
                a_q8,
                m,
                inter,
                stream,
            )?;
            mmq(&q2w.down_proj, output)?;
            return Ok(());
        }

        if self.dequant_q2_0_gn_k.0 == 0 {
            anyhow::bail!(
                "dequant_q2_0_gn_to_bf16 kernel missing in this target build — \
                 packed-Q2 (METRALE_GGUF_NATIVE_Q2) prefill is unavailable"
            );
        }
        let tc = self.dense_gemm_tc_k.0 != 0;
        // 2026-09-25: The three projections share the arena's `q2_dequant_scratch`. That is
        // safe because each dequant and its GEMM run on `stream` before the next projection's
        // dequant.
        let scratch = ctx.buffers.q2_dequant_scratch();
        let q2_gemm = |w: &PackedQ2Weight, input: DevicePtr, out: DevicePtr| -> Result<()> {
            let (n, k) = (w.n, w.k);
            debug_assert!(
                (n as usize) * (k as usize) * 2 <= ctx.buffers.q2_dequant_scratch_bytes(),
                "packed-Q2 FFN dequant scratch too small for [{n},{k}] BF16"
            );
            ops::dequant_q2_0_gn_to_bf16(
                ctx.gpu,
                self.dequant_q2_0_gn_k,
                w.weight,
                scratch,
                n,
                k,
                w.group as u32,
                stream,
            )?;
            let dw = DenseWeight { weight: scratch };
            if tc {
                ops::dense_gemm_tc(
                    ctx.gpu,
                    self.dense_gemm_tc_k,
                    input,
                    &dw,
                    out,
                    m,
                    n,
                    k,
                    stream,
                )?;
            } else {
                ops::dense_gemm(
                    ctx.gpu,
                    self.dense_gemm_bf16_k,
                    input,
                    &dw,
                    out,
                    m,
                    n,
                    k,
                    stream,
                )?;
            }
            Ok(())
        };
        q2_gemm(&q2w.gate_proj, input, gate_out)?;
        q2_gemm(&q2w.up_proj, input, up_out)?;
        ops::silu_mul(
            ctx.gpu,
            self.act_mul,
            gate_out,
            up_out,
            gate_out,
            m * inter,
            stream,
        )?;
        let output = ctx.buffers.moe_output();
        q2_gemm(&q2w.down_proj, gate_out, output)?;
        Ok(())
    }
}
