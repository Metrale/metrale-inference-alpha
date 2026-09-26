// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The per-call arm choices of the NVFP4 prefill branch: which of the NVFP4 MMQ,
//! Q4_K MMQ, W4A4, int8 and BF16 tensor-core arms are on, and the arena buffers their
//! activations use. Each enabled arm logs once per model.
//!
//! Owner: model-layers (dense FFN).
//! Invariants:
//! - The Q4_K arm and the int8 down arm are off whenever the NVFP4 MMQ arm is on.
//! - A buffer field is `DevicePtr::NULL` when its arm is off.

use metrale_gpu_runtime::gpu::{DevicePtr, KernelHandle};

use super::{DenseFfnLayer, FfnActivation};
use crate::layer::ForwardContext;

/// 2026-09-26: The values `nvfp4_prefill_plan` computes for `prefill_nvfp4`, one field per local
/// of the same name there.
#[derive(Debug, Clone, Copy)]
pub(super) struct Nvfp4PrefillPlan {
    pub(super) fp8_m64_prefill: bool,
    pub(super) int8_prefill: bool,
    pub(super) fp4mmq_prefill: bool,
    pub(super) fp4mmq_down: bool,
    pub(super) down_faith2: bool,
    pub(super) int8_a_i8: DevicePtr,
    pub(super) int8_a_scale: DevicePtr,
    pub(super) fp4_prefill: bool,
    pub(super) nvfp4_a_packed: DevicePtr,
    pub(super) nvfp4_a_scale: DevicePtr,
    pub(super) q4k_prefill: bool,
    pub(super) q4k_a: DevicePtr,
    pub(super) fp4_y: DevicePtr,
    pub(super) use_v2: bool,
    pub(super) bf16_kernel: KernelHandle,
    pub(super) bf16_tc_prefill: bool,
}

impl DenseFfnLayer {
    /// 2026-09-26: Resolves the NVFP4 prefill arms from `ctx.levers` and the resolved handles.
    pub(super) fn nvfp4_prefill_plan(&self, ctx: &ForwardContext) -> Nvfp4PrefillPlan {
        let bf16_tc_env = ctx.levers.bf16_tc_prefill;
        let fp8_m64_prefill = self.w4a16_gemm_t_k.0 != 0 && ctx.levers.fp8_m64_prefill;
        let int8_prefill = self.int8_faith2_k.0 != 0 && ctx.levers.int8_prefill;
        if int8_prefill {
            // 2026-09-25: Latched per model (`ModelStats::once`).
            if ctx.stats.once("log:ffn_int8_prefill") {
                tracing::info!(target: "metrale_model_layers::layers::dense_ffn", "[metrale] METRALE_INT8_PREFILL=1: dense-FFN prefill via int8_gemm_faith2 (W4A8 requant→int8 MMA, lossy ~0.99998 cosine)"
                );
            }
        }
        // 2026-09-25: A LoRA adapter turns the NVFP4 MMQ arm off: its gate/up outputs lack
        // `weight_scale_2` until the scaled SiLU·mul applies it, so a delta added to them would be
        // scaled too. The Q4_K arm is off while this arm is on; both use `ffn_act_q8`.
        let fp4mmq_prefill = self.nvfp4_mmq_nc_k.0 != 0
            && self.nvfp4_quant_act_k.0 != 0
            && self.nvfp4_silu_scaled_k.0 != 0
            && matches!(self.activation, FfnActivation::SiLU)
            && self.lora.is_none()
            && ctx.levers.ffn_nvfp4_mmq;
        if fp4mmq_prefill {
            // 2026-09-25: Latched per model (`ModelStats::once`).
            if ctx.stats.once("log:ffn_fp4_mmq_prefill") {
                tracing::info!(target: "metrale_model_layers::layers::dense_ffn", "[metrale] METRALE_FFN_NVFP4_MMQ=1: dense-FFN gate/up prefill via vendored llama NVFP4 W4A4 MMQ (block-scale FP4 MMA, ~80 TFLOP/s vs t_m128 ~51)"
                );
            }
        }
        // 2026-09-25: Down takes the NVFP4 MMQ arm too, unless `METRALE_NO_FFN_NVFP4_MMQ_DOWN` is
        // set; it needs `nvfp4_scale_k` to apply down's `weight_scale_2`.
        let fp4mmq_down =
            fp4mmq_prefill && self.nvfp4_scale_k.0 != 0 && ctx.levers.ffn_nvfp4_mmq_down;
        // 2026-09-25: Under the Q4_K arm, down runs int8 (`int8_gemm_faith2`) unless
        // `METRALE_FFN_MMQ_DOWN_Q4K` is set. Down never takes the Q4_K arm (`w4_gemm!` gets
        // `allow_q4k = false` for it), so with that variable set down falls to the later arms.
        let down_faith2 = self.q4k_mmq_nc_k.0 != 0
            && self.q4k_quant_act_k.0 != 0
            && self.q4k_quant_w_k.0 != 0
            && self.dequant_nvfp4_bf16_k.0 != 0
            && self.int8_faith2_k.0 != 0
            && self.requant_a_int8_k.0 != 0
            && !fp4mmq_prefill
            && ctx.levers.ffn_mmq
            && !ctx.levers.ffn_mmq_down_q4k;
        let (int8_a_i8, int8_a_scale) = if int8_prefill || down_faith2 {
            (ctx.buffers.ffn_act_a(), ctx.buffers.ffn_act_scale())
        } else {
            (DevicePtr::NULL, DevicePtr::NULL)
        };
        let fp4_prefill =
            self.w4a4_gemm_k.0 != 0 && self.quantize_nvfp4_k.0 != 0 && ctx.levers.fp4_prefill;
        if fp4_prefill {
            // 2026-09-25: Latched per model (`ModelStats::once`).
            if ctx.stats.once("log:ffn_fp4_prefill") {
                tracing::info!(target: "metrale_model_layers::layers::dense_ffn", "[metrale] METRALE_FP4_PREFILL=1: dense-FFN prefill via w4a4_gemm (native FP4 MMA sm_121a, W4A4)"
                );
            }
        }
        let (nvfp4_a_packed, nvfp4_a_scale) = if fp4_prefill {
            (ctx.buffers.ffn_act_a(), ctx.buffers.ffn_act_scale())
        } else {
            (DevicePtr::NULL, DevicePtr::NULL)
        };
        let q4k_prefill = self.q4k_mmq_nc_k.0 != 0
            && self.q4k_quant_act_k.0 != 0
            && self.q4k_quant_w_k.0 != 0
            && self.dequant_nvfp4_bf16_k.0 != 0
            && !fp4mmq_prefill
            && ctx.levers.ffn_mmq;
        if q4k_prefill {
            // 2026-09-25: Latched per model (`ModelStats::once`).
            if ctx.stats.once("log:ffn_q4k_prefill") {
                tracing::info!(target: "metrale_model_layers::layers::dense_ffn", "[metrale] METRALE_FFN_MMQ=1: dense-FFN prefill via vendored llama Q4_K MMQ (W4A8, +25%/+10% gate·down vs faith2)"
                );
            }
        }
        let q4k_a = if q4k_prefill {
            ctx.buffers.ffn_act_q8()
        } else {
            DevicePtr::NULL
        };
        // 2026-09-25: The NVFP4 MMQ activation also lives in `ffn_act_q8`; `q4k_prefill` is false
        // whenever this arm is on.
        let fp4_y = if fp4mmq_prefill {
            ctx.buffers.ffn_act_q8()
        } else {
            DevicePtr::NULL
        };
        // 2026-09-25: `METRALE_DISABLE_PREFILL_V2` keeps the BF16 arm on `w4a16_gemm_t_m128_bf16`
        // even when the v2 kernel resolved.
        let use_v2 = self.w4a16_gemm_t_m128_bf16_v2_k.0 != 0 && ctx.levers.prefill_v2;
        let bf16_kernel = if use_v2 {
            self.w4a16_gemm_t_m128_bf16_v2_k
        } else {
            self.w4a16_gemm_t_m128_bf16_k
        };
        // 2026-09-25: The BF16 arm needs the lever and the handle of the kernel actually chosen.
        let bf16_tc_prefill = bf16_kernel.0 != 0 && bf16_tc_env;
        Nvfp4PrefillPlan {
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
        }
    }
}
