// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `Qwen3AttentionLayer` weight installation after construction
//! (transposed NVFP4 copies, the fused q/k/v copy, FP8 and packed Q2_0 weights,
//! LoRA adapters), the FP8 transposes and NVFP4-to-FP8 copies prefill reads, the
//! packed Q2_0 prefill GEMM, and the NVFP4 M128 prefill GEMM dispatcher.
//!
//! Owner: model-layers (qwen3 attention).
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

use super::types::Qwen3AttentionLayer;
use super::types_weights::Fp8TwinSet;
use crate::weight_map::{Fp8Weight, Fp8WeightTransposed, QuantWeight, QuantizedWeight};

impl Qwen3AttentionLayer {
    /// 2026-09-25: The M128-tile NVFP4 prefill GEMM on a transposed weight,
    /// first match: the BF16-MMA kernel under `bf16_tc_proj`; v3 when
    /// `METRALE_W4A16_VARIANT` pins 3; v2 unless it pins 1; else v1. A kernel
    /// that is not loaded is skipped. The arguments are those of
    /// `ops::w4a16_gemm_n128_m128`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn w4a16_gemm_m128_dispatch(
        &self,
        gpu: &dyn GpuBackend,
        dispatch: &crate::layers::ops::GemmDispatch,
        input: DevicePtr,
        weight: &crate::weight_map::QuantizedWeight,
        output: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        stream: u64,
    ) -> anyhow::Result<()> {
        // 2026-09-25: `GemmDispatch::w4a16_variant` (`METRALE_W4A16_VARIANT`):
        // 1, 2 or 3 pins a kernel, 0 is auto (v2).
        let v = dispatch.w4a16_variant;
        // 2026-09-25: Opt-in (`METRALE_BF16_TC_PROJ` present): the BF16-MMA
        // kernel, which `ops::w4a16_gemm_n128_m128_bf16` documents as the base
        // `w4a16_gemm` arithmetic. The lever comes from the process-wide
        // `ModelLevers::get()`, resolved once.
        let bf16_proj = crate::layers::ops::ModelLevers::get().bf16_tc_proj;
        if bf16_proj && self.w4a16_gemm_t_m128_bf16_k.0 != 0 {
            return crate::layers::ops::w4a16_gemm_n128_m128_bf16(
                gpu,
                self.w4a16_gemm_t_m128_bf16_k,
                input,
                weight,
                output,
                m,
                n,
                k,
                stream,
            );
        }
        if v == 3 && self.w4a16_gemm_t_m128_v3_k.0 != 0 {
            crate::layers::ops::w4a16_gemm_n128_m128_v3(
                gpu,
                self.w4a16_gemm_t_m128_v3_k,
                input,
                weight,
                output,
                m,
                n,
                k,
                stream,
            )
        } else if v != 1 && self.w4a16_gemm_t_m128_v2_k.0 != 0 {
            crate::layers::ops::w4a16_gemm_n128_m128_v2(
                gpu,
                self.w4a16_gemm_t_m128_v2_k,
                input,
                weight,
                output,
                m,
                n,
                k,
                stream,
            )
        } else {
            crate::layers::ops::w4a16_gemm_n128_m128(
                gpu,
                self.w4a16_gemm_t_m128_k,
                input,
                weight,
                output,
                m,
                n,
                k,
                stream,
            )
        }
    }

    /// 2026-09-25: Install the transposed NVFP4 q/k/v/o copies the prefill
    /// GEMMs read.
    pub fn set_prefill_weights(
        &mut self,
        q_nvfp4_t: Option<QuantizedWeight>,
        k_nvfp4_t: Option<QuantizedWeight>,
        v_nvfp4_t: Option<QuantizedWeight>,
        o_nvfp4_t: Option<QuantizedWeight>,
    ) {
        self.q_nvfp4_t = q_nvfp4_t;
        self.k_nvfp4_t = k_nvfp4_t;
        self.v_nvfp4_t = v_nvfp4_t;
        self.o_nvfp4_t = o_nvfp4_t;
    }

    /// 2026-09-25: Install packed ternary Q2_0 q/k/v/o weights (the
    /// `qwen35_dense` loader, `METRALE_GGUF_NATIVE_Q2=1`), replacing the decode
    /// weights. Decode runs `q2_0_gemv_vec`; prefill runs `q2_prefill_gemm`.
    pub fn set_packed_q2_weights(
        &mut self,
        q: crate::weight_map::PackedQ2Weight,
        k: crate::weight_map::PackedQ2Weight,
        v: crate::weight_map::PackedQ2Weight,
        o: crate::weight_map::PackedQ2Weight,
        gpu: &dyn metrale_gpu_runtime::gpu::GpuBackend,
    ) {
        self.q_weight = Some(QuantWeight::PackedQ2(q));
        self.k_weight = Some(QuantWeight::PackedQ2(k));
        self.v_weight = Some(QuantWeight::PackedQ2(v));
        self.o_weight = Some(QuantWeight::PackedQ2(o));
        // 2026-09-25: Looked up here, not in the constructor: only targets that
        // serve GGUF carry these kernels, and the boot audit refuses to serve
        // after a failed lookup (`layers/kernel_probe.rs`).
        self.q2_0_mmq_nc_k = crate::layers::try_kernel(gpu, "q2_0_mmq", "metrale_q2_0_mmq128_nc");
        self.q2_0_mmq_wc_k = crate::layers::try_kernel(gpu, "q2_0_mmq", "metrale_q2_0_mmq128_wc");
        self.q4k_quant_act_k =
            crate::layers::try_kernel(gpu, "q4k_mmq", "metrale_q8_1_quantize_ds4_bf16");
    }

    /// 2026-09-25: Prefill GEMM for a packed Q2_0 projection,
    /// `out[m, n] = in[m, k] @ w^T`. With the native MMQ kernels and group 128,
    /// the input is quantized to q8_1 and the packed weight is used directly.
    /// Otherwise the weight is dequantized into the BF16 `scratch` (the arena's
    /// `q2_dequant_scratch`) and a BF16 GEMM runs; both are on `stream`, so
    /// the next projection's dequant cannot overwrite `scratch` before this GEMM
    /// reads it. Nothing is allocated. Errors if the dequant kernel is missing.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn q2_prefill_gemm(
        &self,
        gpu: &dyn GpuBackend,
        w: &crate::weight_map::PackedQ2Weight,
        input: DevicePtr,
        out: DevicePtr,
        scratch: DevicePtr,
        act_q8: DevicePtr,
        m: u32,
        stream: u64,
    ) -> Result<()> {
        let (n, k) = (w.n, w.k);

        // 2026-09-25: Native MMQ (`METRALE_GGUF_NATIVE_Q2_MMQ=1`, group 128 only):
        // no BF16 weight dequant.
        if self.q2_0_mmq_nc_k.0 != 0
            && self.q4k_quant_act_k.0 != 0
            && crate::layers::ops::native_q2_mmq_enabled()
            && w.group == 128
        {
            crate::layers::ops::quantize_act_q8_1(
                gpu,
                self.q4k_quant_act_k,
                input,
                act_q8,
                m,
                k,
                stream,
            )?;
            return crate::layers::ops::q2_0_mmq_gemm(
                gpu,
                self.q2_0_mmq_nc_k,
                self.q2_0_mmq_wc_k,
                act_q8,
                w.weight,
                out,
                m,
                n,
                k,
                stream,
            );
        }

        if self.dequant_q2_0_gn_k.0 == 0 {
            anyhow::bail!(
                "dequant_q2_0_gn_to_bf16 kernel missing — packed-Q2 attention prefill unavailable"
            );
        }
        crate::layers::ops::dequant_q2_0_gn_to_bf16(
            gpu,
            self.dequant_q2_0_gn_k,
            w.weight,
            scratch,
            n,
            k,
            w.group as u32,
            stream,
        )?;
        let dw = crate::weight_map::DenseWeight { weight: scratch };
        if self.dense_gemm_pipelined_k.0 != 0 {
            crate::layers::ops::dense_gemm_bf16_pipelined(
                gpu,
                self.dense_gemm_pipelined_k,
                input,
                &dw,
                out,
                m,
                n,
                k,
                stream,
            )?;
        } else {
            crate::layers::ops::dense_gemm(
                gpu,
                self.dense_gemm_k,
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
    }

    /// 2026-09-25: The packed Q2_0 prefill check, the first check in the Q/K/V
    /// (`paged_qkv.rs`, `cache_skip_qkv.rs`) and O (`paged_oproj.rs`) prefill
    /// projections. For a packed Q2_0 `weight` it runs `q2_prefill_gemm` with
    /// the arena scratch and returns `Some(result)`; otherwise `None`, and the
    /// caller takes its other routes.
    pub(crate) fn try_q2_prefill(
        &self,
        ctx: &crate::layer::ForwardContext,
        weight: Option<&QuantWeight>,
        input: DevicePtr,
        out: DevicePtr,
        m: u32,
        stream: u64,
    ) -> Option<Result<()>> {
        let q2 = weight.and_then(|w| w.as_packed_q2())?;
        debug_assert!(
            (q2.n as usize) * (q2.k as usize) * 2 <= ctx.buffers.q2_dequant_scratch_bytes(),
            "packed-Q2 prefill dequant scratch too small"
        );
        let scratch = ctx.buffers.q2_dequant_scratch();
        let act_q8 = ctx.buffers.q2_act_q8();
        Some(self.q2_prefill_gemm(ctx.gpu, q2, input, out, scratch, act_q8, m, stream))
    }

    /// 2026-09-25: Install the fused `[q | k | v]` transposed copy. Separate from
    /// `set_prefill_weights`, so a loader opts in and the separate copies stay
    /// as the fallback.
    pub fn set_fused_qkv_prefill_weight(&mut self, qkv_nvfp4_t: Option<QuantizedWeight>) {
        self.qkv_nvfp4_t = qkv_nvfp4_t;
    }
    /// 2026-09-25: Install native FP8 checkpoint weights, replacing the decode
    /// weight of each projection given.
    ///
    /// Block-scaled FP8 weights stored here (weight and per-128 `row_scale`)
    /// are also read by the W8A8 prefill, whose `fp8_gemm_t_blockscaled` folds
    /// the per-token activation scale and the per-block weight scale in an FP32
    /// epilogue.
    pub fn set_fp8_weights(
        &mut self,
        q: Option<Fp8Weight>,
        k: Option<Fp8Weight>,
        v: Option<Fp8Weight>,
        o: Option<Fp8Weight>,
    ) {
        if let Some(qw) = q {
            self.q_weight = Some(QuantWeight::Fp8(qw));
        }
        if let Some(kw) = k {
            self.k_weight = Some(QuantWeight::Fp8(kw));
        }
        if let Some(vw) = v {
            self.v_weight = Some(QuantWeight::Fp8(vw));
        }
        if let Some(ow) = o {
            self.o_weight = Some(QuantWeight::Fp8(ow));
        }
    }

    /// 2026-09-25: Install the LoRA adapter weights after construction. `attn`
    /// carries the q/k/v/o pairs; `ffn`, when given, goes to this layer's dense
    /// FFN (an error on any other FFN). It is routed here because `self.ffn` is
    /// `pub(super)`.
    pub fn set_lora_weights(
        &mut self,
        attn: crate::layers::ops::lora_delta::LoraAttnWeights,
        ffn: Option<crate::layers::ops::lora_delta::LoraFfnWeights>,
    ) -> Result<()> {
        self.lora = Some(attn);
        if let Some(f) = ffn {
            match &mut self.ffn {
                crate::layers::FfnComponent::Dense(d) => d.set_lora_weights(f)?,
                _ => anyhow::bail!("LoRA: FFN targets on a non-dense FFN layer"),
            }
        }
        Ok(())
    }

    /// 2026-09-25: Install this layer's MoE router and routed-expert LoRA on its
    /// MoE FFN, `self.ffn` or else `self.moe_ffn`; an error when the layer has
    /// neither. `MoeLayer::set_lora_weights` allocates the scratch.
    pub fn set_moe_lora_weights(
        &mut self,
        router: Option<crate::layers::ops::lora_delta::LoraPair>,
        experts: crate::lora::ExpertLoraLayer,
        kernels: crate::layers::ops::lora_delta::LoraKernels,
        gpu: &dyn GpuBackend,
    ) -> Result<()> {
        if let crate::layers::FfnComponent::Moe(m) = &mut self.ffn {
            return m.set_lora_weights(router, experts, kernels, gpu);
        }
        if let Some(crate::layers::FfnComponent::Moe(m)) = &mut self.moe_ffn {
            return m.set_lora_weights(router, experts, kernels, gpu);
        }
        anyhow::bail!("LoRA: router/expert deltas installed on a layer with no MoE FFN component")
    }

    /// 2026-09-25: Whether both kernels of the W8A8 block-scaled prefill route
    /// are loaded (the activation quantizer and `fp8_gemm_t_blockscaled`). The
    /// `qwen35_dense` loader uses it to decide which FP8 prefill twins to build
    /// (`attn_fp8_twins`).
    pub fn has_w8a8_prefill_kernels(&self) -> bool {
        self.per_token_group_quant_fp8_k.available() && self.fp8_gemm_t_blockscaled_k.0 != 0
    }

    /// 2026-09-25: Build transposed copies of all four FP8 weights for the
    /// prefill GEMMs, with no owner for teardown. Call after `set_fp8_weights`;
    /// it allocates device memory.
    pub fn transpose_fp8_for_prefill(
        &mut self,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> anyhow::Result<()> {
        self.transpose_fp8_for_prefill_selected(gpu, stream, Fp8TwinSet::ALL, None)
    }

    /// 2026-09-25: `transpose_fp8_for_prefill` for the projections in `want`
    /// only, handing each copy to `derived` (when given) so teardown releases it.
    /// `Fp8TwinSet` documents which projection each prefill chain reads.
    pub fn transpose_fp8_for_prefill_selected(
        &mut self,
        gpu: &dyn GpuBackend,
        stream: u64,
        want: Fp8TwinSet,
        derived: Option<&metrale_model_weights::weights::DerivedStore>,
    ) -> anyhow::Result<()> {
        // 2026-09-25: A load-time decision, made before a model exists to carry
        // the dispatch, so it reads `GemmDispatch::from_env()`.
        if crate::layers::ops::GemmDispatch::from_env().cutlass_nvfp4_gemm {
            tracing::info!(
                "Skipping attention FP8 prefill transposes because METRALE_CUTLASS_NVFP4_GEMM=1"
            );
            return Ok(());
        }
        if !want.any() {
            return Ok(());
        }
        if self.w8a16_gemm_t_k.0 == 0 {
            return Ok(());
        }
        let transpose_k = gpu.kernel("w8a16_gemm_t", "transpose_fp8")?;
        let transpose_scale_k = gpu.kernel("w8a16_gemm_t", "transpose_block_scale")?;

        let build = |src: Option<&QuantWeight>| -> anyhow::Result<Option<Fp8WeightTransposed>> {
            let Some(w) = src.and_then(|w| w.as_fp8()) else {
                return Ok(None);
            };
            let t = w.transpose_for_gemm(gpu, transpose_k, transpose_scale_k, stream)?;
            if let Some(d) = derived {
                let (n, k) = (w.n as usize, w.k as usize);
                d.adopt("attn fp8 prefill twin (weight_t)", t.weight_t, n * k);
                d.adopt(
                    "attn fp8 prefill twin (scale_t)",
                    t.scale_t,
                    n.div_ceil(128) * k.div_ceil(128) * 4,
                );
            }
            Ok(Some(t))
        };

        if want.q {
            self.q_fp8w_t = build(self.q_weight.as_ref())?;
        }
        if want.k {
            self.k_fp8w_t = build(self.k_weight.as_ref())?;
        }
        if want.v {
            self.v_fp8w_t = build(self.v_weight.as_ref())?;
        }
        if want.o {
            self.o_fp8w_t = build(self.o_weight.as_ref())?;
        }
        Ok(())
    }

    /// 2026-09-25: Build FP8 copies of the NVFP4 q/k/v weights, and of `attn.o_proj`
    /// when a transposed NVFP4 O weight is installed, for the prefill FP8 routes.
    pub fn predequant_for_prefill(
        &mut self,
        gpu: &dyn GpuBackend,
        config: &metrale_config::ModelConfig,
        stream: u64,
    ) -> Result<()> {
        // 2026-09-25: Skipped under `METRALE_CUTLASS_NVFP4_GEMM=1`, as in
        // `transpose_fp8_for_prefill_selected`, with the same load-time
        // `GemmDispatch::from_env()`.
        if crate::layers::ops::GemmDispatch::from_env().cutlass_nvfp4_gemm {
            tracing::info!(
                "Skipping attention FP8 prefill predequant because METRALE_CUTLASS_NVFP4_GEMM=1"
            );
            return Ok(());
        }
        let predequant_k = gpu.kernel("w4a16", "predequant_nvfp4_to_fp8")?;
        let h = config.hidden_size;
        let nq = config.num_attention_heads;
        let nkv = config.num_key_value_heads;
        let hd = config.head_dim;
        let q_dim = nq * hd;
        let q_proj_dim = if self.gated { q_dim * 2 } else { q_dim };
        let kv_dim = nkv * hd;

        // 2026-09-25: The non-transposed weights: `predequant_to_fp8` reads
        // `[N, K/2]` packed NVFP4.
        if let Some(nvfp4) = self.q_weight.as_ref().and_then(|w| w.as_nvfp4()) {
            self.q_fp8 = Some(nvfp4.predequant_to_fp8(gpu, predequant_k, q_proj_dim, h, stream)?);
        }
        if let Some(nvfp4) = self.k_weight.as_ref().and_then(|w| w.as_nvfp4()) {
            self.k_fp8 = Some(nvfp4.predequant_to_fp8(gpu, predequant_k, kv_dim, h, stream)?);
        }
        if let Some(nvfp4) = self.v_weight.as_ref().and_then(|w| w.as_nvfp4()) {
            self.v_fp8 = Some(nvfp4.predequant_to_fp8(gpu, predequant_k, kv_dim, h, stream)?);
        }
        // 2026-09-25: O uses `attn.o_proj`, the non-transposed weight.
        if self.o_nvfp4_t.is_some() {
            self.o_fp8 =
                Some(
                    self.attn
                        .o_proj
                        .predequant_to_fp8(gpu, predequant_k, h, q_dim, stream)?,
                );
        }
        Ok(())
    }
}
