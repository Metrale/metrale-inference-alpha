// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The two LongCat FFN builders: the dense SwiGLU MLP every sublayer has, and
//! the shortcut MoE that sublayer 0 computes and sublayer 1 adds.
//!
//! Owner: model-arch weight loader (LongCat).
//! Invariants:
//! - The dense FFN always holds NVFP4 weights; `METRALE_LONGCAT_BF16_FFN=1` adds the BF16
//!   weights beside them.
//! - With `METRALE_LONGCAT_FP8_EXPERTS=1` the NVFP4 expert slots are null and the FP8
//!   tables are installed, or the load fails.

use anyhow::{Context, Result};
use metrale_config::ModelConfig;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_weights::weights::WeightStore;

use metrale_model_layers::layers::{FfnComponent, MoeLayer};
use metrale_model_layers::weight_map::{
    DenseWeight, ExpertWeight, MoeWeights, QuantizedWeight, dense, dense_f32_as_bf16,
    quantize_to_nvfp4, quantized_any,
};

pub(super) fn build_dense_ffn(
    store: &WeightStore,
    prefix: &str,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
) -> Result<FfnComponent> {
    use metrale_model_layers::layers::dense_ffn::{DenseFfnLayer, DenseFfnWeights};
    let inter = config.intermediate_size;
    let h = config.hidden_size;
    let absmax_k = gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax")?;
    let quantize_k = gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4")?;
    let stream = gpu.default_stream();
    let q = |name: &str, n: usize, k: usize| -> Result<QuantizedWeight> {
        let w = dense(store, name)?;
        quantize_to_nvfp4(&w, n, k, gpu, absmax_k, quantize_k, stream)
    };
    let weights = DenseFfnWeights {
        gate_proj: q(&format!("{prefix}.gate_proj.weight"), inter, h)?,
        up_proj: q(&format!("{prefix}.up_proj.weight"), inter, h)?,
        down_proj: q(&format!("{prefix}.down_proj.weight"), h, inter)?,
        gate_proj_t: None,
        up_proj_t: None,
        down_proj_t: None,
    };
    let mut layer = DenseFfnLayer::new(weights, gpu)?;

    // 2026-09-25: The three projections are quantized to NVFP4 above from
    // the store's weights. With the lever on, `set_bf16_weights` also installs
    // those weights, and the forward paths dispatch to the BF16 kernels; the
    // NVFP4 copy stays resident.
    if bf16_dense_ffn() {
        layer.set_bf16_weights(
            dense(store, &format!("{prefix}.gate_proj.weight"))?,
            dense(store, &format!("{prefix}.up_proj.weight"))?,
            dense(store, &format!("{prefix}.down_proj.weight"))?,
        );
    }
    Ok(FfnComponent::Dense(layer))
}

/// 2026-09-25: `METRALE_LONGCAT_BF16_FFN=1` runs the per-sublayer dense FFN in BF16.
pub(super) fn bf16_dense_ffn() -> bool {
    std::env::var("METRALE_LONGCAT_BF16_FFN").as_deref() == Ok("1")
}

/// 2026-09-25: The block's shortcut MoE: `mlp.router.*` + `mlp.experts.{e}.*`.
/// No shared-expert tensor is loaded; the shared slot is zero-filled.
pub(super) fn build_shortcut_moe(
    store: &WeightStore,
    layer_prefix: &str,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
) -> Result<FfnComponent> {
    let p = format!("{layer_prefix}.mlp");
    let h = config.hidden_size;
    let inter = config.moe_intermediate_size;
    // 2026-09-25: The router classifier is converted F32 to BF16, or bound as
    // stored when that fails; the correction bias is bound as stored.
    let gate_name = format!("{p}.router.classifier.weight");
    let gate = dense_f32_as_bf16(store, &gate_name, gpu)
        .or_else(|_| dense(store, &gate_name))
        .context("longcat: router classifier")?;
    let correction_bias = dense(store, &format!("{p}.router.e_score_correction_bias"))
        .context("longcat: router e_score_correction_bias")?;

    let alloc_zero = |size: usize| -> Result<DevicePtr> {
        let ptr = gpu.alloc(size)?;
        gpu.memset(ptr, 0, size)?;
        Ok(ptr)
    };
    let group = 16usize;
    let mk_zero = |packed: usize, scale: usize| -> Result<QuantizedWeight> {
        Ok(QuantizedWeight {
            weight: alloc_zero(packed)?,
            weight_scale: alloc_zero(scale)?,
            weight_scale_2: 0.0,
            input_scale: DevicePtr::NULL,
            weight_scale_2_vec: DevicePtr::NULL,
        })
    };
    let shared_expert = ExpertWeight {
        gate_proj: mk_zero(inter * h / 2, inter * (h / group))?,
        up_proj: mk_zero(inter * h / 2, inter * (h / group))?,
        down_proj: mk_zero(h * inter / 2, h * (inter / group))?,
    };
    let shared_expert_gate = DenseWeight {
        weight: alloc_zero(h * 2)?,
    };

    // 2026-09-25: The NVFP4 experts are read through `quantized_any` with the
    // detected variant; a `Bf16Raw` checkpoint is quantized at load.
    let variant = metrale_model_layers::weight_map::detect_nvfp4_variant(store, config);
    let qctx = metrale_model_layers::weight_map::QuantizeCtx {
        absmax_k: gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax")?,
        quantize_k: gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4")?,
        stream: gpu.default_stream(),
    };
    let fp8 = fp8_experts();
    let mut experts = Vec::with_capacity(config.num_experts);
    let mut fp8_experts_vec = Vec::with_capacity(if fp8 { config.num_experts } else { 0 });
    let fp8_quant_k = if fp8 {
        gpu.kernel(
            "quantize_bf16_to_fp8_blockscaled",
            "quantize_bf16_to_fp8_blockscaled",
        )?
    } else {
        metrale_gpu_runtime::gpu::KernelHandle(0)
    };
    for e in 0..config.num_experts {
        let ep = format!("{p}.experts.{e}");
        if fp8 {
            // 2026-09-25: The NVFP4 slot stays null; `set_fp8_experts` below
            // installs the FP8 tables.
            experts.push(ExpertWeight::null());
            fp8_experts_vec.push(metrale_model_layers::weight_map::Fp8ExpertWeight {
                gate_proj: quant_expert_fp8(
                    store,
                    &format!("{ep}.gate_proj"),
                    inter,
                    h,
                    gpu,
                    fp8_quant_k,
                    qctx.stream,
                )?,
                up_proj: quant_expert_fp8(
                    store,
                    &format!("{ep}.up_proj"),
                    inter,
                    h,
                    gpu,
                    fp8_quant_k,
                    qctx.stream,
                )?,
                down_proj: quant_expert_fp8(
                    store,
                    &format!("{ep}.down_proj"),
                    h,
                    inter,
                    gpu,
                    fp8_quant_k,
                    qctx.stream,
                )?,
            });
            continue;
        }
        experts.push(ExpertWeight {
            gate_proj: quantized_any(
                store,
                &format!("{ep}.gate_proj"),
                inter,
                h,
                gpu,
                variant,
                qctx,
            )?,
            up_proj: quantized_any(
                store,
                &format!("{ep}.up_proj"),
                inter,
                h,
                gpu,
                variant,
                qctx,
            )?,
            down_proj: quantized_any(
                store,
                &format!("{ep}.down_proj"),
                h,
                inter,
                gpu,
                variant,
                qctx,
            )?,
        });
    }

    let weights = MoeWeights {
        gate,
        shared_expert,
        shared_expert_gate,
        experts,
        router_pre_norm: None,
        correction_bias: Some(correction_bias),
    };
    let mut moe = MoeLayer::new(weights, config.num_experts, None, gpu, config)?;
    if fp8 {
        // 2026-09-25: A failure to install the FP8 tables fails the load: the
        // NVFP4 expert slots above are null. The FP8 shared slot is zero-filled,
        // like the NVFP4 one.
        let mk_zero_fp8 = |n: usize,
                           k: usize|
         -> Result<metrale_model_layers::weight_map::Fp8Weight> {
            Ok(metrale_model_layers::weight_map::Fp8Weight {
                weight: alloc_zero(n * k)?,
                row_scale: alloc_zero(n.div_ceil(128) * k.div_ceil(128) * 4)?,
                n: n as u32,
                k: k as u32,
                scale_format: metrale_model_layers::weight_map::WeightQuantFormat::Fp8BlockScaled,
            })
        };
        moe.set_fp8_experts(
            &fp8_experts_vec,
            metrale_model_layers::weight_map::Fp8ExpertWeight {
                gate_proj: mk_zero_fp8(inter, h)?,
                up_proj: mk_zero_fp8(inter, h)?,
                down_proj: mk_zero_fp8(h, inter)?,
            },
            gpu,
        )
        .context("longcat: installing FP8 expert pointer tables")?;
    }
    Ok(FfnComponent::Moe(moe))
}

/// 2026-09-25: `METRALE_LONGCAT_FP8_EXPERTS=1` quantizes the routed experts to
/// block-scaled FP8 at load instead of NVFP4.
pub(super) fn fp8_experts() -> bool {
    std::env::var("METRALE_LONGCAT_FP8_EXPERTS").as_deref() == Ok("1")
}

/// 2026-09-25: One expert projection: BF16 from the store to block-scaled FP8,
/// then the BF16 source is freed.
fn quant_expert_fp8(
    store: &WeightStore,
    prefix: &str,
    n: usize,
    k: usize,
    gpu: &dyn GpuBackend,
    quantize_k: metrale_gpu_runtime::gpu::KernelHandle,
    stream: u64,
) -> Result<metrale_model_layers::weight_map::Fp8Weight> {
    let w = store.get(&format!("{prefix}.weight"))?;
    let bf16 = DenseWeight { weight: w.ptr };
    let q = metrale_model_layers::weight_map::quantize_to_fp8_blockscaled(
        &bf16, n, k, gpu, quantize_k, stream,
    )?;
    // 2026-09-25: The kernel reads the BF16 source on `stream`; the free must
    // not race it.
    gpu.synchronize(stream)?;
    gpu.free(w.ptr)?;
    Ok(q)
}
