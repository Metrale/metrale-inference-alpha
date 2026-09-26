// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Detection of the on-disk weight format (`Nvfp4Variant`) and the variant-dispatching quantized-weight loaders.
//!
//! Owner: model-layers (weight loading).
//! Invariants:
//! - A variant declared by `config.quantization_config` wins over tensor-name sniffing.
//! - Detection reads only layer 0 and the whole name list, never a layer chosen by EP rank.

#![allow(unused_imports)]

use anyhow::{Context, Result, bail, ensure};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_weights::weights::{WeightDtype, WeightStore};

use super::*;

/// 2026-09-25: Step (1) of [`detect_nvfp4_variant`]: the variant the config
/// declares, answered without a [`WeightStore`], so the pre-load residency
/// prediction (`qwen35_dense::predicted_residency`) asks the same question as
/// the loader. `None` means the config does not say; the sniffing half of
/// [`detect_nvfp4_variant`] then decides.
pub fn config_declared_variant(config: &metrale_config::ModelConfig) -> Option<Nvfp4Variant> {
    let qc = config.quantization_config.as_ref()?;
    match qc.quant_method.as_str() {
        "modelopt" if qc.quant_algo.eq_ignore_ascii_case("NVFP4") => Some(Nvfp4Variant::Standard),
        "modelopt" if qc.quant_algo.eq_ignore_ascii_case("FP8") => Some(Nvfp4Variant::Fp8Dequanted),
        "compressed-tensors" => {
            // 2026-09-25: `format` selects: containing "fp8" or "float-quant" is
            // block-scaled FP8, anything else NVFP4.
            let fmt = qc.format.to_ascii_lowercase();
            if fmt.contains("fp8") || fmt.contains("float-quant") {
                Some(Nvfp4Variant::Fp8Dequanted)
            } else {
                Some(Nvfp4Variant::CompressedTensors)
            }
        }
        "fp8" => Some(Nvfp4Variant::Fp8Dequanted),
        _ => None,
    }
}

/// 2026-09-25: Detect the on-disk weight format.
///
/// 1. [`config_declared_variant`] wins when it answers. A checkpoint with an
///    `ignore` list can leave a projection unquantized, which tensor-name
///    sniffing would misread.
/// 2. Otherwise the tensor names are sniffed, in this order:
///    compressed-tensors `weight_packed`, then FP8 `weight_scale_inv` or an
///    FP8E4M3 projection weight, then `Bf16Raw` when layer 0's MLP
///    `gate_proj` has no `weight_scale`, and `Standard` last.
pub fn detect_nvfp4_variant(
    store: &WeightStore,
    config: &metrale_config::ModelConfig,
) -> Nvfp4Variant {
    if let Some(declared) = config_declared_variant(config) {
        return declared;
    }

    let lp = config.layer_prefix(0);

    let local_expert = config.local_expert_range().0;
    let moe_sehyo_key = format!("{lp}.mlp.experts.{local_expert}.gate_proj.weight_packed");
    if store.contains(&moe_sehyo_key) {
        return Nvfp4Variant::CompressedTensors;
    }

    let dense_sehyo_key = format!("{lp}.mlp.gate_proj.weight_packed");
    if store.contains(&dense_sehyo_key) {
        return Nvfp4Variant::CompressedTensors;
    }

    let mistral_key = format!("layers.0.experts.{local_expert}.w1.weight_packed");
    if store.contains(&mistral_key) {
        return Nvfp4Variant::CompressedTensors;
    }

    if store.names().any(|k| k.ends_with(".weight_packed")) {
        return Nvfp4Variant::CompressedTensors;
    }

    // 2026-09-25: Two spellings of layer 0, never a layer index derived from the
    // EP rank, so every rank reaches the same answer.
    const ALT_LAYER0_PREFIX: &str = "model.language_model.layers.0";
    let prefixes_to_check = [lp.clone(), ALT_LAYER0_PREFIX.to_string()];
    for pfx in &prefixes_to_check {
        let fp8_key = format!("{pfx}.mlp.experts.{local_expert}.gate_proj.weight_scale_inv");
        if store.contains(&fp8_key) {
            return Nvfp4Variant::Fp8Dequanted;
        }
        let fp8_dense_key = format!("{pfx}.mlp.gate_proj.weight_scale_inv");
        if store.contains(&fp8_dense_key) {
            return Nvfp4Variant::Fp8Dequanted;
        }
        let fp8_attn_key = format!("{pfx}.self_attn.q_proj.weight_scale_inv");
        if store.contains(&fp8_attn_key) {
            return Nvfp4Variant::Fp8Dequanted;
        }
        // 2026-09-25: FP8 with a `weight_scale` instead of `weight_scale_inv` is
        // told apart from NVFP4 by the FP8E4M3 weight dtype; the `weight_scale`
        // checks below would otherwise take it for `Standard`.
        for key in [
            format!("{pfx}.mlp.experts.{local_expert}.gate_proj.weight"),
            format!("{pfx}.mlp.gate_proj.weight"),
            format!("{pfx}.self_attn.q_proj.weight"),
        ] {
            if store
                .get(&key)
                .map(|w| w.dtype == WeightDtype::FP8E4M3)
                .unwrap_or(false)
            {
                return Nvfp4Variant::Fp8Dequanted;
            }
        }
    }
    if store.names().any(|k| k.ends_with(".weight_scale_inv")) {
        return Nvfp4Variant::Fp8Dequanted;
    }

    let any_standard_scale = store.names().any(|k| k.ends_with(".weight_scale"));
    if !any_standard_scale {
        tracing::warn!(
            "No NVFP4/FP8 quantization metadata found (no .weight_packed / .weight_scale_inv / .weight_scale). \
             Falling back to runtime BF16→NVFP4 quantization. Quality will be inferior to a calibrated NVFP4 release."
        );
        return Nvfp4Variant::Bf16Raw;
    }

    // 2026-09-25: A `weight_scale` elsewhere (for example on KV scales) is not
    // enough for `Standard`: layer 0's MLP `gate_proj` must carry one, or
    // `quantized` would fail on that missing key.
    let has_mlp_scale = {
        let k_dense = format!("{lp}.mlp.gate_proj.weight_scale");
        let k_moe = format!("{lp}.mlp.experts.{local_expert}.gate_proj.weight_scale");
        store.contains(&k_dense) || store.contains(&k_moe)
    };
    if !has_mlp_scale {
        tracing::warn!(
            "Partial NVFP4 metadata: `.weight_scale` exists for some tensors (e.g. KV scales) \
             but not for MLP/MoE projections. Falling back to runtime BF16→NVFP4 quantization. \
             For best quality use a fully-quantized NVFP4 release (e.g. Sehyo/*-NVFP4)."
        );
        return Nvfp4Variant::Bf16Raw;
    }

    Nvfp4Variant::Standard
}

/// 2026-09-25: Load a pre-quantized NVFP4 weight by variant: `quantized` for
/// `Standard`, `quantized_v2` for `CompressedTensors`. Panics for
/// `Fp8Dequanted` and `Bf16Raw`, which need [`quantized_any`].
pub fn quantized_auto(
    store: &WeightStore,
    prefix: &str,
    gpu: &dyn GpuBackend,
    variant: Nvfp4Variant,
) -> Result<QuantizedWeight> {
    match variant {
        Nvfp4Variant::Standard => quantized(store, prefix, gpu),
        Nvfp4Variant::CompressedTensors => quantized_v2(store, prefix, gpu),
        Nvfp4Variant::Fp8Dequanted => {
            unreachable!("Fp8Dequanted must use quantized_auto_fp8 with quant context")
        }
        Nvfp4Variant::Bf16Raw => {
            unreachable!("Bf16Raw must use quantized_any with quant context")
        }
    }
}

/// 2026-09-25: Kernels and stream for load-time NVFP4 quantization (`quantize_to_nvfp4`).
#[derive(Clone, Copy)]
pub struct QuantizeCtx {
    pub absmax_k: metrale_gpu_runtime::gpu::KernelHandle,
    pub quantize_k: metrale_gpu_runtime::gpu::KernelHandle,
    pub stream: u64,
}

/// 2026-09-25: Load a weight `[n, k]` as NVFP4 whatever its on-disk format.
///
/// The variant is overridden per key: a key with only a `.weight` is
/// quantized as `Bf16Raw`, and an FP8E4M3 key with a `weight_scale` or
/// `weight_scale_inv` but no NVFP4 global scale is dequantized as
/// `Fp8Dequanted` (unless `variant` is already `Fp8Dequanted` or `Bf16Raw`).
/// `n` and `k` are used only by those two re-quantizing paths.
pub fn quantized_any(
    store: &WeightStore,
    prefix: &str,
    n: usize,
    k: usize,
    gpu: &dyn GpuBackend,
    variant: Nvfp4Variant,
    qctx: QuantizeCtx,
) -> Result<QuantizedWeight> {
    let _t_detect = std::time::Instant::now();
    let has_packed = store.contains(&format!("{prefix}.weight_packed"));
    let has_scale = store.contains(&format!("{prefix}.weight_scale"));
    let has_scale_inv = store.contains(&format!("{prefix}.weight_scale_inv"));
    let has_only_dense =
        !has_packed && !has_scale && !has_scale_inv && store.contains(&format!("{prefix}.weight"));

    // 2026-09-25: Neither override can take an NVFP4 key: `Standard` keys have
    // `weight_scale_2`, `CompressedTensors` keys `weight_packed`, and both are
    // excluded here.
    let has_fp8_dense = !has_packed
        && !store.contains(&format!("{prefix}.weight_global_scale"))
        && !store.contains(&format!("{prefix}.weight_scale_2"))
        && (has_scale || has_scale_inv)
        && store
            .get(&format!("{prefix}.weight"))
            .map(|w| w.dtype == WeightDtype::FP8E4M3)
            .unwrap_or(false);

    let effective_variant = if has_only_dense && !matches!(variant, Nvfp4Variant::Bf16Raw) {
        tracing::debug!("{prefix}: no quantization metadata; falling back to runtime BF16→NVFP4");
        Nvfp4Variant::Bf16Raw
    } else if has_fp8_dense
        && !matches!(variant, Nvfp4Variant::Fp8Dequanted | Nvfp4Variant::Bf16Raw)
    {
        tracing::debug!("{prefix}: FP8 key in an NVFP4 checkpoint; dequant FP8→BF16→NVFP4");
        Nvfp4Variant::Fp8Dequanted
    } else {
        variant
    };

    let _t_detect_ns = _t_detect.elapsed().as_nanos() as u64;
    match effective_variant {
        Nvfp4Variant::Standard => quantized(store, prefix, gpu),
        Nvfp4Variant::CompressedTensors => quantized_v2(store, prefix, gpu),
        Nvfp4Variant::Fp8Dequanted => quantized_from_fp8(
            store,
            prefix,
            n,
            k,
            gpu,
            qctx.absmax_k,
            qctx.quantize_k,
            qctx.stream,
        ),
        Nvfp4Variant::Bf16Raw => {
            use std::sync::atomic::{AtomicU64, Ordering};
            static T_DETECT: AtomicU64 = AtomicU64::new(0);
            static T_GET: AtomicU64 = AtomicU64::new(0);
            static T_QUANT: AtomicU64 = AtomicU64::new(0);
            static T_FREE: AtomicU64 = AtomicU64::new(0);
            static N: AtomicU64 = AtomicU64::new(0);
            T_DETECT.fetch_add(_t_detect_ns, Ordering::Relaxed);
            let _t = std::time::Instant::now();
            let w = store.get(&format!("{prefix}.weight"))?;
            let bf16 = DenseWeight { weight: w.ptr };
            T_GET.fetch_add(_t.elapsed().as_nanos() as u64, Ordering::Relaxed);
            let _t = std::time::Instant::now();
            let q = quantize_to_nvfp4(
                &bf16,
                n,
                k,
                gpu,
                qctx.absmax_k,
                qctx.quantize_k,
                qctx.stream,
            )?;
            T_QUANT.fetch_add(_t.elapsed().as_nanos() as u64, Ordering::Relaxed);
            let _t = std::time::Instant::now();
            // 2026-09-25: The NVFP4 copy is a new allocation, so the store's BF16
            // weight is freed; the store keeps the freed pointer, and nothing may
            // read this key afterwards.
            gpu.free(w.ptr)?;
            T_FREE.fetch_add(_t.elapsed().as_nanos() as u64, Ordering::Relaxed);
            let c = N.fetch_add(1, Ordering::Relaxed) + 1;
            if c.is_multiple_of(512) {
                let ms = |a: &AtomicU64| a.load(Ordering::Relaxed) as f64 / 1.0e6;
                tracing::info!(
                    "quantized_any(Bf16Raw) PROFILE after {c} calls (ms total): detect={:.1} \
                     store_get={:.1} quantize={:.1} free={:.1} | sum={:.1} per_call={:.3}ms",
                    ms(&T_DETECT),
                    ms(&T_GET),
                    ms(&T_QUANT),
                    ms(&T_FREE),
                    ms(&T_DETECT) + ms(&T_GET) + ms(&T_QUANT) + ms(&T_FREE),
                    (ms(&T_DETECT) + ms(&T_GET) + ms(&T_QUANT) + ms(&T_FREE)) / c as f64,
                );
            }
            Ok(q)
        }
    }
}

/// 2026-09-25: Dequantize a block-scaled FP8 weight `[n, k]` to BF16, quantize
/// it to NVFP4, and free the BF16 intermediate.
pub fn quantized_from_fp8(
    store: &WeightStore,
    prefix: &str,
    n: usize,
    k: usize,
    gpu: &dyn GpuBackend,
    absmax_k: metrale_gpu_runtime::gpu::KernelHandle,
    quantize_k: metrale_gpu_runtime::gpu::KernelHandle,
    stream: u64,
) -> Result<QuantizedWeight> {
    let bf16 = dequant_fp8_blockscaled_to_bf16(store, prefix, gpu)?;
    let result = quantize_to_nvfp4(&bf16, n, k, gpu, absmax_k, quantize_k, stream)?;
    gpu.free(bf16.weight)?;
    Ok(result)
}

/// 2026-09-25: Dequantize a block-scaled FP8 weight to BF16, without re-quantizing.
#[allow(dead_code)]
pub(crate) fn dense_from_fp8(
    store: &WeightStore,
    prefix: &str,
    gpu: &dyn GpuBackend,
) -> Result<DenseWeight> {
    dequant_fp8_blockscaled_to_bf16(store, prefix, gpu)
}

/// 2026-09-25: Load attention weights with q/k/v as the raw `weight_packed`
/// NVFP4 pointers (typed `DenseWeight`, not dequantized) and o_proj through `quantized_v2`.
#[allow(dead_code)]
pub(crate) fn load_attention_qwen35(
    store: &WeightStore,
    layer_prefix: &str,
    gpu: &dyn GpuBackend,
) -> Result<AttentionWeights> {
    let p = format!("{layer_prefix}.self_attn");
    let (k_scale, v_scale) = load_kv_scales(store, &p, gpu);
    Ok(AttentionWeights {
        q_proj: dense(store, &format!("{p}.q_proj.weight_packed"))?,
        k_proj: dense(store, &format!("{p}.k_proj.weight_packed"))?,
        v_proj: dense(store, &format!("{p}.v_proj.weight_packed"))?,
        o_proj: quantized_v2(store, &format!("{p}.o_proj"), gpu)?,
        q_norm: dense(store, &format!("{p}.q_norm.weight"))?,
        k_norm: dense(store, &format!("{p}.k_norm.weight"))?,
        q_norm_full: None,
        k_norm_full: None,
        k_scale,
        v_scale,
    })
}

/// 2026-09-25: `quantized_v2` under another name.
#[allow(dead_code)]
pub(crate) fn load_quantized_proj_qwen35(
    store: &WeightStore,
    prefix: &str,
    gpu: &dyn GpuBackend,
) -> Result<QuantizedWeight> {
    quantized_v2(store, prefix, gpu)
}

#[cfg(test)]
mod ep_detection_tests {
    use super::*;
    use metrale_config::ModelConfig;
    use metrale_model_weights::weights::WeightStore;

    /// 2026-09-25: A store whose tensors are all FP8E4M3 with shape `[1]` and NULL pointers.
    fn store_with(names: &[String]) -> WeightStore {
        use std::collections::HashMap;
        let map: HashMap<String, metrale_model_weights::weights::WeightTensor> = names
            .iter()
            .map(|n| {
                (
                    n.clone(),
                    metrale_model_weights::weights::WeightTensor {
                        ptr: metrale_gpu_runtime::gpu::DevicePtr::NULL,
                        shape: vec![1],
                        dtype: metrale_model_weights::weights::WeightDtype::FP8E4M3,
                    },
                )
            })
            .collect();
        WeightStore::from_map(map)
    }

    #[test]
    fn alternate_layer0_fp8_dtype_is_detected_on_every_ep_rank() {
        let mut cfg = ModelConfig::qwen3_next_80b_nvfp4();
        cfg.quantization_config = None;
        let store =
            store_with(&["model.language_model.layers.0.self_attn.q_proj.weight".to_string()]);

        cfg.ep_world_size = 2;
        for ep_rank in 0..2 {
            cfg.ep_rank = ep_rank;
            assert_eq!(
                detect_nvfp4_variant(&store, &cfg),
                Nvfp4Variant::Fp8Dequanted,
                "EP rank {ep_rank} must inspect the same layer-zero checkpoint marker"
            );
        }
    }

    #[test]
    fn scale_inv_suffix_fallback_detects_an_unexpected_prefix() {
        let mut cfg = ModelConfig::qwen3_next_80b_nvfp4();
        cfg.quantization_config = None;
        let store =
            store_with(&["third_party.transformer.blocks.17.attn.q.weight_scale_inv".to_string()]);
        assert_eq!(
            detect_nvfp4_variant(&store, &cfg),
            Nvfp4Variant::Fp8Dequanted
        );
    }
}
