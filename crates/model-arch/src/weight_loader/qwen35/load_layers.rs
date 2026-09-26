// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `load_layers` for the Qwen3.5 MoE loader: resolves the checkpoint's weight
//! format and the `METRALE_*` loader levers, then builds each full-attention and
//! linear-attention layer with its MoE block.
//!
//! Owner: model-arch weight loader (Qwen3.5).
//! Invariants: none beyond the types.

pub(crate) mod attention_arms;
mod fp8_attention_arms;
pub(crate) mod linear_attn_arms;
mod linear_attn_route;
mod load_cx;
mod moe_experts;
mod selectors;
mod tq_plus_weight_rotation;

use anyhow::Result;
use metrale_cache::kv_cache::KvCacheDtype;
use metrale_config::{LayerType, ModelConfig};
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_model_weights::weights::{WeightDtype, WeightStore};

use super::super::{ModelWeightLoader, QuantFormat, WeightFormat};
use load_cx::{LayerIn, LoadCx};
use metrale_model_layers::layer::TransformerLayer;
use metrale_model_layers::layers::{FfnComponent, MoeLayer};
use metrale_model_layers::weight_map::quant_helpers::dense_auto;
use metrale_model_layers::weight_map::{
    Nvfp4Variant, detect_nvfp4_variant, load_moe_qwen35, quantize_to_nvfp4,
};
use selectors::{
    HoloFastMoeMode, holo_fast_moe_layer_selected, holo_fast_moe_mode, holo_moe_down_fp4,
    holo_moe_gateup_fp4, is_holo_modelopt_mixed_precision, layer_dequant_selected,
};

/// 2026-09-25: True when `{prefix}.weight` is FP8E4M3 with a block scale: a
/// `.weight_scale_inv`, or a 2D `.weight_scale`. A scalar `.weight_scale` (per-tensor FP8)
/// returns false.
fn proj_is_native_fp8(store: &WeightStore, prefix: &str) -> bool {
    let is_fp8_weight = store
        .get(&format!("{prefix}.weight"))
        .map(|w| w.dtype == WeightDtype::FP8E4M3)
        .unwrap_or(false);
    let has_block_scale = store.contains(&format!("{prefix}.weight_scale_inv"))
        || store
            .get(&format!("{prefix}.weight_scale"))
            .map(|s| s.shape.len() == 2)
            .unwrap_or(false);
    is_fp8_weight && has_block_scale
}

pub(super) fn load_layers(
    loader: &dyn ModelWeightLoader,
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    layer_kv_dtypes: &[KvCacheDtype],
) -> Result<Vec<Box<dyn TransformerLayer>>> {
    let layer_types = if config.layer_types.is_empty() {
        (0..config.num_hidden_layers)
            .map(|i| config.layer_type(i))
            .collect::<Vec<_>>()
    } else {
        config.layer_types.clone()
    };

    let mut layers: Vec<Box<dyn TransformerLayer>> = Vec::with_capacity(config.num_hidden_layers);
    let mut attn_idx = 0usize;

    // 2026-09-25: The loader's precision schedule is logged when it overrides anything;
    // nothing in this function reads it further.
    let precision = loader.precision_schedule(config);
    if precision.has_any_override() {
        tracing::info!(
            "Precision schedule active: {:?} — overriding per-checkpoint dtype",
            precision,
        );
    }
    let _ = precision;

    let absmax_k = gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax")?;
    let quantize_k = gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4")?;
    let stream = gpu.default_stream();
    let h = config.hidden_size;

    let variant = detect_nvfp4_variant(store, config);
    let weight_format = WeightFormat::detect(store, config);

    // 2026-09-25: `quant_format` is FP8 only for the `Fp8Dequanted` variant (an FP8
    // checkpoint), NVFP4 otherwise.
    let modelopt_mixed_precision = is_holo_modelopt_mixed_precision(config);
    let quant_format = if variant == Nvfp4Variant::Fp8Dequanted {
        QuantFormat::Fp8
    } else {
        QuantFormat::Nvfp4
    };
    let native_fp8 = quant_format == QuantFormat::Fp8;
    // 2026-09-25: The fast-MoE prefill copies qualify by architecture (any MoE served as
    // NVFP4) or for the Holo ModelOpt mixed-precision checkpoint. They are on by default:
    // unless `METRALE_HOLO_LOW_MEMORY_MOE=0`, the low-memory expert layout and a fast-MoE
    // mode are enabled together. The layout is enabled only when a mode resolved
    // (`low_memory_modelopt_moe`): without one it would skip every layer's MoE prefill
    // copies (`skip_moe_prefill_copies`).
    let nvfp4_moe = config.num_experts > 0 && quant_format == QuantFormat::Nvfp4;
    let moe_qualifies = modelopt_mixed_precision || nvfp4_moe;
    let low_memory_requested =
        moe_qualifies && std::env::var("METRALE_HOLO_LOW_MEMORY_MOE").ok().as_deref() != Some("0");
    // 2026-09-25: Unset selects `Full`. A set value goes through the parser, so a typo
    // warns and resolves no mode.
    let holo_fast_moe_mode = if low_memory_requested {
        if std::env::var_os("METRALE_HOLO_FAST_MOE_MODE").is_some() {
            holo_fast_moe_mode()
        } else {
            Some(HoloFastMoeMode::Full)
        }
    } else {
        None
    };
    let low_memory_modelopt_moe = low_memory_requested && holo_fast_moe_mode.is_some();
    let holo_fast_moe_spec = if low_memory_modelopt_moe {
        // 2026-09-25: Unset selects every layer: `0-99999` is an inclusive range
        // (`parse_layer_ranges`).
        Some(
            std::env::var("METRALE_HOLO_FAST_MOE_LAYERS").unwrap_or_else(|_| "0-99999".to_string()),
        )
    } else {
        None
    };
    let native_modelopt_ssm = modelopt_mixed_precision
        && std::env::var("METRALE_HOLO_NATIVE_FP8_SSM").ok().as_deref() == Some("1");
    let native_modelopt_attn = modelopt_mixed_precision
        && std::env::var("METRALE_HOLO_NATIVE_FP8_ATTN")
            .ok()
            .as_deref()
            == Some("1");
    tracing::info!(
        "Weight format: {:?}, NVFP4 variant: {:?}, quant_format: {:?}",
        weight_format,
        variant,
        quant_format,
    );

    // 2026-09-25: Bytes of the transposed MoE prefill tables: gate, up and down (packed FP4
    // plus one scale byte per 16 values) for every expert of every layer. When they exceed
    // free memory less 2 GiB, the transpose is skipped except on selected fast-MoE layers.
    let skip_moe_transpose = moe_experts::moe_transpose_skipped(config, gpu, h);
    if low_memory_modelopt_moe {
        if let (Some(mode), Some(spec)) = (holo_fast_moe_mode, holo_fast_moe_spec.as_deref()) {
            tracing::info!(
                "METRALE_HOLO_LOW_MEMORY_MOE=1: enabling Holo ModelOpt MoE {:?} prefill copies for layers {spec}",
                mode,
            );
        } else {
            tracing::info!(
                "METRALE_HOLO_LOW_MEMORY_MOE=1: skipping Holo ModelOpt MoE transpose/predequant prefill copies"
            );
        }
    }
    if native_modelopt_ssm {
        tracing::info!(
            "METRALE_HOLO_NATIVE_FP8_SSM=1: routing Holo ModelOpt SSM projections through native FP8"
        );
    }
    if native_modelopt_attn {
        tracing::info!(
            "METRALE_HOLO_NATIVE_FP8_ATTN=1: routing Holo ModelOpt attention projections through native FP8"
        );
    }

    let cx = LoadCx {
        store,
        config,
        gpu,
        layer_kv_dtypes,
        variant,
        h,
        absmax_k,
        quantize_k,
        stream,
        modelopt_mixed_precision,
        native_modelopt_ssm,
    };
    for (i, lt) in layer_types.iter().enumerate() {
        let lp = config.layer_prefix(i);
        let input_norm = dense_auto(store, &format!("{lp}.input_layernorm.weight"), gpu)?;
        let post_attn_norm =
            dense_auto(store, &format!("{lp}.post_attention_layernorm.weight"), gpu)?;

        // 2026-09-25: A native-FP8 checkpoint skips the NVFP4 routed experts unless
        // `METRALE_FORCE_NVFP4_MOE=1` (keeps them and skips the FP8 experts) or the experts are
        // fused. `METRALE_FORCE_NVFP4_ALL=1` implies it and also skips the native-FP8
        // attention arm, the FP8-dequant BF16 attention arm and the `Fp8Dequanted`
        // native-FP8 SSM arm.
        let force_nvfp4_all = std::env::var("METRALE_FORCE_NVFP4_ALL").ok().as_deref() == Some("1");
        // 2026-09-25: `METRALE_HOLO_FP4_PROJ_DECODE=1` sends the full-attention and SSM
        // projections to the NVFP4 builders while the MoE experts keep their native-FP8 path
        // (it does not set `force_nvfp4_moe`). It does not override
        // `METRALE_HOLO_NATIVE_FP8_SSM=1`.
        let fp4_proj_decode = std::env::var("METRALE_HOLO_FP4_PROJ_DECODE")
            .ok()
            .as_deref()
            == Some("1");
        let force_nvfp4_moe = force_nvfp4_all
            || std::env::var("METRALE_FORCE_NVFP4_MOE").ok().as_deref() == Some("1");
        // 2026-09-25: Routed experts in the fused layout (`mlp.experts.gate_up_proj`) take the
        // NVFP4 expert path even for an FP8 checkpoint: `load_moe_qwen35` slices the fused
        // tensors, and `load_moe_qwen35_fp8_experts` reads only per-expert tensors.
        let fused_experts = store.contains(&format!("{lp}.mlp.experts.gate_up_proj"));
        let skip_nvfp4_experts = native_fp8 && !force_nvfp4_moe && !fused_experts;
        if skip_nvfp4_experts {
            tracing::info!(
                "FP8: skipping NVFP4 routed experts (FP8 fused MoE batch1/2/3 handles all dispatch)"
            );
        } else if native_fp8 && fused_experts {
            tracing::info!(
                "FP8: routed experts use FUSED layout — loading via NVFP4 expert path (dequant→NVFP4)"
            );
        } else if native_fp8 && force_nvfp4_moe {
            tracing::warn!(
                "METRALE_FORCE_NVFP4_MOE=1: routing MoE through NVFP4 path (diagnostic — slower)"
            );
        }
        let moe_weights = load_moe_qwen35(
            store,
            &lp,
            config.num_experts,
            gpu,
            config,
            variant,
            absmax_k,
            quantize_k,
            stream,
            skip_nvfp4_experts,
        )?;
        // 2026-09-25: For `native_fp8` the router gate stays BF16 (`gate_nvfp4` is None) and
        // the gate GEMM runs dense (`moe/forward_batched_gate.rs`): a 4-bit E2M1 copy is too
        // coarse for the router. Measured 2026-05-25: at late layers the top-8 routing weights
        // sit within [0.105, 0.168], a range narrower than one NVFP4 step. Every other variant
        // quantizes the gate to NVFP4.
        let gate_nvfp4 = if native_fp8 {
            None
        } else {
            Some(quantize_to_nvfp4(
                &moe_weights.gate,
                config.num_experts,
                h,
                gpu,
                absmax_k,
                quantize_k,
                stream,
            )?)
        };
        let mut moe_layer =
            MoeLayer::new(moe_weights, config.num_experts, gate_nvfp4, gpu, config)?;
        // 2026-09-25: `config.dflash_capture_layers` is the drafter's `target_layer_ids`, used as
        // they are (model-engine `factory/build.rs`). On those layers
        // `METRALE_FRANKENSTEIN_DECODE_VIA_PREFILL=1` runs MoE decode through `forward_prefill`
        // (`moe/forward.rs`).
        moe_layer.is_dflash_capture_layer = config.dflash_capture_layers.contains(&i);
        if i == 0 && (holo_moe_gateup_fp4() || holo_moe_down_fp4()) && holo_fast_moe_mode.is_none()
        {
            tracing::warn!(
                "METRALE_HOLO_MOE_GATEUP_FP4/_DOWN_FP4 set but METRALE_HOLO_FAST_MOE_MODE \
                 is not full: the FP4 MoE prefill path needs the shared [K/2,N] tables \
                 and will be IGNORED (FP8 fused path used instead)."
            );
        }
        // 2026-09-25: The NVFP4 MoE prefill copies (transpose, predequant) are built unless the
        // checkpoint is native FP8 without `METRALE_FORCE_NVFP4_MOE`, or the layer is a
        // low-memory layer not selected for fast-MoE copies.
        let fast_holo_moe_layer = low_memory_modelopt_moe
            && holo_fast_moe_mode.is_some()
            && holo_fast_moe_spec
                .as_deref()
                .is_some_and(|spec| holo_fast_moe_layer_selected(spec, i));
        let skip_moe_prefill_copies = low_memory_modelopt_moe && !fast_holo_moe_layer;
        if fast_holo_moe_layer {
            tracing::info!(
                "Layer {i}: selected for Holo ModelOpt {:?} MoE prefill copies",
                holo_fast_moe_mode.expect("checked is_some"),
            );
        }
        if (!native_fp8 || force_nvfp4_moe)
            && (!skip_moe_transpose || fast_holo_moe_layer)
            && !skip_moe_prefill_copies
        {
            match holo_fast_moe_mode {
                Some(HoloFastMoeMode::GateUp) if fast_holo_moe_layer => {
                    moe_layer.transpose_gate_up_for_prefill(gpu, config)?;
                }
                Some(HoloFastMoeMode::Unified) if fast_holo_moe_layer => {
                    moe_layer.transpose_for_prefill_unified(gpu, config)?;
                }
                _ => {
                    moe_layer.transpose_for_prefill(gpu, config)?;
                }
            }
        }
        if (!native_fp8 || force_nvfp4_moe) && !skip_moe_prefill_copies {
            moe_layer.predequant_for_prefill(gpu, config, stream)?;
        }
        // 2026-09-25: `METRALE_HOLO_MOE_GROUPED_CUTLASS=1` on a fast-MoE layer builds the swizzled
        // scale tables for the CUTLASS grouped NVFP4 gate_up (`build_cutlass_grouped_sfb`).
        if fast_holo_moe_layer
            && std::env::var("METRALE_HOLO_MOE_GROUPED_CUTLASS")
                .ok()
                .as_deref()
                == Some("1")
        {
            moe_layer.build_cutlass_grouped_sfb(gpu, config, stream)?;
        }

        // 2026-09-25: `METRALE_FP8_DEQUANT_MOE_TO_BF16=1` (native FP8 only) dequantizes the FP8
        // experts to BF16 at load and installs them with `set_bf16_experts`.
        // `METRALE_FP8_DEQUANT_LAYERS` limits this and the attention dequant below to the
        // layers it lists (`layer_dequant_selected`).
        let layer_sel = layer_dequant_selected(i);
        let dequant_moe_to_bf16 = native_fp8
            && std::env::var("METRALE_FP8_DEQUANT_MOE_TO_BF16")
                .ok()
                .as_deref()
                == Some("1")
            && layer_sel;
        // 2026-09-25: `METRALE_FP8_DEQUANT_ATTN_TO_BF16=1` (native FP8 only) selects the BF16
        // dense attention arm below.
        let dequant_attn_to_bf16 = native_fp8
            && std::env::var("METRALE_FP8_DEQUANT_ATTN_TO_BF16")
                .ok()
                .as_deref()
                == Some("1")
            && layer_sel;

        if dequant_moe_to_bf16 {
            moe_experts::install_bf16_dequant_experts(&cx, &lp, i, &mut moe_layer);
        }

        // 2026-09-25: Native-FP8 routed experts, for an FP8 checkpoint that is not forced to
        // NVFP4, not dequantized to BF16 and not in the fused layout. A load failure is
        // logged, and the MoE is left without FP8 experts.
        if native_fp8 && !force_nvfp4_moe && !dequant_moe_to_bf16 && !fused_experts {
            moe_experts::install_native_fp8_experts(&cx, &lp, i, &mut moe_layer);
        }

        let ffn = FfnComponent::Moe(moe_layer);

        match lt {
            LayerType::FullAttention
                if (native_fp8
                    && dequant_attn_to_bf16
                    && !(force_nvfp4_all || fp4_proj_decode)
                    && proj_is_native_fp8(store, &format!("{lp}.self_attn.q_proj")))
                    || (modelopt_mixed_precision && !native_modelopt_attn && !fp4_proj_decode) =>
            {
                let parts = LayerIn {
                    input_norm,
                    post_attn_norm,
                    ffn,
                };
                let layer =
                    fp8_attention_arms::build_bf16_dense_attention(&cx, &lp, i, attn_idx, parts)?;
                layers.push(Box::new(layer));
                attn_idx += 1;
            }
            LayerType::FullAttention
                if ((native_fp8
                    && proj_is_native_fp8(store, &format!("{lp}.self_attn.q_proj")))
                    || native_modelopt_attn)
                    && !(force_nvfp4_all || fp4_proj_decode) =>
            {
                let parts = LayerIn {
                    input_norm,
                    post_attn_norm,
                    ffn,
                };
                let layer =
                    fp8_attention_arms::build_native_fp8_attention(&cx, &lp, i, attn_idx, parts)?;
                layers.push(Box::new(layer));
                attn_idx += 1;
            }
            LayerType::FullAttention => {
                let layer = attention_arms::build_full_attention_nvfp4(
                    i,
                    store,
                    &lp,
                    gpu,
                    variant,
                    config,
                    h,
                    absmax_k,
                    quantize_k,
                    stream,
                    layer_kv_dtypes[attn_idx],
                    attn_idx,
                    input_norm,
                    post_attn_norm,
                    ffn,
                )?;
                layers.push(layer);
                attn_idx += 1;
            }
            LayerType::LinearAttention => {
                let parts = LayerIn {
                    input_norm,
                    post_attn_norm,
                    ffn,
                };
                let layer = linear_attn_route::build_linear_attention(
                    &cx,
                    &lp,
                    i,
                    force_nvfp4_all,
                    fp4_proj_decode,
                    parts,
                )?;
                layers.push(layer);
            }
            LayerType::SlidingAttention => {
                unreachable!("unexpected SlidingAttention in this loader")
            }
            LayerType::Moe => unreachable!("Qwen3.5 has no standalone MoE layers"),
            // 2026-09-25: GLM-5.3 `deepseek_sparse_attention` needs a DSA indexer and per-query
            // top-k, which this loader does not build; it is refused, not served as dense
            // attention.
            LayerType::SparseAttention => anyhow::bail!(
                "layer {i}: SparseAttention needs a DSA indexer and per-query top-k; Qwen3.5 has neither"
            ),
        }

        if (i + 1) % 10 == 0 || i < 5 {
            let free_gb = gpu.free_memory()? as f64 / (1024.0 * 1024.0 * 1024.0);
            tracing::info!("Loaded layers 0..{} — {free_gb:.1} GB free", i + 1);
            metrale_telemetry::progress::layer(i + 1, config.num_hidden_layers);
        }
    }

    tracing::info!(
        "Qwen3.5 weight loader: {} layers ({} attention, {} linear_attn)",
        layers.len(),
        attn_idx,
        layers.len() - attn_idx,
    );

    Ok(layers)
}
