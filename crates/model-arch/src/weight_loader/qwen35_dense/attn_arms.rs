// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The attention weight builders `attn_layer::load_full_attention` dispatches
//! to: the keep-packed Q2_0 layer, and one arm per NVFP4 variant, each returning the
//! layer's [`AttentionWeights`] with its NVFP4 q/k/v (when built).
//!
//! Owner: model-arch weight loader (Qwen3.5 dense).
//! Invariants:
//! - An arm that returns a NULL `o_proj` leaves it to a later install: the FP8 overlay
//!   (`nvfp4_skipped_arm`) or the BF16 O-proj (`bf16_raw_arm`).

use anyhow::Result;
use metrale_model_layers::layers::Qwen3AttentionLayer;
use metrale_model_layers::weight_map::quant_helpers::dense_auto;
use metrale_model_layers::weight_map::{
    AttentionWeights, DenseWeight, Nvfp4Variant, QuantizedWeight, dense, load_kv_scales,
    quantize_to_nvfp4, quantized_auto,
};
use metrale_model_weights::weights::WeightDtype;

use super::fp8_residency::{self, DerivedResidency};
use super::load_cx::{LayerIn, LoadCx};
use super::packed_q2_from_store;
use crate::tp_shard::{TpShardKind, load_qkvo_tp, shard_dense_bf16, shard_quantized_nvfp4};

/// 2026-09-26: An attention arm's result: the layer's weights and its NVFP4 q, k and v.
pub(super) type AttnParts = (
    AttentionWeights,
    Option<QuantizedWeight>,
    Option<QuantizedWeight>,
    Option<QuantizedWeight>,
);

/// 2026-09-26: The keep-packed Q2_0 attention layer (`attn_q2`): NULL q/k/v/o, then the
/// packed q/k/v/o installed with `set_packed_q2_weights`.
pub(super) fn build_q2_attention(
    cx: &LoadCx<'_>,
    p: &str,
    attn_idx: usize,
    l: LayerIn<'_>,
) -> Result<Qwen3AttentionLayer> {
    let LoadCx {
        store,
        config,
        gpu,
        layer_kv_dtypes,
        ..
    } = *cx;
    let LayerIn {
        lp,
        input_norm,
        post_attn_norm,
        ffn,
        ..
    } = l;
    let (k_scale, v_scale) = load_kv_scales(store, p, gpu);
    let attn = AttentionWeights {
        q_proj: DenseWeight {
            weight: metrale_gpu_runtime::gpu::DevicePtr::NULL,
        },
        k_proj: DenseWeight {
            weight: metrale_gpu_runtime::gpu::DevicePtr::NULL,
        },
        v_proj: DenseWeight {
            weight: metrale_gpu_runtime::gpu::DevicePtr::NULL,
        },
        o_proj: metrale_model_layers::weight_map::QuantizedWeight::null(),
        q_norm: dense(store, &format!("{p}.q_norm.weight"))?,
        k_norm: dense(store, &format!("{p}.k_norm.weight"))?,
        q_norm_full: None,
        k_norm_full: None,
        k_scale,
        v_scale,
    };
    let mut attn_layer = Qwen3AttentionLayer::new(
        input_norm,
        attn,
        post_attn_norm,
        ffn,
        attn_idx,
        None,
        None,
        None,
        gpu,
        layer_kv_dtypes[attn_idx],
        config.fp8_kv_calibration_tokens,
        config,
    )?;
    attn_layer.set_packed_q2_weights(
        packed_q2_from_store(store, &format!("{p}.q_proj"))?,
        packed_q2_from_store(store, &format!("{p}.k_proj"))?,
        packed_q2_from_store(store, &format!("{p}.v_proj"))?,
        packed_q2_from_store(store, &format!("{p}.o_proj"))?,
        gpu,
    );
    tracing::info!(target: "metrale_model_arch::weight_loader::qwen35_dense", "ATTN[{lp}] native keep-packed Q2_0: q/k/v/o 2-bit \
         (q2_0_gemv_vec decode; transient-dequant prefill)"
    );
    Ok(attn_layer)
}

/// 2026-09-26: `Nvfp4Variant::CompressedTensors`: NVFP4 q/k/v/o from disk, sharded for
/// this rank.
pub(super) fn compressed_tensors_arm(
    cx: &LoadCx<'_>,
    p: &str,
    tp_rank: usize,
    tp_size: usize,
) -> Result<AttnParts> {
    let LoadCx {
        store,
        config,
        gpu,
        variant,
        absmax_k,
        quantize_k,
        stream,
        ..
    } = *cx;
    // 2026-09-25: NVFP4 from disk: Q/K/V column-parallel, O row-parallel
    // (`load_qkvo_tp`).
    let group_size = 16usize;
    let load_nvfp4 = |name: &str,
                      full_n: usize,
                      full_k: usize,
                      kind: TpShardKind|
     -> Result<metrale_model_layers::weight_map::QuantizedWeight> {
        let prefix = format!("{p}.{name}");
        // 2026-09-25: A projection without `weight_packed` is loaded with
        // `dense_auto` and quantized to NVFP4 here, as in
        // `qwen35/load_layers/attention_arms.rs`.
        let src = if store.contains(&format!("{prefix}.weight_packed")) {
            quantized_auto(store, &prefix, gpu, variant)?
        } else {
            let dense_bf16 = dense_auto(store, &format!("{prefix}.weight"), gpu)?;
            quantize_to_nvfp4(
                &dense_bf16,
                full_n,
                full_k,
                gpu,
                absmax_k,
                quantize_k,
                stream,
            )?
        };
        if tp_size == 1 {
            return Ok(src);
        }
        let sharded = shard_quantized_nvfp4(
            &src, full_n, full_k, kind, tp_rank, tp_size, group_size, gpu,
        )?;
        gpu.free(src.weight)?;
        gpu.free(src.weight_scale)?;
        Ok(sharded)
    };
    let [q, k, v, o] = load_qkvo_tp(config, load_nvfp4)?;
    let dummy = DenseWeight {
        weight: metrale_gpu_runtime::gpu::DevicePtr::NULL,
    };
    let (k_scale, v_scale) = load_kv_scales(store, p, gpu);
    let attn = AttentionWeights {
        q_proj: dummy,
        k_proj: dummy,
        v_proj: dummy,
        o_proj: o,
        q_norm: dense(store, &format!("{p}.q_norm.weight"))?,
        k_norm: dense(store, &format!("{p}.k_norm.weight"))?,
        q_norm_full: None,
        k_norm_full: None,
        k_scale,
        v_scale,
    };
    Ok((attn, Some(q), Some(k), Some(v)))
}

/// 2026-09-26: `Standard`/`Fp8Dequanted` with `attn_nvfp4` false: NULL q/k/v/o, which the
/// native FP8 overlay replaces.
pub(super) fn nvfp4_skipped_arm(
    cx: &LoadCx<'_>,
    residency: &mut DerivedResidency,
    p: &str,
    attn_fp8: bool,
) -> Result<AttnParts> {
    let LoadCx {
        store, config, gpu, ..
    } = *cx;
    // 2026-09-25: `attn_nvfp4` is false: the FP8 overlay below replaces
    // q/k/v/o and `RouteEnv::attn_nvfp4` found no route that reads NVFP4
    // copies, so none are built.
    let (k_scale, v_scale) = load_kv_scales(store, p, gpu);
    let (nh, hd) = (config.num_attention_heads, config.head_dim);
    let (nkv, hh) = (config.num_key_value_heads, config.hidden_size);
    let q_n = nh * hd * if config.attn_gated { 2 } else { 1 };
    residency.skip(fp8_residency::attn_nvfp4_bytes(q_n, nkv * hd, nh * hd, hh));
    let null = || DenseWeight {
        weight: metrale_gpu_runtime::gpu::DevicePtr::NULL,
    };
    let attn = AttentionWeights {
        q_proj: null(),
        k_proj: null(),
        v_proj: null(),
        o_proj: metrale_model_layers::weight_map::QuantizedWeight::null(),
        q_norm: dense(store, &format!("{p}.q_norm.weight"))?,
        k_norm: dense(store, &format!("{p}.k_norm.weight"))?,
        q_norm_full: None,
        k_norm_full: None,
        k_scale,
        v_scale,
    };
    // 2026-09-25: The NULL `o_proj` relies on the FP8 overlay installed
    // below.
    debug_assert!(attn_fp8, "attn_nvfp4=false must imply a native FP8 overlay");
    Ok((attn, None, None, None))
}

/// 2026-09-26: `Standard`/`Fp8Dequanted` with `attn_nvfp4` true: each projection is
/// sharded as BF16 and this rank's part quantized to NVFP4.
pub(super) fn bf16_then_nvfp4_arm(
    cx: &LoadCx<'_>,
    p: &str,
    tp_rank: usize,
    tp_size: usize,
) -> Result<AttnParts> {
    let LoadCx {
        store,
        config,
        gpu,
        absmax_k,
        quantize_k,
        stream,
        ..
    } = *cx;
    // 2026-09-25: Shard the BF16 projection, then quantize this rank's part
    // to NVFP4.
    let load_bf16_then_nvfp4 = |name: &str,
                                full_n: usize,
                                full_k: usize,
                                kind: TpShardKind|
     -> Result<(
        DenseWeight,
        metrale_model_layers::weight_map::QuantizedWeight,
    )> {
        // 2026-09-25: A UInt8 `.weight` is pre-quantized NVFP4: it loads as
        // it is and is not sharded, so TP must be 1.
        let weight_key = format!("{p}.{name}.weight");
        if matches!(
            store.get(&weight_key).map(|w| w.dtype),
            Ok(WeightDtype::UInt8)
        ) {
            anyhow::ensure!(
                tp_size == 1,
                "pre-quantized NVFP4 weight '{weight_key}' (U8 on disk) \
                 cannot be loaded under tensor parallelism (tp_size={tp_size}): \
                 TP sharding of pre-quantized NVFP4 checkpoints is not yet \
                 implemented. Use tp_size=1, or dequantize this checkpoint to \
                 BF16 first so it goes through the shard-then-requantize path."
            );
            let null_dense = DenseWeight {
                weight: metrale_gpu_runtime::gpu::DevicePtr::NULL,
            };
            let qw = quantized_auto(store, &format!("{p}.{name}"), gpu, Nvfp4Variant::Standard)?;
            return Ok((null_dense, qw));
        }
        let src = dense_auto(store, &weight_key, gpu)?;
        let (sharded_ptr, local_n, local_k) =
            shard_dense_bf16(src.weight, full_n, full_k, kind, tp_rank, tp_size, gpu)?;
        let sharded = DenseWeight {
            weight: sharded_ptr,
        };
        let q = quantize_to_nvfp4(
            &sharded, local_n, local_k, gpu, absmax_k, quantize_k, stream,
        )?;
        if sharded_ptr != src.weight {
            gpu.free(sharded_ptr)?;
        }
        Ok((src, q))
    };
    let [
        (q_dense, q_nvfp4),
        (k_dense, k_nvfp4),
        (v_dense, v_nvfp4),
        (o_dense, o_nvfp4),
    ] = load_qkvo_tp(config, load_bf16_then_nvfp4)?;

    let (k_scale, v_scale) = load_kv_scales(store, p, gpu);

    // 2026-09-25: The BF16 q/k/v/o were only the quantizer's input and the
    // layer gets no BF16 attention weights, so they are freed. They are
    // fresh buffers for an FP8 source; for a BF16 source `dense_auto`
    // returns the store's own pointer, and for a UInt8 source they are
    // NULL.
    gpu.free(q_dense.weight)?;
    gpu.free(k_dense.weight)?;
    gpu.free(v_dense.weight)?;
    gpu.free(o_dense.weight)?;

    let attn = AttentionWeights {
        q_proj: DenseWeight {
            weight: metrale_gpu_runtime::gpu::DevicePtr::NULL,
        },
        k_proj: DenseWeight {
            weight: metrale_gpu_runtime::gpu::DevicePtr::NULL,
        },
        v_proj: DenseWeight {
            weight: metrale_gpu_runtime::gpu::DevicePtr::NULL,
        },
        o_proj: o_nvfp4,
        q_norm: dense(store, &format!("{p}.q_norm.weight"))?,
        k_norm: dense(store, &format!("{p}.k_norm.weight"))?,
        q_norm_full: None,
        k_norm_full: None,
        k_scale,
        v_scale,
    };
    Ok((attn, Some(q_nvfp4), Some(k_nvfp4), Some(v_nvfp4)))
}

/// 2026-09-26: `Nvfp4Variant::Bf16Raw`: BF16 q/k/v in the weights; the BF16 O goes to
/// `o_dense_bf16` for `set_o_dense_bf16`.
pub(super) fn bf16_raw_arm(
    cx: &LoadCx<'_>,
    p: &str,
    tp_rank: usize,
    tp_size: usize,
    o_dense_bf16: &mut Option<DenseWeight>,
) -> Result<AttnParts> {
    let LoadCx {
        store, config, gpu, ..
    } = *cx;
    // 2026-09-25: Bf16Raw: Q/K/V/O stay BF16 and no NVFP4 copy is built; O
    // is installed after the layer is built (`set_o_dense_bf16`).
    let load_bf16_dense =
        |name: &str, full_n: usize, full_k: usize, kind: TpShardKind| -> Result<DenseWeight> {
            let src = dense_auto(store, &format!("{p}.{name}.weight"), gpu)?;
            if tp_size == 1 {
                return Ok(src);
            }
            let (sharded_ptr, _local_n, _local_k) =
                shard_dense_bf16(src.weight, full_n, full_k, kind, tp_rank, tp_size, gpu)?;
            if sharded_ptr != src.weight {
                gpu.free(src.weight)?;
            }
            Ok(DenseWeight {
                weight: sharded_ptr,
            })
        };
    let [q_dense, k_dense, v_dense, o_dense] = load_qkvo_tp(config, load_bf16_dense)?;

    let (k_scale, v_scale) = load_kv_scales(store, p, gpu);

    let attn = AttentionWeights {
        q_proj: q_dense,
        k_proj: k_dense,
        v_proj: v_dense,
        o_proj: metrale_model_layers::weight_map::QuantizedWeight::null(),
        q_norm: dense(store, &format!("{p}.q_norm.weight"))?,
        k_norm: dense(store, &format!("{p}.k_norm.weight"))?,
        q_norm_full: None,
        k_norm_full: None,
        k_scale,
        v_scale,
    };
    *o_dense_bf16 = Some(o_dense);
    // 2026-09-25: The NULL `o_proj` relies on the BF16 O installed after
    // the layer is built (`set_o_dense_bf16`).
    debug_assert!(
        o_dense_bf16.is_some(),
        "Bf16Raw attention must install a dense O-proj to cover the null o_proj"
    );
    Ok((attn, None, None, None))
}
