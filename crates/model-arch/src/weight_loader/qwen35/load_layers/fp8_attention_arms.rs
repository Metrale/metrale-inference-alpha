// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The two full-attention arms of `load_layers` that keep Q/K/V/O out of NVFP4:
//! BF16 dense attention and native FP8 attention.
//!
//! Owner: model-arch weight loader (Qwen3.5).
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_model_layers::layers::Qwen3AttentionLayer;
use metrale_model_layers::weight_map::quant_helpers::dense_auto;
use metrale_model_layers::weight_map::{
    AttentionWeights, DenseWeight, QuantizedWeight, load_fp8_block_scaled_as_fp8weight,
    load_kv_scales,
};

use super::load_cx::{LayerIn, LoadCx};
use crate::tp_shard::{TpShardKind, load_qkvo_tp, shard_fp8_block_scaled};

/// 2026-09-26: Layer `i`'s BF16 dense attention layer, the `attn_idx`-th attention layer.
pub(super) fn build_bf16_dense_attention(
    cx: &LoadCx,
    lp: &str,
    i: usize,
    attn_idx: usize,
    parts: LayerIn,
) -> Result<Qwen3AttentionLayer> {
    let LoadCx {
        store,
        config,
        gpu,
        layer_kv_dtypes,
        modelopt_mixed_precision,
        ..
    } = *cx;
    let LayerIn {
        input_norm,
        post_attn_norm,
        ffn,
    } = parts;
    // 2026-09-25: BF16 dense attention, TP=1 only: FP8 Q/K/V/O dequantized to BF16
    // (`METRALE_FP8_DEQUANT_ATTN_TO_BF16`), or the Holo ModelOpt checkpoint's attention
    // loaded with `dense_auto`. No quantized Q/K/V is passed, and O is installed with
    // `set_o_dense_bf16`.
    if config.tp_world_size.max(1) != 1 {
        anyhow::bail!(
            "BF16-dequant attention supports TP=1 only (got tp={})",
            config.tp_world_size,
        );
    }
    let p = format!("{lp}.self_attn");
    tracing::info!(
        target: "metrale_model_arch::weight_loader::qwen35::load_layers",
        "Layer {i}: dequanting attention Q/K/V/O FP8→BF16 (dense)"
    );
    let load_fp8_dense = |name: &str| -> Result<DenseWeight> {
        if modelopt_mixed_precision {
            dense_auto(store, &format!("{p}.{name}.weight"), gpu)
        } else {
            metrale_model_layers::weight_map::quant_helpers::dequant_fp8_blockscaled_to_bf16(
                store,
                &format!("{p}.{name}"),
                gpu,
            )
        }
    };
    let q_bf16 = load_fp8_dense("q_proj")?;
    let k_bf16 = load_fp8_dense("k_proj")?;
    let v_bf16 = load_fp8_dense("v_proj")?;
    let o_bf16 = load_fp8_dense("o_proj")?;

    let (k_scale, v_scale) = load_kv_scales(store, &p, gpu);
    let dummy_qw = QuantizedWeight::null();
    let attn = AttentionWeights {
        q_proj: q_bf16,
        k_proj: k_bf16,
        v_proj: v_bf16,
        o_proj: dummy_qw,
        q_norm: dense_auto(store, &format!("{p}.q_norm.weight"), gpu)?,
        k_norm: dense_auto(store, &format!("{p}.k_norm.weight"), gpu)?,
        q_norm_full: None,
        k_norm_full: None,
        k_scale,
        v_scale,
    };
    let layer_kv_dtype = layer_kv_dtypes[attn_idx];
    let mut layer = Qwen3AttentionLayer::new(
        input_norm,
        attn,
        post_attn_norm,
        ffn,
        attn_idx,
        None,
        None,
        None,
        gpu,
        layer_kv_dtype,
        config.fp8_kv_calibration_tokens,
        config,
    )?;
    layer.set_o_dense_bf16(o_bf16);
    Ok(layer)
}

/// 2026-09-26: Layer `i`'s native FP8 attention layer, the `attn_idx`-th attention layer.
pub(super) fn build_native_fp8_attention(
    cx: &LoadCx,
    lp: &str,
    i: usize,
    attn_idx: usize,
    parts: LayerIn,
) -> Result<Qwen3AttentionLayer> {
    let LoadCx {
        store,
        config,
        gpu,
        layer_kv_dtypes,
        stream,
        ..
    } = *cx;
    let LayerIn {
        input_norm,
        post_attn_norm,
        ffn,
    } = parts;
    // 2026-09-25: Native FP8 attention: the checkpoint's block-scaled FP8 Q/K/V/O
    // (scales widened to FP32 by `load_fp8_block_scaled_as_fp8weight`) are installed
    // with `set_fp8_weights`, plus transposed copies for prefill. No NVFP4 copy is
    // built.
    let p = format!("{lp}.self_attn");
    tracing::info!(
        target: "metrale_model_arch::weight_loader::qwen35::load_layers",
        "Layer {i}: loading attention FP8 native (zero-copy)"
    );

    let tp_rank = config.tp_rank;
    let tp_size = config.tp_world_size.max(1);
    let block_size = 128usize;
    let load_fp8_proj = |name: &str,
                         _full_n: usize,
                         _full_k: usize,
                         kind: TpShardKind|
     -> Result<metrale_model_layers::weight_map::Fp8Weight> {
        let src = load_fp8_block_scaled_as_fp8weight(store, &format!("{p}.{name}"), gpu)?;
        if tp_size == 1 {
            return Ok(src);
        }
        let sharded = shard_fp8_block_scaled(&src, kind, tp_rank, tp_size, block_size, gpu)?;
        gpu.free(src.weight)?;
        gpu.free(src.row_scale)?;
        Ok(sharded)
    };
    let [q_fp8, k_fp8, v_fp8, o_fp8] = load_qkvo_tp(config, load_fp8_proj)?;
    tracing::info!(
        target: "metrale_model_arch::weight_loader::qwen35::load_layers",
        "Layer {i}: FP8 Q/K/V/O loaded, {:.1} GB free",
        gpu.free_memory()? as f64 / (1024.0 * 1024.0 * 1024.0)
    );

    // 2026-09-25: Q/K/V/O here are NULL placeholders; the projections run from the FP8
    // weights installed by `set_fp8_weights` below.
    let (k_scale, v_scale) = load_kv_scales(store, &p, gpu);
    let dummy = DenseWeight {
        weight: metrale_gpu_runtime::gpu::DevicePtr::NULL,
    };
    let dummy_qw = QuantizedWeight::null();
    let attn = AttentionWeights {
        q_proj: dummy,
        k_proj: dummy,
        v_proj: dummy,
        o_proj: dummy_qw,
        q_norm: dense_auto(store, &format!("{p}.q_norm.weight"), gpu)?,
        k_norm: dense_auto(store, &format!("{p}.k_norm.weight"), gpu)?,
        q_norm_full: None,
        k_norm_full: None,
        k_scale,
        v_scale,
    };

    let layer_kv_dtype = layer_kv_dtypes[attn_idx];
    let mut layer = Qwen3AttentionLayer::new(
        input_norm,
        attn,
        post_attn_norm,
        ffn,
        attn_idx,
        None,
        None,
        None,
        gpu,
        layer_kv_dtype,
        config.fp8_kv_calibration_tokens,
        config,
    )?;

    layer.set_fp8_weights(Some(q_fp8), Some(k_fp8), Some(v_fp8), Some(o_fp8));

    if let Err(e) = layer.transpose_fp8_for_prefill(gpu, stream) {
        tracing::warn!(
            target: "metrale_model_arch::weight_loader::qwen35::load_layers",
            "Layer {i}: FP8 transpose failed, using non-transposed prefill: {e}"
        );
    } else {
        tracing::info!(
            target: "metrale_model_arch::weight_loader::qwen35::load_layers",
            "Layer {i}: FP8 weights transposed for fast prefill"
        );
    }

    Ok(layer)
}
