// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Gemma-4 per-layer construction (`load_layers_impl`).
//!
//! Owner: model-arch weight loader.
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_cache::kv_cache::KvCacheDtype;
use metrale_config::ModelConfig;
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_model_weights::weights::WeightStore;

use super::loader_b::{build_bf16_mlp, build_moe_ffn};
use crate::tp_shard::{TpShardKind, shard_dense_bf16};
use metrale_model_layers::layer::TransformerLayer;
use metrale_model_layers::layers::dense_ffn::DenseFfnWeights;
use metrale_model_layers::layers::{
    DenseFfnLayer, FfnActivation, FfnComponent, Qwen3AttentionLayer,
};
use metrale_model_layers::weight_map::quant_helpers::dense_auto;
use metrale_model_layers::weight_map::{
    AttentionWeights, QuantizeCtx, dense, detect_nvfp4_variant, load_kv_scales, quantize_to_nvfp4,
    quantized_any,
};

/// 2026-09-25: Whether free memory, less 2 GiB, holds the transposed NVFP4 FFN
/// copies for every layer. `METRALE_GEMMA4_FFN_TRANSPOSE=0` answers no.
fn ffn_transpose_fits(config: &ModelConfig, gpu: &dyn GpuBackend) -> bool {
    if std::env::var("METRALE_GEMMA4_FFN_TRANSPOSE")
        .ok()
        .as_deref()
        == Some("0")
    {
        tracing::info!("METRALE_GEMMA4_FFN_TRANSPOSE=0: dense FFN prefill uses the w4a16 fallback");
        return false;
    }
    let h = config.hidden_size;
    let inter = config.intermediate_size;
    // 2026-09-25: NVFP4 is half a byte per weight plus a one-byte scale per 16.
    let per_weight = h * inter / 2 + h * inter / 16;
    let total = 3 * per_weight * config.num_hidden_layers;
    let available = gpu.free_memory().unwrap_or(0);
    let headroom = 2 * 1024 * 1024 * 1024;
    let fits = total <= available.saturating_sub(headroom);
    if !fits {
        tracing::warn!(
            "Skipping Gemma-4 FFN transposition ({:.1} GB needed, {:.1} GB available). \
             Prefill falls back to the untransposed w4a16 GEMM.",
            total as f64 / (1024.0 * 1024.0 * 1024.0),
            available as f64 / (1024.0 * 1024.0 * 1024.0),
        );
    }
    fits
}

pub(super) fn load_layers_impl(
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    layer_kv_dtypes: &[KvCacheDtype],
) -> Result<Vec<Box<dyn TransformerLayer>>> {
    let mut layers: Vec<Box<dyn TransformerLayer>> = Vec::with_capacity(config.num_hidden_layers);

    let variant = detect_nvfp4_variant(store, config);
    tracing::info!("Gemma-4 NVFP4 variant: {:?}", variant);

    // 2026-09-25: Decided once, before any layer allocates; see the
    // `moe_prefill_copies` parameter of `build_moe_ffn`.
    let moe_prefill_copies =
        config.num_experts > 0 && crate::weight_loader::moe_prefill_copies_fit(config, gpu);
    let absmax_k = gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax")?;
    let quantize_k = gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4")?;
    let stream = gpu.default_stream();
    let qctx = QuantizeCtx {
        absmax_k,
        quantize_k,
        stream,
    };
    let h = config.hidden_size;

    for i in 0..config.num_hidden_layers {
        let lp = config.layer_prefix(i);

        // 2026-09-25: `pre_feedforward_layernorm` is passed as the layer's
        // `post_attn_norm`, the norm before the FFN. Gemma-4's `rms_norm` kernel
        // scales by `w`, not `1 + w`, so the norm weights load unchanged.
        let input_norm = dense(store, &format!("{lp}.input_layernorm.weight"))?;
        let pre_ffn_norm = dense(store, &format!("{lp}.pre_feedforward_layernorm.weight"))?;

        // 2026-09-25: Norms of the attention and FFN outputs, before each
        // residual add.
        let post_attn_out_norm = dense(store, &format!("{lp}.post_attention_layernorm.weight"))?;
        let post_ffn_norm_w = dense(store, &format!("{lp}.post_feedforward_layernorm.weight"))?;

        // 2026-09-25: `layer_scalar` is read as BF16 and multiplies the whole
        // hidden state, residual included, at the end of the layer.
        let layer_scalar_key = format!("{lp}.layer_scalar");
        let layer_scalar_val = if store.contains(&layer_scalar_key) {
            let mut scalar_buf = [0u8; 2];
            let wt = store.get(&layer_scalar_key)?;
            gpu.copy_d2h(wt.ptr, &mut scalar_buf)?;
            let scalar_bf16 = u16::from_le_bytes(scalar_buf);
            let scalar_f32 = f32::from_bits((scalar_bf16 as u32) << 16);
            tracing::info!("L{i}: layer_scalar = {scalar_f32:.6}");
            Some(scalar_f32)
        } else {
            None
        };

        // 2026-09-25: A layer is full attention when its `q_proj` rows differ from
        // `num_attention_heads * 256`.
        let p = format!("{lp}.self_attn");
        let sliding_head_dim: usize = 256;
        let q_out_dim = store
            .get(&format!("{p}.q_proj.weight"))
            .map(|w| w.shape[0])
            .unwrap_or(config.num_attention_heads * sliding_head_dim);
        let kv_out_dim = store
            .get(&format!("{p}.k_proj.weight"))
            .map(|w| w.shape[0])
            .unwrap_or(config.num_key_value_heads * sliding_head_dim);
        let is_full_attn = q_out_dim != config.num_attention_heads * sliding_head_dim;

        let attn_is_nvfp4 = store.contains(&format!("{p}.q_proj.weight_scale"));
        let q_dense = if attn_is_nvfp4 {
            use metrale_model_layers::weight_map::fp8_lut::dequant_nvfp4_to_bf16;
            dequant_nvfp4_to_bf16(store, &format!("{p}.q_proj"), q_out_dim, h, gpu)?
        } else {
            dense_auto(store, &format!("{p}.q_proj.weight"), gpu)?
        };
        let k_dense = if attn_is_nvfp4 {
            use metrale_model_layers::weight_map::fp8_lut::dequant_nvfp4_to_bf16;
            dequant_nvfp4_to_bf16(store, &format!("{p}.k_proj"), kv_out_dim, h, gpu)?
        } else {
            dense_auto(store, &format!("{p}.k_proj.weight"), gpu)?
        };
        let v_key = format!("{p}.v_proj.weight");
        let v_dense =
            if store.contains(&v_key) || store.contains(&format!("{p}.v_proj.weight_scale")) {
                if attn_is_nvfp4 {
                    use metrale_model_layers::weight_map::fp8_lut::dequant_nvfp4_to_bf16;
                    dequant_nvfp4_to_bf16(store, &format!("{p}.v_proj"), kv_out_dim, h, gpu)?
                } else {
                    dense_auto(store, &v_key, gpu)?
                }
            } else {
                k_dense
            };
        let o_dense = if attn_is_nvfp4 {
            use metrale_model_layers::weight_map::fp8_lut::dequant_nvfp4_to_bf16;
            dequant_nvfp4_to_bf16(store, &format!("{p}.o_proj"), h, q_out_dim, gpu)?
        } else {
            dense_auto(store, &format!("{p}.o_proj.weight"), gpu)?
        };
        if is_full_attn {
            tracing::info!("L{i}: full attention (Q_dim={q_out_dim}, K_dim={kv_out_dim}, K=V)");
        } else {
            tracing::debug!("L{i}: sliding attention (Q_dim={q_out_dim}, K_dim={kv_out_dim})");
        }

        // 2026-09-25: Attention stays BF16 by default: q/k/v get no NVFP4 copy
        // and o_proj gets a BF16 copy. `METRALE_GEMMA4_BF16_ATTN=0` quantizes
        // them to NVFP4 instead.
        let bf16_attn_default = true;
        let bf16_attn = match std::env::var("METRALE_GEMMA4_BF16_ATTN").ok().as_deref() {
            Some("0") => false,
            Some("1") => true,
            _ => bf16_attn_default,
        };
        // 2026-09-25: `METRALE_GEMMA4_BF16_MLP=1` adds BF16 copies of an NVFP4
        // MLP (`build_bf16_mlp`), which the FFN then uses; off by default.
        let bf16_mlp_default = false;
        let bf16_mlp = match std::env::var("METRALE_GEMMA4_BF16_MLP").ok().as_deref() {
            Some("0") => false,
            Some("1") => true,
            _ => bf16_mlp_default,
        };
        // 2026-09-25: Under TP, q/k/v are column-sharded and o row-sharded by
        // their per-layer on-disk dims before any quantization, which then
        // uses the per-rank dims. When V aliases K, V is re-pointed at the
        // sharded K.
        let tp_rank = config.tp_rank;
        let tp_size = config.tp_world_size.max(1);
        let v_aliases_k = v_dense.weight == k_dense.weight;
        let (mut q_dense, mut k_dense, mut v_dense, mut o_dense) =
            (q_dense, k_dense, v_dense, o_dense);
        let local_q_out = q_out_dim / tp_size;
        let local_kv_out = kv_out_dim / tp_size;
        if tp_size > 1 {
            let (qp, _, _) = shard_dense_bf16(
                q_dense.weight,
                q_out_dim,
                h,
                TpShardKind::ColumnParallel,
                tp_rank,
                tp_size,
                gpu,
            )?;
            if qp != q_dense.weight {
                gpu.free(q_dense.weight)?;
            }
            q_dense.weight = qp;
            let (kp, _, _) = shard_dense_bf16(
                k_dense.weight,
                kv_out_dim,
                h,
                TpShardKind::ColumnParallel,
                tp_rank,
                tp_size,
                gpu,
            )?;
            if kp != k_dense.weight {
                gpu.free(k_dense.weight)?;
            }
            k_dense.weight = kp;
            if v_aliases_k {
                v_dense.weight = k_dense.weight;
            } else {
                let (vp, _, _) = shard_dense_bf16(
                    v_dense.weight,
                    kv_out_dim,
                    h,
                    TpShardKind::ColumnParallel,
                    tp_rank,
                    tp_size,
                    gpu,
                )?;
                if vp != v_dense.weight {
                    gpu.free(v_dense.weight)?;
                }
                v_dense.weight = vp;
            }
            let (op, _, _) = shard_dense_bf16(
                o_dense.weight,
                h,
                q_out_dim,
                TpShardKind::RowParallel,
                tp_rank,
                tp_size,
                gpu,
            )?;
            if op != o_dense.weight {
                gpu.free(o_dense.weight)?;
            }
            o_dense.weight = op;
        }
        let q_out_dim = local_q_out;
        let kv_out_dim = local_kv_out;
        let (q_nvfp4_opt, k_nvfp4_opt, v_nvfp4_opt) = if bf16_attn {
            tracing::info!(
                "L{i}: BF16 attention (dense_gemv path) — skip NVFP4 q/k/v quant for Gemma-4 precision"
            );
            (None, None, None)
        } else {
            let q = quantize_to_nvfp4(&q_dense, q_out_dim, h, gpu, absmax_k, quantize_k, stream)?;
            let k = quantize_to_nvfp4(&k_dense, kv_out_dim, h, gpu, absmax_k, quantize_k, stream)?;
            let v = quantize_to_nvfp4(&v_dense, kv_out_dim, h, gpu, absmax_k, quantize_k, stream)?;
            (Some(q), Some(k), Some(v))
        };
        // 2026-09-25: `AttentionWeights::o_proj` is NVFP4, so it is always built.
        // With BF16 attention the layer also gets `o_dense_bf16`, which the
        // o_proj dispatch takes before the NVFP4 weight.
        let o_nvfp4 = quantize_to_nvfp4(&o_dense, h, q_out_dim, gpu, absmax_k, quantize_k, stream)?;

        let (k_scale, v_scale) = load_kv_scales(store, &p, gpu);

        let attn = AttentionWeights {
            q_proj: q_dense,
            k_proj: k_dense,
            v_proj: v_dense,
            o_proj: o_nvfp4,
            q_norm: dense(store, &format!("{p}.q_norm.weight"))?,
            k_norm: dense(store, &format!("{p}.k_norm.weight"))?,
            q_norm_full: None,
            k_norm_full: None,
            k_scale,
            v_scale,
        };

        // 2026-09-25: The NVFP4 MLP weights are loaded even when BF16 copies are
        // installed with `set_bf16_weights`.
        let gate_proj = quantized_any(
            store,
            &format!("{lp}.mlp.gate_proj"),
            config.intermediate_size,
            h,
            gpu,
            variant,
            qctx,
        )?;
        let up_proj = quantized_any(
            store,
            &format!("{lp}.mlp.up_proj"),
            config.intermediate_size,
            h,
            gpu,
            variant,
            qctx,
        )?;
        let down_proj = quantized_any(
            store,
            &format!("{lp}.mlp.down_proj"),
            h,
            config.intermediate_size,
            gpu,
            variant,
            qctx,
        )?;
        // 2026-09-25: The `Some(wt)` arms of `DenseFfnLayer::forward_prefill`'s
        // `w4_gemm!` read these transposed copies; without them prefill takes
        // an arm over the untransposed weights, down to plain `w4a16_gemm`. The
        // copies are skipped when `bf16_mlp` is on or `ffn_transpose_fits` says
        // they do not fit.
        let want_ffn_t = !bf16_mlp && ffn_transpose_fits(config, gpu);
        let (gate_proj_t, up_proj_t, down_proj_t) = if want_ffn_t {
            (
                Some(gate_proj.transpose_for_gemm(gpu, config.intermediate_size, h)?),
                Some(up_proj.transpose_for_gemm(gpu, config.intermediate_size, h)?),
                Some(down_proj.transpose_for_gemm(gpu, h, config.intermediate_size)?),
            )
        } else {
            (None, None, None)
        };
        let ffn_weights = DenseFfnWeights {
            gate_proj,
            up_proj,
            down_proj,
            gate_proj_t,
            up_proj_t,
            down_proj_t,
        };
        let bf16_mlp_weights = build_bf16_mlp(store, &lp, bf16_mlp, config, gpu, h)?;
        gpu.synchronize(stream)?;
        tracing::info!(
            "L{i}: FFN weights loaded (bf16_mlp={bf16_mlp}), building DenseFfnLayer (GELU)..."
        );
        let mut ffn_layer =
            DenseFfnLayer::new_with_activation(ffn_weights, FfnActivation::GeLU, gpu)?;
        if let Some((g, u, d)) = bf16_mlp_weights {
            ffn_layer.set_bf16_weights(g, u, d);
            tracing::info!("L{i}: BF16 MLP weights installed");
        }
        let ffn = FfnComponent::Dense(ffn_layer);
        gpu.synchronize(stream)?;
        tracing::info!("L{i}: DenseFfnLayer built");

        let moe_ffn = build_moe_ffn(
            store,
            &lp,
            i,
            config,
            gpu,
            variant,
            qctx,
            h,
            absmax_k,
            quantize_k,
            moe_prefill_copies,
            stream,
        )?;

        tracing::info!("L{i}: building attention layer...");

        let layer_kv_dtype = layer_kv_dtypes[i];
        // 2026-09-25: The layer is built from the model config, then given this
        // layer's head dim and KV head count, derived from the projection shapes
        // and `num_attention_heads`.
        let mut layer = Qwen3AttentionLayer::new_ungated(
            input_norm,
            attn,
            pre_ffn_norm,
            ffn,
            i,
            q_nvfp4_opt,
            k_nvfp4_opt,
            v_nvfp4_opt,
            gpu,
            layer_kv_dtype,
            config.fp8_kv_calibration_tokens,
            config,
        )?;
        let actual_head_dim = q_out_dim / config.num_attention_heads;
        let actual_kv_heads = kv_out_dim / actual_head_dim;
        layer.set_dimension_overrides(actual_head_dim, config.num_attention_heads, actual_kv_heads);
        layer.set_attn_scale_override(1.0);
        if bf16_attn {
            layer.set_o_dense_bf16(o_dense);
        }
        layer.set_post_sublayer_norms(post_attn_out_norm, post_ffn_norm_w);
        if let Some(scalar) = layer_scalar_val {
            layer.set_layer_scalar(scalar);
        }
        // 2026-09-25: Every layer gets a V norm with a ones weight: the norm
        // kernel scales by `w`, so ones give a plain RMS normalization. A layer
        // with no `v_proj.weight` also gets `k_eq_v`.
        let is_k_eq_v = !store.contains(&v_key);
        let v_norm_w = super::loader_b::make_v_norm_ones_bf16(gpu, actual_head_dim)?;
        if is_k_eq_v {
            layer.set_k_eq_v(v_norm_w);
        } else {
            layer.set_v_norm(v_norm_w);
        }
        if is_full_attn {
            let rope_angles = ((actual_head_dim as f32 * 0.25) / 2.0) as u32;
            layer.set_rope_overrides(1_000_000.0, rope_angles);
            layer.set_rope_proportional(true);
            layer.set_sliding_window(None);
            tracing::info!(
                "L{i}: full attn: hd={actual_head_dim}, nq={}, nkv={actual_kv_heads}, rope=1M/proportional/angles={rope_angles}, K=V={is_k_eq_v}",
                config.num_attention_heads
            );
        } else {
            layer.set_rope_overrides(10_000.0, actual_head_dim as u32);
            if config.sliding_window > 0 {
                layer.set_sliding_window(Some(config.sliding_window));
            }
            if i < 3 {
                tracing::info!(
                    "L{i}: sliding attn: hd={actual_head_dim}, nkv={actual_kv_heads}, window={:?}",
                    config.sliding_window
                );
            }
        }
        if let Some((moe_comp, pre_norm, post_norm, dense_norm)) = moe_ffn {
            layer.set_moe_ffn(moe_comp, pre_norm, post_norm, dense_norm);
        }

        gpu.synchronize(stream)?;
        tracing::info!("L{i}: attention layer built OK");

        layers.push(Box::new(layer));

        if (i + 1) % 10 == 0 {
            tracing::info!("Loaded layers 0..{}", i + 1);
        }
    }

    tracing::info!(
        "Gemma-4 weight loader: {} layers (all attention, ungated, dense FFN)",
        layers.len(),
    );

    Ok(layers)
}
