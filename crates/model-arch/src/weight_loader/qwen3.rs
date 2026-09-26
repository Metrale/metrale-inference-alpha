// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `Qwen3WeightLoader`, the loader for `qwen3_next` checkpoints: full-attention
//! and linear-attention (GDN) layers, each with an MoE block, plus embedding, final norm,
//! LM head and MTP weights.
//!
//! Owner: model-arch weight loader (Qwen3-Next).
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_cache::kv_cache::KvCacheDtype;
use metrale_config::{LayerType, ModelConfig};
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_model_weights::weights::WeightStore;

use super::{ModelWeightLoader, QuantFormat};
use crate::tp_shard::{TpShardKind, load_qkvo_tp, shard_dense_bf16, shard_fp8_block_scaled};
use metrale_model_layers::layer::TransformerLayer;
use metrale_model_layers::layers::{FfnComponent, MoeLayer, Qwen3AttentionLayer, Qwen3SsmLayer};
use metrale_model_layers::weight_map::loaders_moe::load_mtp;
use metrale_model_layers::weight_map::{
    AttentionWeights, DenseWeight, MtpWeights, Nvfp4Variant, QuantizeCtx, QuantizedWeight, dense,
    detect_nvfp4_variant, load_attention, load_fp8_block_scaled_as_fp8weight, load_kv_scales,
    load_moe, load_moe_qwen35_fp8_experts, load_moe_skip_experts, load_ssm, quantize_to_nvfp4,
};

pub struct Qwen3WeightLoader;

impl ModelWeightLoader for Qwen3WeightLoader {
    fn supports_tp(&self) -> bool {
        // 2026-09-25: Full-attention layers shard on both arms. Linear-attention layers are
        // loaded unsharded (`load_ssm`).
        true
    }

    fn load_layers(
        &self,
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

        let mut layers: Vec<Box<dyn TransformerLayer>> =
            Vec::with_capacity(config.num_hidden_layers);
        let mut attn_idx = 0usize;

        let absmax_k = gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax")?;
        let quantize_k = gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4")?;
        let stream = gpu.default_stream();
        let qctx = QuantizeCtx {
            absmax_k,
            quantize_k,
            stream,
        };

        let variant = detect_nvfp4_variant(store, config);
        let quant_format = if variant == Nvfp4Variant::Fp8Dequanted {
            QuantFormat::Fp8
        } else {
            QuantFormat::Nvfp4
        };
        let native_fp8 = quant_format == QuantFormat::Fp8;
        tracing::info!(
            "Qwen3 weight variant: {:?}, native_fp8: {}",
            variant,
            native_fp8
        );

        let h = config.hidden_size;

        // 2026-09-25: `super::moe_prefill_copies_fit` decides whether the transposed MoE prefill
        // tables fit in free memory; `METRALE_MOE_PREFILL_COPIES=0` forces them off.
        let skip_moe_transpose = !super::moe_prefill_copies_fit(config, gpu);

        for (i, lt) in layer_types.iter().enumerate() {
            let lp = config.layer_prefix(i);
            let input_norm = dense(store, &format!("{lp}.input_layernorm.weight"))?;
            let post_attn_norm = dense(store, &format!("{lp}.post_attention_layernorm.weight"))?;

            let moe_weights = if native_fp8 {
                load_moe_skip_experts(store, &lp, config.num_experts, gpu, config, variant, qctx)?
            } else {
                load_moe(store, &lp, config.num_experts, gpu, config, variant, qctx)?
            };
            // 2026-09-25: `METRALE_BF16_ROUTER=1` keeps the router gate BF16 (`gate_nvfp4` None),
            // so the gate GEMM runs dense on `weights.gate` (`moe/forward_batched_gate.rs`).
            let gate_nvfp4 = if std::env::var("METRALE_BF16_ROUTER").as_deref() == Ok("1") {
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
            if !native_fp8 && !skip_moe_transpose {
                moe_layer.transpose_for_prefill(gpu, config)?;
            }
            if !native_fp8 {
                moe_layer.predequant_for_prefill(gpu, config, stream)?;
            }

            // 2026-09-25: Native FP8: install the checkpoint's FP8 routed experts. A load error is
            // discarded, and the layer then has no routed experts (`load_moe_skip_experts`).
            if native_fp8
                && let Ok(fp8_experts) =
                    load_moe_qwen35_fp8_experts(store, &lp, config.num_experts, gpu, config)
            {
                let sp = format!("{lp}.mlp.shared_expert");
                use metrale_gpu_runtime::gpu::DevicePtr;
                use metrale_model_layers::weight_map::{Fp8ExpertWeight as FEW, Fp8Weight as FW};
                let null_fw = FW {
                    weight: DevicePtr::NULL,
                    row_scale: DevicePtr::NULL,
                    n: 0,
                    k: 0,
                    // 2026-09-25: Stands in for a shared-expert tensor that failed to load, tagged
                    // like the block-scaled loader's output.
                    scale_format:
                        metrale_model_layers::weight_map::WeightQuantFormat::Fp8BlockScaled,
                };
                let sh_gate =
                    load_fp8_block_scaled_as_fp8weight(store, &format!("{sp}.gate_proj"), gpu);
                let sh_up =
                    load_fp8_block_scaled_as_fp8weight(store, &format!("{sp}.up_proj"), gpu);
                let sh_down =
                    load_fp8_block_scaled_as_fp8weight(store, &format!("{sp}.down_proj"), gpu);
                let shared_fp8 = FEW {
                    gate_proj: sh_gate.unwrap_or(null_fw),
                    up_proj: sh_up.unwrap_or(null_fw),
                    down_proj: sh_down.unwrap_or(null_fw),
                };
                if let Err(e) = moe_layer.set_fp8_experts(&fp8_experts, shared_fp8, gpu) {
                    tracing::error!("Layer {i}: FP8 expert tables failed: {e:#}");
                } else {
                    tracing::info!("Layer {i}: MoE experts loaded as native FP8");
                }
            }

            let ffn = FfnComponent::Moe(moe_layer);

            match lt {
                LayerType::FullAttention if native_fp8 => {
                    let p = format!("{lp}.self_attn");
                    let tp_rank = config.tp_rank;
                    let tp_size = config.tp_world_size.max(1);
                    let block_size = 128usize;
                    let load_fp8 = |name: &str,
                                    _full_n: usize,
                                    _full_k: usize,
                                    kind: TpShardKind|
                     -> Result<metrale_model_layers::weight_map::Fp8Weight> {
                        let src =
                            load_fp8_block_scaled_as_fp8weight(store, &format!("{p}.{name}"), gpu)?;
                        if tp_size == 1 {
                            return Ok(src);
                        }
                        let sharded =
                            shard_fp8_block_scaled(&src, kind, tp_rank, tp_size, block_size, gpu)?;
                        gpu.free(src.weight)?;
                        gpu.free(src.row_scale)?;
                        Ok(sharded)
                    };
                    let [q_fp8, k_fp8, v_fp8, o_fp8] = load_qkvo_tp(config, load_fp8)?;

                    let (k_scale, v_scale) = load_kv_scales(store, &p, gpu);
                    let dummy = DenseWeight {
                        weight: metrale_gpu_runtime::gpu::DevicePtr::NULL,
                    };
                    let attn = AttentionWeights {
                        q_proj: dummy,
                        k_proj: dummy,
                        v_proj: dummy,
                        o_proj: QuantizedWeight::null(),
                        q_norm: dense(store, &format!("{p}.q_norm.weight"))?,
                        k_norm: dense(store, &format!("{p}.k_norm.weight"))?,
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
                        tracing::warn!("Layer {i}: FP8 transpose failed: {e}");
                    }
                    layers.push(Box::new(layer));
                    attn_idx += 1;
                }
                LayerType::FullAttention => {
                    let mut attn = load_attention(store, &lp, gpu, variant, qctx, config)?;
                    let tp_rank = config.tp_rank;
                    let tp_size = config.tp_world_size.max(1);
                    // 2026-09-25: Under TP, shard each projection before Q/K/V are quantized. The
                    // sizes below come from `config`, whose head counts are already per rank
                    // (server `serve_phases/topology.rs`).
                    if tp_size > 1 {
                        use crate::tp_shard::TpAttentionDims;
                        let dims = TpAttentionDims::from_config(config);
                        let (qp, _, _) = shard_dense_bf16(
                            attn.q_proj.weight,
                            dims.full_q_n,
                            dims.h,
                            TpShardKind::ColumnParallel,
                            tp_rank,
                            tp_size,
                            gpu,
                        )?;
                        if qp != attn.q_proj.weight {
                            gpu.free(attn.q_proj.weight)?;
                        }
                        attn.q_proj.weight = qp;
                        let (kp, _, _) = shard_dense_bf16(
                            attn.k_proj.weight,
                            dims.full_kv_n,
                            dims.h,
                            TpShardKind::ColumnParallel,
                            tp_rank,
                            tp_size,
                            gpu,
                        )?;
                        if kp != attn.k_proj.weight {
                            gpu.free(attn.k_proj.weight)?;
                        }
                        attn.k_proj.weight = kp;
                        let (vp, _, _) = shard_dense_bf16(
                            attn.v_proj.weight,
                            dims.full_kv_n,
                            dims.h,
                            TpShardKind::ColumnParallel,
                            tp_rank,
                            tp_size,
                            gpu,
                        )?;
                        if vp != attn.v_proj.weight {
                            gpu.free(attn.v_proj.weight)?;
                        }
                        attn.v_proj.weight = vp;
                        let (op, _, _) = shard_dense_bf16(
                            attn.o_proj.weight,
                            dims.h,
                            dims.full_o_in,
                            TpShardKind::RowParallel,
                            tp_rank,
                            tp_size,
                            gpu,
                        )?;
                        if op != attn.o_proj.weight {
                            gpu.free(attn.o_proj.weight)?;
                        }
                        attn.o_proj.weight = op;
                    }
                    let q_nvfp4 = quantize_to_nvfp4(
                        &attn.q_proj,
                        config.num_attention_heads * config.head_dim * 2,
                        h,
                        gpu,
                        absmax_k,
                        quantize_k,
                        stream,
                    )?;
                    let k_nvfp4 = quantize_to_nvfp4(
                        &attn.k_proj,
                        config.num_key_value_heads * config.head_dim,
                        h,
                        gpu,
                        absmax_k,
                        quantize_k,
                        stream,
                    )?;
                    let v_nvfp4 = quantize_to_nvfp4(
                        &attn.v_proj,
                        config.num_key_value_heads * config.head_dim,
                        h,
                        gpu,
                        absmax_k,
                        quantize_k,
                        stream,
                    )?;
                    let layer_kv_dtype = layer_kv_dtypes[attn_idx];
                    let mut layer = Qwen3AttentionLayer::new(
                        input_norm,
                        attn,
                        post_attn_norm,
                        ffn,
                        attn_idx,
                        Some(q_nvfp4),
                        Some(k_nvfp4),
                        Some(v_nvfp4),
                        gpu,
                        layer_kv_dtype,
                        config.fp8_kv_calibration_tokens,
                        config,
                    )?;
                    let qt = q_nvfp4.transpose_for_gemm(
                        gpu,
                        config.num_attention_heads * config.head_dim * 2,
                        h,
                    )?;
                    let kt = k_nvfp4.transpose_for_gemm(
                        gpu,
                        config.num_key_value_heads * config.head_dim,
                        h,
                    )?;
                    let vt = v_nvfp4.transpose_for_gemm(
                        gpu,
                        config.num_key_value_heads * config.head_dim,
                        h,
                    )?;
                    let ot = layer.attn.o_proj.transpose_for_gemm(
                        gpu,
                        h,
                        config.num_attention_heads * config.head_dim,
                    )?;
                    layer.set_prefill_weights(Some(qt), Some(kt), Some(vt), Some(ot));
                    layer.predequant_for_prefill(gpu, config, stream)?;
                    layers.push(Box::new(layer));
                    attn_idx += 1;
                }
                // 2026-09-25: Linear attention takes one path for every variant: `load_ssm`, then
                // QKVZ quantized to NVFP4. It is not sharded under TP.
                LayerType::LinearAttention => {
                    let ssm = load_ssm(store, &lp, gpu, variant, qctx, config)?;
                    let qkvz_nvfp4 = quantize_to_nvfp4(
                        &ssm.in_proj_qkvz,
                        config.ssm_qkvz_size(),
                        h,
                        gpu,
                        absmax_k,
                        quantize_k,
                        stream,
                    )?;
                    layers.push(Box::new(Qwen3SsmLayer::new(
                        input_norm,
                        ssm,
                        post_attn_norm,
                        ffn,
                        Some(qkvz_nvfp4),
                        config,
                        gpu,
                    )?));
                }
                LayerType::SlidingAttention => {
                    unreachable!("unexpected SlidingAttention in this loader")
                }
                LayerType::Moe => unreachable!("Qwen3 has no standalone MoE layers"),
                // 2026-09-25: GLM-5.3 `deepseek_sparse_attention` needs a DSA indexer and per-query
                // top-k, which this loader does not build; it is refused, not served as dense
                // attention.
                LayerType::SparseAttention => anyhow::bail!(
                    "layer {i}: SparseAttention needs a DSA indexer and per-query top-k; Qwen3 has neither"
                ),
            }

            if (i + 1) % 12 == 0 {
                tracing::info!("Loaded layers 0..{}", i + 1);
            }
        }

        tracing::info!(
            "Weight loader: {} layers ({} attention, {} SSM)",
            layers.len(),
            attn_idx,
            layers.len() - attn_idx,
        );

        Ok(layers)
    }

    fn load_embedding(
        &self,
        store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        dense(store, "model.embed_tokens.weight")
    }

    fn load_final_norm(
        &self,
        store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        dense(store, "model.norm.weight")
    }

    fn load_lm_head(
        &self,
        store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        if store.contains("lm_head.weight") {
            dense(store, "lm_head.weight")
        } else {
            dense(store, "model.embed_tokens.weight")
        }
    }

    fn load_mtp_weights(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<Option<MtpWeights>> {
        if !store.contains("mtp.fc.weight") {
            tracing::info!("No MTP weights found — speculative decoding disabled");
            return Ok(None);
        }
        let variant = detect_nvfp4_variant(store, config);
        tracing::info!("Loading MTP weights (variant={:?})...", variant);
        let mtp = load_mtp(store, config.num_experts, gpu, variant)?;
        tracing::info!(
            "MTP weights loaded: fc=[2048,4096], {} experts, attn layer",
            mtp.experts.len(),
        );
        Ok(Some(mtp))
    }
}
