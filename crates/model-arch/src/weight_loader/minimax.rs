// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: MiniMax M2 weight loader (`minimax_m2`): every layer is full attention plus a
//! routed MoE.
//!
//! Owner: model-arch weight loader (MiniMax).
//! Invariants:
//! - Q/K/V are sharded column-parallel and O row-parallel (`load_qkvo_tp`) before being
//!   quantized to NVFP4 at load; the q/k norms are sharded 1-D (`load_qk_norms_tp`).
//! - The per-head `q_norm`/`k_norm` slots are null; `q_norm_full`/`k_norm_full` carry the
//!   full-width norm weights.
//! - `load_mtp_weights_multi` returns no modules when the checkpoint has no
//!   `model.layers.{num_hidden_layers}` tensors, and an error when it has them.

use anyhow::Result;
use metrale_cache::kv_cache::KvCacheDtype;
use metrale_config::ModelConfig;
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_model_weights::weights::WeightStore;

use super::ModelWeightLoader;
use crate::tp_shard::{
    TpShardKind, load_qk_norms_tp, load_qkvo_tp, shard_dense_1d_bf16, shard_dense_bf16,
};
use metrale_model_layers::layer::TransformerLayer;
use metrale_model_layers::layers::{FfnComponent, MoeLayer, Qwen3AttentionLayer};
use metrale_model_layers::weight_map::quant_helpers::dense_auto;
use metrale_model_layers::weight_map::ssm_qwen35_more::load_moe_minimax;
use metrale_model_layers::weight_map::{
    AttentionWeights, DenseWeight, MtpWeights, QuantizedWeight, dense, detect_nvfp4_variant,
    load_kv_scales, quantize_to_nvfp4,
};

pub struct MinimaxM2WeightLoader;

impl ModelWeightLoader for MinimaxM2WeightLoader {
    fn supports_tp(&self) -> bool {
        // 2026-09-25: `load_layers` shards Q/K/V/O and the q/k norms by
        // `config.tp_rank` / `config.tp_world_size`.
        true
    }

    fn load_layers(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
        layer_kv_dtypes: &[KvCacheDtype],
    ) -> Result<Vec<Box<dyn TransformerLayer>>> {
        let absmax_k = gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax")?;
        let quantize_k = gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4")?;
        let stream = gpu.default_stream();
        let h = config.hidden_size;
        let variant = detect_nvfp4_variant(store, config);
        tracing::info!(
            "minimax_m2: loading {} layers, variant={:?}, hidden_size={h}",
            config.num_hidden_layers,
            variant,
        );

        // 2026-09-25: The MoE prefill transpose is not done here; `factory::build`
        // runs it after load (`maybe_run_minimax_m2_moe_transpose`).

        let mut layers: Vec<Box<dyn TransformerLayer>> =
            Vec::with_capacity(config.num_hidden_layers);

        // 2026-09-25: Null weight for the per-head q_norm/k_norm slots; the
        // full-width norms go in `q_norm_full`/`k_norm_full`.
        let dummy_norm = DenseWeight {
            weight: metrale_gpu_runtime::gpu::DevicePtr::NULL,
        };

        for i in 0..config.num_hidden_layers {
            let lp = format!("model.layers.{i}");
            tracing::debug!("minimax_m2: layer {i}");
            let input_norm = dense(store, &format!("{lp}.input_layernorm.weight"))?;
            let post_attn_norm = dense(store, &format!("{lp}.post_attention_layernorm.weight"))?;

            // 2026-09-25: MoE through `load_moe_minimax`, with the router gate
            // also quantized to NVFP4.
            let moe_weights = load_moe_minimax(
                store,
                &lp,
                config.num_experts,
                gpu,
                config,
                variant,
                absmax_k,
                quantize_k,
                stream,
            )?;
            let gate_nvfp4 = quantize_to_nvfp4(
                &moe_weights.gate,
                config.num_experts,
                h,
                gpu,
                absmax_k,
                quantize_k,
                stream,
            )?;
            let mut moe_layer = MoeLayer::new(
                moe_weights,
                config.num_experts,
                Some(gate_nvfp4),
                gpu,
                config,
            )?;
            // 2026-09-25: `predequant_for_prefill` runs here; the prefill
            // transpose runs after load (see above).
            moe_layer.predequant_for_prefill(gpu, config, stream)?;
            let ffn = FfnComponent::Moe(moe_layer);

            // 2026-09-25: Attention: ungated Q, full-width q/k norms.
            let p = format!("{lp}.self_attn");
            // 2026-09-25: Each projection is read through `dense_auto` (a new
            // BF16 buffer for an FP8 or F32 source), TP-sharded and quantized to
            // NVFP4. The BF16 buffers and the store's source tensor are then
            // freed, and the returned `DenseWeight` is null.
            let tp_rank = config.tp_rank;
            let tp_size = config.tp_world_size;
            let load_and_quant = |name: &str,
                                  full_n: usize,
                                  full_k: usize,
                                  kind: TpShardKind|
             -> Result<(DenseWeight, QuantizedWeight)> {
                let wkey = format!("{p}.{name}.weight");
                let scale_key = format!("{p}.{name}.weight_scale_inv");
                let (src_ptr, src_dtype) = {
                    let t = store.get(&wkey)?;
                    (t.ptr, t.dtype)
                };
                let src_is_fp8 = src_dtype == metrale_model_weights::weights::WeightDtype::FP8E4M3;
                let src_is_f32 = src_dtype == metrale_model_weights::weights::WeightDtype::FP32;
                let scale_ptr = if src_is_fp8 && store.contains(&scale_key) {
                    Some(store.get(&scale_key)?.ptr)
                } else {
                    None
                };
                let dense_w = dense_auto(store, &wkey, gpu)?;
                // 2026-09-25: TP-shard the BF16 weight before NVFP4 quantization.
                // With tp_size <= 1 the helper returns the input pointer.
                let (sharded_ptr, local_n, local_k) =
                    shard_dense_bf16(dense_w.weight, full_n, full_k, kind, tp_rank, tp_size, gpu)?;
                let sharded = DenseWeight {
                    weight: sharded_ptr,
                };
                let q = quantize_to_nvfp4(
                    &sharded, local_n, local_k, gpu, absmax_k, quantize_k, stream,
                )?;
                if sharded_ptr != dense_w.weight {
                    gpu.free(sharded_ptr)?;
                }
                if src_is_fp8 {
                    // 2026-09-25: FP8 source: free the BF16 dequant buffer, the
                    // FP8 source and its block scale. The store keeps the stale
                    // pointers.
                    gpu.free(dense_w.weight)?;
                    gpu.free(src_ptr)?;
                    if let Some(sp) = scale_ptr {
                        gpu.free(sp)?;
                    }
                } else {
                    // 2026-09-25: BF16 or F32 source: free the F32-to-BF16
                    // buffer when one was made, and the source.
                    if src_is_f32 {
                        gpu.free(dense_w.weight)?;
                    }
                    gpu.free(src_ptr)?;
                }
                Ok((
                    DenseWeight {
                        weight: metrale_gpu_runtime::gpu::DevicePtr::NULL,
                    },
                    q,
                ))
            };
            // 2026-09-25: Q/K/V column-parallel and O row-parallel through
            // `load_qkvo_tp`, which passes each projection's full dims to
            // `load_and_quant`.
            let [
                (q_dense, q_nvfp4),
                (k_dense, k_nvfp4),
                (v_dense, v_nvfp4),
                (_o_dense, o_nvfp4),
            ] = load_qkvo_tp(config, |name, full_n, full_k, kind| {
                load_and_quant(name, full_n, full_k, kind)
            })?;

            let (k_scale, v_scale) = load_kv_scales(store, &p, gpu);

            // 2026-09-25: Full-width q/k norm weights, sharded 1-D to match the
            // local Q/K outputs; `shard_dense_1d_bf16` returns the input pointer
            // when tp_size <= 1.
            let (q_norm_full, k_norm_full) = load_qk_norms_tp(config, |name, full_dim| {
                let src = dense(store, &format!("{p}.{name}.weight"))?;
                let (ptr, _) = shard_dense_1d_bf16(src.weight, full_dim, tp_rank, tp_size, gpu)?;
                Ok::<_, anyhow::Error>(DenseWeight { weight: ptr })
            })?;

            let attn = AttentionWeights {
                q_proj: q_dense,
                k_proj: k_dense,
                v_proj: v_dense,
                o_proj: o_nvfp4,
                // 2026-09-25: Per-head slots null; the full-width norms follow.
                q_norm: dummy_norm,
                k_norm: dummy_norm,
                q_norm_full: Some(q_norm_full),
                k_norm_full: Some(k_norm_full),
                k_scale,
                v_scale,
            };

            let mut layer = Qwen3AttentionLayer::new_ungated(
                input_norm,
                attn,
                post_attn_norm,
                ffn,
                i,
                Some(q_nvfp4),
                Some(k_nvfp4),
                Some(v_nvfp4),
                gpu,
                layer_kv_dtypes[i],
                config.fp8_kv_calibration_tokens,
                config,
            )?;

            // 2026-09-25: Transposed copies of the attention NVFP4 weights for
            // prefill (`set_prefill_weights`).
            let q_proj_n = config.num_attention_heads * config.head_dim;
            let kv_proj_n = config.num_key_value_heads * config.head_dim;
            let qt = q_nvfp4.transpose_for_gemm(gpu, q_proj_n, h)?;
            let kt = k_nvfp4.transpose_for_gemm(gpu, kv_proj_n, h)?;
            let vt = v_nvfp4.transpose_for_gemm(gpu, kv_proj_n, h)?;
            let ot = layer.attn.o_proj.transpose_for_gemm(gpu, h, q_proj_n)?;
            layer.set_prefill_weights(Some(qt), Some(kt), Some(vt), Some(ot));

            layers.push(Box::new(layer));
        }

        tracing::info!("minimax_m2: built {} layers", layers.len());
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
        dense(store, "lm_head.weight")
    }

    fn load_mtp_weights(
        &self,
        _store: &WeightStore,
        _config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<Option<MtpWeights>> {
        // 2026-09-25: MiniMax's MTP modules go through `load_mtp_weights_multi`.
        Ok(None)
    }

    fn load_mtp_weights_multi(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<Vec<MtpWeights>> {
        // 2026-09-25: MTP modules would start at `model.layers.{num_hidden_layers}`.
        // Without that layer's `input_layernorm.weight` the checkpoint has none:
        // return no modules.
        let first_mtp_idx = config.num_hidden_layers;
        let probe = format!("model.layers.{first_mtp_idx}.input_layernorm.weight");
        if !store.contains(&probe) {
            tracing::info!(
                "minimax_m2: no MTP module weights found in checkpoint \
                 (expected starting at layer {first_mtp_idx}); MTP disabled"
            );
            return Ok(Vec::new());
        }

        // 2026-09-25: Per-module extraction is not implemented, so a checkpoint
        // that has the modules fails the load.
        anyhow::bail!(
            "minimax_m2: MTP module weights detected at layer {first_mtp_idx} \
             but the MiniMax loader hasn't implemented per-module extraction \
             yet. Run with --speculative 0 for non-MTP decode, or await \
             MiniMax M5 phase-3 (populate load_mtp_weights_multi with the \
             concrete Mixtral-convention weight keys)."
        )
    }
}
