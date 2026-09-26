// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `Step3p7WeightLoader`, the Step 3.7 Flash weight loader, with
//! its norm-offset and fused-expert helpers.
//!
//! Owner: model-arch weight loader.
//! Invariants: none beyond the types.
//!
//! Layers are read under `{weight_prefix}.layers.{i}`, with
//! `model.language_model` when the prefix is empty. A layer with
//! `moe.gate.weight` gets a MoE FFN and any other a dense FFN
//! (`load_layers.rs`). Routed experts are read per expert
//! (`moe.experts.{e}.*`) when such tensors exist, otherwise from one fused
//! tensor per projection that `slice_fused_experts` splits by byte offset.
//! MTP modules, probed at `model.layers.{num_hidden_layers}`, are detected
//! but not loaded.

mod load_layers;

use anyhow::Result;
use metrale_cache::kv_cache::KvCacheDtype;
use metrale_config::ModelConfig;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_weights::weights::WeightStore;

use super::ModelWeightLoader;
use metrale_model_layers::layer::TransformerLayer;
use metrale_model_layers::weight_map::{DenseWeight, MtpWeights, QuantizedWeight, dense};

pub struct Step3p7WeightLoader;

/// 2026-09-25: Add 1.0 to each of `size` BF16 norm weights in place, so the
/// standard RMSNorm kernel, `(x / rms) * weight`, computes
/// `(x / rms) * (weight + 1)`. The loader applies it to every norm it loads.
fn offset_norm_weights_plus_one(
    weight: &DenseWeight,
    size: usize,
    gpu: &dyn GpuBackend,
) -> Result<()> {
    let byte_len = size * 2;
    let mut buf = vec![0u8; byte_len];
    gpu.copy_d2h(weight.weight, &mut buf)?;

    for i in 0..size {
        let bits = u16::from_le_bytes([buf[i * 2], buf[i * 2 + 1]]);
        let f32_val = f32::from_bits((bits as u32) << 16);
        let new_val = f32_val + 1.0;
        // 2026-09-25: Round to nearest even when truncating to BF16.
        let f32_bits = new_val.to_bits();
        let new_bits = ((f32_bits + 0x7FFF + ((f32_bits >> 16) & 1)) >> 16) as u16;
        buf[i * 2] = new_bits as u8;
        buf[i * 2 + 1] = (new_bits >> 8) as u8;
    }

    gpu.copy_h2d(&buf, weight.weight)?;
    Ok(())
}

/// 2026-09-25: Split one fused NVFP4 projection into `num_experts`
/// `QuantizedWeight` views of `[n, k]`. Expert `e` starts at `e * n * k / 2`
/// bytes of the packed weight, `e * n * ceil(k / 16)` bytes of the group
/// scales and, when present, `e * n * 4` bytes of the input scale; all share
/// `global_scale_2`. Nothing is copied.
fn slice_fused_experts(
    fused_weight: DevicePtr,
    fused_scale: DevicePtr,
    fused_input_scale: DevicePtr,
    global_scale_2: f32,
    num_experts: usize,
    n: usize,
    k: usize,
) -> Vec<QuantizedWeight> {
    let group_size = 16usize;
    let packed_bytes_per_expert = n * k / 2;
    let scale_bytes_per_expert = n * k.div_ceil(group_size);
    let input_scale_bytes_per_expert = n * 4;

    (0..num_experts)
        .map(|e| QuantizedWeight {
            weight: fused_weight.offset(e * packed_bytes_per_expert),
            weight_scale: fused_scale.offset(e * scale_bytes_per_expert),
            weight_scale_2: global_scale_2,
            input_scale: if fused_input_scale == DevicePtr::NULL {
                DevicePtr::NULL
            } else {
                fused_input_scale.offset(e * input_scale_bytes_per_expert)
            },
            weight_scale_2_vec: DevicePtr::NULL,
        })
        .collect()
}

/// 2026-09-25: Whether any store tensor name starts with
/// `{layer_prefix}.moe.experts.`.
fn has_per_expert_tensors(store: &WeightStore, layer_prefix: &str) -> bool {
    let pattern = format!("{layer_prefix}.moe.experts.");
    let found = store.names().any(|k| k.starts_with(&pattern));
    tracing::debug!("has_per_expert_tensors('{layer_prefix}'): pattern='{pattern}', found={found}");
    found
}

/// 2026-09-25: The store pointers of one fused NVFP4 projection:
/// `(weight, weight_scale, input_scale or NULL, weight_scale_2 as f32)`.
fn load_fused_nvfp4(
    store: &WeightStore,
    prefix: &str,
    gpu: &dyn GpuBackend,
) -> Result<(DevicePtr, DevicePtr, DevicePtr, f32)> {
    let weight = store.get(&format!("{prefix}.weight"))?.ptr;
    let weight_scale = store.get(&format!("{prefix}.weight_scale"))?.ptr;

    let ws2_key = format!("{prefix}.weight_scale_2");
    let ws2_ptr = store.get(&ws2_key)?.ptr;
    let mut ws2_buf = [0u8; 4];
    gpu.copy_d2h(ws2_ptr, &mut ws2_buf)?;
    let weight_scale_2 = f32::from_le_bytes(ws2_buf);

    let is_key = format!("{prefix}.input_scale");
    let input_scale = if store.contains(&is_key) {
        store.get(&is_key)?.ptr
    } else {
        DevicePtr::NULL
    };

    Ok((weight, weight_scale, input_scale, weight_scale_2))
}

impl ModelWeightLoader for Step3p7WeightLoader {
    fn supports_tp(&self) -> bool {
        false
    }

    fn load_layers(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
        layer_kv_dtypes: &[KvCacheDtype],
    ) -> Result<Vec<Box<dyn TransformerLayer>>> {
        load_layers::load_layers(store, config, gpu, layer_kv_dtypes)
    }

    fn load_embedding(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        let prefix = if config.weight_prefix.is_empty() {
            "model.language_model"
        } else {
            &config.weight_prefix
        };
        dense(store, &format!("{prefix}.embed_tokens.weight"))
    }

    fn load_final_norm(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        let prefix = if config.weight_prefix.is_empty() {
            "model.language_model"
        } else {
            &config.weight_prefix
        };
        let w = dense(store, &format!("{prefix}.norm.weight"))?;
        offset_norm_weights_plus_one(&w, config.hidden_size, gpu)?;
        Ok(w)
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
        Ok(None)
    }

    fn load_mtp_weights_multi(
        &self,
        store: &WeightStore,
        config: &ModelConfig,
        _gpu: &dyn GpuBackend,
    ) -> Result<Vec<MtpWeights>> {
        let first_mtp_idx = config.num_hidden_layers;
        let probe = format!("model.layers.{first_mtp_idx}.input_layernorm.weight");
        if !store.contains(&probe) {
            tracing::info!(
                "step3p7: no MTP module weights found \
                 (expected at layer {first_mtp_idx}); MTP disabled"
            );
            return Ok(Vec::new());
        }

        tracing::info!(
            "step3p7: MTP module weights detected at layers {}-{} but MTP loader \
             not yet implemented. Run with --speculative 0 for non-MTP decode.",
            first_mtp_idx,
            first_mtp_idx + config.mtp_num_hidden_layers - 1,
        );
        Ok(Vec::new())
    }
}
