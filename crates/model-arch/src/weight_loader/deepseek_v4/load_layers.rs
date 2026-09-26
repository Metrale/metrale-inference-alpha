// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Builds every DeepSeek-V4 transformer layer from the weight store.
//!
//! Owner: model-arch weight loader.
//! Invariants:
//! - When `hc_mult > 0`, every layer gets a clone of the same model-level
//!   `HcHeadWeights`, or loading fails.

use anyhow::Result;
use metrale_cache::kv_cache::KvCacheDtype;
use metrale_config::ModelConfig;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_weights::weights::WeightStore;

use metrale_model_layers::layer::TransformerLayer;
use metrale_model_layers::layers::qwen3_attention::HcHeadWeights;
use metrale_model_layers::weight_map::DenseWeight;
use metrale_model_layers::weight_map::quant_helpers::dense_auto;

pub fn load_all_layers(
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    layer_kv_dtypes: &[KvCacheDtype],
) -> Result<Vec<Box<dyn TransformerLayer>>> {
    let n = config.num_hidden_layers;
    tracing::info!(
        "DeepSeek-V4 load_layers: num_layers={}, hc_mult={}, hc_sinkhorn_iters={}, hc_eps={}",
        n,
        config.hc_mult,
        config.hc_sinkhorn_iters,
        config.hc_eps,
    );
    tracing::info!(
        "DeepSeek-V4 architecture: hidden_size={}, num_experts={}, q_lora_rank={}, kv_lora_rank={}, o_lora_rank={}, head_dim={}",
        config.hidden_size,
        config.num_experts,
        config.q_lora_rank,
        config.kv_lora_rank,
        config.o_lora_rank,
        config.head_dim,
    );
    tracing::info!(
        "DeepSeek-V4 attention: q_heads={}, kv_heads={}, qk_rope_head_dim={}, qk_nope_head_dim={}",
        config.num_attention_heads,
        config.num_key_value_heads,
        config.qk_rope_head_dim,
        config.qk_nope_head_dim,
    );
    tracing::info!(
        "DeepSeek-V4 KV cache dtype: {:?}",
        layer_kv_dtypes
            .first()
            .copied()
            .unwrap_or(KvCacheDtype::Bf16),
    );

    let mut layers = Vec::with_capacity(n);
    let mut yarn_inv_freq = DevicePtr::NULL;

    // 2026-09-25: The model-level HC head is loaded once and cloned into every layer.
    let hc_head = if config.hc_mult > 0 {
        let hc = config.hc_mult;
        let hc_dim = hc * config.hidden_size;
        let head_fn = super::assemble::load_hc_f32(
            store,
            &["hc_head_fn".to_string(), "model.hc_head.fn".to_string()],
            hc * hc_dim,
            gpu,
        )
        .ok();
        let head_base = super::assemble::load_hc_f32(
            store,
            &["hc_head_base".to_string(), "model.hc_head.base".to_string()],
            hc,
            gpu,
        )
        .ok();
        let head_scale = super::assemble::load_hc_f32(
            store,
            &[
                "hc_head_scale".to_string(),
                "model.hc_head.scale".to_string(),
            ],
            1,
            gpu,
        )
        .ok();
        match (head_fn, head_base, head_scale) {
            (Some(fn_ptr), Some(base_ptr), Some(scale_ptr)) => Some(HcHeadWeights {
                hc_fn: fn_ptr,
                hc_base: base_ptr,
                hc_scale: scale_ptr,
                // 2026-09-25: `None` selects the Sinkhorn head mixer.
                lowrank: None,
            }),
            (fn_ok, base_ok, scale_ok) => {
                anyhow::bail!(
                    "DeepSeek-V4: hc_head weights missing (fn={} base={} scale={}); \
                     tried hc_head_*, model.hc_head.*, head_hc.*",
                    if fn_ok.is_some() { "ok" } else { "MISSING" },
                    if base_ok.is_some() { "ok" } else { "MISSING" },
                    if scale_ok.is_some() { "ok" } else { "MISSING" },
                );
            }
        }
    } else {
        None
    };

    for i in 0..n {
        let lp = format!("layers.{i}");
        let ap = format!("{lp}.attn");

        let input_norm = dense_auto(store, &format!("{lp}.attn_norm.weight"), gpu)?;
        let post_attn_norm = dense_auto(store, &format!("{lp}.ffn_norm.weight"), gpu)?;

        // 2026-09-25: The attention projections get no NVFP4 copy. `dense_auto`
        // loads them as BF16, dequantizing FP8, and `assemble_layer` adds the
        // native FP8 views when the checkpoint stores them as FP8.
        let wq_a = dense_auto(store, &format!("{ap}.wq_a.weight"), gpu)?;
        let wq_b = dense_auto(store, &format!("{ap}.wq_b.weight"), gpu)?;
        let q_a_norm = dense_auto(store, &format!("{ap}.q_norm.weight"), gpu)?;

        let wkv_a = dense_auto(store, &format!("{ap}.wkv.weight"), gpu)?;
        let kv_a_norm = dense_auto(store, &format!("{ap}.kv_norm.weight"), gpu)?;

        let is_v4_flash = config.o_lora_rank > 0;
        let null = DenseWeight {
            weight: DevicePtr::NULL,
        };

        // 2026-09-25: With `o_lora_rank > 0` the layer uses `wo_a` and `wo_b`
        // directly and the absorption views stay NULL.
        let (
            wo_a,
            wo_b,
            wkv_b,
            w_uk_t,
            w_uv,
            wq_b_rope,
            w_qk_absorbed,
            w_uk_block_diag,
            w_uv_block_diag,
        ) = if is_v4_flash {
            let wo_a_w = dense_auto(store, &format!("{ap}.wo_a.weight"), gpu)?;
            let wo_b_w = dense_auto(store, &format!("{ap}.wo_b.weight"), gpu)?;
            (wo_a_w, wo_b_w, null, null, null, null, null, null, null)
        } else {
            // 2026-09-25: Without `o_lora_rank`, `wo_a` serves as `wkv_b` and `wo_b` as the
            // output projection, and the absorption views are built from `wo_a` and `wq_b`.
            let wkv_b_w = dense_auto(store, &format!("{ap}.wo_a.weight"), gpu)?;
            let wkv_b_shape = store.get(&format!("{ap}.wo_a.weight"))?.shape.clone();
            let o_dense = dense_auto(store, &format!("{ap}.wo_b.weight"), gpu)?;
            let wq_b_shape = store.get(&format!("{ap}.wq_b.weight"))?.shape.clone();
            let (w_uk_t, w_uv, wq_b_rope, w_uk_host) = super::compute::build_per_head_views(
                &wkv_b_w,
                &wkv_b_shape,
                &wq_b,
                &wq_b_shape,
                config,
                gpu,
            )?;
            let w_qk_absorbed =
                super::compute::build_w_qk_absorbed(&wq_b, &wq_b_shape, &w_uk_t, config, gpu)?;
            let (w_uk_block_diag, w_uv_block_diag) =
                super::compute::build_block_diagonals(&w_uk_host, &w_uv, config, gpu)?;
            (
                null,
                o_dense,
                wkv_b_w,
                w_uk_t,
                w_uv,
                wq_b_rope,
                w_qk_absorbed,
                w_uk_block_diag,
                w_uv_block_diag,
            )
        };
        yarn_inv_freq = super::compute::ensure_yarn_inv_freq(&mut yarn_inv_freq, config, gpu)?;

        let layer = super::assemble::assemble_layer(
            i,
            &lp,
            false,
            input_norm,
            post_attn_norm,
            wq_a,
            None,
            wq_b,
            None,
            q_a_norm,
            wkv_a,
            None,
            wkv_b,
            kv_a_norm,
            wo_b,
            None,
            w_uk_t,
            w_uv,
            wq_b_rope,
            w_qk_absorbed,
            w_uk_block_diag,
            w_uv_block_diag,
            yarn_inv_freq,
            wo_a,
            hc_head.clone(),
            store,
            config,
            gpu,
            layer_kv_dtypes,
        )?;
        layers.push(layer);
    }
    Ok(layers)
}
