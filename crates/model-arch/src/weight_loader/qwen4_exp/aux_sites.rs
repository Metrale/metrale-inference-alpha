// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Per-layer attach helpers for `qwen4_exp` (QSA indexer, PLE,
//! mHC sites) and the final-norm placeholder.
//!
//! Owner: model-arch weight loader.
//! Invariants: none beyond the types.

use anyhow::{Context, Result};
use metrale_config::ModelConfig;
use metrale_gpu_runtime::gpu::GpuBackend;
use metrale_model_weights::weights::WeightStore;

use super::{mixer_prefix, ones_norm};
use metrale_model_layers::layer::TransformerLayer;
use metrale_model_layers::weight_map::DenseWeight;

/// 2026-09-25: Attach a QSA indexer to a layer whose store has
/// `self_attn.indexer.index_qk_proj.weight`, when `index_topk > 0` and
/// `METRALE_QSA_DISABLE` is not `1`. Other layers are left without one. An
/// error if such a layer is not a `Qwen3AttentionLayer`.
pub(super) fn attach_qsa(
    layer: &mut Box<dyn TransformerLayer>,
    i: usize,
    lp: &str,
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
) -> Result<()> {
    if config.index_topk > 0
        && std::env::var("METRALE_QSA_DISABLE").as_deref() != Ok("1")
        && store.contains(&format!("{lp}.self_attn.indexer.index_qk_proj.weight"))
    {
        // 2026-09-25: The indexer holds the store's device pointers; it copies
        // nothing.
        let up = |name: &str| -> Result<metrale_gpu_runtime::gpu::DevicePtr> {
            Ok(store
                .get(&format!("{lp}.self_attn.indexer.{name}"))
                .with_context(|| format!("qwen4_exp layer {i}: indexer {name}"))?
                .ptr)
        };
        let qsa = metrale_model_layers::layers::qsa::QsaIndexer::new(
            up("index_qk_proj.weight")?,
            up("q_layernorm.weight")?,
            up("k_layernorm.weight")?,
            config.index_n_heads,
            config.index_head_dim,
            config.index_compress_ratio,
            config.index_topk,
            (config.head_dim as f64 * config.partial_rotary_factor) as usize,
            config.rope_theta as f32,
            config.rms_norm_eps as f32,
            config.hidden_size,
            config.num_key_value_heads,
            config.head_dim,
            gpu,
        )
        .with_context(|| format!("qwen4_exp layer {i}: QSA indexer"))?;
        let any = layer
            .as_any_mut()
            .ok_or_else(|| anyhow::anyhow!("qwen4_exp layer {i}: no as_any_mut for QSA"))?;
        any.downcast_mut::<metrale_model_layers::layers::Qwen3AttentionLayer>()
            .ok_or_else(|| {
                anyhow::anyhow!("qwen4_exp layer {i}: indexer on a non-attention layer")
            })?
            .set_qsa(qsa);
    }
    Ok(())
}

/// 2026-09-25: Hand a loaded PLE layer to its host, which must be a
/// `Qwen3SsmLayer` (GDN); any other layer is an error.
pub(super) fn attach_ple(
    layer: &mut Box<dyn TransformerLayer>,
    i: usize,
    p: metrale_model_layers::layers::ple::PleLayer,
) -> Result<()> {
    let any = layer
        .as_any_mut()
        .ok_or_else(|| anyhow::anyhow!("qwen4_exp layer {i}: no as_any_mut for PLE"))?;
    let l = any
        .downcast_mut::<metrale_model_layers::layers::Qwen3SsmLayer>()
        .ok_or_else(|| {
            anyhow::anyhow!(
                "qwen4_exp layer {i} carries PLE but is not a GDN layer; \
                 the injection has nowhere to go"
            )
        })?;
    l.set_ple(p);
    Ok(())
}

pub(super) fn final_norm_placeholder(
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
) -> Result<DenseWeight> {
    let mixer = mixer_prefix(config);
    anyhow::ensure!(
        store.contains(&format!("{mixer}.hc_norm.weight")),
        "qwen4_exp: no `{mixer}.hc_norm.weight` — this architecture is \
         supposed to keep its final normalization in the hyper-connection \
         mixer, and it is not there. Refusing rather than guessing."
    );
    tracing::warn!(
        "qwen4_exp: final norm is a PLACEHOLDER. The real one is \
         `{mixer}.hc_norm` [{}], applied while collapsing the {} residual \
         streams — that is mHC work, not a final-norm substitution.",
        config.hc_mult * config.hidden_size,
        config.hc_mult,
    );
    ones_norm(config.hidden_size, gpu)
}

/// 2026-09-25: Attach both hyper-connection sites to a built layer, which
/// must be a `Qwen3AttentionLayer` or a `Qwen3SsmLayer`; anything else is an
/// error, never a skip.
pub(super) fn attach_hc(
    layer: &mut Box<dyn TransformerLayer>,
    idx: usize,
    attn: metrale_model_layers::layers::qwen3_attention::HcSiteWeights,
    ffn: metrale_model_layers::layers::qwen3_attention::HcSiteWeights,
    head: Option<metrale_model_layers::layers::qwen3_attention::HcHeadWeights>,
    config: &ModelConfig,
) -> Result<()> {
    use metrale_model_layers::layers::qwen3_attention::HcWeights;
    let any = layer.as_any_mut().ok_or_else(|| {
        anyhow::anyhow!("qwen4_exp layer {idx}: no as_any_mut, cannot attach mHC weights")
    })?;
    let w = HcWeights {
        attn,
        ffn,
        head,
        hc_mult: config.hc_mult,
        sinkhorn_iters: 0,
        hc_eps: config.rms_norm_eps as f32,
        // 2026-09-25: Model layer indices, not attention-layer ones.
        is_first_model_layer: idx == 0,
        is_last_model_layer: idx + 1 == config.num_hidden_layers,
    };
    if let Some(l) = any.downcast_mut::<metrale_model_layers::layers::Qwen3AttentionLayer>() {
        l.set_hc_weights(w);
        return Ok(());
    }
    if let Some(l) = any.downcast_mut::<metrale_model_layers::layers::Qwen3SsmLayer>() {
        l.set_hc_weights(w);
        return Ok(());
    }
    anyhow::bail!(
        "qwen4_exp layer {idx}: mHC weights have nowhere to go — the layer is \
         neither Qwen3AttentionLayer nor Qwen3SsmLayer"
    )
}
