// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Loads the DeepSeek-V4 multi-token-prediction (MTP) draft module
//! from the `mtp.0.*` tensors.
//!
//! The body is a V4 transformer layer built by `assemble_layer` from the
//! `mtp.0` prefix. The MTP-specific pieces are the input combiner
//! (`enorm`/`hnorm`, `e_proj`/`h_proj`), the final `norm` and the module's own
//! HC head. The token embedding and the LM head are not loaded here: the
//! proposer, `crate::deepseek_v4_mtp::DeepseekV4MtpHead`, takes the target
//! model's at build time.
//!
//! Owner: model-arch weight loader.
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_cache::kv_cache::KvCacheDtype;
use metrale_config::ModelConfig;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_weights::weights::WeightStore;

use metrale_model_layers::layer::TransformerLayer;
use metrale_model_layers::layers::qwen3_attention::HcHeadWeights;
use metrale_model_layers::weight_map::DenseWeight;
use metrale_model_layers::weight_map::quant_helpers::dense_auto;

/// 2026-09-25: A loaded DeepSeek-V4 MTP draft module: the V4 layer body plus the
/// MTP input combiner, final norm and HC head.
#[allow(dead_code)]
pub struct DeepseekV4MtpModule {
    /// 2026-09-25: The V4 layer body, built by `assemble_layer` from the `mtp.0` prefix.
    pub body: Box<dyn TransformerLayer>,
    /// 2026-09-25: RMSNorm weight applied to the token embedding before `e_proj`.
    pub enorm: DenseWeight,
    /// 2026-09-25: RMSNorm weight applied to the target's hidden state before `h_proj`.
    pub hnorm: DenseWeight,
    /// 2026-09-25: `[hidden, hidden]` projection of the normed embedding; summed with
    /// the `h_proj` branch.
    pub e_proj: DenseWeight,
    /// 2026-09-25: `[hidden, hidden]` projection of the normed hidden state; summed
    /// with the `e_proj` branch.
    pub h_proj: DenseWeight,
    /// 2026-09-25: Final RMSNorm weight, applied before the target's LM head.
    pub norm: DenseWeight,
    /// 2026-09-25: The module's own HC head (`mtp.0.hc_head_*`). The body is built
    /// with `layer_idx = num_hidden_layers`, so it is not the last model layer
    /// and never collapses the HC streams; the proposer does that with these
    /// weights. The body holds a clone of the same pointers. `Some` in every
    /// returned module: `assemble_layer` refuses `hc_mult == 0`.
    pub hc_head: Option<HcHeadWeights>,
}

/// 2026-09-25: Loads the DeepSeek-V4 MTP draft module.
///
/// Returns `Ok(None)`, loading nothing, when `num_mtp_modules == 0` or the
/// store has no `mtp.0.enorm.weight`. Any other missing `mtp.0.*` tensor is an
/// error.
pub fn load_v4_mtp_module(
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    layer_kv_dtypes: &[KvCacheDtype],
) -> Result<Option<DeepseekV4MtpModule>> {
    if config.num_mtp_modules == 0 {
        return Ok(None);
    }
    if !store.contains("mtp.0.enorm.weight") {
        tracing::info!(
            "DeepSeek-V4: num_mtp_modules={} but no mtp.0.* tensors in checkpoint — MTP disabled",
            config.num_mtp_modules
        );
        return Ok(None);
    }

    let prefix = "mtp.0";
    let ap = format!("{prefix}.attn");
    let null = DenseWeight {
        weight: DevicePtr::NULL,
    };

    // 2026-09-25: The body loads the same tensors as the `o_lora_rank > 0` branch
    // of `load_all_layers`, whatever `o_lora_rank` is, and its absorption views
    // stay NULL.
    let input_norm = dense_auto(store, &format!("{prefix}.attn_norm.weight"), gpu)?;
    let post_attn_norm = dense_auto(store, &format!("{prefix}.ffn_norm.weight"), gpu)?;
    let wq_a = dense_auto(store, &format!("{ap}.wq_a.weight"), gpu)?;
    let wq_b = dense_auto(store, &format!("{ap}.wq_b.weight"), gpu)?;
    let q_a_norm = dense_auto(store, &format!("{ap}.q_norm.weight"), gpu)?;
    let wkv_a = dense_auto(store, &format!("{ap}.wkv.weight"), gpu)?;
    let kv_a_norm = dense_auto(store, &format!("{ap}.kv_norm.weight"), gpu)?;
    let wo_a = dense_auto(store, &format!("{ap}.wo_a.weight"), gpu)?;
    let wo_b = dense_auto(store, &format!("{ap}.wo_b.weight"), gpu)?;

    // 2026-09-25: The module's own HC head (`mtp.0.hc_head_*`), separate from the
    // model-level head that `load_all_layers` gives the main layers.
    let hc_head = if config.hc_mult > 0 {
        let hc = config.hc_mult;
        let hc_dim = hc * config.hidden_size;
        let head_fn = super::assemble::load_hc_f32(
            store,
            &[format!("{prefix}.hc_head_fn")],
            hc * hc_dim,
            gpu,
        )?;
        let head_base =
            super::assemble::load_hc_f32(store, &[format!("{prefix}.hc_head_base")], hc, gpu)?;
        let head_scale =
            super::assemble::load_hc_f32(store, &[format!("{prefix}.hc_head_scale")], 1, gpu)?;
        Some(HcHeadWeights {
            hc_fn: head_fn,
            hc_base: head_base,
            hc_scale: head_scale,
            // 2026-09-25: `None` selects the Sinkhorn head mixer.
            lowrank: None,
        })
    } else {
        None
    };

    let mut yarn_inv_freq = DevicePtr::NULL;
    yarn_inv_freq = super::compute::ensure_yarn_inv_freq(&mut yarn_inv_freq, config, gpu)?;

    // 2026-09-25: `layer_idx = num_hidden_layers` makes the body neither the
    // first nor the last model layer, so it neither expands nor collapses the HC
    // streams; the proposer does both.
    let body = super::assemble::assemble_layer(
        config.num_hidden_layers,
        prefix,
        true,
        input_norm,
        post_attn_norm,
        wq_a,
        None,
        wq_b,
        None,
        q_a_norm,
        wkv_a,
        None,
        null,
        kv_a_norm,
        wo_b,
        None,
        null,
        null,
        null,
        null,
        null,
        null,
        yarn_inv_freq,
        wo_a,
        hc_head.clone(),
        store,
        config,
        gpu,
        layer_kv_dtypes,
    )?;

    // 2026-09-25: The norms are loaded unchanged: the proposer applies them with
    // `rms_norm_vanilla`, which scales by `w`, not `1 + w`. `dense_auto` returns every
    // tensor here as BF16, dequantizing FP8 or packed NVFP4.
    let enorm = dense_auto(store, &format!("{prefix}.enorm.weight"), gpu)?;
    let hnorm = dense_auto(store, &format!("{prefix}.hnorm.weight"), gpu)?;
    let norm = dense_auto(store, &format!("{prefix}.norm.weight"), gpu)?;
    let e_proj = dense_auto(store, &format!("{prefix}.e_proj.weight"), gpu)?;
    let h_proj = dense_auto(store, &format!("{prefix}.h_proj.weight"), gpu)?;

    tracing::info!(
        "DeepSeek-V4 MTP module loaded: reused V4 body (MLA + mHC + 256-expert MoE) \
         + combiner (enorm/hnorm + e_proj/h_proj) + final norm"
    );

    Ok(Some(DeepseekV4MtpModule {
        body,
        enorm,
        hnorm,
        e_proj,
        h_proj,
        norm,
        hc_head,
    }))
}
