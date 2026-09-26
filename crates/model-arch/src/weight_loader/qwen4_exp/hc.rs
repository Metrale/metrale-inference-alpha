// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The low-rank multi-hyperconnection (mHC) weights of
//! `qwen4_exp`: the `attn_hyper_connection` and `mlp_hyper_connection` sites
//! of each layer, and the model-level `hyper_connection_mixer`.
//!
//! Owner: model-arch weight loader.
//! Invariants:
//! - Every site loaded here has `lowrank` set and NULL Sinkhorn fields
//!   (`hc_fn`, `hc_base`, `hc_scale`).
//!
//! Each site reads `hc_norm`, `input_mix_weight_down` and
//! `input_mix_weight_up`. The per-layer sites also read
//! `block_inject_weight`; the mixer does not, and its `inject_w` is NULL.

use anyhow::{Context, Result};
use metrale_config::ModelConfig;
use metrale_gpu_runtime::gpu::DevicePtr;
use metrale_model_weights::weights::WeightStore;

use metrale_model_layers::layers::qwen3_attention::{HcHeadWeights, HcLowRank, HcSiteWeights};
use metrale_model_layers::weight_map::dense;

/// 2026-09-25: One hyper-connection site under `prefix`. The callers pass
/// `with_inject = false` only for the model-level mixer.
fn load_site(
    store: &WeightStore,
    prefix: &str,
    rank: usize,
    with_inject: bool,
) -> Result<HcLowRank> {
    let g = |name: &str| -> Result<DevicePtr> {
        dense(store, &format!("{prefix}.{name}.weight"))
            .map(|d| d.weight)
            .with_context(|| format!("qwen4_exp mHC: {prefix}.{name}.weight"))
    };
    Ok(HcLowRank {
        norm_w: g("hc_norm")?,
        down_w: g("input_mix_weight_down")?,
        up_w: g("input_mix_weight_up")?,
        inject_w: if with_inject {
            g("block_inject_weight")?
        } else {
            DevicePtr::NULL
        },
        rank,
    })
}

/// 2026-09-25: A site with NULL Sinkhorn fields; `lowrank` being `Some`
/// selects the low-rank kernels (`ops/hyper_connection_dispatch.rs`).
fn site(lowrank: HcLowRank) -> HcSiteWeights {
    HcSiteWeights {
        hc_fn: DevicePtr::NULL,
        hc_base: DevicePtr::NULL,
        hc_scale: DevicePtr::NULL,
        lowrank: Some(lowrank),
    }
}

/// 2026-09-25: The attention and MLP sites of one layer, in that order.
pub(super) fn load_layer_sites(
    store: &WeightStore,
    lp: &str,
    config: &ModelConfig,
) -> Result<(HcSiteWeights, HcSiteWeights)> {
    let rank = config.hc_lowrank;
    let attn = load_site(store, &format!("{lp}.attn_hyper_connection"), rank, true)?;
    let ffn = load_site(store, &format!("{lp}.mlp_hyper_connection"), rank, true)?;
    Ok((site(attn), site(ffn)))
}

/// 2026-09-25: The model-level mixer, `hyper_connection_mixer` under
/// `embed_prefix(config)`.
pub(super) fn load_head(store: &WeightStore, config: &ModelConfig) -> Result<HcHeadWeights> {
    let prefix = format!("{}.hyper_connection_mixer", super::embed_prefix(config));
    let lowrank = load_site(store, &prefix, config.hc_lowrank, false)?;
    Ok(HcHeadWeights {
        hc_fn: DevicePtr::NULL,
        hc_base: DevicePtr::NULL,
        hc_scale: DevicePtr::NULL,
        lowrank: Some(lowrank),
    })
}
