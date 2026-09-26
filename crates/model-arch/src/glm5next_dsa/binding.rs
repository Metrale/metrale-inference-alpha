// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The tensor contract of one GLM-5.3 DSA block: the 14 `self_attn` tensors
//! [`dsa_tensor_specs`] lists, their dtype and unsharded shape, and a verifier.
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants:
//! - [`verify_dsa_block`] returns `Ok` only when every spec tensor is present with the
//!   spec's dtype, shape and BF16 byte length, and the source holds no other `self_attn.*`
//!   tensor.
//!
//! It uses the tensor types of [`crate::glm5next_kda::binding`]. Shapes are computed from
//! [`Glm5NextDsaConfig`].
//!
//! # What the table catches
//!
//! * `indexer.k_norm` is a LayerNorm with a bias, and the bias is a spec entry, so a
//!   checkpoint without it fails the bind.
//! * `index_kpool_compress_ape` is `[index_kpool, index_head_dim]`, indexed by pool slot.
//! * `q_b_proj` has `qk_head_dim` rows per head and `kv_b_proj` has
//!   `qk_nope_head_dim + v_head_dim` (256 and 512 on GLM-5.3). Both are
//!   `[heads * w, lora]`, so only the shape check tells a swapped width apart.
//! * NoPE: `kv_a_proj_with_mqa` has `kv_cache_dim()` rows, which is `kv_lora_rank` (512),
//!   not 512 + a rope section.

use std::collections::BTreeSet;

use anyhow::{Result, bail};

use super::Glm5NextDsaConfig;
use crate::glm5next_kda::binding::{KdaDtype as Dtype, KdaTensorSource as TensorSource};

/// 2026-09-25: One expected tensor: layer-relative name, dtype, and full (unsharded) shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DsaSpec {
    pub name: String,
    pub dtype: Dtype,
    pub shape: Vec<usize>,
}

/// 2026-09-25: Every `self_attn` tensor a DSA block has, and the only ones it may have.
/// `full_heads` is the unsharded head count: shapes are checked against the full tensors,
/// which `build::build_dsa_weights` shards afterwards.
pub fn dsa_tensor_specs(cfg: &Glm5NextDsaConfig, full_heads: usize) -> Vec<DsaSpec> {
    let x = cfg.hidden;
    let ql = cfg.q_lora_rank;
    let kvl = cfg.kv_lora_rank;
    let qk = cfg.qk_head_dim();
    let kvb = cfg.qk_nope_head_dim + cfg.v_head_dim;
    let ihd = cfg.index_head_dim;
    let ih = cfg.index_heads;

    let s = |name: &str, shape: Vec<usize>| DsaSpec {
        name: name.to_string(),
        dtype: Dtype::Bf16,
        shape,
    };

    vec![
        s("self_attn.q_a_proj.weight", vec![ql, x]),
        s("self_attn.q_a_layernorm.weight", vec![ql]),
        s("self_attn.q_b_proj.weight", vec![full_heads * qk, ql]),
        s(
            "self_attn.kv_a_proj_with_mqa.weight",
            vec![cfg.kv_cache_dim(), x],
        ),
        s("self_attn.kv_a_layernorm.weight", vec![kvl]),
        s("self_attn.kv_b_proj.weight", vec![full_heads * kvb, kvl]),
        s(
            "self_attn.o_proj.weight",
            vec![x, full_heads * cfg.v_head_dim],
        ),
        s("self_attn.indexer.wq_b.weight", vec![ih * ihd, ql]),
        s("self_attn.indexer.wk.weight", vec![ihd, x]),
        s("self_attn.indexer.k_norm.weight", vec![ihd]),
        s("self_attn.indexer.k_norm.bias", vec![ihd]),
        s("self_attn.indexer.weights_proj.weight", vec![ih, x]),
        s("self_attn.indexer.index_kpool_compress_gate", vec![ihd, x]),
        s(
            "self_attn.indexer.index_kpool_compress_ape",
            vec![cfg.index_kpool, ihd],
        ),
    ]
}

/// 2026-09-25: What a successful bind saw.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DsaBindReport {
    pub bound: usize,
    pub total_bytes: usize,
    /// 2026-09-25: Always empty: [`verify_dsa_block`] fails when the source has an
    /// unclaimed `self_attn.*` tensor.
    pub unclaimed: Vec<String>,
}

/// 2026-09-25: Verify one DSA block against the spec table.
///
/// Validates `cfg`, then fails unless every spec is present with its dtype, shape and
/// BF16 byte length and no `self_attn.*` tensor is left unclaimed.
pub fn verify_dsa_block(
    cfg: &Glm5NextDsaConfig,
    full_heads: usize,
    source: &dyn TensorSource,
) -> Result<DsaBindReport> {
    cfg.validate()?;
    let specs = dsa_tensor_specs(cfg, full_heads);
    let mut total_bytes = 0usize;

    for spec in &specs {
        let Some(raw) = source.get(&spec.name) else {
            bail!("DSA bind: missing required tensor `{}`", spec.name);
        };
        if raw.dtype != spec.dtype {
            bail!(
                "DSA bind: `{}` is {:?}, expected {:?}",
                spec.name,
                raw.dtype,
                spec.dtype
            );
        }
        if raw.shape != spec.shape {
            bail!(
                "DSA bind: `{}` has shape {:?}, expected {:?}",
                spec.name,
                raw.shape,
                spec.shape
            );
        }
        let elems: usize = spec.shape.iter().product();
        if raw.bytes.len() != elems * 2 {
            bail!(
                "DSA bind: `{}` carries {} bytes, expected {} for {:?} BF16",
                spec.name,
                raw.bytes.len(),
                elems * 2,
                spec.shape
            );
        }
        total_bytes += raw.bytes.len();
    }

    let claimed: BTreeSet<&str> = specs.iter().map(|s| s.name.as_str()).collect();
    let unclaimed: Vec<String> = source
        .names()
        .into_iter()
        .filter(|n| n.starts_with("self_attn.") && !claimed.contains(n.as_str()))
        .collect();
    if !unclaimed.is_empty() {
        bail!(
            "DSA bind: {} unclaimed self_attn tensor(s), first: {:?}. An unexpected \
             attention tensor means the architecture moved — do not skip it.",
            unclaimed.len(),
            &unclaimed[..unclaimed.len().min(5)]
        );
    }

    Ok(DsaBindReport {
        bound: specs.len(),
        total_bytes,
        unclaimed,
    })
}

#[cfg(test)]
mod tests;
