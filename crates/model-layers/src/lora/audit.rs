// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The per-adapter audit: classify every tensor, check pairs and
//! shapes, and produce the [`AuditedAdapter`] the pack loops read.
//!
//! Owner: model-layers (lora).
//! Invariants: none beyond the types.

use std::collections::BTreeMap;

use anyhow::{Result, bail};
use metrale_config::{ModelConfig, PeftAdapterConfig};
use metrale_model_weights::weights::WeightStore;

use super::*;

/// 2026-09-25: One adapter's classified tensors: `attn` pairs for the slot
/// pool, `router` and `experts` pairs for the expert pool, and the token
/// overlay tensors.
pub(crate) struct AuditedAdapter {
    pub attn: BTreeMap<(usize, LoraModule), [Option<String>; 2]>,
    pub router: expert_pack::RouterMap,
    pub experts: expert_pack::ExpertMap,
    /// 2026-09-25: Token-overlay tensors (embed_tokens / lm_head
    /// `trainable_tokens` and `modules_to_save`), read by `stage_overlay_raw`.
    pub overlay: OverlayTensors,
}

/// 2026-09-25: Set one cell of an `[a_key, b_key]` entry; an error when the
/// cell is already set.
fn set_ab(slot: &mut Option<String>, name: &str, what: &str) -> Result<()> {
    if slot.is_some() {
        bail!("REJECT[duplicate-tensor]: two tensors map to {what}");
    }
    *slot = Some(name.to_string());
    Ok(())
}

/// 2026-09-25: Audit one adapter: `validate_peft_config`, then classify every
/// tensor (a tensor that cannot be placed is an error), then check that each
/// pair has both tensors and that A is `[r, in]` and B is `[out, r]`, then the
/// expert checks, then that every `target_modules` entry matched a pair.
/// Under `METRALE_LORA_ALLOW_PARTIAL`, `is_gdn_key` tensors and unmatched
/// targets are skipped with a warning instead.
pub(crate) fn audit_adapter(
    adapter_store: &WeightStore,
    peft: &PeftAdapterConfig,
    cfg: &ModelConfig,
    max_lora_rank: usize,
) -> Result<AuditedAdapter> {
    validate_peft_config(peft, max_lora_rank)?;

    let mut found: BTreeMap<(usize, LoraModule), [Option<String>; 2]> = BTreeMap::new();
    let mut router: expert_pack::RouterMap = BTreeMap::new();
    let mut experts: expert_pack::ExpertMap = BTreeMap::new();
    let mut overlay = OverlayTensors::default();
    let mut gdn_skipped = 0usize;
    for name in adapter_store.names() {
        // 2026-09-25: Overlay tensors are taken before `classify_key`, which
        // refuses any name without a lora_A/lora_B suffix.
        if let Some(t) = classify_overlay_key(name) {
            overlay.insert(t, name)?;
            continue;
        }
        if super::env::allow_partial_targets() && super::key::is_gdn_key(name) {
            gdn_skipped += 1;
            continue;
        }
        let (layer, target, ab) = classify_key(name, cfg)?;
        match target {
            LoraTarget::Attn(module) => {
                let entry = found.entry((layer, module)).or_default();
                set_ab(
                    &mut entry[ab as usize],
                    name,
                    &format!("layer {layer} {module:?}"),
                )?;
            }
            LoraTarget::Router => {
                let entry = router.entry(layer).or_default();
                set_ab(
                    &mut entry[ab as usize],
                    name,
                    &format!("layer {layer} router"),
                )?;
            }
            LoraTarget::Expert { n, proj } => {
                let entry = experts.entry((layer, n, proj)).or_default();
                set_ab(
                    &mut entry[ab as usize],
                    name,
                    &format!("layer {layer} expert {n} {proj:?}"),
                )?;
            }
        }
    }
    // 2026-09-25: Of the overlay tensors, only `lora_embedding_A/B` is refused.
    reject_pending_overlay(&overlay)?;
    if found.is_empty() && !expert_pack::present(&router, &experts) && overlay.is_empty() {
        bail!("REJECT[empty-adapter]: no lora_A/lora_B or overlay tensors in adapter");
    }

    for ((layer, module), pair) in &found {
        let [Some(a_key), Some(b_key)] = pair else {
            bail!(
                "REJECT[unpaired-tensor]: layer {layer} {module:?} has only one of lora_A/lora_B"
            );
        };
        let (out_dim, in_dim) = module.dims(cfg);
        let a = adapter_store.get(a_key)?;
        let b = adapter_store.get(b_key)?;
        if a.shape != vec![peft.r, in_dim] {
            bail!(
                "REJECT[shape-mismatch]: '{a_key}' is {:?}, expected [{}, {}] (r, in_dim)",
                a.shape,
                peft.r,
                in_dim
            );
        }
        if b.shape != vec![out_dim, peft.r] {
            bail!(
                "REJECT[shape-mismatch]: '{b_key}' is {:?}, expected [{}, {}] (out_dim, r)",
                b.shape,
                out_dim,
                peft.r
            );
        }
    }

    expert_pack::validate(cfg, peft, &router, &experts)?;
    if expert_pack::present(&router, &experts) {
        expert_pack::validate_shapes(adapter_store, cfg, peft, &router, &experts)?;
    }

    if gdn_skipped > 0 {
        tracing::warn!(
            "LoRA PARTIAL LOAD: skipped {gdn_skipped} GDN/linear-attention tensor(s) \
             (out_proj / in_proj_*) — these layers have no v0 delta path, so that \
             part of the adapter is NOT applied."
        );
    }

    for t in &peft.target_modules {
        let last = t.rsplit('.').next().unwrap_or(t);
        let matched = found.keys().any(|(_, m)| m.peft_name() == last)
            || (last == "gate" && !router.is_empty())
            || experts.keys().any(|(_, _, p)| p.peft_name() == last);
        if !matched {
            // 2026-09-25: An unsupported module matches no pair, so refusing
            // here would refuse every partial load `validate_peft_config`
            // allowed.
            if super::env::allow_partial_targets() {
                tracing::warn!(
                    "LoRA PARTIAL LOAD: target_modules entry '{t}' matched no \
                     adapter tensor Metrale Engine can place — skipped."
                );
                continue;
            }
            bail!(
                "REJECT[unmatched-target]: target_modules entry '{t}' matched \
                 no adapter tensor on any full-attention layer"
            );
        }
    }
    Ok(AuditedAdapter {
        attn: found,
        router,
        experts,
        overlay,
    })
}
