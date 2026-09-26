// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The GPU-free half of the token overlay, which replaces whole
//! embed_tokens / lm_head rows for PEFT `trainable_tokens` and
//! `modules_to_save` adapters: tensor-name classification
//! ([`classify_overlay_key`]), the per-adapter collection ([`OverlayTensors`]),
//! and the choice of overridden rows and their sources
//! ([`clamp_trainable_to_vocab`], [`build_override_set`], [`override_source`]).
//! The device build is in [`super::overlay_build`].
//!
//! Tensor kinds:
//! - `…token_adapter.base_layer.weight` (`[R, h]`) and
//!   `…token_adapter.trainable_tokens_delta` (`[T, h]`): trainable id `k`'s
//!   row is replaced by delta row `k`, not added to.
//! - a `modules_to_save` `…embed_tokens.weight` / `lm_head.weight`: a full
//!   table; the rows that differ from the served table are replaced.
//! - `lora_embedding_A/B`: classified so the audit can refuse it.
//!
//! Owner: model-layers (lora).
//! Invariants: none beyond the types.

use anyhow::{Result, bail};

/// 2026-09-25: A base row whose largest absolute difference from the served
/// row exceeds this is overridden (`embed_rowdiff_bf16`).
pub const ROWDIFF_THRESH: f32 = 0.1;

/// 2026-09-25: The table an overlay tensor belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OverlayModule {
    /// 2026-09-25: The input embedding (a name containing `embed_tokens` or
    /// `.shared.`).
    EmbedTokens,
    /// 2026-09-25: The output projection (a name containing `lm_head`).
    LmHead,
}

/// 2026-09-25: The role an overlay tensor plays in
/// [`build_overlay`](super::overlay_build::build_overlay).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OverlayTensorKind {
    /// 2026-09-25: `token_adapter.base_layer.weight`, the adapter's `[R, h]`
    /// table.
    Base,
    /// 2026-09-25: `token_adapter.trainable_tokens_delta`, `[T, h]`
    /// replacement rows.
    Delta,
    /// 2026-09-25: A `modules_to_save` full table (`embed_tokens.weight` /
    /// `lm_head.weight`).
    FullSave,
    /// 2026-09-25: `lora_embedding_A`; refused at audit.
    LoraEmbedA,
    /// 2026-09-25: `lora_embedding_B`; refused at audit.
    LoraEmbedB,
}

/// 2026-09-25: A classified overlay tensor: its table and its role.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OverlayTensor {
    pub module: OverlayModule,
    pub kind: OverlayTensorKind,
}

/// 2026-09-25: Classify a PEFT tensor name as an overlay tensor. `None` for a
/// name without the `base_model.model.` prefix, a `lora_A`/`lora_B` weight
/// (for [`super::classify_key`]), or a name matching no overlay form. Only the
/// module name and suffix are matched, so `model.embed_tokens` and
/// `model.language_model.embed_tokens` both work. The `token_adapter` and
/// `lora_embedding` suffixes are tested before the bare `.weight` form, so
/// `…token_adapter.base_layer.weight` is not taken for a full table.
pub fn classify_overlay_key(key: &str) -> Option<OverlayTensor> {
    let stripped = key.strip_prefix("base_model.model.")?;
    if stripped.ends_with(".lora_A.weight") || stripped.ends_with(".lora_B.weight") {
        return None;
    }
    let module = overlay_module_of(stripped)?;
    let kind = if stripped.ends_with(".token_adapter.base_layer.weight") {
        OverlayTensorKind::Base
    } else if stripped.ends_with(".token_adapter.trainable_tokens_delta") {
        OverlayTensorKind::Delta
    } else if stripped.ends_with(".lora_embedding_A") {
        OverlayTensorKind::LoraEmbedA
    } else if stripped.ends_with(".lora_embedding_B") {
        OverlayTensorKind::LoraEmbedB
    } else if is_full_module_weight(stripped, module) {
        OverlayTensorKind::FullSave
    } else {
        return None;
    };
    Some(OverlayTensor { module, kind })
}

/// 2026-09-25: The table a prefix-stripped name targets; `lm_head` is tested
/// first.
fn overlay_module_of(stripped: &str) -> Option<OverlayModule> {
    if stripped.contains("lm_head") {
        Some(OverlayModule::LmHead)
    } else if stripped.contains("embed_tokens") || stripped.contains(".shared.") {
        Some(OverlayModule::EmbedTokens)
    } else {
        None
    }
}

/// 2026-09-25: True when `stripped` ends with the module's own weight
/// (`embed_tokens.weight` / `lm_head.weight`) and does not contain `.layers.`.
fn is_full_module_weight(stripped: &str, module: OverlayModule) -> bool {
    if stripped.contains(".layers.") {
        return false;
    }
    let leaf_owner = match module {
        OverlayModule::EmbedTokens => "embed_tokens.weight",
        OverlayModule::LmHead => "lm_head.weight",
    };
    stripped.ends_with(leaf_owner)
}

/// 2026-09-25: One adapter's overlay tensor names by (table, role); each
/// field is set at most once.
#[derive(Debug, Default, Clone)]
pub struct OverlayTensors {
    pub embed_base: Option<String>,
    pub embed_delta: Option<String>,
    pub embed_full: Option<String>,
    pub lmhead_base: Option<String>,
    pub lmhead_delta: Option<String>,
    pub lmhead_full: Option<String>,
    /// 2026-09-25: A `lora_embedding_A/B` tensor was seen.
    pub lora_embedding_seen: bool,
}

impl OverlayTensors {
    /// 2026-09-25: Record one classified tensor; an error when its (table,
    /// role) is already set.
    pub fn insert(&mut self, t: OverlayTensor, name: &str) -> Result<()> {
        use OverlayModule::*;
        use OverlayTensorKind::*;
        let slot = match (t.module, t.kind) {
            (_, LoraEmbedA) | (_, LoraEmbedB) => {
                self.lora_embedding_seen = true;
                return Ok(());
            }
            (EmbedTokens, Base) => &mut self.embed_base,
            (EmbedTokens, Delta) => &mut self.embed_delta,
            (EmbedTokens, FullSave) => &mut self.embed_full,
            (LmHead, Base) => &mut self.lmhead_base,
            (LmHead, Delta) => &mut self.lmhead_delta,
            (LmHead, FullSave) => &mut self.lmhead_full,
        };
        if slot.is_some() {
            bail!(
                "REJECT[duplicate-overlay-tensor]: two tensors map to {:?}/{:?}",
                t.module,
                t.kind
            );
        }
        *slot = Some(name.to_string());
        Ok(())
    }

    /// 2026-09-25: True when no overlay tensor was collected.
    pub fn is_empty(&self) -> bool {
        self.embed_base.is_none()
            && self.embed_delta.is_none()
            && self.embed_full.is_none()
            && self.lmhead_base.is_none()
            && self.lmhead_delta.is_none()
            && self.lmhead_full.is_none()
            && !self.lora_embedding_seen
    }
}

/// 2026-09-25: An error when the adapter has a `lora_embedding_A/B` tensor;
/// every other overlay tensor is loaded.
pub fn reject_pending_overlay(overlay: &OverlayTensors) -> Result<()> {
    if overlay.lora_embedding_seen {
        bail!(
            "REJECT[lora-embedding-unimplemented]: classic low-rank embedding LoRA \
             (lora_embedding_A/B) is not yet supported on the decoder path"
        );
    }
    Ok(())
}

/// 2026-09-25: Split `trainable` ids at the served vocab, keeping list order,
/// since delta row `k` belongs to the `k`-th id. Returns `(kept_ids,
/// skipped_count)`:
/// - `idx >= r` (outside the adapter's `[R, h]` table) is an error;
/// - `vocab <= idx < r` is dropped and counted;
/// - `idx < vocab` is kept.
///
/// Also an error: a kept id after a dropped one (dropped ids must be one
/// trailing run, so kept id `k` still matches delta row `k`), and a kept id
/// not greater than the previous kept id.
pub fn clamp_trainable_to_vocab(
    trainable: &[u32],
    r: usize,
    vocab: usize,
) -> Result<(Vec<u32>, usize)> {
    let mut kept = Vec::new();
    let mut skipped = 0usize;
    let mut last_kept: Option<u32> = None;
    for &idx in trainable {
        let i = idx as usize;
        if i >= r {
            bail!("REJECT[trainable-index-out-of-adapter]: id {idx} >= adapter embedding rows {r}");
        }
        if i >= vocab {
            skipped += 1;
            continue;
        }
        if skipped > 0 {
            bail!(
                "REJECT[trainable-order]: served-vocab id {idx} appears after a skipped \
                 vocab-extension id; extension ids must form one trailing suffix"
            );
        }
        if let Some(prev) = last_kept
            && idx <= prev
        {
            bail!(
                "REJECT[trainable-order]: kept id {idx} follows id {prev}; \
                 PEFT trainable-token order must be strictly ascending in the served-vocab prefix"
            );
        }
        last_kept = Some(idx);
        kept.push(idx);
    }
    Ok((kept, skipped))
}

/// 2026-09-25: The vocab rows the overlay replaces: the indices `i` with
/// `row_diff[i]` set, plus `trainable`, sorted ascending without duplicates.
pub fn build_override_set(row_diff: &[bool], trainable: &[u32]) -> Vec<u32> {
    let mut ids: Vec<u32> = row_diff
        .iter()
        .enumerate()
        .filter_map(|(i, &d)| d.then_some(i as u32))
        .collect();
    ids.extend_from_slice(trainable);
    ids.sort_unstable();
    ids.dedup();
    ids
}

/// 2026-09-25: Where an overridden id's row comes from: `Some(k)`, delta row
/// `k`, when `id` is the `k`-th trainable id (this wins over a differing base
/// row); `None`, the adapter's base row `id`.
pub fn override_source(id: u32, trainable: &[u32]) -> Option<usize> {
    trainable.iter().position(|&t| t == id)
}

#[cfg(test)]
#[path = "overlay_tests.rs"]
mod tests;
