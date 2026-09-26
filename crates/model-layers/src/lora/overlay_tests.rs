// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: GPU-free tests for `overlay`: name classification, the tensor
//! collection, trainable-id clamping, the override set and row source, and the
//! `lora_embedding` refusal.
//!
//! Owner: model-layers (lora).
//! Invariants: none beyond the types.

use super::*;

#[test]
fn classifies_trainable_tokens_pair() {
    let base = "base_model.model.model.embed_tokens.token_adapter.base_layer.weight";
    let delta = "base_model.model.model.embed_tokens.token_adapter.trainable_tokens_delta";
    assert_eq!(
        classify_overlay_key(base),
        Some(OverlayTensor {
            module: OverlayModule::EmbedTokens,
            kind: OverlayTensorKind::Base
        })
    );
    assert_eq!(
        classify_overlay_key(delta),
        Some(OverlayTensor {
            module: OverlayModule::EmbedTokens,
            kind: OverlayTensorKind::Delta
        })
    );
}

#[test]
fn classifies_lm_head_and_multimodal_prefix() {
    let lm = "base_model.model.lm_head.token_adapter.base_layer.weight";
    assert_eq!(
        classify_overlay_key(lm),
        Some(OverlayTensor {
            module: OverlayModule::LmHead,
            kind: OverlayTensorKind::Base
        })
    );
    let mm =
        "base_model.model.model.language_model.embed_tokens.token_adapter.trainable_tokens_delta";
    assert_eq!(
        classify_overlay_key(mm),
        Some(OverlayTensor {
            module: OverlayModule::EmbedTokens,
            kind: OverlayTensorKind::Delta
        })
    );
}

#[test]
fn classifies_modules_to_save_full_weight() {
    assert_eq!(
        classify_overlay_key("base_model.model.model.embed_tokens.weight"),
        Some(OverlayTensor {
            module: OverlayModule::EmbedTokens,
            kind: OverlayTensorKind::FullSave
        })
    );
    assert_eq!(
        classify_overlay_key("base_model.model.lm_head.weight"),
        Some(OverlayTensor {
            module: OverlayModule::LmHead,
            kind: OverlayTensorKind::FullSave
        })
    );
}

#[test]
fn classifies_lora_embedding_tier2() {
    for (suffix, kind) in [
        ("lora_embedding_A", OverlayTensorKind::LoraEmbedA),
        ("lora_embedding_B", OverlayTensorKind::LoraEmbedB),
    ] {
        assert_eq!(
            classify_overlay_key(&format!("base_model.model.model.embed_tokens.{suffix}")),
            Some(OverlayTensor {
                module: OverlayModule::EmbedTokens,
                kind,
            })
        );
    }
}

#[test]
fn ordinary_lora_and_layer_weights_are_not_overlays() {
    assert_eq!(
        classify_overlay_key("base_model.model.model.layers.3.self_attn.k_proj.lora_A.weight"),
        None
    );
    assert_eq!(
        classify_overlay_key("base_model.model.model.layers.3.mlp.gate_proj.lora_B.weight"),
        None
    );
    assert_eq!(
        classify_overlay_key("base_model.model.model.layers.3.mlp.down_proj.weight"),
        None
    );
    assert_eq!(classify_overlay_key("model.embed_tokens.weight"), None);
}

#[test]
fn overlay_tensors_collects_and_rejects_dupes() {
    let mut t = OverlayTensors::default();
    assert!(t.is_empty());
    let base = "base_model.model.model.embed_tokens.token_adapter.base_layer.weight";
    t.insert(classify_overlay_key(base).unwrap(), base).unwrap();
    assert!(!t.is_empty());
    assert_eq!(t.embed_base.as_deref(), Some(base));
    let err = t
        .insert(classify_overlay_key(base).unwrap(), base)
        .unwrap_err()
        .to_string();
    assert!(err.contains("REJECT[duplicate-overlay-tensor]"), "{err}");
}

#[test]
fn overlay_tensors_tracks_lora_embedding_flag() {
    let mut t = OverlayTensors::default();
    let k = "base_model.model.model.embed_tokens.lora_embedding_A";
    t.insert(classify_overlay_key(k).unwrap(), k).unwrap();
    assert!(t.lora_embedding_seen);
    assert!(!t.is_empty());
}

#[test]
fn clamp_keeps_in_vocab_ascending() {
    let (kept, skipped) = clamp_trainable_to_vocab(&[10, 42, 99], 200, 200).unwrap();
    assert_eq!(kept, vec![10, 42, 99]);
    assert_eq!(skipped, 0);
}

#[test]
fn clamp_skips_vocab_extension_tail() {
    let (kept, skipped) = clamp_trainable_to_vocab(&[10, 100, 104], 105, 100).unwrap();
    assert_eq!(kept, vec![10]);
    assert_eq!(skipped, 2);
}

#[test]
fn clamp_rejects_index_beyond_adapter() {
    let err = clamp_trainable_to_vocab(&[500], 105, 100)
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("REJECT[trainable-index-out-of-adapter]"),
        "{err}"
    );
}

#[test]
fn clamp_rejects_nonascending_or_nontrailing_ids() {
    for ids in [&[42, 10][..], &[42, 42][..], &[10, 100, 20][..]] {
        let err = clamp_trainable_to_vocab(ids, 105, 100)
            .unwrap_err()
            .to_string();
        assert!(err.contains("REJECT[trainable-order]"), "{ids:?}: {err}");
    }
}

#[test]
fn override_set_unions_sorts_dedups() {
    let diff = vec![false, false, true, false, false, true];
    let ids = build_override_set(&diff, &[5, 1]);
    assert_eq!(ids, vec![1, 2, 5]);
}

#[test]
fn override_set_empty_when_nothing_changes() {
    let diff = vec![false, false, false];
    assert!(build_override_set(&diff, &[]).is_empty());
}

#[test]
fn no_overlay_tensors_passes_gate() {
    assert!(reject_pending_overlay(&OverlayTensors::default()).is_ok());
}

#[test]
fn trainable_tokens_now_loads_not_rejected() {
    let mut t = OverlayTensors::default();
    let k = "base_model.model.model.embed_tokens.token_adapter.base_layer.weight";
    t.insert(classify_overlay_key(k).unwrap(), k).unwrap();
    assert!(reject_pending_overlay(&t).is_ok());
}

#[test]
fn lora_embedding_hits_tier2_reject() {
    let mut t = OverlayTensors::default();
    let k = "base_model.model.model.embed_tokens.lora_embedding_A";
    t.insert(classify_overlay_key(k).unwrap(), k).unwrap();
    let err = reject_pending_overlay(&t).unwrap_err().to_string();
    assert!(
        err.contains("REJECT[lora-embedding-unimplemented]"),
        "{err}"
    );
}

#[test]
fn override_source_delta_wins_over_baked() {
    assert_eq!(override_source(5, &[7, 5, 9]), Some(1));
    assert_eq!(override_source(2, &[7, 5, 9]), None);
}
