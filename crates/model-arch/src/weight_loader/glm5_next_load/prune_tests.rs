// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Which store tensors `Glm5NextWeightLoader::prune_after_load` frees.
//!
//! Owner: model-arch weight loader (GLM-5.3).
//! Invariants: none beyond the types.

use super::is_reuploaded;

#[test]
fn prunes_only_the_reuploaded_layer_tensors() {
    let n = 45;
    // 2026-09-25: Text-stack tensors the binders re-uploaded: freed.
    assert!(is_reuploaded(
        "model.language_model.layers.0.self_attn.q_proj.weight",
        n
    ));
    assert!(is_reuploaded(
        "model.language_model.layers.44.mlp.gate.weight",
        n
    ));
    assert!(is_reuploaded(
        "model.language_model.layers.3.hc_attn_fn.weight",
        n
    ));
    // 2026-09-25: Routed experts: kept, since a U8 expert is bound zero-copy.
    assert!(!is_reuploaded(
        "model.language_model.layers.7.mlp.experts.12.down_proj.weight",
        n
    ));
    // 2026-09-25: The MTP block `layers.{n}`: kept for
    // `glm5_next_mtp::load_glm5next_mtp_module`.
    assert!(!is_reuploaded(
        "model.language_model.layers.45.self_attn.q_proj.weight",
        n
    ));
    // 2026-09-25: Not a layer tensor.
    assert!(!is_reuploaded(
        "model.language_model.embed_tokens.weight",
        n
    ));
    assert!(!is_reuploaded("lm_head.weight", n));
    // 2026-09-25: A non-numeric layer index never matches.
    assert!(!is_reuploaded("model.language_model.layers.x.foo", n));
}
