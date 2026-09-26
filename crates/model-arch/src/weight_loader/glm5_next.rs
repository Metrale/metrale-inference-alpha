// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: GLM-5.3 (`glm5_next`) tensor accounting: `classify` gives the
//! role of a checkpoint tensor name, and `account` totals a name list by role.
//! Nothing is loaded.
//!
//! `classify` returns `None` for a name it does not know, so an unrecognised
//! tensor is reported rather than skipped. The counts below are from the
//! reference checkpoint's name fixture
//! (`crates/model-engine/tests/fixtures/glm53-nvfp4-9e0d74e3-patterns.tsv`),
//! whose test fails if any name in it is unknown.
//!
//! Owner: model-arch weight loader.
//! Invariants: none beyond the types.

use std::collections::BTreeMap;

/// 2026-09-25: What a tensor is for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TensorRole {
    Embedding,
    LmHead,
    FinalNorm,
    /// 2026-09-25: KDA linear-attention projections. The reference checkpoint
    /// has 34 KDA layers.
    KdaProjection,
    KdaConv,
    KdaDecay,
    KdaGate,
    KdaNorm,
    /// 2026-09-25: Sparse-MLA projections, plus every mixer's `o_proj`. The
    /// reference checkpoint has 12 sparse-MLA layers.
    MlaProjection,
    MlaNorm,
    /// 2026-09-25: The DSA top-k indexer of a sparse-MLA layer
    /// (`self_attn.indexer.*`, its `k_norm` excepted).
    Indexer,
    IndexerNorm,
    MoeRouter,
    MoeExpert,
    MoeShared,
    /// 2026-09-25: Dense FFN: `mlp.gate_proj/up_proj/down_proj` with no expert
    /// index. The reference checkpoint has 3 dense-FFN layers.
    DenseFfn,
    /// 2026-09-25: `input_layernorm` and `post_attention_layernorm`.
    LayerNorm,
    HyperConnection,
    /// 2026-09-25: The MTP layer's `eh_proj`. The MTP layer is a numbered
    /// `layers.#` layer; the reference checkpoint has no `mtp.0.*` tensor.
    MtpProjection,
    MtpNorm,
    /// 2026-09-25: The vision tower (`model.visual.*`, `model.vision*`), not part
    /// of the text model.
    Vision,
}

impl TensorRole {
    /// 2026-09-25: Norm-family roles. `norm_sanity_pass` in
    /// `glm5next_tensor_accounting.rs` requires every text tensor whose name
    /// contains `norm` to land in one of them.
    pub fn is_norm(self) -> bool {
        matches!(
            self,
            TensorRole::FinalNorm
                | TensorRole::KdaNorm
                | TensorRole::MlaNorm
                | TensorRole::IndexerNorm
                | TensorRole::LayerNorm
                | TensorRole::MtpNorm
        )
    }

    /// 2026-09-25: Every role except `Vision`.
    pub fn is_text_model(self) -> bool {
        !matches!(self, TensorRole::Vision)
    }
}

/// 2026-09-25: Replace every all-numeric path segment, and the fixture's
/// placeholders `N` and `E`, with `#`, so real and fixture names give one key.
fn normalize(name: &str) -> String {
    name.split('.')
        .map(|seg| {
            if seg.is_empty() {
                seg
            } else if seg.chars().all(|c| c.is_ascii_digit()) || seg == "N" || seg == "E" {
                "#"
            } else {
                seg
            }
        })
        .collect::<Vec<_>>()
        .join(".")
}

/// 2026-09-25: Classify one checkpoint tensor name. `None` means unrecognised;
/// callers must treat it as an error, not as a tensor to skip.
pub fn classify(name: &str) -> Option<TensorRole> {
    let n = normalize(name);
    let s = n.as_str();

    if s == "lm_head.weight" {
        return Some(TensorRole::LmHead);
    }
    if s == "model.language_model.embed_tokens.weight" {
        return Some(TensorRole::Embedding);
    }
    if s == "model.language_model.norm.weight" {
        return Some(TensorRole::FinalNorm);
    }
    if s.starts_with("model.visual.") || s.starts_with("model.vision") {
        return Some(TensorRole::Vision);
    }

    let rest = s.strip_prefix("model.language_model.layers.#.")?;

    // 2026-09-25: MTP-layer names are classified by name alone; `account` finds
    // the MTP layer's index through `is_mtp_only_name`.
    match rest {
        "eh_proj.weight" => return Some(TensorRole::MtpProjection),
        "enorm.weight" | "hnorm.weight" | "shared_head.norm.weight" => {
            return Some(TensorRole::MtpNorm);
        }
        _ => {}
    }

    match rest {
        "self_attn.q_proj.weight"
        | "self_attn.k_proj.weight"
        | "self_attn.v_proj.weight"
        | "self_attn.b_proj.weight"
        | "self_attn.f_a_proj.weight"
        | "self_attn.f_b_proj.weight"
        | "self_attn.g_a_proj.weight"
        | "self_attn.g_b_proj.weight" => return Some(TensorRole::KdaProjection),
        "self_attn.q_conv1d.weight" | "self_attn.k_conv1d.weight" | "self_attn.v_conv1d.weight" => {
            return Some(TensorRole::KdaConv);
        }
        "self_attn.A_log" | "self_attn.dt_bias" => return Some(TensorRole::KdaDecay),
        "self_attn.o_norm.weight" => return Some(TensorRole::KdaNorm),
        _ => {}
    }

    match rest {
        "self_attn.q_a_proj.weight"
        | "self_attn.q_b_proj.weight"
        | "self_attn.kv_a_proj_with_mqa.weight"
        | "self_attn.kv_b_proj.weight" => return Some(TensorRole::MlaProjection),
        "self_attn.q_a_layernorm.weight" | "self_attn.kv_a_layernorm.weight" => {
            return Some(TensorRole::MlaNorm);
        }
        // 2026-09-25: Every mixer has an `o_proj` (46 in the reference
        // checkpoint: 34 KDA, 12 sparse-MLA), so it does not identify the family.
        "self_attn.o_proj.weight" => return Some(TensorRole::MlaProjection),
        _ => {}
    }

    if let Some(idx) = rest.strip_prefix("self_attn.indexer.") {
        return Some(match idx {
            "k_norm.weight" | "k_norm.bias" => TensorRole::IndexerNorm,
            _ => TensorRole::Indexer,
        });
    }

    if rest.starts_with("hc_") {
        return Some(TensorRole::HyperConnection);
    }

    if rest == "input_layernorm.weight" || rest == "post_attention_layernorm.weight" {
        return Some(TensorRole::LayerNorm);
    }

    if let Some(mlp) = rest.strip_prefix("mlp.") {
        if mlp.starts_with("gate.") || mlp == "gate.weight" {
            return Some(TensorRole::MoeRouter);
        }
        if mlp.starts_with("experts.#.") {
            return Some(TensorRole::MoeExpert);
        }
        if mlp.starts_with("shared_experts.") {
            return Some(TensorRole::MoeShared);
        }
        if mlp.starts_with("gate_proj.")
            || mlp.starts_with("up_proj.")
            || mlp.starts_with("down_proj.")
        {
            return Some(TensorRole::DenseFfn);
        }
    }

    None
}

/// 2026-09-25: Whether `name` is an MTP-layer tensor of a numbered layer
/// (`eh_proj`, `enorm`, `hnorm`, `shared_head.norm`). `account` uses it to find
/// the MTP layer's index. The reference checkpoint has no `mtp.0.*` tensor.
pub fn is_mtp_only_name(name: &str) -> bool {
    let n = normalize(name);
    let Some(rest) = n.strip_prefix("model.language_model.layers.#.") else {
        return false;
    };
    matches!(
        rest,
        "eh_proj.weight" | "enorm.weight" | "hnorm.weight" | "shared_head.norm.weight"
    )
}

/// 2026-09-25: The result of [`account`].
#[derive(Debug, Default)]
pub struct Accounting {
    pub total: usize,
    pub by_role: BTreeMap<String, usize>,
    /// 2026-09-25: Names `classify` refused.
    pub unknown: Vec<String>,
    /// 2026-09-25: Layer indices that carry MTP-layer tensors, ascending.
    pub mtp_layers: Vec<usize>,
}

/// 2026-09-25: Total a list of `(tensor_name, count)` pairs by role.
pub fn account<'a, I>(names: I) -> Accounting
where
    I: IntoIterator<Item = (&'a str, usize)>,
{
    let mut acc = Accounting::default();
    let mut mtp = std::collections::BTreeSet::new();
    for (name, count) in names {
        acc.total += count;
        match classify(name) {
            Some(role) => {
                *acc.by_role.entry(format!("{role:?}")).or_insert(0) += count;
            }
            None => acc.unknown.push(name.to_string()),
        }
        if is_mtp_only_name(name)
            && let Some(i) = layer_index(name)
        {
            mtp.insert(i);
        }
    }
    acc.mtp_layers = mtp.into_iter().collect();
    acc
}

/// 2026-09-25: The number after the first `layers` segment, or `None` when it
/// is absent or not a number (as in the fixture's `layers.N.`).
pub fn layer_index(name: &str) -> Option<usize> {
    let mut it = name.split('.');
    while let Some(seg) = it.next() {
        if seg == "layers" {
            return it.next().and_then(|s| s.parse().ok());
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_numeric_and_canonical_segments_alike() {
        assert_eq!(
            normalize("model.language_model.layers.7.self_attn.o_proj.weight"),
            normalize("model.language_model.layers.N.self_attn.o_proj.weight")
        );
        assert_eq!(
            normalize("model.language_model.layers.45.mlp.experts.12.down_proj.weight"),
            normalize("model.language_model.layers.N.mlp.experts.E.down_proj.weight")
        );
    }

    #[test]
    fn mtp_is_found_by_layer_name_not_by_mtp_prefix() {
        assert!(is_mtp_only_name(
            "model.language_model.layers.45.eh_proj.weight"
        ));
        assert!(!is_mtp_only_name("model.language_model.layers.3.eh_proj"));
        // 2026-09-25: An `mtp.0.*` name is not an MTP-layer name.
        assert!(!is_mtp_only_name("model.layers.mtp.0.eh_proj.weight"));
        assert_eq!(
            layer_index("model.language_model.layers.45.eh_proj.weight"),
            Some(45)
        );
    }

    #[test]
    fn unknown_tensor_is_refused_not_skipped() {
        assert_eq!(
            classify("model.language_model.layers.4.self_attn.wat"),
            None
        );
        let acc = account([("model.language_model.layers.4.self_attn.wat", 1)]);
        assert_eq!(acc.unknown.len(), 1);
    }

    #[test]
    fn norm_tensors_land_in_norm_roles() {
        for n in [
            "model.language_model.norm.weight",
            "model.language_model.layers.0.self_attn.o_norm.weight",
            "model.language_model.layers.3.self_attn.q_a_layernorm.weight",
            "model.language_model.layers.3.self_attn.indexer.k_norm.weight",
            "model.language_model.layers.5.input_layernorm.weight",
            "model.language_model.layers.45.enorm.weight",
        ] {
            let r = classify(n).unwrap_or_else(|| panic!("unclassified: {n}"));
            assert!(r.is_norm(), "{n} classified as non-norm {r:?}");
        }
    }
}
