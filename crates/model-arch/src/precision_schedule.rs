// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::doc_lazy_continuation)]
#![allow(clippy::doc_overindented_list_items)]

//! 2026-09-25: A per-(layer, role) weight precision schedule: [`Role`] tags a
//! tensor, [`Dtype`] is a requested precision, and [`PrecisionSchedule`]
//! resolves a (layer, role) pair to a dtype.
//!
//! The only schedule in use is `PrecisionSchedule::default()`, which
//! `ModelWeightLoader::precision_schedule` returns for every loader
//! (`weight_loader/mod.rs`); it resolves every lookup to `Dtype::Inherit`. The
//! qwen35 loader only logs a schedule that has an override
//! (`weight_loader/qwen35/load_layers.rs`), and no loader calls `dtype_for`.
//!
//! Owner: model-arch (weight loading).
//! Invariants:
//! - `dtype_for` resolves in the order: sensitive-layer dtype (Attention,
//!   Expert and SharedExpert tensors only), then the role's dtype, then the
//!   global default; `Inherit` at a step falls through to the next.

use std::collections::BTreeSet;

/// 2026-09-25: Semantic role of a tensor, used for precision lookups. Adding
/// a role means extending the matches in `Role::name`, `Role::from_str` and
/// `PrecisionSchedule::dtype_for`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Role {
    /// 2026-09-25: MoE router gate (hidden × num_experts).
    Router,
    /// 2026-09-25: Final unembedding (hidden × vocab).
    LmHead,
    /// 2026-09-25: Token embedding (vocab × hidden).
    Embedding,
    /// 2026-09-25: Attention Q/K/V/O projection.
    Attention,
    /// 2026-09-25: Expert FFN (gate / up / down per expert).
    Expert,
    /// 2026-09-25: Shared-expert FFN, when the model has one.
    SharedExpert,
    /// 2026-09-25: Layer norm scales (RMSNorm `weight`).
    Norm,
}

impl Role {
    pub fn name(&self) -> &'static str {
        match self {
            Role::Router => "router",
            Role::LmHead => "lm_head",
            Role::Embedding => "embedding",
            Role::Attention => "attention",
            Role::Expert => "expert",
            Role::SharedExpert => "shared_expert",
            Role::Norm => "norm",
        }
    }

    /// 2026-09-25: Parse a role tag (the inverse of `name`). Returns `None`
    /// for an unknown name.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "router" => Some(Role::Router),
            "lm_head" => Some(Role::LmHead),
            "embedding" => Some(Role::Embedding),
            "attention" => Some(Role::Attention),
            "expert" => Some(Role::Expert),
            "shared_expert" => Some(Role::SharedExpert),
            "norm" => Some(Role::Norm),
            _ => None,
        }
    }
}

/// 2026-09-25: Target precision for a tensor. `Inherit` means no override;
/// the other variants request that precision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    /// 2026-09-25: No override: the loader keeps its own format detection.
    Inherit,
    Bf16,
    Fp8,
    Nvfp4,
}

impl Dtype {
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "inherit" => Some(Dtype::Inherit),
            "bf16" => Some(Dtype::Bf16),
            "fp8" => Some(Dtype::Fp8),
            "nvfp4" => Some(Dtype::Nvfp4),
            _ => None,
        }
    }

    /// 2026-09-25: True for every variant but `Inherit`.
    pub fn is_override(&self) -> bool {
        !matches!(self, Dtype::Inherit)
    }
}

/// 2026-09-25: Per-(layer, role) precision schedule: one dtype per role, a
/// global default, and a set of sensitive layers with their own dtype.
#[derive(Debug, Clone)]
pub struct PrecisionSchedule {
    /// 2026-09-25: Dtype for any (layer, role) that nothing more specific
    /// overrides.
    default: Dtype,
    /// 2026-09-25: Per-role dtypes, one field per `Role`.
    router_dtype: Dtype,
    lm_head_dtype: Dtype,
    embedding_dtype: Dtype,
    attention_dtype: Dtype,
    expert_dtype: Dtype,
    shared_expert_dtype: Dtype,
    norm_dtype: Dtype,
    /// 2026-09-25: Layer indices marked sensitive. `Attention`, `Expert` and
    /// `SharedExpert` tensors in these layers get `sensitive_dtype` when it is
    /// not `Inherit`.
    sensitive_layers: BTreeSet<u16>,
    sensitive_dtype: Dtype,
}

impl Default for PrecisionSchedule {
    /// 2026-09-25: Empty schedule: every lookup returns `Inherit`.
    fn default() -> Self {
        Self {
            default: Dtype::Inherit,
            router_dtype: Dtype::Inherit,
            lm_head_dtype: Dtype::Inherit,
            embedding_dtype: Dtype::Inherit,
            attention_dtype: Dtype::Inherit,
            expert_dtype: Dtype::Inherit,
            shared_expert_dtype: Dtype::Inherit,
            norm_dtype: Dtype::Inherit,
            sensitive_layers: BTreeSet::new(),
            sensitive_dtype: Dtype::Inherit,
        }
    }
}

impl PrecisionSchedule {
    /// 2026-09-25: Build a schedule from:
    ///   - `router_dtype`: dtype for the MoE gate
    ///   - `lm_head_dtype`: dtype for the final unembedding
    ///   - `sensitive_block_dtype` + `sensitive_block_indices`: the dtype
    ///     for the weight tensors of the listed layers
    ///   - `default_dtype`: the global default
    ///
    /// The other roles' dtypes are `Inherit`.
    pub fn build(
        router_dtype: Dtype,
        lm_head_dtype: Dtype,
        sensitive_block_indices: &[u16],
        sensitive_block_dtype: Dtype,
        default_dtype: Dtype,
    ) -> Self {
        Self {
            default: default_dtype,
            router_dtype,
            lm_head_dtype,
            embedding_dtype: Dtype::Inherit,
            attention_dtype: Dtype::Inherit,
            expert_dtype: Dtype::Inherit,
            shared_expert_dtype: Dtype::Inherit,
            norm_dtype: Dtype::Inherit,
            sensitive_layers: sensitive_block_indices.iter().copied().collect(),
            sensitive_dtype: sensitive_block_dtype,
        }
    }

    /// 2026-09-25: Resolve the target dtype for a tensor. `layer_idx = None`
    /// is for non-layer tensors (embedding, lm_head, final norm).
    /// Lookup order, skipping any step that yields `Inherit`:
    ///   1. Sensitive-layer dtype (Attention, Expert and SharedExpert only)
    ///   2. Per-role dtype
    ///   3. Global default
    pub fn dtype_for(&self, layer_idx: Option<u16>, role: Role) -> Dtype {
        if let Some(li) = layer_idx
            && matches!(role, Role::Attention | Role::Expert | Role::SharedExpert)
            && self.sensitive_layers.contains(&li)
            && self.sensitive_dtype.is_override()
        {
            return self.sensitive_dtype;
        }
        let role_dtype = match role {
            Role::Router => self.router_dtype,
            Role::LmHead => self.lm_head_dtype,
            Role::Embedding => self.embedding_dtype,
            Role::Attention => self.attention_dtype,
            Role::Expert => self.expert_dtype,
            Role::SharedExpert => self.shared_expert_dtype,
            Role::Norm => self.norm_dtype,
        };
        if role_dtype.is_override() {
            role_dtype
        } else {
            self.default
        }
    }

    /// 2026-09-25: True iff some lookup can return a dtype other than
    /// `Inherit`. The qwen35 loader logs the schedule when it is.
    pub fn has_any_override(&self) -> bool {
        self.default.is_override()
            || self.router_dtype.is_override()
            || self.lm_head_dtype.is_override()
            || self.embedding_dtype.is_override()
            || self.attention_dtype.is_override()
            || self.expert_dtype.is_override()
            || self.shared_expert_dtype.is_override()
            || self.norm_dtype.is_override()
            || (self.sensitive_dtype.is_override() && !self.sensitive_layers.is_empty())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_schedule_is_all_inherit() {
        let s = PrecisionSchedule::default();
        assert!(!s.has_any_override());
        assert_eq!(s.dtype_for(None, Role::LmHead), Dtype::Inherit);
        assert_eq!(s.dtype_for(Some(0), Role::Router), Dtype::Inherit);
        assert_eq!(s.dtype_for(Some(38), Role::Expert), Dtype::Inherit);
    }

    #[test]
    fn role_override_wins_over_default() {
        let s = PrecisionSchedule::build(
            Dtype::Bf16,
            Dtype::Bf16,
            &[], // 2026-09-25: no sensitive layers
            Dtype::Inherit,
            Dtype::Nvfp4,
        );
        assert!(s.has_any_override());
        assert_eq!(s.dtype_for(None, Role::Router), Dtype::Bf16);
        assert_eq!(s.dtype_for(None, Role::LmHead), Dtype::Bf16);
        assert_eq!(s.dtype_for(Some(15), Role::Expert), Dtype::Nvfp4);
    }

    #[test]
    fn sensitive_layer_overrides_bulk_default_for_weights() {
        let s = PrecisionSchedule::build(
            Dtype::Bf16,
            Dtype::Bf16,
            &[0, 1, 38, 39],
            Dtype::Fp8,
            Dtype::Nvfp4,
        );
        assert_eq!(s.dtype_for(Some(0), Role::Expert), Dtype::Fp8);
        assert_eq!(s.dtype_for(Some(38), Role::Attention), Dtype::Fp8);
        assert_eq!(s.dtype_for(Some(1), Role::SharedExpert), Dtype::Fp8);
        assert_eq!(s.dtype_for(Some(5), Role::Expert), Dtype::Nvfp4);

        let sensitive_only = PrecisionSchedule::build(
            Dtype::Inherit,
            Dtype::Inherit,
            &[7],
            Dtype::Fp8,
            Dtype::Inherit,
        );
        assert!(sensitive_only.has_any_override());
        assert_eq!(
            sensitive_only.dtype_for(Some(7), Role::Attention),
            Dtype::Fp8
        );
    }

    #[test]
    fn sensitive_layer_does_not_override_router_or_lm_head() {
        // 2026-09-25: Only Attention, Expert and SharedExpert tensors take the
        // sensitive dtype; the other roles resolve to their role dtype or the
        // default.
        let s = PrecisionSchedule::build(Dtype::Bf16, Dtype::Bf16, &[0], Dtype::Fp8, Dtype::Nvfp4);
        assert_eq!(s.dtype_for(Some(0), Role::Router), Dtype::Bf16);
        assert_eq!(s.dtype_for(Some(0), Role::LmHead), Dtype::Bf16);
        assert_eq!(s.dtype_for(Some(0), Role::Embedding), Dtype::Nvfp4);
        assert_eq!(s.dtype_for(Some(0), Role::Norm), Dtype::Nvfp4);
    }

    #[test]
    fn role_str_round_trips() {
        for r in [
            Role::Router,
            Role::LmHead,
            Role::Embedding,
            Role::Attention,
            Role::Expert,
            Role::SharedExpert,
            Role::Norm,
        ] {
            assert_eq!(Role::from_str(r.name()), Some(r));
        }
        assert_eq!(Role::from_str("nonsense"), None);
    }

    #[test]
    fn dtype_str_parsing() {
        assert_eq!(Dtype::from_str("bf16"), Some(Dtype::Bf16));
        assert_eq!(Dtype::from_str("fp8"), Some(Dtype::Fp8));
        assert_eq!(Dtype::from_str("nvfp4"), Some(Dtype::Nvfp4));
        assert_eq!(Dtype::from_str("inherit"), Some(Dtype::Inherit));
        assert_eq!(Dtype::from_str("bogus"), None);
        assert!(!Dtype::Inherit.is_override());
        assert!(Dtype::Bf16.is_override());
    }
}
