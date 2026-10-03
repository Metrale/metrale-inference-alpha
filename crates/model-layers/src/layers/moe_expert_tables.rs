// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-02: Whether a MoE's transposed NVFP4 prefill tables (the routed and shared experts'
//! K-major copies `MoeLayer::transpose_for_prefill` builds) are built. The serve decides once,
//! before the model loads, from its memory plan (`met circuit memory`'s model at the serve's
//! declared `--max-seq-len` and batch): the tables are built only when the plan fits the util
//! budget with them. Without them MoE prefill runs the grouped GEMM on the row-major experts.
//!
//! Owner: model-layers (MoE).
//! Invariants:
//! - Published once (`OnceLock`); a second publication does not change it.
//! - There is no default: a loader that would build the tables and finds nothing published
//!   refuses to load ([`moe_expert_tables`] is `None`).

use std::sync::OnceLock;

/// 2026-10-02: The decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoeExpertTables {
    /// 2026-10-02: The plan fits with the tables.
    Build,
    /// 2026-10-02: The plan does not fit with them.
    Skip,
}

impl MoeExpertTables {
    /// 2026-10-02: `build` or `skip`, the spelling the memory plan's copy rules and a gate
    /// record's `serve_resolved` use.
    pub fn name(self) -> &'static str {
        match self {
            Self::Build => "build",
            Self::Skip => "skip",
        }
    }
}

static VALUE: OnceLock<MoeExpertTables> = OnceLock::new();

/// 2026-10-02: Publish the serve's decision. Returns the decision in force; a caller that gets a
/// different one should warn.
pub fn set_moe_expert_tables_from_plan(t: MoeExpertTables) -> MoeExpertTables {
    let _ = VALUE.set(t);
    *VALUE.get().expect("just set")
}

/// 2026-10-02: The published decision, if any.
pub fn moe_expert_tables() -> Option<MoeExpertTables> {
    VALUE.get().copied()
}
