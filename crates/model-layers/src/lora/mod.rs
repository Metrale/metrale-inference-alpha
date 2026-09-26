// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: PEFT LoRA adapters: audit, pack into the rank-padded slot pool
//! and the expert pool, slot routing and swap, and token overlays. "LoRA"
//! here means adapters; `kv_lora_rank` / `q_lora_rank` in the model config are
//! MLA dimensions, unrelated.
//!
//! Owner: model-layers (lora).
//! Invariants: none beyond the types.

mod audit;
mod env;
mod expert_apply;
mod expert_pack;
mod key;
mod loading;
mod moe_row_adapter;
mod overlay;
mod overlay_build;
mod overlay_tables;
mod slot_math;
mod target;
mod types;

pub(crate) use audit::{AuditedAdapter, audit_adapter};
pub use env::*;
pub use expert_apply::*;
pub use key::*;
pub use loading::*;
pub use moe_row_adapter::*;
pub use overlay::*;
pub use overlay_build::*;
pub use overlay_tables::*;
pub use slot_math::*;
pub use target::*;
pub use types::*;

// 2026-09-25: Only `fetch_adapter_manifest` needs `cuda`; the rest is host
// code, compiled under test too so its tests run without CUDA.
#[cfg(any(feature = "cuda", test))]
// 2026-09-25: Unix-only: the RDMA swap it serves (model-engine
// `swap_lora_slot_from_peer`) is unix-only.
#[cfg(unix)]
pub mod rdma_stage;

#[cfg(test)]
#[path = "test_support.rs"]
mod test_support;
