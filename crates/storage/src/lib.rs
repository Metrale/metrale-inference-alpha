// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: metrale-storage: the KV, SSM-snapshot, expert and weight storage tiers
//! (NVMe files, host RAM, RDMA peers), the peers that serve them, and their CUDA helpers.
//!
//! Owner: storage.
//! Invariants:
//! - The CUDA driver bindings (`cuda_min`, `cuda_module`, `cuda_graph`) exist only with
//!   the `cuda` feature, so every module that calls the driver is gated on it too.
//! - Without the `cuda` feature, `stubs` supplies `HighSpeedSwap`, `with_local`,
//!   `local_installed` and `install_local` under the same names.

#![deny(warnings)]
#![deny(clippy::all)]

#[cfg(feature = "cuda")]
pub mod cuda_graph;
#[cfg(feature = "cuda")]
pub mod cuda_min;
#[cfg(feature = "cuda")]
pub mod cuda_module;

#[cfg(feature = "cuda")]
pub use cuda_module::{CudaEvent, CudaModule, launch_kernel};

pub mod attention_ref;
pub mod cascade_policy;
pub mod config;
pub mod eviction;
pub mod expert;
pub mod expert_pack;
pub mod expert_peer;
pub mod group;
pub mod kv_paging;
pub mod model_dims;
pub mod predictor_ref;
pub mod projection;
// 2026-09-25: The one-sided RDMA KV backend: a `StorageBackend` that offloads and
// restores KV groups to a `cache_peer` over verbs.
#[cfg(all(feature = "cuda", metrale_rdma_verbs))]
pub mod rdma_kv_backend;
pub mod rdma_snapshot;
pub mod snapshot_swap;
pub mod weight_peer;

// 2026-09-25: `cache_peer` serves remote-RAM arenas over one-sided RDMA; `blade_cap`
// is the commit ledger it and `expert_peer` reserve against. The handshake that
// reserves is compiled only with `metrale_rdma_verbs`, hence the `dead_code` allow.
#[cfg(unix)]
#[allow(dead_code)]
pub(crate) mod blade_cap;
pub mod cache_peer;

// 2026-09-25: Re-exported without the `cuda` feature: metrale-model-engine's
// `high_speed_swap_dims` returns it on every build.
pub use model_dims::ModelDims;

#[cfg(feature = "cuda")]
pub mod layout;

#[cfg(feature = "cuda")]
pub mod backend;
#[cfg(all(feature = "cuda", target_os = "linux"))]
pub mod bench;
#[cfg(feature = "cuda")]
pub mod cascade_backend;
#[cfg(feature = "cuda")]
pub mod expert_arena;
#[cfg(feature = "cuda")]
pub mod expert_tier;
#[cfg(feature = "cuda")]
pub mod ngram_cache;
#[cfg(feature = "cuda")]
mod ngram_cache_fault;
// 2026-09-25: RDMA expert staging, unix only.
#[cfg(all(feature = "cuda", unix))]
pub mod expert_tier_rdma;
#[cfg(feature = "cuda")]
pub mod high_speed_swap;
#[cfg(feature = "cuda")]
pub mod predictor;
#[cfg(all(feature = "cuda", target_os = "linux"))]
pub mod probe;
#[cfg(feature = "cuda")]
pub mod scratch_pool;
#[cfg(feature = "cuda")]
pub mod tiled_attention;
#[cfg(feature = "cuda")]
pub use backend::{PosixBackend, ReadRequest, StorageBackend};
// 2026-09-25: io_uring is Linux-only.
#[cfg(all(feature = "cuda", target_os = "linux"))]
pub use backend::IoUringBackend;
pub use config::HighSpeedSwapConfig;
pub use eviction::EvictionPolicy;
pub use expert::{
    ExpertKey, ExpertLayout, ExpertRecordHeader, ExpertRecordId, ExpertRecordSpec, Proj, ProjBytes,
};
#[cfg(feature = "cuda")]
pub use expert_arena::ExpertArena;
pub use expert_pack::{ExpertFileReader, ExpertFileWriter};
pub use expert_pack::{ExpertIndex, ProjData, ProjView, pack_record, unpack_record};
#[cfg(feature = "cuda")]
pub use expert_tier::{
    ArenaSlot, ExpertResidency, ExpertTier, PosixTier, TierKind, UmaArenaTier, open_tier,
};
#[cfg(all(feature = "cuda", unix))]
pub use expert_tier_rdma::RdmaTier;
#[cfg(feature = "cuda")]
pub use high_speed_swap::{HighSpeedSwap, install_local, local_installed, with_local};
#[cfg(all(feature = "cuda", metrale_rdma_verbs))]
pub use kv_paging::KvPagingBackend;
#[cfg(feature = "cuda")]
pub use ngram_cache::NgramRowCache;
#[cfg(all(feature = "cuda", metrale_rdma_verbs))]
pub use rdma_kv_backend::RdmaKvBackend;
pub use rdma_snapshot::RdmaSnapshotArena;

/// 2026-09-25: `true` when build.rs emitted `metrale_rdma_verbs` for this crate. The
/// verbs shim is built by metrale-gpu-sys, and `rustc-cfg` does not cross crates, so
/// build.rs re-emits the cfg from that crate's `links` metadata.
/// `rdma_verbs_probe_tests` checks it matches `metrale_gpu_sys::verbs_enabled()`.
pub const fn rdma_verbs_enabled() -> bool {
    cfg!(metrale_rdma_verbs)
}

#[cfg(test)]
mod rdma_verbs_probe_tests;

// 2026-09-25: Gated on the `cuda` feature alone: `HighSpeedSwap` picks
// `IoUringBackend` on Linux and `PosixBackend` elsewhere, so any CUDA build gets the
// real one.
#[cfg(not(feature = "cuda"))]
mod stubs;
#[cfg(not(feature = "cuda"))]
pub use stubs::{HighSpeedSwap, install_local, local_installed, with_local};

#[cfg(feature = "cuda")]
pub use predictor::{Predictor, PredictorDims};
#[cfg(all(feature = "cuda", target_os = "linux"))]
pub use probe::{Backend, ProbeConfig, ProbeResult, run_probe};
pub use projection::{PredictorShape, build_projection};
#[cfg(feature = "cuda")]
pub use tiled_attention::{TiledAttention, TiledAttentionDims};
#[cfg(feature = "cuda")]
pub use weight_peer::{WeightManifest, WeightTensorRecord};
#[cfg(feature = "cuda")]
pub mod tier;
