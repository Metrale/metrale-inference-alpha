// SPDX-License-Identifier: MIT OR Apache-2.0

#![deny(warnings)]
#![deny(clippy::all)]

//! 2026-09-26: Low-level bindings: the RDMA verbs C shim with its rail
//! handshake and wire codecs, NCCL (feature `nccl`), NVML and cuFile loaded
//! with `dlopen`, and NVTX ranges (feature `nvtx`).
//!
//! `verbs` and `railset` are compiled only under `cfg(metrale_rdma_verbs)`,
//! which build.rs emits when it compiles the shim. `env`, `handshake` and
//! `wire` are always compiled, so their tests run without rdma-core.
//!
//! Owner: metrale-gpu-sys.
//! Invariants: none beyond the types.

pub mod env;
pub mod handshake;
pub mod wire;

// 2026-09-26: Also compiled under `test`, so its tests run on hosts without
// the shim.
#[cfg(any(metrale_rdma_verbs, test))]
mod rail_count;

#[cfg(metrale_rdma_verbs)]
pub mod verbs;

#[cfg(metrale_rdma_verbs)]
pub mod railset;

#[cfg(metrale_rdma_verbs)]
pub use railset::{Rail, RailSet, RailSpec};
#[cfg(metrale_rdma_verbs)]
pub use verbs::{Gid, MrKeys, Verbs};
pub use wire::{
    CacheServerParams, MODE_TCP, MODE_VERBS, RemoteQp, STATUS_ERR, STATUS_OK, VerbsClientParams,
    VerbsServerParams,
};

/// 2026-09-26: Whether build.rs compiled the verbs shim, i.e. whether
/// `cfg(metrale_rdma_verbs)` is set. Always compiled, so tests can assert it
/// (tests/verbs_probe.rs, metrale-storage's rdma_verbs_probe_tests.rs).
pub const fn verbs_enabled() -> bool {
    cfg!(metrale_rdma_verbs)
}

pub mod cufile;
#[cfg(feature = "nccl")]
pub mod nccl;
pub mod nvml;
#[cfg(feature = "nvtx")]
pub mod nvtx;
