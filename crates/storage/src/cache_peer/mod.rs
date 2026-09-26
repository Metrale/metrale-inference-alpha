// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The cache peer: a process that lends RAM to KV and snapshot
//! clients over one-sided RDMA. A connection gets either its own arena (RAW
//! mode), which the client allocates in, or the shared paging arena of its
//! (kind, blob_bytes), backed by a swap file (`registry`).
//!
//! Owner: metrale-storage peers.
//! Invariants:
//! - A connection starts with the v2 paging header; `blob_bytes == 0`
//!   selects RAW mode, any other value paging.
//! - A paging client is refused unless `RdmaConfig::swap_dir` is set.

// 2026-09-25: The handshake codec is metrale-gpu-sys `wire`;
// `kv_server_params_round_trip` in crates/gpu-sys/tests/wire_roundtrip.rs
// covers `CacheServerParams`.
pub use metrale_gpu_sys::wire::CacheServerParams;

// 2026-09-25: Only the verbs build of the connection handler uses the registry.
#[cfg(metrale_rdma_verbs)]
mod registry;
#[cfg(unix)]
mod server_impl;

#[cfg(unix)]
pub use server_impl::{RdmaConfig, serve};
