// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The weight peer: a daemon that holds models' safetensors shards mapped in
//! its RAM and lets a client read any tensor with one-sided RDMA. The client is
//! `RdmaWeightLoader` in model-weights (`weight_tier_rdma.rs`).
//!
//! Protocol on one TCP connection, lengths little-endian (`wire`, `serve`):
//! 1. The client sends the model request, `[u32 len][len bytes of id or path]`.
//! 2. The peer stages that model (once per directory) and sends the manifest,
//!    `[u32 len][len bytes of JSON]` (`WeightManifest`).
//! 3. The client sends `[u8 transport mode]`; only `MODE_VERBS` is served.
//! 4. The client sends `[u8 n_rails]`. The peer registers every shard mapping for remote
//!    read on each rail and sends one `VerbsServerParams` per rail, whose `layers` holds
//!    each shard's `(base, rkey)`. The client sends `n_rails` again and its QP parameters
//!    per rail; the peer connects, acknowledges, and waits for the client to close the
//!    connection while it reads the tensors.
//!
//! Owner: storage (weight peer).
//! Invariants: `manifest` and `wire` compile on every platform; `serve` and `shard` only
//! on unix.

mod manifest;
#[cfg(unix)]
mod serve;
#[cfg(unix)]
mod shard;
mod wire;

pub use manifest::{WeightManifest, WeightTensorRecord, rail_for_tensor, tensor_remote_addr};
#[cfg(unix)]
pub use serve::{WeightPeerConfig, serve};
pub use wire::{
    MODEL_REQUEST_MAX, read_model_request, read_weight_manifest, write_model_request,
    write_weight_manifest,
};
