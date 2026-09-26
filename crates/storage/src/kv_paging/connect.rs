// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Picks the KV peer backend from `METRALE_KV_PAGING`: the raw one-sided
//! `RdmaKvBackend` or the peer-paging `KvPagingBackend`.
//!
//! Owner: storage, KV paging tier.
//! Invariants:
//! - `METRALE_KV_PAGING` unset or `0` connects `RdmaKvBackend` and reads no other
//!   paging variable; any value other than `0`/`1` is an error.
//! - The paging backend is connected only with block coalescing on, an explicit
//!   `METRALE_KV_PAGING_ARENA_GB`, and a namespace from either `METRALE_KV_PAGING_NS`
//!   or the model fingerprint; a missing one is an error, never a default.

use std::num::NonZeroU64;

use anyhow::{Result, anyhow, bail};

use super::backend::{KvPagingBackend, KvPagingConnect};
use super::ns;
use crate::backend::StorageBackend;
use crate::group::GroupLayout;
use crate::model_dims::ModelDims;

/// 2026-09-25: Connect the KV backend for `peer`. `METRALE_KV_PAGING` unset or `0`
/// returns `RdmaKvBackend::connect(peer, layout)`, whose handshake is the v2 header
/// with `blob_bytes == 0`. `1` resolves the arena size, namespace and salt from the
/// environment and connects `KvPagingBackend`. The namespace override is logged, as
/// are the derived namespace, fingerprint and salt.
pub fn connect_kv_peer_backend(
    peer: &str,
    layout: GroupLayout,
    model: &ModelDims,
    elem_bytes: u32,
    coalesce_blocks: bool,
) -> Result<Box<dyn StorageBackend>> {
    let paging = ns::kv_paging_selected(std::env::var("METRALE_KV_PAGING").ok().as_deref())?;
    if !paging {
        return Ok(Box::new(crate::rdma_kv_backend::RdmaKvBackend::connect(
            peer, layout,
        )?));
    }
    if !coalesce_blocks {
        bail!(
            "METRALE_KV_PAGING=1 requires block coalescing (METRALE_HSS_COALESCE_BLOCKS, the \
             default): the paging record is one whole KV block, and the per-head offload \
             write path cannot be served by a peer-owned block arena"
        );
    }
    let arena_bytes = ns::resolve_arena_bytes_from(
        std::env::var("METRALE_KV_PAGING_ARENA_GB").ok().as_deref(),
        layout.block_bytes(),
    )?;
    let ns = match std::env::var("METRALE_KV_PAGING_NS").ok() {
        Some(raw) => {
            let ns = ns::resolve_kv_ns_from(Some(&raw), NonZeroU64::new(1).expect("nonzero"))?;
            tracing::info!(
                "kv-paging: namespace OVERRIDDEN via METRALE_KV_PAGING_NS={:#018x} — two clients \
                 sharing one explicit ns on one peer WILL cross-serve KV blocks",
                ns.get()
            );
            ns
        }
        None => {
            // 2026-09-25: The derived namespace folds the model fingerprint; a model
            // without one is refused rather than keyed without it.
            let fp = model.model_fp.ok_or_else(|| {
                anyhow!(
                    "METRALE_KV_PAGING=1 requires a model fingerprint (ModelDims::model_fp) and \
                     the loader did not derive one — fix the model config, or set \
                     METRALE_KV_PAGING_NS to an explicit non-zero u64"
                )
            })?;
            let salt =
                ns::resolve_salt_from(std::env::var("METRALE_KV_PAGING_SALT").ok().as_deref())?
                    .unwrap_or_else(rand::random::<u64>);
            let derived = ns::derive_kv_ns(
                fp.get(),
                &layout,
                elem_bytes,
                model.block_size as u32,
                model.head_dim as u32,
                salt,
            );
            tracing::info!(
                "kv-paging: derived namespace {:#018x} (fp {:#018x}, client salt {salt:#018x} — \
                 pin via METRALE_KV_PAGING_SALT to reproduce; the salt makes keys CLIENT-PRIVATE: \
                 capacity pooling yes, cross-client warm hits no)",
                derived.get(),
                fp.get(),
            );
            derived
        }
    };
    tracing::info!(
        "kv-paging: connecting {peer} (arena {:.3} GiB, blob {} B); peer must run with \
         --swap-cap-gb-kv 0 (a KV disk cap turns evictions into unrecoverable KV loss)",
        arena_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        layout.block_bytes(),
    );
    Ok(Box::new(KvPagingBackend::connect(
        peer,
        layout,
        KvPagingConnect { arena_bytes, ns },
    )?))
}
