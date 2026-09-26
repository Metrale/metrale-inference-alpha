// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Env-driven store selection: [`build_tier_store`] for the SSM spill tier,
//! [`build_decode_tier_store`] for the decode cold tier.
//!
//! Owner: model-engine (SSM snapshot tier).
//! Invariants: none beyond the types.

use anyhow::{Result, bail};

use super::fingerprint::{ModelFingerprint, resolve_decode_ns, resolve_swap_ns};
use super::unified::{
    DISK_GB_VAR, TransportSlotArena, build_unified_swap, disk_gb_requested, log_unified_tier,
    ssm_tier_disk_slots, unified_hot_slots,
};
use super::{
    ArenaSnapshotStore, FileSnapshotArena, MemBlobStore, PagingSnapshotStore, RdmaSnapshotStore,
    SnapshotBlobStore, UnifiedSnapshotStore, ssm_tier_unified,
};

/// 2026-09-25: Whether `METRALE_SSM_TIER` is set, to any value. When it is not,
/// `build_ssm_tier_store` returns `None` and snapshot eviction drops the victim.
pub(crate) fn ssm_tier_enabled() -> bool {
    std::env::var_os("METRALE_SSM_TIER").is_some()
}

/// 2026-09-25: Build the SSM spill-tier store. `build_ssm_tier_store` calls it
/// only when [`ssm_tier_enabled`].
///
/// - `METRALE_SSM_RDMA_TIER=host:port` connects to a peer arena of
///   `METRALE_SSM_RDMA_ARENA_SLOTS` slots (512 when unset or unparsable):
///   - with `METRALE_SSM_SWAP=1`, first a [`PagingSnapshotStore`];
///   - then, with `METRALE_SSM_TIER_UNIFIED`, a [`UnifiedSnapshotStore`] over the
///     arena, otherwise a [`RdmaSnapshotStore`].
/// - Without a peer, or when every connect fails (always, in a build without
///   RDMA verbs), a unified host-RAM store with `METRALE_SSM_TIER_UNIFIED`,
///   otherwise `MemBlobStore::new(0)`.
///
/// Connect failures and a failed unified construction log and fall back.
/// `Err` is returned for a bad `METRALE_SSM_SWAP_NS` (parsed before the paging
/// connect) or a bad `METRALE_SSM_TIER_DISK_GB` (parsed only on the unified arms).
pub(crate) fn build_tier_store(
    fp: ModelFingerprint,
    blob_bytes: usize,
) -> Result<std::sync::Arc<dyn SnapshotBlobStore>> {
    use std::sync::Arc;
    // 2026-09-25: Only the unified arms read the disk budget, so say so when it
    // is set without them.
    if disk_gb_requested() && !ssm_tier_unified() {
        tracing::warn!(
            "{DISK_GB_VAR} is set but METRALE_SSM_TIER_UNIFIED is not — the disk budget is \
             INERT (the legacy spill stores have no disk tier to bound)"
        );
    }
    if let Some(peer) = std::env::var("METRALE_SSM_RDMA_TIER")
        .ok()
        .filter(|s| !s.is_empty())
    {
        let slots: usize = std::env::var("METRALE_SSM_RDMA_ARENA_SLOTS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(512);
        let arena_bytes = slots as u64 * blob_bytes as u64;
        // 2026-09-25: `METRALE_SSM_SWAP=1` selects paging mode: the peer, which
        // needs `--swap-dir`, owns residency. A connect failure falls through to
        // the bounded RDMA store.
        if std::env::var("METRALE_SSM_SWAP").ok().as_deref() == Some("1") {
            // 2026-09-25: The namespace is `METRALE_SSM_SWAP_NS` (strict parse)
            // or the model fingerprint. It is resolved before the connect, so a
            // bad override is an `Err`, not a fallback.
            let namespace = resolve_swap_ns(fp)?;
            match metrale_storage::RdmaSnapshotArena::connect_paging(&peer, arena_bytes, blob_bytes)
            {
                Ok(arena) => {
                    tracing::info!(
                        "SSM spill tier = RDMA PAGING peer {peer} ({slots}-slot shared RAM cache × \
                         {blob_bytes} B + NVMe swap = infinite depth; ns={namespace:#x}, model \
                         fingerprint {:#018x})",
                        fp.get(),
                    );
                    return Ok(Arc::new(PagingSnapshotStore::new(
                        Box::new(arena),
                        blob_bytes,
                        namespace,
                    )));
                }
                Err(e) => tracing::warn!(
                    "SSM RDMA paging connect to {peer} failed ({e:#}); trying bounded RDMA"
                ),
            }
        }
        // 2026-09-25: `connect` errors both on a real connect failure and in a
        // build without RDMA verbs, whose stub arena always errors.
        match metrale_storage::RdmaSnapshotArena::connect(&peer, arena_bytes, blob_bytes) {
            Ok(arena) => {
                if ssm_tier_unified() {
                    // 2026-09-25: A `Residency` over the same remote arena: a full
                    // arena spills its least recently used blob to the swap tier
                    // instead of refusing the put.
                    let hot = Box::new(TransportSlotArena {
                        transport: Box::new(arena),
                        slot_bytes: blob_bytes,
                        num_slots: slots,
                    });
                    let (swap, backing) = build_unified_swap(blob_bytes, "marconi-rdma");
                    // 2026-09-25: A bad disk budget is an `Err`, not a fallback
                    // to an unbounded store.
                    let max_disk_slots = ssm_tier_disk_slots(blob_bytes)?;
                    match UnifiedSnapshotStore::new_capped(hot, swap, blob_bytes, max_disk_slots) {
                        Ok(s) => {
                            log_unified_tier(
                                &format!("over RDMA peer {peer}"),
                                slots,
                                blob_bytes,
                                max_disk_slots,
                                backing,
                            );
                            return Ok(Arc::new(s));
                        }
                        Err(e) => {
                            tracing::warn!(
                                "SSM unified residency init failed ({e:#}); \
                                 falling back to host-RAM"
                            );
                            return Ok(Arc::new(MemBlobStore::new(0)));
                        }
                    }
                }
                tracing::info!(
                    "SSM spill tier = RDMA peer {peer} ({slots} slots × {blob_bytes} B = \
                     {:.2} GiB arena)",
                    arena_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
                );
                return Ok(Arc::new(RdmaSnapshotStore::new(
                    Box::new(arena),
                    blob_bytes,
                    slots,
                )));
            }
            Err(e) => tracing::warn!(
                "SSM RDMA tier connect to {peer} failed ({e:#}); falling back to host-RAM"
            ),
        }
    }
    if ssm_tier_unified() {
        // 2026-09-25: An LRU hot arena of `unified_hot_slots()` slots that
        // spills into the swap tier. `VecSlotArena::new` allocates all
        // `slots × blob_bytes` up front.
        let hot_slots = unified_hot_slots();
        let hot = Box::new(metrale_storage::tier::VecSlotArena::new(
            blob_bytes, hot_slots,
        ));
        let (swap, backing) = build_unified_swap(blob_bytes, "marconi-host");
        let max_disk_slots = ssm_tier_disk_slots(blob_bytes)?;
        match UnifiedSnapshotStore::new_capped(hot, swap, blob_bytes, max_disk_slots) {
            Ok(s) => {
                log_unified_tier(
                    "in host RAM",
                    hot_slots,
                    blob_bytes,
                    max_disk_slots,
                    backing,
                );
                return Ok(Arc::new(s));
            }
            Err(e) => tracing::warn!(
                "SSM unified residency init failed ({e:#}); falling back to host-RAM store"
            ),
        }
    }
    Ok(Arc::new(MemBlobStore::new(0)))
}

/// 2026-09-25: Build the decode cold-tier store, selected by `METRALE_SSM_DECODE_TIER`:
///   - `nvme` (needs `METRALE_SSM_DECODE_NVME_DIR`): with `METRALE_SSM_TIER_UNIFIED`
///     and a 4 KiB-multiple `blob_bytes`, an uncapped [`UnifiedSnapshotStore`] over an
///     O_DIRECT swap file; otherwise an [`ArenaSnapshotStore`] over a
///     [`FileSnapshotArena`] of `min_slots + 1` slots.
///   - `peer` (needs `METRALE_SSM_DECODE_RDMA_TIER=host:port`): a
///     [`PagingSnapshotStore`] in the `resolve_decode_ns` namespace. A connect
///     failure is an `Err`.
///   - unset: an unbounded `MemBlobStore::new(0)`.
///   - any other value: an `Err`.
///
/// `METRALE_SSM_TIER_DISK_GB` is never read here: the unified arm calls
/// [`UnifiedSnapshotStore::new`], which is uncapped.
pub(crate) fn build_decode_tier_store(
    fp: ModelFingerprint,
    blob_bytes: usize,
    min_slots: usize,
) -> Result<std::sync::Arc<dyn SnapshotBlobStore>> {
    use std::sync::Arc;
    match std::env::var("METRALE_SSM_DECODE_TIER").ok().as_deref() {
        Some("nvme") => {
            let dir = std::env::var("METRALE_SSM_DECODE_NVME_DIR")
                .ok()
                .filter(|s| !s.is_empty())
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "METRALE_SSM_DECODE_TIER=nvme requires METRALE_SSM_DECODE_NVME_DIR=<dir>"
                    )
                })?;
            if ssm_tier_unified() && blob_bytes > 0 && blob_bytes.is_multiple_of(4096) {
                // 2026-09-25: `new` is uncapped (`max_disk_slots = 0`), so
                // `Residency::make_disk_room` never drops a record from this store.
                std::fs::create_dir_all(&dir)?;
                let path = std::path::Path::new(&dir)
                    .join(format!("metrale-decode-ring.{}.swap", std::process::id()));
                let swap = metrale_storage::tier::DirectSwapFile::create(&path, blob_bytes)?;
                let hot_slots = unified_hot_slots().min(min_slots + 1);
                let hot = Box::new(metrale_storage::tier::VecSlotArena::new(
                    blob_bytes, hot_slots,
                ));
                let store = UnifiedSnapshotStore::new(hot, Box::new(swap), blob_bytes)?;
                tracing::info!(
                    "SSM decode cold tier = UNIFIED residency ({hot_slots} hot RAM slots + \
                     O_DIRECT swap in {dir}; non-dropping by construction ≥ min_slots \
                     {min_slots})"
                );
                return Ok(Arc::new(store));
            }
            if ssm_tier_unified() {
                tracing::info!(
                    "SSM decode cold tier: METRALE_SSM_TIER_UNIFIED set but blob_bytes \
                     {blob_bytes} is not a 4 KiB multiple (O_DIRECT stride); keeping the \
                     sized arena store"
                );
            }
            // 2026-09-25: `min_slots + 1` slots: `ArenaSnapshotStore::put` refuses a
            // new key once every slot is mapped.
            let slots = min_slots + 1;
            let capacity = slots as u64 * blob_bytes as u64;
            let arena = FileSnapshotArena::create(&dir, capacity)?;
            tracing::info!(
                "SSM decode cold tier = LOCAL NVMe {dir} ({slots} slots × {blob_bytes} B = \
                 {:.2} GiB, non-dropping ≥ min_slots {min_slots})",
                capacity as f64 / (1024.0 * 1024.0 * 1024.0),
            );
            Ok(Arc::new(ArenaSnapshotStore::new(
                Box::new(arena),
                blob_bytes,
                slots,
            )))
        }
        Some("peer") => {
            let peer = std::env::var("METRALE_SSM_DECODE_RDMA_TIER")
                .ok()
                .filter(|s| !s.is_empty())
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "METRALE_SSM_DECODE_TIER=peer requires METRALE_SSM_DECODE_RDMA_TIER=host:port"
                    )
                })?;
            // 2026-09-25: The namespace is `METRALE_SSM_DECODE_NS` (strict parse,
            // used unsalted) or `derive_decode_ns_salted`, resolved before the
            // connect.
            let namespace = resolve_decode_ns(fp)?;
            // 2026-09-25: `PagingSnapshotStore::put` never returns `Ok(false)`;
            // this slot count only sets the `arena_bytes` requested from the peer.
            let slots = (min_slots + 1).max(512);
            let arena_bytes = slots as u64 * blob_bytes as u64;
            let arena =
                metrale_storage::RdmaSnapshotArena::connect_paging(&peer, arena_bytes, blob_bytes)?;
            tracing::info!(
                "SSM decode cold tier = RDMA PAGING peer {peer} (non-dropping, ns={namespace:#x}, \
                 model fingerprint {:#018x})",
                fp.get(),
            );
            Ok(Arc::new(PagingSnapshotStore::new(
                Box::new(arena),
                blob_bytes,
                namespace,
            )))
        }
        // 2026-09-25: Unset: `MemBlobStore::new(0)`, which never refuses a put.
        None => {
            tracing::info!("SSM decode cold tier = host-RAM (unbounded, non-dropping)");
            Ok(Arc::new(MemBlobStore::new(0)))
        }
        // 2026-09-25: Any other value is an `Err`: falling back to host RAM would
        // hide a typo.
        Some(other) => bail!(
            "METRALE_SSM_DECODE_TIER={other:?} is not recognized (accepted: \"nvme\", \"peer\", or \
             unset for unbounded host-RAM). Refusing to silently fall back to host-RAM."
        ),
    }
}

#[cfg(test)]
#[path = "selectors_tests.rs"]
mod tests;
