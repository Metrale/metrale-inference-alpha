// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The peer-owned paging store and the client-side fixed-slot arena store
//! (over any [`SnapshotTransport`]: the RDMA arena or a local file).
//!
//! Owner: model-engine SSM tier.
//! Invariants:
//! - Both stores refuse a blob whose length is not `blob_bytes`.
//! - [`RdmaSnapshotStore`] maps a new key only after its write succeeded, and a
//!   failed overwrite unmaps the key.

use std::collections::HashMap;
use std::num::NonZeroU64;
use std::sync::atomic::Ordering;

use anyhow::Result;
use parking_lot::Mutex;

use super::{BlobStoreStats, PagingTransport, SnapshotBlobStore, SnapshotTransport};

/// 2026-09-25: Paging-mode store: the peer owns residency, so this store forwards
/// put, get and remove over a [`PagingTransport`] under namespaced keys. `put`
/// returns `Ok(true)` or an error, never `Ok(false)`, unlike [`RdmaSnapshotStore`],
/// which returns `Ok(false)` when its arena is full.
pub(crate) struct PagingSnapshotStore {
    arena: Box<dyn PagingTransport>,
    blob_bytes: usize,
    /// 2026-09-25: Namespace folded into every key, so stores with different
    /// namespaces on one peer do not share wire keys.
    namespace: NonZeroU64,
}

impl PagingSnapshotStore {
    pub(crate) fn new(
        arena: Box<dyn PagingTransport>,
        blob_bytes: usize,
        namespace: NonZeroU64,
    ) -> Self {
        Self {
            arena,
            blob_bytes,
            namespace,
        }
    }

    /// 2026-09-25: Wire key: `mix64(key, namespace)`, pinned by
    /// `paging_isolation_tests::wire_key_is_mix64_of_key_and_ns`.
    fn wire(&self, key: u64) -> u64 {
        metrale_storage::tier::hash::mix64(key, self.namespace.get())
    }
}

impl SnapshotBlobStore for PagingSnapshotStore {
    fn put(&self, key: u64, bytes: &[u8]) -> Result<bool> {
        if bytes.len() != self.blob_bytes {
            anyhow::bail!(
                "paging put: {} != blob_bytes {}",
                bytes.len(),
                self.blob_bytes
            );
        }
        self.arena.paging_put(self.wire(key), bytes)?;
        Ok(true)
    }
    fn get(&self, key: u64, out: &mut [u8]) -> Result<bool> {
        if out.len() != self.blob_bytes {
            anyhow::bail!(
                "paging get: {} != blob_bytes {}",
                out.len(),
                self.blob_bytes
            );
        }
        self.arena.paging_get(self.wire(key), out)
    }
    fn remove(&self, key: u64) {
        if let Err(e) = self.arena.paging_remove(self.wire(key)) {
            tracing::debug!("paging remove {key:#x} failed: {e:#}");
        }
    }
    // 2026-09-25: Residency lives on the peer; this client does not track it.
    fn len(&self) -> usize {
        0
    }
    fn bytes_resident(&self) -> usize {
        0
    }
}

/// 2026-09-25: Fixed-slot store over a [`SnapshotTransport`] byte arena. Every blob
/// is `blob_bytes` long, so slot `i` lives at offset `i * blob_bytes`; a free list and
/// a `key -> slot` map track residency. A full arena or a wrong-sized blob makes
/// `put` return `Ok(false)`. The map and free list are mutated under the mutex;
/// the transport call runs outside it.
#[allow(dead_code)]
pub(crate) struct RdmaSnapshotStore {
    transport: Box<dyn SnapshotTransport>,
    blob_bytes: usize,
    inner: Mutex<RdmaInner>,
    pub stats: BlobStoreStats,
}

struct RdmaInner {
    map: HashMap<u64, usize>,
    /// 2026-09-25: Free slot indices, reused last-in first-out.
    free: Vec<usize>,
}

#[allow(dead_code)]
impl RdmaSnapshotStore {
    /// 2026-09-25: A store with `arena_slots` slots of `blob_bytes` each. The caller
    /// must give a transport whose arena covers `arena_slots * blob_bytes` bytes.
    pub(crate) fn new(
        transport: Box<dyn SnapshotTransport>,
        blob_bytes: usize,
        arena_slots: usize,
    ) -> Self {
        let free: Vec<usize> = (0..arena_slots).rev().collect();
        Self {
            transport,
            blob_bytes,
            inner: Mutex::new(RdmaInner {
                map: HashMap::new(),
                free,
            }),
            stats: BlobStoreStats::default(),
        }
    }

    #[inline]
    fn offset_of(&self, slot: usize) -> u64 {
        (slot * self.blob_bytes) as u64
    }
}

impl SnapshotBlobStore for RdmaSnapshotStore {
    fn put(&self, key: u64, bytes: &[u8]) -> Result<bool> {
        if bytes.len() != self.blob_bytes {
            self.stats.put_rejects.fetch_add(1, Ordering::Relaxed);
            return Ok(false);
        }
        // 2026-09-25: Pick the slot under the lock; map a new key only after the
        // write succeeds.
        let (slot, was_new) = {
            let mut g = self.inner.lock();
            match g.map.get(&key) {
                Some(&slot) => (slot, false),
                None => {
                    let Some(slot) = g.free.pop() else {
                        self.stats.put_rejects.fetch_add(1, Ordering::Relaxed);
                        return Ok(false);
                    };
                    (slot, true)
                }
            }
        };
        match self.transport.write_blob(self.offset_of(slot), bytes) {
            Ok(()) => {
                if was_new {
                    self.inner.lock().map.insert(key, slot);
                }
                self.stats.puts.fetch_add(1, Ordering::Relaxed);
                Ok(true)
            }
            Err(e) => {
                // 2026-09-25: A new slot returns to the free list. An overwritten
                // slot may hold partial bytes, so its key is unmapped and the slot freed.
                let mut g = self.inner.lock();
                if was_new {
                    g.free.push(slot);
                } else if let Some(s) = g.map.remove(&key) {
                    g.free.push(s);
                }
                Err(e)
            }
        }
    }

    fn get(&self, key: u64, out: &mut [u8]) -> Result<bool> {
        if out.len() != self.blob_bytes {
            self.stats.get_misses.fetch_add(1, Ordering::Relaxed);
            return Ok(false);
        }
        let slot = match self.inner.lock().map.get(&key) {
            Some(&slot) => slot,
            None => {
                self.stats.get_misses.fetch_add(1, Ordering::Relaxed);
                return Ok(false);
            }
        };
        self.transport.read_blob(self.offset_of(slot), out)?;
        self.stats.get_hits.fetch_add(1, Ordering::Relaxed);
        Ok(true)
    }

    fn remove(&self, key: u64) {
        let mut g = self.inner.lock();
        if let Some(slot) = g.map.remove(&key) {
            g.free.push(slot);
        }
    }

    fn len(&self) -> usize {
        self.inner.lock().map.len()
    }

    fn bytes_resident(&self) -> usize {
        self.inner.lock().map.len() * self.blob_bytes
    }
}

/// 2026-09-25: Transport-neutral name for [`RdmaSnapshotStore`], used by the decode
/// tier selector's NVMe arm over [`super::FileSnapshotArena`].
pub(crate) type ArenaSnapshotStore = RdmaSnapshotStore;

#[cfg(test)]
#[path = "arena_store_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "paging_isolation_tests.rs"]
mod paging_isolation_tests;
