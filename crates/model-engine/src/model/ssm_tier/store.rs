// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The [`SnapshotBlobStore`] trait and [`MemBlobStore`], its host-RAM implementation.
//!
//! Owner: model-engine (SSM snapshot tier).
//! Invariants:
//! - `MemBlobStore::bytes_resident` is the summed length of the blobs it holds.
//! - With `cap_bytes > 0`, a `MemBlobStore::put` that returns `Ok(true)` leaves
//!   `bytes_resident() <= cap_bytes`.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::Result;
use parking_lot::Mutex;

/// 2026-09-25: A keyed byte-blob store for the SSM spill tier. One blob is one
/// snapshot: the `h` and `conv` state of every SSM layer, as gathered by
/// `SsmSnapshotPool::spill_slot`.
pub(crate) trait SnapshotBlobStore: Send + Sync {
    /// 2026-09-25: Store `bytes` under `key`, replacing any prior value.
    /// `Ok(false)` means the store refused the write; both callers of
    /// `SsmSnapshotPool::spill_slot` then call `retire_refused_spill`.
    fn put(&self, key: u64, bytes: &[u8]) -> Result<bool>;

    /// 2026-09-25: Copy the blob for `key` into `out`. `Ok(false)` if `key` is
    /// absent. When `out.len()` differs from the blob size nothing is copied, and
    /// the result is `Ok(false)` or `Err` depending on the store.
    fn get(&self, key: u64, out: &mut [u8]) -> Result<bool>;

    /// 2026-09-25: Drop the blob for `key` if present.
    fn remove(&self, key: u64);

    /// 2026-09-25: Number of blobs the store holds (0 for a store whose peer owns
    /// residency).
    fn len(&self) -> usize;

    /// 2026-09-25: Bytes of the blobs in the store's first tier (for
    /// `UnifiedSnapshotStore`, the hot arena only). 0 for a store whose peer
    /// owns residency.
    fn bytes_resident(&self) -> usize;
}

/// 2026-09-25: Per-store operation counters.
#[derive(Default)]
pub(crate) struct BlobStoreStats {
    pub puts: AtomicUsize,
    pub put_rejects: AtomicUsize,
    pub get_hits: AtomicUsize,
    pub get_misses: AtomicUsize,
    pub evictions: AtomicUsize,
}

/// 2026-09-25: Host-RAM spill tier, bounded by `cap_bytes` with FIFO eviction;
/// `cap_bytes == 0` means unbounded.
pub(crate) struct MemBlobStore {
    inner: Mutex<MemInner>,
    bytes: AtomicUsize,
    cap_bytes: usize,
    pub stats: BlobStoreStats,
}

struct MemInner {
    map: HashMap<u64, Vec<u8>>,
    /// 2026-09-25: Insertion order for FIFO eviction. A key is pushed on its
    /// first insert; a re-`put` of an existing key keeps its position.
    order: std::collections::VecDeque<u64>,
}

impl MemBlobStore {
    pub(crate) fn new(cap_bytes: usize) -> Self {
        Self {
            inner: Mutex::new(MemInner {
                map: HashMap::new(),
                order: std::collections::VecDeque::new(),
            }),
            bytes: AtomicUsize::new(0),
            cap_bytes,
            stats: BlobStoreStats::default(),
        }
    }
}

impl SnapshotBlobStore for MemBlobStore {
    fn put(&self, key: u64, bytes: &[u8]) -> Result<bool> {
        // 2026-09-25: A blob larger than the whole cap can never fit, so refuse it
        // before evicting anything.
        if self.cap_bytes != 0 && bytes.len() > self.cap_bytes {
            self.stats.put_rejects.fetch_add(1, Ordering::Relaxed);
            return Ok(false);
        }
        let mut g = self.inner.lock();
        // 2026-09-25: On an overwrite, uncount the old blob's bytes first.
        if let Some(old) = g.map.get(&key) {
            self.bytes.fetch_sub(old.len(), Ordering::Relaxed);
        } else {
            g.order.push_back(key);
        }
        // 2026-09-25: Evict the oldest blobs until the new one fits under the cap.
        if self.cap_bytes != 0 {
            while self.bytes.load(Ordering::Relaxed) + bytes.len() > self.cap_bytes {
                // 2026-09-25: The key being overwritten keeps its FIFO position
                // and its old bytes are already uncounted, so evict the oldest
                // other key.
                let Some(victim_pos) = g.order.iter().position(|&candidate| candidate != key)
                else {
                    break;
                };
                let victim = g
                    .order
                    .remove(victim_pos)
                    .expect("position came from the same queue");
                if let Some(v) = g.map.remove(&victim) {
                    self.bytes.fetch_sub(v.len(), Ordering::Relaxed);
                    self.stats.evictions.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
        self.bytes.fetch_add(bytes.len(), Ordering::Relaxed);
        g.map.insert(key, bytes.to_vec());
        self.stats.puts.fetch_add(1, Ordering::Relaxed);
        Ok(true)
    }

    fn get(&self, key: u64, out: &mut [u8]) -> Result<bool> {
        let g = self.inner.lock();
        match g.map.get(&key) {
            Some(v) if v.len() == out.len() => {
                out.copy_from_slice(v);
                self.stats.get_hits.fetch_add(1, Ordering::Relaxed);
                Ok(true)
            }
            _ => {
                self.stats.get_misses.fetch_add(1, Ordering::Relaxed);
                Ok(false)
            }
        }
    }

    fn remove(&self, key: u64) {
        let mut g = self.inner.lock();
        if let Some(v) = g.map.remove(&key) {
            self.bytes.fetch_sub(v.len(), Ordering::Relaxed);
            g.order.retain(|&k| k != key);
        }
    }

    fn len(&self) -> usize {
        self.inner.lock().map.len()
    }

    fn bytes_resident(&self) -> usize {
        self.bytes.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
#[path = "store_tests.rs"]
mod tests;
