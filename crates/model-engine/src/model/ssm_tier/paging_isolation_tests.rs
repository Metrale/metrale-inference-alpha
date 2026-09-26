// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Cross-model isolation on a shared paging peer, tested over an
//! in-process mock peer.
//!
//! The peer keys residency by the wire key alone, and the tier key is a hash of
//! tokens and adapter id only, so the namespace folded by
//! [`PagingSnapshotStore::wire`] is what keeps two models' state apart. The tests
//! pin both directions: distinct namespaces isolate, equal ones collide.
//!
//! Owner: model-engine (SSM snapshot tier).
//! Invariants: none beyond the types.

use std::collections::HashMap;
use std::num::NonZeroU64;
use std::sync::Arc;

use anyhow::Result;
use metrale_config::{LayerType, ModelConfig};
use parking_lot::Mutex;

use super::super::fingerprint::{ModelFingerprint, derive_decode_ns_salted, mix64};
use super::super::{MockSnapshotTransport, PagingTransport, SnapshotBlobStore, SnapshotTransport};
use super::PagingSnapshotStore;

const BLOB: usize = 64;
/// 2026-09-25: One logical tier key used by both models, as a shared prompt would give.
const K: u64 = 0x5EED_F00D_CAFE_D00D;

// 2026-09-25: A mock paging peer: one wire-key → slot map and one arena shared by
// every client store. It has no LRU eviction, swap or read pins.
pub(super) struct MockPagingPeer {
    blob_bytes: usize,
    inner: Mutex<MockPeerInner>,
    arena: MockSnapshotTransport,
}

struct MockPeerInner {
    /// 2026-09-25: Wire key → slot, as the peer's `Residency` map.
    map: HashMap<u64, usize>,
    free: Vec<usize>,
}

impl MockPagingPeer {
    pub(super) fn new(blob_bytes: usize, slots: usize) -> Self {
        Self {
            blob_bytes,
            inner: Mutex::new(MockPeerInner {
                map: HashMap::new(),
                free: (0..slots).rev().collect(),
            }),
            arena: MockSnapshotTransport::new(blob_bytes * slots),
        }
    }
}

impl PagingTransport for Arc<MockPagingPeer> {
    fn paging_put(&self, key: u64, bytes: &[u8]) -> Result<()> {
        let slot = {
            let mut g = self.inner.lock();
            match g.map.get(&key) {
                Some(&s) => s,
                None => {
                    let s = g.free.pop().expect("mock peer full — size the test arena");
                    g.map.insert(key, s);
                    s
                }
            }
        };
        self.arena
            .write_blob((slot * self.blob_bytes) as u64, bytes)
    }
    fn paging_get(&self, key: u64, out: &mut [u8]) -> Result<bool> {
        let slot = match self.inner.lock().map.get(&key) {
            Some(&s) => s,
            None => return Ok(false),
        };
        self.arena.read_blob((slot * self.blob_bytes) as u64, out)?;
        Ok(true)
    }
    fn paging_remove(&self, key: u64) -> Result<()> {
        let mut g = self.inner.lock();
        if let Some(s) = g.map.remove(&key) {
            g.free.push(s);
        }
        Ok(())
    }
}

fn hybrid() -> ModelConfig {
    ModelConfig::qwen3_next_80b_nvfp4()
}

fn dense() -> ModelConfig {
    let mut c = ModelConfig::qwen3_next_80b_nvfp4();
    c.model_type = "qwen3".to_string();
    c.num_hidden_layers = 28;
    c.layer_types = vec![LayerType::FullAttention; 28];
    c.num_experts = 0;
    c.linear_num_key_heads = 0;
    c.linear_key_head_dim = 0;
    c.linear_num_value_heads = 0;
    c.linear_value_head_dim = 0;
    c
}

fn ns_of(cfg: &ModelConfig) -> NonZeroU64 {
    ModelFingerprint::derive_with_id(cfg, BLOB, "")
        .unwrap()
        .nonzero()
}

fn store(peer: &Arc<MockPagingPeer>, ns: NonZeroU64) -> PagingSnapshotStore {
    PagingSnapshotStore::new(Box::new(peer.clone()), BLOB, ns)
}

#[test]
fn distinct_fingerprints_do_not_cross_serve() {
    let peer = Arc::new(MockPagingPeer::new(BLOB, 8));
    let a = store(&peer, ns_of(&hybrid()));
    let b = store(&peer, ns_of(&dense()));
    a.put(K, &[0xAA; BLOB]).unwrap();
    let mut out = [0x5Au8; BLOB];
    assert!(
        !b.get(K, &mut out).unwrap(),
        "model B must MISS model A's state for the same logical key"
    );
    assert_eq!(out, [0x5A; BLOB], "a miss must leave `out` untouched");
    assert!(a.get(K, &mut out).unwrap());
    assert_eq!(out, [0xAA; BLOB]);
}

// 2026-09-25: Two stores with equal namespaces share entries on the peer, so
// model B reads model A's state as a hit.
#[test]
fn equal_namespaces_cross_serve_the_old_default_bug() {
    let peer = Arc::new(MockPagingPeer::new(BLOB, 8));
    let shared = NonZeroU64::new(metrale_kernels::DECODE_DOMAIN).unwrap();
    let a = store(&peer, shared);
    let b = store(&peer, shared);
    a.put(K, &[0xAA; BLOB]).unwrap();
    let mut out = [0u8; BLOB];
    assert!(
        b.get(K, &mut out).unwrap(),
        "equal namespaces DO collide — this is the pinned bug"
    );
    assert_eq!(
        out, [0xAA; BLOB],
        "model B silently served model A's recurrent state as a cache HIT"
    );
}

#[test]
fn same_fingerprint_round_trips_across_clients() {
    let peer = Arc::new(MockPagingPeer::new(BLOB, 8));
    let ns = ns_of(&hybrid());
    let c1 = store(&peer, ns);
    let c2 = store(&peer, ns);
    let blob: Vec<u8> = (0..BLOB as u8).collect();
    c1.put(K, &blob).unwrap();
    let mut out = vec![0u8; BLOB];
    assert!(
        c2.get(K, &mut out).unwrap(),
        "same model, second client: shared warm-cache HIT"
    );
    assert_eq!(out, blob, "bit-identical restore");
}

#[test]
fn remove_is_namespace_scoped() {
    let peer = Arc::new(MockPagingPeer::new(BLOB, 8));
    let a = store(&peer, ns_of(&hybrid()));
    let b = store(&peer, ns_of(&dense()));
    a.put(K, &[0xAA; BLOB]).unwrap();
    b.remove(K);
    let mut out = [0u8; BLOB];
    assert!(a.get(K, &mut out).unwrap(), "B's remove must not evict A");
    a.remove(K);
    assert!(!a.get(K, &mut out).unwrap(), "A's remove evicts A");
}

#[test]
fn decode_and_marconi_tiers_do_not_cross_serve_on_one_peer() {
    // 2026-09-25: A model's decode namespace (`derive_decode_ns_salted`) differs
    // from its spill-tier namespace (the bare fingerprint).
    let peer = Arc::new(MockPagingPeer::new(BLOB, 8));
    let fp = ModelFingerprint::derive_with_id(&hybrid(), BLOB, "").unwrap();
    let marconi = store(&peer, fp.nonzero());
    let decode = store(&peer, derive_decode_ns_salted(fp.get(), 0x00C1_1E17));
    marconi.put(K, &[0xAA; BLOB]).unwrap();
    let mut out = [0u8; BLOB];
    assert!(
        !decode.get(K, &mut out).unwrap(),
        "decode namespace must not serve Marconi state"
    );
}

#[test]
fn wire_fold_is_deterministic_per_store_instance() {
    let peer = Arc::new(MockPagingPeer::new(BLOB, 8));
    let ns = ns_of(&hybrid());
    let s1 = store(&peer, ns);
    let s2 = store(&peer, ns);
    for key in [0u64, 1, K, u64::MAX] {
        assert_eq!(s1.wire(key), s2.wire(key), "same (ns, key) → same wire key");
    }
    let other = store(&peer, ns_of(&dense()));
    assert_ne!(s1.wire(K), other.wire(K));
}

/// 2026-09-25: The wire key itself is pinned. The determinism test above would
/// still pass if the whole fold changed, while keys already on a peer became
/// unreachable. It also pins `wire(key) == mix64(key, ns)`.
#[test]
fn wire_key_is_mix64_of_key_and_ns() {
    let peer = Arc::new(MockPagingPeer::new(BLOB, 8));
    // 2026-09-25: The golden hybrid fingerprint (`golden_fingerprint_hybrid_moe_is_pinned`).
    let ns = NonZeroU64::new(0x5629_922c_51a1_6a10).unwrap();
    let s = store(&peer, ns);
    let key = 0x0123_4567_89ab_cdef_u64;

    assert_eq!(
        s.wire(key),
        0x51f6_0258_9d95_1e89,
        "persisted wire key rotated"
    );
    assert_eq!(
        s.wire(key),
        mix64(key, ns.get()),
        "wire() must BE mix64(key, ns)"
    );
}
