// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Hash primitives for cache keys that outlive the process: FNV-1a/64
//! and the `mix64` fold.
//!
//! The keys land in the paging peer's swap file, which outlives the client that
//! wrote them, so the same input must give the same `u64` across toolchains,
//! platforms and rebuilds. Hence both functions are written out here over fixed
//! constants instead of coming from `std::hash` or a crate. The users import
//! them from here: model-engine's SSM tier (`ssm_tier/fingerprint.rs`, and the
//! wire key in `ssm_tier/arena_store.rs`) and the KV paging namespace
//! (`kv_paging/ns.rs`).
//!
//! FNV-1a is not collision-resistant: each step is an XOR and a multiply by an
//! odd constant, both invertible, so colliding inputs can be constructed. The
//! keys separate the configs of one trusted fleet; they are no boundary between
//! tenants that choose their own configs.
//!
//! Owner: storage (tier).
//! Invariants: `fnv1a_64` and `mix64` are pure `const fn`s of their inputs.

/// 2026-09-25: FNV-1a/64 offset basis.
pub const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
/// 2026-09-25: FNV-1a/64 prime.
pub const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;

/// 2026-09-25: FNV-1a/64 over `bytes`. It reads bytes, not words, so the result
/// does not depend on endianness.
pub const fn fnv1a_64(bytes: &[u8]) -> u64 {
    let mut h = FNV_OFFSET;
    let mut i = 0;
    while i < bytes.len() {
        h = (h ^ bytes[i] as u64).wrapping_mul(FNV_PRIME);
        i += 1;
    }
    h
}

/// 2026-09-25: The splitmix64 finalizer over `a ^ b * GOLDEN`.
///
/// The fold behind the persisted wire keys: model-engine's
/// `PagingSnapshotStore::wire(key)` is `mix64(key, namespace)`, and the decode
/// namespace folds `mix64(fingerprint, DECODE_DOMAIN)` with the client salt.
/// A change to it changes every stored key.
pub const fn mix64(a: u64, b: u64) -> u64 {
    let mut h = a ^ b.wrapping_mul(GOLDEN);
    h ^= h >> 30;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 27;
    h = h.wrapping_mul(0x94D0_49BB_1331_11EB);
    h ^ (h >> 31)
}

#[cfg(test)]
#[path = "hash_tests.rs"]
mod tests;
