// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests for `kv_paging::ns`: golden values, per-input sensitivity of the
//! namespace, and the strict resolvers.
//!
//! Owner: storage, KV paging tier.
//! Invariants: none beyond the types.

use std::num::NonZeroU64;

use super::*;
use crate::group::GroupLayout;

/// 2026-09-25: 80 layers, 4096 blocks, 8 KV heads, block 16, head_dim 128, 2-byte
/// elements: group_stride 4096 and block_bytes 65536.
fn layout() -> GroupLayout {
    GroupLayout::new(80, 4096, 8, 16, 128, 2, 4096)
}

// 2026-09-25: The fingerprint metrale-model-engine's `fingerprint_tests.rs` pins for
// its hybrid config.
const FP: u64 = 0x5629_922c_51a1_6a10;
const SALT: u64 = 0xD00D_F00D_0000_0001;

fn ns_with(f: impl FnOnce(&mut (u64, GroupLayout, u32, u32, u32, u64))) -> NonZeroU64 {
    let mut a = (FP, layout(), 2u32, 16u32, 128u32, SALT);
    f(&mut a);
    derive_kv_ns(a.0, &a.1, a.2, a.3, a.4, a.5)
}

#[test]
// 2026-09-25: Asserting constants is the point: both are hashed into keys the peer
// stores.
#[allow(clippy::assertions_on_constants)]
fn version_and_domain_frozen() {
    assert_eq!(KV_NS_VERSION, 1);
    assert_eq!(KV_DOMAIN, 0x4B56_5041_4745_0001, "\"KV\" + \"PAGE\" + 1");
    assert_ne!(KV_DOMAIN, 0, "the domain must be a usable ns fallback");
}

#[test]
fn fnv1a_64_matches_reference_vectors() {
    assert_eq!(fnv1a_64(b""), 0xcbf2_9ce4_8422_2325);
    assert_eq!(fnv1a_64(b"a"), 0xaf63_dc4c_8601_ec8c);
    assert_eq!(fnv1a_64(b"foobar"), 0x8594_4171_f739_67e8);
}

/// 2026-09-25: `mix64` makes the wire key on both paging paths (`wire_key` here,
/// `PagingSnapshotStore::wire` in the SSM tier), so a changed constant changes every
/// stored key.
#[test]
fn mix64_frozen_literals() {
    assert_eq!(mix64(0, 0), 0x0);
    assert_eq!(mix64(1, 2), 0xbeeb_8da1_658e_ec67);
    assert_eq!(
        mix64(0x5EED_F00D_CAFE_D00D, 0xD3C0_DE12_A5B6_C7D8),
        0xe567_5d86_f750_1640
    );
}

#[test]
fn derive_kv_ns_golden() {
    assert_eq!(ns_with(|_| {}).get(), 0x74b4_02b0_f421_5375);
}

#[test]
fn wire_key_golden_and_mechanism() {
    let ns = ns_with(|_| {});
    assert_eq!(wire_key(ns, 42), mix64(42, ns.get()));
    assert_eq!(wire_key(ns, 42), 0x6d2d_470e_f594_7d4b);
    assert_eq!(wire_key(ns, 42), wire_key(ns, 42));
    let other = ns_with(|a| a.5 ^= 1);
    assert_ne!(wire_key(ns, 42), wire_key(other, 42));
}

// 2026-09-25: Changing any one input changes the namespace. The model fingerprint does
// not encode `elem_bytes`, `block_size` or `num_blocks`, so only this derivation
// separates layouts that differ in them.

#[test]
fn every_field_flips_the_namespace() {
    let base = ns_with(|_| {});
    let variants: [(&str, NonZeroU64); 9] = [
        ("model_fp", ns_with(|a| a.0 ^= 1)),
        ("elem_bytes", ns_with(|a| a.2 = 1)),
        ("block_size", ns_with(|a| a.3 = 32)),
        ("head_dim", ns_with(|a| a.4 = 64)),
        (
            "num_layers",
            ns_with(|a| a.1 = GroupLayout::new(48, 4096, 8, 16, 128, 2, 4096)),
        ),
        (
            "num_blocks",
            ns_with(|a| a.1 = GroupLayout::new(80, 2048, 8, 16, 128, 2, 4096)),
        ),
        (
            "num_kv_heads",
            ns_with(|a| a.1 = GroupLayout::new(80, 4096, 4, 16, 128, 2, 4096)),
        ),
        (
            "fs_block_size",
            ns_with(|a| a.1 = GroupLayout::new(80, 4096, 8, 16, 128, 2, 512)),
        ),
        ("client_salt", ns_with(|a| a.5 = SALT ^ 0xFFFF)),
    ];
    for (name, v) in variants {
        assert_ne!(v, base, "changing {name} must flip the KV namespace");
    }
}

#[test]
fn flag_selection_is_strict() {
    assert!(!kv_paging_selected(None).unwrap());
    assert!(!kv_paging_selected(Some("0")).unwrap());
    assert!(kv_paging_selected(Some("1")).unwrap());
    assert!(kv_paging_selected(Some("yes")).is_err());
    assert!(kv_paging_selected(Some("")).is_err());
}

/// 2026-09-25: The raw backend's handshake, the v2 header with `blob_bytes == 0`,
/// round-trips for this layout's arena size, which is within the peer's `1 << 42`
/// arena bound.
#[test]
fn flag_off_raw_v2_header_round_trips() {
    use crate::snapshot_swap::{PagingKind, encode_paging_v2_header, parse_paging_header};
    let l = layout();
    let num_groups = (l.num_layers as u64) * 2 * (l.num_blocks as u64) * (l.num_kv_heads as u64);
    // 2026-09-25: The arena size `RdmaKvBackend::connect` requests.
    let total = num_groups * l.group_stride;
    assert!(
        total <= (1 << 42),
        "raw totals stay under the peer's arena sanity bound"
    );
    let w = encode_paging_v2_header(PagingKind::KV, total, 0);
    let first = u64::from_le_bytes(w[0..8].try_into().unwrap());
    let mut c = std::io::Cursor::new(w[8..].to_vec());
    assert_eq!(
        parse_paging_header(first, &mut c).unwrap(),
        (PagingKind::KV, total, 0),
        "flag-OFF clients ride the RAW one-sided mode (blob_bytes == 0)"
    );
}

#[test]
fn cascade_paging_conflict_is_detected() {
    assert!(cascade_conflicts_with_paging(true, Some("1")).unwrap());
    assert!(!cascade_conflicts_with_paging(true, None).unwrap());
    assert!(!cascade_conflicts_with_paging(true, Some("0")).unwrap());
    assert!(!cascade_conflicts_with_paging(false, Some("1")).unwrap());
    assert!(cascade_conflicts_with_paging(true, Some("junk")).is_err());
}

#[test]
fn ns_override_is_strict() {
    let derived = NonZeroU64::new(0xFEED).unwrap();
    assert_eq!(resolve_kv_ns_from(None, derived).unwrap(), derived);
    assert_eq!(
        resolve_kv_ns_from(Some("0xD3C0"), derived).unwrap().get(),
        0xD3C0
    );
    assert_eq!(resolve_kv_ns_from(Some("77"), derived).unwrap().get(), 77);
    assert!(resolve_kv_ns_from(Some("junk"), derived).is_err());
    assert!(
        resolve_kv_ns_from(Some("0"), derived).is_err(),
        "ns=0 stays unrepresentable"
    );
}

#[test]
fn salt_override_is_strict() {
    assert_eq!(resolve_salt_from(None).unwrap(), None);
    assert_eq!(resolve_salt_from(Some("0x10")).unwrap(), Some(0x10));
    assert_eq!(resolve_salt_from(Some("7")).unwrap(), Some(7));
    assert!(resolve_salt_from(Some("nope")).is_err());
}

#[test]
fn arena_resolution_is_strict_and_block_aligned() {
    let bb = 65536u64;
    let err = resolve_arena_bytes_from(None, bb).unwrap_err().to_string();
    assert!(err.contains("METRALE_KV_PAGING_ARENA_GB"), "{err}");
    assert_eq!(resolve_arena_bytes_from(Some("1"), bb).unwrap(), 1 << 30);
    let a = resolve_arena_bytes_from(Some("0.001"), bb).unwrap();
    assert_eq!(a % bb, 0);
    assert!(a > 0 && a <= (0.001 * (1u64 << 30) as f64) as u64);
    assert!(resolve_arena_bytes_from(Some("lots"), bb).is_err());
    assert!(resolve_arena_bytes_from(Some("0"), bb).is_err());
    assert!(resolve_arena_bytes_from(Some("-1"), bb).is_err());
    assert!(resolve_arena_bytes_from(Some("0.00000001"), bb).is_err());
}

#[test]
fn miss_error_names_the_miss_proof_config() {
    let e = super::super::kv_miss_error(3, 77).to_string();
    assert!(e.contains("layer 3"), "{e}");
    assert!(e.contains("--swap-cap-gb-kv 0"), "{e}");
    assert!(e.contains("unrecoverable"), "{e}");
}
