// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Pins `fnv1a_64` and `mix64` to literal values, so a change to
//! either, which would change every key already stored on a peer, fails here.
//!
//! Owner: storage (tier).
//! Invariants: none beyond the types.

use super::{fnv1a_64, mix64};

/// 2026-09-25: `fnv1a_64` of three inputs, including the empty one (the offset
/// basis).
#[test]
fn fnv1a_64_matches_published_reference_vectors() {
    assert_eq!(fnv1a_64(b""), 0xcbf2_9ce4_8422_2325);
    assert_eq!(fnv1a_64(b"a"), 0xaf63_dc4c_8601_ec8c);
    assert_eq!(fnv1a_64(b"foobar"), 0x8594_4171_f739_67e8);
}

#[test]
fn fnv1a_64_is_order_sensitive_and_length_sensitive() {
    assert_ne!(fnv1a_64(b"ab"), fnv1a_64(b"ba"));
    assert_ne!(fnv1a_64(b"a"), fnv1a_64(b"aa"));
}

/// 2026-09-25: `mix64` is the fold behind the persisted wire keys, so its
/// output is pinned to literals.
#[test]
fn mix64_golden_pins() {
    assert_eq!(mix64(1, 0), 0x5692_161d_100b_05e5);
    assert_eq!(mix64(0, 1), 0xe220_a839_7b1d_cdaf);
    assert_eq!(mix64(0xdead_beef, 0xcafe_f00d), 0x5f3e_915c_5c36_4c56);
}

/// 2026-09-25: `mix64(0, 0) == 0`. This is why a namespace built from a mix is a
/// `NonZeroU64` with an explicit fallback for zero (model-engine
/// `ssm_tier/fingerprint.rs` falls back to `DECODE_DOMAIN`).
#[test]
fn mix64_has_a_zero_fixed_point() {
    assert_eq!(mix64(0, 0), 0);
}

/// 2026-09-25: The same key under two namespaces gives two different wire keys.
#[test]
fn mix64_separates_namespaces() {
    let key = 0x1234_5678_9abc_def0;
    assert_ne!(mix64(key, 1), mix64(key, 2));
    assert_ne!(mix64(key, 0), mix64(key, u64::MAX));
}

/// 2026-09-25: Both functions evaluate in a `const` context and give the same
/// value at run time.
#[test]
fn mix64_and_fnv_are_const_and_deterministic() {
    const H: u64 = fnv1a_64(b"metrale");
    const M: u64 = mix64(H, 7);
    assert_eq!(H, fnv1a_64(b"metrale"));
    assert_eq!(M, mix64(H, 7));
}
