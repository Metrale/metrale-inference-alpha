// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `random_u64` succeeds and varies. It seeds model-engine's
//! per-process decode client salt (`ssm_tier/fingerprint.rs`
//! `decode_client_salt`), which a constant source would make the same in every
//! process.
//!
//! Owner: storage (tier).
//! Invariants: none beyond the types.

use super::random_u64;

#[test]
fn random_u64_succeeds_and_varies() {
    // 2026-09-25: For 8 independent uniform draws, all 8 equal has
    // probability 2^-448.
    let draws: Vec<u64> = (0..8).map(|_| random_u64().expect("OS entropy")).collect();
    assert!(
        draws.iter().any(|&d| d != draws[0]),
        "8 identical draws — entropy source is degraded/constant: {draws:?}"
    );
}
