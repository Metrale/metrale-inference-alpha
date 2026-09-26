// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests for `drafter_context_rows`: the GLM drafter's row cap follows the DSA indexer reservation.
//!
//! Owner: model-arch (GLM-5.3 MTP drafter).
//! Invariants: none beyond the types.

use super::drafter_context_rows;
use crate::glm5next_dsa::Glm5NextDsaConfig;

/// 2026-09-25: GLM-5.3's DSA shape; the same values as `glm5next_dsa::state::tests::cfg`.
fn cfg() -> Glm5NextDsaConfig {
    Glm5NextDsaConfig {
        hidden: 4096,
        index_heads: 32,
        index_head_dim: 128,
        index_kpool: 4,
        index_topk: 2048,
        always_select_tail: true,
        local_heads: 64,
        q_lora_rank: 1536,
        kv_lora_rank: 512,
        qk_nope_head_dim: 256,
        qk_rope_head_dim: 0,
        v_head_dim: 256,
        max_context: 16_384,
    }
}

/// 2026-09-25: A context longer than the indexer reservation is clamped to the reservation.
#[test]
fn a_declared_context_past_the_dsa_reservation_does_not_size_the_drafter() {
    let c = cfg();
    assert_eq!(drafter_context_rows(524_288, &c), 16_384);
    assert_eq!(drafter_context_rows(262_144, &c), 16_384);
}

/// 2026-09-25: A context at or below the reservation passes through unchanged.
#[test]
fn a_context_under_the_ceiling_is_untouched() {
    let c = cfg();
    assert_eq!(drafter_context_rows(8_192, &c), 8_192);
    assert_eq!(drafter_context_rows(16_384, &c), 16_384);
}

/// 2026-09-25: The cap is `max_dsa_context`, not a literal: it follows `max_context` and
/// rounds down to whole `index_kpool` pools. A hardcoded 16,384 passes the two tests above
/// and fails this one.
#[test]
fn the_cap_tracks_the_indexer_reservation_not_a_constant() {
    let mut c = cfg();
    c.max_context = 65_536;
    assert_eq!(drafter_context_rows(524_288, &c), 65_536);
    assert_eq!(drafter_context_rows(32_768, &c), 32_768);
    c.max_context = 65_538;
    assert_eq!(
        drafter_context_rows(524_288, &c),
        65_536,
        "whole pools only"
    );
}
