// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: PLE id bit-exactness against the reference.
//!
//! Owner: model-layers (PLE).
//! Invariants: none beyond the types.
//!
//! Pure token arithmetic, with no GPU and no checkpoint. A wrong hash still
//! returns valid rows from the table, so this is the check that catches it.
//!
//! Fixture: `bench/qwen4_exp/ple_id_fixtures.json`, written by
//! `bench/qwen4_exp/ple_golden.py` from the reference's own forward (the ids
//! are recorded, not re-derived).

use super::ids::{PleIdDims, ple_ngram_ids};

#[derive(serde::Deserialize)]
struct Fixture {
    tokens: Vec<u32>,
    eos_token_id: u32,
    ngram_size: usize,
    heads_per_ngram: usize,
    layer_multipliers: Vec<u64>,
    ngram_heads_vocab_sizes: Vec<u64>,
    ngram_heads_offsets: Vec<u64>,
    expected_ids: Vec<Vec<u64>>,
}

fn fixture() -> Fixture {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../bench/qwen4_exp/ple_id_fixtures.json"
    );
    serde_json::from_str(
        &std::fs::read_to_string(path)
            .expect("run bench/qwen4_exp/ple_golden.py to generate the fixture"),
    )
    .expect("ple_id_fixtures.json")
}

fn dims(f: &Fixture) -> PleIdDims {
    PleIdDims {
        ngram_size: f.ngram_size,
        heads_per_ngram: f.heads_per_ngram,
        multipliers: f.layer_multipliers.clone(),
        head_vocab_sizes: f.ngram_heads_vocab_sizes.clone(),
        head_offsets: f.ngram_heads_offsets.clone(),
        eos_token_id: f.eos_token_id,
    }
}

/// 2026-09-25: `tokens` in the fixture is the raw prompt; the reference
/// prepends `context_len` EOS before hashing, so this does the same and then
/// slices the trailing `tokens.len()` rows.
#[test]
fn ids_match_the_reference_bit_exactly() {
    let f = fixture();
    let d = dims(&f);
    d.validate().unwrap();

    let mut ctx = vec![d.eos_token_id; d.context_len()];
    ctx.extend_from_slice(&f.tokens);
    let all = ple_ngram_ids(&d, &ctx);
    let got = &all[all.len() - f.tokens.len()..];

    assert_eq!(got.len(), f.expected_ids.len(), "row count");
    for (t, (g, w)) in got.iter().zip(&f.expected_ids).enumerate() {
        assert_eq!(g, w, "token {t}");
    }
}

/// 2026-09-25: The fixture's prompt has an EOS in the middle, so the reference
/// test above covers the EOS branch; this pins it directly: the two tokens
/// after an EOS must not see across it.
#[test]
fn shift_refuses_to_read_across_eos() {
    let f = fixture();
    let d = dims(&f);
    let eos = d.eos_token_id;

    // 2026-09-25: The hash must depend on the predecessors: position 3 holds
    // token 200 in both, but its 3-gram context is (eos,100,200) vs
    // (999,998,200), so the ids must differ.
    let a = ple_ngram_ids(&d, &[eos, eos, 100, 200, 300]);
    let b = ple_ngram_ids(&d, &[eos, 999, 998, 200, 300]);
    assert_ne!(a[3], b[3], "different predecessors must hash differently");
    assert_ne!(a[2], b[2], "likewise at position 2");

    // 2026-09-25: It must depend on the right window per head block.
    // Position 4 (token 300) has 2-gram context (200,300) in both but 3-gram
    // context (100,200,300) vs (998,200,300). So the first `heads_per_ngram`
    // heads (the order-2 block) must match, and the order-3 block must
    // differ.
    let k = d.heads_per_ngram;
    assert_eq!(
        a[4][..k],
        b[4][..k],
        "order-2 heads see only (200,300) and must agree"
    );
    assert_ne!(
        a[4][k..],
        b[4][k..],
        "order-3 heads see the differing third token and must not agree"
    );

    // 2026-09-25: A segment after a mid-sequence EOS must produce the same ids
    // as the same tokens at the start of a sequence.
    let fresh = ple_ngram_ids(&d, &[eos, eos, 42, 43]);
    let after = ple_ngram_ids(&d, &[7, 8, eos, 42, 43]);
    assert_eq!(
        fresh[2], after[3],
        "first token of a post-EOS segment must hash like a sequence start"
    );
    assert_eq!(
        fresh[3], after[4],
        "second token of a post-EOS segment likewise"
    );
}

/// 2026-09-25: `validate` rejects an even multiplier.
#[test]
fn validate_rejects_an_even_multiplier() {
    let f = fixture();
    let mut d = dims(&f);
    d.multipliers[0] += 1;
    let e = d.validate().unwrap_err().to_string();
    assert!(e.contains("is even"), "{e}");
}

#[test]
fn validate_rejects_a_head_count_mismatch() {
    let f = fixture();
    let mut d = dims(&f);
    d.head_offsets.pop();
    assert!(d.validate().is_err());
}
