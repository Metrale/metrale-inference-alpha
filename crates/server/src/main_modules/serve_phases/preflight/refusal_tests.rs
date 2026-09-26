// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests of the pre-load refusal text.
//!
//! Owner: server startup (`met serve`).
//! Invariants: none beyond the types.
//!
//! The fixture is a 27B-shaped serve: 48 GDN layers, 151.5 MiB of SSM state
//! per sequence, `--max-batch-size 32`.

use super::*;
use clap::Parser as _;

const PER_SEQ_BLOB: usize = 48 * ((48 * 128 * 128 * 4) + ((16 * 128 * 2 + 48 * 128) * 4 * 4));
const GIB: usize = 1024 * 1024 * 1024;

fn args() -> cli::ServeArgs {
    cli::ServeArgs::parse_from([
        "met",
        "Qwen/Qwen3.8-27B-FP8",
        "--max-batch-size",
        "32",
        "--max-seq-len",
        "24576",
    ])
}

fn config() -> ModelConfig {
    let mut c = ModelConfig::qwen3_next_80b_nvfp4();
    c.hidden_size = 5120;
    c.linear_num_key_heads = 16;
    c.linear_key_head_dim = 128;
    c.linear_num_value_heads = 48;
    c.linear_value_head_dim = 128;
    c
}

fn refusal(ring_slots: usize, ring_pinned: bool) -> String {
    reserve_refusal(
        &args(),
        &config(),
        Refusal {
            total_reserve: 46 * GIB,
            free_mem: 14 * GIB,
            seq_len_independent: 8 * GIB,
            ring_requested: 8,
            ring_slots,
            per_seq_blob: PER_SEQ_BLOB,
            ring_pinned,
        },
    )
    .to_string()
}

/// 2026-09-26: The refusal quotes `decode_ring::formula` and the sizes and
/// flags an operator acts on.
#[test]
fn the_refusal_carries_the_ring_formula() {
    let text = refusal(8, false);
    assert!(
        text.contains("ring: 8 slots x 32 seqs x 151.5 MB/seq = 37.88 GB"),
        "{text}"
    );
    assert_eq!(
        decode_ring::formula(8, 32, PER_SEQ_BLOB),
        "ring: 8 slots x 32 seqs x 151.5 MB/seq = 37.88 GB",
        "one formula, three readers",
    );
    for term in ["46.00 GB", "14.00 GB", "--max-seq-len", "--max-batch-size"] {
        assert!(text.contains(term), "{term} missing: {text}");
    }
}

/// 2026-09-26: The ring note's reason follows `ring_pinned`.
#[test]
fn the_refusal_says_why_shrinking_did_not_rescue_the_boot() {
    assert!(
        refusal(8, false).contains("even 0 ring slots does not fit"),
        "{}",
        refusal(8, false),
    );
    assert!(
        refusal(8, true).contains("an explicit --ssm-decode-ring-slots pins the depth"),
        "{}",
        refusal(8, true),
    );
}

/// 2026-09-26: With no ring requested the text has no ring note.
#[test]
fn a_ringless_serve_gets_no_ring_note() {
    let text = reserve_refusal(
        &args(),
        &config(),
        Refusal {
            total_reserve: 46 * GIB,
            free_mem: 14 * GIB,
            seq_len_independent: 8 * GIB,
            ring_requested: 0,
            ring_slots: 0,
            per_seq_blob: PER_SEQ_BLOB,
            ring_pinned: false,
        },
    )
    .to_string();
    assert!(!text.contains("ring:"), "{text}");
}
