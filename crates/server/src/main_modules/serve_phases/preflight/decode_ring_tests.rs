// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests of the preflight ring term and its fit on the pre-load
//! yardstick, with no GPU, model or checkpoint.
//!
//! Owner: server startup (`met serve`).
//! Invariants: none beyond the types.
//!
//! The fixture is a 27B-shaped serve: 48 GDN layers, 151.5 MiB of SSM state
//! per sequence, `--max-batch-size 32`.

use super::*;
use clap::Parser as _;

/// 2026-09-26: The pre-load yardstick; `headroom_tests.rs` covers the
/// post-load one.
const PRE_LOAD: Yardstick = Yardstick::PreLoadFree("no residency prediction in this test");

/// 2026-09-26: 48 layers x (h + conv state) = 158,859,264 B, exactly
/// 151.5 MiB.
const PER_SEQ_BLOB: usize = 48 * ((48 * 128 * 128 * 4) + ((16 * 128 * 2 + 48 * 128) * 4 * 4));
/// 2026-09-26: A 45,823 MiB reserve less its 8-slot ring
/// (8 x 32 x 151.5 = 38,784 MiB).
const RESERVE_WITHOUT_RING: usize = 7_039 * 1024 * 1024;

fn args() -> cli::ServeArgs {
    cli::ServeArgs::parse_from(["met", "Qwen/Qwen3.8-27B-FP8", "--max-batch-size", "32"])
}

/// 2026-09-26: The formula text at depths 8, 1 and 0.
#[test]
fn the_formula_spells_out_snapshots_times_batch_times_per_seq_state() {
    assert_eq!(
        formula(8, 32, PER_SEQ_BLOB),
        "ring: 8 slots x 32 seqs x 151.5 MB/seq = 37.88 GB"
    );
    assert_eq!(
        formula(1, 32, PER_SEQ_BLOB),
        "ring: 1 slots x 32 seqs x 151.5 MB/seq = 4.73 GB"
    );
    assert_eq!(
        formula(0, 32, PER_SEQ_BLOB),
        "ring: 0 slots x 32 seqs x 151.5 MB/seq = 0.00 GB"
    );
}

/// 2026-09-26: With 14.1 GiB free and an 8-slot ring of 37.88 GiB, the fit
/// lowers the depth to 1, the largest ladder depth that fits, and warns.
#[test]
fn the_915_boot_shrinks_to_the_largest_fitting_depth_instead_of_refusing() {
    let args = args();
    let slot = slot_bytes(&args, PER_SEQ_BLOB);
    assert_eq!(slot, 32 * PER_SEQ_BLOB);
    let free = (14.1 * 1024.0 * 1024.0 * 1024.0) as usize;

    let fit = fit_ring(
        &args,
        8,
        slot,
        PER_SEQ_BLOB,
        RESERVE_WITHOUT_RING,
        free,
        &PRE_LOAD,
        false,
    );
    assert_eq!(fit.slots, 1, "8 -> 4 -> 2 -> 1 is the first rung that fits");
    assert!(
        RESERVE_WITHOUT_RING + fit.slots * slot <= free,
        "the depth it kept must actually fit"
    );

    let warning = fit.warning.expect("a shrink must be logged, never silent");
    assert!(
        warning.contains("ring: 1 slots x 32 seqs x 151.5 MB/seq = 4.73 GB"),
        "{warning}"
    );
    assert!(warning.contains("(was 37.88 GB)"), "{warning}");
    assert!(warning.contains("reserve 11.61 of 14.10 GB"), "{warning}");
    assert!(
        warning.contains("Sized from pre-load free memory"),
        "the WARN must name the yardstick it used: {warning}"
    );
    assert!(warning.contains("#915"), "{warning}");
    assert!(
        warning.contains("--ssm-decode-ring-slots"),
        "the warning must name the flag that pins a depth: {warning}"
    );
}

/// 2026-09-26: A reserve that fits keeps the requested depth, with no
/// warning but with a decision line.
#[test]
fn a_reserve_that_fits_is_not_touched() {
    let args = args();
    let slot = slot_bytes(&args, PER_SEQ_BLOB);
    let free = 64 * 1024 * 1024 * 1024;
    let fit = fit_ring(
        &args,
        8,
        slot,
        PER_SEQ_BLOB,
        RESERVE_WITHOUT_RING,
        free,
        &PRE_LOAD,
        false,
    );
    assert_eq!(fit.slots, 8);
    assert!(fit.warning.is_none());
    assert!(
        fit.decision.contains("pre-load free memory"),
        "{}",
        fit.decision
    );
}

/// 2026-09-26: A requested depth of 0, or a 0-byte slot (a model without SSM
/// state), is returned unchanged with no warning.
#[test]
fn a_ringless_serve_is_left_to_the_refusal() {
    let args = args();
    let free = RESERVE_WITHOUT_RING / 2;
    let fit = fit_ring(
        &args,
        0,
        slot_bytes(&args, PER_SEQ_BLOB),
        PER_SEQ_BLOB,
        RESERVE_WITHOUT_RING,
        free,
        &PRE_LOAD,
        false,
    );
    assert_eq!(fit.slots, 0);
    assert!(fit.warning.is_none());
    let fit = fit_ring(&args, 8, 0, 0, RESERVE_WITHOUT_RING, free, &PRE_LOAD, false);
    assert_eq!(fit.slots, 8);
    assert!(fit.warning.is_none());
}

/// 2026-09-26: On the pre-load yardstick, when even depth 0 does not fit, the
/// requested depth is returned unchanged with no warning, for the refusal.
#[test]
fn an_unfittable_reserve_keeps_the_requested_depth_for_the_refusal() {
    let args = args();
    let slot = slot_bytes(&args, PER_SEQ_BLOB);
    let fit = fit_ring(
        &args,
        8,
        slot,
        PER_SEQ_BLOB,
        RESERVE_WITHOUT_RING,
        RESERVE_WITHOUT_RING - 1,
        &PRE_LOAD,
        false,
    );
    assert_eq!(fit.slots, 8, "the refusal must quote what was asked for");
    assert!(fit.warning.is_none());
}

/// 2026-09-26: `--ssm-decode-ring-slots` parses through
/// `ssm_reserve::parse_decode_ring_slots`, and its clap default is `auto`.
#[test]
fn the_flag_parses_through_the_model_side_ssot() {
    use metrale_model_layers::ssm_reserve::parse_decode_ring_slots;
    assert_eq!(parse_decode_ring_slots("auto"), Ok(None));
    assert_eq!(parse_decode_ring_slots("2"), Ok(Some(2)));
    assert!(parse_decode_ring_slots("12").is_err());
    assert_eq!(args().ssm_decode_ring_slots, "auto");
}
