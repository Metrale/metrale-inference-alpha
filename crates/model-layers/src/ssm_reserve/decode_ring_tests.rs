// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests for [`super`] (`ssm_reserve::decode_ring`).
//!
//! Owner: model-layers (SSM reserve).
//! Invariants: none beyond the types.
//!
//! The byte figures model one configuration: 48 GDN layers, a 151.5 MiB
//! per-sequence SSM state blob, `--max-batch-size 32`, a 45,823 MiB reserve
//! of which the 8-slot ring is 38,784 MiB, and 14.1 GiB free.
use super::*;

/// 2026-09-25: Per-sequence SSM state blob: 48 GDN layers x (h + conv) =
/// 158,859,264 B (151.5 MiB), the same derivation as `H_BLOB` + `CONV_BLOB` in
/// `ssm_reserve_tests.rs`.
const PER_SEQ_BLOB: usize = 48 * ((48 * 128 * 128 * 4) + ((16 * 128 * 2 + 48 * 128) * 4 * 4));
/// 2026-09-25: One unit of ring depth at `--max-batch-size 32`.
const SLOT_BYTES: usize = 32 * PER_SEQ_BLOB;
/// 2026-09-25: The 45,823 MiB reserve minus its 8-slot ring (38,784 MiB): the
/// part that does not scale with ring depth.
const RESERVE_WITHOUT_RING: usize = 7_039 * 1024 * 1024;

#[test]
fn decode_ring_decision_matrix() {
    let decide = |layers, spec, override_value, watchdogs| {
        let decision =
            decode_rollback_ring_slots_with(layers, spec, None, override_value, watchdogs);
        (decision.slots, decision.skip_reason)
    };
    let ring = metrale_kernels::DECODE_ROLLBACK_RING_SLOTS;

    for value in ["1", "true", " TRUE "] {
        assert!(
            watchdogs_disabled_from_value(Some(value)),
            "value={value:?}"
        );
    }
    for value in [None, Some(""), Some("0"), Some("false"), Some("yes")] {
        assert!(!watchdogs_disabled_from_value(value), "value={value:?}");
    }

    assert_eq!(decide(0, false, Some("1"), false), (0, None));
    assert_eq!(decide(48, true, Some("1"), true), (ring, None));
    assert_eq!(decide(48, false, Some("0"), false), (0, None));
    assert_eq!(
        decide(48, true, None, false),
        (0, Some("speculative decode active"))
    );
    assert_eq!(
        decide(48, false, None, true),
        (0, Some("watchdogs disabled"))
    );
    assert_eq!(
        decide(48, true, Some("invalid"), true),
        (0, Some("speculative decode active"))
    );
    assert_eq!(decide(48, false, None, false), (ring, None));
}

/// 2026-09-25: A published depth wins over the env override and the implicit
/// skips, so both sizing call sites (preflight and `TransformerModel::new`)
/// see the depth preflight reserved for.
#[test]
fn a_published_depth_outranks_the_default_the_env_and_the_skips() {
    let ring = metrale_kernels::DECODE_ROLLBACK_RING_SLOTS;
    let d = decode_rollback_ring_slots_with(48, false, Some(2), None, false);
    assert_eq!((d.slots, d.skip_reason), (2, None));
    // 2026-09-25: Against `METRALE_SSM_DECODE_RING=1` (depth 8).
    assert_eq!(
        decode_rollback_ring_slots_with(48, false, Some(2), Some("1"), false).slots,
        2
    );
    // 2026-09-25: Against both implicit skips (speculative decode, watchdogs
    // off).
    assert_eq!(
        decode_rollback_ring_slots_with(48, true, Some(4), None, true).slots,
        4
    );
    // 2026-09-25: A published 0 is an explicit off, not an implicit skip, so
    // it carries no skip_reason.
    let zero = decode_rollback_ring_slots_with(48, false, Some(0), None, false);
    assert_eq!((zero.slots, zero.skip_reason), (0, None));
    // 2026-09-25: No SSM layers outranks everything: no recurrent state.
    assert_eq!(
        decode_rollback_ring_slots_with(0, false, Some(ring), None, false).slots,
        0
    );
}

/// 2026-09-25: The parse CLI validation and the serve's publication both use.
#[test]
fn parse_accepts_auto_and_zero_through_eight() {
    assert_eq!(parse_decode_ring_slots("auto"), Ok(None));
    assert_eq!(parse_decode_ring_slots("0"), Ok(Some(0)));
    assert_eq!(
        parse_decode_ring_slots("8"),
        Ok(Some(metrale_kernels::DECODE_ROLLBACK_RING_SLOTS))
    );
    // 2026-09-25: Above the ring's ceiling, or not `auto` or a number, is an
    // error, not a clamp to 8.
    assert!(parse_decode_ring_slots("9").is_err());
    assert!(parse_decode_ring_slots("AUTO").is_err());
    assert!(parse_decode_ring_slots("").is_err());
    assert!(parse_decode_ring_slots("-1").is_err());
}

/// 2026-09-25: With 7,039 MiB of non-ring reserve and 14.1 GiB free, depth 8
/// (37.88 GiB of ring) does not fit and depth 1 does.
#[test]
fn autofit_picks_the_largest_fitting_depth() {
    let free = (14.1 * 1024.0 * 1024.0 * 1024.0) as usize;
    let fitted = fit_decode_ring_slots(8, RESERVE_WITHOUT_RING, SLOT_BYTES, free);
    assert_eq!(fitted, 1, "largest ladder depth that fits in 14.1 GiB");
    assert!(RESERVE_WITHOUT_RING + fitted * SLOT_BYTES <= free);
    assert!(
        RESERVE_WITHOUT_RING + 2 * SLOT_BYTES > free,
        "and 2 does not"
    );

    // 2026-09-25: With more room the fit lands on 4; the ladder holds no 5, 6
    // or 7.
    let roomier = 30 * 1024 * 1024 * 1024;
    assert_eq!(
        fit_decode_ring_slots(8, RESERVE_WITHOUT_RING, SLOT_BYTES, roomier),
        4
    );
    // 2026-09-25: A depth that already fits is kept.
    assert_eq!(
        fit_decode_ring_slots(8, RESERVE_WITHOUT_RING, SLOT_BYTES, 64 * 1024 * 1024 * 1024),
        8
    );
    // 2026-09-25: The fit never raises a depth: `start_slots` is a ceiling.
    assert_eq!(
        fit_decode_ring_slots(2, RESERVE_WITHOUT_RING, SLOT_BYTES, 64 * 1024 * 1024 * 1024),
        2
    );
}

/// 2026-09-25: When the rest of the reserve alone exceeds free memory, the fit
/// returns 0 and the caller must still refuse.
#[test]
fn autofit_is_zero_when_even_a_ringless_reserve_does_not_fit() {
    let free = RESERVE_WITHOUT_RING - 1;
    assert_eq!(
        fit_decode_ring_slots(8, RESERVE_WITHOUT_RING, SLOT_BYTES, free),
        0
    );
    assert!(
        RESERVE_WITHOUT_RING > free,
        "0 slots is still over budget — the caller refuses rather than booting ringless"
    );
    // 2026-09-25: A zero-byte ring also returns 0 when the rest does not fit.
    assert_eq!(fit_decode_ring_slots(8, RESERVE_WITHOUT_RING, 0, free), 0);
}

#[test]
fn the_fit_ladder_is_descending_and_starts_at_the_wired_default() {
    assert_eq!(
        DECODE_RING_FIT_LADDER[0],
        metrale_kernels::DECODE_ROLLBACK_RING_SLOTS
    );
    assert_eq!(*DECODE_RING_FIT_LADDER.last().unwrap(), 0);
    assert!(
        DECODE_RING_FIT_LADDER.windows(2).all(|w| w[0] > w[1]),
        "`fit_decode_ring_slots` returns the FIRST fitting rung, so the ladder \
         must be strictly descending or it would return a smaller depth than fits"
    );
}
