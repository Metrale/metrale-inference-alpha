// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests of the counter tracker, the J/token ratio, the energy
//! ring and request attribution.
//!
//! Owner: telemetry.
//! Invariants: none beyond the types.

use super::*;

#[test]
fn the_first_reading_anchors_and_forward_steps_are_deltas() {
    let mut t = CounterTracker::default();
    assert_eq!(t.advance(1_000), Advance::First);
    assert_eq!(t.advance(1_250), Advance::Delta(250));
    assert_eq!(
        t.advance(1_250),
        Advance::Delta(0),
        "a still counter is zero, not a reset"
    );
}

#[test]
fn a_wrap_is_told_from_a_reset_by_where_the_counter_sat() {
    let mut t = CounterTracker::default();
    t.advance(u64::MAX - 9);
    assert_eq!(t.advance(5), Advance::Wrapped(15));

    let mut t = CounterTracker::default();
    t.advance(1_000_000);
    assert_eq!(
        t.advance(400),
        Advance::Reset(400),
        "mid-range backwards is a restart"
    );

    // 2026-09-26: Backwards from the top quarter to mid-range is a restart too.
    let mut t = CounterTracker::default();
    t.advance(u64::MAX - 9);
    assert_eq!(t.advance(u64::MAX / 2), Advance::Reset(u64::MAX / 2));
}

#[test]
fn joules_per_token_refuses_an_empty_window() {
    assert_eq!(joules_per_token(1_500, 3), Some(0.5));
    assert_eq!(joules_per_token(1_500, 0), None, "no tokens");
    assert_eq!(
        joules_per_token(0, 3),
        None,
        "no energy means nothing was measured"
    );
}

fn point(at_ns: u64, energy_mj: u64, tokens: u64) -> EnergyPoint {
    EnergyPoint {
        at_ns,
        energy_mj,
        tokens,
    }
}

#[test]
fn the_ring_answers_the_newest_point_at_or_before_a_time() {
    let r = EnergyRing::with_capacity(8);
    assert_eq!(r.at_or_before(5), None, "an empty ring answers nothing");
    for i in 1..=5u64 {
        r.push(point(i * 10, i * 100, i));
    }
    assert_eq!(r.at_or_before(30).unwrap().at_ns, 30);
    assert_eq!(r.at_or_before(39).unwrap().at_ns, 30);
    assert_eq!(r.at_or_before(1_000).unwrap().at_ns, 50);
    assert_eq!(
        r.at_or_before(1).unwrap().at_ns,
        10,
        "before the ring: its oldest"
    );
    assert_eq!(r.back(0).unwrap().at_ns, 50);
    assert_eq!(r.back(4).unwrap().at_ns, 10);
    assert_eq!(r.back(5), None);
}

#[test]
fn an_overwritten_ring_keeps_only_live_points() {
    let r = EnergyRing::with_capacity(4);
    for i in 1..=10u64 {
        r.push(point(i, i, i));
    }
    // 2026-09-26: Capacity 4 keeps 3 live points (one slot of writer slack).
    assert_eq!(r.back(2).unwrap().at_ns, 8);
    assert_eq!(r.back(3), None);
    assert_eq!(
        r.at_or_before(2).unwrap().at_ns,
        8,
        "evicted history reads as the oldest live"
    );
}

#[test]
fn a_request_is_charged_its_token_share_of_its_window() {
    let r = EnergyRing::with_capacity(16);
    r.push(point(0, 0, 0));
    r.push(point(100, 6_000, 30));
    r.push(point(200, 9_000, 60));
    // 2026-09-26: 20 tokens over [100, 200]: the window drew 3000 mJ for 30 tokens.
    assert_eq!(request_millijoules(&r, 100, 200, 20), Some(2_000));
    // 2026-09-26: A window with no tokens emitted cannot attribute.
    let idle = EnergyRing::with_capacity(4);
    idle.push(point(0, 0, 5));
    idle.push(point(100, 500, 5));
    assert_eq!(request_millijoules(&idle, 0, 100, 1), None);
}
