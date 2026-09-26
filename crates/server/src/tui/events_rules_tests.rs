// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for the event loop's pure rules in `events_rules.rs`, called directly without a tty.
//!
//! Owner: server tui.
//! Invariants: none beyond the types.

use super::*;
use std::sync::mpsc::channel;

#[test]
fn a_key_release_is_not_an_action() {
    assert!(!key_is_actionable(KeyEventKind::Release));
}

#[test]
fn a_held_key_still_acts() {
    assert!(key_is_actionable(KeyEventKind::Press));
    assert!(key_is_actionable(KeyEventKind::Repeat));
}

#[test]
fn only_the_newest_publication_survives() {
    let (tx, rx) = channel();
    for i in 0..4 {
        tx.send(i).expect("live receiver");
    }
    assert_eq!(newest(&rx), Some(3));
    assert_eq!(newest(&rx), None, "and the queue is drained, not stepped");
}

#[test]
fn an_empty_or_dead_channel_yields_nothing() {
    let (tx, rx) = channel::<u8>();
    assert_eq!(newest(&rx), None);
    drop(tx);
    assert_eq!(
        newest(&rx),
        None,
        "a run that never published is not an error"
    );
}

#[test]
fn a_dead_channel_still_yields_what_it_carried() {
    let (tx, rx) = channel();
    tx.send(7).expect("live receiver");
    drop(tx);
    assert_eq!(newest(&rx), Some(7));
}

#[test]
fn metrics_are_sampled_once_a_second_at_the_ten_hertz_tick() {
    let sampled: Vec<u32> = (1..=25).filter(|t| samples_metrics(*t)).collect();
    assert_eq!(sampled, vec![10, 20]);
}

#[test]
fn the_first_sample_waits_a_full_interval() {
    assert!(!samples_metrics(1));
    assert!(samples_metrics(SAMPLE_EVERY));
}

#[test]
fn the_wrap_costs_one_short_interval_and_nothing_else() {
    // 2026-09-26: The counter wraps. `u32::MAX` is not a multiple of `SAMPLE_EVERY`, so a
    // saturating counter would stop sampling there.
    assert!(samples_metrics(4_294_967_290));
    assert!(!samples_metrics(u32::MAX));
    assert!(samples_metrics(0), "zero is a multiple of everything");
    let after_wrap = (4_294_967_291u32..=u32::MAX)
        .filter(|t| samples_metrics(*t))
        .count();
    assert_eq!(after_wrap, 0, "so the short interval is six ticks, not two");
}

fn idle() -> LibraryPhase {
    LibraryPhase::default()
}

#[test]
fn nothing_lazy_happens_off_the_section_that_needs_it() {
    for s in [Section::Main, Section::Stats, Section::Network] {
        let w = tick_work(
            s,
            LibraryPhase {
                dirty: true,
                ..idle()
            },
        );
        assert_eq!(w, TickWork::default(), "{}", s.label());
    }
}

#[test]
fn entering_the_library_attaches_the_recipes_and_scans_the_cache() {
    let w = tick_work(
        Section::Library,
        LibraryPhase {
            dirty: true,
            ..idle()
        },
    );
    assert!(w.start_scan);
    assert!(w.attach_recipes);
    assert!(w.poll_library);
    assert!(!w.load_history);
}

#[test]
fn the_recipe_fetch_is_not_repeated_once_it_has_run() {
    let w = tick_work(
        Section::Library,
        LibraryPhase {
            dirty: true,
            recipes_attached: true,
            ..idle()
        },
    );
    assert!(!w.attach_recipes);
    assert!(w.start_scan, "the local scan is the half that does repeat");
}

#[test]
fn a_store_that_cannot_exist_is_not_asked_for_again() {
    // 2026-09-26: A failed attach sets `recipes_unavailable`, which stops `attach_recipes`
    // from firing again while `attached()` stays false.
    let w = tick_work(
        Section::Library,
        LibraryPhase {
            recipes_unavailable: true,
            ..idle()
        },
    );
    assert!(!w.attach_recipes);
    assert!(w.poll_library, "and the local half still renders");
}

#[test]
fn a_clean_library_starts_no_scan() {
    assert!(!tick_work(Section::Library, idle()).start_scan);
}

#[test]
fn a_dirty_library_waits_for_the_running_scan_to_finish() {
    // 2026-09-26: The event loop clears `library_dirty` only when `start_scan` is set, so a change
    // made during a scan waits for the next one.
    let mid_scan = LibraryPhase {
        dirty: true,
        scan_in_flight: true,
        ..idle()
    };
    assert!(
        !tick_work(Section::Library, mid_scan).start_scan,
        "so the caller leaves the flag set"
    );
    let settled = LibraryPhase {
        scan_in_flight: false,
        ..mid_scan
    };
    assert!(tick_work(Section::Library, settled).start_scan);
}

#[test]
fn the_benchmarks_section_is_the_only_one_that_reads_history() {
    let w = tick_work(
        Section::Benchmarks,
        LibraryPhase {
            dirty: true,
            ..idle()
        },
    );
    assert!(w.load_history);
    assert!(!w.start_scan, "and it does not do the Library's work");
    assert!(!w.attach_recipes);
    assert!(!w.poll_library);
}

#[test]
fn an_idle_loop_keeps_going() {
    assert_eq!(exit_kind(false, false, false), None);
}

#[test]
fn q_and_ctrl_c_stop_the_server() {
    assert_eq!(exit_kind(true, false, false), Some(Exit::Quit));
}

#[test]
fn detach_leaves_the_server_running() {
    assert_eq!(exit_kind(false, true, false), Some(Exit::Detach));
}

#[test]
fn a_shutdown_from_elsewhere_is_not_a_detach() {
    // 2026-09-26: A shutdown requested elsewhere (a signal, or `tui::stop_and_join`) sets neither
    // `should_quit` nor `detach`.
    assert_eq!(exit_kind(false, false, true), Some(Exit::ShuttingDown));
}

#[test]
fn the_users_intent_outranks_the_flag_it_raised() {
    // 2026-09-26: Ctrl+C and `/quit` set `should_quit` and also request shutdown.
    assert_eq!(exit_kind(true, false, true), Some(Exit::Quit));
    // 2026-09-26: A detach still wins over a shutdown request that raced it.
    assert_eq!(exit_kind(false, true, true), Some(Exit::Detach));
}
