// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests of the Stats section: `fmt_ms` at its millisecond to
//! second seam, and the panes rendered into a buffer, where an idle server
//! shows dashes rather than zeros.
//!
//! Owner: server tui.
//! Invariants: none beyond the types.

use super::*;
use crate::tui::app::Section;
use crate::tui::render::harness::{has, screen};
use metrale_speculative::snapshot::{MtpModeSnap, SchedulerSnapshot};

fn stats_app() -> App {
    let mut a = crate::tui::render::tests::app();
    a.section = Section::Stats;
    a
}

fn sched() -> SchedulerSnapshot {
    SchedulerSnapshot {
        active_seqs: 3,
        prefilling_seqs: 1,
        swapped_seqs: 0,
        pending_len: 2,
        kv_blocks_free: 400,
        kv_blocks_total: 1000,
        ssm_slots_used: 4,
        ssm_slots_total: 16,
        mtp_mode: MtpModeSnap::Mtp,
        delivered_tps: 42.0,
        steps_total: 900,
        published_at: std::time::Instant::now(),
    }
}

#[test]
fn ttft_switches_from_milliseconds_to_seconds_at_one_thousand() {
    assert_eq!(fmt_ms(None), "—", "an unmeasured percentile is not a zero");
    assert_eq!(fmt_ms(Some(0.0)), "0ms");
    assert_eq!(fmt_ms(Some(999.0)), "999ms");
    assert_eq!(fmt_ms(Some(1000.0)), "1.0s");
    assert_eq!(fmt_ms(Some(1024.0)), "1.0s");
    assert_eq!(fmt_ms(Some(2985.0)), "3.0s");
    // 2026-09-26: The unit is chosen before rounding, so a value just under a
    // second renders as a four-digit millisecond count, not "1.0s".
    assert_eq!(fmt_ms(Some(999.6)), "1000ms");
}

#[test]
fn an_idle_server_reports_nothing_measured_rather_than_a_plausible_zero() {
    let rows = screen(&stats_app(), 160, 48);
    for title in ["REQUESTS", "THROUGHPUT", "TTFT", "GPU"] {
        assert!(has(&rows, title), "the {title} tile is drawn:\n{rows:#?}");
    }
    assert!(
        has(&rows, "p50 —"),
        "no requests, no percentile:\n{rows:#?}"
    );
    assert!(has(&rows, "p90 —"));
    assert!(has(&rows, "0.0 tok/s"));
    assert!(
        has(&rows, "prefix-cache hit —"),
        "an unsampled hit rate is a dash, not 0%:\n{rows:#?}"
    );
}

#[test]
fn a_server_under_load_reports_its_measurements_in_the_tiles() {
    let mut a = stats_app();
    a.stats.requests_total = 1007;
    a.stats.requests_active = 8;
    a.stats.gen_tps = 12.55;
    a.stats.ttft_p50_ms = Some(412.0);
    a.stats.ttft_p90_ms = Some(2985.0);
    a.stats.gpu_known = true;
    a.stats.metrale_used_gb = 57.25;
    a.stats.gpu_free_gb = 62.4;
    a.stats.bytes_in_rate = 2048.0;
    a.stats.bytes_out_rate = 3_145_728.0;
    let rows = screen(&a, 160, 48);
    assert!(has(&rows, "1007"), "{rows:#?}");
    assert!(has(&rows, "● 8"));
    assert!(has(&rows, "12.6 tok/s"), "one decimal:\n{rows:#?}");
    assert!(has(&rows, "p50 412ms"));
    assert!(has(&rows, "p90 3.0s"));
    assert!(has(&rows, "metrale 57.2 GB"));
    assert!(has(&rows, "free 62.4"));
    assert!(has(&rows, "↓2.0 KB/s ↑3.0 MB/s"), "{rows:#?}");
}

#[test]
fn the_sequences_pane_shows_the_scheduler_only_once_one_has_published() {
    let mut a = stats_app();
    let bare = screen(&a, 160, 48);
    assert!(has(&bare, "active 0 · prefill 0 · swapped 0 · queue 0"));
    assert!(
        !has(&bare, " KV"),
        "no KV gauge without a scheduler to measure:\n{bare:#?}"
    );

    a.stats.sched = Some(sched());
    a.stats.gpu_known = true;
    a.stats.gpu_total_gb = 119.7;
    a.stats.metrale_used_gb = 57.2;
    a.stats.host_total_gb = 119.7;
    a.stats.host_avail_gb = 40.0;
    let rows = screen(&a, 160, 48);
    assert!(has(&rows, "active 3 · prefill 1 · swapped 0 · queue 2"));
    assert!(has(&rows, "600/1000"), "KV blocks used/total:\n{rows:#?}");
    assert!(has(&rows, "4/16"), "SSM slots:\n{rows:#?}");
    assert!(has(&rows, "57/120"), "GPU GB:\n{rows:#?}");
    assert!(has(&rows, "80/120"), "host RAM GB:\n{rows:#?}");
}

#[test]
fn the_speculation_pane_names_the_gate_and_the_rate_it_delivered() {
    let mut a = stats_app();
    a.stats.sched = Some(sched());
    a.stats.spec_accept = vec![("4".into(), 3, 4)];
    a.stats.prefix_hit_rate = Some(0.735);
    a.stats.prefix_hit_tokens = 18_432;
    a.stats.tool_calls_total = 61;
    a.stats.entropy = 1.234;
    let rows = screen(&a, 160, 48);
    assert!(has(&rows, "MTP gate speculative"), "{rows:#?}");
    assert!(has(&rows, "delivered 42 tok/s"));
    assert!(has(&rows, "accept k=4"));
    assert!(has(&rows, "75%"));
    assert!(has(&rows, "prefix-cache hit 74% · 18432 tok warm"));
    assert!(has(&rows, "tool calls 61 · entropy 1.23"));
}

#[test]
fn a_speculation_k_that_has_never_run_is_left_out_rather_than_shown_as_zero() {
    let mut a = stats_app();
    a.stats.sched = Some(sched());
    a.stats.spec_accept = vec![("3".into(), 0, 0), ("4".into(), 4, 4)];
    let rows = screen(&a, 160, 48);
    assert!(!has(&rows, "accept k=3"), "{rows:#?}");
    // 2026-09-26: A fully accepted K fills every cell, and the empty-track
    // count `w - filled` stays at zero.
    assert!(has(&rows, "accept k=4"));
    assert!(has(&rows, "100%"));
}

#[test]
fn the_ttft_histogram_labels_its_buckets() {
    let mut a = stats_app();
    a.stats.ttft_buckets = vec![
        (0.1, 5),
        (0.5, 12),
        (1.0, 18),
        (3.0, 24),
        (f64::INFINITY, 25),
    ];
    let rows = screen(&a, 160, 48);
    assert!(has(&rows, "TTFT DISTRIBUTION"), "{rows:#?}");
    for label in [".10", ".50", "1", "3"] {
        assert!(
            has(&rows, label),
            "bucket {label} is unlabelled:\n{rows:#?}"
        );
    }
    assert!(
        rows.iter().any(|r| r.contains('█')),
        "the counts are actually drawn:\n{rows:#?}"
    );
}

#[test]
fn the_throughput_chart_captions_both_rates() {
    let mut a = stats_app();
    a.stats.gen_tps = 59.9;
    a.stats.prompt_tps = 1841.0;
    let rows = screen(&a, 160, 48);
    assert!(has(&rows, "gen 60 tok/s · prompt 1841 tok/s"), "{rows:#?}");
}

#[test]
fn the_stats_pane_survives_narrow_and_short_terminals() {
    let mut a = stats_app();
    a.stats.sched = Some(sched());
    a.stats.spec_accept = vec![("4".into(), 3, 4)];
    a.stats.ttft_buckets = vec![(0.1, 5), (1.0, 9), (f64::INFINITY, 9)];
    for (w, h) in [(20u16, 8u16), (20, 40), (40, 12), (60, 20), (200, 3)] {
        let rows = screen(&a, w, h);
        assert_eq!(rows.len(), h as usize, "{w}x{h} drew a partial frame");
    }
}

#[cfg(test)]
mod gauges {
    use super::*;

    /// 2026-09-26: A gauge whose total is zero draws an empty bar, not a full
    /// one, and does not divide by zero.
    #[test]
    fn an_unknown_total_reads_empty_not_full() {
        let mut a = stats_app();
        a.stats.gpu_known = true;
        a.stats.gpu_total_gb = 0.0;
        a.stats.metrale_used_gb = 0.0;
        a.stats.host_total_gb = 0.0;
        let rows = screen(&a, 160, 48);
        assert!(has(&rows, "0/0"), "{rows:#?}");
    }

    #[test]
    fn a_gauge_pushed_past_its_total_clamps_instead_of_overflowing() {
        let mut a = stats_app();
        a.stats.gpu_known = true;
        a.stats.gpu_total_gb = 100.0;
        a.stats.metrale_used_gb = 250.0;
        let rows = screen(&a, 160, 48);
        assert_eq!(rows.len(), 48);
        assert!(has(&rows, "250/100"), "the numbers stay honest:\n{rows:#?}");
    }
}

/// 2026-09-26: Without a GPU memory reading the GPU tile shows a dash, not
/// `metrale 0.0 GB`; with one it shows the reading.
#[test]
fn a_box_with_no_gpu_reading_shows_a_dash_not_zero() {
    let mut a = crate::tui::render::tests::app();
    a.section = crate::tui::app::Section::Stats;
    a.stats.gpu_known = false;
    let rows = screen(&a, 120, 40);
    assert!(
        !has(&rows, "metrale 0.0 GB"),
        "a zero must never be presented as a GPU measurement:\n{rows:#?}"
    );

    // 2026-09-26: With a reading, the tile shows it.
    a.stats.gpu_known = true;
    a.stats.metrale_used_gb = 12.5;
    a.stats.gpu_free_gb = 100.0;
    a.stats.gpu_total_gb = 112.5;
    let rows = screen(&a, 120, 40);
    assert!(
        has(&rows, "metrale 12.5 GB"),
        "a real reading must still be shown:\n{rows:#?}"
    );
}
