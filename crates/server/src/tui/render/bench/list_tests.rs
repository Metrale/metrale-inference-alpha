// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for the suite list's windowing: every entry reachable, the clip indicator, offset clamping and the published page size.
//!
//! Owner: server tui.
//! Invariants: none beyond the types.

use crate::tui::app::{App, Section};
use crate::tui::bench_state::View;
use crate::tui::render::harness::{has, screen};

/// 2026-09-26: A terminal height derived from the registry size, `3n + 20`,
/// so it holds the whole registry on one page as the registry grows: an entry
/// takes at least `COMPACT` (3) rows.
fn tall_enough() -> u16 {
    let n = metrale_bench::registry::all().len();
    u16::try_from(n * 3 + 20).expect("fits a u16 terminal")
}

fn list_app(selected: usize) -> App {
    let mut a = crate::tui::render::tests::app();
    a.section = Section::Benchmarks;
    a.bench.view = View::List;
    a.bench.selected = selected;
    a
}

/// 2026-09-26: Holds for any registry size.
#[test]
fn every_registered_benchmark_is_reachable_by_scrolling() {
    // 2026-09-26: 24 rows clip the suite, so finding an entry means the
    // offset followed the selection. 120 columns so that names are not cut
    // horizontally.
    let all = metrale_bench::registry::all();
    for (i, descriptor) in all.iter().enumerate() {
        let rows = screen(&list_app(i), 120, 24);
        assert!(
            has(&rows, descriptor.name),
            "{} unreachable at 120x24:\n{rows:#?}",
            descriptor.name
        );
    }
}

#[test]
fn the_clip_indicator_appears_only_when_the_list_is_clipped() {
    let n = metrale_bench::registry::all().len();
    // 2026-09-26: 80x24 clips the suite, so the bottom border names the
    // window from entry 1. Both halves are matched on one row, so the footer's
    // "1-7 jump" hint cannot satisfy it.
    let rows = screen(&list_app(0), 80, 24);
    assert!(
        rows.iter()
            .any(|r| r.contains("─ 1-") && r.contains(&format!("of {n} ─"))),
        "clipped list must name its window:\n{rows:#?}"
    );
    // 2026-09-26: A registry-sized terminal holds every entry, so no
    // indicator.
    let rows = screen(&list_app(0), 160, tall_enough());
    assert!(
        !has(&rows, &format!("of {n} ─")),
        "unclipped list must not draw an indicator:\n{rows:#?}"
    );
}

#[test]
fn the_indicator_tracks_the_selection_to_the_bottom() {
    let n = metrale_bench::registry::all().len();
    let rows = screen(&list_app(n - 1), 80, 24);
    assert!(
        has(&rows, &format!("-{n} of {n} ─")),
        "with the last entry selected the window must end at {n}:\n{rows:#?}"
    );
}

#[test]
fn the_offset_is_clamped_at_both_ends() {
    // 2026-09-26: A selection past the registry shows the last page.
    let all = metrale_bench::registry::all();
    let rows = screen(&list_app(all.len() + 40), 80, 24);
    assert!(
        has(&rows, all[all.len() - 1].name),
        "an over-large selection clamps to the last page:\n{rows:#?}"
    );
    let rows = screen(&list_app(0), 80, 24);
    assert!(has(&rows, all[0].name), "{rows:#?}");
}

#[test]
fn compaction_keeps_the_summary_and_duration_on_every_entry() {
    // 2026-09-26: The compact layout drops only the blank separator; the
    // duration line stays.
    let all = metrale_bench::registry::all();
    let rows = screen(&list_app(0), 80, 24);
    assert!(
        has(&rows, all[0].duration_hint),
        "the duration line survives the compact layout:\n{rows:#?}"
    );
}

#[test]
fn the_renderer_publishes_the_page_size_for_the_key_handler() {
    // 2026-09-26: PgUp/PgDn page by the entry count the last frame held.
    let n = metrale_bench::registry::all().len();
    let a = list_app(0);
    screen(&a, 80, 24);
    let page = a.bench.suite_page.get();
    assert!(page > 0, "a viewport of zero would freeze the page keys");
    assert!(
        page < n,
        "80x24 is clipped, so one page is less than the suite"
    );
    let tall = tall_enough();
    screen(&a, 160, tall);
    assert!(
        a.bench.suite_page.get() >= n,
        "a terminal sized from the registry ({n} entries, {tall} rows) holds \
         the whole suite on one page; got {}",
        a.bench.suite_page.get()
    );
}

#[test]
fn the_list_survives_a_12x4_terminal() {
    // 2026-09-26: Nothing readable fits; the windowing math (`visible`
    // floored at one entry, the offset clamp) must not panic.
    let n = metrale_bench::registry::all().len();
    for selected in [0, n - 1] {
        let rows = screen(&list_app(selected), 12, 4);
        assert_eq!(rows.len(), 4, "selected {selected}");
    }
}
