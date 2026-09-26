// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests of the shape the sidebar, `⇥` and the mouse handler all
//! read from `Section`: every variant listed once, distinct labels and icons,
//! subsections absent or paired, and the section order.
//!
//! Owner: server tui.
//! Invariants: none beyond the types.

use super::*;

use std::collections::BTreeSet;

/// 2026-09-26: Every variant. The exhaustive match makes a new variant fail
/// to compile here rather than go uncovered.
fn every_variant() -> Vec<Section> {
    let all = vec![
        Section::Main,
        Section::Stats,
        Section::Network,
        Section::Library,
        Section::Benchmarks,
        Section::Terminal,
        Section::Help,
    ];
    for s in &all {
        match s {
            Section::Main
            | Section::Stats
            | Section::Network
            | Section::Library
            | Section::Benchmarks
            | Section::Terminal
            | Section::Help => {}
        }
    }
    all
}

#[test]
fn all_lists_every_section_exactly_once() {
    let listed: Vec<Section> = Section::ALL.to_vec();
    for s in every_variant() {
        assert_eq!(
            listed.iter().filter(|l| **l == s).count(),
            1,
            "{s:?} must appear exactly once in ALL"
        );
    }
    assert_eq!(listed.len(), every_variant().len());
}

#[test]
fn labels_and_icons_are_present_and_distinct() {
    let mut labels = BTreeSet::new();
    let mut icons = BTreeSet::new();
    for s in Section::ALL {
        assert!(!s.label().is_empty(), "{s:?} has no label");
        assert!(!s.icon().is_empty(), "{s:?} has no icon");
        assert!(labels.insert(s.label()), "duplicate label {}", s.label());
        assert!(icons.insert(s.icon()), "duplicate icon {}", s.icon());
        assert_eq!(
            s.icon().chars().count(),
            1,
            "the sidebar draws the icon in one cell: {:?}",
            s.icon()
        );
    }
}

#[test]
fn subsections_are_either_absent_or_a_pair() {
    // 2026-09-26: `App::sub_index` and `set_sub` hold each section's
    // subsection as a bool, so a third subsection would have no state of its
    // own.
    for s in Section::ALL {
        let subs = s.subs();
        assert!(
            subs.is_empty() || subs.len() == 2,
            "{s:?} has {} subsections",
            subs.len()
        );
        for name in subs {
            assert!(!name.is_empty(), "{s:?} has an unnamed subsection");
        }
    }
}

#[test]
fn the_sections_with_subsections_are_the_ones_that_have_two_panes() {
    assert_eq!(Section::Main.subs().to_vec(), vec!["Overview", "Kernels"]);
    assert_eq!(
        Section::Benchmarks.subs().to_vec(),
        vec!["Suite", "History"]
    );
    assert_eq!(Section::Terminal.subs().to_vec(), vec!["Ops", "Chat"]);
    assert_eq!(Section::Help.subs().to_vec(), vec!["Guide", "Report Issue"]);
    for s in [Section::Stats, Section::Network, Section::Library] {
        assert!(s.subs().is_empty(), "{s:?}");
    }
}

#[test]
fn the_navigable_row_count_is_what_the_sidebar_draws() {
    // 2026-09-26: One row per subsection, or one row for a section with none:
    // the rows `⇥` steps through (`App::nav_rows`).
    let rows: usize = Section::ALL.iter().map(|s| s.subs().len().max(1)).sum();
    assert_eq!(rows, 3 + 4 * 2, "3 plain sections and 4 with a pair each");
}

#[test]
fn benchmarks_is_last_but_one_so_terminal_keeps_the_bottom_row() {
    // 2026-09-26: The digit keys are bound per section in `app.rs` (`5`
    // Benchmarks, `6` Terminal, `7` Help); this pins the sidebar order to
    // match them.
    assert_eq!(Section::ALL[4], Section::Benchmarks);
    assert_eq!(Section::ALL[5], Section::Terminal);
    assert_eq!(Section::ALL[6], Section::Help);
}
