// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for `on_mouse`: sidebar clicks, drag selection, the wheel and the Library search field.
//!
//! Owner: server tui.
//! Invariants: none beyond the types.

use super::*;
use crossterm::event::{KeyModifiers, MouseEvent};
use ratatui::layout::Size;

/// 2026-09-26: Big enough for `render::Chrome`'s 18-column sidebar and 3-row header.
const WIDE: Size = Size {
    width: 120,
    height: 40,
};
/// 2026-09-26: Small enough for the 4-column icon sidebar and the 1-row header.
const NARROW: Size = Size {
    width: 80,
    height: 20,
};

fn app() -> App {
    App::new(clap::Parser::parse_from(["met", "org/m"]))
}

fn mouse(kind: MouseEventKind, column: u16, row: u16) -> MouseEvent {
    MouseEvent {
        kind,
        column,
        row,
        modifiers: KeyModifiers::NONE,
    }
}

fn click(a: &mut App, column: u16, row: u16, size: Size) -> MouseOutcome {
    on_mouse(
        a,
        mouse(MouseEventKind::Down(MouseButton::Left), column, row),
        Some(size),
    )
}

/// 2026-09-26: The selected section, and its subsection when it has any, as `Section/Sub`.
fn at(a: &App) -> String {
    match a.section.subs().get(a.sub_index(a.section)) {
        Some(sub) => format!("{}/{sub}", a.section.label()),
        None => a.section.label().to_string(),
    }
}

#[test]
fn clicking_a_sidebar_section_row_selects_that_section() {
    // 2026-09-26: From row 3, with Main expanded: Main, Overview, Kernels, Stats, Network, Library,
    // Benchmarks, Terminal, Help.
    for (row, expected) in [
        (6, "Stats"),
        (7, "Network"),
        (8, "Library"),
        (9, "Benchmarks/Suite"),
        (10, "Terminal/Ops"),
    ] {
        let mut a = app();
        click(&mut a, 2, row, WIDE);
        assert_eq!(at(&a), expected, "row {row}");
    }
}

#[test]
fn clicking_the_section_already_shown_cycles_its_subsections() {
    let mut a = app();
    click(&mut a, 2, 3, WIDE);
    assert_eq!(at(&a), "Main/Kernels");
    click(&mut a, 2, 3, WIDE);
    assert_eq!(at(&a), "Main/Overview");
}

#[test]
fn clicking_a_subsection_row_selects_that_subsection() {
    let mut a = app();
    click(&mut a, 2, 5, WIDE);
    assert_eq!(at(&a), "Main/Kernels", "└ Kernels is the third row");
    click(&mut a, 2, 4, WIDE);
    assert_eq!(at(&a), "Main/Overview", "and ├ Overview the second");
}

#[test]
fn the_rows_below_shift_with_whichever_section_is_expanded() {
    let mut a = app();
    click(&mut a, 2, 10, WIDE);
    assert_eq!(at(&a), "Terminal/Ops");
    // 2026-09-26: With Terminal expanded, its Ops and Chat rows follow it, at rows 9 and 10.
    click(&mut a, 2, 10, WIDE);
    assert_eq!(at(&a), "Terminal/Chat", "└ Chat");
    click(&mut a, 2, 9, WIDE);
    assert_eq!(at(&a), "Terminal/Ops", "├ Ops");
    click(&mut a, 2, 6, WIDE);
    assert_eq!(at(&a), "Library");
}

#[test]
fn a_narrow_sidebar_draws_no_subsections_and_offsets_nothing() {
    // 2026-09-26: Icons only under a one-row header, so row N + 1 is section N.
    for (row, expected) in [
        (2, "Stats"),
        (3, "Network"),
        (4, "Library"),
        (5, "Benchmarks/Suite"),
        (6, "Terminal/Ops"),
    ] {
        let mut a = app();
        click(&mut a, 1, row, NARROW);
        assert_eq!(at(&a), expected, "row {row}");
    }
}

#[test]
fn a_click_past_the_last_sidebar_row_selects_nothing() {
    let mut a = app();
    click(&mut a, 2, 30, WIDE);
    assert_eq!(at(&a), "Main/Overview", "unchanged");
}

#[test]
fn a_click_on_the_header_navigates_nowhere() {
    let mut a = app();
    click(&mut a, 2, 1, WIDE);
    assert_eq!(at(&a), "Main/Overview");
    assert_eq!(a.selection.expect("armed").anchor, (2, 1));
}

#[test]
fn a_sidebar_click_is_navigation_not_the_start_of_a_drag() {
    let mut a = app();
    a.selection = Some(crate::tui::selection::Selection::new((60, 9)));
    click(&mut a, 2, 8, WIDE);
    assert!(a.selection.is_none());
}

#[test]
fn a_click_in_the_content_area_arms_a_selection() {
    let mut a = app();
    click(&mut a, 40, 9, WIDE);
    let sel = a.selection.expect("armed");
    assert_eq!(sel.anchor, (40, 9));
    assert!(!sel.is_drag(), "nothing is copied until it moves");
}

#[test]
fn a_drag_tracks_the_pointer_and_copies_on_release() {
    let mut a = app();
    click(&mut a, 40, 9, WIDE);
    let out = on_mouse(
        &mut a,
        mouse(MouseEventKind::Drag(MouseButton::Left), 52, 11),
        Some(WIDE),
    );
    assert_eq!(out, MouseOutcome::None, "nothing to copy mid-drag");
    assert_eq!(a.selection.expect("still armed").cursor, (52, 11));

    let out = on_mouse(
        &mut a,
        mouse(MouseEventKind::Up(MouseButton::Left), 52, 11),
        Some(WIDE),
    );
    assert_eq!(out, MouseOutcome::CopySelection);
    assert!(
        a.selection.is_some(),
        "the selection outlives the release: the text is read out of the frame"
    );
}

#[test]
fn a_click_that_never_moved_copies_nothing() {
    let mut a = app();
    click(&mut a, 40, 9, WIDE);
    let out = on_mouse(
        &mut a,
        mouse(MouseEventKind::Up(MouseButton::Left), 40, 9),
        Some(WIDE),
    );
    assert_eq!(out, MouseOutcome::None);
    assert!(a.selection.is_none(), "and clears rather than lingering");
}

#[test]
fn a_release_with_no_button_down_is_harmless() {
    let mut a = app();
    let out = on_mouse(
        &mut a,
        mouse(MouseEventKind::Up(MouseButton::Left), 40, 9),
        Some(WIDE),
    );
    assert_eq!(out, MouseOutcome::None);
    assert!(a.selection.is_none());
}

#[test]
fn a_drag_with_nothing_armed_does_not_invent_a_selection() {
    let mut a = app();
    on_mouse(
        &mut a,
        mouse(MouseEventKind::Drag(MouseButton::Left), 40, 9),
        Some(WIDE),
    );
    assert!(a.selection.is_none());
}

#[test]
fn the_wheel_scrolls_the_active_pane_and_drops_the_highlight() {
    let mut a = app();
    a.log_scroll_max.set(50);
    a.selection = Some(crate::tui::selection::Selection::new((40, 9)));
    on_mouse(&mut a, mouse(MouseEventKind::ScrollUp, 40, 9), Some(WIDE));
    assert_eq!(a.log_scroll, Some(3), "three rows a notch");
    assert!(a.selection.is_none());
    on_mouse(&mut a, mouse(MouseEventKind::ScrollDown, 40, 9), Some(WIDE));
    assert_eq!(a.log_scroll, None);
}

#[test]
fn a_mouse_event_before_the_terminal_size_is_known_is_ignored() {
    let mut a = app();
    let out = on_mouse(
        &mut a,
        mouse(MouseEventKind::Down(MouseButton::Left), 2, 8),
        None,
    );
    assert_eq!(out, MouseOutcome::None);
    assert_eq!(at(&a), "Main/Overview");
    assert!(a.selection.is_none());
}

#[test]
fn buttons_other_than_the_left_one_are_left_to_the_terminal() {
    let mut a = app();
    for kind in [
        MouseEventKind::Down(MouseButton::Right),
        MouseEventKind::Down(MouseButton::Middle),
        MouseEventKind::Up(MouseButton::Right),
        MouseEventKind::Moved,
    ] {
        let out = on_mouse(&mut a, mouse(kind, 40, 9), Some(WIDE));
        assert_eq!(out, MouseOutcome::None, "{kind:?}");
        assert!(a.selection.is_none(), "{kind:?}");
    }
}

#[test]
fn clicking_the_library_search_field_focuses_it_and_a_miss_does_not() {
    let mut a = app();
    a.section = Section::Library;
    let mut terminal =
        ratatui::Terminal::new(ratatui::backend::TestBackend::new(WIDE.width, WIDE.height))
            .expect("backend");
    terminal
        .draw(|f| crate::tui::render::draw(f, &a))
        .expect("draw");
    let rect = a.lib_search_click.get().expect("the field was drawn");

    assert!(!a.lib.filter_editing);
    click(&mut a, rect.x + 2, rect.y, WIDE);
    assert!(a.lib.filter_editing, "the click focuses the field");
    assert!(
        a.selection.is_none(),
        "focusing a field is not the start of a drag"
    );

    let mut b = app();
    b.section = Section::Library;
    terminal
        .draw(|f| crate::tui::render::draw(f, &b))
        .expect("draw");
    let rect = b.lib_search_click.get().expect("drawn");
    click(&mut b, rect.x + 2, rect.y + 1, WIDE);
    assert!(!b.lib.filter_editing, "a miss must not focus the field");
}

#[test]
fn the_search_rect_is_not_published_outside_the_library_list() {
    // 2026-09-26: `render::draw` clears the cell every frame; only the Library list sets it.
    let a = app();
    debug_assert!(a.section != Section::Library);
    let mut terminal =
        ratatui::Terminal::new(ratatui::backend::TestBackend::new(WIDE.width, WIDE.height))
            .expect("backend");
    terminal
        .draw(|f| crate::tui::render::draw(f, &a))
        .expect("draw");
    assert!(a.lib_search_click.get().is_none());
}
