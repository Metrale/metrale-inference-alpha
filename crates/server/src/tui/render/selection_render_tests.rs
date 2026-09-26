// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for what a mouse drag draws. The highlight is a style
//! change only, so the rendered symbols, which the copy reads, stay the same.
//!
//! Owner: server tui.
//! Invariants: none beyond the types.

use super::*;
// 2026-09-26: The shared render fixtures, from `render_tests.rs`.
use super::tests::{app, render};

/// 2026-09-26: Replaces every `<digits>.<digits>s` with `T.Ts` before two
/// renders are compared. The STARTUP pane's elapsed clock is read from a real
/// `Instant` each frame, so it can move between two renders.
fn without_the_clock(frame: &str) -> String {
    let mut out = String::with_capacity(frame.len());
    let b: Vec<char> = frame.chars().collect();
    let mut i = 0;
    while i < b.len() {
        let start = i;
        let mut j = i;
        while j < b.len() && b[j].is_ascii_digit() {
            j += 1;
        }
        if j > i && j < b.len() && b[j] == '.' {
            let mut k = j + 1;
            while k < b.len() && b[k].is_ascii_digit() {
                k += 1;
            }
            if k > j + 1 && k < b.len() && b[k] == 's' {
                out.push_str("T.Ts");
                i = k + 1;
                continue;
            }
        }
        out.push(b[start]);
        i = start + 1;
    }
    out
}

#[test]
fn a_drag_highlights_what_it_covers_and_a_click_highlights_nothing() {
    use crate::tui::selection::Selection;
    let mut a = app();
    a.section = Section::Main;

    // 2026-09-26: A click (no movement) paints nothing.
    a.selection = Some(Selection::new((10, 5)));
    let clean = render(&a, 120, 40);

    // 2026-09-26: A drag reverses cells: styling changes, symbols do not.
    a.selection = Some(Selection {
        anchor: (10, 5),
        cursor: (30, 5),
    });
    let dragged = render(&a, 120, 40);
    assert_eq!(
        without_the_clock(&clean),
        without_the_clock(&dragged),
        "the highlight is a style change; it must not alter the text"
    );
}

#[test]
fn the_selection_highlight_survives_hostile_geometry() {
    // 2026-09-26: A drag that ends past the frame must not index outside the
    // buffer.
    use crate::tui::selection::Selection;
    let mut a = app();
    a.selection = Some(Selection {
        anchor: (0, 0),
        cursor: (250, 250),
    });
    for (w, h) in [(1u16, 1u16), (40, 12), (200, 50)] {
        let _ = render(&a, w, h);
    }
}

#[test]
fn the_highlight_actually_sets_reverse_video_on_the_covered_cells() {
    // 2026-09-26: The symbol test above passes even if nothing is
    // highlighted; this one checks the reverse video itself.
    use crate::tui::selection::Selection;
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;
    use ratatui::style::Modifier;

    let mut a = app();
    a.section = Section::Main;
    a.selection = Some(Selection {
        anchor: (10, 5),
        cursor: (30, 5),
    });

    let mut t = Terminal::new(TestBackend::new(120, 40)).expect("backend");
    t.draw(|f| draw(f, &a)).expect("draw");
    let buf = t.backend().buffer();

    let rev = |x: u16, y: u16| buf[(x, y)].modifier.contains(Modifier::REVERSED);
    assert!(rev(10, 5), "the first covered cell is highlighted");
    assert!(rev(20, 5), "and the middle");
    assert!(rev(30, 5), "and the last");
    assert!(!rev(9, 5), "but not the cell before it");
    assert!(!rev(31, 5), "nor the cell after");
    assert!(!rev(20, 4), "nor another row");
}

#[test]
fn the_buffer_to_copy_from_is_the_completed_frame_not_the_current_one() {
    // 2026-09-26: ratatui's `Terminal::swap_buffers` resets the buffer that
    // becomes current after each draw, so between frames
    // `current_buffer_mut()` is blank. The text is in the `CompletedFrame`
    // that `draw()` returns, which is why the event loop copies right after a
    // draw (`copy_after_draw`), not in the mouse handler.
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;

    let a = app();
    let mut t = Terminal::new(TestBackend::new(120, 40)).expect("backend");

    let rendered: String = {
        let frame = t.draw(|f| draw(f, &a)).expect("draw");
        frame.buffer.content().iter().map(|c| c.symbol()).collect()
    };
    assert!(
        rendered.contains("Benchmarks"),
        "the completed frame holds the rendered text:\n{rendered}"
    );

    let current: String = t
        .current_buffer_mut()
        .content()
        .iter()
        .map(|c| c.symbol())
        .collect();
    assert_ne!(
        current.trim(),
        rendered.trim(),
        "current_buffer_mut() is NOT the frame just drawn — extracting from it \
         copies a blank screen"
    );
}

#[test]
fn a_selection_does_not_survive_a_keystroke() {
    // 2026-09-26: The selection is stored in screen cells, which mean
    // something else on the next screen, so any key ends it.
    use crate::tui::selection::Selection;
    use crossterm::event::{KeyCode, KeyEvent};

    let mut a = app();
    a.section = Section::Main;
    a.selection = Some(Selection {
        anchor: (10, 5),
        cursor: (30, 5),
    });
    assert!(a.selection.is_some());

    a.on_key(KeyEvent::from(KeyCode::Char('4')));
    assert!(
        a.selection.is_none(),
        "a keystroke must end the selection, not carry it to the next screen"
    );
}

#[test]
fn a_cleared_selection_paints_nothing() {
    use ratatui::Terminal;
    use ratatui::backend::TestBackend;
    use ratatui::style::Modifier;

    let mut a = app();
    a.section = Section::Main;
    a.selection = None;

    let mut t = Terminal::new(TestBackend::new(120, 40)).expect("backend");
    t.draw(|f| draw(f, &a)).expect("draw");
    let buf = t.backend().buffer();
    let reversed = buf
        .content()
        .iter()
        .filter(|c| c.modifier.contains(Modifier::REVERSED))
        .count();
    // 2026-09-26: The dashboard uses REVERSED for its own chips and badges,
    // so compare against this count instead of zero.
    let before = reversed;

    a.selection = Some(crate::tui::selection::Selection {
        anchor: (10, 5),
        cursor: (30, 5),
    });
    let mut t2 = Terminal::new(TestBackend::new(120, 40)).expect("backend");
    t2.draw(|f| draw(f, &a)).expect("draw");
    let after = t2
        .backend()
        .buffer()
        .content()
        .iter()
        .filter(|c| c.modifier.contains(Modifier::REVERSED))
        .count();
    assert_eq!(
        after,
        before + 21,
        "a live selection adds exactly its own cells; a cleared one adds none"
    );
}
