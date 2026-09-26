// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for [`super::App`]: navigation order, digit keys, boot section, kernel table, wheel and toasts.
//!
//! Owner: server tui.
//! Invariants: none beyond the types.

use super::*;

/// 2026-09-26: The ⇥ order contains the subsection rows, in the order the sidebar draws them.
#[test]
fn nav_rows_include_subsections_in_sidebar_order() {
    let labels: Vec<String> = App::nav_rows()
        .iter()
        .map(|(s, i)| match s.subs().get(*i) {
            Some(sub) => format!("{}/{}", s.label(), sub),
            None => s.label().to_string(),
        })
        .collect();
    assert_eq!(
        labels,
        [
            "Main/Overview",
            "Main/Kernels",
            "Stats",
            "Network",
            "Library",
            "Benchmarks/Suite",
            "Benchmarks/History",
            "Terminal/Ops",
            "Terminal/Chat",
            "Help/Guide",
            "Help/Report Issue",
        ]
    );
}

/// 2026-09-26: Digit key `n` selects `Section::ALL[n - 1]`.
#[test]
fn digit_keys_match_the_sidebar_order() {
    use clap::Parser;
    use crossterm::event::{KeyCode, KeyEvent};
    let mut app = App::new(crate::cli::ServeArgs::parse_from(["met", "some/model"]));
    for (i, section) in Section::ALL.iter().enumerate() {
        let digit = char::from_digit(i as u32 + 1, 10).expect("<=9 sections");
        app.on_key(KeyEvent::from(KeyCode::Char(digit)));
        assert_eq!(
            app.section,
            *section,
            "key {digit} must select {}",
            section.label()
        );
    }
}

#[test]
fn benchmarks_declares_both_of_its_subsections() {
    assert_eq!(Section::Benchmarks.subs(), &["Suite", "History"]);
    assert_eq!(
        Section::Benchmarks.icon().chars().count(),
        1,
        "sidebar is 1 cell wide"
    );
}

/// 2026-09-26: A section without subsections still contributes a ⇥ stop.
#[test]
fn every_section_is_reachable() {
    let rows = App::nav_rows();
    for s in Section::ALL {
        assert!(
            rows.iter().any(|(r, _)| *r == s),
            "{} unreachable",
            s.label()
        );
    }
}

/// 2026-09-26: Chat scrollback: `None` follows the tip, and scrolling down to or past the bottom
/// restores `None` rather than parking at `Some(0)`.
#[test]
fn chat_scroll_returns_to_follow_at_the_bottom() {
    let mut c = crate::tui::chat::ChatState::default();
    assert_eq!(c.scroll, None, "starts following");
    c.scroll_by(3);
    assert_eq!(c.scroll, Some(3));
    c.scroll_by(-1);
    assert_eq!(c.scroll, Some(2));
    c.scroll_by(-5);
    assert_eq!(c.scroll, None, "overshoot resumes follow");
    c.scroll_by(10);
    c.follow();
    assert_eq!(c.scroll, None);
}

#[test]
fn the_watchdog_command_toggles_the_running_run_not_a_process_global() {
    // 2026-09-26: `/watchdog on|off` sets the levers of the run in `app.run` and no other.
    use clap::Parser as _;
    let mut app = App::new(crate::cli::ServeArgs::parse_from(["met", "some/model"]));
    let levers = std::sync::Arc::new(crate::scheduler::levers::SchedLevers::from_env(None));
    let other = std::sync::Arc::new(crate::scheduler::levers::SchedLevers::from_env(None));
    app.run = Some(crate::tui::RunHandles {
        levers: levers.clone(),
        snapshot: std::sync::Arc::new(metrale_speculative::snapshot::SnapshotCell::default()),
    });

    crate::tui::commands::execute("/watchdog on", &mut app);
    assert!(levers.loop_watchdog(), "the attached run is armed");
    assert!(!other.loop_watchdog(), "another run is untouched");

    crate::tui::commands::execute("/watchdog off", &mut app);
    assert!(!levers.loop_watchdog());
}

#[test]
fn the_watchdog_command_says_so_when_no_run_is_attached() {
    // 2026-09-26: Before a run is attached there is nothing to toggle, and the command says so.
    use clap::Parser as _;
    let mut app = App::new(crate::cli::ServeArgs::parse_from(["met", "some/model"]));
    crate::tui::commands::execute("/watchdog on", &mut app);
    assert!(
        app.ops.output.iter().any(|l| l.contains("no run yet")),
        "got {:?}",
        app.ops.output
    );
}

#[test]
fn a_no_argument_boot_opens_the_library_rather_than_an_empty_main() {
    // 2026-09-26: Without a model the boot lands on the Library, where one can be started.
    use clap::Parser as _;
    let mut args = crate::cli::ServeArgs::parse_from(["met", "m"]);
    args.model = None;
    let app = App::new(args);
    assert!(app.awaiting_model);
    assert_eq!(app.section, Section::Library);
}

#[test]
fn a_boot_with_a_model_still_opens_main() {
    use clap::Parser as _;
    let args = crate::cli::ServeArgs::parse_from(["met", "org/m"]);
    let app = App::new(args);
    assert!(!app.awaiting_model, "a model was named");
    assert_eq!(app.section, Section::Main);
}

#[test]
fn launching_from_the_library_stops_claiming_there_is_no_model() {
    use clap::Parser as _;
    let mut args = crate::cli::ServeArgs::parse_from(["met", "m"]);
    args.model = None;
    let mut app = App::new(args);
    assert!(app.awaiting_model);
    // 2026-09-26: No host is attached, so the launch is refused and `awaiting_model` stays set.
    app.launch_selected_recipe();
    assert!(app.awaiting_model, "a refused launch loaded nothing");
}

#[test]
fn changing_section_asks_for_a_full_repaint() {
    // 2026-09-26: A section change sets `repaint`; ratatui's diff cannot repair cells where its
    // buffer and the terminal have diverged, and a section change redraws the whole content area.
    use clap::Parser as _;
    let mut app = App::new(crate::cli::ServeArgs::parse_from(["met", "m"]));
    app.repaint = false;
    app.jump(Section::Library);
    assert!(app.repaint, "a real change repaints");

    app.repaint = false;
    app.jump(Section::Library);
    assert!(
        !app.repaint,
        "jumping to the section already shown does not"
    );
}

#[test]
fn the_kernel_table_is_not_built_before_a_model_exists() {
    // 2026-09-26: `on_tick` builds the table only when `ready`, not `awaiting_model`, and a live
    // model is known; a no-model boot is `awaiting_model` with no model name.
    use clap::Parser as _;
    let mut args = crate::cli::ServeArgs::parse_from(["met", "m"]);
    args.model = None;
    let mut app = App::new(args);
    app.progress.ready = true;
    app.on_tick();
    assert!(app.kernels.is_none(), "nothing to describe yet");
    assert!(app.kernels_for.is_none());
}

#[test]
fn the_kernel_table_is_built_once_per_model() {
    use clap::Parser as _;
    let mut app = App::new(crate::cli::ServeArgs::parse_from(["met", "org/a"]));
    app.progress.ready = true;
    app.on_tick();
    assert_eq!(
        app.kernels_for.as_deref(),
        Some("org/a"),
        "built for the model"
    );
    let built = app.kernels.is_some();
    assert!(built);

    // 2026-09-26: A tick with the same model does not rebuild.
    app.on_tick();
    assert_eq!(app.kernels_for.as_deref(), Some("org/a"));
}

#[test]
fn the_wheel_scrolls_every_section_that_has_anything_to_scroll() {
    use crate::tui::app::{MainSub, TermSub};
    use crate::tui::section::Section;

    let mut a = App::new(clap::Parser::parse_from(["met", "m"]));

    // 2026-09-26: Main ▸ Overview: the log offset counts back from the newest line, so wheel-up
    // enters history and wheel-down returns to following.
    a.section = Section::Main;
    a.main_sub = MainSub::Overview;
    a.log_scroll_max.set(100);
    a.scroll(-3);
    assert_eq!(a.log_scroll, Some(3), "wheel-up enters history");
    a.scroll(3);
    assert_eq!(a.log_scroll, None, "wheel-down returns to following newest");
    a.scroll(3);
    assert_eq!(a.log_scroll, None, "and cannot scroll past the newest line");

    // 2026-09-26: Main ▸ Kernels: a plain viewport offset, clamped at zero.
    a.main_sub = MainSub::Kernels;
    a.kernel_scroll_max.set(100);
    a.scroll(3);
    assert_eq!(a.kernel_scroll, 3);
    a.scroll(-99);
    assert_eq!(a.kernel_scroll, 0, "clamped, not wrapped into a huge usize");

    a.section = Section::Terminal;
    a.term_sub = TermSub::Chat;
    a.chat_scroll_max.set(100);
    a.scroll(-3);
    assert_eq!(a.chat.scroll, Some(3));

    // 2026-09-26: Stats and Network have nothing to scroll.
    for s in [Section::Stats, Section::Network] {
        a.section = s;
        a.scroll(3);
        a.scroll(-3);
    }
}

#[test]
fn scrolling_stops_at_the_limits_instead_of_running_away() {
    use crate::tui::app::{MainSub, TermSub};
    use crate::tui::section::Section;

    let mut a = App::new(clap::Parser::parse_from(["met", "m"]));
    a.section = Section::Main;
    a.main_sub = MainSub::Overview;
    a.log_scroll_max.set(10);

    for _ in 0..50 {
        a.scroll(-3);
    }
    assert_eq!(a.log_scroll, Some(10), "clamped at the oldest line");

    // 2026-09-26: Coming back takes four wheel-downs, not fifty.
    for _ in 0..4 {
        a.scroll(3);
    }
    assert_eq!(a.log_scroll, None, "back to following the newest line");

    a.log_scroll_max.set(0);
    a.scroll(-9);
    assert_eq!(a.log_scroll, None);

    a.main_sub = MainSub::Kernels;
    a.kernel_scroll_max.set(5);
    for _ in 0..20 {
        a.scroll(3);
    }
    assert_eq!(a.kernel_scroll, 5, "cannot scroll past the last row");
    for _ in 0..20 {
        a.scroll(-3);
    }
    assert_eq!(a.kernel_scroll, 0, "nor above the first");

    a.section = Section::Terminal;
    a.term_sub = TermSub::Chat;
    a.chat_scroll_max.set(4);
    for _ in 0..20 {
        a.scroll(-3);
    }
    assert_eq!(a.chat.scroll, Some(4), "clamped at the oldest message");
}

/// 2026-09-26: Error toasts last 12 s and info toasts 5 s; with the 3-toast render cap, an error
/// that never left would crowd out what follows.
#[test]
fn error_toasts_outlive_info_toasts_but_still_leave() {
    use clap::Parser;
    let mut a = App::new(crate::cli::ServeArgs::parse_from(["met", "some/model"]));
    a.toast("plain note", false);
    a.toast("something failed", true);
    let aged = Instant::now() - std::time::Duration::from_secs(6);
    for t in &mut a.toasts {
        t.at = aged;
    }
    a.on_tick();
    assert_eq!(a.toasts.len(), 1, "at 6s the info toast is gone");
    assert!(a.toasts[0].error);

    a.toasts[0].at = Instant::now() - std::time::Duration::from_secs(13);
    a.on_tick();
    assert!(a.toasts.is_empty(), "at 13s the error leaves too");
}
