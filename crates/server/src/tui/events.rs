// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The TUI event loop, run on the "metrale-tui" thread. Each
//! iteration waits up to 50 ms for a crossterm event, drains the data
//! channels, runs the tick work when `TICK` (100 ms) has passed, draws a
//! frame, and checks the exit conditions.
//!
//! Owner: server tui.
//! Invariants:
//! - `run` releases `TUI_ACTIVE` on every return: the `ActiveClaim` is taken
//!   first, and on the normal exit it is dropped after the terminal guard.

use std::sync::atomic::Ordering;
use std::sync::mpsc::Receiver;
use std::time::{Duration, Instant};

use crossterm::event::{Event, MouseButton, MouseEventKind};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;

use super::app::{App, Section};
use super::capture_layer::ProgressEvent;
use super::events_rules::{Exit, LibraryPhase, exit_kind, key_is_actionable, newest, tick_work};
use super::init::{ActiveClaim, TUI_ACTIVE};
use super::terminal_guard::TerminalGuard;
use super::{events_rules, render, shutdown};

const TICK: Duration = Duration::from_millis(100);

pub fn run(
    mut app: App,
    progress_rx: Receiver<ProgressEvent>,
    levers_rx: Receiver<crate::tui::RunHandles>,
) {
    // 2026-09-26: Taken first, so its drop releases `TUI_ACTIVE` on every
    // return below. `tui::start` sets the flag before spawning this thread.
    let claim = ActiveClaim::claim();
    super::terminal_guard::install_panic_hook();
    let guard = match TerminalGuard::enter() {
        Ok(g) => g,
        Err(e) => {
            tracing::warn!("TUI unavailable ({e}); continuing with plain logs");
            return;
        }
    };
    debug_assert!(
        TUI_ACTIVE.load(Ordering::SeqCst),
        "start() claims the terminal"
    );
    let mut terminal = match Terminal::new(CrosstermBackend::new(std::io::stdout())) {
        Ok(t) => t,
        Err(e) => {
            drop(guard);
            tracing::warn!("TUI terminal init failed ({e}); plain logs");
            return;
        }
    };

    let mut last_tick = Instant::now();
    // 2026-09-26: Set when a drag ends; consumed after the next draw, whose
    // buffer holds the selected text.
    let mut copy_after_draw = false;
    let mut copy_result: Option<(Result<usize, String>, bool)> = None;
    let mut clear_selection = false;
    let mut ticks: u32 = 0;

    // 2026-09-26: The loop `break`s with its `Exit`, which the code after the
    // loop acts on.
    let exit = loop {
        if crossterm::event::poll(Duration::from_millis(50)).unwrap_or(false) {
            match crossterm::event::read() {
                Ok(Event::Key(k)) if key_is_actionable(k.kind) => app.on_key(k),
                Ok(Event::Mouse(m)) => {
                    let size = terminal.size().ok();
                    // 2026-09-26: The copy waits for the next draw. ratatui
                    // 0.29 resets the buffer it swaps in after each draw, so
                    // the text is only in the `CompletedFrame` that
                    // `terminal.draw` returns.
                    if on_mouse(&mut app, m, size) == MouseOutcome::CopySelection {
                        copy_after_draw = true;
                    }
                }
                Ok(Event::Resize(..)) => {
                    let _ = terminal.clear();
                }
                // 2026-09-26: `TerminalGuard::enter` enables bracketed paste,
                // so a paste arrives as one event.
                Ok(Event::Paste(text)) => app.on_paste(text),
                _ => {}
            }
        }
        // 2026-09-26: The newest published run wins; a hot-swap republishes it.
        if let Some(h) = newest(&levers_rx) {
            app.run = Some(h);
        }
        for ev in progress_rx.try_iter() {
            app.progress.apply(ev);
        }
        app.chat.pump();
        app.bench.pump();
        app.help.pump();
        if let Some((text, error)) = app.help.take_message() {
            app.toast(text, error);
        }
        // 2026-09-26: Downloads are pumped whatever section is shown, so one
        // started in the Library still settles and reports elsewhere.
        if let Some(settled) = app.download.pump() {
            app.library_dirty = true;
            if let crate::tui::download_state::Settled::Finished(_) = settled {
                app.repaint = true;
            }
        }
        if let Some((text, error)) = app.download.last_message.take() {
            app.toast(text, error);
        }
        // 2026-09-26: A pending download starts once no job is running, also
        // when the running job finished while the switch prompt was open; the
        // prompt is then dropped.
        app.start_pending_download();
        if app.download_switch.is_some() && app.download.job.is_none() {
            app.download_switch = None;
        }
        if last_tick.elapsed() >= TICK {
            last_tick = Instant::now();
            ticks = ticks.wrapping_add(1);
            app.on_tick();
            if events_rules::samples_metrics(ticks) {
                app.stats.sample(app.run.as_ref());
            }
            // 2026-09-26: The Library reducer cannot reach `App`, so it raises a
            // flag the tick moves into `App::library_dirty`.
            if std::mem::take(&mut app.lib.mark_dirty) {
                app.library_dirty = true;
            }
            // 2026-09-26: The tick's lazy work, decided by
            // `events_rules::tick_work`.
            let work = tick_work(
                app.section,
                LibraryPhase {
                    dirty: app.library_dirty,
                    scan_in_flight: app.lib.scan_in_flight(),
                    recipes_attached: app.lib.attached(),
                    recipes_unavailable: app.lib.recipes_unavailable(),
                },
            );
            // 2026-09-26: The scan runs on its own thread and `poll_scan` below
            // only `try_recv`s. The dirty flag is cleared only when a scan
            // starts.
            if work.start_scan {
                app.library_dirty = false;
                app.lib.start_scan(app.args.cache_dir.as_deref());
            }
            if work.attach_recipes {
                app.attach_recipes();
            }
            if work.poll_library {
                if let Some(found) = app.lib.poll_scan() {
                    app.library = found;
                    app.lib.rebuild(&app.library);
                }
                app.lib.poll(&app.library);
                app.lib.poll_date();
                // 2026-09-26: Ask for the visible recipe's commit date.
                // `want_date_for` returns at once while a date fetch is
                // pending, for an id already dated, and for a recipe with its
                // own `updated`.
                if let Some(id) = app.lib.visible_recipe_id() {
                    app.lib.want_date_for(&id);
                }
            }
            // 2026-09-26: `load_history` returns at once until
            // `history_loaded` is cleared.
            if work.load_history {
                app.bench.load_history();
            }
            // 2026-09-26: Whatever section is shown, so a pre-flight started in
            // Benchmarks still resolves elsewhere.
            app.bench.poll_preflight();
        }
        if std::mem::take(&mut app.repaint) {
            let _ = terminal.clear();
        }
        match terminal.draw(|f| render::draw(f, &app)) {
            Err(e) => {
                tracing::warn!("TUI draw error: {e}; detaching");
                // 2026-09-26: A failed draw leaves the loop as `/detach` does;
                // the server keeps serving.
                break Exit::Detach;
            }
            Ok(frame) => {
                // 2026-09-26: `CompletedFrame` borrows the buffer just rendered,
                // which holds the on-screen text.
                if std::mem::take(&mut copy_after_draw)
                    && let Some(sel) = app.selection
                {
                    let text = super::selection::extract(frame.buffer, frame.area, &sel);
                    copy_result = Some((super::clipboard::copy(&text), text.is_empty()));
                    // 2026-09-26: The selection is in screen cells, so it is
                    // cleared once read.
                    clear_selection = true;
                }
            }
        }
        // 2026-09-26: Both of these need `&mut app`, which the frame borrow
        // above forbids.
        if std::mem::take(&mut clear_selection) {
            app.selection = None;
        }
        if let Some((res, was_empty)) = copy_result.take() {
            match res {
                // 2026-09-26: "sent", not "copied": `Ok` means the sequence
                // was written, not that the terminal applied it.
                Ok(n) => app.toast(
                    format!("sent {n} characters to the terminal clipboard (OSC 52)"),
                    false,
                ),
                // 2026-09-26: An empty selection is logged at debug level, not
                // toasted.
                Err(e) if was_empty => tracing::debug!("copy skipped: {e}"),
                Err(e) => app.toast(e, true),
            }
        }
        if let Some(e) = exit_kind(app.should_quit, app.detach, shutdown::requested()) {
            break e;
        }
    };
    // 2026-09-26: Stop the recipe refresh from starting further files; nothing
    // will render its answer.
    app.lib.cancel_refresh();
    // 2026-09-26: Restore the terminal, then release the claim, so no log line
    // is written into the alternate screen. Both drops come before the log
    // lines below, which need stdout.
    drop(guard);
    drop(claim);
    let log = super::init::tee_file_path().unwrap_or("-");
    match exit {
        Exit::Quit => shutdown::request("TUI quit"),
        Exit::Detach => {
            tracing::info!("TUI detached — plain logs resume (full history: {log})")
        }
        // 2026-09-26: Not "detached": shutdown was already requested elsewhere
        // (a signal, `tui::stop_and_join`), so the server is draining.
        Exit::ShuttingDown => {
            tracing::info!("TUI closed — shutdown already under way (full history: {log})")
        }
    }
}

/// 2026-09-26: What the caller must do after a mouse event it cannot do
/// itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MouseOutcome {
    None,
    /// 2026-09-26: A drag finished: read the selection out of the rendered
    /// frame and copy.
    CopySelection,
}

/// 2026-09-26: What a clicked sidebar row draws.
enum SidebarRow {
    /// 2026-09-26: Index into [`Section::ALL`].
    Section(usize),
    /// 2026-09-26: Subsection of the active section, the only one drawing any.
    Sub(usize),
}

/// 2026-09-26: Map a visual sidebar row back to what is drawn on it. The
/// active section's `subs` rows sit directly under it and shift every later
/// section down by `subs`.
fn sidebar_row(active_idx: usize, subs: usize, visual: usize) -> SidebarRow {
    if visual <= active_idx {
        SidebarRow::Section(visual)
    } else if visual <= active_idx + subs {
        SidebarRow::Sub(visual - active_idx - 1)
    } else {
        SidebarRow::Section(visual - subs)
    }
}

fn on_mouse(
    app: &mut App,
    m: crossterm::event::MouseEvent,
    size: Option<ratatui::layout::Size>,
) -> MouseOutcome {
    let Some(size) = size else {
        return MouseOutcome::None;
    };
    // 2026-09-26: The renderer's own breakpoints (`render::Chrome`).
    let chrome = render::Chrome::of(size);
    match m.kind {
        MouseEventKind::Down(MouseButton::Left) => {
            if m.column < chrome.sidebar_w && m.row >= chrome.header_h {
                let visual = (m.row - chrome.header_h) as usize;
                let active_idx = Section::ALL
                    .iter()
                    .position(|s| *s == app.section)
                    .unwrap_or(0);
                // 2026-09-26: The renderer draws `Section::subs` under the
                // active section only in the full-width sidebar.
                let subs = if chrome.full_sidebar() {
                    app.section.subs().len()
                } else {
                    0
                };
                match sidebar_row(active_idx, subs, visual) {
                    SidebarRow::Section(i) => app.sidebar_click(i),
                    SidebarRow::Sub(i) => app.sidebar_sub_click(i),
                }
                // 2026-09-26: A sidebar click does not start a selection.
                app.selection = None;
            } else if app
                .lib_search_click
                .get()
                .is_some_and(|r| r.contains(ratatui::layout::Position::new(m.column, m.row)))
            {
                // 2026-09-26: The Library search field, at the rect the
                // renderer drew last frame (`App::lib_search_click`).
                app.lib.filter_editing = true;
                app.selection = None;
            } else {
                // 2026-09-26: Anywhere else a press starts a selection; nothing
                // is copied unless the pointer moves before release.
                app.selection = Some(super::selection::Selection::new((m.column, m.row)));
            }
        }
        MouseEventKind::Drag(MouseButton::Left) => {
            if let Some(sel) = app.selection.as_mut() {
                sel.cursor = (m.column, m.row);
            }
        }
        MouseEventKind::Up(MouseButton::Left) => {
            // 2026-09-26: Copy on release only after a drag; a plain click
            // clears the selection.
            return match app.selection {
                Some(sel) if sel.is_drag() => MouseOutcome::CopySelection,
                _ => {
                    app.selection = None;
                    MouseOutcome::None
                }
            };
        }
        // 2026-09-26: Scrolling moves the content under the selection, so it
        // clears the selection.
        MouseEventKind::ScrollUp => {
            app.selection = None;
            app.scroll(-3);
        }
        MouseEventKind::ScrollDown => {
            app.selection = None;
            app.scroll(3);
        }
        _ => {}
    }
    MouseOutcome::None
}

#[cfg(test)]
#[path = "events_tests.rs"]
mod tests;
