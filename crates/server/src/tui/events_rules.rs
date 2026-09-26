// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The event loop's decisions, as pure functions of values the
//! loop holds. `events::run` needs a real terminal, so its rules live here
//! where tests can call them.
//!
//! Owner: server tui.
//! Invariants: none beyond the types.

use std::sync::mpsc::Receiver;

use crossterm::event::KeyEventKind;

use super::section::Section;

/// 2026-09-26: Ticks between metrics samples; with the 100 ms `TICK`, at most
/// one sample a second.
pub const SAMPLE_EVERY: u32 = 10;

/// 2026-09-26: Should this key event reach the reducer? Every kind but
/// `Release`, so `Press` and `Repeat` both do.
pub fn key_is_actionable(kind: KeyEventKind) -> bool {
    kind != KeyEventKind::Release
}

/// 2026-09-26: The newest value on a channel, draining everything older. Used
/// for [`super::RunHandles`], which a hot-swap republishes; an older handle
/// names the levers and snapshot of a model no longer loaded.
pub fn newest<T>(rx: &Receiver<T>) -> Option<T> {
    rx.try_iter().last()
}

/// 2026-09-26: Is this the tick that samples metrics? Counted in ticks, not
/// wall time: a tick fires once at least `TICK` has passed, so a slow loop
/// samples less often rather than in a burst.
///
/// The caller's counter wraps. `u32::MAX` is not a multiple of `SAMPLE_EVERY`,
/// so a saturating counter would stop sampling there; wrapping costs one
/// six-tick interval every 2^32 ticks (about 13.6 years at 100 ms).
pub fn samples_metrics(ticks: u32) -> bool {
    ticks.is_multiple_of(SAMPLE_EVERY)
}

/// 2026-09-26: The Library state that [`tick_work`] reads, passed as values.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LibraryPhase {
    /// 2026-09-26: `App::library_dirty`: a rescan of the local cache is
    /// wanted.
    pub dirty: bool,
    /// 2026-09-26: A background scan is running; its result describes the
    /// cache as it was when it started.
    pub scan_in_flight: bool,
    /// 2026-09-26: `LibState::attached`: the recipe store root is set.
    pub recipes_attached: bool,
    /// 2026-09-26: `ArtifactStore::discover` failed once. It fails only on
    /// environment the process does not change (`METRALE_HOME` set but empty,
    /// or neither it nor a non-empty `HOME`).
    pub recipes_unavailable: bool,
}

/// 2026-09-26: What a tick owes the sections.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TickWork {
    /// 2026-09-26: Start a background scan of the local cache and clear
    /// `App::library_dirty`.
    pub start_scan: bool,
    /// 2026-09-26: Call `App::attach_recipes`: attach the recipe store and
    /// start the recipe refresh, or mark the recipes unavailable.
    pub attach_recipes: bool,
    /// 2026-09-26: Drain the Library's pollers (scan, index, recipe date).
    pub poll_library: bool,
    /// 2026-09-26: Call `BenchState::load_history`.
    pub load_history: bool,
}

/// 2026-09-26: The lazy-work rules for one tick.
pub fn tick_work(section: Section, lib: LibraryPhase) -> TickWork {
    let in_library = section == Section::Library;
    TickWork {
        // 2026-09-26: Not while a scan is in flight: `LibState::start_scan`
        // ignores that call, and the dirty flag must stay set until a scan
        // that started after the change runs.
        start_scan: in_library && lib.dirty && !lib.scan_in_flight,
        // 2026-09-26: Not after a failed attach (`recipes_unavailable`): each
        // failure logs a warning and rebuilds the catalogue.
        attach_recipes: in_library && !lib.recipes_attached && !lib.recipes_unavailable,
        poll_library: in_library,
        load_history: section == Section::Benchmarks,
    }
}

/// 2026-09-26: Why the event loop stopped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Exit {
    /// 2026-09-26: The user asked to stop the server: `q`, Ctrl+C, `/quit`.
    Quit,
    /// 2026-09-26: The dashboard goes away and the server keeps serving:
    /// `/detach`, or a failed draw.
    Detach,
    /// 2026-09-26: Shutdown was already requested elsewhere through
    /// `shutdown::request`: a signal listener, `tui::stop_and_join`, a lost
    /// CUDA context, or the end of a benchmark gate run.
    ShuttingDown,
}

/// 2026-09-26: Why the loop should stop, or `None` to keep going.
/// `should_quit` outranks `shutdown_requested` because `/quit` and Ctrl+C set
/// both.
pub fn exit_kind(should_quit: bool, detach: bool, shutdown_requested: bool) -> Option<Exit> {
    if should_quit {
        return Some(Exit::Quit);
    }
    if detach {
        return Some(Exit::Detach);
    }
    shutdown_requested.then_some(Exit::ShuttingDown)
}

#[cfg(test)]
#[path = "events_rules_tests.rs"]
mod tests;
