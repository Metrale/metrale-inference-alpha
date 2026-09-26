// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Footer key hints for the three sections whose footer depends on
//! their inner state: Library, Help and Benchmarks.
//!
//! Owner: server tui.
//! Invariants: none beyond the types.

use crate::tui::app::App;

/// 2026-09-26: The Library's footer, by view, open modal and edit mode.
pub(super) fn library_hints(app: &App) -> &'static str {
    use crate::tui::lib_state::View;
    if app.lib.filter_editing {
        return "type to search · ⏎ keep · Esc clear";
    }
    match (app.lib.view, app.lib.editing) {
        (View::Cards, _) => "j/k move · ⏎ configure · d download · u updates · Esc back",
        // 2026-09-26: An open modal owns the keyboard, so the footer names its
        // keys, not the form's. The borrow preview's Enter applies rather than
        // selects, so it has its own line.
        (View::Config, _)
            if matches!(
                app.lib.modal,
                Some(crate::tui::lib_modal::ConfigModal::Preview { .. })
            ) =>
        {
            "j/k scroll · ⏎ apply changes · Esc back to recipes"
        }
        // 2026-09-26: The add-picker's help panel scrolls on `J`/`K`
        // (`lib_modal.rs`).
        (View::Config, _)
            if matches!(
                app.lib.modal,
                Some(crate::tui::lib_modal::ConfigModal::Add { .. })
            ) =>
        {
            "j/k move · J/K scroll help · ⏎ add · Esc cancel"
        }
        (View::Config, _) if app.lib.modal.is_some() => "j/k move · ⏎ select · Esc cancel",
        (View::Config, true) => "⏎ commit · Esc cancel",
        // 2026-09-26: A start is refused without weights
        // (`LibState::selected_has_weights`), so the footer names the way to
        // get them instead of `s`.
        (View::Config, false) if !app.lib.selected_has_weights() => {
            "⚠ weights not downloaded · Esc then d to download · ⏎ edit"
        }
        (View::Config, false) => {
            "j/k move · ⏎ edit · a add · x remove · b borrow · d recipe defaults · s START · Esc back"
        }
        // 2026-09-26: `x stop` only while a download job exists. `u` works in
        // both List and Cards (`lib_keys.rs`), so both footers name it.
        (View::List, _) if app.download.job.is_some() => {
            "j/k move · ⏎ configure · d download · x stop · u updates · / search · r refresh"
        }
        (View::List, _) => {
            "j/k move · ⏎ configure · d download · u updates · / search · r refresh · ? help"
        }
    }
}

/// 2026-09-26: The Help footer: the guide's keys, the editing keys, or the
/// keys of the report's current `ReportPhase`.
pub(super) fn help_hints(app: &App) -> &'static str {
    use crate::tui::help_state::{HelpSub, ReportPhase};
    if app.help.sub == HelpSub::Guide {
        return "⇥ cycle · 7 Report Issue · 1-7 jump · ? help · q quit";
    }
    if app.help.is_editing() {
        return "type · Esc done editing";
    }
    match app.help.phase {
        ReportPhase::Compose => "j/k field · ⏎ edit/toggle · s review & submit · 1-7 jump",
        ReportPhase::Preview => "j/k scroll · y send · a toggle logs · Esc back",
        ReportPhase::RequestingCode => "Esc cancel",
        ReportPhase::WaitingAuth { .. } => "c copy code · Esc cancel",
        ReportPhase::Submitting => "posting…",
        ReportPhase::Done { .. } => "c copy link · Esc compose another",
        ReportPhase::Failed { .. } => "s retry · Esc back",
    }
}

/// 2026-09-26: The Benchmarks footer, by subsection, view and edit mode.
pub(super) fn bench_hints(app: &App) -> &'static str {
    use crate::tui::app::BenchSub;
    use crate::tui::bench_state::View;
    if app.bench_sub == BenchSub::History {
        return "j/k run · PgUp/PgDn table · c card · ⇥ Suite↔History · ? help";
    }
    match (app.bench.view, app.bench.editing) {
        (View::List, _) if app.bench.frame.is_some() => {
            "j/k select · PgUp/PgDn page · ⏎ configure · v last run · ⇥ Suite↔History · ? help"
        }
        (View::List, _) => {
            "j/k select · PgUp/PgDn page · ⏎ configure · ⇥ Suite↔History · 1-7 jump · ? help"
        }
        (View::Variants, _) => "j/k variant · ⏎ choose model · Esc back · ? help",
        (View::Params, true) => "⏎ commit · Esc cancel",
        (View::Params, false) => "j/k move · ⏎ edit · d defaults · p probe · s START · Esc back",
        (View::Run, _) => "c cancel · j/k scroll · Esc back to suite",
    }
}
