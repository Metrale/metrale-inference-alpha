// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for what each key does in the Benchmarks views. No executor is attached, so
//! every start is refused and the refusal is what the tests observe.
//!
//! Owner: server tui.
//! Invariants: none beyond the types.

use crossterm::event::{KeyCode, KeyEvent};

use super::*;
use crate::tui::bench_preflight::{Phase, Preflight};

fn key(code: KeyCode) -> KeyEvent {
    KeyEvent::from(code)
}

/// 2026-09-26: The first benchmark's form, pointed at an endpoint nothing is listening on.
///
/// The target is set directly rather than through `attach`, which also
/// installs an executor that a stray `s` would use to start a run.
fn state() -> BenchState {
    let mut s = BenchState::default();
    s.select(0);
    s.target = metrale_bench::TargetEndpoint::local(8888, "test-model");
    // 2026-09-26: Reload so the two target rows carry the endpoint set above.
    s.select(0);
    s
}

fn params() -> BenchState {
    let mut s = state();
    s.view = View::Params;
    s
}

fn toast(outcome: Outcome) -> String {
    match outcome {
        Outcome::Toast { text, error } => {
            assert!(error, "a refusal is an error toast: {text}");
            text
        }
        Outcome::None => panic!("expected a toast"),
    }
}

fn is_silent(outcome: Outcome) -> bool {
    matches!(outcome, Outcome::None)
}

#[test]
fn the_arrow_keys_and_the_vim_keys_are_the_same_keys() {
    let mut arrows = state();
    let mut vim = state();
    arrows.on_key(key(KeyCode::Down), BenchSub::Suite);
    vim.on_key(key(KeyCode::Char('j')), BenchSub::Suite);
    assert_eq!(arrows.selected, vim.selected);
    assert_eq!(arrows.selected, 1);
    arrows.on_key(key(KeyCode::Up), BenchSub::Suite);
    vim.on_key(key(KeyCode::Char('k')), BenchSub::Suite);
    assert_eq!(arrows.selected, vim.selected);
    assert_eq!(arrows.selected, 0);
}

#[test]
fn an_unbound_key_in_the_list_changes_nothing() {
    let mut s = state();
    assert!(is_silent(
        s.on_key(key(KeyCode::Char('z')), BenchSub::Suite)
    ));
    assert_eq!(s.selected, 0);
    assert_eq!(s.view, View::List);
}

#[test]
fn the_run_pane_is_only_reachable_from_the_list_once_there_is_a_run_to_see() {
    let mut s = state();
    s.on_key(key(KeyCode::Char('v')), BenchSub::Suite);
    assert_eq!(s.view, View::List, "nothing has run yet");
}

#[test]
fn a_finished_run_stays_reachable_after_navigating_away() {
    let mut s = state();
    s.frame = Some(metrale_bench::BenchmarkResult {
        status: metrale_bench::RunStatus::Completed,
        phase: "done".into(),
        progress: None,
        summary: Vec::new(),
        table: None,
        verdict: None,
        metrics: std::collections::BTreeMap::new(),
        log: Vec::new(),
        dataset_fingerprint: None,
        elapsed: std::time::Duration::from_secs(1),
        hardware_state: None,
    });
    s.on_key(key(KeyCode::Char('v')), BenchSub::Suite);
    assert_eq!(s.view, View::Run);
}

#[test]
fn opening_a_benchmark_that_is_not_running_opens_its_form() {
    // 2026-09-26: Nothing is running, so Enter opens the form.
    let mut s = state();
    s.on_key(key(KeyCode::Right), BenchSub::Suite);
    assert_eq!(s.view, View::Params);
}

#[test]
fn row_navigation_in_the_form_is_clamped_to_the_fields_that_exist() {
    let mut s = params();
    for _ in 0..s.row_count() + 5 {
        s.on_key(key(KeyCode::Char('j')), BenchSub::Suite);
    }
    assert_eq!(s.row, s.row_count() - 1, "the last row is the Model field");
    for _ in 0..s.row_count() + 5 {
        s.on_key(key(KeyCode::Up), BenchSub::Suite);
    }
    assert_eq!(s.row, 0);
}

#[test]
fn typing_edits_only_the_focused_row() {
    let mut s = params();
    let untouched = s.edit[1].clone();
    s.on_key(key(KeyCode::Enter), BenchSub::Suite);
    for c in "12".chars() {
        s.on_key(key(KeyCode::Char(c)), BenchSub::Suite);
    }
    s.on_key(key(KeyCode::Backspace), BenchSub::Suite);
    assert!(s.edit[0].ends_with('1'), "{}", s.edit[0]);
    assert_eq!(s.edit[1], untouched);
    s.on_key(key(KeyCode::Enter), BenchSub::Suite);
    assert!(!s.is_editing(), "Enter commits and hands the keyboard back");
}

#[test]
fn backspace_on_an_empty_buffer_is_not_an_underflow() {
    let mut s = params();
    s.on_key(key(KeyCode::Enter), BenchSub::Suite);
    for _ in 0..s.edit[0].len() + 5 {
        s.on_key(key(KeyCode::Backspace), BenchSub::Suite);
    }
    assert_eq!(s.edit[0], "");
    s.on_key(key(KeyCode::Enter), BenchSub::Suite);
    assert!(
        s.row_error(0).is_some(),
        "an empty value is refused, not kept"
    );
}

#[test]
fn a_cancelled_edit_of_the_endpoint_restores_the_committed_url() {
    // 2026-09-26: Esc restores the committed value, not the schema default.
    let mut s = params();
    s.row = s.specs.len();
    s.on_key(key(KeyCode::Enter), BenchSub::Suite);
    s.on_key(key(KeyCode::Char('x')), BenchSub::Suite);
    s.on_key(key(KeyCode::Esc), BenchSub::Suite);
    assert_eq!(s.edit[s.row], s.target.base_url);
    assert!(!s.is_editing());
}

#[test]
fn a_cancelled_edit_of_the_model_restores_the_committed_model() {
    let mut s = params();
    s.row = s.specs.len() + 1;
    s.on_key(key(KeyCode::Enter), BenchSub::Suite);
    s.on_key(key(KeyCode::Backspace), BenchSub::Suite);
    s.on_key(key(KeyCode::Esc), BenchSub::Suite);
    assert_eq!(s.edit[s.row], "test-model");
}

#[test]
fn the_coherence_probe_toggles_both_ways() {
    let mut s = params();
    assert_eq!(s.coherence, metrale_bench::CoherencePolicy::Probe);
    s.on_key(key(KeyCode::Char('p')), BenchSub::Suite);
    assert_eq!(s.coherence, metrale_bench::CoherencePolicy::Skip);
    s.on_key(key(KeyCode::Char('p')), BenchSub::Suite);
    assert_eq!(s.coherence, metrale_bench::CoherencePolicy::Probe);
}

#[test]
fn leaving_the_form_keeps_what_was_typed_into_it() {
    let mut s = params();
    s.edit[0] = "7".into();
    s.commit_row(0);
    let committed = s.values.clone();
    s.on_key(key(KeyCode::Esc), BenchSub::Suite);
    assert_eq!(s.view, View::List);
    s.on_key(key(KeyCode::Enter), BenchSub::Suite);
    assert_eq!(s.view, View::Params);
    assert_eq!(s.values, committed, "stepping back is not a reset");
}

#[test]
fn re_selecting_the_same_benchmark_from_the_list_does_reset_the_form() {
    let mut s = params();
    s.edit[0] = "7".into();
    s.commit_row(0);
    s.on_key(key(KeyCode::Char('d')), BenchSub::Suite);
    assert_eq!(s.edit[0], s.specs[0].default.to_edit_string());
}

#[test]
fn starting_without_an_executor_toasts_the_reason_rather_than_switching_view() {
    let mut s = params();
    let text = toast(s.on_key(key(KeyCode::Char('s')), BenchSub::Suite));
    assert!(text.contains("executor"), "{text}");
    assert_eq!(s.view, View::Params, "a refused start stays on the form");
}

#[test]
fn a_start_refused_for_an_invalid_field_says_how_many_need_fixing() {
    let mut s = params();
    s.edit[0] = "nonsense".into();
    s.commit_row(0);
    let text = toast(s.on_key(key(KeyCode::Char('s')), BenchSub::Suite));
    assert!(text.contains("1 field(s) need fixing"), "{text}");
}

#[test]
fn the_consent_gate_only_opens_for_the_benchmark_that_runs_model_authored_shell() {
    let mut s = params();
    let index = metrale_bench::registry::all()
        .iter()
        .position(|d| d.needs_confirmation)
        .expect("the agentic benchmark requires confirmation");
    s.select(index);
    s.view = View::Params;
    assert!(is_silent(
        s.on_key(key(KeyCode::Char('s')), BenchSub::Suite)
    ));
    assert!(s.confirm_open);
    // 2026-09-26: `y` proceeds, and the start is then refused for lack of an executor.
    let text = toast(s.on_key(key(KeyCode::Char('y')), BenchSub::Suite));
    assert!(text.contains("executor"), "{text}");
    assert!(!s.confirm_open, "the gate closes once answered");
}

#[test]
fn the_form_underneath_the_consent_gate_is_not_edited_by_the_answer() {
    let mut s = params();
    let index = metrale_bench::registry::all()
        .iter()
        .position(|d| d.needs_confirmation)
        .expect("a benchmark that asks");
    s.select(index);
    s.view = View::Params;
    s.on_key(key(KeyCode::Char('s')), BenchSub::Suite);
    let before = s.edit.clone();
    // 2026-09-26: `d` would reset the form if it reached the form.
    s.on_key(key(KeyCode::Char('d')), BenchSub::Suite);
    assert!(!s.confirm_open, "anything but y backs out");
    assert_eq!(s.edit, before);
}

#[test]
fn only_esc_answers_a_pre_flight_that_is_still_checking() {
    let mut s = params();
    s.preflight = Some(Preflight::pending());
    for code in [
        KeyCode::Char('d'),
        KeyCode::Char('p'),
        KeyCode::Char('j'),
        KeyCode::Enter,
    ] {
        s.on_key(key(code), BenchSub::Suite);
        assert!(s.preflight.is_some(), "{code:?} must not answer the check");
    }
    assert_eq!(s.row, 0, "and must not move the form underneath");
    assert_eq!(s.coherence, metrale_bench::CoherencePolicy::Probe);
    s.on_key(key(KeyCode::Esc), BenchSub::Suite);
    assert!(s.preflight.is_none(), "Esc abandons the run");
    assert_eq!(s.view, View::Params);
}

#[test]
fn a_reported_concern_can_be_overruled_and_the_run_proceeds() {
    // 2026-09-26: A concern can be overruled: `p`, `P` and Enter each try to start the run.
    for code in [KeyCode::Char('p'), KeyCode::Char('P'), KeyCode::Enter] {
        let mut s = params();
        s.preflight = Some(Preflight::with_concern("a concern".into()));
        let text = toast(s.on_key(key(code), BenchSub::Suite));
        assert!(text.contains("executor"), "{code:?} tried to start: {text}");
        assert!(s.preflight.is_none(), "{code:?} dismissed the modal");
    }
}

#[test]
fn any_other_key_backs_out_of_a_reported_concern() {
    let mut s = params();
    s.preflight = Some(Preflight::with_concern("a concern".into()));
    assert!(is_silent(
        s.on_key(key(KeyCode::Char('n')), BenchSub::Suite)
    ));
    assert!(s.preflight.is_none());
    assert_eq!(s.view, View::Params, "back to the form, not into a run");
}

#[test]
fn a_pending_pre_flight_reports_that_it_is_still_checking() {
    let s = Preflight::pending();
    assert!(s.is_checking());
    assert_eq!(s.phase, Phase::Checking);
}

#[test]
fn the_run_pane_scrolls_and_clamps_at_the_top() {
    let mut s = state();
    s.view = View::Run;
    s.table_scroll_max.set(30);
    for _ in 0..3 {
        s.on_key(key(KeyCode::Char('j')), BenchSub::Suite);
    }
    assert_eq!(s.table_scroll, 3);
    for _ in 0..10 {
        s.on_key(key(KeyCode::Up), BenchSub::Suite);
    }
    assert_eq!(s.table_scroll, 0, "no underflow past the first row");
    s.on_key(key(KeyCode::Down), BenchSub::Suite);
    s.on_key(key(KeyCode::Home), BenchSub::Suite);
    assert_eq!(s.table_scroll, 0);
    s.on_key(key(KeyCode::Char('j')), BenchSub::Suite);
    s.on_key(key(KeyCode::Char('g')), BenchSub::Suite);
    assert_eq!(s.table_scroll, 0);
}

#[test]
fn cancelling_when_nothing_is_running_says_nothing_and_does_nothing() {
    let mut s = state();
    s.view = View::Run;
    assert!(is_silent(
        s.on_key(key(KeyCode::Char('c')), BenchSub::Suite)
    ));
    assert_eq!(s.status, "", "there is no run to report on");
    assert_eq!(s.view, View::Run);
}

#[test]
fn the_run_pane_steps_back_to_the_list() {
    for code in [KeyCode::Esc, KeyCode::Left, KeyCode::Char('h')] {
        let mut s = state();
        s.view = View::Run;
        s.on_key(key(code), BenchSub::Suite);
        assert_eq!(s.view, View::List, "{code:?}");
    }
}

#[test]
fn history_navigation_on_an_empty_history_does_not_move_a_cursor_it_has_no_rows_for() {
    let mut s = state();
    assert!(s.history.is_empty());
    s.on_key(key(KeyCode::Char('j')), BenchSub::History);
    s.on_key(key(KeyCode::Down), BenchSub::History);
    assert_eq!(s.history_row, 0);
    assert_eq!(s.view, View::List, "History does not drive the Suite view");
}

#[test]
fn history_navigation_is_clamped_to_the_rows_that_exist() {
    let mut s = state();
    s.history = vec![record(), record(), record()];
    for _ in 0..10 {
        s.on_key(key(KeyCode::Char('j')), BenchSub::History);
    }
    assert_eq!(s.history_row, 2);
    for _ in 0..10 {
        s.on_key(key(KeyCode::Char('k')), BenchSub::History);
    }
    assert_eq!(s.history_row, 0);
}

/// 2026-09-26: A minimal persisted run, for the History pane's cursor.
fn record() -> metrale_bench::RunRecord {
    let descriptor = metrale_bench::registry::find("concurrency-sweep").expect("registered");
    metrale_bench::RunRecord::new(
        descriptor,
        &metrale_bench::ParamValues::default(),
        &metrale_bench::TargetEndpoint::local(8888, "m"),
        Default::default(),
        metrale_bench::RunSource::Tui,
        crate::cli::METRALE_VERSION,
        metrale_bench::BenchmarkResult {
            status: metrale_bench::RunStatus::Completed,
            phase: "done".into(),
            progress: None,
            summary: Vec::new(),
            table: None,
            verdict: None,
            metrics: std::collections::BTreeMap::new(),
            log: Vec::new(),
            dataset_fingerprint: None,
            elapsed: std::time::Duration::from_secs(1),
            hardware_state: None,
        },
    )
}

#[test]
fn the_run_table_clamps_at_its_published_ceiling_and_answers_g_end() {
    let mut s = state();
    s.view = View::Run;
    s.table_scroll_max.set(12);
    for _ in 0..40 {
        s.on_key(key(KeyCode::Char('j')), BenchSub::Suite);
    }
    assert_eq!(s.table_scroll, 12, "j stops at the last row, never banks");
    s.on_key(key(KeyCode::Char('k')), BenchSub::Suite);
    assert_eq!(s.table_scroll, 11, "one press up means one row up");

    s.on_key(key(KeyCode::Char('g')), BenchSub::Suite);
    assert_eq!(s.table_scroll, 0);
    s.on_key(key(KeyCode::Char('G')), BenchSub::Suite);
    assert_eq!(s.table_scroll, 12);
    s.on_key(key(KeyCode::End), BenchSub::Suite);
    assert_eq!(s.table_scroll, 12, "End is the same jump");
}

/// 2026-09-26: In History, j/k select the run, PgUp/PgDn page the stored table, and a new selection
/// resets the table offset.
#[test]
fn the_history_table_pages_and_a_selection_change_resets_it() {
    let mut s = state();
    s.history_table_scroll_max.set(20);
    s.on_key(key(KeyCode::PageDown), BenchSub::History);
    assert_eq!(s.history_table_scroll, 5);
    for _ in 0..10 {
        s.on_key(key(KeyCode::PageDown), BenchSub::History);
    }
    assert_eq!(s.history_table_scroll, 20, "clamped at the ceiling");
    s.on_key(key(KeyCode::PageUp), BenchSub::History);
    assert_eq!(s.history_table_scroll, 15);

    s.on_key(key(KeyCode::Char('k')), BenchSub::History);
    assert_eq!(
        s.history_table_scroll, 0,
        "the offset described another run's table"
    );
}
