// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for `LibState`: the joined list, the config form, refresh anchoring, and the launch result channel.
//!
//! Owner: server tui.
//! Invariants: none beyond the types.

use super::*;
use crate::recipe::Recipe;

fn real_recipe() -> Recipe {
    // 2026-09-26: A real recipe fixture, so edits are checked against the
    // recipe's actual `serve_args`.
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/recipes/qwen3.6/qwen3.6-35b-a3b-fp8-mtp.yaml");
    let text = std::fs::read_to_string(&path).expect("fixture");
    Recipe::parse("qwen3.6/flagship", &text).expect("parses")
}

fn local_of(model: &str) -> LibraryEntry {
    LibraryEntry {
        id: model.into(),
        snapshot_dir: Default::default(),
        size_bytes: 1024,
        has_weights: true,
        model_type: "qwen3_5_moe".into(),
        quant: "fp8".into(),
        layers: 40,
        hidden: 4096,
        heads: 32,
        experts: 128,
        context: 65536,
        optimized: true,
    }
}

fn state_with_recipe() -> LibState {
    let recipe = real_recipe();
    let local = vec![local_of(&recipe.model)];
    let mut s = LibState {
        index: Index {
            recipes: vec![recipe],
            ..Index::default()
        },
        ..LibState::default()
    };
    s.rebuild(&local);
    s
}

#[test]
fn the_list_populates_from_cache_without_a_network() {
    let s = state_with_recipe();
    assert_eq!(s.rows.len(), 1);
    assert!(s.current().expect("a row").runnable_now());
}

#[test]
fn a_model_with_no_recipe_opens_on_starting_points() {
    // 2026-09-26: A local-only checkpoint opens the cards on synthesized
    // starting points; their rules are tested in `lib_start_tests`.
    let mut s = LibState::default();
    s.rebuild(&[local_of("org/orphan")]);
    s.open_cards().expect("no longer refused");
    assert_eq!(s.view, View::Cards);
    assert!(s.cards().iter().all(|c| c.starting_point.is_some()));
}

#[test]
fn the_form_shows_every_recipe_key_with_its_value() {
    let mut s = state_with_recipe();
    s.open_cards().expect("opens the cards");
    s.open_config().expect("opens the form");
    let rows = s.config_rows();
    let recipe = s.config_recipe().expect("recipe");
    assert_eq!(rows.len(), recipe.defaults.len());
    assert!(rows.iter().all(|r| !r.changed), "nothing edited yet");
    let row = rows.iter().find(|r| r.key == "port").expect("port");
    assert_eq!(row.key, "port");
    assert_eq!(row.value, "8888");
}

#[test]
fn a_valid_edit_is_kept_and_marked() {
    let mut s = state_with_recipe();
    s.open_cards().expect("opens the cards");
    s.open_config().expect("opens the form");
    let rows = s.config_rows();
    s.row = rows
        .iter()
        .position(|r| r.key == "max_model_len")
        .expect("key");
    s.editing = true;
    s.edit_buffer = "4096".into();
    s.commit_edit();

    assert!(s.error.is_none(), "{:?}", s.error);
    assert!(!s.editing);
    let row = s
        .config_rows()
        .into_iter()
        .find(|r| r.key == "max_model_len")
        .expect("key");
    assert_eq!(row.value, "4096");
    assert!(
        row.changed,
        "an edited row is marked as differing from the recipe"
    );
}

#[test]
fn an_invalid_edit_is_rejected_and_not_kept() {
    // 2026-09-26: A rejected value does not enter `overrides`.
    let mut s = state_with_recipe();
    s.open_cards().expect("opens the cards");
    s.open_config().expect("opens the form");
    let rows = s.config_rows();
    s.row = rows.iter().position(|r| r.key == "scheduler").expect("key");
    s.editing = true;
    s.edit_buffer = "nonsense".into();
    s.commit_edit();

    let err = s.error.clone().expect("rejected");
    assert!(
        err.contains("--scheduler") || err.contains("nonsense"),
        "{err}"
    );
    assert!(
        s.overrides.is_empty(),
        "the bad value must not enter the overrides"
    );
    let row = s
        .config_rows()
        .into_iter()
        .find(|r| r.key == "scheduler")
        .expect("key");
    assert_eq!(row.value, "slai", "still the recipe's value");
    assert!(!row.changed);
}

#[test]
fn an_empty_edit_is_refused_rather_than_silently_clearing_a_flag() {
    let mut s = state_with_recipe();
    s.open_cards().expect("opens the cards");
    s.open_config().expect("opens the form");
    s.editing = true;
    s.edit_buffer = "   ".into();
    s.commit_edit();
    assert!(
        s.error.as_deref().is_some_and(|e| e.contains("empty")),
        "{:?}",
        s.error
    );
    assert!(s.overrides.is_empty());
}

#[test]
fn the_whole_config_is_validated_not_just_the_field() {
    // 2026-09-26: An out-of-range `gpu_memory_utilization` is refused and not
    // kept. The fixture sets that key.
    let mut s = state_with_recipe();
    s.open_cards().expect("opens the cards");
    s.open_config().expect("opens the form");
    let rows = s.config_rows();
    if let Some(i) = rows.iter().position(|r| r.key == "gpu_memory_utilization") {
        s.row = i;
        s.editing = true;
        s.edit_buffer = "9.0".into();
        s.commit_edit();
        assert!(s.error.is_some(), "an out-of-range value must be caught");
        assert!(s.overrides.is_empty());
    }
}

#[test]
fn resetting_returns_to_the_recipes_own_values() {
    let mut s = state_with_recipe();
    s.open_cards().expect("opens the cards");
    s.open_config().expect("opens the form");
    s.row = s
        .config_rows()
        .iter()
        .position(|r| r.key == "port")
        .expect("port");
    s.editing = true;
    s.edit_buffer = "9999".into();
    s.commit_edit();
    assert_eq!(s.overrides.len(), 1);

    s.reset_overrides();
    assert!(s.overrides.is_empty());
    let row = s
        .config_rows()
        .into_iter()
        .find(|r| r.key == "port")
        .expect("port");
    assert_eq!(row.value, "8888");
    assert!(!row.changed);
}

#[test]
fn the_preview_argv_reflects_the_edits() {
    let mut s = state_with_recipe();
    s.open_cards().expect("opens the cards");
    s.open_config().expect("opens the form");
    s.row = s
        .config_rows()
        .iter()
        .position(|r| r.key == "port")
        .expect("port");
    s.editing = true;
    s.edit_buffer = "9999".into();
    s.commit_edit();

    let argv = s.preview_argv().expect("renders");
    let i = argv.iter().position(|a| a == "--port").expect("present");
    assert_eq!(argv[i + 1], "9999");
    assert_eq!(
        argv.iter().filter(|a| *a == "--port").count(),
        1,
        "specified once: {argv:?}"
    );
}

#[test]
fn the_filter_narrows_and_the_selection_stays_in_range() {
    let mut s = state_with_recipe();
    s.rebuild(&[local_of("org/other")]);
    s.selected = s.visible().len().saturating_sub(1);
    s.filter = "zzz-matches-nothing".into();
    s.rebuild(&[]);
    assert!(s.visible().is_empty());
    assert_eq!(s.selected, 0, "a filtered-out selection cannot dangle");
    assert!(s.current().is_none());
}

#[test]
fn a_refresh_without_a_store_is_a_no_op_not_a_panic() {
    let mut s = LibState::default();
    s.refresh();
    assert!(!s.fetching, "nothing to refresh against");
    assert!(!s.poll(&[]), "and polling is harmless");
}

#[test]
fn a_field_error_carries_the_actionable_line_not_the_header() {
    // 2026-09-26: The field shows the violation and its fix, not the report
    // header.
    let report = concat!(
        "Metrale Engine CLI: 1 invalid flag combination — fix before serving:\n\n",
        "  [1] --ep-size 2 exceeds --world-size 1.\n",
        "      why: expert parallelism cannot span more ranks than exist.\n",
        "      fix: raise --world-size to at least --ep-size, or lower --ep-size.\n"
    );
    let line = problem_line(report);
    assert!(line.contains("--ep-size 2 exceeds"), "{line}");
    assert!(
        line.contains("raise --world-size"),
        "carries the fix: {line}"
    );
    assert!(
        !line.contains("Metrale Engine CLI:"),
        "not the header: {line}"
    );
    assert!(!line.contains('\n'), "one line, for one field: {line}");
}

#[test]
fn a_clap_error_without_a_numbered_block_still_reads() {
    let line = problem_line("error: invalid value 'x' for '--port <PORT>'");
    assert!(line.contains("invalid value"), "{line}");
}

#[test]
fn a_refresh_keeps_the_selection_on_the_same_model_not_the_same_row() {
    // 2026-09-26: `catalogue::join` sorts by rank, then model id, so a new
    // row can shift the selected model's index; the selection follows the
    // model.
    let mut s = state_with_recipe();
    let mine = s.current().expect("a row").model.clone();
    s.rebuild(&[local_of(&mine), local_of("aaa/sorts-first")]);
    let before = s.current().expect("a row").model.clone();
    assert_eq!(before, mine, "still the model I had selected");

    s.rebuild(&[local_of("aaa/sorts-first"), local_of(&mine)]);
    assert_eq!(
        s.current().expect("a row").model,
        mine,
        "a reshuffle must not move the selection to another model"
    );
}

#[test]
fn a_refresh_that_removes_the_open_model_steps_back_to_the_list() {
    // 2026-09-26: When the open model leaves the list, the view returns to
    // the list.
    let mut s = state_with_recipe();
    s.open_cards().expect("opens");
    assert_eq!(s.view, View::Cards);
    s.index = Index::default();
    s.rebuild(&[]);
    assert_eq!(
        s.view,
        View::List,
        "cannot stay in a vanished model's cards"
    );
}

#[test]
fn a_shorter_recipe_list_cannot_leave_the_card_index_dangling() {
    let mut s = state_with_recipe();
    s.open_cards().expect("opens");
    s.card = 5;
    s.rebuild(&[local_of(&real_recipe().model)]);
    assert!(
        s.card < s.cards().len().max(1),
        "card {} out of {} ",
        s.card,
        s.cards().len()
    );
}

#[test]
fn edits_do_not_survive_onto_a_recipe_the_user_never_opened() {
    // 2026-09-26: When the recipe being edited leaves the list, its edits are
    // dropped instead of carried onto the recipe that inherits its index.
    let mut s = state_with_recipe();
    s.open_cards().expect("opens");
    s.open_config().expect("opens");
    s.row = s
        .config_rows()
        .iter()
        .position(|r| r.key == "port")
        .expect("port");
    s.editing = true;
    s.edit_buffer = "9100".into();
    s.commit_edit();
    assert_eq!(s.overrides.len(), 1);

    s.index = Index::default();
    s.rebuild(&[local_of(&real_recipe().model)]);
    assert!(
        s.overrides.is_empty(),
        "edits to a vanished recipe are dropped"
    );
    assert!(!s.editing);
}

#[test]
fn a_multi_node_recipe_can_still_be_opened_and_read() {
    // 2026-09-26: `model_swap::swap` refuses world_size > 1 and the card says
    // so, but the form still opens for reading.
    let mut recipe = real_recipe();
    recipe.min_nodes = 2;
    let local = vec![local_of(&recipe.model)];
    let mut s = LibState {
        index: Index {
            recipes: vec![recipe],
            ..Index::default()
        },
        ..LibState::default()
    };
    s.rebuild(&local);
    s.open_cards().expect("cards open");
    s.open_config().expect("and so does the form");

    // 2026-09-26: `Recipe::argv_edited` emits `--world-size` from
    // `min_nodes`.
    let argv = s.preview_argv().expect("renders");
    let i = argv
        .iter()
        .position(|a| a == "--world-size")
        .expect("world-size is emitted from min_nodes");
    assert_eq!(argv[i + 1], "2");
}

#[test]
fn a_launch_that_fails_after_spawning_is_reported_not_swallowed() {
    // 2026-09-26: `launch` returns once the loader thread starts; a later
    // failure arrives once through `poll_launch`.
    let mut s = state_with_recipe();
    assert!(s.poll_launch().is_none(), "nothing in flight yet");

    // 2026-09-26: Stands in for the loader thread's channel.
    let (tx, rx) = std::sync::mpsc::channel::<String>();
    s.launch_result = Some(rx);
    assert!(
        s.poll_launch().is_none(),
        "still loading — nothing said yet"
    );

    tx.send("this build has no compiled kernels for qwen3_6_moe".into())
        .expect("send");
    let got = s.poll_launch().expect("the failure surfaces");
    assert!(got.contains("no compiled kernels"), "{got}");
    assert!(
        s.poll_launch().is_none(),
        "and it is reported once, not every tick"
    );
}

#[test]
fn a_launch_that_succeeds_reports_nothing() {
    // 2026-09-26: The loader drops its sender on success; a disconnect with no
    // message is not an error.
    let mut s = state_with_recipe();
    let (tx, rx) = std::sync::mpsc::channel::<String>();
    s.launch_result = Some(rx);
    drop(tx);
    assert!(s.poll_launch().is_none(), "silence means it worked");
}

/// 2026-09-26: A launch is in flight from spawn until `poll_launch` sees its
/// channel settle, including a disconnect with no message.
#[test]
fn launch_in_flight_tracks_the_result_channel() {
    let mut s = LibState::default();
    assert!(!s.launch_in_flight(), "nothing launched yet");

    let (tx, rx) = std::sync::mpsc::channel::<String>();
    s.launch_result = Some(rx);
    assert!(s.launch_in_flight(), "loader thread is out");

    drop(tx);
    assert!(s.launch_in_flight(), "still in flight until polled");
    assert!(s.poll_launch().is_none(), "silence means it worked");
    assert!(!s.launch_in_flight(), "and the guard lets go");
}
