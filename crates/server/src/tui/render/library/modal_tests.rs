// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Render tests for the config form's pickers and its added and
//! removed rows.
//!
//! Owner: server tui.
//! Invariants: none beyond the types.
//!
//! The harness reads cell symbols, not styles, so these assertions do not
//! depend on colour.

use crossterm::event::{KeyCode, KeyEvent};

use super::super::tests::{lib, local, recipe};
use crate::tui::app::App;
use crate::tui::render::harness::{has, screen};

fn form() -> App {
    let r = recipe("qwen3.6-35b-a3b-fp8-mtp");
    let model = r.model.clone();
    let mut a = lib(vec![r], vec![local(&model, true)]);
    a.lib.open_cards().expect("cards");
    a.lib.open_config().expect("form");
    a
}

fn press(a: &mut App, code: KeyCode) {
    a.lib.on_key(KeyEvent::from(code));
}

fn select_row(a: &mut App, key: &str) {
    a.lib.row = a
        .lib
        .config_rows()
        .iter()
        .position(|r| r.key == key)
        .unwrap_or_else(|| panic!("{key} is not on the form"));
}

#[test]
fn the_options_picker_names_the_flag_and_marks_the_current_value() {
    let mut a = form();
    select_row(&mut a, "kv_cache_dtype");
    press(&mut a, KeyCode::Enter);
    let rows = screen(&a, 160, 48);
    assert!(has(&rows, "KV-CACHE-DTYPE"), "titled by flag:\n{rows:#?}");
    // 2026-09-26: The fixture recipe sets `kv_cache_dtype: bf16`.
    assert!(has(&rows, "✓ bf16"), "current value marked:\n{rows:#?}");
    assert!(has(&rows, "▌"), "the cursor bar is a glyph, not a hue");
}

#[test]
fn a_sixteen_row_option_list_scrolls_rather_than_clips() {
    let mut a = form();
    select_row(&mut a, "kv_cache_dtype");
    press(&mut a, KeyCode::Enter);
    press(&mut a, KeyCode::Char('G'));
    // 2026-09-26: A 14-row terminal forces the list to scroll.
    let rows = screen(&a, 100, 14);
    assert!(
        has(&rows, "fp8k_turbo2v"),
        "the cursor's row is inside the window at the bottom:\n{rows:#?}"
    );
    assert!(
        !has(&rows, "✓ bf16"),
        "the top of the list scrolled out:\n{rows:#?}"
    );
    assert!(
        has(&rows, "16/16"),
        "the clipped list says where you are:\n{rows:#?}"
    );
}

#[test]
fn the_add_picker_lists_flags_with_their_help() {
    let mut a = form();
    press(&mut a, KeyCode::Char('a'));
    let rows = screen(&a, 200, 48);
    assert!(has(&rows, "ADD A SETTING"), "{rows:#?}");
    // 2026-09-26: A flag the recipe does not set, with the first line of its
    // clap help.
    assert!(has(&rows, "block_size"), "{rows:#?}");
    assert!(has(&rows, "KV cache block size"), "{rows:#?}");
}

#[test]
fn a_removed_row_reads_removed_in_words_not_in_colour() {
    let mut a = form();
    select_row(&mut a, "scheduler");
    press(&mut a, KeyCode::Char('x'));
    let rows = screen(&a, 200, 50);
    assert!(has(&rows, "✗"), "the gutter mark:\n{rows:#?}");
    assert!(
        has(&rows, "removed — server default fifo"),
        "the value column names what the server will do:\n{rows:#?}"
    );
    // 2026-09-26: The command preview proves the flag is not passed.
    assert!(
        !has(&rows, "--scheduler"),
        "a removed flag must not survive into the launch command:\n{rows:#?}"
    );
}

#[test]
fn an_added_row_is_marked_and_reaches_the_launch_command() {
    let mut a = form();
    a.lib.overrides.insert("block_size".into(), "32".into());
    let rows = screen(&a, 200, 50);
    assert!(has(&rows, "+ block_size"), "the + gutter mark:\n{rows:#?}");
    assert!(has(&rows, "--block-size 32"), "{rows:#?}");
}

#[test]
fn removals_count_toward_the_changed_tally_in_the_title() {
    let mut a = form();
    select_row(&mut a, "scheduler");
    press(&mut a, KeyCode::Char('x'));
    let rows = screen(&a, 200, 50);
    assert!(
        has(&rows, "1 changed"),
        "a removal is a change to the launch:\n{rows:#?}"
    );
}

#[test]
fn the_footer_teaches_the_picker_keys_while_one_is_open() {
    let a = form();
    let rows = screen(&a, 200, 50);
    assert!(has(&rows, "a add"), "add is discoverable:\n{rows:#?}");
    assert!(has(&rows, "x remove"), "remove is discoverable:\n{rows:#?}");
    let mut a = form();
    select_row(&mut a, "kv_cache_dtype");
    press(&mut a, KeyCode::Enter);
    let rows = screen(&a, 200, 50);
    assert!(
        has(&rows, "⏎ select · Esc cancel"),
        "the footer answers for the option picker:\n{rows:#?}"
    );
    let mut a = form();
    press(&mut a, KeyCode::Char('a'));
    let rows = screen(&a, 200, 50);
    assert!(
        has(&rows, "J/K scroll help · ⏎ add"),
        "the footer answers for the add picker's own keys:\n{rows:#?}"
    );
}

#[test]
fn the_pickers_render_at_every_size_without_panicking() {
    for (w, h) in [(160u16, 48u16), (100, 30), (80, 24), (40, 12), (12, 4)] {
        for open in ["options", "add"] {
            let mut a = form();
            if open == "options" {
                select_row(&mut a, "kv_cache_dtype");
                press(&mut a, KeyCode::Enter);
            } else {
                press(&mut a, KeyCode::Char('a'));
            }
            let out = screen(&a, w, h);
            assert!(!out.is_empty(), "{open} at {w}x{h} drew nothing");
        }
    }
}

/// 2026-09-26: The 35B fixture's form, with the 27B fixture in the index as a
/// donor.
fn form_with_donor() -> App {
    let flagship = recipe("qwen3.6-35b-a3b-fp8-mtp");
    let donor = recipe("qwen3.6-27b-nvfp4");
    let model = flagship.model.clone();
    let mut a = lib(vec![flagship, donor], vec![local(&model, true)]);
    a.lib.open_cards().expect("cards");
    a.lib.open_config().expect("form");
    a
}

#[test]
fn the_borrow_picker_names_each_donors_measured_model() {
    let mut a = form_with_donor();
    press(&mut a, KeyCode::Char('b'));
    let rows = screen(&a, 200, 50);
    assert!(has(&rows, "BORROW PARAMETERS FROM"), "{rows:#?}");
    assert!(has(&rows, "qwen3.6/qwen3.6-27b-nvfp4"), "{rows:#?}");
    assert!(
        has(&rows, "measured on nvidia/Qwen3.6-27B-NVFP4"),
        "{rows:#?}"
    );
}

#[test]
fn the_preview_shows_old_to_new_and_says_it_is_not_a_measurement() {
    let mut a = form_with_donor();
    press(&mut a, KeyCode::Char('b'));
    press(&mut a, KeyCode::Enter);
    let rows = screen(&a, 200, 50);
    assert!(has(&rows, "65536 → 32768"), "old beside new:\n{rows:#?}");
    assert!(
        has(&rows, "not set → qwen3_coder"),
        "an added key says it was absent:\n{rows:#?}"
    );
    assert!(
        has(&rows, "not measured on this model"),
        "the honesty header:\n{rows:#?}"
    );
    assert!(
        has(&rows, "keep their values"),
        "what the borrow does NOT touch is stated:\n{rows:#?}"
    );
    // 2026-09-26: The preview's Enter applies rather than selects.
    assert!(has(&rows, "⏎ apply changes"), "{rows:#?}");
}

#[test]
fn an_applied_borrow_marks_the_form_with_its_provenance() {
    let mut a = form_with_donor();
    press(&mut a, KeyCode::Char('b'));
    press(&mut a, KeyCode::Enter);
    press(&mut a, KeyCode::Enter);
    let rows = screen(&a, 200, 50);
    assert!(
        has(&rows, "borrowed — values from qwen3.6/qwen3.6-27b-nvfp4"),
        "the form says where its values came from:\n{rows:#?}"
    );
    assert!(
        has(&rows, "not a measurement for this model"),
        "and that they are copies, in words that survive NO_COLOR:\n{rows:#?}"
    );
    // 2026-09-26: Borrowed rows get the `•` gutter mark of any change.
    assert!(has(&rows, "• max_model_len"), "{rows:#?}");
}

#[test]
fn the_borrow_surfaces_render_at_every_size_without_panicking() {
    for (w, h) in [(160u16, 48u16), (100, 30), (80, 24), (40, 12), (12, 4)] {
        for depth in [1u8, 2] {
            let mut a = form_with_donor();
            press(&mut a, KeyCode::Char('b'));
            if depth == 2 {
                press(&mut a, KeyCode::Enter);
            }
            let out = screen(&a, w, h);
            assert!(
                !out.is_empty(),
                "borrow depth {depth} at {w}x{h} drew nothing"
            );
        }
    }
}

/// 2026-09-26: The add picker with the cursor on `check_kernels`, whose clap
/// help (`ServeArgs::check_kernels`) runs to several paragraphs.
fn add_picker_on_check_kernels(w: u16, h: u16) -> (App, Vec<String>) {
    let mut a = form();
    press(&mut a, KeyCode::Char('a'));
    let Some(crate::tui::lib_modal::ConfigModal::Add { fields, .. }) = &a.lib.modal else {
        panic!("add picker open");
    };
    let target = fields
        .iter()
        .position(|f| f.key == "check_kernels")
        .expect("check_kernels is addable");
    for _ in 0..target {
        press(&mut a, KeyCode::Char('j'));
    }
    let rows = screen(&a, w, h);
    (a, rows)
}

#[test]
fn the_side_panel_shows_the_help_beyond_the_first_line() {
    let (_, rows) = add_picker_on_check_kernels(200, 50);
    assert!(
        has(&rows, "CHECK_KERNELS"),
        "titled by the flag:\n{rows:#?}"
    );
    // 2026-09-26: "CLAMPED" is in the third paragraph, which the one-line row
    // cannot show.
    assert!(
        has(&rows, "CLAMPED"),
        "a later paragraph is readable:\n{rows:#?}"
    );
}

#[test]
fn the_panel_follows_the_cursor_and_shift_j_scrolls_it() {
    // 2026-09-26: A short terminal, so the help overflows the panel.
    let (mut a, rows) = add_picker_on_check_kernels(200, 24);
    // 2026-09-26: From the help's first paragraph, beyond what the list row's
    // clipped line holds.
    let panel_only = "reporting any";
    assert!(has(&rows, panel_only), "top of the help:\n{rows:#?}");
    assert!(has(&rows, "J/K 1/"), "position and binding:\n{rows:#?}");
    for _ in 0..8 {
        press(&mut a, KeyCode::Char('J'));
    }
    let rows = screen(&a, 200, 24);
    assert!(
        !has(&rows, panel_only),
        "the opening lines scrolled away:\n{rows:#?}"
    );
    assert!(has(&rows, "J/K 9/"), "{rows:#?}");
    press(&mut a, KeyCode::Char('j'));
    let rows = screen(&a, 200, 24);
    assert!(
        !has(&rows, "CHECK_KERNELS"),
        "the panel answers for the highlighted row only:\n{rows:#?}"
    );
}

#[test]
fn a_narrow_terminal_drops_the_panel_whole_and_keeps_the_list() {
    // 2026-09-26: At 80 columns the content area has less than 50 list
    // columns plus the 34-column panel (`HELP_PANEL_TEXT_W` + 4).
    let (_, rows) = add_picker_on_check_kernels(80, 30);
    assert!(
        has(&rows, "ADD A SETTING"),
        "the picker survives:\n{rows:#?}"
    );
    assert!(
        !has(&rows, "CHECK_KERNELS"),
        "no panel sliver at 80 columns:\n{rows:#?}"
    );
    assert!(has(&rows, "…"), "ellipsis on clipped rows:\n{rows:#?}");
}
