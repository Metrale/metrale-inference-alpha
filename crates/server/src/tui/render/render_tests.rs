// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Render tests over `TestBackend`: every section and Benchmarks
//! view at several sizes, which catches a layout that panics (a `Rect` past the
//! frame, an underflowing subtraction), plus content checks. `app()` is the
//! shared fixture for the render tests.
//!
//! Owner: server tui.
//! Invariants: none beyond the types.

use ratatui::Terminal;
use ratatui::backend::TestBackend;

use super::draw;
use crate::tui::app::{App, BenchSub, Section};
use crate::tui::bench_state::View;

pub(super) fn app() -> App {
    use clap::Parser;
    let mut app = App::new(crate::cli::ServeArgs::parse_from([
        "met",
        "nvidia/Qwen3.6-27B-NVFP4",
    ]));
    // 2026-09-26: No `attach` (it needs a `BenchmarkExecutor`); `select(0)`
    // loads the first benchmark's form, which `attach` would also do.
    app.bench.select(0);
    app
}

pub(super) fn render(app: &App, width: u16, height: u16) -> String {
    let mut terminal = Terminal::new(TestBackend::new(width, height)).expect("backend");
    terminal.draw(|f| draw(f, app)).expect("draw");
    terminal
        .backend()
        .buffer()
        .content()
        .iter()
        .map(|c| c.symbol())
        .collect()
}

/// 2026-09-26: The wide sidebar and tall header (160×48, 100×30), the narrow
/// sidebar and one-line header (80×24, under `Chrome::of`'s 96 columns and 28
/// rows), and a small 40×12.
const SIZES: [(u16, u16); 4] = [(160, 48), (100, 30), (80, 24), (40, 12)];

#[test]
fn every_section_renders_at_every_size() {
    for section in Section::ALL {
        for (w, h) in SIZES {
            let mut a = app();
            a.section = section;
            let out = render(&a, w, h);
            assert!(
                !out.is_empty(),
                "{} at {w}x{h} drew nothing",
                section.label()
            );
        }
    }
}

#[test]
fn every_benchmarks_view_renders_at_every_size() {
    for sub in [BenchSub::Suite, BenchSub::History] {
        for view in [View::List, View::Params, View::Run] {
            for (w, h) in SIZES {
                let mut a = app();
                a.section = Section::Benchmarks;
                a.bench_sub = sub;
                a.bench.view = view;
                let out = render(&a, w, h);
                assert!(!out.is_empty(), "bench view at {w}x{h} drew nothing");
            }
        }
    }
}

#[test]
fn the_suite_list_shows_the_benchmarks_and_their_provenance() {
    let mut a = app();
    a.section = Section::Benchmarks;
    // 2026-09-26: The list scrolls with the cursor, so one frame need not
    // hold every name: select each entry in turn, which proves each one is
    // reachable.
    for (i, descriptor) in metrale_bench::registry::all().iter().enumerate() {
        a.bench.select(i);
        let out = render(&a, 160, 48);
        assert!(out.contains(descriptor.name), "missing {}", descriptor.name);
    }
    a.bench.select(0);
    let out = render(&a, 160, 48);
    assert!(out.contains("OFFICIAL"), "first-party badge is missing");
    assert!(out.contains("Metrale Engine"), "author is missing");
}

#[test]
fn the_parameter_form_shows_every_field_plus_the_endpoint() {
    let mut a = app();
    a.section = Section::Benchmarks;
    a.bench.view = View::Params;
    let out = render(&a, 160, 48);
    for spec in &a.bench.specs {
        assert!(out.contains(spec.label), "missing field {}", spec.label);
    }
    assert!(out.contains("TARGET"));
    assert!(out.contains("START"), "the start key must be discoverable");
}

#[test]
fn the_confirmation_modal_says_what_it_will_do() {
    let mut a = app();
    a.section = Section::Benchmarks;
    let index = metrale_bench::registry::all()
        .iter()
        .position(|d| d.needs_confirmation)
        .expect("one benchmark runs shell");
    a.bench.select(index);
    a.bench.view = View::Params;
    a.bench.confirm_open = true;
    let out = render(&a, 160, 48);
    assert!(out.contains("shell"), "the consent gate must name the risk");
    assert!(out.contains("sandbox"));
}

#[test]
fn the_glow_ring_is_titled_only_while_a_benchmark_runs() {
    let mut a = app();
    a.section = Section::Stats;
    assert!(
        !render(&a, 160, 48).contains("⏱"),
        "an idle ring carries no title"
    );
    a.bench.glow = true;
    let running = render(&a, 160, 48);
    assert!(
        running.contains("⏱"),
        "the run signal must follow you out of the Benchmarks section"
    );
    // 2026-09-26: The ring names the selected benchmark
    // (`BenchState::descriptor`).
    let name = a.bench.descriptor().expect("a benchmark is selected").name;
    assert!(running.contains(name), "ring title must carry {name:?}");
}

#[test]
fn the_history_pane_says_so_when_there_is_nothing_to_show() {
    let mut a = app();
    a.section = Section::Benchmarks;
    a.bench_sub = BenchSub::History;
    let out = render(&a, 160, 48);
    assert!(out.contains("No runs recorded yet"));
    assert!(out.contains(".metrale/runs"), "say where they will appear");
}

#[test]
fn a_terminal_one_cell_wide_does_not_panic() {
    for (w, h) in [(1, 1), (2, 3), (1, 40), (40, 1)] {
        let mut a = app();
        a.section = Section::Benchmarks;
        let _ = render(&a, w, h);
    }
}

/// 2026-09-26: The Library panes at realistic and very small sizes.
mod library {
    use super::*;
    use crate::tui::lib_state::View as LibView;

    fn with_rows() -> App {
        let mut app = app();
        app.section = Section::Library;
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/recipes/qwen3.6/qwen3.6-35b-a3b-fp8-mtp.yaml");
        let recipe = crate::recipe::Recipe::parse(
            "qwen3.6/flagship",
            &std::fs::read_to_string(path).expect("fixture"),
        )
        .expect("parses");
        app.library = vec![crate::tui::data::library::LibraryEntry {
            id: recipe.model.clone(),
            snapshot_dir: Default::default(),
            size_bytes: 34_900_000_000,
            has_weights: true,
            model_type: "qwen3_6_moe".into(),
            quant: "fp8".into(),
            layers: 40,
            hidden: 4096,
            heads: 32,
            experts: 128,
            context: 65536,
            optimized: true,
        }];
        app.lib.index = crate::recipe::fetch::Index {
            recipes: vec![recipe],
            ..Default::default()
        };
        app.lib.rebuild(&app.library);
        app
    }

    /// 2026-09-26: The title's row count agrees with the rows drawn.
    #[test]
    fn the_title_agrees_with_the_rows_it_draws() {
        let app = with_rows();
        let out = render(&app, 200, 50);
        assert!(out.contains("MODELS"), "the panel is drawn");
        assert!(
            !out.contains("MODELS ─ 0"),
            "a populated list must not claim 0 rows:\n{out}"
        );
        assert!(
            !out.contains("no models or recipes yet"),
            "the empty hint must not appear beside real rows:\n{out}"
        );
        assert!(out.contains("Qwen3.6-35B-A3B-FP8"), "the row is drawn");
    }

    #[test]
    fn the_empty_state_says_what_to_do() {
        let mut app = app();
        app.section = Section::Library;
        let out = render(&app, 200, 50);
        assert!(out.contains("press r to fetch recipes"), "{out}");
    }

    #[test]
    fn the_config_pane_renders_and_shows_the_command() {
        let mut app = with_rows();
        app.lib.open_cards().expect("opens");
        app.lib.open_config().expect("opens");
        assert_eq!(app.lib.view, LibView::Config);
        let out = render(&app, 200, 50);
        assert!(out.contains("SETTINGS"), "{out}");
        assert!(out.contains("met serve"), "the command preview: {out}");
    }

    #[test]
    fn the_cards_pane_shows_the_recipe_and_its_rationale() {
        let mut app = with_rows();
        app.lib.open_cards().expect("opens");
        assert_eq!(app.lib.view, LibView::Cards);
        let out = render(&app, 200, 50);
        assert!(out.contains("recipe"), "the header counts them: {out}");
        // 2026-09-26: The fixture's description starts "THE FLAGSHIP".
        assert!(out.contains("FLAGSHIP"), "the recipe's own text: {out}");
        assert!(out.contains("configure and start"), "{out}");
    }

    /// 2026-09-26: A one-recipe model still gets a card, titled in the
    /// singular.
    #[test]
    fn one_recipe_still_renders_a_card() {
        let mut app = with_rows();
        assert_eq!(app.lib.cards().len(), 1, "the fixture has one");
        app.lib.open_cards().expect("opens");
        let out = render(&app, 200, 50);
        assert!(
            out.contains("1 recipe"),
            "singular, not \"1 recipes\": {out}"
        );
    }

    #[test]
    fn the_library_survives_hostile_sizes() {
        let app = with_rows();
        for (w, h) in [(40, 12), (60, 20), (80, 24), (120, 30), (240, 80)] {
            let _ = render(&app, w, h);
        }
        let mut cards = with_rows();
        cards.lib.open_cards().expect("opens");
        for (w, h) in [(40, 12), (60, 20), (80, 24), (240, 80)] {
            let _ = render(&cards, w, h);
        }
        let mut config = with_rows();
        config.lib.open_cards().expect("opens");
        config.lib.open_config().expect("opens");
        for (w, h) in [(40, 12), (60, 20), (80, 24), (240, 80)] {
            let _ = render(&config, w, h);
        }
    }
}

/// 2026-09-26: Two frames into one terminal, empty Library then populated, so
/// text the first frame left behind would show; a single-frame test cannot see
/// that.
#[test]
fn the_library_leaves_nothing_behind_when_it_fills_in() {
    let mut terminal = Terminal::new(TestBackend::new(200, 50)).expect("backend");

    let mut app = app();
    app.section = Section::Library;
    terminal.draw(|f| draw(f, &app)).expect("draw");

    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/recipes/qwen3.6/qwen3.6-35b-a3b-fp8-mtp.yaml");
    let recipe = crate::recipe::Recipe::parse(
        "qwen3.6/flagship",
        &std::fs::read_to_string(path).expect("fixture"),
    )
    .expect("parses");
    app.lib.index = crate::recipe::fetch::Index {
        recipes: vec![recipe],
        ..Default::default()
    };
    app.lib.rebuild(&[]);
    terminal.draw(|f| draw(f, &app)).expect("draw");

    let out: String = terminal
        .backend()
        .buffer()
        .content()
        .iter()
        .map(|c| c.symbol())
        .collect();
    assert!(
        !out.contains("no models or recipes yet"),
        "the empty hint survived into the populated frame:\n{out}"
    );
    assert!(
        !out.contains("MODELS ─ 0"),
        "the empty title survived into the populated frame:\n{out}"
    );
    assert!(out.contains("MODELS ─ 1"), "the new title is drawn:\n{out}");
}

/// 2026-09-26: The Benchmarks pre-flight modal and the run log.
mod preflight {
    use super::*;
    use crate::tui::bench_preflight::Preflight;

    fn on_params() -> App {
        let mut app = app();
        app.section = Section::Benchmarks;
        app.bench_sub = BenchSub::Suite;
        app.bench.view = View::Params;
        app
    }

    #[test]
    fn the_checking_modal_shows_a_spinner_and_the_target() {
        let mut app = on_params();
        app.bench.preflight = Some(Preflight::pending());
        let out = render(&app, 200, 50);
        assert!(out.contains("CHECKING THE ENDPOINT"), "{out}");
        assert!(out.contains("known-answer"), "says what it is doing: {out}");
    }

    /// 2026-09-26: A long concern wraps; its tail stays on screen.
    #[test]
    fn a_concern_is_wrapped_not_truncated() {
        let mut app = on_params();
        let long = "http://127.0.0.1:8888 is serving \"nvidia/Qwen3.6-27B-NVFP4\", which did not \
                    answer as expected (recall answered nothing). This benchmark may be aimed at \
                    a different model, or the checkpoint may be a base (non-instruct) one — the \
                    run is still valid, but read the numbers with that in mind.";
        app.bench.preflight = Some(Preflight::with_concern(long.to_string()));
        let out = render(&app, 200, 50);
        assert!(out.contains("BEFORE YOU START"), "{out}");
        assert!(out.contains("run it anyway"), "offers to proceed: {out}");
        assert!(out.contains("back to the form"), "offers to go back: {out}");
        assert!(
            out.contains("with that in mind"),
            "the end of the reason was lost:\n{out}"
        );
    }

    /// 2026-09-26: A long log line wraps inside its panel.
    #[test]
    fn run_log_lines_wrap_inside_the_panel() {
        let mut app = on_params();
        app.bench.view = View::Run;
        let tail = "and this tail must still be on screen";
        app.bench.log.push_back(metrale_bench::LogLine {
            level: metrale_bench::LogLevel::Warn,
            text: format!(
                "http://127.0.0.1:8888 is serving a model that did not answer as expected, \
                 which usually means the benchmark is aimed somewhere else — {tail}"
            ),
        });
        let out = render(&app, 120, 40);
        assert!(
            out.contains(tail),
            "the line was truncated, not wrapped:\n{out}"
        );
    }
}

#[test]
fn the_benchmark_detail_pane_says_when_the_measurement_last_changed() {
    // 2026-09-26: The detail pane shows the descriptor's `updated` date.
    let mut a = app();
    a.section = Section::Benchmarks;
    a.bench.view = View::List;
    let out = render(&a, 200, 50);
    assert!(out.contains("Updated"), "the row is drawn:\n{out}");
    let d = a
        .bench
        .descriptor()
        .expect("a benchmark is selected")
        .updated;
    assert!(out.contains(d), "and carries the date {d}:\n{out}");
}

#[test]
fn the_clear_chat_prompt_names_what_it_will_discard() {
    use crate::tui::app::TermSub;
    use crate::tui::chat::{ChatMessage, Role};
    let mut a = app();
    a.section = Section::Terminal;
    a.term_sub = TermSub::Chat;
    a.chat
        .transcript
        .push(ChatMessage::new(Role::User, "hello".into()));
    a.chat
        .transcript
        .push(ChatMessage::new(Role::Model, "hi".into()));
    a.confirm_chat_clear = true;
    let out = render(&a, 120, 40);
    assert!(out.contains("CLEAR THE CONVERSATION?"), "{out}");
    assert!(
        out.contains("2 turns will be discarded"),
        "the stake is named in the user's own units:\n{out}"
    );
    assert!(
        out.contains("any other key"),
        "the way out is named:\n{out}"
    );
}

#[test]
fn the_new_chat_key_is_discoverable_from_the_chat_pane() {
    use crate::tui::app::TermSub;
    let mut a = app();
    a.section = Section::Terminal;
    a.term_sub = TermSub::Chat;
    let out = render(&a, 160, 48);
    assert!(
        out.contains("Ctrl+N new chat"),
        "a reset nobody can find is no reset:\n{out}"
    );
}
