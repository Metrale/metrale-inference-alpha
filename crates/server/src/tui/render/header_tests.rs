// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for what the header and `logo::badges` claim about the
//! server, above all with no model (`app.awaiting_model`).
//!
//! Owner: server tui.
//! Invariants: none beyond the types.

#[test]
fn the_status_pill_does_not_claim_to_be_serving_with_no_model() {
    // 2026-09-26: `progress.ready` with `awaiting_model` still set must read
    // NO MODEL, not SERVING.
    use clap::Parser as _;
    let mut args = crate::cli::ServeArgs::parse_from(["met", "m"]);
    args.model = None;
    let mut app = crate::tui::app::App::new(args);
    assert!(app.awaiting_model, "no model was named");

    app.progress.ready = true;
    let pill = super::status_pill(&app);
    assert!(
        pill.content.contains("NO MODEL"),
        "a bound socket is not a loaded model: {:?}",
        pill.content
    );

    app.awaiting_model = false;
    assert!(super::status_pill(&app).content.contains("SERVING"));
}

#[test]
fn the_chip_strip_describes_the_running_config_not_the_boot_argv() {
    // 2026-09-26: The chips are built from whichever `ServeArgs` they are
    // given, so a different argv gives a different strip.
    use clap::Parser as _;
    let boot = crate::cli::ServeArgs::parse_from(["met", "m"]);
    let boot_chips = crate::tui::logo::badges(&boot, false);
    let boot_text: String = boot_chips.iter().map(|b| b.text.clone()).collect();

    let mut live = crate::cli::ServeArgs::parse_from(["met", "org/loaded"]);
    live.max_batch_size = boot.max_batch_size + 7;
    let live_chips = crate::tui::logo::badges(&live, false);
    let live_text: String = live_chips.iter().map(|b| b.text.clone()).collect();

    assert!(live_text.contains("org/loaded"), "names the live model");
    assert_ne!(
        boot_text, live_text,
        "the strip must be able to differ from the boot argv"
    );
    assert!(
        live_text.contains(&(boot.max_batch_size + 7).to_string()),
        "and it reports the live batch size: {live_text}"
    );
}

#[test]
fn the_chip_strip_asserts_nothing_about_a_model_that_is_not_loaded() {
    // 2026-09-26: With `awaiting_model`, every chip but the address would be a
    // clap default, so none is drawn.
    use clap::Parser as _;
    let args = crate::cli::ServeArgs::parse_from(["met"]);
    let text: String = crate::tui::logo::badges(&args, true)
        .iter()
        .map(|b| b.text.clone())
        .collect::<Vec<_>>()
        .join(" ");

    for claim in ["kv ", "lm ", "mtp ", "batch ", "ctx ", "sched ", "<model>"] {
        assert!(
            !text.contains(claim),
            "awaiting strip must not claim {claim:?}: {text}"
        );
    }
    assert!(
        text.contains(&format!(":{}", args.port)),
        "the bound port is true with no model: {text}"
    );
    assert!(
        text.contains("Library"),
        "and it says how to fix it: {text}"
    );
}

#[test]
fn the_header_mini_strip_does_not_claim_a_kv_dtype_with_no_model() {
    use clap::Parser as _;
    let mut args = crate::cli::ServeArgs::parse_from(["met", "m"]);
    args.model = None;
    let mut app = crate::tui::app::App::new(args);
    assert!(app.awaiting_model, "no model was named");
    let line = super::header_line(&app);
    assert!(!line.contains("kv "), "no dtype with no model: {line}");
    assert!(line.contains("Library"), "says the way out: {line}");
    assert!(
        line.contains(&format!(":{}", app.args.port)),
        "keeps the bound port: {line}"
    );

    app.awaiting_model = false;
    let line = super::header_line(&app);
    assert!(
        line.contains("kv "),
        "a loaded model has a KV dtype: {line}"
    );
}

/// 2026-09-26: What the header draws on the frame, through `render::draw`.
mod rendered {
    use crate::tui::render::harness::{has, screen};

    /// 2026-09-26: The header rows: three on a tall terminal, one on a short
    /// one.
    fn head(rows: &[String], tall: bool) -> Vec<String> {
        rows[..if tall { 3 } else { 1 }].to_vec()
    }

    #[test]
    fn a_tall_terminal_gets_the_wordmark_and_the_mini_strip() {
        let a = crate::tui::render::tests::app();
        let rows = screen(&a, 160, 48);
        let head = head(&rows, true);
        assert!(has(&head, "M E T R A L E"), "{head:#?}");
        assert!(has(&head, "E N G I N E"), "{head:#?}");
        assert!(has(&head, "SERVING") || has(&head, "LOADING"), "{head:#?}");
        assert!(has(&head, "up 0:00:"), "the uptime clock:\n{head:#?}");
        assert!(
            has(&head, "kv "),
            "and the strip that says what is running:\n{head:#?}"
        );
    }

    #[test]
    fn a_short_terminal_gets_one_row_and_drops_the_strip_rather_than_the_status() {
        // 2026-09-26: Below 28 rows (`Chrome::of`) the header is one line: no
        // mini-strip, but the pill stays.
        let a = crate::tui::render::tests::app();
        let rows = screen(&a, 160, 27);
        let head = head(&rows, false);
        assert!(has(&head, "Metrale Engine"), "{head:#?}");
        assert!(!has(&head, "M E T R A L E"), "{head:#?}");
        assert!(has(&head, "SERVING") || has(&head, "LOADING"), "{head:#?}");
        assert!(has(&head, "up 0:00:"));
    }

    #[test]
    fn the_pill_and_the_strip_agree_about_whether_a_model_is_loaded() {
        let mut a = crate::tui::render::tests::app();
        a.awaiting_model = true;
        a.progress.ready = true;
        let head = head(&screen(&a, 160, 48), true);
        assert!(has(&head, "NO MODEL"), "{head:#?}");
        assert!(has(&head, "press 4 for Library"), "{head:#?}");
        assert!(
            !has(&head, "kv "),
            "no dtype for a process that loaded nothing:\n{head:#?}"
        );
    }

    #[test]
    fn the_header_draws_at_widths_too_narrow_for_anything_it_wants_to_say() {
        for (w, h) in [(1u16, 1u16), (8, 30), (20, 48), (40, 28)] {
            assert_eq!(
                screen(&crate::tui::render::tests::app(), w, h).len(),
                h as usize,
                "{w}x{h}"
            );
        }
    }
}

/// 2026-09-26: Uptime never repeats a string as time advances, across the hour
/// and day boundaries.
#[test]
fn uptime_keeps_counting_past_an_hour_a_day_and_a_hundred_minutes() {
    use super::fmt_uptime;

    assert_eq!(fmt_uptime(0), "up 0:00:00");
    assert_eq!(fmt_uptime(59), "up 0:00:59");
    assert_eq!(fmt_uptime(60), "up 0:01:00");
    assert_eq!(fmt_uptime(100 * 60), "up 1:40:00", "100 minutes is 1h40m");
    assert_eq!(fmt_uptime(3_600), "up 1:00:00");
    assert_eq!(fmt_uptime(86_399), "up 23:59:59");
    assert_eq!(fmt_uptime(86_400), "up 1d 00:00");
    assert_eq!(fmt_uptime(3 * 86_400 + 4 * 3_600 + 12 * 60), "up 3d 04:12");

    let mut seen = std::collections::HashSet::new();
    for m in 0..(48 * 60) {
        assert!(
            seen.insert(fmt_uptime(m * 60)),
            "minute {m} rendered a string already used earlier"
        );
    }
}
