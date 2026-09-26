// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Render tests for the download surfaces: the per-row progress
//! line, the header chip, and the footer's stop hint. The fixtures (`app`,
//! `render`) come from `render_tests.rs`.
//!
//! Owner: server tui.
//! Invariants: none beyond the types.

use crate::tui::app::{App, Section};

use super::tests::{app, render};

/// 2026-09-26: An `App` on the Library whose only row is `model`, so the
/// per-row progress line is drawn.
fn app_with_row(model: &str) -> App {
    let mut a = app();
    a.section = Section::Library;
    a.library = vec![crate::tui::data::library::LibraryEntry {
        id: model.to_string(),
        snapshot_dir: Default::default(),
        size_bytes: 1024,
        has_weights: false,
        model_type: "qwen3_6_moe".into(),
        quant: "nvfp4".into(),
        layers: 40,
        hidden: 4096,
        heads: 32,
        experts: 128,
        context: 65536,
        optimized: false,
    }];
    a.lib.rebuild(&a.library);
    a
}

/// 2026-09-26: Under one percent, the bar still has a filled cell and the
/// percentage has a decimal.
#[test]
fn a_download_under_one_percent_still_looks_alive() {
    let mut a = app_with_row("org/big");
    let root = std::env::temp_dir().join("metrale-render-dl");
    std::fs::create_dir_all(&root).ok();
    a.download.start("org/big", root);
    {
        let job = a.download.job.as_mut().expect("a job");
        job.total = 19_800_000_000;
        job.done = 149_000_000;
        job.rate_bps = 1_600_000.0;
    }
    let out = render(&a, 200, 50);

    // 2026-09-26: 149 MB of 19.8 GB is about 0.75%, and 0.75% of 12 cells
    // rounds to zero; `progress_line` fills one cell anyway.
    assert!(
        out.contains('▓'),
        "some of the bar must be filled once bytes have moved:\n{out}"
    );
    assert!(
        !out.contains("  0%"),
        "sub-1% progress must not render as a bare 0%:\n{out}"
    );
    assert!(out.contains("0.8%"), "one decimal below 10%:\n{out}");
    // 2026-09-26: The rate fits in the list pane at this width.
    assert!(out.contains("MB/s"), "the rate must be visible:\n{out}");
}

#[test]
fn a_download_at_exactly_zero_bytes_shows_an_empty_bar_honestly() {
    // 2026-09-26: The one-cell minimum applies only once a byte has moved.
    let mut a = app_with_row("org/big");
    let root = std::env::temp_dir().join("metrale-render-dl0");
    std::fs::create_dir_all(&root).ok();
    a.download.start("org/big", root);
    {
        let job = a.download.job.as_mut().expect("a job");
        job.total = 19_800_000_000;
        job.done = 0;
    }
    let out = render(&a, 200, 50);
    assert!(out.contains('░'), "the empty track is drawn:\n{out}");
    assert!(out.contains("0.0%"));
}

/// 2026-09-26: The header chip shows the download from every section.
#[test]
fn a_download_is_visible_from_every_section() {
    for section in crate::tui::section::Section::ALL {
        let mut a = app_with_row("nvidia/Qwen3-80B-NVFP4");
        a.section = section;
        let root = std::env::temp_dir().join("metrale-render-chip");
        std::fs::create_dir_all(&root).ok();
        a.download.start("nvidia/Qwen3-80B-NVFP4", root);
        {
            let job = a.download.job.as_mut().expect("a job");
            job.total = 19_800_000_000;
            job.done = 8_300_000_000;
            job.rate_bps = 96_000_000.0;
        }
        let out = render(&a, 120, 32);
        assert!(
            out.contains("42%"),
            "{section:?}: the chip must carry the percentage:\n{out}"
        );
        assert!(
            out.contains("Qwen3-80B-NVFP4"),
            "{section:?}: the chip must name the model:\n{out}"
        );
    }
}

/// 2026-09-26: With no total the chip shows bytes moved, not a percentage.
#[test]
fn an_unknown_total_shows_bytes_not_a_fake_percentage() {
    let mut a = app_with_row("org/big");
    a.section = crate::tui::section::Section::Stats;
    let root = std::env::temp_dir().join("metrale-render-chip2");
    std::fs::create_dir_all(&root).ok();
    a.download.start("org/big", root);
    {
        let job = a.download.job.as_mut().expect("a job");
        job.total = 0;
        job.done = 3_400_000_000;
    }
    let out = render(&a, 120, 32);
    assert!(out.contains("3.2 GB"), "bytes moved:\n{out}");
    assert!(!out.contains("0%"), "never a fabricated percentage:\n{out}");
}

/// 2026-09-26: A cancelling download's chip says "stopping".
#[test]
fn a_cancelling_download_says_stopping_in_the_chip() {
    let mut a = app_with_row("org/big");
    a.section = crate::tui::section::Section::Network;
    let root = std::env::temp_dir().join("metrale-render-chip3");
    std::fs::create_dir_all(&root).ok();
    a.download.start("org/big", root);
    {
        let job = a.download.job.as_mut().expect("a job");
        job.total = 1_000;
        job.done = 500;
        job.cancelling = true;
    }
    let out = render(&a, 120, 32);
    assert!(out.contains("stopping"), "the chip must say so:\n{out}");
}

/// 2026-09-26: With no download job the Library footer has no `x stop`.
#[test]
fn no_download_means_no_chip_and_no_stop_hint() {
    let mut a = app_with_row("org/big");
    a.section = crate::tui::section::Section::Library;
    let out = render(&a, 120, 32);
    assert!(
        !out.contains("x stop"),
        "the stop hint is a false claim with nothing to stop:\n{out}"
    );
}
