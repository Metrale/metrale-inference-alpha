// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for the header chips `badges` derives from `ServeArgs`, and for the logo rows and colors.
//!
//! Owner: server tui.
//! Invariants: none beyond the types.

use super::*;
use clap::Parser as _;

fn args(extra: &[&str]) -> crate::cli::ServeArgs {
    let mut argv = vec!["met", "org/m"];
    argv.extend_from_slice(extra);
    crate::cli::ServeArgs::parse_from(argv)
}

fn strip(a: &crate::cli::ServeArgs) -> String {
    badges(a, false)
        .iter()
        .map(|b| b.text.clone())
        .collect::<Vec<_>>()
        .join(" | ")
}

#[test]
fn the_model_chip_prefers_the_served_name_over_the_path() {
    // 2026-09-26: `--model-name` first, then `--model`, then a placeholder.
    let mut a = args(&[]);
    a.model = Some("/scratch/checkpoints/run-17".into());
    a.model_name = Some("org/pretty".into());
    let chips = badges(&a, false);
    assert_eq!(chips[0].text, "org/pretty");
    assert_eq!(chips[0].tint, BadgeTint::Model);

    a.model_name = None;
    assert_eq!(badges(&a, false)[0].text, "/scratch/checkpoints/run-17");

    a.model = None;
    assert_eq!(
        badges(&a, false)[0].text,
        "<model>",
        "a placeholder, not a panic"
    );
}

#[test]
fn speculation_is_reported_as_off_rather_than_omitted() {
    // 2026-09-26: With no drafter the strip says "spec off" rather than
    // omitting the chip.
    let a = args(&[]);
    assert!(!a.speculative);
    let chips = badges(&a, false);
    let spec = chips
        .iter()
        .find(|b| b.text.starts_with("spec"))
        .expect("a chip");
    assert_eq!(spec.text, "spec off");
    assert_eq!(spec.tint, BadgeTint::Neutral);
}

#[test]
fn a_speculating_run_reports_the_depth_the_scheduler_will_use() {
    // 2026-09-26: `--num-drafts N` is shown as K = N+1 tokens per verify step.
    let a = args(&["--speculative", "--num-drafts", "3"]);
    let text = strip(&a);
    assert!(text.contains("MTP k=4"), "K is drafts + 1: {text}");
    assert!(!text.contains("spec off"), "{text}");

    for flag in ["--self-speculative", "--ngram-speculative"] {
        let a = args(&[flag]);
        assert!(
            strip(&a).contains("MTP k="),
            "{flag} is speculation too: {}",
            strip(&a)
        );
    }
}

#[test]
fn dflash_displaces_the_mtp_chip_rather_than_sitting_beside_it() {
    // 2026-09-26: `badges` checks `dflash` before the MTP flags, so a DFlash
    // run shows one drafter chip.
    let a = args(&["--dflash"]);
    let text = strip(&a);
    assert!(text.contains("DFlash"), "{text}");
    assert!(!text.contains("MTP k="), "one drafter, one chip: {text}");
    assert!(!text.contains("spec off"), "{text}");
}

#[test]
fn the_role_chip_distinguishes_a_head_from_a_worker() {
    // 2026-09-26: Rank 0 reads "head", any other rank "worker".
    let head = args(&["--world-size", "2"]);
    assert!(strip(&head).contains("head 0/2"), "{}", strip(&head));

    let worker = args(&["--world-size", "2", "--rank", "1"]);
    assert!(strip(&worker).contains("worker 1/2"), "{}", strip(&worker));
}

#[test]
fn the_prefix_cache_chip_appears_only_when_it_is_on() {
    // 2026-09-26: The chip names the SSM slot count, and appears only when
    // prefix caching is on.
    let off = args(&[]);
    assert!(!strip(&off).contains("prefix-cache"), "{}", strip(&off));

    let on = args(&["--enable-prefix-caching", "--ssm-cache-slots", "256"]);
    let text = strip(&on);
    assert!(text.contains("prefix-cache"), "{text}");
    assert!(text.contains("256"), "and the pinned slot count: {text}");
}

#[test]
fn the_context_chip_rounds_only_when_the_rounding_is_exact() {
    // 2026-09-26: Only exact multiples of 1024 are shown as `k`.
    for (tokens, want) in [
        (65536usize, "ctx 64k"),
        (4096, "ctx 4k"),
        (1024, "ctx 1k"),
        (60000, "ctx 60000"),
        (1023, "ctx 1023"),
    ] {
        let mut a = args(&[]);
        a.max_seq_len = tokens;
        assert!(
            strip(&a).contains(want),
            "{tokens} should read {want}: {}",
            strip(&a)
        );
    }
}

#[test]
fn the_quant_chip_names_all_three_dtypes_that_can_differ() {
    // 2026-09-26: KV, LM head and MTP dtypes are separate flags; the chip
    // shows all three.
    let mut a = args(&[]);
    a.kv_cache_dtype = Some("fp8".into());
    a.lm_head_dtype = "bf16".into();
    a.mtp_quantization = "nvfp4".into();
    let chip = badges(&a, false)
        .into_iter()
        .find(|b| b.text.starts_with("kv "))
        .expect("a chip");
    assert_eq!(chip.text, "kv fp8 · lm bf16 · mtp nvfp4");
    assert_eq!(chip.tint, BadgeTint::Quant);
}

#[test]
fn the_awaiting_strip_says_the_way_out_and_the_bound_port() {
    // 2026-09-26: Awaiting a model, the strip is the Library hint and the port
    // only.
    let a = args(&["--port", "9123"]);
    let chips = badges(&a, true);
    assert_eq!(chips.len(), 2, "nothing else may be asserted: {chips:?}");
    assert!(chips[0].text.contains("Library"), "{:?}", chips[0].text);
    assert_eq!(chips[1].text, ":9123");
    assert!(chips.iter().all(|b| b.tint == BadgeTint::Neutral));
}

#[test]
fn the_address_is_the_last_chip_so_it_survives_a_narrow_terminal() {
    // 2026-09-26: The port chip is last.
    let a = args(&["--port", "8871"]);
    let chips = badges(&a, false);
    assert_eq!(chips.last().expect("chips").text, ":8871");
}

#[test]
fn the_three_row_logo_is_exactly_three_rows_of_equal_chevron_width() {
    // 2026-09-26: `render::header` draws the tall logo into three rows.
    let lines = three_line(None);
    assert_eq!(lines.len(), 3);
    for (i, line) in lines.iter().enumerate() {
        let chevrons: Vec<&str> = line
            .spans
            .iter()
            .map(|s| s.content.as_ref())
            .filter(|c| CHEVRON_ROWS.contains(c))
            .collect();
        assert_eq!(chevrons.len(), 3, "row {i} draws three chevrons");
        assert!(chevrons.iter().all(|c| *c == CHEVRON_ROWS[i]));
    }
    let wordmark: String = lines[1].spans.iter().map(|s| s.content.as_ref()).collect();
    assert!(wordmark.contains("M E T R A L E"), "{wordmark}");
}

#[test]
fn the_one_line_logo_is_three_chevrons_and_the_name() {
    let line = one_line(None);
    let text: String = line.spans.iter().map(|s| s.content.as_ref()).collect();
    assert_eq!(text, "❯❯❯ Metrale Engine");
}

/// 2026-09-26: Pin the color mode before reading any color.
///
/// `theme::depth` reads `COLORTERM` on every call, and the sibling test
/// `wave_brightens_one_chevron_at_a_time` sets it to "truecolor". Setting the
/// same value here makes colors read before and after that set agree.
fn pin_truecolor() {
    unsafe { std::env::set_var("COLORTERM", "truecolor") };
}

#[test]
fn a_steady_logo_uses_the_brand_colors_unmodified() {
    // 2026-09-26: `None` is the steady logo; no chevron is dimmed.
    pin_truecolor();
    let want = [
        theme::PURPLE.color(),
        theme::CYAN.color(),
        theme::GREEN.color(),
    ];
    let got: Vec<Color> = three_line(None)[0]
        .spans
        .iter()
        .filter(|s| CHEVRON_ROWS.contains(&s.content.as_ref()))
        .map(|s| s.style.fg.expect("a chevron is colored"))
        .collect();
    assert_eq!(got, want, "purple, cyan, green, left to right");
}

#[test]
fn the_wave_returns_to_where_it_started_every_three_steps() {
    // 2026-09-26: Three chevrons, so step 3 is step 0.
    pin_truecolor();
    for step in 0..3usize {
        assert_eq!(
            format!("{:?}", one_line(Some(step)).spans),
            format!("{:?}", one_line(Some(step + 3)).spans),
            "step {step}"
        );
    }
}

/// 2026-09-26: `--hermetic` gets its own chip and removes the prefix-cache
/// chip.
#[test]
fn the_hermetic_chip_names_the_regime_rather_than_leaving_a_gap() {
    let off = strip(&args(&[]));
    assert!(!off.contains("HERMETIC"), "{off}");

    let on = strip(&args(&["--hermetic"]));
    assert!(on.contains("HERMETIC"), "the regime must be named: {on}");
    assert!(
        !on.contains("prefix-cache"),
        "and the channel it closes must not still be advertised: {on}"
    );
}
