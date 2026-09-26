// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for `flag_values`: every value [`options_for_flag`]
//! offers survives clap parsing and `validate_serve_args`, and a value
//! outside the set is refused; plus the `--kv-high-precision-layers` and
//! tristate parsers.
//!
//! Owner: server CLI (`met serve`).
//! Invariants: none beyond the types.

use super::options_for_flag;
use crate::cli::validate_serve_args;
use clap::Parser;

/// 2026-09-26: The flags with a closed value set. A new `options_for_flag`
/// entry missing here fails the count check below.
const ENUMERATED: &[&str] = &[
    "lm-head-dtype",
    "mtp-quantization",
    "scheduler",
    "scheduler-config",
    "telemetry",
    "ssm-h-dtype",
    "mtp-gate",
    "tool-call-parser",
    "kv-cache-dtype",
    "ssm-batched-recurrent",
    "content-loop-watchdog",
    "tool-grammar",
];

/// 2026-09-26: Flags a value needs beside it to validate: `--ssm-h-dtype f16`
/// without `--gdn-fused-norm` is refused by `validate_serve_args`.
fn companions(flag: &str) -> &'static [&'static str] {
    match flag {
        "ssm-h-dtype" => &["--gdn-fused-norm"],
        _ => &[],
    }
}

fn round_trip(flag: &str, value: &str) -> Result<(), String> {
    let mut argv: Vec<String> = ["met", "serve", "dummy/model", "--model-name", "dummy"]
        .map(String::from)
        .to_vec();
    argv.push(format!("--{flag}"));
    argv.push(value.to_string());
    argv.extend(companions(flag).iter().map(|s| s.to_string()));
    let cli = crate::cli::Cli::try_parse_from(argv).map_err(|e| e.to_string())?;
    let crate::cli::Command::Serve(args) = cli.command else {
        unreachable!("this test parses a serve command");
    };
    validate_serve_args(&args)
}

#[test]
fn every_offered_value_is_accepted_by_parse_and_validate() {
    for flag in ENUMERATED {
        let options = options_for_flag(flag).expect("listed flags have options");
        assert!(!options.is_empty(), "--{flag} offers nothing");
        for value in &options {
            round_trip(flag, value)
                .unwrap_or_else(|e| panic!("--{flag} {value} is offered but refused: {e}"));
        }
    }
}

#[test]
fn a_value_outside_the_set_is_refused_for_every_enumerated_flag() {
    // 2026-09-26: A value outside the set is refused, naming the flag.
    for flag in ENUMERATED {
        let err = round_trip(flag, "zzz-not-a-value")
            .expect_err(&format!("--{flag} accepted a value outside its set"));
        assert!(err.contains(flag), "the refusal names the flag: {err}");
    }
}

#[test]
fn the_enumerated_list_and_the_registry_agree_on_membership() {
    for flag in ENUMERATED {
        assert!(
            options_for_flag(flag).is_some(),
            "--{flag} is tested here but not in options_for_flag"
        );
    }
    // 2026-09-26: `options_for_flag` is a match, not a table, so the reverse
    // direction is checked by count over the TUI's serve fields.
    let known = crate::tui::lib_fields::serve_fields()
        .iter()
        .filter(|s| options_for_flag(&s.flag).is_some())
        .count();
    assert_eq!(
        known,
        ENUMERATED.len(),
        "options_for_flag knows a flag this test does not (or vice versa)"
    );
}

/// 2026-09-26: Pins the forms `KvHighPrecisionLayers` accepts, the one parse
/// `validate_serve_args` and `serve_phases::kv_cache` share.
#[test]
fn kv_high_precision_layers_accepts_exactly_the_documented_forms() {
    use super::KvHighPrecisionLayers as K;
    let p = |s: &str| s.parse::<K>();
    assert_eq!(p("auto"), Ok(K::Auto));
    assert_eq!(p("max"), Ok(K::All));
    assert_eq!(p("all"), Ok(K::All));
    assert_eq!(p("0"), Ok(K::Count(0)));
    assert_eq!(p("6"), Ok(K::Count(6)));
    assert_eq!(p("  AuTo "), Ok(K::Auto));
    assert!(p("atuo").is_err());
    assert!(p("-1").is_err());
    assert!(p("").is_err());
    assert!(p("2.5").is_err());
}

#[test]
fn kv_high_precision_layers_resolves_against_the_attention_layer_count() {
    use super::{AUTO_KV_HIGH_PRECISION_LAYERS, KvHighPrecisionLayers as K};
    assert_eq!(K::All.resolve(48), 48);
    assert_eq!(K::Auto.resolve(48), AUTO_KV_HIGH_PRECISION_LAYERS);
    assert_eq!(K::Count(6).resolve(48), 6);
    // 2026-09-26: `0` stays 0, so `serve_phases::kv_cache` can defer to
    // `auto_high_precision_layers`.
    assert_eq!(K::Count(0).resolve(48), 0);
}

/// 2026-09-26: The error text names the accepted forms.
#[test]
fn the_kv_high_precision_layers_error_names_the_accepted_forms() {
    let e = "atuo".parse::<super::KvHighPrecisionLayers>().unwrap_err();
    assert!(e.contains("auto"), "{e}");
    assert!(e.contains("max"), "{e}");
    assert!(e.contains("all"), "{e}");
}

/// 2026-09-26: The tristate flags take exactly `auto`, `on` and `off`; boolean
/// spellings such as `true` and `false` are refused.
#[test]
fn a_tristate_is_exactly_auto_on_off() {
    use super::{TRISTATES, Tristate};
    let parsed: Vec<Tristate> = TRISTATES.iter().map(|v| v.parse().unwrap()).collect();
    assert_eq!(parsed, [Tristate::Auto, Tristate::On, Tristate::Off]);
    assert_eq!(
        parsed.iter().map(|t| t.pinned()).collect::<Vec<_>>(),
        [None, Some(true), Some(false)]
    );
    for bad in ["true", "false", "1", "0", "yes", "ON", ""] {
        assert!(
            bad.parse::<Tristate>().is_err(),
            "{bad:?} parsed as a tristate"
        );
        assert!(
            round_trip("tool-grammar", bad).is_err(),
            "--tool-grammar {bad:?}"
        );
    }
}
