// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for the `dump-serve-options` document: presence-only reporting, value sets, aliases, key spelling and the lever half.
//!
//! Owner: server CLI.
//! Invariants: none beyond the types.

use super::*;

fn flag<'a>(m: &'a Manifest, key: &str) -> &'a Flag {
    m.flags
        .iter()
        .find(|f| f.key == key)
        .unwrap_or_else(|| panic!("{key} is not in the manifest"))
}

/// 2026-09-26: `presence_only` and `is_bool` are reported per flag, checked on the bare
/// toggles and value flags listed here, plus the auto/on/off set of `--tool-grammar`.
#[test]
fn presence_only_is_reported_per_flag_not_guessed() {
    let m = build();
    for key in [
        "video_allow_ffmpeg",
        "gdn_fused_norm",
        "no_ssm_tail_midchunk",
    ] {
        let f = flag(&m, key);
        assert!(f.presence_only && f.is_bool, "{key} is a bare flag");
        assert!(f.options.is_empty(), "{key} offers {:?}", f.options);
    }
    for key in ["ssm_h_dtype", "ssm_batched_recurrent"] {
        assert!(!flag(&m, key).presence_only, "{key} takes a value");
        assert!(!flag(&m, key).is_bool, "{key} is not a bool");
    }
    assert_eq!(
        flag(&m, "tool_grammar").options,
        ["auto", "on", "off"].map(String::from)
    );
}

/// 2026-09-26: Closed value sets come from `cli::flag_values`, the lists
/// `validate_serve_args` checks against.
#[test]
fn enumerated_values_come_from_the_validator_not_from_prose() {
    let m = build();
    assert_eq!(
        flag(&m, "scheduler").options,
        vec!["fifo".to_owned(), "slai".to_owned()]
    );
    assert!(
        flag(&m, "kv_cache_dtype").options.len() >= 16,
        "the KV catalog has 16 variants, the manifest offered {}",
        flag(&m, "kv_cache_dtype").options.len()
    );
    assert!(
        flag(&m, "lm_head_dtype")
            .options
            .contains(&"nvfp4".to_owned()),
        "lm_head_dtype must carry its closed set"
    );
}

/// 2026-09-26: A recipe key spelled differently from its flag travels as a
/// `recipe_aliases` entry on that flag: `max_model_len` on `--max-seq-len` and
/// `tensor_parallel` on `--tp-size`.
#[test]
fn recipe_aliases_travel_with_the_flag_that_owns_them() {
    let m = build();
    let seq = flag(&m, "max_seq_len");
    assert_eq!(seq.flag, "max-seq-len");
    assert!(
        seq.recipe_aliases.contains(&"max_model_len".to_owned()),
        "max_model_len must map to --max-seq-len, got {:?}",
        seq.recipe_aliases
    );

    let tp = flag(&m, "tp_size");
    assert!(tp.recipe_aliases.contains(&"tensor_parallel".to_owned()));

    assert!(flag(&m, "port").recipe_aliases.is_empty());
}

/// 2026-09-26: `cli_aliases` carries clap aliases that `get_long()` does not report:
/// `--bind` has the alias `host`.
#[test]
fn clap_aliases_are_captured_not_just_the_primary_name() {
    let m = build();
    let bind = flag(&m, "bind");
    assert!(
        bind.cli_aliases.contains(&"host".to_owned()),
        "--bind accepts --host; the manifest reported {:?}",
        bind.cli_aliases
    );
    assert!(flag(&m, "port").cli_aliases.is_empty());
}

#[test]
fn flags_are_the_cli_spelling_and_keys_the_underscored_one() {
    let m = build();
    for f in &m.flags {
        assert!(!f.flag.contains('_'), "the flag is hyphenated: {}", f.flag);
        assert!(!f.flag.starts_with('-'), "no leading dashes: {}", f.flag);
        assert_eq!(f.key, f.flag.replace('-', "_"));
    }
}

/// 2026-09-26: clap's own `help` and `version` arguments are not listed as flags.
#[test]
fn clap_own_arguments_are_not_flags_a_recipe_can_set() {
    let m = build();
    for k in ["help", "version"] {
        assert!(
            !m.flags.iter().any(|f| f.key == k),
            "{k} must not appear as a settable flag"
        );
    }
}

/// 2026-09-26: The document carries `SCHEMA_VERSION`, a non-empty engine version and
/// more than 90 flags.
#[test]
fn the_document_declares_its_shape_and_its_engine() {
    let m = build();
    assert_eq!(m.schema_version, SCHEMA_VERSION);
    assert!(!m.engine_version.is_empty());
    assert!(
        m.flags.len() > 90,
        "ServeArgs has ~100 settable flags; the manifest found {}",
        m.flags.len()
    );
}

/// 2026-09-26: No two flags share a key.
#[test]
fn every_key_is_unique() {
    let m = build();
    let mut keys: Vec<&str> = m.flags.iter().map(|f| f.key.as_str()).collect();
    keys.sort_unstable();
    let before = keys.len();
    keys.dedup();
    assert_eq!(before, keys.len(), "duplicate keys in the manifest");
}

/// 2026-09-26: Every lever that names a flag names a serve flag in the document, and the
/// levers listed below are present with class `runtime`.
#[test]
fn every_lever_flag_is_a_serve_flag() {
    let m = build();
    assert!(m.levers.len() > 500, "only {} levers", m.levers.len());
    let with_flag: Vec<&Lever> = m.levers.iter().filter(|l| l.flag.is_some()).collect();
    assert!(
        with_flag.len() >= 10,
        "only {} levers name a flag",
        with_flag.len()
    );
    for l in with_flag {
        let long = l.flag.unwrap_or_default().trim_start_matches("--");
        assert!(
            m.flags.iter().any(|f| f.flag == long),
            "{}: {long} is not a serve flag",
            l.env
        );
    }
    for env in [
        "METRALE_FP8_ROWWISE",
        "METRALE_MTP_DCUT_RATIO",
        "METRALE_MTP_K_LADDER",
    ] {
        assert!(
            m.levers
                .iter()
                .any(|l| l.env == env && l.class == "runtime"),
            "{env} is a recipe-declared lever"
        );
    }
}
