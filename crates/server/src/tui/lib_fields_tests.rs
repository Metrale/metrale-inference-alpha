// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests that `lib_fields` lists `met serve`'s clap flags under keys `schema::flag_for` renders back.
//!
//! Owner: server tui.
//! Invariants: none beyond the types.

use super::{serve_fields, spec_for_key};

#[test]
fn every_listed_key_renders_back_to_its_own_flag() {
    for spec in serve_fields() {
        assert_eq!(
            crate::recipe::schema::flag_for(&spec.key).as_deref(),
            Some(spec.flag.as_str()),
            "{} does not render to --{}",
            spec.key,
            spec.flag
        );
    }
}

#[test]
fn the_positional_model_and_repeatable_flags_are_not_offered() {
    let keys: Vec<&str> = serve_fields().iter().map(|s| s.key.as_str()).collect();
    assert!(!keys.contains(&"model"), "the recipe owns the model");
    for repeatable in ["lora_adapter", "lora_stageable", "lora_stageable_disk"] {
        assert!(
            !keys.contains(&repeatable),
            "{repeatable} takes several values; a defaults: map holds one"
        );
    }
    assert!(keys.contains(&"kv_cache_dtype"));
    assert!(keys.contains(&"port"));
}

#[test]
fn defaults_are_clap_defaults_not_inventions() {
    assert_eq!(
        spec_for_key("port")
            .expect("port exists")
            .default
            .as_deref(),
        Some("8888")
    );
    assert_eq!(
        spec_for_key("speculative")
            .expect("speculative exists")
            .default
            .as_deref(),
        Some("false")
    );
    assert_eq!(
        spec_for_key("model_name")
            .expect("model_name exists")
            .default,
        None,
        "clap declares no default; the form must ask, not guess"
    );
}

#[test]
fn enumerated_and_boolean_fields_carry_their_option_sets() {
    let kv = spec_for_key("kv_cache_dtype").expect("kv_cache_dtype exists");
    let expected: Vec<String> = metrale_cache::kv_cache::KvCacheDtype::ALL
        .iter()
        .map(|d| d.name().to_string())
        .collect();
    assert_eq!(kv.options, expected, "the picker offers exactly the enum");

    let spec = spec_for_key("speculative").expect("speculative exists");
    assert_eq!(spec.options, ["true", "false"], "booleans are a closed set");

    let free = spec_for_key("max_seq_len").expect("max_seq_len exists");
    assert!(free.options.is_empty(), "numbers stay free text");
}

#[test]
fn recipe_spellings_reach_the_same_spec_as_native_ones() {
    let renamed = spec_for_key("max_model_len").expect("resolves through RENAMES");
    assert_eq!(renamed.flag, "max-seq-len");
    let host = spec_for_key("host").expect("resolves through RENAMES");
    assert_eq!(host.flag, "bind");
}

#[test]
fn the_full_help_is_captured_not_only_its_first_line() {
    let spec = spec_for_key("check_kernels").expect("a serve flag");
    assert!(
        spec.help.starts_with("Resolve all kernels"),
        "the row keeps clap's short help: {:?}",
        spec.help
    );
    assert!(
        !spec.help.contains('\n'),
        "the row's line is one line: {:?}",
        spec.help
    );
    assert!(
        spec.help_full.contains("CLAMPED at 255"),
        "a later paragraph survives into help_full: {:?}",
        spec.help_full
    );
}
