// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for the recipe key → `met serve` flag mapping.
//!
//! Owner: server (recipe).
//! Invariants: none beyond the types.

use super::*;

#[test]
fn most_keys_are_the_flag_with_dashes() {
    assert_eq!(
        flag_for("gpu_memory_utilization").as_deref(),
        Some("gpu-memory-utilization")
    );
    assert_eq!(flag_for("port").as_deref(), Some("port"));
}

#[test]
fn the_three_renames_are_applied() {
    // 2026-09-26: `ServeArgs` has no `max_model_len` or `tensor_parallel` field, so
    // clap would reject either key unrenamed.
    assert_eq!(flag_for("max_model_len").as_deref(), Some("max-seq-len"));
    assert_eq!(flag_for("tensor_parallel").as_deref(), Some("tp-size"));
    assert_eq!(flag_for("host").as_deref(), Some("bind"));
}

#[test]
fn a_presence_only_boolean_is_a_bare_flag_and_a_false_one_is_omitted() {
    // 2026-09-26: `--speculative` is a clap SetTrue flag: it takes no value, so
    // `false` can only be expressed by not passing it. Only an override may say
    // `false`; `check_recipe_default` refuses it in a recipe's own `defaults:`.
    assert_eq!(
        argv_for("speculative", "true"),
        Some(vec!["--speculative".to_string()])
    );
    assert_eq!(argv_for("speculative", "false"), None);
    assert_eq!(argv_for("enable_prefix_caching", "false"), None);
    assert_eq!(
        argv_for("video_allow_ffmpeg", "true"),
        Some(vec!["--video-allow-ffmpeg".to_string()])
    );
}

#[test]
fn an_enum_keeps_its_value_and_a_boolean_never_carries_one() {
    // 2026-09-26: `--tool-grammar` is an auto/on/off enum: MODEL.toml turns the
    // grammar off for some models, so the command line pins it either way.
    assert_eq!(
        argv_for("tool_grammar", "on"),
        Some(vec!["--tool-grammar".into(), "on".into()])
    );
    // 2026-09-26: A default-on behaviour's flag is named for its off state
    // (`--no-ssm-tail-midchunk`).
    assert_eq!(argv_for("gdn_fused_norm", "false"), None);
    assert_eq!(
        argv_for("no_ssm_tail_midchunk", "true"),
        Some(vec!["--no-ssm-tail-midchunk".into()])
    );
}

#[test]
fn a_recipe_default_may_only_turn_a_boolean_on() {
    assert_eq!(check_recipe_default("speculative", "true"), Ok(()));
    let err = check_recipe_default("speculative", "false").unwrap_err();
    assert!(err.contains("restates the default"), "{err}");
    assert!(err.contains("--speculative"), "names the flag: {err}");
    let err = check_recipe_default("no_ssm_tail_midchunk", "yes").unwrap_err();
    assert!(err.contains("takes no value"), "{err}");
    // 2026-09-26: Value flags and enums pass, whatever their value.
    assert_eq!(check_recipe_default("tool_grammar", "auto"), Ok(()));
    assert_eq!(check_recipe_default("scheduler", "fifo"), Ok(()));
    assert_eq!(check_recipe_default("max_model_len", "false"), Ok(()));
}

#[test]
fn values_pass_through_verbatim() {
    assert_eq!(
        argv_for("max_model_len", "65536"),
        Some(vec!["--max-seq-len".into(), "65536".into()])
    );
}
