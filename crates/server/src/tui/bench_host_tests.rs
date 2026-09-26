// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for the serve-matrix roster (`classify`, `roster`) and the round argv. No GPU.
//!
//! Owner: server tui.
//! Invariants: none beyond the types.

use super::*;
use crate::tui::data::library::LibraryEntry;

fn entry(id: &str, has_weights: bool, optimized: bool) -> LibraryEntry {
    LibraryEntry {
        id: id.into(),
        snapshot_dir: std::path::PathBuf::from("/nonexistent"),
        size_bytes: 0,
        has_weights,
        model_type: "qwen3_6_moe".into(),
        quant: "nvfp4".into(),
        layers: 40,
        hidden: 2048,
        heads: 16,
        experts: 256,
        context: 262_144,
        optimized,
    }
}

#[test]
fn a_complete_supported_checkpoint_is_runnable() {
    assert_eq!(classify(&entry("org/m", true, true)), None);
}

#[test]
fn a_half_downloaded_checkpoint_is_absent_for_weights_not_for_kernels() {
    assert_eq!(
        classify(&entry("org/m", false, false)),
        Some(Absence::NoWeights)
    );
}

#[test]
fn a_downloaded_model_this_build_has_no_kernels_for_is_skipped_with_that_reason() {
    assert_eq!(
        classify(&entry("org/m", true, false)),
        Some(Absence::NoKernels)
    );
    assert!(Absence::NoKernels.reason().contains("kernels"));
}

#[test]
fn an_unreadable_config_is_not_reported_as_an_unsupported_architecture() {
    // 2026-09-26: An empty `model_type` is classified `NoConfig`. `library::scan` writes "?" for an
    // unparseable config, so this case is a parsed config without a `model_type`.
    let mut e = entry("org/m", true, false);
    e.model_type = String::new();
    assert_eq!(classify(&e), Some(Absence::NoConfig));
    assert!(Absence::NoConfig.reason().contains("unreadable"));
}

fn bound_host() -> Arc<ModelHost> {
    let host = Arc::new(ModelHost::empty());
    host.set_bound("127.0.0.1".into(), 8899);
    host
}

#[test]
fn a_round_is_served_on_the_port_this_server_already_bound() {
    // 2026-09-26: The round uses the port this server already bound.
    let h = TuiServeHost::new(bound_host(), None);
    let args = h
        .argv_for(
            "org/m",
            ServeOptions {
                max_seq_len: 32_768,
                speculative: false,
            },
        )
        .expect("a valid command line");
    assert_eq!(args.port, 8899);
    assert_eq!(args.model.as_deref(), Some("org/m"));
    assert_eq!(args.max_seq_len, 32_768);
    assert!(!args.speculative);
    assert_eq!(
        h.endpoint("org/m").expect("bound").base_url,
        "http://127.0.0.1:8899"
    );
}

#[test]
fn the_mtp_arm_is_the_only_thing_the_speculative_option_changes() {
    let h = TuiServeHost::new(bound_host(), None);
    let opts = |speculative| ServeOptions {
        max_seq_len: 16_384,
        speculative,
    };
    let off = h.argv_for("org/m", opts(false)).expect("valid");
    let on = h.argv_for("org/m", opts(true)).expect("valid");
    assert!(!off.speculative && on.speculative);
    assert_eq!(off.max_seq_len, on.max_seq_len);
    assert_eq!(off.port, on.port);
}

#[test]
fn a_round_cannot_be_built_before_the_server_has_bound() {
    let h = TuiServeHost::new(Arc::new(ModelHost::empty()), None);
    let err = h
        .argv_for(
            "org/m",
            ServeOptions {
                max_seq_len: 4096,
                speculative: false,
            },
        )
        .unwrap_err()
        .to_string();
    assert!(err.contains("binding its port"), "{err}");
}

#[test]
fn a_cache_directory_with_nothing_in_it_yields_an_empty_roster() {
    // 2026-09-26: The roster is derived from the cache scan, so an empty cache gives an empty roster.
    let dir = tempfile::tempdir().expect("scratch dir");
    let h = TuiServeHost::new(bound_host(), Some(dir.path().to_path_buf()));
    assert!(h.roster().expect("a scan of an empty cache").is_empty());
}

#[test]
fn a_half_downloaded_checkpoint_reaches_the_roster_as_a_named_absence() {
    // 2026-09-26: A skipped checkpoint stays in the roster with its reason.
    let dir = tempfile::tempdir().expect("scratch dir");
    let snapshot = dir
        .path()
        .join("models--org--half")
        .join("snapshots")
        .join("deadbeef");
    std::fs::create_dir_all(&snapshot).expect("mock cache");
    std::fs::write(snapshot.join("config.json"), "{}").expect("config");

    let h = TuiServeHost::new(bound_host(), Some(dir.path().to_path_buf()));
    let roster = h.roster().expect("a scan");
    assert_eq!(roster.len(), 1, "{roster:?}");
    assert_eq!(roster[0].model, "org/half");
    assert_eq!(roster[0].absent, Some(Absence::NoWeights));
}

#[test]
fn the_cache_override_is_carried_into_every_rounds_command_line() {
    let dir = tempfile::tempdir().expect("scratch dir");
    let h = TuiServeHost::new(bound_host(), Some(dir.path().to_path_buf()));
    let args = h
        .argv_for(
            "org/m",
            ServeOptions {
                max_seq_len: 4096,
                speculative: false,
            },
        )
        .expect("valid");
    assert_eq!(args.cache_dir.as_deref(), Some(dir.path()));
}

#[test]
fn a_scanned_id_reaches_the_round_argv_exactly_as_the_cache_spells_it() {
    let h = TuiServeHost::new(bound_host(), None);
    for id in [
        "unsloth/Qwen3.6-27B-NVFP4",
        "nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4",
        "org/name.with.dots",
    ] {
        let args = h
            .argv_for(
                id,
                ServeOptions {
                    max_seq_len: 4096,
                    speculative: false,
                },
            )
            .expect("valid");
        assert_eq!(args.model.as_deref(), Some(id));
    }
}

#[test]
fn restoring_a_box_that_was_serving_nothing_is_a_no_op_not_a_teardown() {
    // 2026-09-26: With `original` unset, `restore` returns before any swap, so nothing touches the GPU.
    let h = TuiServeHost::new(bound_host(), None);
    let rt = tokio::runtime::Runtime::new().expect("runtime");
    rt.block_on(ServeHost::restore(&h))
        .expect("nothing to put back");
}
