// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for `validate_serve_args`; `validate_tests_b.rs` holds the rest.
//!
//! Owner: server CLI.
//! Invariants: none beyond the types.
use super::*;
use clap::Parser;

fn parse(extra: &[&str]) -> ServeArgs {
    let mut argv = vec!["met", "serve", "dummy/model", "--model-name", "dummy"];
    argv.extend_from_slice(extra);
    match super::super::Cli::parse_from(argv).command {
        super::super::Command::Serve(a) => a,
        super::super::Command::Benchmark(_)
        | super::super::Command::DumpServeOptions
        | super::super::Command::SyncRecipes
        | super::super::Command::Doctor => {
            unreachable!("this test parses a serve command")
        }
    }
}

#[test]
fn defaults_are_valid() {
    assert!(validate_serve_args(&parse(&[])).is_ok());
}

#[test]
fn warmup_prompt_is_refused_because_nothing_implements_it() {
    let err = validate_serve_args(&parse(&["--warmup-prompt", "/tmp/warm.txt"]))
        .expect_err("an inert flag must not be accepted in silence");
    assert!(err.contains("--warmup-prompt"), "{err}");
    assert!(
        err.contains("not implemented"),
        "the operator must be told the flag does nothing, not merely that it is \
         disallowed: {err}"
    );
    assert!(
        err.contains("fix:"),
        "a diagnostic without a fix is half of one: {err}"
    );
    assert!(
        err.contains("throwaway request"),
        "must name the way to warm the server: {err}"
    );
}

#[test]
fn omitting_warmup_prompt_is_fine() {
    assert!(parse(&[]).warmup_prompt.is_none());
    assert!(validate_serve_args(&parse(&[])).is_ok());
}

#[test]
fn kv_high_precision_layers_typo_is_refused_before_the_weight_load() {
    let err = validate_serve_args(&parse(&["--kv-high-precision-layers", "atuo"]))
        .expect_err("a typo must be refused, not silently resolved to 0");
    assert!(err.contains("--kv-high-precision-layers"), "{err}");
    assert!(err.contains("atuo"), "must quote the rejected value: {err}");
    assert!(err.contains("fix:"), "{err}");
    assert!(
        err.contains("auto") && err.contains("max"),
        "must name the accepted keywords: {err}"
    );
}

#[test]
fn every_documented_kv_high_precision_layers_form_is_accepted() {
    for form in ["0", "2", "64", "auto", "max", "all", "AUTO", "Max"] {
        assert!(
            validate_serve_args(&parse(&["--kv-high-precision-layers", form])).is_ok(),
            "{form} is documented as valid but was refused"
        );
    }
}

#[test]
fn fp8_calibration_requires_fp8_kv() {
    let err = validate_serve_args(&parse(&[
        "--kv-cache-dtype",
        "bf16",
        "--fp8-kv-calibration-tokens",
        "256",
    ]))
    .unwrap_err();
    assert!(err.contains("--fp8-kv-calibration-tokens"));
    assert!(err.contains("fix:"));
    assert!(
        validate_serve_args(&parse(&[
            "--kv-cache-dtype",
            "fp8",
            "--fp8-kv-calibration-tokens",
            "256",
        ]))
        .is_ok()
    );
}

#[test]
fn an_absent_lever_flag_parses_as_unspecified() {
    // 2026-09-26: An absent flag publishes nothing (`KernelFlagPlan`, and
    // `mtp_gate_force` for `--mtp-gate`), which leaves the named `METRALE_*`
    // variable in charge.
    let a = parse(&[]);
    assert!(!a.no_ssm_tail_midchunk, "METRALE_SSM_TAIL_MIDCHUNK");
    assert!(a.mtp_gate.is_none(), "METRALE_MTP_GATE_FORCE");
    assert!(a.ssm_h_dtype.is_none(), "METRALE_SSM_H_FP16");
    assert!(!a.gdn_fused_norm, "METRALE_GDN_FUSED_NORM");
    assert_eq!(
        a.ssm_batched_recurrent, "auto",
        "METRALE_SSM_BATCHED_RECURRENT"
    );
    assert!(!a.exact_verify, "--exact-verify");
    assert!(!a.prefill_varlen_batch, "METRALE_PREFILL_VARLEN");
    assert!(!a.prefill_codispatch, "METRALE_PREFILL_CODISPATCH");

    let a = parse(&["--no-ssm-tail-midchunk", "--mtp-gate", "force"]);
    assert!(a.no_ssm_tail_midchunk, "given, it still wins");
    assert_eq!(a.mtp_gate.as_deref(), Some("force"));
}

#[test]
fn the_bare_gdn_switches_still_mean_on() {
    let a = parse(&[
        "--gdn-fused-norm",
        "--exact-verify",
        "--prefill-varlen-batch",
        "--prefill-codispatch",
    ]);
    assert!(a.gdn_fused_norm && a.exact_verify);
    assert!(a.prefill_varlen_batch && a.prefill_codispatch);
}

#[test]
fn exact_verify_refuses_the_f16_h_state() {
    let err = validate_serve_args(&parse(&[
        "--exact-verify",
        "--ssm-h-dtype",
        "f16",
        "--gdn-fused-norm",
    ]))
    .unwrap_err();
    assert!(err.contains("--exact-verify"), "{err}");
    assert!(validate_serve_args(&parse(&["--exact-verify"])).is_ok());
    assert!(validate_serve_args(&parse(&["--ssm-h-dtype", "f16", "--gdn-fused-norm"])).is_ok());
}

#[test]
fn f16_h_state_still_needs_the_fused_norm_arm() {
    assert!(validate_serve_args(&parse(&["--ssm-h-dtype", "f16"])).is_err());
    assert!(
        validate_serve_args(&parse(&["--ssm-h-dtype", "f16", "--gdn-fused-norm"])).is_ok(),
        "the supported pairing"
    );
}

#[test]
fn ssm_rollback_mode_values_and_typos() {
    let a = parse(&[]);
    assert_eq!(a.ssm_rollback_mode, "snapshot");
    assert!(validate_serve_args(&a).is_ok());
    assert!(validate_serve_args(&parse(&["--ssm-rollback-mode", "replay"])).is_ok());
    let err = validate_serve_args(&parse(&["--ssm-rollback-mode", "Replay"])).unwrap_err();
    assert!(err.contains("--ssm-rollback-mode"), "{err}");
    assert!(err.contains("snapshot"), "{err}");
}

#[test]
fn ssm_decode_ring_slots_values_and_typos() {
    let a = parse(&[]);
    assert_eq!(a.ssm_decode_ring_slots, "auto");
    assert!(validate_serve_args(&a).is_ok());
    for depth in ["0", "1", "2", "4", "8"] {
        assert!(
            validate_serve_args(&parse(&["--ssm-decode-ring-slots", depth])).is_ok(),
            "depth {depth} is inside the ring's range"
        );
    }
    let err = validate_serve_args(&parse(&["--ssm-decode-ring-slots", "9"])).unwrap_err();
    assert!(err.contains("--ssm-decode-ring-slots"), "{err}");
    let err = validate_serve_args(&parse(&["--ssm-decode-ring-slots", "AUTO"])).unwrap_err();
    assert!(err.contains("auto"), "names the valid values: {err}");
}

#[test]
fn a_mistyped_mtp_gate_is_still_caught() {
    let err = validate_serve_args(&parse(&["--mtp-gate", "always"])).unwrap_err();
    assert!(err.contains("--mtp-gate"), "{err}");
    assert!(err.contains("auto, force"), "names the valid values: {err}");
}

#[test]
fn require_auth_needs_a_token() {
    assert!(validate_serve_args(&parse(&["--require-auth"])).is_err());
    assert!(validate_serve_args(&parse(&["--require-auth", "--auth-token", "sk-x"])).is_ok());
}

#[test]
fn num_drafts_needs_speculative() {
    assert!(validate_serve_args(&parse(&["--num-drafts", "2"])).is_err());
    assert!(validate_serve_args(&parse(&["--num-drafts", "2", "--speculative"])).is_ok());
}

#[test]
fn rank_must_be_below_world_size() {
    assert!(validate_serve_args(&parse(&["--rank", "2", "--world-size", "2"])).is_err());
    assert!(validate_serve_args(&parse(&["--rank", "1", "--world-size", "2"])).is_ok());
}

#[test]
fn ep_size_cannot_exceed_world_size() {
    assert!(validate_serve_args(&parse(&["--ep-size", "2"])).is_err());
    assert!(validate_serve_args(&parse(&["--ep-size", "2", "--world-size", "2"])).is_ok());
}

#[test]
fn disable_thinking_conflicts_with_budget() {
    assert!(
        validate_serve_args(&parse(&[
            "--disable-thinking",
            "--max-thinking-budget",
            "2048"
        ]))
        .is_err()
    );
}
