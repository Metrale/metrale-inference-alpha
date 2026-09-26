// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests for the target-defaults resolver, on hand-written copies
//! of the gb10 and Hopper declarations: an empty environment resolves to the
//! declaration with every value sourced from the target, and each lever is
//! overridable from the environment and says so.
//!
//! This file tests the resolver. `crates/kernels/tests/target_defaults.rs`
//! parses the real `kernels/*/HARDWARE.toml` with the build-script parser.
//!
//! Owner: model-layers ops (target serving defaults).
//! Invariants: none beyond the types.

use super::*;
use metrale_kernels::TargetDefaults;

/// 2026-09-25: A copy of `kernels/gb10/HARDWARE.toml` `[defaults]`. Apart from
/// the two `w8a8_prefill_max_m_*` rows it equals `build_defaults::baseline`,
/// which `gb10_declares_the_baseline_apart_from_the_measured_w8a8_ceiling`
/// (`crates/kernels/tests/target_defaults.rs`) checks on the real file.
const GB10: TargetDefaults = TargetDefaults {
    hw: "gb10",
    lm_head_batchm_max: 8,
    ssm_batched_recurrent: false,
    gdn_prefill_tc: false,
    ssm_ba_gates_hopper: false,
    fp8_act_quant_hopper: false,
    decode_split_silu: true,
    attn_decode_splitk: "legacy",
    ffn_m16_tc: false,
    attn_m16_tc: false,
    lm_head_m16_tc: false,
    attn_ncol_gemv: false,
    ffn_gateup_fused: false,
    w8a8_prefill_max_m_widening: 64,
    w8a8_prefill_max_m_narrowing: 384,
};

/// 2026-09-25: A copy of `kernels/hopper/HARDWARE.toml` `[defaults]`.
const HOPPER: TargetDefaults = TargetDefaults {
    hw: "hopper",
    lm_head_batchm_max: 16,
    ssm_batched_recurrent: true,
    gdn_prefill_tc: true,
    ssm_ba_gates_hopper: true,
    fp8_act_quant_hopper: true,
    decode_split_silu: true,
    attn_decode_splitk: "auto",
    ffn_m16_tc: false,
    attn_m16_tc: true,
    lm_head_m16_tc: true,
    attn_ncol_gemv: false,
    ffn_gateup_fused: true,
    w8a8_prefill_max_m_widening: u32::MAX,
    w8a8_prefill_max_m_narrowing: u32::MAX,
};

fn with(defaults: &TargetDefaults, env: &[(&str, &str)]) -> TargetLevers {
    let env: Vec<(String, String)> = env
        .iter()
        .map(|(k, v)| ((*k).to_owned(), (*v).to_owned()))
        .collect();
    resolve(defaults, |name| {
        env.iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.to_owned())
    })
}

fn empty(defaults: &TargetDefaults) -> TargetLevers {
    with(defaults, &[])
}

/// 2026-09-25: With no `METRALE_*` set, the Hopper table resolves to its
/// declaration, and the batched-recurrent row reports the target as its source.
#[test]
fn hopper_resolves_its_recipe_from_an_empty_environment() {
    let l = empty(&HOPPER);
    assert!(
        l.ssm_batched_recurrent.value,
        "+6% on the serve, md5-identical output"
    );
    assert_eq!(
        l.ssm_batched_recurrent.source,
        Source::Target,
        "with an empty environment every value must be attributed to the \
         TARGET — an ` (env)` tag here would mean the log credits a prefix \
         nobody typed"
    );
    assert!(
        l.gdn_prefill_tc.value,
        "round 13: the tensor-core GDN prefill family is the H100 default — \
         C=1 TTFT -39.6%/-44.7%, C=16 aggregate +21.5%/+31.4%, coherency 4/4, \
         determinism 8/8 x 3"
    );
    assert!(
        l.ssm_ba_gates_hopper.value,
        "the BA-gates twin is bit-identical to its parent, so it ships on: its \
         worst case is a null and off it re-reads every activation row 96 \
         times, once per BA output"
    );
    assert!(l.decode_split_silu.value);
    assert!(
        l.ssm_ba_gates_hopper.value,
        "round 14: the BA-gates twin is bit-identical to its parent, so it is \
         on without an accuracy receipt and its worst case is a null"
    );
    assert_eq!(l.lm_head_batchm_max.value, 16);
    assert_eq!(l.hw, "hopper");
}

/// 2026-09-25: With no `METRALE_*` set, the gb10 table resolves to its
/// declaration, and the listed rows report the target as their source.
#[test]
fn gb10_with_an_empty_environment_is_todays_behaviour() {
    let l = empty(&GB10);
    assert_eq!(l.lm_head_batchm_max.value, DENSE_GEMV_BATCHM_DECODE_MAX_M);
    assert!(!l.ssm_batched_recurrent.value);
    assert!(
        !l.gdn_prefill_tc.value,
        "the scalar GDN prefill spine stays GB10's default. Round 13 promoted \
         the tensor-core family on HOPPER, on an H100 receipt; a 48-SM GB10 is \
         the part the 48-CTA grid nearly fills, so that number does not \
         transfer by argument and this row waits for a GB10 A/B"
    );
    assert!(
        !l.ssm_ba_gates_hopper.value,
        "GB10 does not compile the twin at all — the row is declared so the \
         lever list is one list, not to change anything"
    );
    assert!(l.decode_split_silu.value);
    // 2026-09-25: The two rows where gb10 differs from the baseline.
    assert_eq!(l.w8a8_prefill_max_m_widening.value, 64);
    assert_eq!(l.w8a8_prefill_max_m_narrowing.value, 384);
    for source in [
        l.lm_head_batchm_max.source,
        l.ssm_batched_recurrent.source,
        l.decode_split_silu.source,
        l.w8a8_prefill_max_m_widening.source,
        l.w8a8_prefill_max_m_narrowing.source,
    ] {
        assert_eq!(source, Source::Target);
    }
}

/// 2026-09-25: [`BASELINE_BATCHM_MAX`] is `DENSE_GEMV_BATCHM_DECODE_MAX_M` and the
/// gb10 copy's band. `build_defaults::baseline` repeats the value as a literal
/// because `metrale-kernels` cannot depend on this crate.
#[test]
fn the_baseline_band_is_the_frozen_one() {
    assert_eq!(BASELINE_BATCHM_MAX, DENSE_GEMV_BATCHM_DECODE_MAX_M);
    assert_eq!(
        BASELINE_BATCHM_MAX, GB10.lm_head_batchm_max,
        "kernels/gb10 declares the frozen band; if this fails one of the two \
         moved without the other"
    );
}

/// 2026-09-25: A lever the target declares on is turned off by each off
/// spelling, and reports the environment as its source.
#[test]
fn a_declared_on_lever_can_be_turned_off_by_the_environment() {
    for off in ["0", "false", "off", "no", "OFF", " 0 "] {
        let l = with(&HOPPER, &[("METRALE_SSM_BATCHED_RECURRENT", off)]);
        assert!(!l.ssm_batched_recurrent.value, "`{off}` must read as off");
        assert!(l.ssm_batched_recurrent.from_env());
    }
}

/// 2026-09-25: A lever the target declares off is turned on by `=1`.
#[test]
fn a_declared_off_lever_is_still_armed_by_the_bare_one() {
    let l = with(&GB10, &[("METRALE_SSM_BATCHED_RECURRENT", "1")]);
    assert!(l.ssm_batched_recurrent.value);
    assert!(l.ssm_batched_recurrent.from_env());
}

/// 2026-09-25: `METRALE_GDN_PREFILL_TC=0` (and every other off spelling) turns
/// the tensor-core GDN prefill off rather than arming it by being present; `=1`
/// arms it. The remnant twins read the same resolved bit
/// (`ssm_gdn_remnants_tests::the_twins_read_the_spines_resolved_lever`).
#[test]
fn the_tensor_core_prefill_spine_reads_zero_as_off_not_as_present() {
    for off in ["0", "false", "off", "no", "OFF", " 0 "] {
        let l = with(&GB10, &[("METRALE_GDN_PREFILL_TC", off)]);
        assert!(
            !l.gdn_prefill_tc.value,
            "`{off}` must disarm the spine, not arm it by being present"
        );
        assert!(l.gdn_prefill_tc.from_env());
    }
    let on = with(&GB10, &[("METRALE_GDN_PREFILL_TC", "1")]);
    assert!(on.gdn_prefill_tc.value);
    assert!(on.gdn_prefill_tc.from_env());
}

/// 2026-09-25: `METRALE_NO_DECODE_SPLIT_SILU` is presence-gated: any value,
/// including `0` and empty, turns the lever off over the declaration.
#[test]
fn the_legacy_kill_switch_still_forces_the_lever_off() {
    for value in ["1", "0", ""] {
        let l = with(&HOPPER, &[("METRALE_NO_DECODE_SPLIT_SILU", value)]);
        assert!(
            !l.decode_split_silu.value,
            "METRALE_NO_DECODE_SPLIT_SILU={value:?} is PRESENCE-gated and must \
             force the lever off whatever it is set to"
        );
        assert!(l.decode_split_silu.from_env());
    }
}

/// 2026-09-25: The head band has no "off": `0`, garbage and empty keep the
/// target's declaration, and every value, declared or overridden, is clamped to
/// `DENSE_GEMV_BATCHM_MAX_M`, above which `dense_gemv_batchm` returns an error.
#[test]
fn the_head_band_clamps_and_has_no_off() {
    assert_eq!(
        resolve_batchm_max(8, Some("64")).value,
        DENSE_GEMV_BATCHM_MAX_M,
        "clamped to the kernel's row bound, not passed through"
    );
    assert_eq!(resolve_batchm_max(8, Some("12")).value, 12);
    for keep in [Some("0"), Some("banana"), Some(""), None] {
        assert_eq!(
            resolve_batchm_max(8, keep).value,
            8,
            "{keep:?} must keep the target's declaration"
        );
        assert_eq!(resolve_batchm_max(8, keep).source, Source::Target);
    }
    // 2026-09-25: A declaration above the kernel bound is clamped too.
    assert_eq!(
        resolve_batchm_max(64, None).value,
        DENSE_GEMV_BATCHM_MAX_M,
        "the clamp is on the resolved value, whichever rung it came from"
    );
}

/// 2026-09-25: The boot line names each lever with its value and tags an
/// environment-sourced one. Tested through [`format_levers`], because
/// [`summary_line`] caches the real environment's resolution for the whole
/// test binary.
#[test]
fn the_summary_line_names_every_lever_and_flags_the_environment() {
    let line = format_levers(&with(&HOPPER, &[("METRALE_LM_HEAD_BATCHM_MAX", "12")]));
    assert!(line.starts_with("target defaults (hopper): "), "{line}");
    for field in [
        "sm_count=",
        "lm_head_batchm_max=12 (env)",
        "ssm_batched_recurrent=on",
        "gdn_prefill_tc=on",
        "ssm_ba_gates_hopper=on",
        "fp8_act_quant_hopper=on",
        "decode_split_silu=on",
        "attn_decode_splitk=auto",
        "ffn_gateup_fused=on",
        "w8a8_prefill_max_m=max/max",
    ] {
        assert!(line.contains(field), "missing `{field}` in:\n{line}");
    }
    let clean = format_levers(&empty(&HOPPER));
    assert!(!clean.contains("(env)"), "{clean}");
    assert!(
        format_levers(&empty(&GB10)).contains("attn_decode_splitk=legacy"),
        "{}",
        format_levers(&empty(&GB10))
    );
}

/// 2026-09-25: The split-K policy row: declaration first, environment second.
///
/// `METRALE_ATTN_DECODE_SPLITK=0` resolves to `Pinned(1)` (one split), so the
/// line reads `attn_decode_splitk=1 (env)`: the label is the resolved policy,
/// not the string that was typed.
#[test]
fn the_split_k_policy_resolves_and_reports_like_every_other_lever() {
    assert_eq!(
        empty(&HOPPER).attn_decode_splitk.value,
        SplitkPolicy::Auto,
        "an H100 serve with an empty environment must reach the split count \
         that fills 132 SMs — the whole content of #928"
    );
    assert_eq!(
        empty(&GB10).attn_decode_splitk.value,
        SplitkPolicy::Legacy,
        "GB10 is unchanged"
    );
    assert_eq!(empty(&HOPPER).attn_decode_splitk.source, Source::Target);

    let off = with(&HOPPER, &[("METRALE_ATTN_DECODE_SPLITK", "0")]);
    assert_eq!(off.attn_decode_splitk.value, SplitkPolicy::Pinned(1));
    assert_eq!(off.attn_decode_splitk.source, Source::Env);
    assert!(
        format_levers(&off).contains("attn_decode_splitk=1 (env)"),
        "{}",
        format_levers(&off)
    );

    let on = with(&GB10, &[("METRALE_ATTN_DECODE_SPLITK", "auto")]);
    assert_eq!(on.attn_decode_splitk.value, SplitkPolicy::Auto);
    assert!(
        format_levers(&on).contains("attn_decode_splitk=auto (env)"),
        "{}",
        format_levers(&on)
    );

    // 2026-09-25: A misspelling keeps the declaration and its target source.
    let typo = with(&HOPPER, &[("METRALE_ATTN_DECODE_SPLITK", "atuo")]);
    assert_eq!(typo.attn_decode_splitk.value, SplitkPolicy::Auto);
    assert_eq!(typo.attn_decode_splitk.source, Source::Target);
}

// 2026-09-25: The `fp8_act_quant_hopper` row's tests, as a child module so they
// share the fixtures above.
#[path = "target_defaults_actquant_tests.rs"]
mod actquant;

/// 2026-09-25: An empty `hw` prints as `unknown`.
#[test]
fn an_anonymous_build_still_prints_a_readable_line() {
    let anon = TargetDefaults { hw: "", ..GB10 };
    assert!(
        format_levers(&empty(&anon)).starts_with("target defaults (unknown): "),
        "{}",
        format_levers(&empty(&anon))
    );
}

/// 2026-09-25: The process-wide resolution reads this binary's own baked
/// declaration.
#[test]
fn the_process_resolution_reads_this_binarys_declaration() {
    assert_eq!(resolved().hw, declared().hw);
    assert_eq!(
        resolved().hw,
        metrale_kernels::TARGET_DEFAULTS.hw,
        "one table, one resolution"
    );
}

// 2026-09-25: The M16 tensor-core rows' tests, as a child module so they share
// the fixtures above.
#[path = "target_defaults_m16_tests.rs"]
mod m16;

/// 2026-09-25: Hopper's head band of 16 resolves from the declaration alone,
/// with no ` (env)` tag.
#[test]
fn hopper_resolves_the_widened_head_band_from_its_declaration() {
    let h = empty(&HOPPER);
    assert_eq!(h.lm_head_batchm_max.value, 16);
    assert!(!h.lm_head_batchm_max.from_env());
    assert_eq!(empty(&GB10).lm_head_batchm_max.value, BASELINE_BATCHM_MAX);
    assert!(format_levers(&h).contains("lm_head_batchm_max=16"));
    assert!(!format_levers(&h).contains("lm_head_batchm_max=16 (env)"));
}
/// 2026-09-25: The `ffn_gateup_fused` row's tests, as a child module so they
/// share the fixtures above.
#[path = "target_defaults_gateup_tests.rs"]
mod gateup;
