// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests the checked-in HARDWARE.toml `[defaults]` and
//! `[hardware] sm_count` values, parsed with the build script's own
//! `build_defaults.rs`, and the constants build.rs generates from them.
//!
//! Owner: metrale-kernels tests.
//! Invariants: none beyond the types.
//!
//! metrale-model-layers' `target_defaults_tests` test the runtime resolver
//! instead. This is an integration test because cargo does not run a build
//! script's own unit tests.

#[path = "../build_defaults.rs"]
mod build_defaults;

use build_defaults::{
    BASELINE_SM_COUNT, Defaults, baseline, literal, parse_defaults, read_defaults, read_sm_count,
    sm_count_literal,
};

use std::path::PathBuf;

/// 2026-09-25: The hardware sets whose HARDWARE.toml has a `[defaults]` table.
const DECLARING: &[&str] = &["gb10", "hopper", "b200", "b300"];

fn kernels_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("crates/kernels is two levels below the workspace root")
        .join("kernels")
}

fn declared(hw: &str) -> Defaults {
    read_defaults(&kernels_root(), hw)
}

/// 2026-09-25: The values `kernels/hopper/HARDWARE.toml` declares.
#[test]
fn hopper_declares_what_an_h100_serve_runs_with() {
    let d = declared("hopper");
    assert_eq!(d.hw, "hopper");
    assert!(
        d.ssm_batched_recurrent,
        "+6% on the serve, md5-identical output to the per-sequence launches"
    );
    assert!(d.decode_split_silu);
    // 2026-09-25: `true` arms the kernel; it still declines a launch below
    // `FP8_QUANT_MIN_CTAS_PER_SM * sm_count` CTAs (metrale-model-layers
    // `ops/fp8_act_quant_floor.rs`).
    assert!(
        d.fp8_act_quant_hopper,
        "the FP8 activation-quant twin is Hopper's default: 3.30-3.59x and \
         63.7-68.4% of HBM at prefill widths against the parent's 18.6-19.1%, \
         bit-identical, with the decode-width loss handled by the CTA floor \
         rather than by this row (`FP8-ACT-QUANT-ATTRIBUTION.md`)"
    );
    // 2026-09-25: On only here. Its strided SiLU consumer,
    // `silu_mul_strided.cu`, is in `kernels/hopper/common` and nowhere else.
    assert!(
        d.ffn_gateup_fused,
        "one cuBLASLt call at N=34816 on the decode band, not two at N=17408"
    );
    assert_eq!(
        d.attn_decode_splitk, "auto",
        "H100 serves paged-decode attention with the occupancy-filling split          count; `legacy` is the rule that gave it 24 CTAs on 132 SMs"
    );
    assert!(
        d.attn_m16_tc,
        "round 9 cell W: +5.26% C=16 aggregate, -6.38% TPOT, against a 0.15% \
         rep spread"
    );
    assert!(
        !d.ffn_m16_tc,
        "the same kernel family on the dense-FFN arm measured -5.2% (round 6 \
         cell J); one kernel, two rows, two verdicts"
    );
    assert!(
        d.lm_head_m16_tc,
        "round 9 cell Y: +4.09% C=16 aggregate on the BF16 decode head"
    );
    assert_eq!(d.lm_head_batchm_max, 16);
    assert_ne!(
        d.lm_head_batchm_max,
        baseline("hopper").lm_head_batchm_max,
        "hopper's band is its own; gb10's frozen 8 stays the baseline"
    );
    assert!(
        d.gdn_prefill_tc,
        "round 13: the tensor-core GDN prefill family is Hopper's default — \
         -39.6%/-44.7% on C=1 TTFT, +21.5%/+31.4% on C=16 aggregate"
    );
    assert!(
        d.ssm_ba_gates_hopper,
        "the BA-gates twin is bit-identical to its parent and Hopper-only; it \
         is on because it cannot change output and off it re-reads every \
         activation row 96 times (`SSM-BA-GATES-ATTRIBUTION.md`)"
    );
}

/// 2026-09-25: `kernels/gb10` declares the baseline except the two W8A8
/// prefill ceilings, checked as an equality against `baseline` so a third
/// difference fails.
#[test]
fn gb10_declares_the_baseline_apart_from_the_measured_w8a8_ceiling() {
    let d = declared("gb10");

    // 2026-09-25: Two rows because the crossover depends on the shape.
    // Measured 2026-09-11 on GB10 at the Qwen3.8-27B dims (recorded in
    // kernels/gb10/HARDWARE.toml): gate/up (N=17408, K=5120) crosses at
    // M~64-128, down (N=5120, K=17408) at M~384-512.
    assert_eq!(d.w8a8_prefill_max_m_widening, 64);
    assert_eq!(d.w8a8_prefill_max_m_narrowing, 384);
    assert_eq!(baseline("gb10").w8a8_prefill_max_m_widening, u32::MAX);
    assert_eq!(baseline("gb10").w8a8_prefill_max_m_narrowing, u32::MAX);

    // 2026-09-25: Every other field equals the baseline.
    let normalised = Defaults {
        w8a8_prefill_max_m_widening: u32::MAX,
        w8a8_prefill_max_m_narrowing: u32::MAX,
        ..d
    };
    assert_eq!(
        normalised,
        baseline("gb10"),
        "apart from the W8A8 prefill ceiling, kernels/gb10/HARDWARE.toml \
         [defaults] must restate the pre-existing hardcoded defaults and \
         nothing else — it exists to SAY what GB10 serves with"
    );
}

/// 2026-09-25: b200 declares the baseline, not hopper's values.
#[test]
fn b200_declares_the_conservative_table_not_hoppers() {
    let d = declared("b200");
    assert_eq!(d, baseline("b200"));
    assert!(
        !d.ffn_gateup_fused && declared("hopper").ffn_gateup_fused,
        "the fused gate+up decode GEMM is ON for Hopper on a Hopper receipt \
         and OFF here for want of one"
    );
    assert!(
        !d.ssm_batched_recurrent && declared("hopper").ssm_batched_recurrent,
        "the batched GDN recurrence is ON for Hopper on a Hopper receipt and \
         OFF here for want of one — B200 must not inherit a measured recipe by \
         resemblance"
    );
    assert!(
        !d.gdn_prefill_tc && declared("hopper").gdn_prefill_tc,
        "the GDN prefill family is ON for Hopper on a Hopper receipt (round 13) \
         and OFF here for want of one — the same rule, stated on the row that \
         most recently moved"
    );
    assert!(
        !d.ssm_ba_gates_hopper && declared("hopper").ssm_ba_gates_hopper,
        "the BA-gates twin is Hopper-only source; B200's common/ does not link \
         it, so the row is inert here and must read false"
    );
    assert!(
        !d.fp8_act_quant_hopper && declared("hopper").fp8_act_quant_hopper,
        "the FP8 activation-quant twin is Hopper-only source; B200's common/ \
         does not link it, so the row is inert here and must read false — and \
         its floor is `2 * sm_count` CTAs, which on 148 SMs is a threshold \
         nobody has measured"
    );
}

/// 2026-09-25: metal, strix and strix-hip have no `[defaults]` table and
/// resolve to the baseline.
#[test]
fn the_silent_targets_resolve_to_the_baseline() {
    for hw in ["metal", "strix", "strix-hip"] {
        assert_eq!(
            declared(hw),
            baseline(hw),
            "kernels/{hw}/HARDWARE.toml declares no [defaults] and must be \
             byte-for-byte unaffected"
        );
    }
}

/// 2026-09-25: Every `DECLARING` target writes every lever, even where the
/// value equals the baseline, so the file shows every value without the
/// reader knowing `baseline`.
#[test]
fn every_declaring_target_states_every_lever() {
    for hw in DECLARING {
        let raw = std::fs::read_to_string(kernels_root().join(hw).join("HARDWARE.toml"))
            .unwrap_or_else(|e| panic!("kernels/{hw}/HARDWARE.toml: {e}"));
        for lever in [
            "lm_head_batchm_max",
            "ssm_batched_recurrent",
            "gdn_prefill_tc",
            "ssm_ba_gates_hopper",
            "decode_split_silu",
            "attn_decode_splitk",
            "ffn_m16_tc",
            "attn_m16_tc",
            "lm_head_m16_tc",
            "attn_ncol_gemv",
            "ffn_gateup_fused",
            "fp8_act_quant_hopper",
            "w8a8_prefill_max_m_widening",
            "w8a8_prefill_max_m_narrowing",
        ] {
            assert!(
                raw.contains(&format!("\n{lever} = ")),
                "kernels/{hw}/HARDWARE.toml [defaults] must declare `{lever}` \
                 explicitly, not inherit it from the baseline"
            );
        }
    }
}

/// 2026-09-25: The `[hardware] sm_count` each target declares, or
/// `BASELINE_SM_COUNT` where it declares none.
#[test]
fn every_target_declares_the_sm_count_of_its_own_part() {
    let root = kernels_root();
    assert_eq!(
        read_sm_count(&root, "hopper"),
        132,
        "H100 SXM5/PCIe/NVL and H200 SXM5 are all GH100 with 132 SMs"
    );
    assert_eq!(read_sm_count(&root, "gb10"), 48, "DGX Spark GB10");
    assert_eq!(read_sm_count(&root, "b200"), 148, "GB100, 148 SMs enabled");
    // 2026-09-25: These trees declare no sm_count.
    for hw in ["metal", "strix", "strix-hip"] {
        assert_eq!(read_sm_count(&root, hw), BASELINE_SM_COUNT, "{hw}");
    }
    // 2026-09-25: A missing tree falls back rather than panicking, because
    // this also runs on the METRALE_SKIP_BUILD path.
    assert_eq!(read_sm_count(&root, "no-such-hw"), BASELINE_SM_COUNT);
}

/// 2026-09-25: A zero `sm_count` panics, naming the file.
#[test]
#[should_panic(expected = "sm_count = 0 is not a positive u32")]
fn a_zero_sm_count_fails_the_build() {
    let toml: toml::Value = toml::from_str("[hardware]\nsm_count = 0\n").unwrap();
    let _ = build_defaults::parse_sm_count("fictional", &toml);
}

/// 2026-09-25: `sm_count_literal` emits `TARGET_SM_COUNT` as a `u32` const.
#[test]
fn the_sm_count_literal_is_a_compilable_const() {
    let line = sm_count_literal(132);
    assert!(
        line.contains("pub const TARGET_SM_COUNT: u32 = 132;"),
        "{line}"
    );
    assert!(line.contains("Auto-generated by build.rs"), "{line}");
}

/// 2026-09-25: An absent key takes the baseline value; one declared key
/// changes one field.
#[test]
fn absent_keys_fall_through_to_the_baseline() {
    let toml: toml::Value = toml::from_str("[defaults]\nlm_head_batchm_max = 16\n").unwrap();
    let d = parse_defaults("fictional", &toml);
    assert_eq!(d.lm_head_batchm_max, 16);
    assert_eq!(
        Defaults {
            lm_head_batchm_max: baseline("fictional").lm_head_batchm_max,
            ..d
        },
        baseline("fictional"),
        "one declared key must move one field"
    );
}

/// 2026-09-25: A misspelt lever name panics rather than reading as agreement
/// with the baseline.
#[test]
#[should_panic(expected = "has no key `ssm_batched_recurrent_misspelt`")]
fn an_unknown_lever_name_fails_the_build() {
    let toml: toml::Value =
        toml::from_str("[defaults]\nssm_batched_recurrent_misspelt = true\n").unwrap();
    let _ = parse_defaults("fictional", &toml);
}

/// 2026-09-25: A value of the wrong type panics, naming the key.
#[test]
#[should_panic(expected = "[defaults] ssm_batched_recurrent must be a bool")]
fn a_mistyped_value_fails_the_build_naming_the_key() {
    let toml: toml::Value =
        toml::from_str("[defaults]\nssm_batched_recurrent = \"yes\"\n").unwrap();
    let _ = parse_defaults("fictional", &toml);
}

/// 2026-09-25: With no HARDWARE.toml, `read_defaults` returns the baseline
/// rather than panicking, because the generator also runs on the
/// `METRALE_SKIP_BUILD` path. A normal build panics on a bad HARDWARE.toml in
/// `resolve_targets`.
#[test]
fn a_missing_hardware_toml_resolves_to_the_baseline() {
    let nowhere = kernels_root().join("no-such-hardware-tree-for-tests");
    assert_eq!(
        read_defaults(&nowhere, "gb10"),
        baseline("gb10"),
        "a missing tree must not fail a skip build"
    );
}

/// 2026-09-25: The generated `TARGET_DEFAULTS` initialiser, which `lib.rs`
/// `include!`s, names the listed fields with hopper's values. Checked as text,
/// because that output is compiled only when metrale-kernels itself builds.
#[test]
fn the_generated_constant_names_every_field() {
    let generated = literal(&declared("hopper"));
    assert!(generated.contains("pub const TARGET_DEFAULTS: TargetDefaults = TargetDefaults {"));
    for field in [
        "hw: \"hopper\"",
        "lm_head_batchm_max: 16",
        "ssm_batched_recurrent: true",
        "gdn_prefill_tc: true",
        "ssm_ba_gates_hopper: true",
        "fp8_act_quant_hopper: true",
        "decode_split_silu: true",
        "attn_decode_splitk: \"auto\"",
        "ffn_gateup_fused: true",
        "w8a8_prefill_max_m_widening: 4294967295",
        "w8a8_prefill_max_m_narrowing: 4294967295",
    ] {
        assert!(
            generated.contains(field),
            "generated constant is missing `{field}`:\n{generated}"
        );
    }
}

/// 2026-09-25: The `TARGET_DEFAULTS` and `TARGET_SM_COUNT` baked into this
/// binary equal what its own hardware tree declares.
#[test]
fn the_baked_constant_matches_its_own_hardware_tree() {
    let baked = metrale_kernels::TARGET_DEFAULTS;
    // 2026-09-25: Without `METRALE_TARGET_HW` build.rs bakes the default
    // tree, gb10; the check holds for whichever tree it is.
    let declared = read_defaults(&kernels_root(), baked.hw);
    assert_eq!(baked.hw, declared.hw);
    assert_eq!(baked.lm_head_batchm_max, declared.lm_head_batchm_max);
    assert_eq!(baked.ssm_batched_recurrent, declared.ssm_batched_recurrent);
    assert_eq!(baked.gdn_prefill_tc, declared.gdn_prefill_tc);
    assert_eq!(baked.ssm_ba_gates_hopper, declared.ssm_ba_gates_hopper);
    assert_eq!(baked.fp8_act_quant_hopper, declared.fp8_act_quant_hopper);
    assert_eq!(baked.decode_split_silu, declared.decode_split_silu);
    assert_eq!(baked.attn_decode_splitk, declared.attn_decode_splitk);
    assert_eq!(baked.ffn_gateup_fused, declared.ffn_gateup_fused);
    assert_eq!(
        baked.w8a8_prefill_max_m_widening,
        declared.w8a8_prefill_max_m_widening
    );
    assert_eq!(
        baked.w8a8_prefill_max_m_narrowing,
        declared.w8a8_prefill_max_m_narrowing
    );
    assert_eq!(
        metrale_kernels::TARGET_SM_COUNT,
        read_sm_count(&kernels_root(), baked.hw),
        "the baked SM count and the baked defaults must come from ONE tree"
    );
}

#[test]
fn b300_declares_conservative_defaults_until_measured_on_b300() {
    assert_eq!(declared("b300"), baseline("b300"));
    assert_eq!(read_sm_count(&kernels_root(), "b300"), 148);
}
