// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Lever resolution tests: the default and the spelling of every switch, driven through the production `from_values`.
//!
//! Owner: model-layers ops.
//! Invariants: none beyond the types.

use super::*;
// 2026-09-25: `resolve` is private to `model_levers`; this module is its child, so it can reach
// `from_values`.
use super::resolve::from_values;
use std::collections::HashMap;

/// 2026-09-25: Resolve against a fixed map instead of the process environment. `set_var` is
/// process-global and would race the other tests in this binary.
fn resolve(values: &[(&str, &str)]) -> ModelLevers {
    let values: HashMap<_, _> = values.iter().copied().collect();
    from_values(
        |name| values.get(name).map(|value| (*value).to_owned()),
        |name| values.contains_key(name),
        0,
        crate::drafter_context::DrafterContext::BOTH,
        0.0,
        // 2026-09-25: The compiled target's `[defaults] decode_split_silu`, fixed on here;
        // `target_defaults_tests` checks the declarations.
        true,
    )
}

/// 2026-09-25: More lever tests, in a child module so they share `resolve`.
#[path = "model_levers_hot_path_tests.rs"]
mod hot_path_levers;

#[test]
fn the_opt_out_lever_is_on_by_default_and_every_opt_in_is_off() {
    let d = ModelLevers::defaults();
    assert_eq!(
        resolve(&[]),
        d,
        "absent environment uses the public default"
    );
    assert_eq!(
        d,
        ModelLevers {
            gdn_regresident: true,
            gdn_wy17: true,
            gdn_wyn: true,
            gemv_sw: true,
            ffn_small_m: true,
            // 2026-09-25: The opt-out levers are listed by hand, so a new lever's polarity is
            // stated in the test rather than taken from `Default`.
            ssm_gemv_batch4: true,
            // 2026-09-25: The dense-FFN opt-outs, turned off by the presence of their variable.
            decode_split_silu: true,
            ffn_nvfp4_mmq: true,
            ffn_nvfp4_mmq_down: true,
            prefill_v2: true,
            // 2026-09-25: The Nemotron prefill opt-outs.
            ssm_w4a4: true,
            ssd: true,
            ssm_persistent: true,
            moe_zero_intermediates: true,
            shared_w4a4: true,
            max_decode_seqs: 1,
            drafter: crate::drafter_context::DrafterContext::BOTH,
            ..ModelLevers::default()
        }
    );
}

#[test]
fn exact_one_opt_ins_map_to_their_own_fields() {
    let cases = [
        (
            "METRALE_KV_POISON",
            [true, false, false, false, false, false],
        ),
        (
            "METRALE_GDN_BATCHED_FLA",
            [false, true, false, false, false, false],
        ),
        (
            "METRALE_DECODE_FFN_VIA_GEMM",
            [false, false, true, false, false, false],
        ),
        (
            "METRALE_MOE_UNION_STATS",
            [false, false, false, true, false, false],
        ),
        (
            "METRALE_DFLASH_CONTIG_ATTN",
            [false, false, false, false, true, false],
        ),
        ("METRALE_K4_DIAG", [false, false, false, false, false, true]),
    ];
    for (name, expected) in cases {
        let d = resolve(&[(name, "1")]);
        assert_eq!(
            [
                d.kv_poison,
                d.gdn_batched_fla,
                d.decode_ffn_via_gemm,
                d.moe_union_stats,
                d.dflash_contig_attn,
                d.k4_diag
            ],
            expected,
            "{name}"
        );
    }
    assert!(!resolve(&[("METRALE_K4_DIAG", "true")]).k4_diag);
}

#[test]
fn truthy_opt_ins_map_independently_and_presence_is_distinct() {
    let cases = [
        (
            "METRALE_HOLO_MOE_DOWN_FP4",
            [true, false, false, false, false],
        ),
        (
            "METRALE_HOLO_MOE_GATEUP_FP4",
            [false, true, false, false, false],
        ),
        ("METRALE_LORA_EAGER", [false, false, true, false, false]),
        ("METRALE_LORA_ROTATE", [false, false, false, true, false]),
        ("METRALE_DIAG_GEMMA4", [false, false, false, false, true]),
    ];
    for (name, expected) in cases {
        let d = resolve(&[(name, "TrUe")]);
        assert_eq!(
            [
                d.holo_moe_down_fp4,
                d.holo_moe_gateup_fp4,
                d.lora_eager,
                d.lora_rotate,
                d.gemma4_diag
            ],
            expected,
            "{name}"
        );
    }
    assert!(resolve(&[("METRALE_BF16_TC_PROJ", "0")]).bf16_tc_proj);
    // 2026-09-25: `TQ_PLUS_WEIGHT_ROTATION` is value-gated (`1` or `true`, any case), unlike the
    // presence-gated `METRALE_BF16_TC_PROJ` above.
    assert!(!resolve(&[]).weight_pre_rotated);
    assert!(resolve(&[("TQ_PLUS_WEIGHT_ROTATION", "1")]).weight_pre_rotated);
    assert!(resolve(&[("TQ_PLUS_WEIGHT_ROTATION", "TRUE")]).weight_pre_rotated);
    assert!(!resolve(&[("TQ_PLUS_WEIGHT_ROTATION", "0")]).weight_pre_rotated);

    // 2026-09-25: The SSM decode levers differ in polarity: `ssm_ms_profile`, `ssm_detail_profile`
    // and `gdn_fused_conv` are `=1` opt-ins, `ssm_gemv_batch4` is on unless `=0`, and
    // `moe_legacy_pertoken_decode` stores the positive of a variable whose call site reads the
    // negation.
    let d = resolve(&[]);
    assert!(!d.ssm_ms_profile, "profiling is off unless asked for");
    assert!(!d.ssm_detail_profile);
    assert!(!d.gdn_fused_conv);
    assert!(!d.moe_legacy_pertoken_decode, "default is token-major MoE");
    assert!(d.ssm_gemv_batch4, "batch-4 GEMV ships ON");

    assert!(resolve(&[("METRALE_SSM_MS_PROFILE", "1")]).ssm_ms_profile);
    assert!(resolve(&[("METRALE_SSM_DETAIL_PROFILE", "1")]).ssm_detail_profile);
    assert!(resolve(&[("METRALE_GDN_FUSED_CONV", "1")]).gdn_fused_conv);
    assert!(resolve(&[("METRALE_MOE_LEGACY_PERTOKEN_DECODE", "1")]).moe_legacy_pertoken_decode);
    assert!(!resolve(&[("METRALE_SSM_GEMV_BATCH4", "0")]).ssm_gemv_batch4);
    // 2026-09-25: `=0` does not arm a value-gated opt-in; it does arm the presence-gated
    // `METRALE_BF16_TC_PROJ`.
    assert!(!resolve(&[("METRALE_GDN_FUSED_CONV", "0")]).gdn_fused_conv);
}

#[test]
fn kill_switches_and_zero_opt_outs_keep_their_distinct_polarities() {
    let d = resolve(&[
        ("METRALE_NO_GDN_REGRESIDENT", "1"),
        ("METRALE_NO_GEMV_SW", "1"),
        ("METRALE_GDN_WY17", "0"),
        ("METRALE_GDN_WYN", "0"),
        ("METRALE_FFN_SMALLM", "0"),
    ]);
    assert!(!d.gdn_regresident);
    assert!(!d.gemv_sw);
    assert!(!d.gdn_wy17);
    assert!(!d.gdn_wyn);
    assert!(!d.ffn_small_m);
    assert!(resolve(&[("METRALE_NO_GDN_REGRESIDENT", "0")]).gdn_regresident);
    assert!(resolve(&[("METRALE_GDN_WY17", "1")]).gdn_wy17);
}

#[test]
fn externally_resolved_shadow_and_drafter_values_are_carried() {
    let d = from_values(
        |_| None,
        |_| false,
        7,
        crate::drafter_context::DrafterContext::OFF,
        0.42,
        true,
    );
    assert_eq!(d.shadow_topk, 7);
    assert_eq!(
        d.draft_conf_tau, 0.42,
        "the confidence clamp is resolved OUTSIDE `from_values` and carried \
         in, like shadow_topk and drafter — reading it inside broke the \
         function's purity and made a sibling test fail under parallelism"
    );
    assert_eq!(d.drafter, crate::drafter_context::DrafterContext::OFF);
}
