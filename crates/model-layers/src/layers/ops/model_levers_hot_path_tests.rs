// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Default and spelling tests for the dense-FFN, MoE, MTP, decode-step and Nemotron levers: presence, strict `1`, truthy, and case-sensitive truthy.
//!
//! Owner: model-layers ops.
//! Invariants: none beyond the types.
//!
//! A child of `tests`, so `resolve` and the imports come from the parent.

use super::*;

/// 2026-09-25: The eleven dense-FFN levers are presence-gated: `=0` arms an opt-in and does not
/// re-enable an opt-out. Four of them (`METRALE_NO_*`, `METRALE_DISABLE_PREFILL_V2`) store the
/// opposite of their variable's name.
#[test]
fn the_dense_ffn_levers_are_presence_gated_and_their_polarities_hold() {
    let d = resolve(&[]);
    assert!(d.decode_split_silu, "split SiLU+down ships ON");
    assert!(d.ffn_nvfp4_mmq, "gate/up NVFP4 MMQ ships ON");
    assert!(d.ffn_nvfp4_mmq_down, "down NVFP4 MMQ ships ON");
    assert!(d.prefill_v2, "the v2 BF16 prefill kernel ships ON");
    assert!(!d.bf16_tc_prefill);
    assert!(!d.fp8_m64_prefill);
    assert!(!d.int8_prefill);
    assert!(!d.int8_faith5);
    assert!(!d.ffn_mmq);
    assert!(
        !d.ffn_mmq_down_q4k,
        "down stays on the NVFP4 hybrid by default"
    );
    assert!(!d.fp4_prefill);

    // 2026-09-25: Every opt-in arms on presence alone, including `=0`.
    let armed: [(&str, fn(&ModelLevers) -> bool); 7] = [
        ("METRALE_BF16_TC_PREFILL", |l| l.bf16_tc_prefill),
        ("METRALE_FP8_M64_PREFILL", |l| l.fp8_m64_prefill),
        ("METRALE_INT8_PREFILL", |l| l.int8_prefill),
        ("METRALE_INT8_FAITH5", |l| l.int8_faith5),
        ("METRALE_FFN_MMQ", |l| l.ffn_mmq),
        ("METRALE_FFN_MMQ_DOWN_Q4K", |l| l.ffn_mmq_down_q4k),
        ("METRALE_FP4_PREFILL", |l| l.fp4_prefill),
    ];
    for (name, read) in armed {
        assert!(read(&resolve(&[(name, "1")])), "{name} did not arm");
        assert!(
            read(&resolve(&[(name, "0")])),
            "{name} is presence-gated: `=0` still arms it"
        );
    }

    // 2026-09-25: Every kill switch disables on presence alone, including `=0`.
    let killed: [(&str, fn(&ModelLevers) -> bool); 4] = [
        ("METRALE_NO_DECODE_SPLIT_SILU", |l| l.decode_split_silu),
        ("METRALE_NO_FFN_NVFP4_MMQ", |l| l.ffn_nvfp4_mmq),
        ("METRALE_NO_FFN_NVFP4_MMQ_DOWN", |l| l.ffn_nvfp4_mmq_down),
        ("METRALE_DISABLE_PREFILL_V2", |l| l.prefill_v2),
    ];
    for (name, read) in killed {
        assert!(!read(&resolve(&[(name, "1")])), "{name} did not kill");
        assert!(
            !read(&resolve(&[(name, "0")])),
            "{name} is presence-gated: `=0` does NOT re-enable"
        );
    }

    // 2026-09-25: The down-projection kill switch and the gate/up one are independent.
    assert!(resolve(&[("METRALE_NO_FFN_NVFP4_MMQ_DOWN", "1")]).ffn_nvfp4_mmq);
    assert!(resolve(&[("METRALE_NO_FFN_NVFP4_MMQ", "1")]).ffn_nvfp4_mmq_down);
}

/// 2026-09-25: The decode-step levers: `ssm_save_dump` is presence-gated, while `ep_graphs` and
/// `gdn_decode_graph` accept only `1` or `true` (case-sensitive).
#[test]
fn the_decode_step_levers_keep_their_two_different_spellings() {
    let d = resolve(&[]);
    assert!(!d.ssm_save_dump);
    assert!(!d.ep_graphs);
    assert!(!d.gdn_decode_graph);

    // 2026-09-25: Presence: any value arms it, `0` and empty included.
    assert!(resolve(&[("METRALE_SSM_SAVE_DUMP", "1")]).ssm_save_dump);
    assert!(resolve(&[("METRALE_SSM_SAVE_DUMP", "0")]).ssm_save_dump);
    assert!(resolve(&[("METRALE_SSM_SAVE_DUMP", "")]).ssm_save_dump);

    // 2026-09-25: Truthy: `1` or `true`, nothing else.
    for (name, read) in [
        (
            "METRALE_EP_GRAPHS",
            (|l: &ModelLevers| l.ep_graphs) as fn(&ModelLevers) -> bool,
        ),
        ("METRALE_GDN_DECODE_GRAPH", |l: &ModelLevers| {
            l.gdn_decode_graph
        }),
    ] {
        assert!(read(&resolve(&[(name, "1")])), "{name} at =1");
        assert!(read(&resolve(&[(name, "true")])), "{name} at =true");
        assert!(!read(&resolve(&[(name, "0")])), "{name} armed at =0");
        assert!(
            !read(&resolve(&[(name, "")])),
            "{name} is truthy-gated, not presence-gated"
        );
        // 2026-09-25: Case-sensitive: `TRUE` must not arm graph capture.
        assert!(
            !read(&resolve(&[(name, "TRUE")])),
            "{name} must stay case-SENSITIVE: `TRUE` did not arm it before"
        );
    }
    // 2026-09-25: For contrast, `METRALE_LORA_EAGER` is case-insensitive.
    assert!(resolve(&[("METRALE_LORA_EAGER", "TRUE")]).lora_eager);
}

/// 2026-09-25: The MoE-forward, MTP-drafter and FP8 attention levers: all ten are strict `=1`
/// opt-ins, off by default. `fp32_routing` is the last term of
/// `MoeLayer::fp32_routing_active`, whose other four terms are weight and kernel conditions.
#[test]
fn the_moe_forward_and_mtp_levers_are_strict_opt_ins() {
    let d = resolve(&[]);
    assert!(!d.fp32_routing);
    assert!(!d.fp32_gate);
    assert!(!d.frankenstein_decode_via_prefill);
    assert!(!d.k2_diag);
    assert!(!d.mtp_debug_norms);
    assert!(!d.mtp_chain_postnorm);
    assert!(!d.mtp_target_postnorm);
    assert!(!d.mtp_kv_exact);
    assert!(!d.moe_fp8_grouped_decode_target);
    assert!(!d.fp8_attn_m32);

    let cases: [(&str, fn(&ModelLevers) -> bool); 10] = [
        ("METRALE_FP32_ROUTING", |l| l.fp32_routing),
        ("METRALE_FP32_GATE", |l| l.fp32_gate),
        ("METRALE_FRANKENSTEIN_DECODE_VIA_PREFILL", |l| {
            l.frankenstein_decode_via_prefill
        }),
        ("METRALE_K2_DIAG", |l| l.k2_diag),
        ("METRALE_MTP_DEBUG_NORMS", |l| l.mtp_debug_norms),
        ("METRALE_MTP_CHAIN_POSTNORM", |l| l.mtp_chain_postnorm),
        ("METRALE_MTP_TARGET_POSTNORM", |l| l.mtp_target_postnorm),
        ("METRALE_MTP_KV_EXACT", |l| l.mtp_kv_exact),
        ("METRALE_FP8_MOE_GROUPED_DECODE", |l| {
            l.moe_fp8_grouped_decode_target
        }),
        ("METRALE_FP8_ATTN_M32", |l| l.fp8_attn_m32),
    ];
    for (name, read) in cases {
        assert!(read(&resolve(&[(name, "1")])), "{name} did not arm at =1");
        assert!(!read(&resolve(&[(name, "0")])), "{name} armed at =0");
        assert!(
            !read(&resolve(&[(name, "true")])),
            "{name} is strict `1`, not truthy — that is how it was spelled"
        );
    }
    // 2026-09-25: The two FP32 levers resolve independently.
    assert!(!resolve(&[("METRALE_FP32_ROUTING", "1")]).fp32_gate);
    assert!(!resolve(&[("METRALE_FP32_GATE", "1")]).fp32_routing);
}

/// 2026-09-25: The draft-confidence clamp, through the pure `parse_draft_conf_tau`, and the
/// carried lever when `from_values` is given no floor.
#[test]
fn the_draft_confidence_clamp_holds_at_both_ends() {
    use crate::speculative::parse_draft_conf_tau as tau;
    assert_eq!(
        tau(None),
        0.0,
        "unset means OFF — three sites gate on `> 0.0`"
    );
    assert_eq!(tau(Some("")), 0.0);
    assert_eq!(tau(Some("junk")), 0.0);
    assert_eq!(tau(Some("0.7")), 0.7);
    assert_eq!(
        tau(Some("5.0")),
        0.99,
        "the upper clamp is load-bearing: an unclamped 5.0 puts the floor \
         above any achievable confidence and discards EVERY draft, turning \
         speculation off with nothing logged"
    );
    assert_eq!(
        tau(Some("-1")),
        0.0,
        "and the lower clamp cannot go negative"
    );
    // 2026-09-25: `from_values` carries the floor it is given (0.0 in `resolve`).
    assert_eq!(
        resolve(&[]).draft_conf_tau,
        0.0,
        "`from_values` takes the resolved tau as an INPUT; if it read the \
         environment itself this would depend on the ambient process"
    );
}

/// 2026-09-25: `ModelLevers::dflash_debug_dump_full` and metrale-model-arch's
/// `DFlashLevers::debug_dump_full` resolve the same variable. `TransformerModel` holds the drafter
/// as a `dyn DraftProposer` and cannot read its levers, so each struct carries its own copy. The
/// check reads both resolution lines from source, because a runtime check would need `set_var`.
#[test]
fn the_two_halves_of_the_dflash_dump_name_the_same_flag() {
    const FLAG: &str = "METRALE_DFLASH_DEBUG_DUMP_FULL";
    let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    for (rel, field) in [
        (
            "layers/ops/model_levers_resolve.rs",
            "dflash_debug_dump_full:",
        ),
        (
            "../../model-arch/src/dflash_head/levers.rs",
            "debug_dump_full:",
        ),
    ] {
        let text = std::fs::read_to_string(src.join(rel)).unwrap_or_else(|e| panic!("{rel}: {e}"));
        let line = text
            .lines()
            .find(|l| l.trim_start().starts_with(field))
            .unwrap_or_else(|| panic!("{rel} no longer resolves `{field}`"));
        // 2026-09-25: Compare the quoted name exactly: `contains` would accept a longer name
        // that has the flag as a prefix.
        let named = line.split('"').nth(1).unwrap_or_else(|| {
            panic!("{rel}: `{field}` names no quoted variable: {}", line.trim())
        });
        assert_eq!(
            named, FLAG,
            "{rel} resolves `{field}` from `{named}`, not `{FLAG}` — the two \
             halves of the dump would no longer arm together"
        );
    }
}

/// 2026-09-25: The MoE routed-prefill levers: four `=1` opt-ins, the tri-state
/// `moe_prefill_exact_tiles`, whose unset `None` lets the call site apply its per-checkpoint
/// default, and the numeric load factor.
#[test]
fn the_moe_prefill_levers_keep_the_tri_state_distinguishable() {
    let d = resolve(&[]);
    assert!(!d.moe_grouped_cutlass);
    assert!(!d.moe_grouped_down);
    assert!(!d.moe_prefill_zero);
    assert!(!d.moe_prefill_fp8_down);
    assert_eq!(
        d.moe_prefill_exact_tiles, None,
        "unset must defer to the checkpoint, not decide"
    );
    assert_eq!(d.moe_prefill_max_load_factor, None);

    assert!(resolve(&[("METRALE_HOLO_MOE_GROUPED_CUTLASS", "1")]).moe_grouped_cutlass);
    assert!(resolve(&[("METRALE_HOLO_MOE_GROUPED_DOWN", "1")]).moe_grouped_down);
    assert!(resolve(&[("METRALE_MOE_PREFILL_ZERO", "1")]).moe_prefill_zero);
    assert!(resolve(&[("METRALE_MOE_PREFILL_FP8_DOWN", "1")]).moe_prefill_fp8_down);
    // 2026-09-25: These four are value-gated, not presence-gated.
    assert!(!resolve(&[("METRALE_MOE_PREFILL_ZERO", "0")]).moe_prefill_zero);

    assert_eq!(
        resolve(&[("METRALE_MOE_PREFILL_EXACT_TILES", "1")]).moe_prefill_exact_tiles,
        Some(true)
    );
    assert_eq!(
        resolve(&[("METRALE_MOE_PREFILL_EXACT_TILES", "0")]).moe_prefill_exact_tiles,
        Some(false),
        "`0` is an explicit OFF, not an absent lever — the p90 measured -5.0% \
         there and +4.9% at ON, so both directions must stay reachable"
    );
    assert_eq!(
        resolve(&[("METRALE_MOE_PREFILL_EXACT_TILES", "yes")]).moe_prefill_exact_tiles,
        None
    );

    assert_eq!(
        resolve(&[("METRALE_MOE_PREFILL_MAX_LOAD_FACTOR", "4")]).moe_prefill_max_load_factor,
        Some(4)
    );
    // 2026-09-25: `0` means no cap, `None`, not a cap of zero.
    assert_eq!(
        resolve(&[("METRALE_MOE_PREFILL_MAX_LOAD_FACTOR", "0")]).moe_prefill_max_load_factor,
        None
    );
    assert_eq!(
        resolve(&[("METRALE_MOE_PREFILL_MAX_LOAD_FACTOR", "x")]).moe_prefill_max_load_factor,
        None
    );
}

/// 2026-09-25: The eight Nemotron prefill levers are presence-gated: `=0` arms an opt-in and does
/// not re-enable an opt-out. Five are kill switches whose field stores the opposite of their
/// variable's name.
#[test]
fn the_nemotron_prefill_levers_are_presence_gated() {
    let d = resolve(&[]);
    assert!(d.ssm_w4a4);
    assert!(d.ssd);
    assert!(d.ssm_persistent);
    assert!(
        d.moe_zero_intermediates,
        "arena buffers are cleared by default"
    );
    assert!(d.shared_w4a4);
    assert!(
        !d.moe_max_m_tiles_estimate,
        "the unsafe-to-serve bound is opt-in"
    );
    assert!(!d.moe_w4a4);
    assert!(!d.shared_w4a4_down);

    let killed: [(&str, fn(&ModelLevers) -> bool); 5] = [
        ("METRALE_NO_SSM_W4A4", |l| l.ssm_w4a4),
        ("METRALE_NO_SSD", |l| l.ssd),
        ("METRALE_NO_SSM_PERSISTENT", |l| l.ssm_persistent),
        ("METRALE_MOE_NO_ZERO_INTERMEDIATES", |l| {
            l.moe_zero_intermediates
        }),
        ("METRALE_NO_SHARED_W4A4", |l| l.shared_w4a4),
    ];
    for (name, read) in killed {
        assert!(!read(&resolve(&[(name, "1")])), "{name} did not kill");
        assert!(
            !read(&resolve(&[(name, "0")])),
            "{name} is presence-gated: `=0` does NOT re-enable"
        );
    }

    let armed: [(&str, fn(&ModelLevers) -> bool); 3] = [
        ("METRALE_MOE_MAX_M_TILES_ESTIMATE", |l| {
            l.moe_max_m_tiles_estimate
        }),
        ("METRALE_MOE_W4A4", |l| l.moe_w4a4),
        ("METRALE_SHARED_W4A4_DOWN", |l| l.shared_w4a4_down),
    ];
    for (name, read) in armed {
        assert!(read(&resolve(&[(name, "1")])), "{name} did not arm");
        assert!(
            read(&resolve(&[(name, "0")])),
            "{name} is presence-gated: `=0` still arms it"
        );
    }

    // 2026-09-25: The shared-expert up and down levers resolve independently.
    assert!(!resolve(&[("METRALE_NO_SHARED_W4A4", "1")]).shared_w4a4_down);
    assert!(resolve(&[("METRALE_SHARED_W4A4_DOWN", "1")]).shared_w4a4);
}

/// 2026-09-25: The five batched-decode levers: `mla_perseq_fallback` and `conc_hsd` accept `1` or
/// `true` (case-sensitive), the other three only `1`.
#[test]
fn the_batched_decode_levers_keep_their_neighbours_spellings() {
    let d = resolve(&[]);
    assert!(!d.mla_perseq_fallback);
    assert!(!d.hc_perseq_decode);
    assert!(!d.decode_batch_log);
    assert!(!d.ms_profile);
    assert!(!d.conc_hsd);

    // 2026-09-25: Strict `1`: `true` does not arm these.
    for (name, read) in [
        (
            "METRALE_HC_PERSEQ_DECODE",
            (|l: &ModelLevers| l.hc_perseq_decode) as fn(&ModelLevers) -> bool,
        ),
        ("METRALE_DECODE_BATCH_LOG", |l: &ModelLevers| {
            l.decode_batch_log
        }),
        ("METRALE_MS_PROFILE", |l: &ModelLevers| l.ms_profile),
    ] {
        assert!(read(&resolve(&[(name, "1")])), "{name} at =1");
        assert!(!read(&resolve(&[(name, "true")])), "{name} is strict `1`");
        assert!(!read(&resolve(&[(name, "0")])), "{name} at =0");
    }

    // 2026-09-25: Truthy but case-sensitive: `true` arms, `TRUE` does not.
    for (name, read) in [
        (
            "METRALE_MLA_PERSEQ_FALLBACK",
            (|l: &ModelLevers| l.mla_perseq_fallback) as fn(&ModelLevers) -> bool,
        ),
        ("METRALE_CONC_HSD", |l: &ModelLevers| l.conc_hsd),
    ] {
        assert!(read(&resolve(&[(name, "1")])), "{name} at =1");
        assert!(read(&resolve(&[(name, "true")])), "{name} at =true");
        assert!(
            !read(&resolve(&[(name, "TRUE")])),
            "{name} must stay case-SENSITIVE"
        );
        assert!(!read(&resolve(&[(name, "0")])), "{name} at =0");
    }

    // 2026-09-25: `METRALE_MS_PROFILE` and `METRALE_SSM_MS_PROFILE` are two variables; setting
    // either must not move the other.
    assert!(!resolve(&[("METRALE_MS_PROFILE", "1")]).ssm_ms_profile);
    assert!(!resolve(&[("METRALE_SSM_MS_PROFILE", "1")]).ms_profile);
}
