// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: How [`ModelLevers`] is read: the resolution table (`from_values`) and the `get`, `from_env` and `defaults` constructors.
//!
//! Owner: model-layers ops.
//! Invariants:
//! - `from_values` reads only its two closures and its arguments; it never touches the process
//!   environment, so the tests drive this production table directly.
//! - [`ModelLevers::get`] resolves once per process (`OnceLock`).
//! - `resolve(&[])` in the tests equals [`ModelLevers::defaults`]
//!   (`the_opt_out_lever_is_on_by_default_and_every_opt_in_is_off`).

use super::ModelLevers;
use crate::layers::ops::gemv_sw;

pub(super) fn from_values(
    mut value: impl FnMut(&str) -> Option<String>,
    mut present: impl FnMut(&str) -> bool,
    shadow_topk: usize,
    drafter: crate::drafter_context::DrafterContext,
    // 2026-09-25: Passed in, like `shadow_topk` and `drafter`, because
    // `speculative::draft_conf_tau()` reads the process environment and this function must not.
    draft_conf_tau: f32,
    // 2026-09-25: The compiled target's `[defaults] decode_split_silu`
    // (`target_defaults::declared()` in `from_env`), passed in so a test can choose it.
    default_split_silu: bool,
) -> ModelLevers {
    fn opt_in(value: Option<&str>) -> bool {
        value == Some("1")
    }
    fn opt_out(value: Option<&str>) -> bool {
        value != Some("0")
    }
    fn opt_in_truthy(value: Option<&str>) -> bool {
        value.is_some_and(|value| value == "1" || value.eq_ignore_ascii_case("true"))
    }
    /// 2026-09-25: `"1"` or `"true"` exactly: case-sensitive, unlike [`opt_in_truthy`], so `TRUE`
    /// does not arm. Used by `ep_graphs`, `gdn_decode_graph`, `mla_perseq_fallback` and `conc_hsd`.
    fn opt_in_truthy_exact(value: Option<&str>) -> bool {
        matches!(value, Some("1") | Some("true"))
    }

    ModelLevers {
        max_decode_seqs: 1,
        shadow_topk,
        kv_poison: opt_in(value("METRALE_KV_POISON").as_deref()),
        drafter,
        gdn_regresident: value("METRALE_NO_GDN_REGRESIDENT").as_deref() != Some("1"),
        gdn_batched_fla: opt_in(value("METRALE_GDN_BATCHED_FLA").as_deref()),
        gdn_wy17: opt_out(value("METRALE_GDN_WY17").as_deref()),
        gdn_wyn: opt_out(value("METRALE_GDN_WYN").as_deref()),
        ffn_small_m: opt_out(value("METRALE_FFN_SMALLM").as_deref()),
        gemv_sw: gemv_sw::gemv_sw_from(value("METRALE_NO_GEMV_SW").as_deref()),
        decode_ffn_via_gemm: opt_in(value("METRALE_DECODE_FFN_VIA_GEMM").as_deref()),
        holo_moe_down_fp4: opt_in_truthy(value("METRALE_HOLO_MOE_DOWN_FP4").as_deref()),
        holo_moe_gateup_fp4: opt_in_truthy(value("METRALE_HOLO_MOE_GATEUP_FP4").as_deref()),
        moe_union_stats: opt_in(value("METRALE_MOE_UNION_STATS").as_deref()),
        fp32_routing: opt_in(value("METRALE_FP32_ROUTING").as_deref()),
        fp32_gate: opt_in(value("METRALE_FP32_GATE").as_deref()),
        frankenstein_decode_via_prefill: opt_in(
            value("METRALE_FRANKENSTEIN_DECODE_VIA_PREFILL").as_deref(),
        ),
        k2_diag: opt_in(value("METRALE_K2_DIAG").as_deref()),
        dflash_debug_dump_full: opt_in(value("METRALE_DFLASH_DEBUG_DUMP_FULL").as_deref()),
        mtp_debug_norms: opt_in(value("METRALE_MTP_DEBUG_NORMS").as_deref()),
        mtp_chain_postnorm: opt_in(value("METRALE_MTP_CHAIN_POSTNORM").as_deref()),
        mtp_target_postnorm: opt_in(value("METRALE_MTP_TARGET_POSTNORM").as_deref()),
        mtp_kv_exact: opt_in(value("METRALE_MTP_KV_EXACT").as_deref()),
        moe_fp8_grouped_decode_target: opt_in(value("METRALE_FP8_MOE_GROUPED_DECODE").as_deref()),
        fp8_attn_m32: opt_in(value("METRALE_FP8_ATTN_M32").as_deref()),
        draft_conf_tau,
        // 2026-09-25: The target's declaration (`kernels/<hw>/HARDWARE.toml` `[defaults]
        // decode_split_silu`; undeclared is on, `crates/kernels/build_defaults.rs`) arrives as
        // `default_split_silu`. The presence of `METRALE_NO_DECODE_SPLIT_SILU`, any value,
        // turns it off (`resolve_toggle`'s `legacy_off`).
        decode_split_silu: crate::layers::ops::target_defaults::resolve_toggle(
            default_split_silu,
            None,
            present("METRALE_NO_DECODE_SPLIT_SILU"),
        )
        .value,
        bf16_tc_prefill: present("METRALE_BF16_TC_PREFILL"),
        fp8_m64_prefill: present("METRALE_FP8_M64_PREFILL"),
        int8_prefill: present("METRALE_INT8_PREFILL"),
        int8_faith5: present("METRALE_INT8_FAITH5"),
        ffn_nvfp4_mmq: !present("METRALE_NO_FFN_NVFP4_MMQ"),
        ffn_nvfp4_mmq_down: !present("METRALE_NO_FFN_NVFP4_MMQ_DOWN"),
        ffn_mmq: present("METRALE_FFN_MMQ"),
        ffn_mmq_down_q4k: present("METRALE_FFN_MMQ_DOWN_Q4K"),
        fp4_prefill: present("METRALE_FP4_PREFILL"),
        prefill_v2: !present("METRALE_DISABLE_PREFILL_V2"),
        moe_grouped_cutlass: opt_in(value("METRALE_HOLO_MOE_GROUPED_CUTLASS").as_deref()),
        moe_grouped_down: opt_in(value("METRALE_HOLO_MOE_GROUPED_DOWN").as_deref()),
        moe_prefill_exact_tiles: match value("METRALE_MOE_PREFILL_EXACT_TILES").as_deref() {
            Some("0") => Some(false),
            Some("1") => Some(true),
            _ => None,
        },
        moe_prefill_max_load_factor: value("METRALE_MOE_PREFILL_MAX_LOAD_FACTOR")
            .as_deref()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&factor| factor > 0),
        moe_prefill_zero: opt_in(value("METRALE_MOE_PREFILL_ZERO").as_deref()),
        moe_prefill_fp8_down: opt_in(value("METRALE_MOE_PREFILL_FP8_DOWN").as_deref()),
        ssm_w4a4: !present("METRALE_NO_SSM_W4A4"),
        ssd: !present("METRALE_NO_SSD"),
        ssm_persistent: !present("METRALE_NO_SSM_PERSISTENT"),
        moe_zero_intermediates: !present("METRALE_MOE_NO_ZERO_INTERMEDIATES"),
        moe_max_m_tiles_estimate: present("METRALE_MOE_MAX_M_TILES_ESTIMATE"),
        moe_w4a4: present("METRALE_MOE_W4A4"),
        shared_w4a4: !present("METRALE_NO_SHARED_W4A4"),
        shared_w4a4_down: present("METRALE_SHARED_W4A4_DOWN"),
        dflash_contig_attn: opt_in(value("METRALE_DFLASH_CONTIG_ATTN").as_deref()),
        lora_eager: opt_in_truthy(value("METRALE_LORA_EAGER").as_deref()),
        lora_rotate: opt_in_truthy(value("METRALE_LORA_ROTATE").as_deref()),
        k4_diag: opt_in(value("METRALE_K4_DIAG").as_deref()),
        gemma4_diag: opt_in_truthy(value("METRALE_DIAG_GEMMA4").as_deref()),
        mla_perseq_fallback: opt_in_truthy_exact(value("METRALE_MLA_PERSEQ_FALLBACK").as_deref()),
        hc_perseq_decode: opt_in(value("METRALE_HC_PERSEQ_DECODE").as_deref()),
        decode_batch_log: opt_in(value("METRALE_DECODE_BATCH_LOG").as_deref()),
        ms_profile: opt_in(value("METRALE_MS_PROFILE").as_deref()),
        conc_hsd: opt_in_truthy_exact(value("METRALE_CONC_HSD").as_deref()),
        ssm_save_dump: present("METRALE_SSM_SAVE_DUMP"),
        ep_graphs: opt_in_truthy_exact(value("METRALE_EP_GRAPHS").as_deref()),
        gdn_decode_graph: opt_in_truthy_exact(value("METRALE_GDN_DECODE_GRAPH").as_deref()),
        bf16_tc_proj: present("METRALE_BF16_TC_PROJ"),
        weight_pre_rotated: opt_in_truthy(value("TQ_PLUS_WEIGHT_ROTATION").as_deref()),
        ssm_ms_profile: opt_in(value("METRALE_SSM_MS_PROFILE").as_deref()),
        ssm_detail_profile: opt_in(value("METRALE_SSM_DETAIL_PROFILE").as_deref()),
        ssm_gemv_batch4: opt_out(value("METRALE_SSM_GEMV_BATCH4").as_deref()),
        gdn_fused_conv: opt_in(value("METRALE_GDN_FUSED_CONV").as_deref()),
        moe_legacy_pertoken_decode: opt_in(value("METRALE_MOE_LEGACY_PERTOKEN_DECODE").as_deref()),
    }
}

impl ModelLevers {
    /// 2026-09-25: The process-wide levers, resolved from the environment on the first call and
    /// cached for the life of the process. Prefer this to [`Self::from_env`] outside model
    /// construction. `max_decode_seqs` is 1 here. `ModelLevers` is `Copy`, so
    /// `*ModelLevers::get()` gives an owned value.
    pub fn get() -> &'static Self {
        static LEVERS: std::sync::OnceLock<ModelLevers> = std::sync::OnceLock::new();
        LEVERS.get_or_init(Self::from_env)
    }

    /// 2026-09-25: Resolve from the environment on every call. `TransformerModel::new` uses it so
    /// each model gets its own copy, whose `max_decode_seqs` it then sets. The
    /// `from_env_is_called_only_where_it_is_allowed` test fails when another file calls it.
    pub fn from_env() -> Self {
        from_values(
            metrale_config::levers::var,
            |var| metrale_config::levers::var_os(var).is_some(),
            crate::speculative::shadow_topk(),
            crate::drafter_context::resolve_from_env(),
            crate::speculative::draft_conf_tau(),
            crate::layers::ops::target_defaults::declared().decode_split_silu,
        )
    }

    /// 2026-09-25: What `from_values` returns with no variable set, a target that declares
    /// `decode_split_silu` on, `DrafterContext::BOTH`, no draft-confidence floor and no shadow
    /// top-k: every opt-in off, every opt-out on. Tests build a context with this instead of
    /// mutating the environment.
    pub fn defaults() -> Self {
        Self {
            max_decode_seqs: 1,
            shadow_topk: 0,
            kv_poison: false,
            drafter: crate::drafter_context::DrafterContext::BOTH,
            gdn_regresident: true,
            gdn_wy17: true,
            gdn_wyn: true,
            ffn_small_m: true,
            gemv_sw: true,
            // 2026-09-25: On unless `METRALE_SSM_GEMV_BATCH4=0`. Every opt-out lever must be listed
            // here or `the_opt_out_lever_is_on_by_default_and_every_opt_in_is_off` fails.
            ssm_gemv_batch4: true,
            // 2026-09-25: The dense-FFN opt-outs: on, and turned off by the presence of their
            // variable at any value, `=0` included.
            decode_split_silu: true,
            ffn_nvfp4_mmq: true,
            ffn_nvfp4_mmq_down: true,
            prefill_v2: true,
            // 2026-09-25: The Nemotron prefill opt-outs, presence-gated the same way.
            ssm_w4a4: true,
            ssd: true,
            ssm_persistent: true,
            moe_zero_intermediates: true,
            shared_w4a4: true,
            ..Self::default()
        }
    }
}
