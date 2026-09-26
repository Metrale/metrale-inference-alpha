// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `kernels/<hw>/HARDWARE.toml` `[defaults]` and `[hardware] sm_count`, parsed for build.rs and baked into `metrale_kernels::TARGET_DEFAULTS` / `TARGET_SM_COUNT`.
//!
//! Owner: metrale-kernels build.
//! Invariants:
//! - An unknown `[defaults]` key, a mistyped value or a non-positive
//!   `sm_count` panics; an absent key takes the [`baseline`] value.
//!
//! Included via `#[path = "build_defaults.rs"] mod build_defaults;`. Its own
//! file, with no `super::` dependencies, so `tests/target_defaults.rs` can
//! compile the same code against the real `kernels/*/HARDWARE.toml`: cargo
//! never runs a build script's `#[cfg(test)]` modules.

/// 2026-09-25: One target's `[defaults]` table, owned (build-time shape).
///
/// Mirrors `metrale_kernels::TargetDefaults` field for field; [`literal`] emits
/// that type's `const` initialiser. Two shapes rather than one because the
/// runtime type holds `&'static str` (it is a baked constant) and a parser
/// needs `String`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Defaults {
    pub hw: String,
    pub lm_head_batchm_max: u32,
    pub ssm_batched_recurrent: bool,
    pub gdn_prefill_tc: bool,
    pub ssm_ba_gates_hopper: bool,
    pub fp8_act_quant_hopper: bool,
    pub decode_split_silu: bool,
    pub attn_decode_splitk: String,
    pub ffn_m16_tc: bool,
    pub attn_m16_tc: bool,
    pub lm_head_m16_tc: bool,
    pub attn_ncol_gemv: bool,
    pub ffn_gateup_fused: bool,
    pub w8a8_prefill_max_m_widening: u32,
    pub w8a8_prefill_max_m_narrowing: u32,
}

/// 2026-09-25: What a target that declares no `[defaults]` table gets, and
/// what each key a table omits falls back to.
///
/// `kernels/metal`, `kernels/strix` and `kernels/strix-hip` declare no table.
/// `kernels/gb10` declares these values explicitly except the two W8A8
/// prefill ceilings, which `tests/target_defaults.rs`
/// (`gb10_declares_the_baseline_apart_from_the_measured_w8a8_ceiling`) pins.
pub(crate) fn baseline(hw: &str) -> Defaults {
    Defaults {
        hw: hw.to_string(),
        // 2026-09-25: Equal to `ops::DENSE_GEMV_BATCHM_DECODE_MAX_M` in
        // metrale-model-layers, which this crate sits below and cannot name.
        // Tested as a chain: `tests/target_defaults.rs` pins gb10's table to
        // this baseline, and model-layers'
        // `target_defaults_tests::the_baseline_band_is_the_frozen_one` pins
        // gb10's value to that constant.
        lm_head_batchm_max: 8,
        ssm_batched_recurrent: false,
        gdn_prefill_tc: false,
        ssm_ba_gates_hopper: false,
        fp8_act_quant_hopper: false,
        decode_split_silu: true,
        // 2026-09-25: `metrale_kernels::attn_splitk::SplitkPolicy::Legacy`.
        attn_decode_splitk: "legacy".to_string(),
        ffn_m16_tc: false,
        attn_m16_tc: false,
        lm_head_m16_tc: false,
        attn_ncol_gemv: false,
        // 2026-09-25: The fused gate+up decode GEMM needs the strided-SiLU
        // consumer `silu_mul_strided.cu`, which only `kernels/hopper/common`
        // carries, so the row is inert on every other tree.
        ffn_gateup_fused: false,
        // 2026-09-25: No cap: a target that has not measured W8A8 losing to
        // W8A16 at large M declares none.
        w8a8_prefill_max_m_widening: u32::MAX,
        w8a8_prefill_max_m_narrowing: u32::MAX,
    }
}

/// 2026-09-25: What a target that declares no `[hardware] sm_count` gets: 48,
/// the value of `metrale_core::device::sm121::NUM_SMS` (GB10). gb10, metal,
/// strix and strix-hip declare none, so all four resolve to 48.
pub(crate) const BASELINE_SM_COUNT: u32 = 48;

/// 2026-09-25: `[hardware] sm_count`, or [`BASELINE_SM_COUNT`].
///
/// A non-positive or non-integer declaration panics: a zero SM count would
/// divide a grid-sizing rule by nothing, and a string must not read as
/// "absent".
pub(crate) fn parse_sm_count(hw: &str, hw_toml: &toml::Value) -> u32 {
    let Some(value) = hw_toml.get("hardware").and_then(|h| h.get("sm_count")) else {
        return BASELINE_SM_COUNT;
    };
    let n = value.as_integer().unwrap_or_else(|| {
        panic!("kernels/{hw}/HARDWARE.toml: [hardware] sm_count must be an integer")
    });
    u32::try_from(n).ok().filter(|&v| v > 0).unwrap_or_else(|| {
        panic!("kernels/{hw}/HARDWARE.toml: [hardware] sm_count = {n} is not a positive u32")
    })
}

/// 2026-09-25: Read `kernels/<hw>/HARDWARE.toml` and parse its `[hardware] sm_count`.
///
/// Missing or unparseable file falls back to the baseline, exactly as
/// [`read_defaults`] does and for the same reason (the `METRALE_SKIP_BUILD` path
/// may have no `kernels/` tree at all).
pub(crate) fn read_sm_count(kernels_root: &std::path::Path, hw: &str) -> u32 {
    let path = kernels_root.join(hw).join("HARDWARE.toml");
    let Ok(text) = std::fs::read_to_string(&path) else {
        return BASELINE_SM_COUNT;
    };
    let Ok(toml) = toml::from_str::<toml::Value>(&text) else {
        return BASELINE_SM_COUNT;
    };
    parse_sm_count(hw, &toml)
}

/// 2026-09-25: The generated `TARGET_SM_COUNT` initialiser, appended beside
/// [`literal`]'s.
pub(crate) fn sm_count_literal(sm_count: u32) -> String {
    format!(
        "// Auto-generated by build.rs from kernels/<hw>/HARDWARE.toml [hardware] sm_count.\n\
         /// Streaming multiprocessors on the hardware this binary's kernels were\n\
         /// built for, from `kernels/<hw>/HARDWARE.toml` `[hardware] sm_count`.\n\
         ///\n\
         /// Cross-checked at boot against the driver's own\n\
         /// `CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT`\n\
         /// (`metrale_gpu_runtime::cuda_backend::arch_preflight::check_sm_count`), which\n\
         /// WARNS on a mismatch and names both numbers.\n\
         pub const TARGET_SM_COUNT: u32 = {sm_count};\n"
    )
}

/// 2026-09-25: Parse `[defaults]` out of a `kernels/<hw>/HARDWARE.toml`.
///
/// Every key is optional and falls back to [`baseline`]. An unknown key
/// panics: a mistyped lever name would otherwise read as "this target agrees
/// with the baseline".
pub(crate) fn parse_defaults(hw: &str, hw_toml: &toml::Value) -> Defaults {
    let mut out = baseline(hw);
    let Some(table) = hw_toml.get("defaults").and_then(|d| d.as_table()) else {
        return out;
    };

    let boolean = |key: &str, v: &toml::Value| -> bool {
        v.as_bool().unwrap_or_else(|| {
            panic!("kernels/{hw}/HARDWARE.toml: [defaults] {key} must be a bool")
        })
    };
    let unsigned = |key: &str, v: &toml::Value| -> u32 {
        let n = v.as_integer().unwrap_or_else(|| {
            panic!("kernels/{hw}/HARDWARE.toml: [defaults] {key} must be an integer")
        });
        u32::try_from(n).unwrap_or_else(|_| {
            panic!("kernels/{hw}/HARDWARE.toml: [defaults] {key} = {n} is not a u32")
        })
    };
    let string = |key: &str, v: &toml::Value| -> String {
        v.as_str()
            .unwrap_or_else(|| {
                panic!("kernels/{hw}/HARDWARE.toml: [defaults] {key} must be a string")
            })
            .to_string()
    };

    for (key, value) in table {
        match key.as_str() {
            "lm_head_batchm_max" => out.lm_head_batchm_max = unsigned(key, value),
            "w8a8_prefill_max_m_widening" => out.w8a8_prefill_max_m_widening = unsigned(key, value),
            "w8a8_prefill_max_m_narrowing" => {
                out.w8a8_prefill_max_m_narrowing = unsigned(key, value)
            }
            "ssm_batched_recurrent" => out.ssm_batched_recurrent = boolean(key, value),
            "gdn_prefill_tc" => out.gdn_prefill_tc = boolean(key, value),
            "ssm_ba_gates_hopper" => out.ssm_ba_gates_hopper = boolean(key, value),
            "fp8_act_quant_hopper" => out.fp8_act_quant_hopper = boolean(key, value),
            "decode_split_silu" => out.decode_split_silu = boolean(key, value),
            "attn_decode_splitk" => out.attn_decode_splitk = string(key, value),
            "ffn_m16_tc" => out.ffn_m16_tc = boolean(key, value),
            "attn_m16_tc" => out.attn_m16_tc = boolean(key, value),
            "lm_head_m16_tc" => out.lm_head_m16_tc = boolean(key, value),
            "attn_ncol_gemv" => out.attn_ncol_gemv = boolean(key, value),
            "ffn_gateup_fused" => out.ffn_gateup_fused = boolean(key, value),
            other => panic!(
                "kernels/{hw}/HARDWARE.toml: [defaults] has no key `{other}`. \
                 The lever list is the field list of `TargetDefaults` \
                 (crates/kernels/src/target_defaults.rs); adding a lever \
                 means adding it there, in `build_defaults.rs`, in the \
                 metrale-model-layers resolver and in every target's table, in the one \
                 commit that lands the arm which reads it."
            ),
        }
    }
    out
}

/// 2026-09-25: The generated `TARGET_DEFAULTS` initialiser build.rs writes into `OUT_DIR`.
///
/// Emitted even under `METRALE_SKIP_BUILD=1` (which returns before any kernel
/// is compiled): the constant is configuration, not a kernel blob, and
/// CPU-only builds such as `scripts/check.sh` set that flag.
pub(crate) fn literal(d: &Defaults) -> String {
    format!(
        "// Auto-generated by build.rs from kernels/{hw}/HARDWARE.toml [defaults] — do not edit.\n\
         pub const TARGET_DEFAULTS: TargetDefaults = TargetDefaults {{\n\
         \x20   hw: \"{hw}\",\n\
         \x20   lm_head_batchm_max: {batchm},\n\
         \x20   ssm_batched_recurrent: {batched_recurrent},\n\
         \x20   gdn_prefill_tc: {gdn_tc},\n\
         \x20   ssm_ba_gates_hopper: {ba_gates},\n\
         \x20   fp8_act_quant_hopper: {act_quant},\n\
         \x20   decode_split_silu: {split_silu},\n\
         \x20   attn_decode_splitk: \"{splitk}\",\n\
         \x20   ffn_m16_tc: {ffn_m16_tc},\n\
         \x20   attn_m16_tc: {attn_m16_tc},\n\
         \x20   lm_head_m16_tc: {lm_head_m16_tc},\n\
         \x20   attn_ncol_gemv: {attn_ncol_gemv},\n\
         \x20   ffn_gateup_fused: {gateup_fused},\n\
         \x20   w8a8_prefill_max_m_widening: {w8a8_wide},\n\
         \x20   w8a8_prefill_max_m_narrowing: {w8a8_narrow},\n\
         }};\n",
        hw = d.hw,
        batchm = d.lm_head_batchm_max,
        batched_recurrent = d.ssm_batched_recurrent,
        gdn_tc = d.gdn_prefill_tc,
        ba_gates = d.ssm_ba_gates_hopper,
        act_quant = d.fp8_act_quant_hopper,
        split_silu = d.decode_split_silu,
        splitk = d.attn_decode_splitk,
        ffn_m16_tc = d.ffn_m16_tc,
        attn_m16_tc = d.attn_m16_tc,
        lm_head_m16_tc = d.lm_head_m16_tc,
        attn_ncol_gemv = d.attn_ncol_gemv,
        gateup_fused = d.ffn_gateup_fused,
        w8a8_wide = d.w8a8_prefill_max_m_widening,
        w8a8_narrow = d.w8a8_prefill_max_m_narrowing,
    )
}

/// 2026-09-25: Read `kernels/<hw>/HARDWARE.toml` and parse its `[defaults]`.
///
/// A missing or unparsable file falls back to [`baseline`] rather than
/// panicking, because this runs on the `METRALE_SKIP_BUILD` path too, where
/// the `kernels/` tree may not be present. The compiling build already panics
/// on a missing or bad HARDWARE.toml before it gets here.
pub(crate) fn read_defaults(kernels_root: &std::path::Path, hw: &str) -> Defaults {
    let path = kernels_root.join(hw).join("HARDWARE.toml");
    let Ok(text) = std::fs::read_to_string(&path) else {
        return baseline(hw);
    };
    let Ok(toml) = toml::from_str::<toml::Value>(&text) else {
        return baseline(hw);
    };
    parse_defaults(hw, &toml)
}
