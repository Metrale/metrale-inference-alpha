// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The compiled target's serving defaults, resolved against the
//! environment: the target's declaration first, a `METRALE_*` override second.
//!
//! The declaration is [`metrale_kernels::TARGET_DEFAULTS`], which `build.rs`
//! bakes from the `[defaults]` table of the compiled target's
//! `kernels/<hw>/HARDWARE.toml`. A target with no table gets
//! `build_defaults::baseline` (`crates/kernels/build_defaults.rs`).
//!
//! # The override grammar (boolean levers, [`resolve_toggle`])
//!
//! | input | meaning |
//! |---|---|
//! | variable absent | the target's declaration |
//! | `0`, `false`, `off`, `no` (any case, trimmed) | off, from the environment |
//! | any other value, including empty | on, from the environment |
//!
//! Two presence-gated kill switches outrank both the declaration and the
//! positive variable, whatever their value: `METRALE_NO_DECODE_SPLIT_SILU`
//! (`decode_split_silu`) and `METRALE_NO_ATTN_DECODE_BATCH` (`attn_ncol_gemv`).
//! The head band ([`resolve_batchm_max`]), the W8A8 row caps
//! ([`resolve_max_m`]) and the split-K policy have their own rules, documented
//! on their resolvers.
//!
//! # One resolution, one log line
//!
//! [`resolved`] is the process-wide resolution. The dispatch sites read it,
//! and `metrale-server` logs it once as [`summary_line`], which tags values
//! that came from the environment with ` (env)`. `ModelLevers` resolves
//! `decode_split_silu` itself, from [`declared`] through the same
//! [`resolve_toggle`].
//!
//! # Adding a lever
//!
//! A lever needs all of: the field in `metrale_kernels::TargetDefaults`
//! (`crates/kernels/src/target_defaults.rs`), the parse arm in
//! `crates/kernels/build_defaults.rs`, the row in every
//! `kernels/<hw>/HARDWARE.toml` that has a `[defaults]` table, the
//! [`TargetLevers`] field and its arm in [`resolve`], its entry in
//! [`format_levers`], and a test. `parse_defaults` panics on an unknown key,
//! so a row without its parse arm fails the build.
//!
//! Owner: model-layers ops (target serving defaults).
//! Invariants:
//! - [`resolved`] keeps its first completed resolution; later calls return
//!   that table.
//! - A value is tagged [`Source::Env`] exactly when an environment variable
//!   decided it.

use metrale_kernels::attn_splitk::{self, SplitkPolicy};

use super::gemm_quant::{DENSE_GEMV_BATCHM_DECODE_MAX_M, DENSE_GEMV_BATCHM_MAX_M};

/// 2026-09-25: Where a resolved value came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    /// 2026-09-25: `kernels/<hw>/HARDWARE.toml` `[defaults]`, or the baseline.
    Target,
    /// 2026-09-25: A `METRALE_*` variable in the process environment.
    Env,
}

impl Source {
    /// 2026-09-25: The suffix the boot line appends to an environment-sourced value.
    pub fn tag(self) -> &'static str {
        match self {
            Source::Target => "",
            Source::Env => " (env)",
        }
    }
}

/// 2026-09-25: A resolved lever and where it came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Resolved<T> {
    pub value: T,
    pub source: Source,
}

impl<T> Resolved<T> {
    fn target(value: T) -> Self {
        Self {
            value,
            source: Source::Target,
        }
    }
    fn env(value: T) -> Self {
        Self {
            value,
            source: Source::Env,
        }
    }
    /// 2026-09-25: True when an environment variable decided the value.
    pub fn from_env(&self) -> bool {
        self.source == Source::Env
    }
}

/// 2026-09-25: The override grammar for a boolean lever, as a pure function.
///
/// `raw` is the positive variable's value (`None` = absent). `legacy_off` is
/// the presence of the matching `METRALE_NO_*` kill switch, which wins over
/// everything, so a stale positive variable cannot veto it.
pub fn resolve_toggle(default_on: bool, raw: Option<&str>, legacy_off: bool) -> Resolved<bool> {
    if legacy_off {
        return Resolved::env(false);
    }
    match raw {
        None => Resolved::target(default_on),
        Some(v) => match v.trim().to_ascii_lowercase().as_str() {
            "0" | "false" | "off" | "no" => Resolved::env(false),
            _ => Resolved::env(true),
        },
    }
}

/// 2026-09-25: The BF16 decode head's batched-GEMV band: the widest `m` that
/// takes `dense_gemv_batchm`. Wider batches take a tile GEMM that reassociates
/// the reduction, so moving the edge changes the head's output bits.
///
/// Clamped to [`DENSE_GEMV_BATCHM_MAX_M`], the kernel's row bound:
/// `dense_gemv_batchm` returns an error above it, which would fail every
/// decode step. An unparseable or `0` environment value keeps the target's
/// declaration; the band has no "off".
pub fn resolve_batchm_max(default_max: u32, raw: Option<&str>) -> Resolved<u32> {
    let clamp = |v: u32| v.min(DENSE_GEMV_BATCHM_MAX_M);
    match raw
        .and_then(|v| v.trim().parse::<u32>().ok())
        .filter(|&v| v > 0)
    {
        Some(v) => Resolved::env(clamp(v)),
        None => Resolved::target(clamp(default_max)),
    }
}

/// 2026-09-25: Upper `M` for the W8A8 dense-FFN prefill, per projection shape.
///
/// Unlike [`resolve_batchm_max`], a parsed `0` is honoured: it means "never
/// take the W8A8 arm on this shape". Anything that is not a u32 keeps the
/// target's declaration.
pub fn resolve_max_m(default_max: u32, raw: Option<&str>) -> Resolved<u32> {
    match raw.and_then(|v| v.trim().parse::<u32>().ok()) {
        Some(v) => Resolved::env(v),
        None => Resolved::target(default_max),
    }
}

/// 2026-09-25: Every serving lever this target declares, resolved against the
/// environment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TargetLevers {
    /// 2026-09-25: The `kernels/<hw>` tree this binary was compiled from.
    pub hw: &'static str,
    pub lm_head_batchm_max: Resolved<u32>,
    pub ssm_batched_recurrent: Resolved<bool>,
    pub gdn_prefill_tc: Resolved<bool>,
    pub ssm_ba_gates_hopper: Resolved<bool>,
    pub fp8_act_quant_hopper: Resolved<bool>,
    pub decode_split_silu: Resolved<bool>,
    pub attn_decode_splitk: Resolved<SplitkPolicy>,
    /// 2026-09-25: The `w8a16_gemm_m16` tier on the dense-FFN decode arm.
    pub ffn_m16_tc: Resolved<bool>,
    /// 2026-09-25: The `w8a16_gemm_m16` tiers on the attention decode projections.
    pub attn_m16_tc: Resolved<bool>,
    /// 2026-09-25: The `dense_gemm_m16_bf16` arm on the BF16 decode head.
    pub lm_head_m16_tc: Resolved<bool>,
    /// 2026-09-25: `w8a16_gemv_batch16_ncol{2,4}` on the attention decode
    /// projections. Every target with a `[defaults]` table declares it off.
    pub attn_ncol_gemv: Resolved<bool>,
    pub ffn_gateup_fused: Resolved<bool>,
    pub w8a8_prefill_max_m_widening: Resolved<u32>,
    pub w8a8_prefill_max_m_narrowing: Resolved<u32>,
}

/// 2026-09-25: The whole table, as a pure function of a declaration and a
/// variable lookup, so tests resolve any target's table without touching the
/// process environment.
pub fn resolve(
    defaults: &metrale_kernels::TargetDefaults,
    mut var: impl FnMut(&str) -> Option<String>,
) -> TargetLevers {
    let split_silu_off = var("METRALE_NO_DECODE_SPLIT_SILU").is_some();

    TargetLevers {
        hw: defaults.hw,
        lm_head_batchm_max: resolve_batchm_max(
            defaults.lm_head_batchm_max,
            var("METRALE_LM_HEAD_BATCHM_MAX").as_deref(),
        ),
        w8a8_prefill_max_m_widening: resolve_max_m(
            defaults.w8a8_prefill_max_m_widening,
            var("METRALE_W8A8_PREFILL_MAX_M_WIDENING").as_deref(),
        ),
        w8a8_prefill_max_m_narrowing: resolve_max_m(
            defaults.w8a8_prefill_max_m_narrowing,
            var("METRALE_W8A8_PREFILL_MAX_M_NARROWING").as_deref(),
        ),
        // 2026-09-25: `kernels/hopper` declares it on, the other tables off. A
        // pinned `--ssm-batched-recurrent` outranks this row (`serve_flags.rs`).
        ssm_batched_recurrent: resolve_toggle(
            defaults.ssm_batched_recurrent,
            var("METRALE_SSM_BATCHED_RECURRENT").as_deref(),
            false,
        ),
        // 2026-09-25: `METRALE_GDN_PREFILL_TC=0` turns the tensor-core GDN
        // prefill off, and with it the remnant twins that read the same bit.
        gdn_prefill_tc: resolve_toggle(
            defaults.gdn_prefill_tc,
            var("METRALE_GDN_PREFILL_TC").as_deref(),
            false,
        ),
        // 2026-09-25: The BA-gates twin. `kernels/hopper` declares it on; it has
        // no `METRALE_NO_*` spelling.
        ssm_ba_gates_hopper: resolve_toggle(
            defaults.ssm_ba_gates_hopper,
            var("METRALE_SSM_BA_GATES_HOPPER").as_deref(),
            false,
        ),
        // 2026-09-25: The FP8 activation-quant twin. `kernels/hopper` declares
        // it on; it has no `METRALE_NO_*` spelling. When on, the twin still
        // passes a CTA-count floor (`fp8_act_quant_floor.rs`) before it takes a
        // launch; `METRALE_FP8_ACT_QUANT_HOPPER=0` declines it at every width.
        fp8_act_quant_hopper: resolve_toggle(
            defaults.fp8_act_quant_hopper,
            var("METRALE_FP8_ACT_QUANT_HOPPER").as_deref(),
            false,
        ),
        // 2026-09-25: The declaration and the presence-gated
        // `METRALE_NO_DECODE_SPLIT_SILU`; there is no positive variable.
        decode_split_silu: resolve_toggle(defaults.decode_split_silu, None, split_silu_off),
        // 2026-09-25: The paged-decode split-K policy. The rule is
        // `metrale_kernels::attn_splitk::resolve_policy`, which the buffer arena
        // (`gpu-runtime/src/buffers/sizes.rs`, through `policy_from_env`) also
        // calls to size the split-K workspace, so the split count a launch uses
        // and the one the workspace was sized for come from one function.
        attn_decode_splitk: {
            let (policy, from_env) = attn_splitk::resolve_policy(
                defaults.attn_decode_splitk,
                var("METRALE_ATTN_DECODE_SPLITK").as_deref(),
            );
            if from_env {
                Resolved::env(policy)
            } else {
                Resolved::target(policy)
            }
        },
        // 2026-09-25: Two rows for one kernel family. `METRALE_M16_TC` is an
        // umbrella read for both rows, under the same grammar, when the row's
        // own variable is absent, so `METRALE_M16_TC=0` turns both off. The
        // row's own variable wins when both are set.
        ffn_m16_tc: resolve_toggle(
            defaults.ffn_m16_tc,
            var("METRALE_FFN_M16_TC")
                .or_else(|| var("METRALE_M16_TC"))
                .as_deref(),
            false,
        ),
        attn_m16_tc: resolve_toggle(
            defaults.attn_m16_tc,
            var("METRALE_ATTN_M16_TC")
                .or_else(|| var("METRALE_M16_TC"))
                .as_deref(),
            false,
        ),
        // 2026-09-25: Not under `METRALE_M16_TC`: only its own variable
        // overrides the declaration.
        lm_head_m16_tc: resolve_toggle(
            defaults.lm_head_m16_tc,
            var("METRALE_LM_HEAD_M16_TC").as_deref(),
            false,
        ),
        // 2026-09-25: The presence-gated `METRALE_NO_ATTN_DECODE_BATCH` outranks
        // both the declaration and `METRALE_ATTN_NCOL_GEMV`.
        attn_ncol_gemv: resolve_toggle(
            defaults.attn_ncol_gemv,
            var("METRALE_ATTN_NCOL_GEMV").as_deref(),
            var("METRALE_NO_ATTN_DECODE_BATCH").is_some(),
        ),
        // 2026-09-25: `kernels/hopper` declares it on. `=0` turns the arm off, so
        // gate and up run as two GEMMs. On a target whose tree lacks
        // `silu_mul_strided.cu`, turning it on finds no kernel
        // (`try_target_kernel` returns 0) and the layer declines the arm.
        ffn_gateup_fused: resolve_toggle(
            defaults.ffn_gateup_fused,
            var("METRALE_FFN_GATEUP_FUSED").as_deref(),
            false,
        ),
    }
}

/// 2026-09-25: The process-wide resolution, computed on the first call and
/// cached.
///
/// Cached because the dispatch sites read it per projection per step, and the
/// route must stay constant across CUDA-graph replays: a per-call environment
/// read could change the launch set between capture and replay.
pub fn resolved() -> &'static TargetLevers {
    static LEVERS: std::sync::OnceLock<TargetLevers> = std::sync::OnceLock::new();
    LEVERS.get_or_init(|| {
        resolve(
            &metrale_kernels::TARGET_DEFAULTS,
            metrale_config::levers::var,
        )
    })
}

/// 2026-09-25: The baked declaration this binary carries, for callers that
/// resolve from their own inputs (`ModelLevers`).
pub fn declared() -> &'static metrale_kernels::TargetDefaults {
    &metrale_kernels::TARGET_DEFAULTS
}

/// 2026-09-25: `target defaults (<hw>): …`, the one line naming every resolved
/// value and which came from the environment. Built here so the line and the
/// resolution are the same code.
pub fn summary_line() -> String {
    format_levers(resolved())
}

/// 2026-09-25: [`summary_line`] over a table the caller already has. Pure, so
/// tests can format any target's line without touching the process
/// environment or the cached [`resolved`] table.
pub fn format_levers(l: &TargetLevers) -> String {
    let onoff =
        |r: Resolved<bool>| format!("{}{}", if r.value { "on" } else { "off" }, r.source.tag());
    // 2026-09-25: `u32::MAX` is the no-cap baseline; it prints as `max`.
    let cap = |v: u32| {
        if v == u32::MAX {
            "max".to_string()
        } else {
            v.to_string()
        }
    };
    format!(
        "target defaults ({hw}): sm_count={sms} \
         lm_head_batchm_max={batchm}{batchm_src} \
         ssm_batched_recurrent={recurrent} gdn_prefill_tc={gdn_tc} \
         ssm_ba_gates_hopper={ba_gates} decode_split_silu={silu} \
         attn_decode_splitk={splitk}{splitk_src} ffn_m16_tc={ffn_m16_tc} \
         attn_m16_tc={attn_m16_tc} lm_head_m16_tc={lm_head_m16_tc} \
         attn_ncol_gemv={attn_ncol_gemv} ffn_gateup_fused={gateup} \
         fp8_act_quant_hopper={act_quant} \
         w8a8_prefill_max_m={w8a8_wide}/{w8a8_narrow}{w8a8_src}",
        hw = if l.hw.is_empty() { "unknown" } else { l.hw },
        // 2026-09-25: Not a lever: the target's `[hardware] sm_count`, which
        // `arch_preflight::check_sm_count` compares with the device at boot.
        sms = metrale_kernels::TARGET_SM_COUNT,
        batchm = l.lm_head_batchm_max.value,
        batchm_src = l.lm_head_batchm_max.source.tag(),
        recurrent = onoff(l.ssm_batched_recurrent),
        gdn_tc = onoff(l.gdn_prefill_tc),
        ba_gates = onoff(l.ssm_ba_gates_hopper),
        act_quant = onoff(l.fp8_act_quant_hopper),
        silu = onoff(l.decode_split_silu),
        splitk = l.attn_decode_splitk.value.label(),
        splitk_src = l.attn_decode_splitk.source.tag(),
        ffn_m16_tc = onoff(l.ffn_m16_tc),
        attn_m16_tc = onoff(l.attn_m16_tc),
        lm_head_m16_tc = onoff(l.lm_head_m16_tc),
        attn_ncol_gemv = onoff(l.attn_ncol_gemv),
        gateup = onoff(l.ffn_gateup_fused),
        // 2026-09-25: Printed as widening/narrowing, with the widening row's
        // source tag.
        w8a8_wide = cap(l.w8a8_prefill_max_m_widening.value),
        w8a8_narrow = cap(l.w8a8_prefill_max_m_narrowing.value),
        w8a8_src = l.w8a8_prefill_max_m_widening.source.tag(),
    )
}

/// 2026-09-25: The head band a target without a `[defaults]` table gets.
/// `build_defaults::baseline` (`crates/kernels/build_defaults.rs`) repeats it
/// as a literal. `target_defaults_tests::the_baseline_band_is_the_frozen_one`
/// checks this constant against `DENSE_GEMV_BATCHM_DECODE_MAX_M` and the test
/// file's copy of the gb10 declaration;
/// `gb10_declares_the_baseline_apart_from_the_measured_w8a8_ceiling`
/// (`crates/kernels/tests/target_defaults.rs`) checks the real gb10
/// declaration against the literal.
pub const BASELINE_BATCHM_MAX: u32 = DENSE_GEMV_BATCHM_DECODE_MAX_M;

#[cfg(test)]
#[path = "target_defaults_tests.rs"]
mod tests;
