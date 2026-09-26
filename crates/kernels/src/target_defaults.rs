// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The serving defaults baked from the compiled target's
//! `kernels/<hw>/HARDWARE.toml` `[defaults]` table.
//!
//! Each field is a lever whose default is the target's declaration. The
//! metrale-model-layers resolver (`layers::ops::target_defaults::resolve`)
//! lets a parseable environment variable override it and formats the
//! `target defaults (<hw>): …` line naming every resolved value.
//!
//! Owner: kernels crate (target defaults).
//! Invariants:
//! - A `[defaults]` key with no arm in `build_defaults::parse_defaults`
//!   panics the build.
//! - A target without a `[defaults]` table, or a key it omits, gets
//!   `build_defaults::baseline`.

/// 2026-09-25: One compiled target's serving defaults. A `const` that
/// `build.rs` writes into `OUT_DIR/target_ptx.rs` and `lib.rs` `include!`s,
/// so [`crate::TARGET_DEFAULTS`] is known at compile time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TargetDefaults {
    /// 2026-09-25: The `kernels/<hw>` directory this binary was built for
    /// (`METRALE_TARGET_HW`, else `build_diagnose::DEFAULT_HW`). It is set even
    /// when that tree has no HARDWARE.toml: `read_defaults` then returns
    /// `baseline(hw)`.
    pub hw: &'static str,
    /// 2026-09-25: Upper edge of the BF16 decode head's batched-GEMV band
    /// (metrale-model-engine `model/trait_impl/lm_head_batched.rs`). The
    /// resolver clamps it to `DENSE_GEMV_BATCHM_MAX_M` (`resolve_batchm_max`).
    /// hopper declares 16; gb10, b200 and b300 declare the baseline 8.
    pub lm_head_batchm_max: u32,
    /// 2026-09-25: Batched recurrent launch on the GDN decode path
    /// (metrale-model-layers `layers/qwen3_ssm/gdn_flags.rs`). True on
    /// hopper.
    pub ssm_batched_recurrent: bool,
    /// 2026-09-25: The tensor-core GDN chunked-prefill family, including the
    /// `gated_delta_rule_chunk_delta_h_tcfuse_x2` state spine
    /// (`kernels/hopper/common/gated_delta_rule_chunk_tc.cu`), selected in
    /// metrale-model-layers `layers/ops/ssm_gdn_a3.rs`. True on hopper.
    pub gdn_prefill_tc: bool,
    /// 2026-09-25: `dense_gemm_ba_gates_prefill_hopper`, one CTA per token,
    /// for the SSM BA projection and GDN gate transforms
    /// (metrale-model-layers `layers/ops/ssm_ba_gates_hopper.rs`). True on
    /// hopper. Its source exists only under `kernels/hopper`, so the row has
    /// no effect elsewhere. `SSM-BA-GATES-ATTRIBUTION.md` holds the
    /// measurements.
    pub ssm_ba_gates_hopper: bool,
    /// 2026-09-25: `per_token_group_quant_fp8_hopper`
    /// (`kernels/hopper/common/fp8_act_quant_hopper.cu`) for the per-token FP8
    /// activation quantizer. True on hopper. The twin takes a launch only when
    /// its grid reaches the CTA floor in metrale-model-layers
    /// `layers/ops/fp8_act_quant_floor.rs`. Its source exists only under
    /// `kernels/hopper`, so the row has no effect elsewhere.
    /// `FP8-ACT-QUANT-ATTRIBUTION.md` holds the measurements.
    pub fp8_act_quant_hopper: bool,
    /// 2026-09-25: Split SiLU+down on the decode path
    /// (`ModelLevers::decode_split_silu`). True on every declaring target;
    /// `METRALE_NO_DECODE_SPLIT_SILU` turns it off.
    pub decode_split_silu: bool,
    /// 2026-09-25: How the paged-decode attention path picks its KV split
    /// count: `legacy`, `auto` or a pinned count, parsed by
    /// [`crate::attn_splitk::parse`]. hopper declares `auto`; every other
    /// target declares or defaults to `legacy`.
    pub attn_decode_splitk: &'static str,
    /// 2026-09-25: The `w8a16_gemm_m16` tensor-core tier on the dense-FFN
    /// decode arm (metrale-model-layers `layers/dense_ffn_m16_tc.rs`). False
    /// on every target, hopper included; the hopper HARDWARE.toml records
    /// why.
    pub ffn_m16_tc: bool,
    /// 2026-09-25: The `w8a16_gemm_m16{,_strided}` tensor-core tiers on the
    /// decode Q/K/V and o_proj projections. True on hopper.
    pub attn_m16_tc: bool,
    /// 2026-09-25: The `dense_gemm_m16_bf16` tensor-core arm on the BF16
    /// decode head (metrale-model-engine `model/trait_impl/lm_head_batched.rs`).
    /// True on hopper. It reassociates the K reduction relative to
    /// `dense_gemv_bf16`, so a near-tie argmax can change.
    pub lm_head_m16_tc: bool,
    /// 2026-09-25: `w8a16_gemv_batch16_ncol{2,4}` on the decode attention
    /// projections (metrale-model-layers
    /// `layers/qwen3_attention/attn_ncol_gemv.rs`). False on every target.
    pub attn_ncol_gemv: bool,
    /// 2026-09-25: One fused `[gate | up]` GEMM at `N = 2 * intermediate`
    /// instead of two at `N = intermediate` (metrale-model-layers
    /// `layers/dense_ffn_gateup_fused.rs`). True only on hopper; its
    /// strided-SiLU consumer (`kernels/hopper/common/silu_mul_strided.cu`)
    /// exists only in the hopper tree.
    pub ffn_gateup_fused: bool,
    /// 2026-09-25: Upper `M` for the W8A8 block-scaled dense-FFN prefill on a
    /// widening projection (`n > k`: gate/up), read in metrale-model-layers
    /// `layers/dense_ffn_w8a8_prefill.rs`. `u32::MAX` means no cap. gb10
    /// declares 64; its HARDWARE.toml records the measurement.
    pub w8a8_prefill_max_m_widening: u32,
    /// 2026-09-25: Upper `M` for the same path on a narrowing projection
    /// (`n <= k`: down). gb10 declares 384. See
    /// [`Self::w8a8_prefill_max_m_widening`].
    pub w8a8_prefill_max_m_narrowing: u32,
}
