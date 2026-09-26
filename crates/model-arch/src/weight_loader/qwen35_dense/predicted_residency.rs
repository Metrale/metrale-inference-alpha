// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The derived-weight bytes of the native-FP8 dense route,
//! predicted from the config and the environment before the checkpoint loads.
//!
//! Owner: model-arch weight loader (Qwen3.5 dense).
//! Invariants:
//! - `predicted_derived_bytes` reads no environment and allocates nothing;
//!   `Fp8RouteInputs::from_env` does the reads.
//! - A route whose plan keeps an NVFP4 copy (`METRALE_DENSE_FP8_KEEP_NVFP4`,
//!   a CUTLASS NVFP4 attention lever, `METRALE_ATTN_W4A4`) gets
//!   [`DerivedBytesEstimate::Unavailable`].
//!
//! The server's preflight (`post_load_yardstick` in
//! `serve_phases/preflight/headroom.rs`) adds the prediction to the
//! checkpoint's on-disk size to estimate the memory in use before the KV
//! cache. When the prediction is unavailable, it uses pre-load free memory.
//! Which copies are built is [`fp8_residency::DenseFp8Plan::resolve`]; their
//! sizes are shape arithmetic over the config. The loader tallies NVFP4
//! copies through `DerivedResidency::skip`, never `keep`, so a route that
//! builds them is not predicted.

use metrale_config::ModelConfig;

use super::fp8_residency::{self, RouteEnv, TwinsBuilt};
use metrale_model_layers::layers::qwen3_attention::Fp8TwinSet;
use metrale_model_layers::weight_map::Nvfp4Variant;

/// 2026-09-25: The derived bytes the native-FP8 dense loader will keep, by
/// term.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PredictedDerived {
    /// 2026-09-25: `Fp8Weight::transpose_for_gemm` twins for the projections
    /// [`fp8_residency::DenseFp8Plan`] selects, summed over full-attention
    /// layers.
    pub attn_fp8_twins: u64,
    /// 2026-09-25: The fused `[QKV|Z]` FP8 weight, its block-scale grids, the
    /// `out_proj` block-scale grid and the interleaved `in_proj_ba`, summed
    /// over linear-attention layers (`ssm_concat_bytes`).
    pub ssm_fp8_concat: u64,
    /// 2026-09-25: The E4M3 bytes of the fused `[2*inter, hidden]` gate+up
    /// weight, summed over layers; its scale grid is not counted. 0 when the
    /// fusion is not selected.
    pub ffn_gateup_fused: u64,
    /// 2026-09-25: The `mlp.gate_proj.weight` and `mlp.up_proj.weight` bytes
    /// that `prune_after_load` releases on the same layers. Equal to
    /// [`Self::ffn_gateup_fused`]: the fused weight is those two tensors side
    /// by side.
    pub ffn_gateup_pruned: u64,
    /// 2026-09-25: Which derived families the prediction expects.
    pub twins: TwinsBuilt,
    /// 2026-09-25: The twin set the attention term was priced at.
    pub attn_twin_set: Fp8TwinSet,
}

impl PredictedDerived {
    /// 2026-09-25: Derived bytes above the on-disk checkpoint size, which
    /// `headroom.rs` adds to that size. The fusion enters as a difference,
    /// which is zero: the on-disk size still counts the gate and up tensors
    /// that `prune_after_load` releases once the fused copy exists
    /// (`the_gateup_fusion_is_residency_neutral`).
    pub fn total(&self) -> u64 {
        self.attn_fp8_twins
            + self.ssm_fp8_concat
            + self.ffn_gateup_fused.saturating_sub(self.ffn_gateup_pruned)
    }
}

/// 2026-09-25: A prediction, or the reason there is none.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DerivedBytesEstimate {
    /// 2026-09-25: The native-FP8 dense route will run, with these bytes.
    NativeFp8Dense(PredictedDerived),
    /// 2026-09-25: No prediction; the string says why, and the preflight
    /// carries it in `Yardstick::PreLoadFree`.
    Unavailable(&'static str),
}

impl DerivedBytesEstimate {
    /// 2026-09-25: [`PredictedDerived::total`], or `None` when unavailable.
    pub fn bytes(&self) -> Option<u64> {
        match self {
            Self::NativeFp8Dense(p) => Some(p.total()),
            Self::Unavailable(_) => None,
        }
    }

    pub fn reason(&self) -> Option<&'static str> {
        match self {
            Self::NativeFp8Dense(_) => None,
            Self::Unavailable(why) => Some(*why),
        }
    }
}

/// 2026-09-25: The route conditions of `load_layers` that are known before the
/// checkpoint loads.
#[derive(Clone, Copy, Debug)]
pub struct Fp8RouteInputs {
    /// 2026-09-25: `METRALE_DENSE_FP8=1` (`qwen35_dense::dense_fp8_enabled`).
    pub dense_fp8: bool,
    /// 2026-09-25: `config.tp_world_size.max(1)`; the FP8 dense route needs 1.
    pub tp_size: usize,
    /// 2026-09-25: The variant the config declares (`config_declared_variant`),
    /// or `None` when it declares none.
    pub declared_variant: Option<Nvfp4Variant>,
    /// 2026-09-25: `METRALE_NO_GDN_FP8` is unset, a clause of
    /// `qwen35_dense::gdn_fp8_arm_selected`.
    pub gdn_fp8: bool,
    /// 2026-09-25: Both `qwen3_attention::W8A8_PREFILL_KERNELS` are loaded for
    /// this target; it decides whether the Q and O FP8 twins are planned.
    pub w8a8_prefill_kernels: bool,
    /// 2026-09-25: The environment half of the loader's plan.
    pub route: RouteEnv,
    /// 2026-09-25: `gateup_fused::ffn_gateup_fused()`, which
    /// `qwen35_dense::ffn_gateup_fused_selected` also requires.
    pub ffn_gateup_fused: bool,
}

impl Fp8RouteInputs {
    /// 2026-09-25: Resolve each condition from the environment and the config.
    /// `w8a8_prefill_kernels` comes from the caller, which asks
    /// `qwen3_attention::w8a8_prefill_kernels_loaded` of the loaded kernel set.
    pub fn from_env(config: &ModelConfig, w8a8_prefill_kernels: bool) -> Self {
        Self {
            // 2026-09-25: The predicate of `qwen35_dense::dense_fp8_enabled`.
            dense_fp8: std::env::var("METRALE_DENSE_FP8").as_deref() == Ok("1"),
            tp_size: config.tp_world_size.max(1),
            declared_variant: metrale_model_layers::weight_map::config_declared_variant(config),
            // 2026-09-25: The predicate of `gdn_fp8_arm_selected`.
            gdn_fp8: std::env::var_os("METRALE_NO_GDN_FP8").is_none(),
            w8a8_prefill_kernels,
            route: RouteEnv::from_env(),
            ffn_gateup_fused:
                metrale_model_layers::layers::dense_ffn::gateup_fused::ffn_gateup_fused(),
        }
    }
}

/// 2026-09-25: Bytes the native-FP8 dense loader will derive on top of the
/// checkpoint, or why that is not predicted. Reads only its arguments.
pub fn predicted_derived_bytes(
    config: &ModelConfig,
    route: &Fp8RouteInputs,
) -> DerivedBytesEstimate {
    // 2026-09-25: `factory::loader_for_config` picks `Qwen35DenseWeightLoader`
    // when `is_qwen35_dense()` holds.
    if !config.is_qwen35_dense() {
        return DerivedBytesEstimate::Unavailable("not the Qwen3.5-dense loader");
    }
    if !route.dense_fp8 {
        return DerivedBytesEstimate::Unavailable("METRALE_DENSE_FP8 is not 1");
    }
    if route.tp_size != 1 {
        return DerivedBytesEstimate::Unavailable("--tp-size > 1 takes the NVFP4 route");
    }
    if route.declared_variant != Some(Nvfp4Variant::Fp8Dequanted) {
        return DerivedBytesEstimate::Unavailable(
            "config.json does not declare a block-scaled FP8 checkpoint",
        );
    }
    if route.route.keep_nvfp4 {
        return DerivedBytesEstimate::Unavailable(
            "METRALE_DENSE_FP8_KEEP_NVFP4 restores the pre-#915 fallback copies",
        );
    }

    // 2026-09-25: Decline when the plan keeps an NVFP4 copy (module docs).
    // With `ffn_fp8` true, `ffn_nvfp4` is false, so only `attn_nvfp4` can hold.
    let plan = fp8_residency::DenseFp8Plan::resolve(fp8_residency::DenseFp8Inputs {
        ffn_fp8: true,
        attn_fp8: true,
        keep_nvfp4: false,
        dispatch: route.route.dispatch,
        w8a8_kernels: route.w8a8_prefill_kernels,
        attn_w4a4: route.route.attn_w4a4,
        attn_prefill_q_t: route.route.attn_prefill_q_t,
    });
    if plan.ffn_nvfp4 || plan.attn_nvfp4 {
        return DerivedBytesEstimate::Unavailable(
            "an NVFP4 fallback lever (METRALE_CUTLASS_NVFP4_* / METRALE_ATTN_W4A4) is set",
        );
    }

    let hidden = config.hidden_size;
    let (nh, hd) = (config.num_attention_heads, config.head_dim);
    let nkv = config.num_key_value_heads;
    let q_n = nh * hd * if config.attn_gated { 2 } else { 1 };
    let attn_layers = config.num_attention_layers() as u64;
    let attn_fp8_twins = attn_layers
        * fp8_residency::attn_fp8_twin_bytes(plan.attn_fp8_twins, q_n, nkv * hd, nh * hd, hidden)
            as u64;

    let ssm_layers = config.num_ssm_layers() as u64;
    let ssm_fp8_concat = if route.gdn_fp8 && ssm_layers > 0 {
        ssm_layers * ssm_concat_bytes(config) as u64
    } else {
        0
    };

    // 2026-09-25: The clauses of `qwen35_dense::ffn_gateup_fused_selected` that
    // need no checkpoint: the switch, a dense model, and both widths whole
    // 128-blocks. `inter` falls back as `qwen35_dense::ffn_inter` does.
    let inter = if config.intermediate_size > 0 {
        config.intermediate_size
    } else {
        config.moe_intermediate_size
    };
    let ffn_layers = config.num_hidden_layers as u64;
    let fused_here = route.ffn_gateup_fused
        && config.num_experts == 0
        && inter > 0
        && inter.is_multiple_of(128)
        && hidden.is_multiple_of(128);
    let (ffn_gateup_fused, ffn_gateup_pruned) = if fused_here {
        let (w, _scales) = fp8_residency::ffn_gateup_fused_parts(hidden, inter);
        (ffn_layers * w as u64, ffn_layers * w as u64)
    } else {
        (0, 0)
    };

    DerivedBytesEstimate::NativeFp8Dense(PredictedDerived {
        attn_fp8_twins,
        ssm_fp8_concat,
        ffn_gateup_fused,
        ffn_gateup_pruned,
        twins: TwinsBuilt {
            ffn_nvfp4: false,
            attn_nvfp4: false,
            attn_fp8: plan.attn_fp8_twins.any() && attn_layers > 0,
            ssm_fp8_concat: ssm_fp8_concat > 0,
            ffn_gateup_fused: ffn_gateup_fused > 0,
        },
        attn_twin_set: plan.attn_fp8_twins,
    })
}

/// 2026-09-25: What one native-FP8 GDN layer keeps beyond the checkpoint
/// bytes: the four buffers the loader's native FP8 GDN arm adopts and counts
/// with `residency.keep`.
///
/// - The fused `[QKV|Z]` E4M3 weight from `concat_fp8_block_scaled`.
/// - Its FP32 block-scale grid: the two source grids side by side, so the sum
///   of two grids, not one grid over the fused N.
/// - The `out_proj` block-scale grid.
/// - `in_proj_ba`, the `[2*nv, hidden]` BF16 interleave of `in_proj_a` and
///   `in_proj_b`.
///
/// The per-projection source grids are freed after the concat, so they are
/// not counted.
fn ssm_concat_bytes(config: &ModelConfig) -> usize {
    let hidden = config.hidden_size;
    let block_grid = |n: usize, k: usize| n.div_ceil(128) * k.div_ceil(128) * 4;
    let qkv_n = config.ssm_qkv_size();
    let z_n = config.ssm_z_size();
    let qkvz_bytes = config.ssm_qkvz_size() * hidden;
    let qkvz_scale_bytes = block_grid(qkv_n, hidden) + block_grid(z_n, hidden);
    let out_scale_bytes = block_grid(hidden, config.ssm_z_size());
    let ba_bytes = config.linear_num_value_heads * 2 * hidden * 2;
    qkvz_bytes + qkvz_scale_bytes + out_scale_bytes + ba_bytes
}

#[cfg(test)]
#[path = "predicted_residency_tests.rs"]
mod tests;
