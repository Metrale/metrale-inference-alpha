// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: What the `met serve` command line publishes into the kernel-path
//! cells; `serve_flags::publish_kernel_flags` performs the plan.
//!
//! Owner: server startup (`met serve`).
//! Invariants:
//! - `KernelFlagPlan::from_args` reads only its argument: no environment, no
//!   process cells, no logging.
//! - A top-level `None` publishes nothing, so that cell resolves from its
//!   `METRALE_*` variable on first read; publishing a default would seal it.

use crate::cli::ServeArgs;
use crate::cli::flag_values::Tristate;

/// 2026-09-26: The GDN cell, which the command line owns as a whole once any GDN flag is
/// given (`--ssm-h-dtype`, `--gdn-fused-norm`, `--ssm-batched-recurrent`
/// `on|off`, `--exact-verify`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GdnPlan {
    pub h_f16: bool,
    pub h_f16_pool: bool,
    pub fused_norm: bool,
    /// 2026-09-26: `None`: the compiled target's `HARDWARE.toml` default decides,
    /// with `METRALE_SSM_BATCHED_RECURRENT` over it. A hard `false` would turn
    /// off `kernels/hopper`'s default on every serve that names another GDN flag.
    pub batched_recurrent: Option<bool>,
    pub exact_verify: bool,
}

/// 2026-09-26: Every kernel-path selection the command line makes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct KernelFlagPlan {
    /// 2026-09-26: `None` when no GDN flag was given.
    pub gdn: Option<GdnPlan>,
    /// 2026-09-26: Always published: no environment fallback exists.
    pub w4a4_downcast: bool,
    /// 2026-09-26: Only ever on together with `w4a4_downcast`.
    pub w4a4_downcast_wide: bool,
    pub prefill_codispatch: Option<bool>,
    pub prefill_varlen: Option<bool>,
    pub ssm_tail_midchunk: Option<bool>,
    pub hermetic: bool,
}

impl KernelFlagPlan {
    /// 2026-09-26: The plan for a command line that passed `validate_serve_args`.
    pub(crate) fn from_args(args: &ServeArgs) -> Self {
        let batched_recurrent = Tristate::validated(&args.ssm_batched_recurrent).pinned();
        let gdn_given = args.ssm_h_dtype.is_some()
            || args.gdn_fused_norm
            || batched_recurrent.is_some()
            || args.exact_verify;
        let gdn = gdn_given.then(|| {
            // 2026-09-26: The decode `validate_serve_args` checks, so the
            // validator and the published cell read the flag alike; `f16-pool`
            // sets both bits.
            let (h_f16, h_f16_pool) = metrale_model_layers::layers::qwen3_ssm::ssm_h_dtype_bits(
                args.ssm_h_dtype.as_deref(),
            );
            GdnPlan {
                h_f16,
                h_f16_pool,
                fused_norm: args.gdn_fused_norm,
                batched_recurrent,
                exact_verify: args.exact_verify,
            }
        });
        Self {
            gdn,
            w4a4_downcast: args.w4a4_downcast,
            w4a4_downcast_wide: args.w4a4_downcast && args.w4a4_downcast_wide,
            prefill_codispatch: args.prefill_codispatch.then_some(true),
            prefill_varlen: args.prefill_varlen_batch.then_some(true),
            ssm_tail_midchunk: args.no_ssm_tail_midchunk.then_some(false),
            hermetic: args.hermetic,
        }
    }
}

#[cfg(test)]
#[path = "kernel_flag_plan_tests.rs"]
mod tests;
