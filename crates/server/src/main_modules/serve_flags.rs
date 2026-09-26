// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Publish the command line's kernel-path selections into the
//! process-wide cells the kernel crates read, and log what is in force.
//!
//! Owner: server startup (`met serve`).
//! Invariants:
//! - A cell is published only when the command line gave one of its flags;
//!   otherwise it resolves from its `METRALE_*` fallback on first read. Once
//!   one of the four GDN flags is given, the GDN cell is published whole.
//! - The two w4a4 cells (no fallback) and `--ssm-rollback-mode` (explicit
//!   clap default) are published on every serve.

use crate::cli;

/// 2026-09-26: Publish the command line's kernel-path selections to the crates
/// that dispatch on them, then log the resolved values. Called once per
/// process, from `startup`, before the first model is built.
pub(crate) fn publish_kernel_flags(args: &cli::ServeArgs) {
    let plan = super::kernel_flag_plan::KernelFlagPlan::from_args(args);
    // 2026-09-26: The GDN cell (`GdnFlags`) is one `OnceLock` fed by four flags:
    // `--ssm-h-dtype`, `--gdn-fused-norm`, `--ssm-batched-recurrent` and
    // `--exact-verify`. With none of them given nothing is published and
    // `gdn_flags::flags()` resolves from the environment on first read. With
    // any of them given the command line sets the whole cell; an absent
    // `--ssm-batched-recurrent` still takes the target default, which
    // `METRALE_SSM_BATCHED_RECURRENT` can override.
    if let Some(gdn) = plan.gdn {
        let flags = metrale_model_layers::layers::qwen3_ssm::GdnFlags {
            h_f16: gdn.h_f16,
            h_f16_pool: gdn.h_f16_pool,
            fused_norm: gdn.fused_norm,
            batched_recurrent: gdn.batched_recurrent.unwrap_or(
                metrale_model_layers::layers::ops::target_defaults::resolved()
                    .ssm_batched_recurrent
                    .value,
            ),
            exact_verify: gdn.exact_verify,
        };
        let in_force = metrale_model_layers::layers::qwen3_ssm::gdn_flags::set_from_cli(flags);
        if in_force != flags {
            tracing::warn!(
                "GDN flags were already resolved from the environment ({in_force:?}); \
                 the command line's ({flags:?}) did NOT take effect"
            );
        }
        warn_shadowed_env();
    }
    // 2026-09-26: `--w4a4-downcast` is always published: it has no environment
    // fallback, and the model build reads it (`w4a16_gemv_tiers.rs`).
    let w4a4 = metrale_model_layers::layers::ops::w4a4_proj::set_w4a4_downcast_from_cli(
        plan.w4a4_downcast,
    );
    if w4a4 != plan.w4a4_downcast {
        tracing::warn!(
            "w4a4-downcast was already resolved ({w4a4}); the command line's \
             ({}) did NOT take effect",
            plan.w4a4_downcast
        );
    }
    let wide = metrale_model_layers::layers::ops::w4a4_proj::set_w4a4_wide_from_cli(
        plan.w4a4_downcast_wide,
    );
    if args.w4a4_downcast_wide && !wide {
        tracing::warn!("--w4a4-downcast-wide needs --w4a4-downcast; it did NOT take effect");
    }
    // 2026-09-26: Published only when given; otherwise
    // `prefill_codispatch_enabled()` reads `METRALE_PREFILL_CODISPATCH` on
    // first use.
    if let Some(on) = plan.prefill_codispatch {
        let in_force = metrale_model_layers::layers::ops::set_prefill_codispatch_from_cli(on);
        if in_force != on {
            tracing::warn!(
                "prefill-codispatch was already resolved ({in_force}); the command \
                 line's ({on}) did NOT take effect"
            );
        }
    }
    // 2026-09-26: `--ssm-rollback-mode` has an explicit clap default
    // ("snapshot"), so it is published on every serve. `validate_serve_args`
    // parsed it with the same `FromStr`, so this parse cannot fail.
    let rollback = args
        .ssm_rollback_mode
        .parse::<metrale_model_layers::ssm_reserve::SsmRollbackMode>()
        .expect("validated by validate_serve_args");
    let rollback_in_force = metrale_model_layers::ssm_reserve::set_ssm_rollback_mode(rollback);
    if rollback_in_force != rollback {
        tracing::warn!(
            "ssm-rollback-mode was already resolved ({rollback_in_force:?}); the command \
             line's ({rollback:?}) did NOT take effect"
        );
    }
    // 2026-09-26: `auto`, the clap default, publishes nothing: the
    // `METRALE_SSM_DECODE_RING` fallback stays reachable and the preflight
    // auto-fit can publish the depth it fits. An explicit N is published here,
    // before preflight, and the auto-fit leaves an explicit depth unchanged.
    if let Some(slots) =
        metrale_model_layers::ssm_reserve::parse_decode_ring_slots(&args.ssm_decode_ring_slots)
            .expect("validated by validate_serve_args")
    {
        let in_force = metrale_model_layers::ssm_reserve::set_decode_ring_slots(slots);
        if in_force != slots {
            tracing::warn!(
                "ssm-decode-ring-slots was already resolved ({in_force}); the command                  line's ({slots}) did NOT take effect"
            );
        }
    }
    // 2026-09-26: `--prefill-varlen-batch` has its own cell, separate from the
    // GDN cell. Published only when given; otherwise `METRALE_PREFILL_VARLEN`
    // decides.
    if let Some(varlen) = plan.prefill_varlen {
        let in_force = metrale_model_layers::layers::ops::set_prefill_varlen_from_cli(varlen);
        if in_force != varlen {
            tracing::warn!(
                "prefill-varlen-batch was already resolved ({in_force}); the command \
                 line's ({varlen}) did NOT take effect"
            );
        }
        if std::env::var_os("METRALE_PREFILL_VARLEN").is_some() {
            tracing::warn!(
                "METRALE_PREFILL_VARLEN is set but was OVERRIDDEN: `--prefill-varlen-batch` \
                 on the command line owns the decision. Drop the flag to let the \
                 environment decide."
            );
        }
    }
    // 2026-09-26: `None` unless `--no-ssm-tail-midchunk` was given. `None`
    // publishes nothing, so `METRALE_SSM_TAIL_MIDCHUNK` decides.
    metrale_gpu_runtime::set_ssm_tail_midchunk(plan.ssm_tail_midchunk);
    // 2026-09-26: Published before any prefill: `hermetic_enabled` caches its
    // first answer. Only `true` is published.
    metrale_gpu_runtime::set_hermetic(plan.hermetic);
    // 2026-09-26: Both lines below log resolved values, not the raw arguments,
    // because most of them can also come from the environment. `summary_line`
    // names each per-target serving default, its value, and ` (env)` when the
    // environment set it.
    tracing::info!(
        "{}",
        metrale_model_layers::layers::ops::target_defaults::summary_line()
    );
    let gdn = metrale_model_layers::layers::qwen3_ssm::gdn_flags::flags();
    tracing::info!(
        "kernel flags: ssm_h_dtype={} gdn_fused_norm={} ssm_batched_recurrent={} \
         exact_verify={} ssm_tail_midchunk={} mtp_gate={} ssm_rollback_mode={:?} \
         ssm_decode_ring_slots={} prefill_varlen_batch={}",
        if gdn.h_f16 { "f16" } else { "f32" },
        gdn.fused_norm,
        gdn.batched_recurrent,
        // 2026-09-26: Resolved: an FP16 h-state turns exact verify off.
        gdn.verify_exact_active(),
        metrale_gpu_runtime::ssm_tail_midchunk_enabled(),
        if crate::scheduler::levers::resolve_mtp_gate_force(args.mtp_gate_force()) {
            "force"
        } else {
            "auto"
        },
        metrale_model_layers::ssm_reserve::ssm_rollback_mode(),
        // 2026-09-26: `auto` unless `--ssm-decode-ring-slots` published a
        // depth; preflight logs the depth it fits.
        match metrale_model_layers::ssm_reserve::published_decode_ring_slots() {
            Some(slots) => slots.to_string(),
            None => "auto".to_string(),
        },
        // 2026-09-26: Resolved, so it may come from the environment.
        metrale_model_layers::layers::ops::prefill_varlen_enabled(),
    );
}

/// 2026-09-26: Environment variables `warn_shadowed_env` reports as overridden
/// once any GDN flag is given, because the command line then sets the whole
/// GDN cell. `METRALE_SSM_H_FP16` and `METRALE_GDN_FUSED_NORM` are then
/// ignored. `METRALE_SSM_BATCHED_RECURRENT` still applies when
/// `--ssm-batched-recurrent` is absent (`target_defaults::resolved`), although
/// it is listed here.
const SHADOWED_BY_CLI: &[(&str, &str)] = &[
    ("METRALE_SSM_H_FP16", "--ssm-h-dtype f16"),
    ("METRALE_GDN_FUSED_NORM", "--gdn-fused-norm"),
    ("METRALE_SSM_BATCHED_RECURRENT", "--ssm-batched-recurrent"),
];

fn warn_shadowed_env() {
    for (var, flag) in SHADOWED_BY_CLI {
        if std::env::var_os(var).is_some() {
            tracing::warn!(
                "{var} is set but was OVERRIDDEN: a GDN flag on the command line publishes \
                 all three kernel selections at once, so this variable did not decide \
                 anything. Pass `{flag}` instead, or drop the GDN flags entirely to let the \
                 environment decide."
            );
        }
    }
}
