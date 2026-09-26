// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Boot preconditions for an FP16 GDN h-state
//! (`--ssm-h-dtype f16` or `f16-pool`).
//!
//! Owner: server startup (`met serve`).
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_config::ModelConfig;

use crate::cli;

/// 2026-09-26: Refuse the configurations the FP16 h-state kernels do not
/// cover; an FP32 h-state kernel run over an FP16 h-state gives wrong output
/// rather than an error. `Ok` at once when the FP16 h-state is off or the
/// model has no SSM layers.
///
/// With `f16-pool` the h pools hold 2 bytes per element and prefill runs its
/// FP32 kernels over a per-slot FP32 staging blob (`qwen3_ssm::ssm_h_fp16`);
/// the refusals are the same for `f16` and `f16-pool`.
pub(super) fn ssm_h_fp16_preconditions(args: &cli::ServeArgs, config: &ModelConfig) -> Result<()> {
    // 2026-09-26: `ssm_h_fp16_enabled` is the resolution the kernels
    // dispatch on: `--ssm-h-dtype`, else `METRALE_SSM_H_FP16`.
    if !metrale_model_layers::layers::qwen3_ssm::ssm_h_fp16_enabled()
        || config.num_ssm_layers() == 0
    {
        return Ok(());
    }
    // 2026-09-26: The DFlash verify width is gamma + 1, and
    // `MAX_F16_TWIN_DFLASH_GAMMA` is the largest gamma whose width has an FP16
    // twin; gamma 16 reaches the FP32-only `gated_delta_rule_wy17`. An unset
    // `--dflash-gamma` passes: the drafter head takes `default_dflash_gamma`,
    // which is clamped to that bound.
    if args.dflash
        && args
            .dflash_gamma
            .is_some_and(|g| g > metrale_model_layers::layers::qwen3_ssm::MAX_F16_TWIN_DFLASH_GAMMA)
    {
        anyhow::bail!(
            "--ssm-h-dtype f16 with --dflash-gamma {} : the verify width is gamma + 1 = {}, \
             which dispatches gated_delta_rule_wy17 — the one WY family with no FP16 h-state \
             twin, and the one that strides its h intermediates in FP32 elements. An FP32 \
             kernel over an FP16 h-state emits fluent garbage rather than an error. Use \
             --dflash-gamma <= {}, drop --dflash, or use --ssm-h-dtype f32.",
            args.dflash_gamma.unwrap_or_default(),
            args.dflash_gamma.unwrap_or_default() + 1,
            metrale_model_layers::layers::qwen3_ssm::MAX_F16_TWIN_DFLASH_GAMMA,
        );
    }
    // 2026-09-26: MTP verify at up to 3 drafts (K <= 4 rows) runs the
    // `gated_delta_rule_wy{2,3,4}_f16` and `wy{2,3}_resident_f16` twins.
    if args.self_speculative || args.ngram_speculative {
        anyhow::bail!(
            "--ssm-h-dtype f16 supports --speculative (MTP) only. The self-speculative and \
             ngram-speculative verify paths still write the h-state as FP32, and an FP32 \
             kernel over an FP16 pool produces fluent garbage rather than an error. Run \
             without --self-speculative/--ngram-speculative, or use --ssm-h-dtype f32."
        );
    }
    if args.speculative && args.resolved_num_drafts() > 3 {
        anyhow::bail!(
            "--ssm-h-dtype f16 supports up to 3 drafts (K <= 4 verify rows); --num-drafts is \
             {}. Wider verify widths dispatch the wyN (K=5..8) / wy17 arms, which have no \
             FP16 h-state twin. Lower --num-drafts to 3, or use --ssm-h-dtype f32.",
            args.resolved_num_drafts()
        );
    }
    if !metrale_model_layers::layers::qwen3_ssm::gdn_fused_norm_enabled() {
        anyhow::bail!(
            "--ssm-h-dtype f16 requires --gdn-fused-norm — the non-fused decode arms \
             (gated_delta_rule_decode, ..._decode_f32_strided) have no FP16 twin in stage 1."
        );
    }
    if std::env::var("METRALE_GDN_FUSED_CONV").ok().as_deref() == Some("1") {
        anyhow::bail!(
            "--ssm-h-dtype f16 is incompatible with METRALE_GDN_FUSED_CONV=1 —              gated_delta_rule_decode_f32_conv_norm has no FP16 twin in stage 1."
        );
    }
    if config.linear_key_head_dim != 128 || config.linear_value_head_dim != 128 {
        anyhow::bail!(
            "--ssm-h-dtype f16 needs linear head dims 128/128 (the FP16 twins size their shared              memory for k_dim == 128); this model is {}/{}",
            config.linear_key_head_dim,
            config.linear_value_head_dim
        );
    }
    if metrale_model_layers::layers::qwen3_ssm::ssm_h_f16_pool_enabled() {
        tracing::info!(
            "--ssm-h-dtype f16-pool: GDN h-state stored FP16 AND every h pool SIZED at 2 \
             bytes/element. Prefill runs its unchanged FP32 kernels over a per-slot FP32 \
             staging blob and narrows back, so the pool holds FP16 at all times. NUMERICS: \
             the recurrence now carries FP16 state across prefill chunk boundaries as well \
             as decode steps, with round-to-nearest-even and no stochastic rounding — do NOT \
             publish a number from this mode without ssm-state-poisoning-gate, decode-floor, \
             bfcl-subset and the agentic gate."
        );
    } else {
        tracing::info!(
            "--ssm-h-dtype f16: GDN h-state stored FP16 during decode AND MTP verify (pool \
             stays FP32-sized; prefill unchanged). Scan replica at n=128: 183 -> 84 ms/step."
        );
    }
    Ok(())
}
