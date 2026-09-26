// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Prefill budget and KV-cache dtype resolution for serve.
//!
//! Owner: server startup (`met serve`).
//! Invariants:
//! - `PrefillBudget::prefill_budget` is at most 65535 when `block_size > 0`.
//! - `PrefillBudget::max_batch_tokens` is at least `prefill_budget +
//!   max_batch_size`.

use anyhow::Result;

use metrale_config::ModelConfig;

use crate::cli;

pub(crate) struct PrefillBudget {
    pub(crate) prefill_budget: usize,
    pub(crate) max_batch_tokens: usize,
    pub(crate) spec_tokens: usize,
}

pub(crate) fn resolve_prefill_budget(
    args: &cli::ServeArgs,
    ssm_prefill_chunk: usize,
) -> PrefillBudget {
    let spec_tokens = super::preflight::spec_reserve_tokens(args);
    let user_set_prefill = args.max_prefill_tokens != 8192;
    let prefill_budget_pre_hss = if user_set_prefill && args.max_prefill_tokens > 0 {
        args.max_prefill_tokens
    } else if ssm_prefill_chunk > 0 {
        ssm_prefill_chunk
    } else if args.max_prefill_tokens > 0 {
        args.max_prefill_tokens
    } else {
        args.max_seq_len
    };
    let prefill_budget = if args.high_speed_swap {
        let hss_cap_tokens = args.high_speed_swap_cache_blocks_per_seq as usize * args.block_size;
        let hss_chunk_max = hss_cap_tokens.saturating_sub(args.max_batch_size);
        let clamped = prefill_budget_pre_hss.min(hss_chunk_max);
        if clamped < prefill_budget_pre_hss {
            tracing::info!(
                "--high-speed-swap: clamping max_prefill_tokens from {} to {} \
                 (cap {} × bs {} − max_batch_size {}) to keep chunked prefill \
                 within the rolling HBM window",
                prefill_budget_pre_hss,
                clamped,
                args.high_speed_swap_cache_blocks_per_seq,
                args.block_size,
                args.max_batch_size,
            );
        }
        if args.max_seq_len > hss_cap_tokens {
            tracing::info!(
                "--high-speed-swap engaged: cap={} blocks × bs={} = {} tokens HBM-resident, \
                 --max-seq-len={} tokens total. Prefill slides will advance per-layer offload \
                 cursors as the window moves so older blocks stay reachable on disk.",
                args.high_speed_swap_cache_blocks_per_seq,
                args.block_size,
                hss_cap_tokens,
                args.max_seq_len,
            );
        }
        clamped
    } else {
        prefill_budget_pre_hss
    };
    // 2026-09-26: Clamp the chunk to the largest multiple of `block_size` that
    // is at most `CUDA_MAX_GRID_DIM`. This also re-chunks an unchunked prefill
    // (`--max-prefill-tokens 0` with a longer `--max-seq-len`).
    const CUDA_MAX_GRID_DIM: usize = 65535;
    let prefill_budget = if prefill_budget > CUDA_MAX_GRID_DIM && args.block_size > 0 {
        let safe = (CUDA_MAX_GRID_DIM / args.block_size) * args.block_size;
        tracing::warn!(
            "prefill chunk={} exceeds the CUDA grid-Y limit ({}); the prefill \
             kernels map one grid block per token, so a larger chunk overflows \
             grid.y → cuLaunchKernel CUDA_ERROR_INVALID_VALUE. Clamping chunk to \
             {} (largest block-aligned size under the limit). Lower \
             --max-prefill-tokens to silence this.",
            prefill_budget,
            CUDA_MAX_GRID_DIM,
            safe,
        );
        safe
    } else {
        prefill_budget
    };
    // 2026-09-26: `max_batch_tokens` defaults to `prefill_budget +
    // max_batch_size`, raised to `spec_tokens` when that is larger.
    // `METRALE_MAX_BATCH_TOKENS` can raise it further; a value below the
    // default, or one that does not parse, is ignored with a warning.
    let default_max_batch_tokens = (prefill_budget + args.max_batch_size)
        .max(spec_tokens)
        .max(args.max_batch_size);
    let max_batch_tokens = match std::env::var("METRALE_MAX_BATCH_TOKENS") {
        Ok(v) => match v.parse::<usize>() {
            Ok(n) if n >= default_max_batch_tokens => {
                tracing::info!(
                    "METRALE_MAX_BATCH_TOKENS override: {} (default would be {})",
                    n,
                    default_max_batch_tokens
                );
                n
            }
            Ok(n) => {
                tracing::warn!(
                    "METRALE_MAX_BATCH_TOKENS={} ignored — must be >= default {}",
                    n,
                    default_max_batch_tokens
                );
                default_max_batch_tokens
            }
            Err(e) => {
                tracing::warn!("METRALE_MAX_BATCH_TOKENS parse error: {e}");
                default_max_batch_tokens
            }
        },
        Err(_) => default_max_batch_tokens,
    };
    tracing::info!(
        "Prefill config: ssm_prefill_chunk={}, args.max_prefill_tokens={}, prefill_budget={}, max_batch_tokens={}",
        ssm_prefill_chunk,
        args.max_prefill_tokens,
        prefill_budget,
        max_batch_tokens,
    );
    if args.max_prefill_tokens == 0 && args.max_seq_len > 32768 {
        tracing::warn!(
            "--max-prefill-tokens=0 with --max-seq-len={} disables chunked prefill. \
             Long agentic sessions may eventually fail with 'CUDA kernel launch failed (status 1)' \
             when an unchunked prefill exceeds device launch grid limits. \
             Consider --max-prefill-tokens=8192 (default) for sessions that grow past 32K tokens.",
            args.max_seq_len,
        );
    }
    PrefillBudget {
        prefill_budget,
        max_batch_tokens,
        spec_tokens,
    }
}

pub(crate) struct KvCacheConfig {
    pub(crate) effective_kv_dtype_str: String,
    pub(crate) kv_dtype: metrale_cache::kv_cache::KvCacheDtype,
    pub(crate) layer_dtypes: Vec<metrale_cache::kv_cache::KvCacheDtype>,
    pub(crate) hss_cache_blocks_per_seq: Option<u32>,
}

/// 2026-09-26: Where the effective KV cache dtype came from
/// (`resolve_kv_dtype_str`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum KvDtypeSource {
    /// 2026-09-26: Explicit `--kv-cache-dtype`, equal to the MODEL.toml default
    /// or with no MODEL.toml default. Not logged.
    Cli,
    /// 2026-09-26: Explicit `--kv-cache-dtype` that differs from a non-empty
    /// MODEL.toml default: used, with a warning.
    CliMismatchingModelDefault,
    ModelDefault,
    EngineDefault,
}

/// 2026-09-26: Resolve the effective KV cache dtype. An explicit
/// `--kv-cache-dtype` wins, including the engine default value "fp8". An
/// omitted flag uses MODEL.toml `[behavior].default_kv_dtype` when it is
/// non-empty, else `cli::DEFAULT_KV_CACHE_DTYPE`.
pub(crate) fn resolve_kv_dtype_str(
    cli_dtype: Option<&str>,
    model_default: &str,
) -> (String, KvDtypeSource) {
    match (cli_dtype, model_default) {
        (Some(user), md) if md.is_empty() || user == md => (user.to_string(), KvDtypeSource::Cli),
        (Some(user), _) => (user.to_string(), KvDtypeSource::CliMismatchingModelDefault),
        (None, "") => (
            cli::DEFAULT_KV_CACHE_DTYPE.to_string(),
            KvDtypeSource::EngineDefault,
        ),
        (None, md) => (md.to_string(), KvDtypeSource::ModelDefault),
    }
}

pub(crate) fn resolve_kv_cache_config(
    args: &cli::ServeArgs,
    config: &ModelConfig,
    behavior_default_kv_dtype: &str,
    // 2026-09-26: Number of `*.k_scale` tensors in the checkpoint (0 when it
    // ships no FP8 KV scales); only the FP8-KV log below reads it.
    fp8_kv_scale_count: usize,
) -> Result<KvCacheConfig> {
    let (effective_kv_dtype_str, kv_dtype_source) =
        resolve_kv_dtype_str(args.kv_cache_dtype.as_deref(), behavior_default_kv_dtype);
    match kv_dtype_source {
        KvDtypeSource::Cli | KvDtypeSource::EngineDefault => {}
        KvDtypeSource::ModelDefault => tracing::info!(
            "KV cache dtype: {} (from MODEL.toml default_kv_dtype, override with --kv-cache-dtype)",
            effective_kv_dtype_str,
        ),
        KvDtypeSource::CliMismatchingModelDefault => tracing::warn!(
            "KV cache dtype: {} (user override). MODEL.toml recommends '{}' for this \
             model — mismatched KV dtype is a known cause of decode-path corruption \
             (e.g. gemma `<unused>` collapse, mistral character-token loops on NVFP4 KV). \
             Pass --kv-cache-dtype {} to use the recommended value.",
            effective_kv_dtype_str,
            behavior_default_kv_dtype,
            behavior_default_kv_dtype,
        ),
    }
    let kv_dtype: metrale_cache::kv_cache::KvCacheDtype = effective_kv_dtype_str.parse()?;
    if kv_dtype == metrale_cache::kv_cache::KvCacheDtype::Fp8 {
        let has_ckpt_scales = fp8_kv_scale_count > 0;
        if config.fp8_kv_calibration_tokens > 0 {
            if has_ckpt_scales {
                tracing::info!(
                    "FP8 KV online calibration is ON, but this checkpoint already ships {} \
                     per-layer k_scale/v_scale tensors (full-data calibrated, tighter than a \
                     runtime estimate). Prefer --fp8-kv-calibration-tokens 0 to use them directly.",
                    fp8_kv_scale_count,
                );
            } else {
                // 2026-09-26: Each attention layer logs "FP8 KV scales frozen
                // after N tokens (requested M)" when its window closes.
                tracing::info!(
                    "FP8 KV cache with online calibration (checkpoint ships no k/v scales): \
                     accumulating per-tensor K/V amax over the first {} observed tokens \
                     (across requests, readiness probe included) before freezing the scales.{}",
                    config.fp8_kv_calibration_tokens,
                    if args.fp8_kv_calibration_tokens.is_none() {
                        " (auto-enabled from MODEL.toml)"
                    } else {
                        ""
                    },
                );
            }
        } else if has_ckpt_scales {
            tracing::info!(
                "FP8 KV cache using {} per-layer k_scale/v_scale tensors from the checkpoint — \
                 no calibration needed.",
                fp8_kv_scale_count,
            );
        } else {
            tracing::warn!(
                "FP8 KV cache selected but the checkpoint ships NO k_scale/v_scale tensors \
                 (defaulting to 1.0, which silently clips BF16 into E4M3 range [-448, 448] and \
                 destroys dynamic range). Enable --fp8-kv-calibration-tokens 256 for online \
                 calibration, or use --kv-cache-dtype nvfp4/bf16."
            );
        }
    }
    let num_attn_layers = config.num_attention_layers();
    // 2026-09-26: Parsed with `cli::flag_values::KvHighPrecisionLayers`, the
    // parser `validate_serve_args` also uses; a parse error fails the boot.
    let kv_hp_layers: usize = args
        .kv_high_precision_layers
        .parse::<crate::cli::flag_values::KvHighPrecisionLayers>()
        .map_err(|why| {
            anyhow::anyhow!(
                "--kv-high-precision-layers '{}': {why}",
                args.kv_high_precision_layers
            )
        })?
        .resolve(num_attn_layers);
    let kv_hp_layers = match (
        kv_hp_layers,
        crate::main_modules::auto_high_precision_layers(kv_dtype, num_attn_layers),
    ) {
        (0, Some(auto_hp)) => {
            tracing::info!(
                "Auto-enabling --kv-high-precision-layers {} for {} ({}/{} attn layers BF16; \
                 scaled with attn-layer count to keep accumulated turbo quant error tractable)",
                auto_hp,
                effective_kv_dtype_str,
                (auto_hp * 2).min(num_attn_layers),
                num_attn_layers,
            );
            auto_hp
        }
        _ => kv_hp_layers,
    };
    if kv_hp_layers == 0 && kv_dtype != metrale_cache::kv_cache::KvCacheDtype::Bf16 {
        tracing::warn!(
            "⚠ --kv-high-precision-layers is 0: all KV cache layers use {} precision. \
             NVFP4 models may hallucinate or lose coherence at long context. \
             Consider --kv-high-precision-layers max (or 2-5) for better quality.",
            effective_kv_dtype_str,
        );
    }
    let layer_dtypes = crate::main_modules::build_layer_kv_dtypes(
        kv_dtype,
        num_attn_layers,
        kv_hp_layers,
        metrale_cache::kv_cache::KvCacheDtype::Bf16,
    );
    let hss_cache_blocks_per_seq = if args.high_speed_swap {
        Some(args.high_speed_swap_cache_blocks_per_seq)
    } else {
        None
    };
    Ok(KvCacheConfig {
        effective_kv_dtype_str,
        kv_dtype,
        layer_dtypes,
        hss_cache_blocks_per_seq,
    })
}

#[cfg(test)]
mod tests {
    use super::{KvDtypeSource, resolve_kv_dtype_str};

    /// 2026-09-26: An explicit `--kv-cache-dtype fp8` on a model whose MODEL.toml
    /// default is bf16 serves fp8, reported as a mismatch.
    #[test]
    fn explicit_cli_value_equal_to_engine_default_beats_model_default() {
        assert_eq!(
            resolve_kv_dtype_str(Some("fp8"), "bf16"),
            ("fp8".to_string(), KvDtypeSource::CliMismatchingModelDefault)
        );
    }

    #[test]
    fn explicit_cli_value_matching_model_default_is_silent() {
        assert_eq!(
            resolve_kv_dtype_str(Some("bf16"), "bf16"),
            ("bf16".to_string(), KvDtypeSource::Cli)
        );
    }

    #[test]
    fn explicit_cli_value_without_model_default_is_used_as_is() {
        assert_eq!(
            resolve_kv_dtype_str(Some("nvfp4"), ""),
            ("nvfp4".to_string(), KvDtypeSource::Cli)
        );
    }

    #[test]
    fn omitted_flag_falls_back_to_model_default() {
        assert_eq!(
            resolve_kv_dtype_str(None, "bf16"),
            ("bf16".to_string(), KvDtypeSource::ModelDefault)
        );
    }

    #[test]
    fn omitted_flag_without_model_default_uses_engine_default() {
        assert_eq!(
            resolve_kv_dtype_str(None, ""),
            (
                crate::cli::DEFAULT_KV_CACHE_DTYPE.to_string(),
                KvDtypeSource::EngineDefault
            )
        );
    }
}
