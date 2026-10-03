// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The pre-load reserve preflight: the device memory the SSM
//! pools, snapshot regions, decode-rollback ring, buffer arena and CUDA
//! headroom will need, refused before any weight loads when it does not fit.
//! GPU init and the post-load audit are re-exported from sub-modules.
//!
//! Owner: server startup (`met serve`).
//! Invariants:
//! - `preflight_reserve` returns `Ok` only when `inference_reserve +
//!   buffer_arena_bytes` is at most `free_mem`.

use anyhow::Result;

use metrale_config::ModelConfig;

use crate::cli;

mod decode_ring;
mod gpu_backend;
mod headroom;
mod mamba2_spec;
mod per_sequence_state;
mod post_load_audit;
mod refusal;
pub(crate) mod reserve_plan;
mod runtime_headroom;
mod ssm_h_fp16;
#[cfg(any(feature = "cuda", feature = "metal"))]
pub(crate) use gpu_backend::init_gpu_backend;
pub(crate) use headroom::PostLoadInputs;
pub(crate) use post_load_audit::post_load_memory_audit;
use {
    mamba2_spec::refuse_speculation_over_mamba2, per_sequence_state::per_sequence_reserve,
    ssm_h_fp16::ssm_h_fp16_preconditions,
};

pub(crate) struct ReservePreflight {
    /// 2026-10-01: At the slot ceiling (`SlotRequest::ceiling`).
    pub(crate) inference_reserve: usize,
    pub(crate) buffer_arena_bytes: usize,
    pub(crate) gdn_two_phase_bytes: usize,
    pub(crate) ssm_prefill_chunk: usize,
    pub(crate) max_batch_tokens_pre: usize,
    /// 2026-10-01: The reserve at any slot count, for the build (`--max-batch-size auto`).
    pub(crate) plan: reserve_plan::ReservePlan,
}

pub(crate) fn preflight_reserve(
    args: &cli::ServeArgs,
    config: &ModelConfig,
    free_mem: usize,
    // 2026-09-26: What the decode-ring auto-fit needs to predict post-load KV
    // headroom (`headroom::post_load_yardstick`). The caller gathers it: none
    // of it is in `args` or `config`.
    post_load: &PostLoadInputs<'_>,
) -> Result<ReservePreflight> {
    // 2026-09-26: `args.dflash` counts as speculative here because
    // `TransformerModel::new` allocates the SSM rollback pools whenever DFlash
    // capture layers exist (`has_mtp` includes `dflash_kgamma > 0`).
    let spec_on_pool = args.speculative_proposer_requested();
    refuse_speculation_over_mamba2(args, config)?;
    ssm_h_fp16_preconditions(args, config)?;
    // 2026-09-26: A DFlash serve's verify pools are γ + 1 rows wide on every
    // slot. 2026-09-30: γ is the one the build sizes them for
    // (`ServeArgs::serve_dflash_gamma`); until then this peeked the drafter's block
    // size and ignored a pinned `--dflash-gamma`.
    let pool_num_drafts = if args.dflash {
        args.serve_dflash_gamma()
    } else {
        args.resolved_num_drafts()
    };
    // 2026-10-01: The slot count the preflight checks: `--max-batch-size N`, or `auto`'s ceiling
    // (`SlotRequest::ceiling`); the build sizes KV at the count it resolves (`reserve_plan.rs`).
    let ceiling = args.max_batch_size.ceiling();
    // 2026-09-30: Rollback mode is `--ssm-rollback-mode`, published by `serve_flags` before this
    // runs; the h narrowing is `--ssm-h-dtype f16-pool`.
    let h_f16_pool = metrale_model_layers::layers::qwen3_ssm::ssm_h_f16_pool_enabled();
    let rollback = metrale_model_layers::ssm_reserve::ssm_rollback_mode();
    // 2026-09-30: One layer's h (FP32 width) and conv unit, from the pool plan.
    let unit = metrale_model_layers::ssm_reserve::PoolPlan::new(
        config,
        &metrale_model_layers::ssm_reserve::pool_counts(
            &metrale_model_layers::ssm_reserve::PoolShape {
                max_slots: 1,
                spec: spec_on_pool,
                num_intermediates: pool_num_drafts + 1,
                num_drafts: pool_num_drafts,
                uniform_h: args.dflash,
                rollback,
            },
        ),
        h_f16_pool,
    )?;
    let (h_state_bytes, conv_state_bytes) = (unit.h_f32_unit, unit.conv_unit);
    let spec_tokens_pre = spec_reserve_tokens(args);
    // 2026-09-26: An SSM model prefills in chunks of `--max-prefill-tokens`
    // when it is set to anything but 8192 (and above 0), else of 8192; either
    // way at most `max_seq_len`. The chunk bounds `max_batch_tokens_pre`, which
    // sizes the buffer arena and the GDN two-phase term below.
    let ssm_prefill_chunk: usize = if config.num_ssm_layers() > 0 {
        if args.max_prefill_tokens != 8192 && args.max_prefill_tokens > 0 {
            args.max_seq_len.min(args.max_prefill_tokens)
        } else {
            args.max_seq_len.min(8192)
        }
    } else {
        0
    };
    let user_set_prefill_pre = args.max_prefill_tokens != 8192;
    let prefill_budget_pre = if user_set_prefill_pre && args.max_prefill_tokens > 0 {
        args.max_prefill_tokens
    } else if ssm_prefill_chunk > 0 {
        ssm_prefill_chunk
    } else if args.max_prefill_tokens > 0 {
        args.max_prefill_tokens
    } else {
        args.max_seq_len
    };
    let max_batch_tokens_pre = prefill_budget_pre.max(spec_tokens_pre).max(ceiling);
    let buffer_arena_bytes = metrale_gpu_runtime::buffers::BufferSizes::from_config(
        config,
        max_batch_tokens_pre,
        args.max_seq_len,
        args.block_size,
        ceiling,
    )
    .total_bytes();
    // 2026-09-26: Marconi snapshot slots, from
    // `ssm_reserve::marconi_snapshot_slots`, which `TransformerModel::new` also
    // calls: 0 while prefix caching is inactive, unless
    // `METRALE_SSM_MARCONI_FULL` is present.
    let marconi = metrale_model_layers::ssm_reserve::marconi_snapshot_slots(
        args.ssm_cache_slots,
        metrale_model_layers::ssm_reserve::prefix_caching_active(
            args.prefix_caching_enabled(),
            config.kv_only_prefix_cache_is_safe(),
        ),
    );
    if let Some(reason) = marconi.skip_reason {
        tracing::info!(
            "SSM snapshot pool: Marconi region SKIPPED ({}) — {} slot(s) x {} layer(s) \
             = {} MB not reserved (restore with --enable-prefix-caching, or \
             METRALE_SSM_MARCONI_FULL to over-reserve)",
            reason,
            args.ssm_cache_slots,
            config.num_ssm_layers(),
            (args.ssm_cache_slots * config.num_ssm_layers() * (h_state_bytes + conv_state_bytes))
                / (1024 * 1024),
        );
    }
    // 2026-09-26: One sequence's SSM state across all SSM layers. Marconi
    // reserves one per cache slot; a decode-ring slot is one per batch
    // sequence (`decode_ring::slot_bytes`).
    let per_seq_blob = config.num_ssm_layers() * (h_state_bytes + conv_state_bytes);
    let marconi_bytes = marconi.slots * per_seq_blob;
    let gdn_two_phase_bytes: usize = {
        let key_dim = config.linear_num_key_heads * config.linear_key_head_dim;
        let value_dim = config.linear_num_value_heads * config.linear_value_head_dim;
        let nv = config.linear_num_value_heads;
        let conv_dim = key_dim * 2 + value_dim;
        if conv_dim > 0 && config.num_ssm_layers() > 0 {
            let sl = max_batch_tokens_pre;
            sl * conv_dim * 2 + sl * nv * 2 * 4 + sl * value_dim * 2 + sl * value_dim * 2
        } else {
            0
        }
    };
    // 2026-10-01: Every reserve term as a function of the slot count (`reserve_plan.rs`): the
    // SSM pool as `SsmStatePool::new` allocates it (M5, dummy slots included), the f16 staging
    // arena and the replay ring over the pool's slots, Marconi, the GDN two-phase scratch, the
    // runtime headroom (`runtime_headroom.rs`, which replaced the flat 4 GiB / 512 MiB
    // `cuda_headroom`), the per-sequence state and the decode-rollback ring. The preflight checks
    // the slot ceiling; the build sizes KV at the count it resolves.
    let per_sequence_bytes = per_sequence_reserve(args, config) / ceiling.max(1);
    let mut plan = reserve_plan::ReservePlan {
        config: config.clone(),
        spec: spec_on_pool,
        num_drafts: pool_num_drafts,
        uniform_h: args.dflash,
        rollback,
        h_f16_pool,
        marconi_bytes,
        gdn_two_phase_bytes,
        budget_bytes: (post_load.total_mem as f64 * args.gpu_memory_utilization) as usize,
        per_sequence_bytes,
        ring_slots: 0,
        per_seq_blob,
    };
    let at_ceiling = plan.terms(ceiling)?;
    let fixed_reserve = at_ceiling.fixed();
    let ssm_pool_bytes = at_ceiling.ssm_pool;
    let ssm_h_stage_bytes = at_ceiling.ssm_h_stage;
    let runtime_headroom = at_ceiling.runtime;
    // 2026-09-26: Decode-rollback ring: the requested depth, then the largest
    // depth that fits (`decode_ring::autofit`). A shrunk depth is published
    // (`set_decode_ring_slots`) so `TransformerModel::new` allocates the depth
    // reserved here.
    let ring_requested = decode_ring::requested_slots(args, config);
    let ring_slot_bytes = decode_ring::slot_bytes(args, per_seq_blob);
    // 2026-09-26: The auto-fit's yardstick: predicted post-load KV headroom
    // where the load's residency can be predicted, else pre-load free memory
    // (`headroom.rs`).
    let yardstick =
        headroom::post_load_yardstick(args, config, post_load, fixed_reserve, buffer_arena_bytes);
    let fit = decode_ring::autofit(
        args,
        ring_requested,
        ring_slot_bytes,
        per_seq_blob,
        fixed_reserve + buffer_arena_bytes,
        free_mem,
        &yardstick,
    );
    tracing::info!("{}", fit.decision);
    if let Some(warning) = &fit.warning {
        tracing::warn!("SSM decode-rollback ring auto-fit — {}", warning);
    }
    let ssm_snapshot_bytes = marconi_bytes + fit.slots * ring_slot_bytes;
    plan.ring_slots = fit.slots;
    let inference_reserve: usize = plan.inference_reserve(ceiling)?;
    let total_reserve = inference_reserve + buffer_arena_bytes;
    if total_reserve > free_mem {
        return Err(refusal::reserve_refusal(
            args,
            config,
            refusal::Refusal {
                total_reserve,
                free_mem,
                seq_len_independent: ssm_pool_bytes
                    + ssm_h_stage_bytes
                    + ssm_snapshot_bytes
                    + runtime_headroom.total(),
                ring_requested,
                ring_slots: fit.slots,
                per_seq_blob,
                ring_pinned: metrale_model_layers::ssm_reserve::published_decode_ring_slots()
                    .is_some(),
            },
        ));
    }
    tracing::info!(
        "Preflight reserve: inference={} MB, buffer_arena={} MB (pre-load free: {:.1} GB); {}",
        inference_reserve / (1024 * 1024),
        buffer_arena_bytes / (1024 * 1024),
        free_mem as f64 / (1024.0 * 1024.0 * 1024.0),
        decode_ring::formula(fit.slots, ceiling, per_seq_blob),
    );
    // 2026-10-01: The named terms at the ceiling (the debug breakdown before 2026-10-01).
    let t = plan.terms(ceiling)?;
    let mb = |b: usize| b / (1024 * 1024);
    tracing::info!(
        "Preflight reserve terms at {} slot(s): ssm_pool={} MB, ssm_h_stage={} MB, \
         replay_ring={} MB, marconi={} MB ({} slots), gdn_two_phase={} MB ({} tokens), \
         runtime headroom={} MB (gdn carry stash {} + driver fixed {} + driver bookkeeping {}), \
         per_sequence={} MB, decode_ring={} MB",
        ceiling,
        mb(t.ssm_pool),
        mb(t.ssm_h_stage),
        mb(t.replay_ring),
        mb(t.marconi),
        marconi.slots,
        mb(t.gdn_two_phase),
        max_batch_tokens_pre,
        mb(t.runtime.total()),
        mb(t.runtime.carry_stash),
        mb(t.runtime.driver_fixed),
        mb(t.runtime.driver_bookkeeping),
        mb(t.per_sequence),
        mb(t.decode_ring),
    );
    Ok(ReservePreflight {
        inference_reserve,
        buffer_arena_bytes,
        gdn_two_phase_bytes,
        ssm_prefill_chunk,
        max_batch_tokens_pre,
        plan,
    })
}

/// 2026-09-26: Rows one sequence's speculative step can occupy.
/// `max_batch_tokens_pre` here and `resolve_prefill_budget` (`kv_cache.rs`)
/// take a max against it.
///
/// DFlash: γ + 1, with the serve's γ (`ServeArgs::serve_dflash_gamma`). MTP, self- and
/// n-gram speculation:
/// `num_drafts + 2`. Otherwise 1.
pub(crate) fn spec_reserve_tokens(args: &cli::ServeArgs) -> usize {
    if args.dflash {
        args.serve_dflash_gamma() + 1
    } else if args.speculative || args.self_speculative || args.ngram_speculative {
        args.verify_pool_drafts() + 2
    } else {
        1
    }
}
