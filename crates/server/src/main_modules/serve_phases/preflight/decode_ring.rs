// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Preflight's decode-rollback ring term: the depth the flags ask
//! for, its byte formula, and the fit that lowers the depth until it fits.
//!
//! Owner: server startup (`met serve`).
//! Invariants:
//! - `fit_ring` never returns more slots than `requested`, and returns a
//!   warning only when it returns fewer.
//! - `autofit` publishes a depth through `ssm_reserve::set_decode_ring_slots`
//!   only when the fitted depth differs from the requested one.
//!
//! The ring is the term that yields because a smaller ring only removes
//! rollback anchors (a sequence with none declines through
//! `RollbackFallback::NoSsmSnapshot`), while `--max-batch-size` is the serve's
//! concurrency. The fit walks `ssm_reserve::DECODE_RING_FIT_LADDER`.

use metrale_config::ModelConfig;

use super::headroom::Yardstick;
use crate::cli;

const MIB: f64 = 1024.0 * 1024.0;
const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

/// 2026-09-26: The ring depth preflight reserves for, the decision line, and a
/// warning when the fit lowered the depth.
pub(super) struct RingFit {
    pub(super) slots: usize,
    /// 2026-09-26: The fit decision with every term; the caller logs it at
    /// INFO on every boot, whether or not the depth changed.
    pub(super) decision: String,
    /// 2026-09-26: `Some` only when the fit lowered the depth; the caller logs
    /// it at WARN.
    pub(super) warning: Option<String>,
}

/// 2026-09-26: The ring depth the flags and environment ask for, before any
/// fit.
///
/// The depth comes from `ssm_reserve::decode_rollback_ring_slots`, which
/// `TransformerModel::new` also calls to size the allocation. The
/// `use_speculative` passed here must match the one `build_model` passes:
/// `args.speculative || args.dflash`.
///
/// `METRALE_SSM_RESERVE_RING_FULL` set to any value, `0` included, reserves
/// `DECODE_ROLLBACK_RING_SLOTS` on a model with SSM layers whatever the
/// speculative flags; it changes the reserve only, not the allocation.
pub(super) fn requested_slots(args: &cli::ServeArgs, config: &ModelConfig) -> usize {
    if std::env::var("METRALE_SSM_RESERVE_RING_FULL").is_ok() {
        return if config.num_ssm_layers() > 0 {
            metrale_kernels::DECODE_ROLLBACK_RING_SLOTS
        } else {
            0
        };
    }
    metrale_model_layers::ssm_reserve::decode_rollback_ring_slots(
        config.num_ssm_layers(),
        args.speculative || args.dflash,
    )
    .slots
}

/// 2026-09-26: Bytes one ring slot costs: `--max-batch-size` x the
/// per-sequence SSM state blob. Preflight passes this one value to the fit and
/// multiplies the fitted depth by it for the reserve.
pub(super) fn slot_bytes(args: &cli::ServeArgs, per_seq_blob: usize) -> usize {
    args.max_batch_size * per_seq_blob
}

/// 2026-09-26: `slots x batch x per-seq state bytes` as text. The preflight
/// INFO line, the shrink warning and the refusal all quote it.
pub(super) fn formula(slots: usize, max_batch: usize, per_seq_blob: usize) -> String {
    format!(
        "ring: {slots} slots x {max_batch} seqs x {:.1} MB/seq = {:.2} GB",
        per_seq_blob as f64 / MIB,
        (slots * max_batch * per_seq_blob) as f64 / GIB,
    )
}

/// 2026-09-26: Fit the ring to its yardstick, then publish the fitted depth
/// if it differs from `requested`.
///
/// `reserve_without_ring` is the whole reserve without the ring term, and
/// `free_mem` is pre-load free memory; [`Yardstick::ladder_basis`] decides
/// which pair the ladder is measured against. A depth already published (an
/// explicit `--ssm-decode-ring-slots N`, published by `publish_kernel_flags`
/// before preflight) is kept as is, so the operator gets that depth or a
/// refusal.
pub(super) fn autofit(
    args: &cli::ServeArgs,
    requested: usize,
    slot_bytes: usize,
    per_seq_blob: usize,
    reserve_without_ring: usize,
    free_mem: usize,
    yardstick: &Yardstick,
) -> RingFit {
    // 2026-09-26: The published-depth read and the publish stay here, so
    // `fit_ring` touches no process-global state and tests can call it.
    let explicit = metrale_model_layers::ssm_reserve::published_decode_ring_slots().is_some();
    let fit = fit_ring(
        args,
        requested,
        slot_bytes,
        per_seq_blob,
        reserve_without_ring,
        free_mem,
        yardstick,
        explicit,
    );
    if fit.slots != requested {
        // 2026-09-26: `decode_rollback_ring_slots` returns a published depth
        // ahead of the environment and the speculative skip, so
        // `TransformerModel::new` allocates the depth reserved here.
        metrale_model_layers::ssm_reserve::set_decode_ring_slots(fit.slots);
    }
    fit
}

/// 2026-09-26: Core of [`autofit`] without global state. `explicit` is
/// `published_decode_ring_slots().is_some()`.
///
/// Returns `requested` unchanged, with no warning, when `requested` or
/// `slot_bytes` is 0, when `explicit`, when the requested depth fits, or when
/// on the pre-load yardstick even depth 0 does not fit.
#[allow(clippy::too_many_arguments)]
pub(super) fn fit_ring(
    args: &cli::ServeArgs,
    requested: usize,
    slot_bytes: usize,
    per_seq_blob: usize,
    reserve_without_ring: usize,
    free_mem: usize,
    yardstick: &Yardstick,
    explicit: bool,
) -> RingFit {
    let (beside, limit) = yardstick.ladder_basis(reserve_without_ring, free_mem);
    let asked = beside.saturating_add(requested.saturating_mul(slot_bytes));
    let decide = |slots: usize| describe(args, yardstick, slots, slot_bytes, beside, limit);
    if requested == 0 || slot_bytes == 0 || explicit || asked <= limit {
        return RingFit {
            slots: requested,
            decision: decide(requested),
            warning: None,
        };
    }
    let fitted = metrale_model_layers::ssm_reserve::fit_decode_ring_slots(
        requested, beside, slot_bytes, limit,
    );
    let fitted_total = beside.saturating_add(fitted.saturating_mul(slot_bytes));
    if fitted_total > limit && matches!(yardstick, Yardstick::PreLoadFree(_)) {
        // 2026-09-26: Even depth 0 does not fit in pre-load free memory, so
        // the ring is not the cause: keep the requested depth and let the
        // caller refuse, quoting it. The post-load headroom is an estimate,
        // so on that yardstick the fit drops to 0 instead and leaves the
        // decision to the KV budget stage.
        return RingFit {
            slots: requested,
            decision: decide(requested),
            warning: None,
        };
    }
    RingFit {
        slots: fitted,
        decision: decide(fitted),
        warning: Some(format!(
            "{} (was {:.2} GB); {}. Sized from {}, not from --max-batch-size {} (#915): \
             rollback depth yields, concurrency does not. Pass --ssm-decode-ring-slots N to \
             pin a depth (and be refused rather than shrunk).",
            formula(fitted, args.max_batch_size, per_seq_blob),
            (requested * slot_bytes) as f64 / GIB,
            if fitted_total > limit {
                format!(
                    "even 0 slots does not clear the {:.2} GB KV floor inside {:.2} GB of \
                     predicted headroom — the KV budget stage will decide, on measured bytes",
                    beside as f64 / GIB,
                    limit as f64 / GIB,
                )
            } else {
                format!(
                    "{} {:.2} of {:.2} GB",
                    yardstick.total_label(),
                    fitted_total as f64 / GIB,
                    limit as f64 / GIB,
                )
            },
            yardstick.name(),
            args.max_batch_size,
        )),
    }
}

/// 2026-09-26: The decision line: the yardstick, every term behind it, and
/// the depth chosen.
fn describe(
    args: &cli::ServeArgs,
    yardstick: &Yardstick,
    slots: usize,
    slot_bytes: usize,
    beside: usize,
    limit: usize,
) -> String {
    let ring_bytes = slots.saturating_mul(slot_bytes);
    match yardstick {
        Yardstick::PostLoad(h) => format!(
            "SSM decode-ring fit against the predicted post-load KV headroom: {}",
            h.describe(args.gpu_memory_utilization, ring_bytes, slots),
        ),
        Yardstick::PreLoadFree(why) => format!(
            "SSM decode-ring fit against pre-load free memory ({why}): ring({slots}) {:.2} + \
             rest of reserve {:.2} = {:.2} of {:.2} GB free",
            ring_bytes as f64 / GIB,
            beside as f64 / GIB,
            (beside + ring_bytes) as f64 / GIB,
            limit as f64 / GIB,
        ),
    }
}

#[cfg(test)]
#[path = "decode_ring_tests.rs"]
mod tests;
