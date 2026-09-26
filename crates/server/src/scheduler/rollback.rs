// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Decode-time rollback: when the content-loop, fuzzy-repetition
//! or inter-tool prose watchdog trips, [`rollback_to_boundary`] rewinds the
//! sequence to its last boundary token and lets generation continue. When
//! the rollback is declined, the calling watchdog ends the sequence.
//!
//! Owner: scheduler.
//! Invariants:
//! - On a model with SSM layers, a rollback completes only after the SSM
//!   snapshot taken at the target boundary has been restored. With no such
//!   snapshot, or when the restore fails, the rollback is declined.
//!
//! ## Attention KV rewind
//!
//! Lowering `seq.seq_len` and popping `seq.tokens` is the whole attention
//! rewind; `spec_step.rs` rewinds rejected drafts the same way. The block
//! table is left as it is.
//!
//! ## SSM state rewind
//!
//! An SSM layer's recurrent state is updated in place on every token, so
//! lowering a cursor cannot undo it. Each sequence keeps an
//! [`SsmDecodeRing`] of snapshots taken at boundary tokens during decode
//! ([`snapshot_boundary_if_ssm`]), and a rollback on a model with SSM
//! layers may only target a boundary that has one. Pure-attention models
//! ([`Model::has_ssm_layers`] is `false`) may roll back to any boundary.
//! MODEL.toml `[behavior].rollback_resteer` (default `true`) turns the
//! feature off.
//!
//! [`Model::has_ssm_layers`]: metrale_model_engine::traits::ModelSsmState::has_ssm_layers

use metrale_model_engine::traits::Model;

use super::ssm_decode_ring::SsmDecodeRing;
use super::*;

/// 2026-09-25: Outcome of a [`rollback_to_boundary`] attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RollbackOutcome {
    /// 2026-09-25: The sequence was rewound to a boundary; generation
    /// continues.
    RolledBack {
        /// 2026-09-25: Trailing tokens dropped from `output_tokens`.
        dropped: usize,
    },
    /// 2026-09-25: No rollback happened. The callers then end the
    /// sequence.
    Fallback(RollbackFallback),
}

/// 2026-09-25: Why a rollback was declined.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RollbackFallback {
    /// 2026-09-25: MODEL.toml `[behavior].rollback_resteer = false`.
    Disabled,
    /// 2026-09-25: The sequence has a `cancel_flag`: a streaming request
    /// not resumed from swap. Tokens already sent to the client cannot be
    /// taken back, so the client would see both the dropped text and its
    /// regeneration.
    StreamUnsafe,
    /// 2026-09-25: The sequence already has `ROLLBACK_RESTEER_CAP`
    /// rollbacks.
    CapReached,
    /// 2026-09-25: There is no boundary mask for this tokenizer, or no
    /// boundary token at least `min_keep` tokens before the end of the
    /// output.
    NoBoundary,
    /// 2026-09-25: Model with SSM layers only: the sequence's snapshot
    /// ring is disabled, no candidate boundary has a live snapshot, or
    /// restoring the snapshot failed. Rolling back the tokens without the
    /// SSM state would continue from state that still includes the
    /// dropped tokens.
    NoSsmSnapshot,
    /// 2026-09-25: The model reports `decode_rollback_unsupported()`: a
    /// layer keeps per-sequence state that neither lowering the KV cursor
    /// nor the SSM ring rewinds.
    LayerStateNotRewindable,
}

/// 2026-09-25: The index in `output_tokens` of the last token with
/// `mask[id]` set, searching only indices that leave at least `min_keep`
/// tokens after it. A rollback keeps that token and drops the rest.
/// `None` when there is none.
pub fn find_last_boundary(output_tokens: &[u32], mask: &[bool], min_keep: usize) -> Option<usize> {
    let n = output_tokens.len();
    if n <= min_keep {
        return None;
    }
    let search_end = n - 1 - min_keep;
    for idx in (0..=search_end).rev() {
        let id = output_tokens[idx] as usize;
        if id < mask.len() && mask[id] {
            return Some(idx);
        }
    }
    None
}

/// 2026-09-25: Like [`find_last_boundary`], but a boundary at index `i`
/// counts only if `ring` holds a snapshot for position `i + 1`, the
/// `keep_len` of a rollback to it.
pub fn find_last_boundary_with_snapshot(
    output_tokens: &[u32],
    mask: &[bool],
    min_keep: usize,
    ring: &SsmDecodeRing,
) -> Option<usize> {
    let n = output_tokens.len();
    if n <= min_keep {
        return None;
    }
    let search_end = n - 1 - min_keep;
    for idx in (0..=search_end).rev() {
        let id = output_tokens[idx] as usize;
        let is_boundary = id < mask.len() && mask[id];
        if is_boundary && ring.slot_for_position(idx + 1).is_some() {
            return Some(idx);
        }
    }
    None
}

/// 2026-09-25: Roll `a` back to its last boundary token so generation
/// continues from there, dropping at least `min_keep` tokens.
///
/// Declined, in this order, when: `rollback_resteer` is off; the sequence
/// has a `cancel_flag`; the model reports `decode_rollback_unsupported()`;
/// `ROLLBACK_RESTEER_CAP` is reached; there is no boundary mask; on a model
/// with SSM layers, the ring is disabled, no boundary has a snapshot or
/// the restore fails; on other models, no boundary is found. A declined
/// call leaves the token buffers and counters of `a` unchanged.
///
/// On success it restores the SSM snapshot (SSM models), drops the ring
/// entries after the boundary, applies `apply_rollback` and increments
/// `rollback_count`. No tokens are injected.
pub fn rollback_to_boundary(
    a: &mut ActiveSeq,
    min_keep: usize,
    model: &dyn Model,
    sched: &crate::scheduler::sched_ctx::SchedCtx,
) -> RollbackOutcome {
    if !sched.watchdog.rollback_resteer {
        return RollbackOutcome::Fallback(RollbackFallback::Disabled);
    }
    // 2026-09-25: `cancel_flag` is `Some` for streaming requests, except
    // after a swap-in: `resume_swapped_seq` sets it to `None`.
    if a.cancel_flag.is_some() {
        return RollbackOutcome::Fallback(RollbackFallback::StreamUnsafe);
    }
    if model.decode_rollback_unsupported() {
        return RollbackOutcome::Fallback(RollbackFallback::LayerStateNotRewindable);
    }
    if a.rollback_count >= metrale_kernels::ROLLBACK_RESTEER_CAP {
        return RollbackOutcome::Fallback(RollbackFallback::CapReached);
    }
    let mask = match sched.masks.boundary.as_ref() {
        Some(m) => m.clone(),
        None => return RollbackOutcome::Fallback(RollbackFallback::NoBoundary),
    };

    // 2026-09-25: A disabled ring on an SSM model is a decline, not the
    // pure-attention path: there is no snapshot to restore.
    let hybrid = model.has_ssm_layers();
    if hybrid && !a.ssm_rollback_ring.is_enabled() {
        return RollbackOutcome::Fallback(RollbackFallback::NoSsmSnapshot);
    }
    let (boundary_idx, ssm_slot) = if hybrid {
        match find_last_boundary_with_snapshot(
            &a.output_tokens,
            &mask,
            min_keep,
            &a.ssm_rollback_ring,
        ) {
            Some(i) => {
                // 2026-09-25: `Some`: the search matched this position.
                let slot = a.ssm_rollback_ring.slot_for_position(i + 1);
                (i, slot)
            }
            None => {
                // 2026-09-25: Tell "no boundary" from "no snapshot at any
                // boundary"; the callers log the reason.
                let reason = if find_last_boundary(&a.output_tokens, &mask, min_keep).is_some() {
                    RollbackFallback::NoSsmSnapshot
                } else {
                    RollbackFallback::NoBoundary
                };
                return RollbackOutcome::Fallback(reason);
            }
        }
    } else {
        match find_last_boundary(&a.output_tokens, &mask, min_keep) {
            Some(i) => (i, None),
            None => return RollbackOutcome::Fallback(RollbackFallback::NoBoundary),
        }
    };

    let keep_len = boundary_idx + 1;
    let dropped = a.output_tokens.len() - keep_len;
    debug_assert!(dropped >= min_keep);

    // 2026-09-25: Restore before touching the token buffers, so a failed
    // restore declines with them unchanged.
    if let Some(slot) = ssm_slot {
        if let Err(e) = sched
            .io
            .dev
            .apply(crate::scheduler::io::Effect::SsmSnapshotRestore { seq: &a.seq, slot })
        {
            tracing::error!(
                error = %e,
                ring_slot = slot,
                keep_len,
                "SSM decode-snapshot restore failed; declining rollback"
            );
            return RollbackOutcome::Fallback(RollbackFallback::NoSsmSnapshot);
        }
        // 2026-09-25: Drop the snapshots taken after the boundary; the
        // boundary's own snapshot stays.
        a.ssm_rollback_ring.truncate_after(keep_len);
    }

    apply_rollback(a, keep_len, dropped);
    a.rollback_count = a.rollback_count.saturating_add(1);
    RollbackOutcome::RolledBack { dropped }
}

/// 2026-09-25: On a model with SSM layers and an enabled ring, if the last
/// token of `output_tokens` is a boundary token, record it in the ring and
/// save the SSM state into the returned slot. The recorded position is
/// `output_tokens.len()`, which is the `keep_len` of a rollback to that
/// token.
///
/// `decode_logits_step.rs` calls it after pushing each token sampled
/// outside thinking. A failed save is logged and the new entry removed,
/// so no entry names a slot that was not written.
pub fn snapshot_boundary_if_ssm(
    a: &mut ActiveSeq,
    model: &dyn Model,
    sched: &crate::scheduler::sched_ctx::SchedCtx,
) {
    if !model.has_ssm_layers() || !a.ssm_rollback_ring.is_enabled() {
        return;
    }
    // 2026-09-25: Without a boundary mask a rollback declines anyway.
    let Some(mask) = sched.masks.boundary.clone() else {
        return;
    };
    let Some(&last) = a.output_tokens.last() else {
        return;
    };
    let id = last as usize;
    if id >= mask.len() || !mask[id] {
        return;
    }
    let token_position = a.output_tokens.len();
    let Some(slot) = a.ssm_rollback_ring.record(token_position) else {
        return;
    };
    if let Err(e) = sched
        .io
        .dev
        .apply(crate::scheduler::io::Effect::SsmSnapshotSave { seq: &a.seq, slot })
    {
        tracing::warn!(
            error = %e,
            ring_slot = slot,
            token_position,
            "SSM decode-snapshot save failed; dropping ring entry"
        );
        a.ssm_rollback_ring
            .truncate_after(token_position.saturating_sub(1));
    }
}

mod rewind;
pub use rewind::RomHead;
use rewind::apply_rollback;
#[cfg(test)]
use rewind::{grammar_rewind, rewind_buffers};

#[cfg(test)]
#[path = "rollback_tests.rs"]
mod rollback_tests;
