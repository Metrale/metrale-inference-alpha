// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-01: The pre-load reserve as a plan over the slot count: every term the KV budget
//! subtracts, evaluated at any `--max-batch-size`, so the build can size the KV pool for the
//! count it resolves (`--max-batch-size auto`) by the same arithmetic the preflight checked at
//! the ceiling. Split from `preflight.rs`.
//!
//! Owner: server startup (`met serve`).
//! Invariants:
//! - `ssm_pool` is `ssm_reserve::PoolPlan::total`, the bytes `SsmStatePool::new` allocates for
//!   that slot count; the carry stash is `GdnCarrySizes::total` (`runtime_headroom.rs`).
//! - Every term is a pure function of the plan's fields and the slot count.

use anyhow::Result;
use metrale_config::ModelConfig;
use metrale_model_layers::ssm_reserve::{self, PoolShape, SsmRollbackMode};

use super::runtime_headroom::{RuntimeHeadroom, carry_stash_bytes};

/// 2026-10-01: What the reserve depends on besides the slot count.
#[derive(Debug, Clone)]
pub(crate) struct ReservePlan {
    pub(super) config: ModelConfig,
    /// 2026-10-01: The verify pools exist (`PoolShape::spec`).
    pub(super) spec: bool,
    /// 2026-10-01: Drafts per verify (γ under DFlash).
    pub(super) num_drafts: usize,
    /// 2026-10-01: DFlash: every verify slot holds `K - 1` h intermediates.
    pub(super) uniform_h: bool,
    pub(super) rollback: SsmRollbackMode,
    pub(super) h_f16_pool: bool,
    /// 2026-10-01: Marconi snapshot slots x the per-sequence blob.
    pub(super) marconi_bytes: usize,
    /// 2026-10-01: The GDN two-phase prefill scratch at the prefill chunk.
    pub(super) gdn_two_phase_bytes: usize,
    /// 2026-10-01: `total_memory x --gpu-memory-utilization`.
    pub(super) budget_bytes: usize,
    /// 2026-10-01: Per-sequence owned state (DSA indexer, proposer), per slot.
    pub(super) per_sequence_bytes: usize,
    /// 2026-10-01: Decode-rollback ring depth (the preflight's fit) and the per-sequence blob.
    pub(super) ring_slots: usize,
    pub(super) per_seq_blob: usize,
}

/// 2026-10-01: The reserve's terms at one slot count, in bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ReserveTerms {
    pub ssm_pool: usize,
    pub ssm_h_stage: usize,
    pub replay_ring: usize,
    pub marconi: usize,
    pub gdn_two_phase: usize,
    pub runtime: RuntimeHeadroom,
    pub per_sequence: usize,
    pub decode_ring: usize,
}

impl ReserveTerms {
    /// 2026-10-01: Every term except the decode-rollback ring, the only one the preflight's
    /// auto-fit shrinks.
    pub(crate) fn fixed(&self) -> usize {
        self.ssm_pool
            + self.ssm_h_stage
            + self.replay_ring
            + self.marconi
            + self.gdn_two_phase
            + self.runtime.total()
            + self.per_sequence
    }

    /// 2026-10-01: The inference reserve.
    pub(crate) fn total(&self) -> usize {
        self.fixed() + self.decode_ring
    }
}

impl ReservePlan {
    /// 2026-10-01: The terms at `slots` slots.
    pub(crate) fn terms(&self, slots: usize) -> Result<ReserveTerms> {
        let counts = ssm_reserve::pool_counts(&PoolShape {
            max_slots: slots,
            spec: self.spec,
            num_intermediates: self.num_drafts + 1,
            num_drafts: self.num_drafts,
            uniform_h: self.uniform_h,
            rollback: self.rollback,
        });
        let pool = ssm_reserve::PoolPlan::new(&self.config, &counts, self.h_f16_pool)?;
        let verify_slots = counts.verify.as_ref().map(|v| v.slots());
        let replay_ring = match verify_slots {
            Some(v) if self.rollback == SsmRollbackMode::Replay => {
                ssm_reserve::ssm_replay_ring_bytes(
                    pool.layers,
                    ssm_reserve::ssm_replay_row_bytes(
                        self.config.ssm_qkvz_size(),
                        self.config.linear_num_value_heads,
                    ),
                    // 2026-10-02: Widened by a prompt-lookup copy tier, as the pool widens it.
                    ssm_reserve::tier_rows(self.num_drafts + 1, ssm_reserve::copy_tier()),
                    v,
                )
            }
            _ => 0,
        };
        Ok(ReserveTerms {
            ssm_pool: pool.total(),
            ssm_h_stage: ssm_reserve::ssm_h_prefill_stage_bytes(
                counts.slots,
                pool.h_f32_unit,
                self.h_f16_pool,
            ),
            replay_ring,
            marconi: self.marconi_bytes,
            gdn_two_phase: self.gdn_two_phase_bytes,
            runtime: RuntimeHeadroom::new(
                self.budget_bytes,
                carry_stash_bytes(&self.config, verify_slots, self.h_f16_pool),
            ),
            per_sequence: self.per_sequence_bytes * slots.max(1),
            decode_ring: self.ring_slots * slots * self.per_seq_blob,
        })
    }

    /// 2026-10-01: The inference reserve at `slots` slots.
    pub(crate) fn inference_reserve(&self, slots: usize) -> Result<usize> {
        Ok(self.terms(slots)?.total())
    }
}

#[cfg(test)]
#[path = "reserve_plan_tests.rs"]
mod tests;
