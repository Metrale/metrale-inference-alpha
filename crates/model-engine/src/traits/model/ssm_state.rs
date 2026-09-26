// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: `ModelSsmState`, one of the supertraits `Model` is made of. Its methods, default
//! bodies and docs are the ones `Model` declared before the split.
//!
//! Owner: model-engine.
//! Invariants: none beyond the types.

use crate::traits::SequenceState;
use anyhow::Result;

/// 2026-09-26: Recurrent (SSM) state: checkpoint and rollback around a verify, decode-rollback
/// snapshots, and the commit of an accepted prefix on the secondary stream.
pub trait ModelSsmState {
    fn checkpoint_ssm_states(&self, seq: &mut SequenceState) -> Result<()>;

    /// 2026-09-25: Restore the SSM state to where `num_accepted` tokens of the last verify were
    /// accepted.
    fn rollback_ssm_states(&self, seq: &mut SequenceState, num_accepted: usize) -> Result<()>;

    /// 2026-09-25: Whether the model has recurrent SSM layers, whose state is advanced in place
    /// each token. Lowering `seq_len` rewinds paged KV but not that state, so a decode rollback
    /// must also restore it from a snapshot. Default `false`.
    fn has_ssm_layers(&self) -> bool {
        false
    }

    /// 2026-09-26: Whether some layer keeps per-sequence state that lowering the KV cursor does
    /// not rewind (`LayerCapabilities::decode_rollback_unsupported`); the scheduler then declines
    /// a decode rollback (`rollback_to_boundary`). Default `false`.
    fn decode_rollback_unsupported(&self) -> bool {
        false
    }

    /// 2026-09-25: Decode-rollback SSM snapshot slots per active sequence; the scheduler sizes
    /// each sequence's snapshot ring from it. `0` (the default) means no decode-rollback
    /// snapshots. `TransformerModel` returns the depth `ssm_reserve::decode_rollback_ring_slots`
    /// chose.
    fn decode_rollback_ring_slots(&self) -> usize {
        0
    }

    /// 2026-09-25: Save `seq`'s SSM `h_state` and `conv_state` (every SSM layer) into its
    /// decode-rollback ring slot `ring_slot`, in `0..decode_rollback_ring_slots()`. Default:
    /// `Ok(())`.
    fn save_decode_ssm_snapshot(&self, _seq: &SequenceState, _ring_slot: usize) -> Result<()> {
        Ok(())
    }

    /// 2026-09-25: Restore `seq`'s SSM state from ring slot `ring_slot`, written earlier by
    /// [`Self::save_decode_ssm_snapshot`]. Default: `Ok(())`.
    fn restore_decode_ssm_snapshot(&self, _seq: &SequenceState, _ring_slot: usize) -> Result<()> {
        Ok(())
    }

    /// 2026-09-25: During decode, save a block-aligned SSM snapshot at a checkpoint boundary, so
    /// the next turn's prefix-cache hit can restore from decode-produced state. The scheduler
    /// calls it after a step's SSM state is final. Default: no-op.
    fn decode_marconi_checkpoint(&self, _seq: &mut SequenceState) {}

    /// 2026-09-25: Start the SSM checkpoint copies on the secondary stream; call
    /// [`Self::sync_secondary`] before the next verify. Default: the synchronous
    /// [`Self::checkpoint_ssm_states`].
    fn start_checkpoint_async(&self, seq: &mut SequenceState) -> Result<()> {
        self.checkpoint_ssm_states(seq)
    }

    /// 2026-09-25: Roll the SSM state back to `num_accepted` and checkpoint it again, on the
    /// secondary stream. Default: the synchronous rollback, then checkpoint.
    fn start_rollback_and_checkpoint_async(
        &self,
        seq: &mut SequenceState,
        num_accepted: usize,
    ) -> Result<()> {
        self.rollback_ssm_states(seq, num_accepted)?;
        self.checkpoint_ssm_states(seq)
    }

    /// 2026-09-25: Wait for the secondary stream's work. Default: `Ok(())`.
    fn sync_secondary(&self) -> Result<()> {
        Ok(())
    }

    /// 2026-09-25: Commit the accepted prefix of a verify onto the live SSM `h_state` and
    /// `conv_state`, on the secondary stream (pair with [`Self::sync_secondary`]). Default:
    /// `Ok(())`.
    fn commit_accepted_prefix(
        &self,
        _seq: &mut SequenceState,
        _num_accepted: usize,
        _k: usize,
    ) -> Result<()> {
        Ok(())
    }

    /// 2026-09-25: SSM snapshot-pool occupancy `(used, total)`, or `None` without a snapshot
    /// pool (the default; `TransformerModel` also returns `None` when the pool has no slots).
    fn ssm_snapshot_occupancy(&self) -> Option<(u32, u32)> {
        None
    }
}
