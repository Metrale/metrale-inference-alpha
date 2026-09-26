// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: `ModelVerify`, one of the supertraits `Model` is made of. Its methods, default
//! bodies and docs are the ones `Model` declared before the split.
//!
//! Owner: model-engine.
//! Invariants: none beyond the types.

use super::VerifyBatchedOpts;
use crate::traits::SequenceState;
use anyhow::{Result, bail};

/// 2026-09-26: Speculative verify passes: the fixed-width, γ-wide, DFlash and batched verifies, and the
/// GDN fold after a write-on-accept batch.
pub trait ModelVerify {
    /// 2026-09-25: Run `tokens` through the model as one verify pass and return the argmax id at
    /// each position; every token advances the KV and SSM state.
    fn decode_verify(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        stream: u64,
    ) -> Result<Vec<u32>>;

    /// 2026-09-25: K=2 verify (`[last, draft]`); returns the argmax id at each position.
    fn decode_verify_graphed(
        &self,
        tokens: &[u32; 2],
        seq: &mut SequenceState,
        stream: u64,
    ) -> Result<[u32; 2]>;

    /// 2026-09-25: K=3 verify (1 verified token and 2 drafts); returns 3 argmax ids.
    fn decode_verify_graphed_k3(
        &self,
        tokens: &[u32; 3],
        seq: &mut SequenceState,
        stream: u64,
    ) -> Result<[u32; 3]>;

    /// 2026-09-25: K=4 verify (1 verified token and 3 drafts); returns 4 argmax ids.
    fn decode_verify_graphed_k4(
        &self,
        tokens: &[u32; 4],
        seq: &mut SequenceState,
        stream: u64,
    ) -> Result<[u32; 4]>;

    /// 2026-09-25: Whether [`Self::decode_verify_batched`] can run `ks.len()` sequences with
    /// `ks[i]` verify rows each (the sequence's draft count plus one; the rows may differ per
    /// sequence). Default `false`, and there is no default batched implementation: a loop of
    /// per-sequence verifies would leave only the last sequence's rows in the shared logits
    /// buffer.
    fn can_batch_verify(&self, _ks: &[usize]) -> bool {
        false
    }

    /// 2026-09-25: Verify `ks.len()` sequences in one forward. Rows are flat and sequence-major
    /// (`tokens.len() == Σ ks`): sequence `i` holds `[last_verified, d0, ..]` in rows
    /// `off_i..off_i + ks[i]`, with `off_i = Σ_{t<i} ks[t]`. Returns the `Σ ks` argmax ids in the
    /// same order. On success each sequence advances by its own `ks[i]`, and the caller rewinds
    /// by its verdict.
    ///
    /// Call only when [`Self::can_batch_verify`] is true. With `opts.write_on_accept` the caller
    /// must run [`Self::gdn_fold_accepted`] with every verdict before any
    /// `commit_accepted_prefix`. Default: an error.
    fn decode_verify_batched(
        &self,
        tokens: &[u32],
        ks: &[usize],
        seqs: &mut [&mut SequenceState],
        stream: u64,
        opts: VerifyBatchedOpts,
    ) -> Result<Vec<u32>> {
        let _ = (tokens, ks, seqs, stream, opts);
        bail!("decode_verify_batched: unsupported by this model")
    }

    /// 2026-09-25: DFlash verify of `tokens` (1 verified token and γ drafts). Default:
    /// [`Self::decode_verify`].
    fn decode_verify_graphed_kgamma(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        stream: u64,
    ) -> Result<Vec<u32>> {
        self.decode_verify(tokens, seq, stream)
    }

    /// 2026-09-25: DFlash verify. Default: [`Self::decode_verify_graphed_kgamma`].
    fn decode_verify_dflash(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        stream: u64,
    ) -> Result<Vec<u32>> {
        self.decode_verify_graphed_kgamma(tokens, seq, stream)
    }

    /// 2026-09-25: DFlash decode and verify in one forward: `tokens[0]` is the accepted token and
    /// `tokens[1..]` the draft block. Default: [`Self::decode_verify_graphed_kgamma`].
    fn decode_and_verify_fused(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        stream: u64,
    ) -> Result<Vec<u32>> {
        self.decode_verify_graphed_kgamma(tokens, seq, stream)
    }

    /// 2026-09-25: After a batched verify run with `write_on_accept`, commit the GDN `h` state of
    /// each slot `slots[i]` at `accepted_rows[i]` verify rows (anchor included), in the verify's
    /// batch order. `Ok(true)` means the `h` states were committed here, and
    /// `commit_accepted_prefix` then restores only conv state for those slots. `Ok(false)`
    /// means nothing was folded. Default: `Ok(false)`.
    fn gdn_fold_accepted(
        &self,
        _slots: &[usize],
        _accepted_rows: &[u32],
        _k_rows: usize,
    ) -> Result<bool> {
        Ok(false)
    }
}
