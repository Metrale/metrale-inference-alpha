// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The rank-agreed decision whether a prefix-cache hit restores an SSM
//! snapshot, and at which depth.
//!
//! On a multi-rank world the ranks first agree on the matched length (`ep_min_u32`
//! in `prefix_lookup`). The restore itself depends on rank-local state: each rank's
//! snapshot pool, the candidate's depth, and the hidden, session and aux gates. Ranks
//! that restore different depths process different token counts, and their
//! collectives no longer pair up. So each rank folds its gates into one proposal, the
//! depth `T` it can restore or `0`; the proposals are exchanged over the same
//! rooted-broadcast schedule; and a snapshot is restored only when every rank
//! proposed the same nonzero `T`. A minimum could pick a depth some rank does not hold.
//!
//! Owner: model-engine prefill (SSM prefix cache).
//! Invariants:
//! - `agree` returns `Some(T)` only when every proposal equals the same nonzero `T`.

use anyhow::Result;
use metrale_comm::CommBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

use super::super::super::types::TransformerModel;

/// 2026-09-25: The rank-local inputs to the restore decision, as plain values so the
/// decision is testable without a GPU.
#[derive(Debug, Clone, Copy)]
pub(in crate::model) struct LocalGates {
    /// 2026-09-25: Depth of this rank's candidate snapshot; `0` when it holds none for
    /// the matched prefix.
    pub snap_tok: usize,
    /// 2026-09-25: Matched prefix length (on a multi-rank world, the minimum over ranks).
    pub matched: usize,
    /// 2026-09-25: Prompt length.
    pub total: usize,
    /// 2026-09-25: `marconi_min_tokens()`: a shallower snapshot is not restored.
    pub min_tokens: usize,
    /// 2026-09-25: The candidate carries a stashed last-token hidden (only the prompt-end
    /// leaf save stashes one).
    pub has_hidden: bool,
    /// 2026-09-25: `METRALE_MARCONI_EXACT=1`: the exact full-prompt restore is enabled.
    pub exact_enabled: bool,
    /// 2026-09-25: The candidate is a session tail snapshot.
    pub is_tail: bool,
    /// 2026-09-25: `session_matches` for this sequence's session hash: the candidate is
    /// untagged or tagged with that hash, or the hash is 0.
    pub session_ok: bool,
    /// 2026-09-25: Some layer carries per-sequence aux state (`requires_aux_state`).
    pub needs_aux: bool,
    /// 2026-09-25: The candidate stored aux blobs.
    pub has_aux: bool,
}

/// 2026-09-25: This rank's proposal: the snapshot depth it can restore, or `0`. All
/// gates are evaluated here, before the exchange, so no rank declines after the
/// agreement.
pub(in crate::model) fn local_proposal(g: &LocalGates) -> u32 {
    if g.snap_tok == 0 {
        return 0;
    }
    let exact = g.snap_tok == g.matched && g.matched == g.total;
    // 2026-09-25: An exact restore needs the stashed hidden for the first token
    // (`finalize_last`'s exact-restore fixup).
    let exact_without_hidden = exact && !g.has_hidden;
    // 2026-09-25: The exact full-prompt restore is refused unless `METRALE_MARCONI_EXACT=1`.
    let bypass_exact = exact && !g.exact_enabled;
    let ok = g.snap_tok >= g.min_tokens
        && g.matched <= g.total
        && !exact_without_hidden
        && !bypass_exact
        && (!g.is_tail || g.session_ok)
        && (!g.needs_aux || g.has_aux);
    if ok { g.snap_tok as u32 } else { 0 }
}

/// 2026-09-25: All-or-nothing agreement over every rank's proposal, this rank's included:
/// `Some(T)` if and only if every rank proposed the same nonzero `T`.
pub(in crate::model) fn agree(proposals: &[u32]) -> Option<u32> {
    let first = *proposals.first()?;
    (first != 0 && proposals.iter().all(|&p| p == first)).then_some(first)
}

/// 2026-09-25: The position the suffix prefill resumes from, given the agreed decision.
/// A pure function of rank-agreed inputs, so the replay length (`total - skip_point`)
/// is the same on every rank.
pub(in crate::model) fn skip_point(
    skip: bool,
    snap_tok: usize,
    matched: usize,
    total: usize,
    has_ssm: bool,
) -> usize {
    if skip && !has_ssm {
        matched
    } else if skip && matched == total && snap_tok == matched {
        matched
    } else if skip {
        snap_tok
    } else {
        0
    }
}

/// 2026-09-25: Collect one `u32` from every rank of `comm` with `world` rooted
/// broadcasts (root `r` contributes element `r`). Every rank returns the same vector,
/// so any pure function of it is a rank-agreed decision; `ep_min_u32` is its minimum.
///
/// `buf` is a rank-local 4-byte device buffer the broadcasts go through.
pub(in crate::model) fn gather_u32_via_broadcast(
    gpu: &dyn GpuBackend,
    comm: &dyn CommBackend,
    buf: DevicePtr,
    world: usize,
    val: u32,
) -> Result<Vec<u32>> {
    let stream = gpu.default_stream();
    let mut out = Vec::with_capacity(world);
    for root in 0..world {
        let v = if comm.rank() == root {
            gpu.copy_h2d(&val.to_le_bytes(), buf)?;
            comm.broadcast(buf.0, 4, root)?;
            val
        } else {
            comm.broadcast(buf.0, 4, root)?;
            gpu.synchronize(stream)?;
            let mut bytes = [0u8; 4];
            gpu.copy_d2h(buf, &mut bytes)?;
            u32::from_le_bytes(bytes)
        };
        out.push(v);
    }
    Ok(out)
}

impl TransformerModel {
    /// 2026-09-25: Every rank's `val`, indexed by rank. Single-rank: `vec![val]`.
    pub(in crate::model) fn ep_gather_u32(&self, val: u32) -> Result<Vec<u32>> {
        let Some(comm) = self.comm.as_ref() else {
            return Ok(vec![val]);
        };
        // 2026-09-25: Loop over the ranks of the communicator: pure TP has
        // `ep_world_size == 1` but a `tp_world_size`-wide communicator.
        let world = self.config.ep_world_size.max(self.config.tp_world_size);
        gather_u32_via_broadcast(
            self.gpu.as_ref(),
            comm.as_ref(),
            self.ep_cmd_buf,
            world,
            val,
        )
    }
}
