// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Wire format and fire/skip decision of the decode-time Marconi checkpoint, as pure functions (tested in `prefill_b/snap_agree_tests.rs`).
//!
//! Owner: model-engine prefix cache.
//! Invariants:
//! - [`EP_CMD_DECODE_CKPT`] is above the decode-token range and differs from every other
//!   worker opcode; the compile-time asserts below enforce it.
//! - [`decode_ckpt_payload`] returns what [`encode_ckpt_payload`] was given, for token and
//!   block counts below 2^32.

use anyhow::{Result, bail};

/// 2026-09-25: EP worker command: save the decode-time Marconi checkpoint rank 0 just saved,
/// at the same `(slot, token, session)`. A restore is served only when every rank proposes
/// the same snapshot depth (`snap_agree::agree`), so a checkpoint only rank 0 holds is never
/// restored.
///
/// Wire shape: the `(seq_id, cmd)` preamble, then one bulk broadcast of
/// [`EP_CKPT_WORDS`] u32; see [`encode_ckpt_payload`].
pub(in crate::model) const EP_CMD_DECODE_CKPT: u32 = 0xFFFF_FFF8;

// 2026-09-25: The worker dispatches any code it does not match as a decode token id, so a
// command sits above `0xFFFF_FFEF`. It must also differ from the other codes, by last byte: E0 batched
// decode, F0 prefill chunk, F1 alloc-slot, F2/F3/F4 verify K=2/3/4, F5 MTP propose, F6/F7
// reserved, FF shutdown.
const _: () = assert!(
    EP_CMD_DECODE_CKPT > 0xFFFF_FFEF,
    "EP_CMD_DECODE_CKPT would be dispatched as a decode token id"
);
const _: () = assert!(
    EP_CMD_DECODE_CKPT != 0xFFFF_FFE0,
    "collides with batched decode"
);
const _: () = assert!(
    EP_CMD_DECODE_CKPT != 0xFFFF_FFF0,
    "collides with prefill chunk"
);
const _: () = assert!(
    EP_CMD_DECODE_CKPT != 0xFFFF_FFF1,
    "collides with alloc-slot"
);
const _: () = assert!(
    EP_CMD_DECODE_CKPT != 0xFFFF_FFF2,
    "collides with verify K=2"
);
const _: () = assert!(
    EP_CMD_DECODE_CKPT != 0xFFFF_FFF3,
    "collides with verify K=3"
);
const _: () = assert!(
    EP_CMD_DECODE_CKPT != 0xFFFF_FFF4,
    "collides with verify K=4"
);
const _: () = assert!(
    EP_CMD_DECODE_CKPT != metrale_model_layers::speculative::EP_CMD_MTP_PROPOSE,
    "collides with MTP propose"
);
const _: () = assert!(
    EP_CMD_DECODE_CKPT != 0xFFFF_FFF6,
    "reserved: DFlash EP_CMD_VERIFY_KGAMMA (A113)"
);
const _: () = assert!(
    EP_CMD_DECODE_CKPT != 0xFFFF_FFF7,
    "reserved: DFlash ctx-commit (A113)"
);
const _: () = assert!(EP_CMD_DECODE_CKPT != 0xFFFF_FFFF, "collides with shutdown");

/// 2026-09-25: Payload width of [`EP_CMD_DECODE_CKPT`], in u32 words.
pub(in crate::model) const EP_CKPT_WORDS: usize = 6;

/// 2026-09-25: What a decode checkpoint covers: the registered token count and the
/// block-table prefix it was taken over. The head computes it and every worker receives it
/// verbatim.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::model) struct CkptPlan {
    /// 2026-09-25: Tokens the saved state covers (`seq.tokens.len()`). It exceeds
    /// `end_block * block_size` by less than one block, because `end_block` is
    /// `tokens_len / block_size`.
    pub snap_tokens: usize,
    /// 2026-09-25: Complete KV blocks the checkpoint spans.
    pub end_block: usize,
}

/// 2026-09-25: Everything the fire/skip decision reads, as plain values, so the decision, and
/// with it whether an EP command is sent, is testable with no GPU.
#[derive(Debug, Clone, Copy)]
pub(in crate::model) struct CkptInputs {
    /// 2026-09-25: `ssm_snapshots.is_enabled() && prefix_cache.is_active()`. False means no save
    /// and no EP command.
    pub enabled: bool,
    pub num_ssm_layers: usize,
    pub hss_window_start: usize,
    pub slot_idx: usize,
    pub tokens_len: usize,
    pub block_size: usize,
    pub block_table_len: usize,
    pub last_ckpt_block: usize,
    /// 2026-09-25: `METRALE_DECODE_CKPT_BLOCKS`, with its default already applied.
    pub interval: usize,
}

/// 2026-09-25: The cheap half of [`decode_ckpt_plan`], which calls it too, so the decode path
/// can return before reading the env var or taking the KV lock.
pub(in crate::model) fn ckpt_preconditions(
    enabled: bool,
    num_ssm_layers: usize,
    hss_window_start: usize,
    slot_idx: usize,
) -> bool {
    enabled && num_ssm_layers != 0 && hss_window_start == 0 && slot_idx != usize::MAX
}

/// 2026-09-25: The rank-0 fire/skip decision, as a pure function. `None` means nothing is
/// saved and no [`EP_CMD_DECODE_CKPT`] goes on the wire.
///
/// The vision-pad veto is not here: it scans the whole token slice, so the caller applies it
/// after this decision says "fire".
pub(in crate::model) fn decode_ckpt_plan(i: &CkptInputs) -> Option<CkptPlan> {
    if !ckpt_preconditions(i.enabled, i.num_ssm_layers, i.hss_window_start, i.slot_idx) {
        return None;
    }
    if i.block_size == 0 || i.interval == 0 {
        return None;
    }
    // 2026-09-25: The block count comes from `tokens_len`, the slice that is registered, not
    // from `seq_len`.
    let end_block = i.tokens_len / i.block_size;
    if end_block == 0
        || !end_block.is_multiple_of(i.interval)
        || end_block == i.last_ckpt_block
        // 2026-09-25: Only blocks in the block table. The prefill checkpoint's `kv_valid_tokens`
        // guard is not used: only prefill advances that field, so it would veto every decode
        // checkpoint past the prompt.
        || end_block > i.block_table_len
    {
        return None;
    }
    Some(CkptPlan {
        snap_tokens: i.tokens_len,
        end_block,
    })
}

/// 2026-09-25: Encode the [`EP_CMD_DECODE_CKPT`] payload: the plan, then the head's session
/// and adapter identity, low word first for each u64. Token and block counts are truncated
/// to u32.
pub(in crate::model) fn encode_ckpt_payload(
    plan: CkptPlan,
    session_hash: u64,
    adapter_id: u64,
) -> [u32; EP_CKPT_WORDS] {
    [
        plan.snap_tokens as u32,
        plan.end_block as u32,
        session_hash as u32,
        (session_hash >> 32) as u32,
        adapter_id as u32,
        (adapter_id >> 32) as u32,
    ]
}

/// 2026-09-25: Inverse of [`encode_ckpt_payload`]: `(plan, session_hash, adapter_id)` as rank 0
/// sent them. Errors when the payload is not [`EP_CKPT_WORDS`] words.
pub(in crate::model) fn decode_ckpt_payload(w: &[u32]) -> Result<(CkptPlan, u64, u64)> {
    if w.len() != EP_CKPT_WORDS {
        bail!(
            "EP_CMD_DECODE_CKPT: payload is {} words, expected {EP_CKPT_WORDS}",
            w.len()
        );
    }
    Ok((
        CkptPlan {
            snap_tokens: w[0] as usize,
            end_block: w[1] as usize,
        },
        w[2] as u64 | ((w[3] as u64) << 32),
        w[4] as u64 | ((w[5] as u64) << 32),
    ))
}
