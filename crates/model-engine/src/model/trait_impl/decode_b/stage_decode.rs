// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The unfused arm of `mixed_forward_dispatch`: run `decode_batch` and stage its
//! logits rows at the top of the logits arena, where the prefill that follows does not write.
//!
//! Owner: model-engine (decode).
//! Invariants: the staged rows sit above the prefill row and the 64 KiB scratch band, or the
//! call fails before any copy.

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::super::super::types::TransformerModel;
use crate::traits::{ModelForward, SequenceState};

impl TransformerModel {
    /// 2026-09-26: `decode_batch` over the decode rows, then copy its `n_decode` logits rows to
    /// the top of the logits arena and return where they now are; `DevicePtr::NULL` when there
    /// are no decode rows.
    pub(super) fn mixed_stage_decode_logits(
        &self,
        decode_tokens: &[u32],
        decode_seqs: &mut [&mut SequenceState],
        n_decode: usize,
        stream: u64,
    ) -> Result<DevicePtr> {
        let decode_logits = if !decode_tokens.is_empty() {
            let live = self.decode_batch(decode_tokens, decode_seqs, stream)?;
            // 2026-09-25: The prefill below writes the same logits buffer (its LM
            // head rows at the base; MoE uses the start of `logits` as shared-gate
            // scratch), and the scheduler reads the decode rows only after
            // `mixed_forward` returns. So copy the decode rows to the top of the
            // logits arena, above both. The copy runs on the backend's default
            // stream, where `decode()` wrote them; `stream` is still the caller's.
            let v = self.config.vocab_size;
            let elem: usize = if self.decode_logits_fp32() { 4 } else { 2 };
            let bytes = n_decode * v * elem;
            let arena = self.buffers.sizes().logits;
            let staged_off = arena
                .checked_sub(bytes)
                .filter(|&off| off >= 65536 + v * elem)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "mixed fallback: {bytes} B of decode logits do not                              fit above the prefill row + scratch band in the                              {arena} B logits arena"
                    )
                })?;
            let staged = self.buffers.logits().offset(staged_off);
            self.gpu
                .copy_d2d_async(live, staged, bytes, self.gpu.default_stream())?;
            staged
        } else {
            DevicePtr::NULL
        };
        Ok(decode_logits)
    }
}
