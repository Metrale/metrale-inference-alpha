// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Whole-prompt drafter context: the capture of prefill hidden rows, and its consume at the end of the sequence's own prefill.
//!
//! `mtp_prefill_hidden` is one shared buffer stamped with a generation
//! (`mtp_prefill_capture_gen`). `ensure_drafter_context` prefills the drafter from it only
//! while the sequence still owns that generation and the capture covers its prompt. At C >= 2
//! the next sequence's prefill restarts the capture before the previous sequence's first
//! propose, so the capture is consumed at the end of each sequence's prefill instead
//! ([`TransformerModel::try_eager_drafter_prefill`]). The `Model` wrappers `prefill`,
//! `prefill_chunk`, `prefill_twophase` and `mixed_forward` in `mod.rs` call it after their
//! dispatch returns. `METRALE_NO_MTP_EAGER_DRAFTER` turns the eager consume off, leaving the
//! consume at the first propose.
//!
//! Owner: model-engine speculative decoding.
//! Invariants:
//! - A capture append extends the tracked length only while the sequence owns the current
//!   capture generation.

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::super::types::TransformerModel;
use crate::traits::SequenceState;
use metrale_model_layers::layer::ForwardContext;

/// 2026-09-25: True when `METRALE_NO_MTP_EAGER_DRAFTER` is set, to any value: the capture is
/// then consumed only at the first propose. Read once per process.
pub fn eager_drafter_disabled() -> bool {
    static OFF: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OFF.get_or_init(|| std::env::var("METRALE_NO_MTP_EAGER_DRAFTER").is_ok())
}

impl TransformerModel {
    /// 2026-09-25: Copy this prefill chunk's hidden rows (`[proc_count, h]` BF16 at the head of
    /// the hidden buffer) into the whole-prompt capture at row `chunk_start`.
    ///
    /// `chunk_start == 0` restarts the tracked length; a chunk that extends it appends; any
    /// other chunk (prefix-cache reuse, warm restore) leaves the length short, and the
    /// coverage check in `ensure_drafter_context` then skips the drafter prefill.
    pub(super) fn try_mtp_prefill_capture(
        &self,
        seq: &mut SequenceState,
        chunk_start: usize,
        proc_count: usize,
        stream: u64,
    ) -> Result<()> {
        self.try_mtp_prefill_capture_from(
            seq,
            chunk_start,
            proc_count,
            self.buffers.hidden_states(),
            stream,
        )
    }

    /// 2026-09-25: [`Self::try_mtp_prefill_capture`] with an explicit source pointer.
    ///
    /// The mixed forward lays out `[decode rows | prefill rows]`, so its
    /// prefill hiddens start at `hidden + padded_n * h * 2`; the buffer head
    /// there holds decode rows.
    pub(super) fn try_mtp_prefill_capture_from(
        &self,
        seq: &mut SequenceState,
        chunk_start: usize,
        proc_count: usize,
        src: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        if self.mtp_prefill_hidden.is_null() || proc_count == 0 {
            return Ok(());
        }
        use std::sync::atomic::Ordering;
        if chunk_start + proc_count > self.mtp_prefill_capacity {
            return Ok(());
        }
        let len = self.mtp_prefill_capture_len.load(Ordering::Relaxed);
        // 2026-09-25: The capture buffer is one shared slot. Chunk 0 claims it under a fresh
        // generation; an append counts only while this sequence still owns the current
        // generation, because another sequence's chunk 0 may have restarted the capture in
        // between. On a mismatch the length stays short and the drafter prefill is skipped.
        let contiguous_from_zero = if chunk_start == 0 {
            let generation = self.mtp_prefill_capture_gen.fetch_add(1, Ordering::Relaxed) + 1;
            seq.mtp_capture_gen = generation;
            Some(proc_count)
        } else if chunk_start == len
            && seq.mtp_capture_gen != 0
            && seq.mtp_capture_gen == self.mtp_prefill_capture_gen.load(Ordering::Relaxed)
        {
            Some(len + proc_count)
        } else {
            None
        };
        // 2026-09-25: A warm turn's chunk starts at the reused-prefix boundary, which the
        // tracker above rejects. With `mtp_carry_drafter_enabled`, the carry path records the
        // written interval by absolute row (`mtp_store_range`), so it wants the write wherever
        // the chunk starts. The source is this chunk's rows; only the destination is absolute.
        let carry_on = metrale_model_layers::mtp_carry::mtp_carry_drafter_enabled(&self.levers);
        if contiguous_from_zero.is_none() && !carry_on {
            return Ok(());
        }
        let h = self.config.hidden_size;
        let bf16 = 2usize;
        self.gpu.copy_d2d_async(
            src,
            self.mtp_prefill_hidden.offset(chunk_start * h * bf16),
            proc_count * h * bf16,
            stream,
        )?;
        if let Some(new_len) = contiguous_from_zero {
            self.mtp_prefill_capture_len
                .store(new_len, Ordering::Relaxed);
        }
        if carry_on {
            // 2026-09-25: Stamped with this sequence's ticket. A write by a different owner
            // replaces the interval rather than extending it (`stamped_merge`).
            let mut r = self.mtp_store_range.lock();
            *r = metrale_model_layers::mtp_carry::stamped_merge(
                *r,
                seq.mtp_store_gen,
                chunk_start,
                proc_count,
            );
        }
        Ok(())
    }

    /// 2026-09-25: Consume the whole-prompt capture at the end of this sequence's prefill,
    /// while it still owns the capture generation. Returns at once unless `is_last`, the
    /// caller's last-chunk flag.
    ///
    /// The callers pass the `stream` their prefill ran on, so this is ordered after the
    /// capture copy it reads.
    ///
    /// Never fails a prefill: a drafter with fewer rows costs acceptance, not
    /// correctness, because the target verifies every draft.
    pub(super) fn try_eager_drafter_prefill(
        &self,
        seq: &mut SequenceState,
        is_last: bool,
        stream: u64,
    ) {
        if !is_last || eager_drafter_disabled() || self.mtp_prefill_hidden.is_null() {
            return;
        }
        let Some(proposer) = self.proposer.clone() else {
            return;
        };
        // 2026-09-25: A proposer whose prefill writes the shared forward scratch cannot run
        // here, because the target's prefill still owns those buffers. See
        // `DraftProposer::prefill_uses_shared_buffers`.
        if proposer.prefill_uses_shared_buffers() {
            return;
        }
        if seq.proposer_state.is_none() {
            return;
        }
        let ctx = ForwardContext {
            buffers: &self.buffers,
            hc_row_offset: 0,
            gpu: self.gpu.as_ref(),
            config: &self.config,
            dispatch: &self.dispatch,
            moe_lora_route: self.decode_moe_route(),
            derived: &self.derived,
            levers: &self.levers,
            stats: &self.stats,
            attn_metadata: None,
            profile: false,
            comm: None,
            graph_capture: false,
            decode_step: false,
            gdn_exact_replay: false,
            gdn_write_on_accept: false,
            token_ids: None,
            host_token_ids: None,
            routed_lora_layers: None,
            midchunk_capture: None,
        };
        self.ensure_drafter_context(proposer.as_ref(), seq, &ctx, stream);
        if metrale_model_layers::speculative::mtp_accept_debug() {
            let rows =
                proposer.drafter_rows(seq.proposer_state.as_mut().expect("checked above").as_mut());
            let captured = self
                .mtp_prefill_capture_len
                .load(std::sync::atomic::Ordering::Relaxed);
            tracing::info!(
                "MTP drafter coverage: prompt_len={} captured={captured} drafter_rows={rows}",
                seq.prompt_len,
            );
        }
        static LOGGED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
            tracing::info!(
                "MTP eager drafter prefill ENGAGED: the whole-prompt capture is consumed at \
                 end-of-prefill, so every concurrent sequence (not only the last-prefilled) \
                 can build drafter KV over its own prompt"
            );
        }
    }
}
