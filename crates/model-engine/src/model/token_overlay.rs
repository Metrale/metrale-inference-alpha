// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Token-overlay hooks. `apply_embed_overlay` replaces the embedding rows
//! of overridden vocab ids after the embed gather and before `scale_embeddings`;
//! `apply_lmhead_overlay` recomputes the logit columns of overridden ids after the
//! base lm_head projection and before softcap.
//!
//! Both hooks return without a launch when no overlay set is installed
//! (`self.overlays` is `None`). Every caller passes a null `seq_slot`, so the
//! route is the single slot from [`TransformerModel::overlay_active_slot`].
//!
//! Owner: model-engine (LoRA token overlays).
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::types::TransformerModel;
use metrale_model_layers::layers::ops::token_overlay;

impl TransformerModel {
    /// 2026-09-25: Record a single request's `adapter_slot` as the overlay route
    /// for the next forward. The single-sequence `Model` entries in
    /// `trait_impl/mod.rs` call it.
    pub(super) fn stamp_overlay_route(&self, adapter_slot: i32) {
        self.overlay_route_slot
            .store(adapter_slot, std::sync::atomic::Ordering::Relaxed);
    }

    /// 2026-09-25: Record a batch's route: the shared `adapter_slot` when every
    /// sequence agrees, else `i32::MIN` (the hooks then skip). An empty batch
    /// records `-1`.
    pub(super) fn stamp_overlay_route_batch(&self, seqs: &[&mut crate::traits::SequenceState]) {
        let slot = match seqs.split_first() {
            Some((first, rest)) if rest.iter().all(|s| s.adapter_slot == first.adapter_slot) => {
                first.adapter_slot
            }
            Some(_) => i32::MIN,
            None => -1,
        };
        self.overlay_route_slot
            .store(slot, std::sync::atomic::Ordering::Relaxed);
    }

    /// 2026-09-25: The slot whose overlay the hooks apply, or `-1` to skip. It is
    /// the recorded route when `LoraWeights::routed_prefill_slot` accepts it: an
    /// in-range slot other than the pool's `active`. A route on the active slot,
    /// `-1`, `i32::MIN` (mixed batch), an out-of-range slot or no LoRA pool gives
    /// `-1`.
    pub(super) fn overlay_active_slot(&self) -> i32 {
        let Some(l) = self.lora.as_ref() else {
            return -1;
        };
        let req = self
            .overlay_route_slot
            .load(std::sync::atomic::Ordering::Relaxed);
        if req == i32::MIN {
            // 2026-09-25: Mixed-adapter batch: skip rather than apply one
            // adapter's overlay to every row.
            return -1;
        }
        l.routed_prefill_slot(req).map(|s| s as i32).unwrap_or(-1)
    }

    /// 2026-09-25: Overlay the embedding rows of overridden vocab ids in place on
    /// `out`. `ids_dev` is the device `u32[num_tokens]` token-id buffer for these
    /// rows. A null `seq_slot` selects the single [`Self::overlay_active_slot`]
    /// route; a device `seq_slot` buffer would route each row.
    pub(super) fn apply_embed_overlay(
        &self,
        ids_dev: DevicePtr,
        seq_slot: DevicePtr,
        out: DevicePtr,
        num_tokens: u32,
        stream: u64,
    ) -> Result<()> {
        let Some(set) = self.overlays.as_ref() else {
            return Ok(());
        };
        if self.overlay_kernels.embed_overlay.0 == 0 || num_tokens == 0 {
            return Ok(());
        }
        let active = self.overlay_active_slot();
        if seq_slot.is_null() && active < 0 {
            return Ok(());
        }
        token_overlay::embed_overlay_routed(
            self.gpu.as_ref(),
            self.overlay_kernels.embed_overlay,
            ids_dev,
            seq_slot,
            active,
            set.embed_slot_map_table,
            set.embed_rows_table,
            set.embed_n_table,
            out,
            num_tokens,
            self.config.hidden_size as u32,
            set.vocab,
            stream,
        )
    }

    /// 2026-09-25: Overlay the logit columns of overridden vocab ids in place on
    /// `logits` (`[m, vocab]`). `is_fp32` selects the f32-logits kernel (the
    /// single-token `lm_head` passes `use_fp32_logits`); otherwise bf16.
    pub(super) fn apply_lmhead_overlay(
        &self,
        hidden: DevicePtr,
        seq_slot: DevicePtr,
        logits: DevicePtr,
        m: u32,
        is_fp32: bool,
        stream: u64,
    ) -> Result<()> {
        let Some(set) = self.overlays.as_ref() else {
            return Ok(());
        };
        if set.max_n_override == 0 || m == 0 {
            return Ok(()); // 2026-09-25: no slot overrides an lm_head column.
        }
        let kernel = if is_fp32 {
            self.overlay_kernels.lmhead_overlay_f32
        } else {
            self.overlay_kernels.lmhead_overlay_bf16
        };
        if kernel.0 == 0 {
            return Ok(());
        }
        let active = self.overlay_active_slot();
        if seq_slot.is_null() && active < 0 {
            return Ok(());
        }
        token_overlay::lmhead_overlay_routed(
            self.gpu.as_ref(),
            kernel,
            hidden,
            seq_slot,
            active,
            set.lmhead_rows_table,
            set.lmhead_ids_table,
            set.n_override_table,
            logits,
            m,
            set.max_n_override,
            self.config.hidden_size as u32,
            self.config.vocab_size as u32,
            stream,
        )
    }
}
