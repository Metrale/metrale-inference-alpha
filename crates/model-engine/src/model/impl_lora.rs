// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: LoRA on `TransformerModel`: startup install (`set_lora_weights`), per-request
//! adapter slot resolution and refs, the per-decode MoE-LoRA route, and the token-overlay build.
//!
//! Owner: model-engine.
//! Invariants: none beyond the types.

use anyhow::{Context, Result};

use metrale_gpu_runtime::gpu::DevicePtr;

use super::types::TransformerModel;
use metrale_model_layers::layers::ops;

impl TransformerModel {
    /// 2026-09-25: Adapter id for a request's `adapter_slot` (`< 0` resolves
    /// to the active slot): a hash of the slot's name and generation, or `0`
    /// when no LoRA pool is resident or the slot is empty or out of range.
    pub fn adapter_id_for_slot(&self, slot: i32) -> u64 {
        match self.lora.as_ref() {
            Some(lw) => lw.adapter_id_for_slot(slot),
            None => 0,
        }
    }

    /// 2026-09-25: Take a ref on the pool slot a sequence uses (`< 0`
    /// resolves to the active slot, as in [`Self::adapter_id_for_slot`]).
    /// Returns the resolved index; the caller stores it and releases that
    /// index at terminal free, so a rotate that changes `active` in between
    /// cannot move the release to another counter. Returns `-1` (nothing
    /// acquired) when no LoRA pool is resident or the slot is out of range.
    pub fn acquire_adapter_slot(&self, slot: i32) -> i32 {
        match self.lora.as_ref() {
            Some(lw) => lw.acquire_slot(slot),
            None => -1,
        }
    }

    /// 2026-09-25: Release a ref taken by [`Self::acquire_adapter_slot`], by
    /// the resolved index it returned. `-1` and no pool are no-ops.
    pub fn release_adapter_slot(&self, resolved: i32) {
        if let Some(lw) = self.lora.as_ref() {
            lw.release_slot(resolved);
        }
    }

    /// 2026-09-25: The MoE-LoRA route for one request's `adapter_slot`, via
    /// `resolve_moe_lora_route`: `Skip` for a base request (`adapter_slot <
    /// 0`), `Fold` for the active adapter, `Refuse` for any other adapter, and
    /// `Fold` when no LoRA pool is resident.
    pub(crate) fn moe_lora_route(
        &self,
        adapter_slot: i32,
    ) -> metrale_model_layers::layer::MoeLoraRoute {
        let (active, has) = match self.lora.as_ref() {
            Some(lw) => (lw.active as i32, true),
            None => (-1, false),
        };
        metrale_model_layers::lora::resolve_moe_lora_route(adapter_slot, active, has)
    }

    /// 2026-09-25: The MoE-LoRA route last stamped at a `Model` decode entry,
    /// read into the decode and verify `ForwardContext`s.
    pub(crate) fn decode_moe_route(&self) -> metrale_model_layers::layer::MoeLoraRoute {
        match self
            .decode_moe_route
            .load(std::sync::atomic::Ordering::Relaxed)
        {
            0 => metrale_model_layers::layer::MoeLoraRoute::Skip,
            2 => metrale_model_layers::layer::MoeLoraRoute::Refuse,
            _ => metrale_model_layers::layer::MoeLoraRoute::Fold,
        }
    }

    fn store_decode_moe_route(&self, route: metrale_model_layers::layer::MoeLoraRoute) {
        let v = match route {
            metrale_model_layers::layer::MoeLoraRoute::Skip => 0,
            metrale_model_layers::layer::MoeLoraRoute::Fold => 1,
            metrale_model_layers::layer::MoeLoraRoute::Refuse => 2,
        };
        self.decode_moe_route
            .store(v, std::sync::atomic::Ordering::Relaxed);
    }

    /// 2026-09-25: Stamp the per-decode MoE route from a single request's `adapter_slot`.
    pub(crate) fn stamp_decode_moe_single(&self, adapter_slot: i32) {
        self.store_decode_moe_route(self.moe_lora_route(adapter_slot));
    }

    /// 2026-09-25: Stamp the per-decode MoE route for a decode batch: `Refuse`
    /// if any row routes to a non-active adapter (the batched and mixed decode
    /// entries then bail host-side through `ensure_decode_route_servable`),
    /// else `Fold` if any row folds, else `Skip` (every row base, or an empty
    /// batch).
    pub(crate) fn stamp_decode_moe_batch(&self, seqs: &[&mut crate::traits::SequenceState]) {
        use metrale_model_layers::layer::MoeLoraRoute;
        let mut any_fold = false;
        for s in seqs.iter() {
            match self.moe_lora_route(s.adapter_slot) {
                MoeLoraRoute::Refuse => {
                    self.store_decode_moe_route(MoeLoraRoute::Refuse);
                    return;
                }
                MoeLoraRoute::Fold => any_fold = true,
                MoeLoraRoute::Skip => {}
            }
        }
        self.store_decode_moe_route(if any_fold {
            MoeLoraRoute::Fold
        } else {
            MoeLoraRoute::Skip
        });
    }

    pub fn set_lora_weights(
        &mut self,
        mut lora: Option<metrale_model_layers::lora::LoraWeights>,
    ) -> Result<()> {
        if let Some(ref lw) = lora {
            // 2026-09-25: `lora_rotatable` (the `lora_rotate` lever or
            // METRALE_LORA_PEER) is what permits rotate and swap. A pool with
            // several adapters does not set it by itself.
            self.lora_rotatable =
                self.levers.lora_rotate || metrale_model_layers::lora::lora_peer_env().is_some();
            let kernels = ops::lora_delta::LoraKernels::new(self.gpu.as_ref())?;
            let active = lw.active_layers().to_vec();
            let tables = lw.tables.clone();
            let scale_table = lw.scale_table;
            let installed = self.install_lora_layers(&active, kernels, &tables, scale_table)?;
            // 2026-09-25: `slots` is padded to `max_loras` with empty-named
            // placeholders; report only the named adapters.
            let resident: Vec<String> = lw
                .adapter_names()
                .into_iter()
                .filter(|n| !n.is_empty())
                .collect();
            tracing::info!(
                "LoRA: {} adapter(s) resident [{}], active '{}' installed on \
                 {installed} layers (r={}, max_rank={}, max_loras={}, \
                 pool={:.1} MiB, rotatable={})",
                resident.len(),
                resident.join(", "),
                lw.name,
                lw.adapter_config.r,
                lw.max_rank,
                lw.max_loras,
                lw.pool_bytes as f64 / (1024.0 * 1024.0),
                self.lora_rotatable,
            );
        }
        // 2026-09-25: Build the token-overlay tables from the adapter's staged
        // raw overlay tensors against the served embed and lm_head tables. The
        // builder runs only when some slot shipped overlay tensors; otherwise
        // `self.overlays` is left as it was.
        if let Some(ref mut lw) = lora {
            let raws = std::mem::take(&mut lw.overlay_raw);
            if raws.iter().any(|r| r.is_some()) {
                self.build_token_overlays(lw.max_loras, &raws)?;
            }
        }
        self.lora = lora;
        Ok(())
    }

    /// 2026-09-25: Token-overlay build: `build_overlay` row-diffs each staged
    /// slot's raw overlay tensors against the served embed/lm_head tables and
    /// compacts the override rows, then the `[max_loras]` device tables are
    /// built. Sets `self.overlays` only when some slot overrides a row.
    fn build_token_overlays(
        &mut self,
        max_loras: usize,
        raws: &[Option<metrale_model_layers::lora::OverlayRawSlot>],
    ) -> Result<()> {
        // 2026-09-25: Treated as tied when the lm_head shares the embed buffer
        // or an NVFP4/FP8 head is served; tied, the logit recompute reuses the
        // embed override rows.
        let tied = self.lm_head_weight.weight.0 == self.embed_tokens.weight.0
            || self.lm_head_nvfp4.is_some()
            || self.lm_head_fp8.is_some();
        let vocab = self.config.vocab_size;
        let h = self.config.hidden_size;
        let served_embed = self.embed_tokens.weight;
        let served_lmhead = self.lm_head_weight.weight;
        let stream = self.gpu.default_stream();
        let mut overlays: Vec<Option<metrale_model_layers::lora::EmbedOverlay>> =
            (0..max_loras).map(|_| None).collect();
        for (k, raw) in raws.iter().enumerate() {
            if let Some(slot) = raw
                && k < max_loras
            {
                overlays[k] = metrale_model_layers::lora::build_overlay(
                    self.gpu.as_ref(),
                    &self.overlay_kernels,
                    slot,
                    served_embed,
                    served_lmhead,
                    vocab,
                    h,
                    tied,
                    stream,
                )?;
            }
        }
        let set = metrale_model_layers::lora::TokenOverlaySet::from_slots(
            self.gpu.as_ref(),
            overlays,
            max_loras,
            tied,
        )?;
        if set.any_active() {
            tracing::info!(
                "LoRA overlay: token-overlay tables built (max_n_override={}, tied={})",
                set.max_n_override,
                tied,
            );
            self.overlays = Some(set);
        }
        Ok(())
    }
}
