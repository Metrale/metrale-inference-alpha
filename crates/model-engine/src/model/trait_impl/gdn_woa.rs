// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Model side of the GDN write-on-accept K=4 verify: the one-time stash binding and the post-verdict fold gate.
//!
//! The batched verify asks for write-on-accept per call
//! (`VerifyBatchedOpts::write_on_accept`, set only by the DFlash batched
//! step). This file owns the two model-level facts the fold needs and the
//! layer cannot know:
//!
//! * whether the verify that just completed ran under the request with its
//!   pointer tables staged (`gdn_woa_eligible`, cleared at the start of every
//!   batched verify, set at its end, consumed here), and
//! * the stash memory, allocated on the first request and bound into every
//!   GDN layer before any capture bakes its address.
//!
//! Owner: model-engine speculative decoding.
//! Invariants:
//! - Once bound, the stash and flag addresses never change.
//! - A fold consumes `gdn_woa_eligible`, so at most one fold per verify can run.
//!
//! provenance-id: 526f6e616c6420522e205374657369616b

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::TransformerModel;
use metrale_model_layers::layer::{VERIFY_WY_LAYER_STRIDE_BYTES, VERIFY_WY_TABLE_SEQS};

impl TransformerModel {
    /// 2026-09-25: Bind the write-on-accept stash into every GDN layer, allocating it on
    /// the first call. Returns whether at least one layer is bound. The batched verify calls
    /// it before its graph decision, so the allocation never lands inside a graph capture.
    pub(super) fn gdn_woa_bind(&self) -> Result<bool> {
        let mut bound = self.gdn_woa_bound.lock();
        if !bound.1.is_null() {
            return Ok(bound.2 > 0);
        }
        // 2026-09-25: The per-layer stash width comes from the layer; a model with no capable
        // layer binds nothing and allocates nothing.
        let seq_floats = self
            .layers
            .iter()
            .filter_map(|l| l.gdn_woa_stash_seq_floats())
            .max()
            .unwrap_or(0);
        let n_ssm = self.config.num_ssm_layers();
        if seq_floats == 0 || n_ssm == 0 {
            // 2026-09-25: Mark as probed so the next request returns without the walk.
            *bound = (DevicePtr::NULL, DevicePtr(u64::MAX), 0);
            return Ok(false);
        }
        // 2026-09-25: The fold declines batches wider than the pointer table, so the stash
        // holds at most `VERIFY_WY_TABLE_SEQS` sequences, fewer when `max_decode_seqs` is lower.
        let seqs = VERIFY_WY_TABLE_SEQS.min((self.levers.max_decode_seqs as usize).max(2));
        let stash_bytes = n_ssm * seqs * seq_floats * 4;
        let flag_bytes = n_ssm * 4;
        let flags = self.gpu.alloc(flag_bytes)?;
        self.gpu.memset(flags, 0, flag_bytes)?;
        let stash = self.gpu.alloc(stash_bytes)?;
        tracing::info!(
            "GDN write-on-accept: bound {:.1} MB stash ({} GDN layers x {} seqs) on first request",
            stash_bytes as f64 / 1e6,
            n_ssm,
            seqs
        );
        let mut ssm_idx = 0usize;
        for (i, layer) in self.layers.iter().enumerate() {
            if self.config.layer_type(i) != metrale_config::LayerType::LinearAttention {
                continue;
            }
            layer.gdn_woa_bind(
                flags.offset(ssm_idx * 4),
                stash.offset(ssm_idx * seqs * seq_floats * 4),
                seqs,
            );
            ssm_idx += 1;
        }
        *bound = (flags, stash, seqs);
        Ok(true)
    }

    /// 2026-09-25: The post-verdict fold. Declines (`Ok(false)`, the host h restore runs)
    /// unless the batched verify that just completed ran under a write-on-accept request
    /// with its pointer tables staged; the flag is consumed, so a second call folds nothing.
    ///
    /// `gdn_woa_folded_slots` is cleared on every call, so it lists only the slots of the
    /// most recent fold. Its reader (`async_chkpt.rs`) consumes the entry for its own
    /// `slot_idx` to skip the h restore, and slot indices are reused, so an entry left from
    /// an earlier batch would skip a restore that is needed.
    pub(super) fn gdn_fold_accepted_dispatch(
        &self,
        slots: &[usize],
        accepted_rows: &[u32],
        k_rows: usize,
    ) -> Result<bool> {
        self.gdn_woa_folded_slots.lock().clear();
        let eligible = self
            .gdn_woa_eligible
            .swap(false, std::sync::atomic::Ordering::AcqRel);
        if !eligible
            || self.gdn_woa_na_tab.is_null()
            || self.verify_wy_tables.is_null()
            || accepted_rows.is_empty()
            || accepted_rows.len() > VERIFY_WY_TABLE_SEQS
        {
            return Ok(false);
        }
        let mut host = [0u32; VERIFY_WY_TABLE_SEQS];
        host[..accepted_rows.len()].copy_from_slice(accepted_rows);
        let bytes: Vec<u8> = host.iter().flat_map(|v| v.to_le_bytes()).collect();
        let stream = self.gpu.default_stream();
        self.gpu
            .copy_h2d_async(&bytes, self.gdn_woa_na_tab, stream)?;
        let mut ssm_idx = 0usize;
        let mut any = false;
        for (i, layer) in self.layers.iter().enumerate() {
            if self.config.layer_type(i) != metrale_config::LayerType::LinearAttention {
                continue;
            }
            let h_table = self
                .verify_wy_tables
                .offset(ssm_idx * VERIFY_WY_LAYER_STRIDE_BYTES);
            any |= layer.gdn_fold_accepted(
                self.gpu.as_ref(),
                h_table,
                self.gdn_woa_na_tab,
                k_rows,
                accepted_rows.len(),
                stream,
            )?;
            ssm_idx += 1;
        }
        if any {
            self.gdn_woa_folded_slots.lock().extend_from_slice(slots);
        }
        Ok(any)
    }
}
