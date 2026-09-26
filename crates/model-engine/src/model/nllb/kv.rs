// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Per-sequence KV for the served NLLB model.
//!
//! A sequence owns two kinds of KV: the cross-attention K/V, `[enc_len, d]` per decoder
//! layer, computed once from the encoder at prefill and read by every decode step; and
//! the decoder self-attention K/V, `[cache_rows, d]` per layer, filled one row per
//! decoded token. The model keeps one `NllbSeqKv` per `slot_idx`, created in
//! `alloc_sequence` and freed in `free_sequence`.
//!
//! Owner: model-engine.
//! Invariants: none beyond the types.

use anyhow::{Result, bail};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

/// 2026-09-25: One sequence's decoder self-attention KV and cross-attention KV.
pub(super) struct NllbSeqKv {
    /// 2026-09-25: Decoder self-attention K per layer (`self_v` holds V), `[cache_rows, d]` bf16.
    pub self_k: Vec<DevicePtr>,
    pub self_v: Vec<DevicePtr>,
    /// 2026-09-25: Cross-attention K/V per layer, `[enc_len, d]` bf16; empty until
    /// `alloc_cross`.
    pub cross_k: Vec<DevicePtr>,
    pub cross_v: Vec<DevicePtr>,
    /// 2026-09-25: Encoder length behind the cross-KV; `0` while it is empty.
    pub enc_len: usize,
    /// 2026-09-25: Next decoder row to write: the decoder tokens processed since
    /// the last `alloc_cross`.
    pub dec_pos: usize,
    cache_rows: usize,
}

impl NllbSeqKv {
    /// 2026-09-25: Allocate the decoder self-attention KV. The cross-KV stays
    /// empty until `alloc_cross`, since its size depends on the source length.
    pub(super) fn new(
        gpu: &dyn GpuBackend,
        dec_layers: usize,
        cache_rows: usize,
        d: usize,
    ) -> Result<Self> {
        let mut self_k = Vec::with_capacity(dec_layers);
        let mut self_v = Vec::with_capacity(dec_layers);
        for _ in 0..dec_layers {
            self_k.push(gpu.alloc(cache_rows * d * 2)?);
            self_v.push(gpu.alloc(cache_rows * d * 2)?);
        }
        Ok(Self {
            self_k,
            self_v,
            cross_k: Vec::new(),
            cross_v: Vec::new(),
            enc_len: 0,
            dec_pos: 0,
            cache_rows,
        })
    }

    /// 2026-09-25: Allocate the cross-attention KV for a source of `enc_len`
    /// rows, after freeing any previous cross buffers, and reset `dec_pos` to 0.
    pub(super) fn alloc_cross(
        &mut self,
        gpu: &dyn GpuBackend,
        dec_layers: usize,
        enc_len: usize,
        d: usize,
    ) -> Result<()> {
        self.free_cross(gpu)?;
        for _ in 0..dec_layers {
            self.cross_k.push(gpu.alloc(enc_len * d * 2)?);
            self.cross_v.push(gpu.alloc(enc_len * d * 2)?);
        }
        self.enc_len = enc_len;
        self.dec_pos = 0;
        Ok(())
    }

    /// 2026-09-25: Fails when the self-attention cache has no row left for
    /// another decoder token.
    pub(super) fn ensure_room(&self) -> Result<()> {
        if self.dec_pos >= self.cache_rows {
            bail!(
                "nllb: decoder length exceeded cache_rows={} — raise --max-model-len",
                self.cache_rows
            );
        }
        Ok(())
    }

    fn free_cross(&mut self, gpu: &dyn GpuBackend) -> Result<()> {
        for p in self.cross_k.drain(..).chain(self.cross_v.drain(..)) {
            gpu.free(p)?;
        }
        self.enc_len = 0;
        Ok(())
    }

    /// 2026-09-25: Free every device buffer this sequence owns.
    pub(super) fn free(mut self, gpu: &dyn GpuBackend) -> Result<()> {
        self.free_cross(gpu)?;
        for p in self.self_k.drain(..).chain(self.self_v.drain(..)) {
            gpu.free(p)?;
        }
        Ok(())
    }
}
