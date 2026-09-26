// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The PLE per-sequence carry (token history and conv state):
//! snapshot and restore for the layer's aux state, reset, the decode
//! prestage, and release.
//!
//! Owner: model-layers (PLE).
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

use super::{PleLayer, PleSeqState};
use crate::layers::ple::ids::ple_ngram_ids;

impl PleLayer {
    /// 2026-09-25: The aux blob: `hist_len` (u32), the history (u32 each),
    /// then the conv state's f32 bytes, all little-endian.
    pub fn snapshot_aux(
        &self,
        st: &PleSeqState,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<Vec<u8>> {
        let conv_bytes = self.state_len * self.hc_mult * self.hidden * 4;
        let mut blob = Vec::with_capacity(4 + st.history.len() * 4 + conv_bytes);
        blob.extend_from_slice(&(st.history.len() as u32).to_le_bytes());
        for t in &st.history {
            blob.extend_from_slice(&t.to_le_bytes());
        }
        let off = blob.len();
        blob.resize(off + conv_bytes, 0);
        gpu.copy_d2h_on_stream(st.conv, &mut blob[off..], stream)?;
        Ok(blob)
    }

    /// 2026-09-25: Restore a blob from [`Self::snapshot_aux`] and clear
    /// `prestaged_va`. Errors when the blob's size does not match.
    pub fn restore_aux(
        &self,
        st: &mut PleSeqState,
        blob: &[u8],
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        anyhow::ensure!(blob.len() >= 4, "PLE aux blob truncated");
        let n = u32::from_le_bytes(blob[..4].try_into().unwrap()) as usize;
        let conv_bytes = self.state_len * self.hc_mult * self.hidden * 4;
        anyhow::ensure!(
            blob.len() == 4 + n * 4 + conv_bytes,
            "PLE aux blob size mismatch"
        );
        st.history = blob[4..4 + n * 4]
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        st.prestaged_va = None;
        gpu.copy_h2d_async(&blob[4 + n * 4..], st.conv, stream)?;
        Ok(())
    }
}

impl PleLayer {
    /// 2026-09-25: Fresh sequence: EOS-filled history and a zeroed conv
    /// state.
    pub(super) fn reset(
        &self,
        st: &mut PleSeqState,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        st.history = vec![self.dims.eos_token_id; self.dims.context_len()];
        st.prestaged_va = None;
        let zeros = vec![0u8; self.state_len * self.hc_mult * self.hidden * 4];
        gpu.copy_h2d_async(&zeros, st.conv, stream)?;
        Ok(())
    }

    /// 2026-09-25: The per-step host work of a decode step: the n-gram hash,
    /// the row gather (`gather_host`, which uploads the slots into
    /// `slots_dev` from host memory) and the history advance. It runs
    /// through `decode_prestage` before any graph replay or capture; the
    /// next `forward` consumes `prestaged_va` and does not advance the
    /// history again.
    pub fn prestage(
        &self,
        st: &mut PleSeqState,
        tokens: &[u32],
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        if st.history.len() != self.dims.context_len() {
            self.reset(st, gpu, stream)?;
        }
        let mut window = st.history.clone();
        window.extend_from_slice(tokens);
        let all = ple_ngram_ids(&self.dims, &window);
        let rows = &all[all.len() - tokens.len()..];
        let flat: Vec<u64> = rows.iter().flat_map(|r| r.iter().copied()).collect();
        let va = self.gather_host(&flat, gpu, stream)?;
        let keep = self.dims.context_len();
        st.history = window[window.len() - keep..].to_vec();
        st.prestaged_va = Some(va);
        st.last_staged_va = va;
        Ok(())
    }

    /// 2026-09-25: Free one sequence's PLE carry; `conv` is a bare
    /// `DevicePtr`, so dropping `PleSeqState` frees nothing. Idempotent:
    /// `conv` is nulled, and the state cleared, even when the free fails,
    /// whose error is returned.
    pub fn release_seq_state(&self, st: &mut PleSeqState, gpu: &dyn GpuBackend) -> Result<()> {
        if st.conv.is_null() {
            return Ok(());
        }
        let r = gpu.free(st.conv);
        st.conv = DevicePtr(0);
        st.history.clear();
        st.prestaged_va = None;
        st.last_staged_va = 0;
        r
    }
}
