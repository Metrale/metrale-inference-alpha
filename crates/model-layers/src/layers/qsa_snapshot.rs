// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Serialise and restore one sequence's QSA indexer state, the aux blob
//! stored with an SSM prefix-cache snapshot.
//!
//! Owner: model-layers (QSA).
//! Invariants:
//! - `restore_aux` sets the counters only after the size checks pass and the
//!   raw-key upload is issued; `pooled` stays 0 unless re-pooling was issued.

use anyhow::Result;
use metrale_gpu_runtime::gpu::GpuBackend;

use super::{QsaIndexer, QsaSeqState};
use crate::layers::ops;

impl QsaIndexer {
    /// 2026-09-25: Aux blob: `[ingested u64][pooled u64][raw_keys bf16 bytes]`.
    /// Block keys are not serialised: `restore_aux` re-pools them from the raw
    /// keys with one `qsa_block_pool` launch.
    pub fn snapshot_aux(
        &self,
        st: &QsaSeqState,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<Vec<u8>> {
        let hd = self.hd as usize;
        let key_bytes = st.ingested * hd * 2;
        let mut blob = Vec::with_capacity(16 + key_bytes);
        blob.extend_from_slice(&(st.ingested as u64).to_le_bytes());
        blob.extend_from_slice(&(st.pooled as u64).to_le_bytes());
        let off = blob.len();
        blob.resize(off + key_bytes, 0);
        if key_bytes > 0 {
            gpu.copy_d2h_on_stream(st.raw_keys, &mut blob[off..], stream)?;
        }
        Ok(blob)
    }

    /// 2026-09-25: Restore the blob from [`Self::snapshot_aux`] on a prefix-cache hit:
    /// upload the raw keys, reset the counters, re-pool the block keys.
    pub fn restore_aux(
        &self,
        st: &mut QsaSeqState,
        blob: &[u8],
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        anyhow::ensure!(blob.len() >= 16, "QSA aux blob truncated");
        let ingested = u64::from_le_bytes(blob[..8].try_into().unwrap()) as usize;
        let pooled = u64::from_le_bytes(blob[8..16].try_into().unwrap()) as usize;
        let hd = self.hd as usize;
        anyhow::ensure!(
            blob.len() == 16 + ingested * hd * 2,
            "QSA aux blob size mismatch"
        );
        anyhow::ensure!(ingested <= self.max_tokens, "QSA aux exceeds key cache");
        if ingested > 0 {
            gpu.copy_h2d_async(&blob[16..], st.raw_keys, stream)?;
        }
        st.ingested = ingested;
        st.pooled = 0;
        if pooled > 0 {
            ops::qsa_block_pool(
                gpu,
                self.k_pool_k,
                st.raw_keys,
                self.k_norm_w,
                st.block_keys,
                0,
                pooled as u32,
                self.ratio,
                self.hd,
                self.rot,
                self.theta,
                self.eps,
                stream,
            )?;
            st.pooled = pooled;
        }
        Ok(())
    }
}
