// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Gemma-4 decode-logits diagnostic (`METRALE_DIAG_GEMMA4`).
//!
//! Owner: model-engine (decode).
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_gpu_runtime::gpu::GpuBackend;

use super::super::types::TransformerModel;

impl TransformerModel {
    /// 2026-09-25: With `METRALE_DIAG_GEMMA4` set to `1` or `true`, log the logit
    /// range and the top 5 (id, logit) pairs after a decode step, to tell a
    /// near-tie from a confident bad pick. The flag also sets `suppress_graphs` at
    /// load (impl_a1.rs), so decode runs eagerly and the synchronise here is legal;
    /// FP8-KV calibration clears that flag once it freezes.
    pub(super) fn diag_gemma4_decode_logits(&self, token: u32, stream: u64) -> Result<()> {
        if !std::env::var("METRALE_DIAG_GEMMA4").is_ok_and(|v| v == "1" || v == "true") {
            return Ok(());
        }
        self.gpu.synchronize(stream)?;
        let n_logits = self.config.vocab_size;
        // 2026-09-25: Read the buffer `lm_head` wrote: `logits_fp32_buf` when
        // `use_fp32_logits`, else the BF16 logits buffer.
        let logit_vals: Vec<f32> = if self.use_fp32_logits {
            let mut buf = vec![0u8; n_logits * 4];
            if let Err(e) = self.gpu.copy_d2h(self.logits_fp32_buf, &mut buf) {
                tracing::error!("METRALE_DIAG_GEMMA4: copy_d2h(logits_fp32_buf): {e:#}");
            }
            buf.chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        } else {
            let mut buf = vec![0u8; n_logits * 2];
            if let Err(e) = self.gpu.copy_d2h(self.buffers.logits(), &mut buf) {
                tracing::error!("METRALE_DIAG_GEMMA4: copy_d2h(logits BF16): {e:#}");
            }
            buf.chunks_exact(2)
                .map(|c| {
                    let bits = u16::from_le_bytes([c[0], c[1]]);
                    f32::from_bits((bits as u32) << 16)
                })
                .collect()
        };
        let max = logit_vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min = logit_vals.iter().cloned().fold(f32::INFINITY, f32::min);
        let mut idx: Vec<usize> = (0..logit_vals.len()).collect();
        idx.sort_by(|&a, &b| {
            logit_vals[b]
                .partial_cmp(&logit_vals[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let top5: Vec<(usize, f32)> = idx.iter().take(5).map(|&i| (i, logit_vals[i])).collect();
        tracing::warn!(
            "DIAG decode logits: max={max:.4} min={min:.4} prev_token={token} top5={top5:?}",
        );
        Ok(())
    }
}
