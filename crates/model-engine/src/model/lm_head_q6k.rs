// SPDX-License-Identifier: MIT OR Apache-2.0
// provenance-id: 526f6e616c6420522e205374657369616b

//! 2026-09-25: The output head served from raw Q6_K blocks (DeepSeek-V4.1 GGUF).
//!
//! The blocks stay as the store holds them. For each group of up to `MAX_M` hidden
//! rows, `kquant_q8_1_rows_bf16` quantises the rows to `block_q8_1`, then
//! `kquant_mmvq_q6_k_w` computes their logits over the whole vocab. The factory
//! installs this head only when the store holds `lm_head.weight` as
//! [`WeightDtype::Q6K`].
//!
//! Owner: model-engine.
//! Invariants: none beyond the types.
//!
//! [`WeightDtype::Q6K`]: metrale_model_weights::weights::WeightDtype::Q6K

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, KernelHandle};

use super::types::TransformerModel;
use metrale_model_layers::layers::ops;

/// 2026-09-25: Rows of hidden state one `kquant_mmvq_*_w` launch takes (`KQ_MAX_M` in kquant_moe.cu).
const MAX_M: u32 = 8;

/// 2026-09-25: The resident Q6_K head and the two kernels that serve it.
pub(crate) struct LmHeadQ6k {
    /// 2026-09-25: `[vocab][hidden / 256]` raw 210-byte super-blocks.
    blocks: DevicePtr,
    mmvq_w: KernelHandle,
    q8_rows: KernelHandle,
    /// 2026-09-25: `block_q8_1 [MAX_M][hidden / 32]` scratch for the quantised hidden rows.
    q8_scratch: DevicePtr,
}

impl TransformerModel {
    /// 2026-09-25: Install the raw Q6_K head. `blocks` is the store's
    /// `lm_head.weight` pointer, `[vocab, hidden]` in Q6_K super-blocks. Fails
    /// when `hidden_size` is not a multiple of 256.
    pub fn set_lm_head_q6k(&mut self, blocks: DevicePtr) -> Result<()> {
        let h = self.config.hidden_size as u32;
        let v = self.config.vocab_size as u32;
        anyhow::ensure!(
            h.is_multiple_of(256),
            "Q6_K lm_head: hidden {h} is not a multiple of 256"
        );
        let mmvq_w = self.gpu.kernel(ops::KQUANT_MODULE, "kquant_mmvq_q6_k_w")?;
        let q8_rows = self
            .gpu
            .kernel(ops::KQUANT_MODULE, "kquant_q8_1_rows_bf16")?;
        let q8_scratch = self.gpu.alloc(ops::kquant_q8_1_rows_bytes(MAX_M, h))?;
        tracing::info!(
            "LM head served from raw Q6_K blocks (K-quant GEMV, vocab={v}, {} MB resident)",
            ops::kquant_weight_bytes(v, h, ops::Q6K_BLOCK_BYTES) >> 20
        );
        self.lm_head_q6k = Some(LmHeadQ6k {
            blocks,
            mmvq_w,
            q8_rows,
            q8_scratch,
        });
        Ok(())
    }

    /// 2026-09-25: Logits for `m` bf16 hidden rows (`[m, hidden]`, contiguous)
    /// into `logits` (`[m, vocab]` bf16) on the Q6_K head. `Ok(false)` when no
    /// Q6_K head is installed, so the caller falls through to its own paths.
    pub(super) fn lm_head_q6k_run(
        &self,
        hidden: DevicePtr,
        m: u32,
        logits: DevicePtr,
        stream: u64,
    ) -> Result<bool> {
        let Some(ref q6) = self.lm_head_q6k else {
            return Ok(false);
        };
        let h = self.config.hidden_size as u32;
        let v = self.config.vocab_size as u32;
        let gpu = self.gpu.as_ref();
        let mut done = 0u32;
        while done < m {
            let mm = (m - done).min(MAX_M);
            let x = hidden.offset(done as usize * h as usize * 2);
            let out = logits.offset(done as usize * v as usize * 2);
            ops::kquant_q8_1_rows(gpu, q6.q8_rows, x, q6.q8_scratch, mm, h, stream)?;
            ops::kquant_mmvq_w(
                gpu,
                q6.mmvq_w,
                q6.blocks,
                q6.q8_scratch,
                out,
                v,
                h,
                mm,
                stream,
            )?;
            done += mm;
        }
        Ok(true)
    }
}
