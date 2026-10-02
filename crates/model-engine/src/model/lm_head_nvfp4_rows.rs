// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-10-02: The NVFP4 LM head at its declared W4A16 on the tensor-core row tiles
//! (`ops::w4a16_tc_rows`), at every row count in 64-row calls. Installed under
//! `--weight-quantization declared` for a checkpoint that declares an NVFP4 head with 16-bit
//! activations (nvidia/Qwen3.6-35B-A3B-NVFP4: `W4A16_NVFP4`). It stands in for every other NVFP4
//! head path, among them the transposed tile GEMM, which casts activations to E4M3 (below the
//! declared precision), and the by-rows GEMV/GEMM switch, whose summation order depends on the
//! row count. A row's logits here do not depend on the row count.
//!
//! Owner: model-engine (LM head).
//! Invariants: `lm_head_nvfp4_rows_run` writes `logits[rows, vocab]` BF16 and returns `true`
//! exactly when the row tiles are installed.

use anyhow::{Context, Result};
use metrale_gpu_runtime::gpu::DevicePtr;
use metrale_model_layers::layers::ops;

use super::types::TransformerModel;

impl TransformerModel {
    /// 2026-10-02: Install the W4A16 row-tile head when the declared-precision policy declares an
    /// NVFP4 head with 16-bit activations, the model loaded the NVFP4 head, the target ships the
    /// kernels, the head fits their shape contract, and the vocab is not sharded.
    pub fn install_declared_lm_head_w4a16_rows(&mut self) -> Result<()> {
        use metrale_config::weight_quantization::{LmHeadChoice, LmHeadFormat};
        let policy = metrale_config::WeightQuantPolicy::for_checkpoint(
            metrale_model_layers::layers::weight_quantization(),
            self.config.quantization_config.as_ref(),
            metrale_model_layers::layers::kernel_caps(),
        );
        let declared_a16 = policy.lm_head() == LmHeadChoice::Declared(LmHeadFormat::Nvfp4)
            && policy.declared("lm_head").activation.is_none();
        let (h, v) = (
            self.config.hidden_size as u32,
            self.config.vocab_size as u32,
        );
        if !declared_a16
            || self.lm_head_nvfp4.is_none()
            || self.config.tp_world_size > 1
            || !self.gpu.has_module(ops::W4A16_TC_ROWS_MODULE)
            || !ops::w4a16_tc_rows_shape_ok(1, v, h, h, v)
        {
            return Ok(());
        }
        self.lm_head_rows.nvfp4_rows = true;
        tracing::info!(
            "lm_head: declared NVFP4 W4A16 on the tensor-core row tiles at every row count"
        );
        Ok(())
    }

    /// 2026-10-02: `logits[rows, vocab]` from `hidden[rows, hidden]` through the W4A16 row tiles;
    /// `Ok(false)`, launching nothing, unless they are installed.
    pub(super) fn lm_head_nvfp4_rows_run(
        &self,
        hidden: DevicePtr,
        rows: usize,
        logits: DevicePtr,
        stream: u64,
    ) -> Result<bool> {
        if !self.lm_head_rows.nvfp4_rows {
            return Ok(false);
        }
        let head = self
            .lm_head_nvfp4
            .as_ref()
            .context("W4A16 row-tile lm_head needs the NVFP4 head")?;
        let (h, v) = (self.config.hidden_size, self.config.vocab_size);
        let mut done = 0;
        while done < rows {
            let m = (rows - done).min(ops::W4A16_TC_ROWS_MAX_M as usize);
            ops::w4a16_tc_rows(
                self.gpu.as_ref(),
                hidden.offset(done * h * 2),
                head,
                logits.offset(done * v * 2),
                m as u32,
                v as u32,
                h as u32,
                h as u32,
                v as u32,
                stream,
            )?;
            done += m;
        }
        Ok(true)
    }
}
