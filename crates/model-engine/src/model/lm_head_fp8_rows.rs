// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-28: The FP8 E4M3 LM head (per-row F32 scales, `lm_head_fp8`) at any row count, in
//! as few weight passes as the kernels allow:
//! - declared W8A8 (`set_lm_head_w8a8`, installed when the checkpoint declares the head FP8
//!   W8A8): `ops::w8a8_proj` at every row count, in `ops::W8A8_MAX_ROWS`-row calls (one weight
//!   pass per 128 rows), never per row; a row's logits do not depend on the row count;
//! - otherwise W8A16: `dense_gemv_fp8w` for 1 row, `dense_gemv_fp8w_batch2` for 2, and the
//!   register-tiled `fp8_gemv_rowscale_batch{8,16}_rt2` for 3..=16 and in 16-row chunks above.
//!   Before this, every row count but 2 ran one GEMV per row, reading the 1.27 GB head once
//!   per row (4x at the decode-floor recipe's 4 verify rows).
//!
//! Owner: model-engine (LM head).
//! Invariants: `lm_head_fp8_run` writes `logits[rows, vocab]` BF16 and returns `true` exactly
//! when `lm_head_fp8` is set.

use anyhow::{Context, Result, ensure};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_model_layers::layers::W8a8Ctx;
use metrale_model_layers::layers::ops::{self, W8a8Weight};
use metrale_model_layers::weight_map::{Fp8Weight, WeightQuantFormat};

use super::types::TransformerModel;

/// 2026-09-28: The multi-row FP8 head kernels, and the W8A8 head when installed. 2026-10-02:
/// `nvfp4_rows`, set when the NVFP4 head runs the declared W4A16 row tiles
/// (`lm_head_nvfp4_rows.rs`).
pub(crate) struct LmHeadRows {
    rt8: KernelHandle,
    rt16: KernelHandle,
    w8a8: Option<(W8a8Ctx, W8a8Weight)>,
    pub(super) nvfp4_rows: bool,
}

impl LmHeadRows {
    pub(crate) fn resolve(gpu: &dyn GpuBackend) -> Self {
        let rt = |f| metrale_model_layers::layers::try_kernel(gpu, "fp8_gemv_rt", f);
        Self {
            rt8: rt("fp8_gemv_rowscale_batch8_rt2"),
            rt16: rt("fp8_gemv_rowscale_batch16_rt2"),
            w8a8: None,
            nvfp4_rows: false,
        }
    }
}

impl TransformerModel {
    /// 2026-09-28: Install the W8A8 head when the declared-precision policy wants the head run
    /// W8A8 (`WeightQuantPolicy::fp8_decode_act("lm_head") == Some(Fp8)`) and the model loaded
    /// the FP8 head (`lm_head_fp8`, the declared head format under `declared`).
    pub fn install_declared_lm_head_w8a8(&mut self) -> Result<()> {
        let policy = metrale_config::WeightQuantPolicy::for_checkpoint(
            metrale_model_layers::layers::weight_quantization(),
            self.config.quantization_config.as_ref(),
            metrale_model_layers::layers::kernel_caps(),
        );
        let fp8 = policy.fp8_decode_act("lm_head")
            == Some(metrale_config::weight_quantization::ActFormat::Fp8);
        if !fp8 || self.lm_head_fp8.is_none() {
            return Ok(());
        }
        let ctx = W8a8Ctx::new(self.gpu.as_ref(), self.config.hidden_size as u32)?;
        self.set_lm_head_w8a8(ctx)?;
        tracing::info!("W8A8 lm_head (declared FP8 W8A8, per-row E4M3 read in place)");
        Ok(())
    }

    /// 2026-09-28: Run the FP8 head W8A8 (the checkpoint's declared precision) from now on.
    /// Needs `lm_head_fp8`, and kernels and scratch that serve one row.
    pub fn set_lm_head_w8a8(&mut self, ctx: W8a8Ctx) -> Result<()> {
        let fp8 = self
            .lm_head_fp8
            .context("W8A8 lm_head needs the FP8 head (lm_head_fp8)")?;
        let w = W8a8Weight::new(&[Fp8Weight {
            weight: fp8.weight,
            row_scale: fp8.row_scale,
            n: self.config.vocab_size as u32,
            k: self.config.hidden_size as u32,
            scale_format: WeightQuantFormat::Fp8PerRow,
        }])?;
        ensure!(
            ctx.available(&w, 1),
            "W8A8 lm_head kernels or scratch unavailable"
        );
        self.lm_head_rows.w8a8 = Some((ctx, w));
        Ok(())
    }

    /// 2026-09-28: `logits[rows, vocab]` from `hidden[rows, hidden]` through the FP8 head;
    /// `Ok(false)`, launching nothing, when the model has no FP8 head.
    pub(super) fn lm_head_fp8_run(
        &self,
        hidden: DevicePtr,
        rows: usize,
        logits: DevicePtr,
        stream: u64,
    ) -> Result<bool> {
        let Some(ref fp8) = self.lm_head_fp8 else {
            return Ok(false);
        };
        let gpu = self.gpu.as_ref();
        let (h, v) = (self.config.hidden_size, self.config.vocab_size);
        if let Some((ref ctx, ref w)) = self.lm_head_rows.w8a8 {
            // 2026-09-28: Any row count: `W8A8_MAX_ROWS`-row calls (the scratch's capacity); the
            // quantization is per row, so chunking does not change any row's bits.
            let mut done = 0;
            while done < rows {
                let m = (rows - done).min(ops::W8A8_MAX_ROWS);
                let (x, out) = (hidden.offset(done * h * 2), logits.offset(done * v * 2));
                ensure!(
                    ctx.proj(gpu, w, x, h as u32, m, out, v as u32, stream)?,
                    "W8A8 lm_head: {m} rows not servable"
                );
                done += m;
            }
            return Ok(true);
        }
        let k = &self.lm_head_rows;
        let mut done = 0;
        while done < rows {
            let (x, out, left) = (
                hidden.offset(done * h * 2),
                logits.offset(done * v * 2),
                rows - done,
            );
            let step = match left {
                1 => {
                    ops::dense_gemv_fp8w(
                        gpu,
                        self.dense_gemv_fp8w_kernel,
                        x,
                        fp8,
                        out,
                        v as u32,
                        h as u32,
                        stream,
                    )?;
                    1
                }
                2 if self.dense_gemv_fp8w_batch2_kernel.0 != 0 => {
                    let kh = self.dense_gemv_fp8w_batch2_kernel;
                    ops::dense_gemv_fp8w_batch2(gpu, kh, x, fp8, out, v as u32, h as u32, stream)?;
                    2
                }
                3..=8 if k.rt8.0 != 0 => {
                    let m = left as u32;
                    ops::fp8_gemv_rowscale_batch8_rt2(
                        gpu, k.rt8, x, fp8, out, m, v as u32, h as u32, stream,
                    )?;
                    left
                }
                _ if left > 2 && k.rt16.0 != 0 => {
                    let m = left.min(16);
                    let (mu, (vu, hu)) = (m as u32, (v as u32, h as u32));
                    ops::fp8_gemv_rowscale_batch16_rt2(
                        gpu, k.rt16, x, fp8, out, mu, vu, hu, stream,
                    )?;
                    m
                }
                _ => {
                    ops::dense_gemv_fp8w(
                        gpu,
                        self.dense_gemv_fp8w_kernel,
                        x,
                        fp8,
                        out,
                        v as u32,
                        h as u32,
                        stream,
                    )?;
                    1
                }
            };
            done += step;
        }
        Ok(true)
    }
}
