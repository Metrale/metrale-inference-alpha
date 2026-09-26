// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The BF16 copies of the per-row FP8 `in_proj_qkvz` and
//! `out_proj` that the two `METRALE_FP8_ROWWISE` GDN prefill arms multiply
//! (`trait_prefill_proj.rs`, `trait_prefill_helper.rs`).
//!
//! The copies live in the arena slab `BufferSizes::ssm_rowwise_w_bf16`, which
//! `metrale_gpu_runtime::buffers::sizes_rowwise` sizes as `num_ssm_layers` x
//! (`in_proj_qkvz` + `out_proj`) when `METRALE_FP8_ROWWISE=1` and as 0
//! otherwise. Each layer carves its two slices on its first prefill through
//! the arm and keeps their addresses in `qkvz_rowwise_bf16` /
//! `out_proj_rowwise_bf16`. The conversion is exact: every FP8 E4M3 value is
//! representable in BF16.
//!
//! Owner: model-layers (qwen3_ssm).
//! Invariants:
//! - No device allocation: a slice comes from the slab, or the call errors
//!   (slab absent or exhausted).
//! - A layer's slot is written only after its dequant launch returned `Ok`.

use anyhow::{Context, Result};
use std::sync::atomic::{AtomicU64, Ordering};

use super::Qwen3SsmLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;
use crate::weight_map::Fp8Weight;
use metrale_gpu_runtime::gpu::DevicePtr;

impl Qwen3SsmLayer {
    /// 2026-09-25: The BF16 `in_proj_qkvz` for the row-wise prefill arm.
    pub(super) fn rowwise_qkvz_bf16(
        &self,
        ctx: &ForwardContext,
        fp8w: &Fp8Weight,
        stream: u64,
    ) -> Result<DevicePtr> {
        self.rowwise_bf16(&self.qkvz_rowwise_bf16, ctx, fp8w, "in_proj_qkvz", stream)
    }

    /// 2026-09-25: The BF16 `out_proj` for the row-wise prefill arm.
    pub(super) fn rowwise_out_proj_bf16(
        &self,
        ctx: &ForwardContext,
        fp8w: &Fp8Weight,
        stream: u64,
    ) -> Result<DevicePtr> {
        self.rowwise_bf16(&self.out_proj_rowwise_bf16, ctx, fp8w, "out_proj", stream)
    }

    /// 2026-09-25: Dequantize `fp8w` into a fresh slice of the slab on the first
    /// call, store the slice in `slot`, and return it; while `slot` is non-zero,
    /// return it without launching. Allocates nothing (`rowwise_alloc_tests.rs`).
    fn rowwise_bf16(
        &self,
        slot: &AtomicU64,
        ctx: &ForwardContext,
        fp8w: &Fp8Weight,
        what: &str,
        stream: u64,
    ) -> Result<DevicePtr> {
        let cached = slot.load(Ordering::Relaxed);
        if cached != 0 {
            return Ok(DevicePtr(cached));
        }
        let bytes = ops::dequant_fp8_bf16_bytes(fp8w);
        let dst = ctx
            .buffers
            .take_ssm_rowwise_w_bf16(bytes)
            .with_context(|| format!("ssm prefill: row-wise BF16 {what} weight"))?;
        ops::dequant_fp8_bf16_into(ctx.gpu, fp8w, dst, stream)
            .with_context(|| format!("ssm prefill: row-wise BF16 dequant of {what}"))?;
        slot.store(dst.0, Ordering::Relaxed);
        Ok(dst)
    }
}
