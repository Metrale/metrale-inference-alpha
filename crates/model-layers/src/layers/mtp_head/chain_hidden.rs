// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Hidden inputs of an MTP propose: `chain_hidden` picks the buffer draft
//! `j > 0` reads as the previous position's hidden, and `target_postnorm_rows`
//! applies the target's final norm to rows that come from the target.
//!
//! Owner: model-layers (MTP head).
//! Invariants:
//! - `MtpHead`'s single-sequence `propose` and the batched `propose_batch_impl`
//!   read the chain hidden only through `chain_hidden`. The multi-module
//!   `MultiModuleMtpHead::propose` does not use it.

use metrale_gpu_runtime::gpu::DevicePtr;

use super::MtpHead;
use crate::layer::ForwardContext;

impl MtpHead {
    /// 2026-09-25: Row `i` is sequence `i`'s hidden from the position just drafted:
    /// the final-normed row its LM head read (`norm_output`) under
    /// `ModelLevers::mtp_chain_postnorm`, else the pre-norm residual stream
    /// (`hidden_states`). The next position's `pre_fc_norm_hidden` norm reads it
    /// before that position's fc GEMM overwrites `hidden_states`.
    pub(super) fn chain_hidden(ctx: &ForwardContext) -> DevicePtr {
        if ctx.levers.mtp_chain_postnorm {
            ctx.buffers.norm_output()
        } else {
            ctx.buffers.hidden_states()
        }
    }
}

impl MtpHead {
    /// 2026-09-25: The hidden a drafter row feeds to `pre_fc_norm_hidden`, for `rows`
    /// contiguous rows at `src`. Under `ModelLevers::mtp_target_postnorm`, target
    /// rows (`target_rows`: the first draft, the drafter prefill, the catch-up
    /// rows) go through the target's final norm into `tmp`, which is returned.
    /// Otherwise, or when no target final norm was supplied, `src` is returned.
    /// `tmp` must hold `rows` x hidden BF16 and must not alias `src`.
    pub(super) fn target_postnorm_rows(
        &self,
        ctx: &ForwardContext,
        target_rows: bool,
        src: DevicePtr,
        tmp: DevicePtr,
        rows: u32,
        stream: u64,
    ) -> anyhow::Result<DevicePtr> {
        if !(target_rows && ctx.levers.mtp_target_postnorm)
            || self.target_final_norm.weight.is_null()
        {
            return Ok(src);
        }
        crate::layers::ops::rms_norm(
            ctx.gpu,
            self.rms_norm_k,
            src,
            &self.target_final_norm,
            tmp,
            rows,
            ctx.config.hidden_size as u32,
            ctx.config.rms_norm_eps as f32,
            stream,
        )?;
        Ok(tmp)
    }

    /// 2026-09-25: One-row [`Self::target_postnorm_rows`].
    pub(super) fn target_postnorm_row(
        &self,
        ctx: &ForwardContext,
        target_row: bool,
        src: DevicePtr,
        tmp: DevicePtr,
        stream: u64,
    ) -> anyhow::Result<DevicePtr> {
        self.target_postnorm_rows(ctx, target_row, src, tmp, 1, stream)
    }
}
