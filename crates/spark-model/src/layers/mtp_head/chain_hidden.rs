// SPDX-License-Identifier: AGPL-3.0-only

//! THE chain input of an MTP propose: which buffer draft `j > 0` reads as
//! "the previous position's hidden". One reader for both chains (the
//! single-sequence `propose` and the batched `propose_batch_impl`), so the
//! lever cannot steer one lane and not the other.

use spark_runtime::gpu::DevicePtr;

use super::MtpHead;
use crate::layer::ForwardContext;

impl MtpHead {
    /// Row `i` of the returned buffer is sequence `i`'s hidden from the
    /// position just drafted: the drafter's final-normed hidden
    /// (`norm_output`, what its LM head read) under
    /// `ModelLevers::mtp_chain_postnorm`, else its pre-norm residual stream
    /// (`hidden_states`). Both are written by every forward position and are
    /// read by the next position's `pre_fc_norm_hidden` BEFORE that position
    /// writes either buffer (step 2 precedes steps 4/5, same stream).
    pub(super) fn chain_hidden(ctx: &ForwardContext) -> DevicePtr {
        if ctx.levers.mtp_chain_postnorm {
            ctx.buffers.norm_output()
        } else {
            ctx.buffers.hidden_states()
        }
    }
}

impl MtpHead {
    /// The hidden a drafter row feeds to `pre_fc_norm_hidden`, for `rows`
    /// contiguous rows at `src`.
    ///
    /// Under `ModelLevers::mtp_target_postnorm`, rows that come from the
    /// TARGET (`target_rows`: the first draft, drafter prefill, exact-KV
    /// catch-up) are first run through the target's final norm into `tmp`,
    /// which is returned in their place — the tensor the reference hands its
    /// MTP. Drafter-chain rows (`target_rows == false`) and the lever-off
    /// path return `src` unchanged. `tmp` must hold `rows` x hidden BF16 and
    /// must not alias `src`.
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

    /// One-row [`Self::target_postnorm_rows`].
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
