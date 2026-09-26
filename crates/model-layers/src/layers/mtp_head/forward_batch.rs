// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Batched cross-sequence MTP propose. Drafts chain within a
//! sequence but not across sequences, so each draft position is one drafter
//! forward over n sequences, with n-row weight projections (`proj_rows`,
//! `gemm_rows`) and per-row attention.
//!
//! Owner: model-layers (MTP head).
//! Invariants:
//! - `propose_batch_impl` is called only from `DraftProposer::propose_batch`,
//!   after `MtpHead::can_propose_batch` admits n (2 <= n <= `propose_batch_max`).
//! - fc, k and v must be BF16 and q and o BF16 or weight-only NVFP4; any other
//!   projection returns an error.

use anyhow::{Result, ensure};
use metrale_gpu_runtime::gpu::DevicePtr;

use super::row_dispatch;
use super::{MtpHead, MtpProposerState, ProjectionWeight};
use crate::layer::ForwardContext;
use crate::layers::ops;
use crate::weight_map::DenseWeight;

use crate::layers::mtp_meta::pack_mtp_attn_meta;
mod position;

/// 2026-09-25: Byte offset in `scratch` of the per-row FP32 top-1
/// log-probabilities. The n argmax ids occupy `scratch[0..n*4)`, and
/// n <= `PROPOSE_META_SEQS` (32) keeps them below this offset.
const LP_SCRATCH_OFF: usize = 256;

impl MtpHead {
    /// 2026-09-25: n-row projection of a BF16 or weight-only NVFP4 weight.
    /// BF16 goes to [`Self::gemm_rows`]. NVFP4 runs one `ops::w4a16_gemv_batchm`
    /// launch on the narrowest resolved `w4a16_gemv_batch{4..8}` tier that
    /// covers `m` (that op takes a tensor-core entry when `gemv_tc::tc_kernel`
    /// resolves one), else one `w4a16_decode_gemv` per row. Any other
    /// precision returns an error.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn proj_rows(
        &self,
        gpu: &dyn metrale_gpu_runtime::gpu::GpuBackend,
        input: DevicePtr,
        w: &ProjectionWeight,
        output: DevicePtr,
        m: usize,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<()> {
        match w {
            ProjectionWeight::Bf16(d) => self.gemm_rows(gpu, input, d, output, m, n, k, stream),
            ProjectionWeight::Nvfp4(q) => {
                let kh = self.w4a16_batchm.kernel(m as u32);
                if kh.0 != 0 {
                    return ops::w4a16_gemv_batchm(
                        gpu, kh, input, q, output, m as u32, n, k, stream,
                    );
                }
                for r in 0..m {
                    ops::w4a16_decode_gemv(
                        gpu,
                        self.w4a16_gemv_k,
                        self.w4a16_gemv_sw_k,
                        self.gemv_sw,
                        input.offset(r * k as usize * 2),
                        q,
                        output.offset(r * n as usize * 2),
                        n,
                        k,
                        stream,
                    )?;
                }
                Ok(())
            }
            _ => anyhow::bail!("propose_batch: FP8 projection (can_propose_batch lied)"),
        }
    }

    /// 2026-09-25: n-row BF16 projection. The tensor-core GEMV
    /// (`ops::dense_gemv_tc::try_dense_gemv_tc`, off when `METRALE_NO_MTP_TC`
    /// is set non-empty) runs first. When it launches nothing,
    /// [`row_dispatch::drafter_row_kernel`] picks the batched GEMV, the
    /// pipelined tile GEMM or a per-row GEMV loop. Every arm reads `input` as
    /// `[m, k]` contiguous and writes m contiguous rows of n elements.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn gemm_rows(
        &self,
        gpu: &dyn metrale_gpu_runtime::gpu::GpuBackend,
        input: DevicePtr,
        w: &DenseWeight,
        output: DevicePtr,
        m: usize,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<()> {
        if ops::dense_gemv_tc::try_dense_gemv_tc(gpu, input, w, output, m as u32, n, k, n, stream)?
        {
            return Ok(());
        }
        match row_dispatch::drafter_row_kernel(
            m,
            n,
            k,
            self.dense_gemv_batchm_k.0 != 0,
            row_dispatch::kv_gemv_pinned(),
            row_dispatch::small_m_tier_off(),
        ) {
            row_dispatch::RowKernel::Batchm => ops::dense_gemv_batchm(
                gpu,
                self.dense_gemv_batchm_k,
                input,
                w,
                output,
                m as u32,
                n,
                k,
                n,
                stream,
            ),
            row_dispatch::RowKernel::Pipelined => ops::dense_gemm_bf16_pipelined(
                gpu,
                self.dense_gemm_pipelined_k,
                input,
                w,
                output,
                m as u32,
                n,
                k,
                stream,
            ),
            row_dispatch::RowKernel::GemvLoop => {
                let gemv_k = self.dense_gemv_k.unwrap();
                for r in 0..m {
                    ops::dense_gemv(
                        gpu,
                        gemv_k,
                        input.offset(r * k as usize * 2),
                        w,
                        output.offset(r * n as usize * 2),
                        n,
                        k,
                        stream,
                    )?;
                }
                Ok(())
            }
        }
    }
}
