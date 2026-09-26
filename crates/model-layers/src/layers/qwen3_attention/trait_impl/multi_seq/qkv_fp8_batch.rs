// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Batched native-FP8 (block-scaled W8A16) Q/K/V projections for multi-sequence decode: one launch per projection for all `n` rows.
//!
//! Owner: model-layers (qwen3 attention, multi-sequence decode).
//! Invariants:
//! - Every arm writes Q, K and V at the per-row offsets the per-sequence loop in `qkv.rs` uses: Q at 0, K at `q_proj_bytes`, V after K, rows `per_seq_qkv` bytes apart.
//! - The `batch4`/`batch16` strided arms and the N-column arm are bit-identical per row to the scalar `w8a16_gemv`. The `m16` MMA arm, the M32 tile arm and the W8A8 arm are not.

use anyhow::Result;

use super::ctx::MultiSeqCtx;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use crate::layers::ops;
use crate::layers::qwen3_attention::Qwen3AttentionLayer;
use crate::weight_map::{Fp8Weight, WeightQuantFormat};

/// 2026-09-25: The signature every strided arm in `ms_qkv_batchm_fp8_gemv`
/// shares, so the arm choice yields one function pointer.
type StridedBatchGemv = fn(
    &dyn GpuBackend,
    KernelHandle,
    DevicePtr,
    DevicePtr,
    DevicePtr,
    DevicePtr,
    u32,
    u32,
    u32,
    u32,
    u32,
    u64,
) -> Result<()>;

/// 2026-09-25: Kill switch for this tier: `METRALE_NO_FP8_QKV_BATCH` set to any
/// value keeps the per-sequence scalar loop. Read once per process, so the
/// choice cannot change between a CUDA-graph capture and its replays.
pub(super) fn fp8_batchm_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("METRALE_NO_FP8_QKV_BATCH").is_none())
}

/// 2026-09-25: MAX_M of `w8a16_gemv_batch16_strided` (its wrapper refuses `m`
/// outside `1..=16`): the widest batch the GEMV arms serve. Wider batches take
/// the 32-row M-tile arm.
const FP8_QKV_GEMV_MAX_ROWS: usize = 16;

impl Qwen3AttentionLayer {
    /// 2026-09-25: Whether the batched FP8 tier serves this step. `enabled` is the
    /// kill switch, passed in so tests can drive both arms without the
    /// process-global `OnceLock` in [`fp8_batchm_enabled`].
    ///
    /// Branch only on `c.n`, the row count the caller launched with. On the
    /// batched decode path that is `padded_n`, the width the decode graph cache
    /// is keyed by. The row band is [`Self::fp8_qkv_batch_band_admits`].
    pub(super) fn ms_qkv_batchm_fp8_selected(&self, c: &MultiSeqCtx<'_>, enabled: bool) -> bool {
        if !enabled || !self.fp8_qkv_batch_band_admits(c.n) {
            return false;
        }
        if self.w8a16_gemv_batch4_strided_k.0 == 0 || self.w8a16_gemv_batch16_strided_k.0 == 0 {
            return false;
        }
        let Some((q, k, v)) = self.qkv_fp8_block_scaled() else {
            return false;
        };
        // 2026-09-25: Whole 128-wide blocks on every dimension, the granularity
        // of the `[N/128, K/128]` block-scale grid.
        let kv_dim = c.nkv * c.hd;
        let dims_ok = c.h.is_multiple_of(128)
            && c.q_proj_dim.is_multiple_of(128)
            && kv_dim.is_multiple_of(128);
        // 2026-09-25: Row pitches must be whole BF16 elements, and A rows 16-byte
        // aligned (the kernel loads activations as `uint4`); `normed` rows are
        // `h` elements apart.
        let strides_ok = c.per_seq_qkv.is_multiple_of(c.bf16) && c.h.is_multiple_of(8);
        let shapes_ok = q.n == c.q_proj_dim && k.n == kv_dim && v.n == kv_dim;
        dims_ok && strides_ok && shapes_ok
    }

    /// 2026-09-25: The row band of the batched tier. `ms_qkv_batchm_fp8_gemv`
    /// splits it at the same `FP8_QKV_GEMV_MAX_ROWS`.
    ///
    /// * `2..=FP8_QKV_GEMV_MAX_ROWS`: the strided GEMV arms.
    /// * wider: `w8a16_gemm_pipelined_m32_strided`, whose `grid.y` tiles M, so the
    ///   band has no upper edge of its own. Only when that handle is non-zero
    ///   (it is zero unless `ModelLevers::fp8_attn_m32` is on); otherwise those
    ///   widths keep the per-sequence loop.
    pub(super) fn fp8_qkv_batch_band_admits(&self, n: usize) -> bool {
        n >= 2 && (n <= FP8_QKV_GEMV_MAX_ROWS || self.w8a16_gemm_pipelined_m32_k.0 != 0)
    }

    /// 2026-09-25: q, k and v, when all three are native FP8 with 2D block scales
    /// (`Fp8BlockScaled`), the only scale layout the `w8a16` kernels index.
    fn qkv_fp8_block_scaled(&self) -> Option<(&Fp8Weight, &Fp8Weight, &Fp8Weight)> {
        // 2026-09-25: A free fn, not a closure: closure lifetime inference cannot
        // tie the borrow of `w` to the returned reference here.
        fn block_scaled(w: &Option<crate::weight_map::QuantWeight>) -> Option<&Fp8Weight> {
            w.as_ref()
                .and_then(|w| w.as_fp8())
                .filter(|w| w.scale_format == WeightQuantFormat::Fp8BlockScaled)
        }
        Some((
            block_scaled(&self.q_weight)?,
            block_scaled(&self.k_weight)?,
            block_scaled(&self.v_weight)?,
        ))
    }

    /// 2026-09-25: One launch per projection for all `n` rows, written into
    /// `qkv_buf` at the per-row offsets the per-sequence loop uses (Q at 0, K at
    /// `q_proj_bytes`, V after K).
    ///
    /// - Gated Q is deinterleaved here by one strided `deinterleave_qg` launch,
    ///   unless a q adapter is resident; then `ms_qkv_deinterleave_q` does it
    ///   after the LoRA fold.
    /// - LoRA and the q/k RMS norms are not applied here: `ms_phase_qkv` runs
    ///   `ms_qkv_apply_lora` and `ms_qkv_norms` after every projection branch.
    pub(super) fn ms_qkv_batchm_fp8(&self, c: &MultiSeqCtx<'_>) -> Result<()> {
        let MultiSeqCtx {
            fwd,
            n,
            stream,
            h,
            nq,
            nkv,
            hd,
            bf16,
            per_seq_qkv,
            qkv_buf,
            ..
        } = *c;
        let (q, k, v) = self
            .qkv_fp8_block_scaled()
            .expect("ms_qkv_batchm_fp8 entered without block-scaled FP8 q/k/v");

        // 2026-09-25: `normed` is contiguous `[n, h]`. `qkv_buf` rows are
        // `per_seq_qkv` bytes apart; the kernels take the pitch in BF16 elements.
        debug_assert_eq!(per_seq_qkv % bf16, 0);
        let a_stride = h as u32;
        let c_stride = (per_seq_qkv / bf16) as u32;
        let kv_dim = nkv * hd;

        // 2026-09-25: The W8A8 cuBLASLt arm (`w8a8_decode.rs`) first; when it
        // declines (`Ok(false)`), the GEMV arms. The gated deinterleave below
        // runs after either.
        if !self.try_ms_qkv_decode_w8a8(c, q, k, v, kv_dim)? {
            self.ms_qkv_batchm_fp8_gemv(c, q, k, v, kv_dim, a_stride, c_stride)?;
        }

        if self.gated && !self.q_lora_active() {
            ops::deinterleave_qg(
                fwd.gpu,
                self.deinterleave_qg_k,
                qkv_buf,
                n as u32,
                nq,
                hd,
                c_stride,
                stream,
            )?;
        }
        Ok(())
    }

    /// 2026-09-25: One strided launch per projection. By row count: `batch4`
    /// up to 4 rows, the M32 tile above `FP8_QKV_GEMV_MAX_ROWS`, and in between
    /// the `m16` MMA arm when `self.m16_tc`, else the N-column arm when
    /// configured, else `batch16`.
    #[allow(clippy::too_many_arguments)]
    fn ms_qkv_batchm_fp8_gemv(
        &self,
        c: &MultiSeqCtx<'_>,
        q: &Fp8Weight,
        k: &Fp8Weight,
        v: &Fp8Weight,
        kv_dim: u32,
        a_stride: u32,
        c_stride: u32,
    ) -> Result<()> {
        let MultiSeqCtx {
            fwd,
            n,
            stream,
            h,
            bf16,
            q_proj_dim,
            q_proj_bytes,
            qkv_buf,
            normed,
            ..
        } = *c;
        let kv_bytes = kv_dim as usize * bf16;

        // 2026-09-25: `tc` is the `w8a16_gemm_m16_strided` MMA arm (at most 16
        // rows). It reassociates the K reduction, so it is not bit-identical to
        // the GEMV arms. `self.m16_tc` is the resolved `attn_m16_tc` setting: the
        // target's `HARDWARE.toml` `[defaults]` row, overridden by
        // `METRALE_ATTN_M16_TC`, else by `METRALE_M16_TC`.
        let tc = self.m16_tc && self.w8a16_gemm_m16_strided_k.0 != 0 && h.is_multiple_of(128);
        let (launch, kernel): (StridedBatchGemv, KernelHandle) = if n <= 4 {
            (
                ops::w8a16_gemv_batch4_strided,
                self.w8a16_gemv_batch4_strided_k,
            )
        } else if n > FP8_QKV_GEMV_MAX_ROWS {
            // 2026-09-25: Above 16 rows, `w8a16_gemm_pipelined_m32_strided`: one
            // launch per projection with `grid.y = ceil(n/32)`. Not bit-identical
            // to the GEMV arms (its oracle grades it on a tolerance). It comes
            // before the `tc` and N-column arms, which serve at most 16 rows.
            (
                ops::w8a16_gemm_pipelined_m32_strided,
                self.w8a16_gemm_pipelined_m32_k,
            )
        } else if tc {
            crate::layers::qwen3_attention::attn_m16_tc_route::log_qkv_m16_tc_route(fwd.stats);
            (ops::w8a16_gemm_m16_strided, self.w8a16_gemm_m16_strided_k)
        } else if let Some(route) = self.ncol_strided_route(n) {
            // 2026-09-25: N-column-blocked GEMV (`attn_ncol_gemv.rs`): one thread
            // owns N_COLS adjacent output columns and keeps the scalar kernel's
            // per-row reduction order, so it stays bit-exact. It sits below the
            // `tc` arm, so `m16_tc` wins when both are on.
            route
        } else {
            (
                ops::w8a16_gemv_batch16_strided,
                self.w8a16_gemv_batch16_strided_k,
            )
        };
        let gemv = |w: &Fp8Weight, out: DevicePtr, n_out: u32| {
            launch(
                fwd.gpu,
                kernel,
                normed,
                w.weight,
                w.row_scale,
                out,
                n as u32,
                n_out,
                h as u32,
                a_stride,
                c_stride,
                stream,
            )
        };

        // 2026-09-25: `q_proj_dim` is `2 * q_dim` when gated (`[Q|gate]`) and
        // `q_dim` otherwise, so this one width serves both.
        gemv(q, qkv_buf, q_proj_dim)?;
        gemv(k, qkv_buf.offset(q_proj_bytes), kv_dim)?;
        gemv(v, qkv_buf.offset(q_proj_bytes + kv_bytes), kv_dim)?;
        Ok(())
    }
}

#[cfg(test)]
#[path = "qkv_fp8_batch_tests.rs"]
mod tests;
