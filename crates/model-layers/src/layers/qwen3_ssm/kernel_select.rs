// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Kernel-selection helpers for [`Qwen3SsmLayer`]: the W4A16
//! batch-m GEMV tier, the deep-K tile GEMM, and the multi-seq decode
//! projection GEMM.
//!
//! Owner: model-layers (qwen3_ssm).
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use super::Qwen3SsmLayer;
use crate::layers::ops;
use crate::weight_map::QuantizedWeight;

impl Qwen3SsmLayer {
    /// 2026-09-25: The W4A16 batch-m GEMV handle for `m` rows, as chosen by
    /// `W4a16BatchmTiers::kernel`. `KernelHandle(0)` when no resolved tier
    /// serves `m`; callers check `.0 != 0` before launching.
    pub(super) fn w4a16_batchm_kernel(&self, m: usize) -> KernelHandle {
        self.w4a16_batchm.kernel(m as u32)
    }

    /// 2026-09-25: The transposed-weight tile GEMM for reduction depth `k`:
    /// `w4a16_gemm_t_k64_k` when `k >= w4a16_k64_min_k()`, `k` is a multiple
    /// of 64 and that handle resolved, else `w4a16_gemm_t_k`. The dense-FFN and
    /// attention multi-seq QKV paths read the same `w4a16_k64_min_k()`.
    pub(super) fn deep_k_gemm(&self, k: u32) -> KernelHandle {
        if k >= crate::layers::w4a16_k64_min_k()
            && k.is_multiple_of(64)
            && self.w4a16_gemm_t_k64_k.0 != 0
        {
            self.w4a16_gemm_t_k64_k
        } else {
            self.w4a16_gemm_t_k
        }
    }

    /// 2026-09-25: The NVFP4 tile GEMM for the multi-seq decode and batched
    /// verify projections (QKVZ in, `out_proj` out), with the M tile chosen by
    /// row count.
    ///
    /// `deep_k_gemm`'s launch has `ceil(m/64)` CTA rows, each reading the whole
    /// weight. `w4a16_gemm_t_m128_k` has `ceil(m/128)` rows, so it is used when
    /// the handle resolved, `m >= ssm_m128_min_m()` and `ceil(n/128)` alone is
    /// at least `sm_count` (the halved CTA count still covers every SM).
    /// Otherwise the launch is
    /// `deep_k_gemm(k)`, or the narrow-N `w4a16_gemm_t_k64_n64_k` twin when the
    /// deep-K kernel was chosen, the twin resolved and `k64_n64_wins(m, n)`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn ms_proj_gemm(
        &self,
        gpu: &dyn GpuBackend,
        input: DevicePtr,
        weight: &QuantizedWeight,
        output: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<()> {
        if let Some(min_m) = super::gdn_flags::ssm_m128_min_m()
            && m >= min_m
            && n.div_ceil(128) >= self.sm_count
            && self.w4a16_gemm_t_m128_k.0 != 0
        {
            return ops::w4a16_gemm_n128_m128(
                gpu,
                self.w4a16_gemm_t_m128_k,
                input,
                weight,
                output,
                m,
                n,
                k,
                stream,
            );
        }
        let wide = self.deep_k_gemm(k);
        if wide.0 == self.w4a16_gemm_t_k64_k.0
            && self.w4a16_gemm_t_k64_n64_k.0 != 0
            && crate::layers::k64_n64_wins(m, n)
        {
            return ops::w4a16_gemm(
                gpu,
                self.w4a16_gemm_t_k64_n64_k,
                input,
                weight,
                output,
                m,
                n,
                k,
                stream,
            );
        }
        ops::w4a16_gemm_n128(gpu, wide, input, weight, output, m, n, k, stream)
    }
}
