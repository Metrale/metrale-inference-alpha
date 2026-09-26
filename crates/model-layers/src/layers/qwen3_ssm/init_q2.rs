// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The packed Q2_0 fused `in_proj_qkvz`: its install and its
//! prefill GEMM.
//!
//! Owner: model-layers (qwen3_ssm).
//! Invariants: none beyond the types.

use super::*;

impl Qwen3SsmLayer {
    /// 2026-09-25: Install the packed Q2_0 fused `in_proj_qkvz`. Decode runs
    /// `q2_0_gemv_vec` on it and prefill runs `qkvz_q2_prefill_gemm`. The layer
    /// must be `sequential_qkvz`: both paths write the projection straight into
    /// the deinterleaved buffer, and decode reads `qkvz_q2` only on the
    /// sequential branch.
    ///
    /// The MMQ kernels are looked up here rather than in `new`, so a layer that
    /// never installs a packed weight issues no lookup and leaves no failed
    /// row in the boot kernel audit.
    pub fn set_packed_q2_qkvz(
        &mut self,
        qkvz: crate::weight_map::PackedQ2Weight,
        gpu: &dyn GpuBackend,
    ) {
        self.qkvz_q2 = Some(qkvz);
        self.q2_0_mmq_nc_k = super::super::try_kernel(gpu, "q2_0_mmq", "metrale_q2_0_mmq128_nc");
        self.q2_0_mmq_wc_k = super::super::try_kernel(gpu, "q2_0_mmq", "metrale_q2_0_mmq128_wc");
        self.q4k_quant_act_k =
            super::super::try_kernel(gpu, "q4k_mmq", "metrale_q8_1_quantize_ds4_bf16");
    }

    /// 2026-09-25: `out[m, n] = input[m, k] @ W^T` for the packed qkvz.
    ///
    /// With both MMQ handles resolved, `METRALE_GGUF_NATIVE_Q2_MMQ=1` and group
    /// size 128, it quantizes `input` to q8_1 into `act_q8` and runs
    /// `q2_0_mmq_gemm`. Otherwise it dequantizes the weight to BF16 into
    /// `scratch` and runs `dense_gemm_bf16_pipelined` (or `dense_gemm` when that
    /// handle is 0) on the same `stream`. Allocates nothing. Errors when no
    /// packed qkvz is installed or the dequant kernel is missing.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn qkvz_q2_prefill_gemm(
        &self,
        gpu: &dyn GpuBackend,
        input: DevicePtr,
        out: DevicePtr,
        scratch: DevicePtr,
        act_q8: DevicePtr,
        m: u32,
        stream: u64,
    ) -> Result<()> {
        let w = self
            .qkvz_q2
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("qkvz_q2_prefill_gemm: no packed qkvz installed"))?;
        let (n, k) = (w.n, w.k);

        if self.q2_0_mmq_nc_k.0 != 0
            && self.q4k_quant_act_k.0 != 0
            && crate::layers::ops::native_q2_mmq_enabled()
            && w.group == 128
        {
            crate::layers::ops::quantize_act_q8_1(
                gpu,
                self.q4k_quant_act_k,
                input,
                act_q8,
                m,
                k,
                stream,
            )?;
            return crate::layers::ops::q2_0_mmq_gemm(
                gpu,
                self.q2_0_mmq_nc_k,
                self.q2_0_mmq_wc_k,
                act_q8,
                w.weight,
                out,
                m,
                n,
                k,
                stream,
            );
        }

        if self.dequant_q2_0_gn_k.0 == 0 {
            anyhow::bail!(
                "dequant_q2_0_gn_to_bf16 kernel missing — packed-Q2 GDN prefill unavailable"
            );
        }
        crate::layers::ops::dequant_q2_0_gn_to_bf16(
            gpu,
            self.dequant_q2_0_gn_k,
            w.weight,
            scratch,
            n,
            k,
            w.group as u32,
            stream,
        )?;
        let dw = DenseWeight { weight: scratch };
        if self.dense_gemm_pipelined_k.0 != 0 {
            crate::layers::ops::dense_gemm_bf16_pipelined(
                gpu,
                self.dense_gemm_pipelined_k,
                input,
                &dw,
                out,
                m,
                n,
                k,
                stream,
            )?;
        } else {
            crate::layers::ops::dense_gemm(
                gpu,
                self.dense_gemm_k,
                input,
                &dw,
                out,
                m,
                n,
                k,
                stream,
            )?;
        }
        Ok(())
    }
}
