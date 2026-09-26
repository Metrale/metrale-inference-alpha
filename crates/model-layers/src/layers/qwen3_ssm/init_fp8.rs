// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: FP8 weight-install setters and the NVFP4→FP8 `out_proj`
//! pre-dequant for [`Qwen3SsmLayer`].
//!
//! Owner: model-layers (qwen3_ssm).
//! Invariants:
//! - `set_fp8_decode_weights` panics unless each weight is tagged
//!   `Fp8BlockScaled`, and `set_fp8_rowwise_prefill_weights` unless each is
//!   tagged `Fp8PerRow`, so neither FP8 layout reaches the other's readers.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

use super::Qwen3SsmLayer;
use crate::weight_map::Fp8Weight;

impl Qwen3SsmLayer {
    /// 2026-09-25: Install block-scaled FP8 weights as `qkvz_fp8w` /
    /// `out_proj_fp8w`. Panics unless each is tagged `Fp8BlockScaled`: their
    /// readers include `w8a16_gemv`, which computes
    /// `sum_k A[k] * E4M3_LUT[B[n,k]] * block_scale[n/128, k/128]` with FP32
    /// scales (`kernels/gb10/common/w8a16_gemv.cu`).
    ///
    /// It does not touch `qkvz_fp8` / `out_proj_fp8`: those feed
    /// `fp8_gemm_n128`, which takes no scale argument, so block-scaled bytes
    /// there would be read unscaled. `set_fp8_prefill_only_weights` installs
    /// them.
    pub fn set_fp8_decode_weights(&mut self, qkvz: Option<Fp8Weight>, out_proj: Option<Fp8Weight>) {
        if let Some(ref w) = qkvz {
            w.scale_format.expect(
                crate::weight_map::WeightQuantFormat::Fp8BlockScaled,
                "set_fp8_decode_weights::qkvz (w8a16_gemv expects [N/BS,K/BS] BF16 block scales)",
            );
        }
        if let Some(ref w) = out_proj {
            w.scale_format.expect(
                crate::weight_map::WeightQuantFormat::Fp8BlockScaled,
                "set_fp8_decode_weights::out_proj (w8a16_gemv expects [N/BS,K/BS] BF16 block scales)",
            );
        }
        self.qkvz_fp8w = qkvz;
        self.out_proj_fp8w = out_proj;
    }

    /// 2026-09-25: Install per-row FP8 weights as `qkvz_fp8w_rowwise` /
    /// `out_proj_fp8w_rowwise`. Panics unless each is tagged `Fp8PerRow`. Only
    /// the row-wise prefill arms read them (`trait_prefill_proj.rs`,
    /// `trait_prefill_helper.rs`); decode never does.
    pub fn set_fp8_rowwise_prefill_weights(
        &mut self,
        qkvz: Option<Fp8Weight>,
        out_proj: Option<Fp8Weight>,
    ) {
        for (w, what) in [(&qkvz, "qkvz"), (&out_proj, "out_proj")] {
            if let Some(w) = w {
                w.scale_format.expect(
                    crate::weight_map::WeightQuantFormat::Fp8PerRow,
                    "set_fp8_rowwise_prefill_weights (cuBLASLt row-wise expects [N] f32)",
                );
                let _ = what;
            }
        }
        self.qkvz_fp8w_rowwise = qkvz;
        self.out_proj_fp8w_rowwise = out_proj;
    }

    /// 2026-09-25: Install unscaled FP8 weights as `qkvz_fp8` / `out_proj_fp8`,
    /// the operands of the `fp8_gemm_n128` arms in prefill and in the batched
    /// decode/verify projections (`trait_decode_batched.rs`). A `None`
    /// argument leaves that field as it was.
    pub fn set_fp8_prefill_only_weights(
        &mut self,
        qkvz_fp8: Option<DevicePtr>,
        out_proj_fp8: Option<DevicePtr>,
    ) {
        if qkvz_fp8.is_some() {
            self.qkvz_fp8 = qkvz_fp8;
        }
        if out_proj_fp8.is_some() {
            self.out_proj_fp8 = out_proj_fp8;
        }
    }

    /// 2026-09-25: When the layer holds `out_proj_nvfp4_t`, dequantize the NVFP4
    /// `out_proj` to FP8 into a new allocation and store it as `out_proj_fp8`.
    /// QKVZ is not converted. Synchronizes `stream`.
    pub fn predequant_for_prefill(
        &mut self,
        gpu: &dyn GpuBackend,
        config: &metrale_config::ModelConfig,
        stream: u64,
    ) -> Result<()> {
        let predequant_k = gpu.kernel("w4a16", "predequant_nvfp4_to_fp8")?;
        let h = config.hidden_size;
        let qkvz_size = config.ssm_qkvz_size();
        let value_dim = config.linear_num_value_heads * config.linear_value_head_dim;

        let _ = qkvz_size;
        // 2026-09-25: `predequant_nvfp4_to_fp8` reads a packed `[N, K/2]` weight,
        // so this passes the untransposed `ssm.out_proj`.
        if self.out_proj_nvfp4_t.is_some() {
            self.out_proj_fp8 = Some(self.ssm.out_proj.predequant_to_fp8(
                gpu,
                predequant_k,
                h,
                value_dim,
                stream,
            )?);
        }
        Ok(())
    }
}
