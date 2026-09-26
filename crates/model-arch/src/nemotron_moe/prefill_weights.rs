// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `NemotronMoeLayer` prefill-side weight preparation (the
//! transposed-NVFP4 expert and shared-expert copies, and the FP8 shared-expert
//! and latent-projection copies) and the dense-GEMM prefill dispatcher.
//!
//! Owner: model-arch (Nemotron-H).
//! Invariants:
//! - `prepare_prefill_weights` never fails: a copy that cannot be built
//!   leaves its field `None` and prefill uses the next arm.

use anyhow::Result;
use metrale_config::ModelConfig;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

use super::NemotronMoeLayer;
use super::build_ptr_table_from_weights;
use metrale_model_layers::layers::ops;
use metrale_model_layers::weight_map::DenseWeight;

impl NemotronMoeLayer {
    /// 2026-09-25: Dense BF16 GEMM for prefill: `dense_gemm_bf16_pipelined`
    /// when it resolved, else `dense_gemm_bf16`. Used for the gate GEMM and,
    /// without FP8 copies, the fc1 / fc2 latent GEMMs.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn dense_gemm_prefill(
        &self,
        gpu: &dyn GpuBackend,
        input: DevicePtr,
        weight: &DenseWeight,
        output: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<()> {
        if self.dense_gemm_pipelined_k.0 != 0 {
            ops::dense_gemm_bf16_pipelined(
                gpu,
                self.dense_gemm_pipelined_k,
                input,
                weight,
                output,
                m,
                n,
                k,
                stream,
            )
        } else {
            ops::dense_gemm(
                gpu,
                self.dense_gemm_k,
                input,
                weight,
                output,
                m,
                n,
                k,
                stream,
            )
        }
    }

    /// 2026-09-25: Build the prefill-only weight copies; the loader calls it
    /// after construction. Routed experts are transposed only when
    /// `moe_latent_size == 0`.
    pub fn prepare_prefill_weights(&mut self, gpu: &dyn GpuBackend, config: &ModelConfig) {
        let h = config.hidden_size;
        let inter = self.moe_inter;
        let shared_inter = config.shared_expert_intermediate_size;

        // 2026-09-25: A table is installed only when every expert transposed;
        // without it the sorted path uses the base grouped GEMM.
        if self.moe_latent_size == 0 {
            let expert_k = h;
            let mut up_t = Vec::new();
            let mut down_t = Vec::new();
            for expert in &self.weights.experts {
                if let Ok(ut) = expert.up_proj.transpose_for_gemm(gpu, inter, expert_k) {
                    up_t.push(ut);
                }
                if let Ok(dt) = expert.down_proj.transpose_for_gemm(gpu, expert_k, inter) {
                    down_t.push(dt);
                }
            }
            if up_t.len() == self.weights.experts.len()
                && let Ok(ptrs) = build_ptr_table_from_weights(&up_t, gpu)
            {
                self.up_ptrs_t = Some(ptrs);
            }
            if down_t.len() == self.weights.experts.len()
                && let Ok(ptrs) = build_ptr_table_from_weights(&down_t, gpu)
            {
                self.down_ptrs_t = Some(ptrs);
            }
        }

        // 2026-09-25: The shared expert's copies, in every layer shape: with
        // `METRALE_SHARED_FP8_PREFILL` set (any value) and
        // `fp8_gemm_t_m128_mfast` resolved, pre-dequantized FP8 copies;
        // otherwise, or if either failed, transposed NVFP4 copies. Under native
        // FP8 the NVFP4 shared weights are `QuantizedWeight::null()` and every
        // copy here is derived from them, so none is built.
        let native_shared =
            self.weights.shared_up_fp8.is_some() || self.weights.shared_down_fp8.is_some();
        let fp8_prefill = !native_shared && std::env::var("METRALE_SHARED_FP8_PREFILL").is_ok();
        if fp8_prefill
            && self.fp8_gemm_m128_k.0 != 0
            && let Ok(pdq_k) = gpu.kernel("w4a16", "predequant_nvfp4_to_fp8")
        {
            self.shared_up_pd_fp8 = self
                .weights
                .shared_up
                .predequant_to_fp8(gpu, pdq_k, shared_inter, h, 0)
                .ok();
            self.shared_down_pd_fp8 = self
                .weights
                .shared_down
                .predequant_to_fp8(gpu, pdq_k, h, shared_inter, 0)
                .ok();
        }
        if !native_shared && (self.shared_up_pd_fp8.is_none() || self.shared_down_pd_fp8.is_none())
        {
            self.shared_up_t = self
                .weights
                .shared_up
                .transpose_for_gemm(gpu, shared_inter, h)
                .ok();
            self.shared_down_t = self
                .weights
                .shared_down
                .transpose_for_gemm(gpu, h, shared_inter)
                .ok();
        }

        // 2026-09-25: Under the same `fp8_prefill` gate, FP8 E4M3 copies of the
        // BF16 fc1 / fc2 latent projections for `fp8_gemm_t_m128_mfast`.
        let lat = self.moe_latent_size;
        if lat > 0
            && fp8_prefill
            && self.fp8_gemm_m128_k.0 != 0
            && let Ok(b2f) = gpu.kernel("w4a16", "bf16_to_fp8")
        {
            let conv = |w: &DenseWeight, n: usize, k: usize| -> Option<DevicePtr> {
                let dst = gpu.alloc(n * k).ok()?;
                metrale_model_layers::layers::ops::bf16_to_fp8(
                    gpu,
                    b2f,
                    w.weight,
                    dst,
                    (n * k) as u32,
                    0,
                )
                .ok()?;
                gpu.synchronize(0).ok()?;
                Some(dst)
            };
            self.fc1_pd_fp8 = self
                .weights
                .fc1_latent_proj
                .as_ref()
                .and_then(|w| conv(w, lat, h));
            self.fc2_pd_fp8 = self
                .weights
                .fc2_latent_proj
                .as_ref()
                .and_then(|w| conv(w, h, lat));
        }
    }
}
