// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `Qwen3SsmLayer::new_sequential`, the constructor for
//! checkpoints whose fused QKVZ weight is already in `[Q|K|V|Z]` row order.
//!
//! Owner: model-layers (qwen3_ssm).
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_gpu_runtime::gpu::GpuBackend;

use super::super::Qwen3SsmLayer;
use crate::layers::FfnComponent;
use crate::weight_map::{DenseWeight, QuantizedWeight, SsmWeights};

impl Qwen3SsmLayer {
    /// 2026-09-25: `new` plus `sequential_qkvz = true` and the two transposed
    /// NVFP4 prefill weights. With `sequential_qkvz` the QKVZ projection writes
    /// straight into the deinterleaved buffer and `deinterleave_qkvz` is not
    /// launched. Called by the `qwen35` and `qwen35_dense` weight loaders.
    pub fn new_sequential(
        input_norm: DenseWeight,
        ssm: SsmWeights,
        post_attn_norm: DenseWeight,
        ffn: FfnComponent,
        qkvz_nvfp4: Option<QuantizedWeight>,
        qkvz_nvfp4_t: Option<QuantizedWeight>,
        out_proj_nvfp4_t: Option<QuantizedWeight>,
        config: &metrale_config::ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<Self> {
        let mut layer = Self::new(
            input_norm,
            ssm,
            post_attn_norm,
            ffn,
            qkvz_nvfp4,
            config,
            gpu,
        )?;
        layer.sequential_qkvz = true;
        layer.qkvz_nvfp4_t = qkvz_nvfp4_t;
        layer.out_proj_nvfp4_t = out_proj_nvfp4_t;
        Ok(layer)
    }
}
