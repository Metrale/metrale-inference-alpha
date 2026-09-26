// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Pool-free allocation of a GLM-5.3 KDA layer's per-sequence state.
//!
//! The state is the SSM pool's `SsmLayerState`, not a GLM type:
//! `rollback_ssm_states_dispatch` downcasts the state of every `LinearAttention` layer to that
//! type, and every KDA layer is `LinearAttention`.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_gpu_runtime::gpu::GpuBackend;

use crate::glm5next_kda::Glm5NextKdaConfig;
use metrale_model_layers::layer::SsmLayerState;

/// 2026-09-25: Allocate a KDA layer's recurrent and conv state outside the SSM pool, FP32
/// (`h_is_f16: false`), zeroed, with no checkpoints or intermediates. The model path uses pool
/// slots instead (`Glm5NextLayer::uses_ssm_pool`).
pub fn alloc_kda_ssm_state(gpu: &dyn GpuBackend, cfg: &Glm5NextKdaConfig) -> Result<SsmLayerState> {
    let h_bytes = cfg.recurrent_state_elems() * 4;
    let conv_bytes = cfg.conv_state_elems() * 4;
    let h_state = gpu.alloc(h_bytes)?;
    let conv_state = gpu.alloc(conv_bytes)?;
    gpu.memset_async(h_state, 0, h_bytes, 0)?;
    gpu.memset_async(conv_state, 0, conv_bytes, 0)?;
    gpu.synchronize(0)?;
    Ok(SsmLayerState {
        h_state,
        conv_state,
        h_state_checkpoint: None,
        conv_state_checkpoint: None,
        h_state_intermediates: Vec::new(),
        conv_state_intermediates: Vec::new(),
        h_is_f16: false,
        h_prefill_stage: None,
        ple: None,
    })
}
