// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `ple_seq_state`, the PLE per-sequence carry lookup used by the
//! GDN layer's `TransformerLayer` impl.
//!
//! Owner: model-layers (qwen3_ssm).
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_gpu_runtime::gpu::GpuBackend;

use crate::layer::LayerState;

/// 2026-09-25: The PLE per-sequence carry stored in a sequence's
/// `SsmLayerState`, created on first use. Errors if the state is not an
/// `SsmLayerState`.
pub(super) fn ple_seq_state<'a>(
    ple: &crate::layers::ple::PleLayer,
    state: &'a mut dyn LayerState,
    gpu: &dyn GpuBackend,
) -> Result<&'a mut crate::layers::ple::PleSeqState> {
    let ssm = state
        .as_any_mut()
        .downcast_mut::<crate::layer::SsmLayerState>()
        .ok_or_else(|| anyhow::anyhow!("PLE host layer state is not SsmLayerState"))?;
    if ssm.ple.is_none() {
        ssm.ple = Some(ple.new_seq_state(gpu)?);
    }
    Ok(ssm.ple.as_mut().expect("just created"))
}
