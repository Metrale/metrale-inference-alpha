// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Stages the per-stream `h_state` pointer table that the batched GDN
//! prefill kernels take (`float* const* h_state_ptrs`), once per SSM layer, for
//! `prefill_ssm_batched_layer`.
//!
//! Under `--ssm-h-dtype f16-pool` a pool slot is sized at 2 bytes per element, but
//! the batched kernels read and write FP32. [`TransformerModel::stage_h_state_ptrs`]
//! then widens each slot into the sequence's FP32 staging blob and points the table
//! there, and [`TransformerModel::narrow_h_state_stages`] writes the blobs back into
//! the slots. On an FP32-sized pool the table holds the slot pointers and the narrow
//! launches nothing.
//!
//! Owner: model-engine prefill (batched SSM).
//! Invariants: none beyond the types.

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::super::super::types::TransformerModel;
use crate::traits::SequenceState;
use metrale_model_layers::layer::SsmLayerState;

impl TransformerModel {
    /// 2026-09-25: Upload one `u64` device pointer per stream, in `seqs` order, to
    /// `scratch() + scratch_offset_bytes`, and return that address. Each entry is the
    /// stream's `h_state` for `layer_idx`, or its FP32 staging blob on an f16-sized
    /// pool. Errors when `seqs` is empty, when a stream's state for `layer_idx` is not
    /// an `SsmLayerState`, or when the f16-to-f32 kernel did not resolve.
    pub(in crate::model) fn stage_h_state_ptrs(
        &self,
        layer_idx: usize,
        seqs: &mut [&mut SequenceState],
        scratch_offset_bytes: usize,
        stream: u64,
    ) -> Result<DevicePtr> {
        let n = seqs.len();
        if n == 0 {
            anyhow::bail!("stage_h_state_ptrs called with zero streams");
        }
        let mut h_ptrs: Vec<u64> = Vec::with_capacity(n);
        for (i, seq) in seqs.iter_mut().enumerate() {
            let ssm_state = seq.layer_states[layer_idx]
                .as_any_mut()
                .downcast_mut::<SsmLayerState>()
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "stage_h_state_ptrs: stream {i} layer {layer_idx} \
                         is not an SsmLayerState (got non-SSM layer in \
                         SSM batched dispatch)"
                    )
                })?;
            // 2026-09-25: f16-sized pool: widen the slot into the FP32 staging blob and
            // point the table at the blob. FP32-sized pool (`None`): the slot pointer,
            // with no launch.
            match ssm_state.h_prefill_stage {
                None => h_ptrs.push(ssm_state.h_state.0),
                Some(stage) => {
                    if self.ssm_h_f16_to_f32_kernel.0 == 0 {
                        anyhow::bail!(
                            "--ssm-h-dtype f16-pool: ssm_h_dtype::ssm_h_state_f16_to_f32 did \
                             not resolve — refusing to point the batched GDN prefill kernels \
                             at 2-byte-sized pool slots they would write as FP32"
                        );
                    }
                    metrale_model_layers::layers::ops::ssm_h_state_f16_to_f32(
                        self.gpu.as_ref(),
                        self.ssm_h_f16_to_f32_kernel,
                        ssm_state.h_state,
                        stage,
                        (self.ssm_pool.h_bytes / 4) as u64,
                        stream,
                    )?;
                    h_ptrs.push(stage.0);
                }
            }
        }

        let dst = self.buffers.scratch().offset(scratch_offset_bytes);
        // 2026-09-25: SAFETY: the loop pushes exactly one entry per stream and every
        // early exit returns, so `h_ptrs.len() == n` and the `n * 8` bytes are all
        // initialised elements of a live `Vec<u64>`.
        let bytes = unsafe {
            std::slice::from_raw_parts(h_ptrs.as_ptr() as *const u8, n * std::mem::size_of::<u64>())
        };
        self.gpu.copy_h2d_async(bytes, dst, stream)?;
        Ok(dst)
    }

    /// 2026-09-25: Epilogue of [`Self::stage_h_state_ptrs`]: narrow each stream's FP32
    /// staging blob back into its 2-byte-sized pool slot. Returns at once on an
    /// FP32-sized pool.
    ///
    /// Call it only after a batched GDN kernel consumed the table. Widening leaves the
    /// slots unchanged, so a pass that fails before this call keeps each slot's
    /// pre-pass value. The per-request fallback in `batched_layer.rs` skips it: it
    /// goes through `prefill_gdn_full`, which widens and narrows each sequence itself.
    pub(in crate::model) fn narrow_h_state_stages(
        &self,
        layer_idx: usize,
        seqs: &mut [&mut SequenceState],
        stream: u64,
    ) -> Result<()> {
        if self.ssm_pool.h_prefill_stage_pool.is_none() {
            return Ok(());
        }
        for (i, seq) in seqs.iter_mut().enumerate() {
            let ssm_state = seq.layer_states[layer_idx]
                .as_any_mut()
                .downcast_mut::<SsmLayerState>()
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "narrow_h_state_stages: stream {i} layer {layer_idx} is not an \
                         SsmLayerState"
                    )
                })?;
            let Some(stage) = ssm_state.h_prefill_stage else {
                anyhow::bail!(
                    "narrow_h_state_stages: stream {i} layer {layer_idx} has no FP32 staging \
                     blob under an f16-sized h pool — its slot cannot hold the FP32 the \
                     batched GDN kernel just wrote"
                );
            };
            if self.ssm_h_f32_to_f16_kernel.0 == 0 {
                anyhow::bail!(
                    "--ssm-h-dtype f16-pool: ssm_h_dtype::ssm_h_state_f32_to_f16 did not \
                     resolve — the batched prefill h-state cannot be narrowed back into its \
                     2-byte-sized slot"
                );
            }
            metrale_model_layers::layers::ops::ssm_h_state_f32_to_f16(
                self.gpu.as_ref(),
                self.ssm_h_f32_to_f16_kernel,
                stage,
                ssm_state.h_state,
                (self.ssm_pool.h_bytes / 4) as u64,
                stream,
            )?;
        }
        Ok(())
    }
}
