// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `TransformerModel::decode_forward_body`: the single-token decode
//! forward (each layer's decode, periodic Mamba-2 state normalisation, final RMS
//! norm, LM head), run eagerly or inside a CUDA-graph capture.
//!
//! Owner: model-engine (decode).
//! Invariants: none beyond the types.

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use anyhow::Result;
use metrale_cache::kv_cache::PagedKvCache;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

use super::super::types::TransformerModel;
use crate::traits::ModelForward;
use crate::traits::{Model, SequenceState};
use metrale_model_layers::layer::{ForwardContext, TransformerLayer};
use metrale_model_layers::layers::ops;

impl TransformerModel {
    /// 2026-09-25: Single-token decode forward body. `decode_dispatch_with` runs it
    /// once per step, eagerly or inside a capture, and runs it again eagerly when
    /// the capture fails (capture records without executing, so the step still
    /// runs once).
    pub(super) fn decode_forward_body(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        seq: &mut SequenceState,
        kv_cache: &mut PagedKvCache,
        ctx: &ForwardContext,
        probe_layers: bool,
        use_graphs: bool,
        stream: u64,
    ) -> Result<()> {
        for (i, layer) in self.layers.iter().enumerate() {
            layer.decode(
                hidden,
                residual,
                seq.layer_states[i].as_mut(),
                kv_cache,
                seq.seq_len,
                &mut seq.block_table,
                &mut seq.disk_block_ids,
                &mut seq.disk_last_offloaded_per_layer,
                ctx,
                stream,
            )?;
            // 2026-09-25: With `probe_layers` (first decode step, eager,
            // `ssm_save_dump`), log each layer's post-layer sum of |hidden|, to
            // find the first layer that diverges between two runs.
            if probe_layers {
                self.gpu.synchronize(stream).ok();
                let mut hb = vec![0u8; self.config.hidden_size * 2];
                if self.gpu.copy_d2h(hidden, &mut hb).is_ok() {
                    let mut s = 0f64;
                    for c in hb.chunks_exact(2) {
                        let bits = u16::from_le_bytes([c[0], c[1]]);
                        let v = f32::from_bits((bits as u32) << 16) as f64;
                        if v.is_finite() {
                            s += v.abs();
                        }
                    }
                    tracing::warn!("METRALE_LAYER_H[step0] L{i} hidden_sabs={s:.6}");
                }
            }
            // 2026-09-25: DFlash hidden capture: copies row 0 of `hidden_states()`
            // when layer `i` is a capture layer; no-op without DFlash.
            self.try_dflash_capture(i, 0, stream)?;
        }
        // 2026-09-25: MLA (`kv_lora_rank > 0`): synchronise before the final norm,
        // eagerly only, since a synchronise is illegal during capture. Kernels on
        // one stream are already ordered, so capture loses nothing without it.
        if self.config.kv_lora_rank > 0 && !use_graphs {
            self.gpu.synchronize(stream)?;
        }

        // 2026-09-25: Mamba-2 models (`mamba_num_heads > 0`): normalise the SSM state
        // every 64 tokens. A failure is logged and decode continues.
        if self.config.mamba_num_heads > 0
            && seq.seq_len > 0
            && seq.seq_len.is_multiple_of(64)
            && let Err(e) = self.normalize_ssm_states(seq, stream)
        {
            tracing::warn!("Periodic SSM state normalization failed: {e:#}");
        }

        let normed = self.buffers.norm_output();
        let h = self.config.hidden_size as u32;
        let eps = self.config.rms_norm_eps as f32;
        self.final_norm_apply(hidden, normed, 1, h, eps, stream)?;

        self.lm_head(normed, stream)?;
        Ok(())
    }
}
