// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The ctx rows of `forward_block`'s non-paged path: the precompute diagnostic,
//! the debug pattern and dump of the first ctx slot, and the `fc` projection and `hidden_norm`
//! of the `eff_ctx` most recent rows.
//!
//! Owner: model-arch (DFlash drafter).
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;
use metrale_model_layers::layer::ForwardContext;
use metrale_model_layers::layers::ops;

use super::dims::BlockDims;
use crate::dflash_head::BlockDiffusionDraftHead;

impl BlockDiffusionDraftHead {
    /// 2026-09-26: The `METRALE_DFLASH_PRECOMPUTE=1` diagnostic run of `precompute_ctx_kv`.
    pub(super) fn block_precompute_diag(
        &self,
        d: &BlockDims<'_>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let BlockDims {
            levers,
            ctx_base_ptr,
            ctx_total,
            eff_ctx,
            position,
            ..
        } = *d;
        // 2026-09-25: METRALE_DFLASH_PRECOMPUTE=1 on the non-paged path also runs
        // `precompute_ctx_kv` over the same ctx rows, at positions
        // `position - eff_ctx + i`, as a diagnostic: the layers below do not read its
        // output. It writes the paged cache only with METRALE_DFLASH_PRECOMPUTE_COMMIT=1,
        // because this path does not ensure a valid block table.
        if levers.precompute
            && let Some(base) = ctx_base_ptr
            && eff_ctx > 0
        {
            let start_slot = ctx_total.saturating_sub(eff_ctx);
            let abs_start = position.saturating_sub(eff_ctx);
            let slot_positions: Vec<i32> = (0..eff_ctx).map(|i| (abs_start + i) as i32).collect();
            let dump_commit = levers.precompute_commit;
            self.precompute_ctx_kv(
                base,
                start_slot,
                eff_ctx,
                &slot_positions,
                self.scratch.slot_mapping_dev,
                ctx,
                stream,
                dump_commit,
            )?;
        }
        Ok(())
    }

    /// 2026-09-26: The first `eff_ctx` ctx slot's debug pattern and dump; returns the slot of
    /// the oldest projected ctx row.
    pub(super) fn block_ctx_seed(
        &self,
        d: &BlockDims<'_>,
        base: DevicePtr,
        dump_bf16: &impl Fn(&str, DevicePtr, usize) -> Result<()>,
    ) -> Result<usize> {
        let BlockDims {
            gpu,
            levers,
            ctx_total,
            eff_ctx,
            ctx_slot_bytes,
            ..
        } = *d;
        let start_slot = ctx_total.saturating_sub(eff_ctx);
        // 2026-09-25: METRALE_DFLASH_DEBUG_FORCE_PATTERN=1 overwrites the first
        // projected ctx slot with `0.01 * (i + 1) * (j + 1) / target_hidden_size`
        // (i = captured layer, j = column), truncated to BF16.
        if levers.force_pattern && eff_ctx > 0 {
            let n_rows = self.target_layer_ids.len();
            let n_cols = self.target_hidden_size;
            let mut bytes = Vec::with_capacity(n_rows * n_cols * 2);
            for i in 0..n_rows {
                for j in 0..n_cols {
                    let v = 0.01_f32 * ((i + 1) as f32) * ((j + 1) as f32) / (n_cols as f32);
                    // 2026-09-25: f32 to BF16 by dropping the low 16 bits.
                    let bits = v.to_bits();
                    let bf16_bits = (bits >> 16) as u16;
                    bytes.extend_from_slice(&bf16_bits.to_le_bytes());
                }
            }
            gpu.copy_h2d(&bytes, base.offset(start_slot * ctx_slot_bytes))?;
        }
        if eff_ctx > 0 {
            dump_bf16(
                "step0.input.target_hidden_stack[0]",
                base.offset(start_slot * ctx_slot_bytes),
                10,
            )?;
        }
        Ok(start_slot)
    }

    /// 2026-09-26: The `fc` projection and `hidden_norm` of the `eff_ctx` ctx rows from
    /// `start_slot`, into `scratch.fc_proj`.
    pub(super) fn block_ctx_project(
        &self,
        d: &BlockDims<'_>,
        stream: u64,
        base: DevicePtr,
        start_slot: usize,
        dump_bf16: &impl Fn(&str, DevicePtr, usize) -> Result<()>,
    ) -> Result<()> {
        let BlockDims {
            gpu,
            h,
            bf16,
            eff_ctx,
            target_hidden_dim,
            ctx_slot_bytes,
            ..
        } = *d;
        for i in 0..eff_ctx {
            let src_slot = base.offset((start_slot + i) * ctx_slot_bytes);
            let dst_slot = self.scratch.fc_proj.offset(i * self.hidden_size * bf16);
            ops::dense_gemv(
                gpu,
                self.kernels.dense_gemv,
                src_slot,
                &self.fc,
                dst_slot,
                h,
                target_hidden_dim as u32,
                stream,
            )?;
        }
        if eff_ctx > 0 {
            dump_bf16("step0.fc_proj.pre_norm[0]", self.scratch.fc_proj, 10)?;
            ops::rms_norm(
                gpu,
                self.kernels.rms_norm,
                self.scratch.fc_proj,
                &self.hidden_norm,
                self.scratch.fc_proj,
                eff_ctx as u32,
                h,
                self.rms_norm_eps,
                stream,
            )?;
            dump_bf16(
                "step0.fc_proj.post_hidden_norm[0]",
                self.scratch.fc_proj,
                10,
            )?;
        }
        Ok(())
    }

    /// 2026-09-26: The JSON the `METRALE_DFLASH_DEBUG_DUMP_FULL=1` dump writes beside the ctx
    /// slots.
    pub(super) fn dump_full_meta(&self, d: &BlockDims<'_>) -> String {
        let BlockDims {
            width,
            eff_ctx,
            last_token,
            position,
            ..
        } = *d;
        format!(
            "{{\n  \"last_token\": {},\n  \"position\": {},\n  \"eff_ctx\": {},\n  \"n_layers_captured\": {},\n  \"target_hidden_size\": {},\n  \"gamma\": {},\n  \"hidden_size\": {},\n  \"num_kv_heads\": {},\n  \"head_dim\": {},\n  \"num_drafter_layers\": {},\n  \"rope_theta\": {}\n}}\n",
            last_token,
            position,
            eff_ctx,
            self.target_layer_ids.len(),
            self.target_hidden_size,
            width,
            self.hidden_size,
            self.num_kv_heads,
            self.head_dim,
            self.num_layers,
            self.rope_theta,
        )
    }
}
