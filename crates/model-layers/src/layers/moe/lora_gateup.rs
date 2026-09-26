// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: MoE LoRA gate/up-projection folds, for prefill and decode.
//!
//! Owner: model-layers (MoE).
//! Invariants:
//! - Both hooks call the down-fold launchers (`moe_lora_grouped_down`,
//!   `moe_lora_gather_bgmv`) with `x_gather = 1`, so the shrink reads the
//!   token's `expert_input` row: `sorted_token_ids[r]` in prefill, `row / top_k`
//!   in decode.
//! - Both return without launching when no gate/up pair is installed. Without
//!   a per-row adapter map they also return on route `Skip` and fail on `Refuse`.

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::MoeLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;

impl MoeLayer {
    /// 2026-09-25: Fold the routed-expert gate/up LoRA deltas in place onto the
    /// sorted `expert_gate_out`/`expert_up_out` (`te` rows). The prefill callers
    /// run it before `silu_mul` overwrites `expert_gate_out`. `expert_input` is
    /// token-major; the shrink kernel gathers each sorted row's token.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn apply_expert_lora_prefill_gateup(
        &self,
        expert_gate_out: DevicePtr,
        expert_up_out: DevicePtr,
        expert_input: DevicePtr,
        expert_offsets: DevicePtr,
        sorted_token_ids: DevicePtr,
        te: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let Some(ref l) = self.lora else {
            return Ok(());
        };
        if l.gate_route.is_none() && l.up_route.is_none() {
            return Ok(());
        }
        if !self.moe_route_gate(ctx, "expert-gateup")? {
            return Ok(());
        }
        // 2026-09-25: Windows of at most `l.cap` rows, because `l.xa` is
        // `[cap, max_rank]`. Gate and up share `l.xa`: both launch in order on one
        // stream, so gate's expand finishes before up's shrink writes it.
        for (off, end) in ops::grouped_down_windows(te, l.cap) {
            if let Some(ref gate) = l.gate_route {
                ops::moe_lora_grouped_down(
                    ctx.gpu,
                    &l.kernels,
                    gate,
                    expert_input,
                    expert_gate_out,
                    expert_offsets,
                    sorted_token_ids,
                    DevicePtr::NULL,
                    l.xa,
                    off,
                    end,
                    1,
                    stream,
                )?;
            }
            if let Some(ref up) = l.up_route {
                ops::moe_lora_grouped_down(
                    ctx.gpu,
                    &l.kernels,
                    up,
                    expert_input,
                    expert_up_out,
                    expert_offsets,
                    sorted_token_ids,
                    DevicePtr::NULL,
                    l.xa,
                    off,
                    end,
                    1,
                    stream,
                )?;
            }
        }
        Ok(())
    }

    /// 2026-09-25: Decode form: fold the gate/up deltas in place onto the
    /// slot-major `expert_gate_out`/`expert_up_out` (`n_slots` rows). The decode
    /// callers run it before the silu+down kernel reads them. Slot `row` reads
    /// `expert_input` row `row / top_k`; `indices_dev` holds each slot's expert id.
    /// Fails when `n_slots` exceeds the LoRA scratch cap.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn apply_expert_lora_decode_gateup(
        &self,
        expert_gate_out: DevicePtr,
        expert_up_out: DevicePtr,
        expert_input: DevicePtr,
        indices_dev: DevicePtr,
        n_slots: u32,
        top_k: u32,
        row_adapter: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let Some(ref l) = self.lora else {
            return Ok(());
        };
        if l.gate_route.is_none() && l.up_route.is_none() {
            return Ok(());
        }
        // 2026-09-25: With a per-row `row_adapter` map the fold always launches and
        // the kernel skips rows whose entry is negative; without one,
        // `moe_route_gate` decides.
        if row_adapter == DevicePtr::NULL && !self.moe_route_gate(ctx, "expert-gateup-decode")? {
            return Ok(());
        }
        anyhow::ensure!(
            n_slots <= l.cap,
            "MoE expert LoRA decode gate/up-fold: n_slots ({n_slots}) exceeds LoRA scratch cap \
             ({}); raise METRALE_LORA_EXPERT_MAX_TOKENS to >= num_tokens*top_k.",
            l.cap
        );
        if let Some(ref gate) = l.gate_route {
            ops::moe_lora_gather_bgmv(
                ctx.gpu,
                &l.kernels,
                gate,
                expert_input,
                expert_gate_out,
                indices_dev,
                row_adapter,
                l.xa,
                n_slots,
                top_k,
                1,
                stream,
            )?;
        }
        if let Some(ref up) = l.up_route {
            ops::moe_lora_gather_bgmv(
                ctx.gpu,
                &l.kernels,
                up,
                expert_input,
                expert_up_out,
                indices_dev,
                row_adapter,
                l.xa,
                n_slots,
                top_k,
                1,
                stream,
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "lora_gateup_tests.rs"]
mod tests;
