// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: MoE router (`mlp.gate`) LoRA fold for [`MoeLayer`]: the one-entry
//! route build, the prefill and single-sequence decode fold
//! (`apply_router_lora_prefill`), and the batched decode fold
//! (`apply_router_lora_batched`, a `moe_lora_gather_bgmv` launch with `top_k = 1`).
//!
//! Owner: model-layers (MoE).
//! Invariants:
//! - Both folds return without launching when the layer has no router delta.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

use super::MoeLayer;
use crate::layer::MoeLoraRoute;
use crate::layers::ops;
use crate::layers::ops::lora_delta::LoraPair;
use crate::layers::ops::moe_lora_grouped::{MoeExpertRoute, pack_expert_tables};
use crate::lora::apply_router_lora;

/// 2026-09-25: The router route's only `(expert_id, a_addr, b_addr, scale)`
/// entry: expert 0, with the pair's A/B addresses and scale.
pub(super) fn router_expert_entry(rp: &LoraPair) -> (u16, u64, u64, f32) {
    (0, rp.a.weight.0, rp.b.weight.0, rp.scale)
}

impl MoeLayer {
    /// 2026-09-25: Build the one-entry route (expert 0 is the router pair) that the
    /// batched router fold passes to `moe_lora_gather_bgmv`. `k_in`, `n_out` and
    /// `max_rank` come from the pair.
    pub(super) fn build_router_route(
        rp: &LoraPair,
        gpu: &dyn GpuBackend,
    ) -> Result<MoeExpertRoute> {
        let tables = pack_expert_tables(&[router_expert_entry(rp)])
            .expect("single entry => pack_expert_tables returns Some");
        let up = |vals: &[u8]| -> Result<DevicePtr> {
            let d = gpu.alloc(vals.len())?;
            gpu.copy_h2d(vals, d)?;
            Ok(d)
        };
        let a_bytes: Vec<u8> = tables.a.iter().flat_map(|p| p.to_le_bytes()).collect();
        let b_bytes: Vec<u8> = tables.b.iter().flat_map(|p| p.to_le_bytes()).collect();
        let s_bytes: Vec<u8> = tables.scale.iter().flat_map(|s| s.to_le_bytes()).collect();
        Ok(MoeExpertRoute {
            a_table: up(&a_bytes)?,
            b_table: up(&b_bytes)?,
            scale_table: up(&s_bytes)?,
            n_experts: tables.n_experts,
            k_in: rp.k_in,
            n_out: rp.n_out,
            max_rank: rp.max_rank,
        })
    }

    /// 2026-09-25: Fold the router LoRA delta onto `gate_logits` (`n` rows) in
    /// place. Returns without launching when the layer has no router delta or the
    /// route is `Skip`, and fails on `Refuse`. The prefill paths call it after the
    /// router GEMM, and single-sequence decode calls it with `n = 1`.
    pub(crate) fn apply_router_lora_prefill(
        &self,
        router_in: DevicePtr,
        gate_logits: DevicePtr,
        n: u32,
        ctx: &crate::layer::ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let Some(ref l) = self.lora else {
            return Ok(());
        };
        let Some(ref rp) = l.router else {
            return Ok(());
        };
        if !self.moe_route_gate(ctx, "router")? {
            return Ok(());
        }
        apply_router_lora(
            ctx.gpu,
            &l.kernels,
            rp,
            router_in,
            gate_logits,
            n,
            l.cap,
            l.xa,
            l.delta,
            stream,
        )
    }

    /// 2026-09-25: Batched decode form: fold the router delta onto the whole
    /// batch's `gate_logits` in place with a one-entry gather. With a per-row
    /// `row_adapter` map the kernel skips rows whose entry is negative; without
    /// one, `moe_route_gate` decides. Fails when `fp32_gate` is set and the route
    /// is not `Skip`, or when `n` exceeds the LoRA scratch cap.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn apply_router_lora_batched(
        &self,
        router_in: DevicePtr,
        gate_logits: DevicePtr,
        n: u32,
        row_adapter: DevicePtr,
        fp32_gate: bool,
        ctx: &crate::layer::ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let Some(ref l) = self.lora else {
            return Ok(());
        };
        let Some(ref rr) = l.router_route else {
            return Ok(());
        };
        // 2026-09-25: The gather kernel folds into BF16 logits, so FP32 gate logits
        // are refused, but only when the batch folds: a `Skip` batch folds nothing.
        let folds = !matches!(ctx.moe_lora_route, MoeLoraRoute::Skip);
        anyhow::ensure!(
            !(fp32_gate && folds),
            "MoE LoRA batched router fold requires BF16 gate_logits; the FP32-gate path \
             (METRALE_FP32_GATE / fp32 routing) has no BF16-ULP oracle against the single-stream \
             router fold. Unset the FP32-gate flag to serve a router-adapted adapter \
             concurrently, or route router adapters single-stream."
        );
        // 2026-09-25: As in `apply_expert_lora_decode_down`, the host-side
        // `moe_route_gate` is consulted only without a per-row map; with one, the
        // launch is unconditional so a captured graph does not fix one route.
        if row_adapter == DevicePtr::NULL && !self.moe_route_gate(ctx, "router-decode")? {
            return Ok(());
        }
        anyhow::ensure!(
            n <= l.cap,
            "MoE LoRA batched router fold: n ({n}) exceeds LoRA scratch cap ({}); raise \
             METRALE_LORA_EXPERT_MAX_TOKENS to >= num_tokens.",
            l.cap
        );
        // 2026-09-25: `router_zero_indices` (all 0) sends every row to the route's
        // single entry. With `top_k = 1`, `row / top_k == row`, so `row_adapter[row]`
        // is the token's own entry and `x_gather = 0` reads its `router_in` row.
        ops::moe_lora_gather_bgmv(
            ctx.gpu,
            &l.kernels,
            rr,
            router_in,
            gate_logits,
            l.router_zero_indices,
            row_adapter,
            l.xa,
            n,
            1,
            0,
            stream,
        )
    }
}
