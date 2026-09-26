// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: MoE LoRA on `MoeLayer`: the installed router and routed-expert
//! deltas with their apply scratch, the per-request fold gate, the routed-expert
//! down-projection folds (prefill and decode), and the decode refusal. A fold adds
//! a BF16 delta onto a base output buffer; the base GEMMs are unchanged.
//!
//! Owner: model-layers (MoE).
//! Invariants:
//! - With `self.lora == None` the fold hooks in this file return `Ok` without
//!   launching, and `reject_decode_lora` does not refuse.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

use super::MoeLayer;
use crate::layer::{ForwardContext, MoeLoraRoute};
use crate::layers::ops;
use crate::layers::ops::lora_delta::{LoraKernels, LoraPair};
use crate::layers::ops::moe_lora_grouped::{MoeExpertRoute, pack_expert_tables};
use crate::lora::{ExpertLoraLayer, ExpertProj};

/// 2026-09-25: Row capacity of the LoRA apply scratch: METRALE_LORA_EXPERT_MAX_TOKENS
/// when it parses as a positive integer, else 4096; read once per process.
/// Prefill expert folds run in windows of this many rows; a decode fold over
/// more rows errors.
fn max_tokens() -> u32 {
    static V: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("METRALE_LORA_EXPERT_MAX_TOKENS")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&t: &u32| t > 0)
            .unwrap_or(4096)
    })
}

/// 2026-09-25: Column count of the `delta` scratch. Its two readers are the
/// router fold's expand (`router_n_out` = num_experts columns) and the decode
/// down fold's activation recompute (`down_k_in` = moe_inter columns); the
/// gate/up folds do not use it. At least 1, so the allocation is never empty.
pub(super) fn lora_delta_cols(router_n_out: Option<u32>, down_k_in: Option<u32>) -> usize {
    let r = router_n_out.map_or(0, |v| v as usize);
    let d = down_k_in.map_or(0, |v| v as usize);
    r.max(d).max(1)
}

/// 2026-09-25: One MoE layer's installed router and routed-expert LoRA, with apply scratch.
pub(crate) struct MoeLoraWeights {
    /// 2026-09-25: Router (`mlp.gate`) delta on the routing logits; `None` if unadapted.
    pub(super) router: Option<LoraPair>,
    pub(super) kernels: LoraKernels,
    /// 2026-09-25: Scratch capacity in rows (`max_tokens()`).
    pub(super) cap: u32,
    /// 2026-09-25: `[cap, max_rank]` BF16 shrink scratch, indexed by the row within
    /// the current window (prefill) or by flat slot (decode). The gate, up and
    /// down folds reuse it in turn on one stream; each shrink precedes its expand.
    pub(super) xa: DevicePtr,
    /// 2026-09-25: `[cap, lora_delta_cols(..)]` BF16 scratch for the router fold's
    /// expand and the decode down fold's activation recompute.
    pub(super) delta: DevicePtr,
    /// 2026-09-25: Device route tables for the expert down_proj fold (per-expert
    /// A/B/scale; `k_in` = moe_inter, `n_out` = hidden). `None` without `Down` pairs.
    expert_route: Option<MoeExpertRoute>,
    /// 2026-09-25: Expert gate_proj fold route (`k_in` = hidden, `n_out` = moe_inter).
    /// `None` without `Gate` pairs.
    pub(super) gate_route: Option<MoeExpertRoute>,
    /// 2026-09-25: Expert up_proj fold route (dims as gate). `None` without `Up` pairs.
    pub(super) up_route: Option<MoeExpertRoute>,
    /// 2026-09-25: The router pair as a one-entry expert route (expert 0), so the
    /// batched decode router fold runs on `moe_lora_gather_bgmv`. `None` exactly
    /// when `router` is `None`.
    pub(super) router_route: Option<MoeExpertRoute>,
    /// 2026-09-25: `[cap]` u32 zeros: the router gather fold's `indices`, sending every
    /// row to entry 0 (the router pair). `DevicePtr::NULL` without a router pair.
    pub(super) router_zero_indices: DevicePtr,
}

impl MoeLayer {
    /// 2026-09-25: Install this layer's router and routed-expert LoRA, allocating
    /// zeroed `xa` `[cap, max_rank]` and `delta` `[cap, lora_delta_cols(..)]` and the
    /// device route tables. With neither a router nor an expert pair, sets
    /// `lora = None`.
    pub(crate) fn set_lora_weights(
        &mut self,
        router: Option<LoraPair>,
        experts: ExpertLoraLayer,
        kernels: LoraKernels,
        gpu: &dyn GpuBackend,
    ) -> Result<()> {
        if router.is_none() && experts.is_empty() {
            self.lora = None;
            return Ok(());
        }
        let max_rank = router
            .iter()
            .chain(experts.pairs.values())
            .map(|p| p.max_rank)
            .max()
            .unwrap_or(0) as usize;
        let cap = max_tokens() as usize;
        let router_n_out = router.as_ref().map(|p| p.n_out);
        let down_k_in = experts
            .pairs
            .iter()
            .filter(|((_, p), _)| *p == ExpertProj::Down)
            .map(|(_, pr)| pr.k_in)
            .max();
        let delta_cols = lora_delta_cols(router_n_out, down_k_in);
        let xa = gpu.alloc(cap * max_rank.max(1) * 2)?;
        let delta = gpu.alloc(cap * delta_cols * 2)?;
        gpu.memset(xa, 0, cap * max_rank.max(1) * 2)?;
        gpu.memset(delta, 0, cap * delta_cols * 2)?;
        let expert_route = Self::build_expert_route(&experts, ExpertProj::Down, gpu)?;
        let gate_route = Self::build_expert_route(&experts, ExpertProj::Gate, gpu)?;
        let up_route = Self::build_expert_route(&experts, ExpertProj::Up, gpu)?;
        let router_route = match router.as_ref() {
            Some(rp) => Some(Self::build_router_route(rp, gpu)?),
            None => None,
        };
        let router_zero_indices = if router.is_some() {
            let d = gpu.alloc(cap * 4)?;
            gpu.memset(d, 0, cap * 4)?;
            d
        } else {
            DevicePtr::NULL
        };
        tracing::info!(
            "MoE LoRA installed: router={}, {} expert pair(s) (gate={} up={} down={}), \
             cap={cap} rows, scratch={:.2} MiB",
            router.is_some(),
            experts.pairs.len(),
            gate_route.as_ref().map_or(0, |r| r.n_experts),
            up_route.as_ref().map_or(0, |r| r.n_experts),
            expert_route.as_ref().map_or(0, |r| r.n_experts),
            (cap * (max_rank.max(1) + delta_cols) * 2) as f64 / (1024.0 * 1024.0),
        );
        self.lora = Some(MoeLoraWeights {
            router,
            kernels,
            cap: cap as u32,
            xa,
            delta,
            expert_route,
            gate_route,
            up_route,
            router_route,
            router_zero_indices,
        });
        Ok(())
    }

    /// 2026-09-25: Upload the device route tables for the layer's `proj` pairs:
    /// dense `[n_experts]` u64 A and B addresses and f32 scales, 0 for an unadapted
    /// expert, with `n_experts` = highest adapted id + 1 for this projection.
    /// `None` when no pair targets `proj`. `k_in`, `n_out` and `max_rank` are read
    /// from one pair, so every pair of a projection must share them.
    pub(super) fn build_expert_route(
        experts: &ExpertLoraLayer,
        proj: ExpertProj,
        gpu: &dyn GpuBackend,
    ) -> Result<Option<MoeExpertRoute>> {
        let entries: Vec<(u16, u64, u64, f32)> = experts
            .pairs
            .iter()
            .filter(|((_, p), _)| *p == proj)
            .map(|((e, _), pair)| (*e, pair.a.weight.0, pair.b.weight.0, pair.scale))
            .collect();
        let Some(tables) = pack_expert_tables(&entries) else {
            return Ok(None);
        };
        let sample = experts
            .pairs
            .iter()
            .find(|((_, p), _)| *p == proj)
            .map(|(_, pr)| pr)
            .expect("pack_expert_tables returned Some => a matching pair exists");
        let up = |vals: &[u8]| -> Result<DevicePtr> {
            let d = gpu.alloc(vals.len())?;
            gpu.copy_h2d(vals, d)?;
            Ok(d)
        };
        let a_bytes: Vec<u8> = tables.a.iter().flat_map(|p| p.to_le_bytes()).collect();
        let b_bytes: Vec<u8> = tables.b.iter().flat_map(|p| p.to_le_bytes()).collect();
        let s_bytes: Vec<u8> = tables.scale.iter().flat_map(|s| s.to_le_bytes()).collect();
        Ok(Some(MoeExpertRoute {
            a_table: up(&a_bytes)?,
            b_table: up(&b_bytes)?,
            scale_table: up(&s_bytes)?,
            n_experts: tables.n_experts,
            k_in: sample.k_in,
            n_out: sample.n_out,
            max_rank: sample.max_rank,
        }))
    }

    /// 2026-09-25: Per-request fold gate on `ctx.moe_lora_route`: `Fold` gives
    /// `Ok(true)`, `Skip` (a base or non-active request) `Ok(false)`, and `Refuse`
    /// (a mixed batch, or an adapter that is not the active one) an error rather
    /// than folding one adapter onto rows it does not own.
    pub(super) fn moe_route_gate(&self, ctx: &ForwardContext, path: &str) -> Result<bool> {
        match ctx.moe_lora_route {
            MoeLoraRoute::Fold => Ok(true),
            MoeLoraRoute::Skip => Ok(false),
            MoeLoraRoute::Refuse => anyhow::bail!(
                "MoE LoRA (Feature-1) cannot honor per-row adapter identity in this {path} pass \
                 (packed/mixed batch, or a non-active adapter under single-active phase-1); \
                 refusing rather than folding one adapter onto rows it does not own. The \
                 device-side per-row grouped fold is the follow-up (docs/design/lora-solid.md \
                 Incr 1/3)."
            ),
        }
    }

    /// 2026-09-25: Fold the routed-expert down_proj LoRA deltas onto the sorted
    /// `expert_down_out` (`[te, hidden]`) before the weighted reduce, so the routing
    /// weight scales base + delta. `x` is the activated sorted `expert_gate_out`.
    ///
    /// Launches `moe_lora_grouped_down` per window of `cap` rows; it reads
    /// `expert_offsets` on the device (no host copy). The NVFP4, BF16 and FP8
    /// grouped prefills all call it. A no-op without MoE LoRA, without down pairs,
    /// or when `moe_route_gate` returns false.
    pub(crate) fn apply_expert_lora_prefill_down(
        &self,
        expert_gate_out: DevicePtr,
        expert_down_out: DevicePtr,
        expert_offsets: DevicePtr,
        sorted_token_ids: DevicePtr,
        te: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let Some(ref l) = self.lora else {
            return Ok(());
        };
        let Some(ref route) = l.expert_route else {
            return Ok(());
        };
        if !self.moe_route_gate(ctx, "expert-down")? {
            return Ok(());
        }
        // 2026-09-25: `xa` holds `cap` rows, so te rows are folded in contiguous
        // windows of `cap`, each shrink before its expand on `stream`. The fold
        // has no cross-row reduction, so the windowing cannot change a row's result.
        for (off, end) in ops::grouped_down_windows(te, l.cap) {
            // 2026-09-25: No per-row adapter map (NULL): every row folds; a base
            // request was skipped by `moe_route_gate` above.
            ops::moe_lora_grouped_down(
                ctx.gpu,
                &l.kernels,
                route,
                expert_gate_out,
                expert_down_out,
                expert_offsets,
                sorted_token_ids,
                DevicePtr::NULL,
                l.xa,
                off,
                end,
                0,
                stream,
            )?;
        }
        Ok(())
    }

    /// 2026-09-25: Fold the routed-expert down_proj LoRA deltas onto the slot-major
    /// decode `expert_down_out` (`[n_slots, hidden]`, n_slots = rows * top_k) in
    /// place, before `moe_weighted_sum_blend`, so the routing weight scales
    /// base + delta. `indices_dev` is the `[n_slots]` u32 expert-id array of the
    /// expert GEMVs.
    ///
    /// `x = act(gate) * up` is recomputed into `l.delta` with `moe_act_mul`, then
    /// `moe_lora_gather_bgmv` folds. `row_adapter` is the device `[rows]` i32 map
    /// (`< 0` = base row, skipped) or `DevicePtr::NULL` to fold every row after
    /// the request-level `moe_route_gate`. Errors if `n_slots > cap`. A no-op
    /// without MoE LoRA or without down pairs.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn apply_expert_lora_decode_down(
        &self,
        expert_gate_out: DevicePtr,
        expert_up_out: DevicePtr,
        expert_down_out: DevicePtr,
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
        let Some(ref route) = l.expert_route else {
            return Ok(());
        };
        // 2026-09-25: With a `row_adapter` map the launch is unconditional and base
        // rows are skipped on the device, so a captured graph holds no host
        // branch on the route; `moe_route_gate` is consulted only without a map.
        if row_adapter == DevicePtr::NULL && !self.moe_route_gate(ctx, "expert-down-decode")? {
            return Ok(());
        }
        anyhow::ensure!(
            n_slots <= l.cap,
            "MoE expert LoRA decode down-fold: n_slots ({n_slots}) exceeds LoRA scratch cap \
             ({}); raise METRALE_LORA_EXPERT_MAX_TOKENS to >= num_tokens*top_k.",
            l.cap
        );
        // 2026-09-25: `l.delta` receives x as packed `[n_slots, k_in]` BF16.
        ops::moe_silu_mul(
            ctx.gpu,
            self.moe_act_mul,
            expert_gate_out,
            expert_up_out,
            l.delta,
            n_slots * route.k_in,
            stream,
        )?;
        ops::moe_lora_gather_bgmv(
            ctx.gpu,
            &l.kernels,
            route,
            l.delta,
            expert_down_out,
            indices_dev,
            row_adapter,
            l.xa,
            n_slots,
            top_k,
            0,
            stream,
        )
    }

    /// 2026-09-25: Refusal for decode passes that cannot fold MoE LoRA (`forward`
    /// with several sequences, `forward_atomic_c4_decode`): errors when an adapter
    /// is installed and the batch route is not `Skip`; `Ok` otherwise.
    pub(crate) fn reject_decode_lora(&self, ctx: &ForwardContext, path: &str) -> Result<()> {
        // 2026-09-25: A `Skip` batch has no delta to fold, so base requests decode
        // while an adapter is installed. The model stamps the route per batch
        // (`stamp_decode_moe_single` / `stamp_decode_moe_batch`).
        if self.lora.is_some() && !matches!(ctx.moe_lora_route, MoeLoraRoute::Skip) {
            anyhow::bail!(
                "MoE LoRA (Feature-1) is prefill-only in phase 1; the {path} decode/verify \
                 path does not yet fold the expert/router delta for an adapter-routed request. \
                 Use the adapter for prefill-logit scoring, or wait for the decode-fold followup \
                 (docs/design/lora-solid.md Incr-4)."
            );
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "lora_tests.rs"]
mod tests;
