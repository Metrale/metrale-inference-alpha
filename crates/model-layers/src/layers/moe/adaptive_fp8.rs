// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Hopper short-prefill W8A8: exact-arithmetic M16/M128 expert buckets.
//! Device routing counts choose the bucket; sparse small buckets fold back into
//! M128. Both lists and counters belong to the arena and survive graph replay.
//!
//! Owner: model-layers (MoE).
//! Invariants: none beyond the types.

use std::sync::Mutex;

use super::*;
use metrale_gpu_runtime::kernel_args::KernelLaunch;

pub(super) struct SideJoin<'a> {
    gpu: &'a dyn GpuBackend,
    main: u64,
    side: u64,
    done: u64,
}

impl SideJoin<'_> {
    pub(super) fn side(&self) -> u64 {
        self.side
    }
}

impl Drop for SideJoin<'_> {
    fn drop(&mut self) {
        // 2026-09-25: Queue the wait on `main` so the next launch there sees side-stream
        // writes. If the GPU wait cannot be queued, block on the side stream.
        if self.gpu.record_event(self.done, self.side).is_err()
            || self.gpu.stream_wait_event(self.main, self.done).is_err()
        {
            let _ = self.gpu.synchronize(self.side);
        }
    }
}

fn overlap_k2048(k: u32) -> bool {
    k == 2048
}

/// 2026-09-25: Short C1-shape prefill only. Shared W8A8 and the router read the same
/// hidden state and write different buffers; other shapes stay serial.
pub(super) fn overlap_shared_router(rows: usize, experts: u32, topk: u32) -> bool {
    (65..=128).contains(&rows) && experts == 256 && topk == 8
}

pub(super) fn begin_shared_side<'a>(gpu: &'a dyn GpuBackend, main: u64) -> Result<SideJoin<'a>> {
    let (side, ready, done) = k2048_side(gpu)?;
    gpu.record_event(ready, main)?;
    gpu.stream_wait_event(side, ready)?;
    Ok(SideJoin {
        gpu,
        main,
        side,
        done,
    })
}

fn k2048_side(gpu: &dyn GpuBackend) -> Result<(u64, u64, u64)> {
    static SLOT: Mutex<Option<(u64, u64, u64)>> = Mutex::new(None);
    let mut slot = SLOT.lock().unwrap_or_else(|err| err.into_inner());
    if let Some(ready) = *slot {
        return Ok(ready);
    }
    let created = (
        gpu.create_stream()?,
        gpu.create_event()?,
        gpu.create_event()?,
    );
    *slot = Some(created);
    Ok(created)
}

pub(super) fn adaptive_sm_count(gpu: &dyn GpuBackend) -> Result<u32> {
    if gpu.has_module("moe_bucket_builder") && gpu.has_module("moe_w8a8_m16") {
        gpu.sm_count()
    } else {
        Ok(0)
    }
}

#[derive(Debug)]
struct Plan {
    small_items: u32,
    large_items: u32,
    small_grid: u32,
    large_grid: u32,
    threshold: u32,
}

#[allow(clippy::too_many_arguments)]
fn plan(
    rows: usize,
    experts: usize,
    topk: usize,
    decode: bool,
    n: u32,
    k: u32,
    sms: u32,
    available: bool,
) -> Option<Plan> {
    // 2026-09-25: Only the qualified short-prefill dimensions. Decode and other models keep
    // their existing route; widening needs its own numerical/performance gate.
    if !available
        || decode
        || !(65..=128).contains(&rows)
        || experts != 256
        || topk != 8
        || sms == 0
        || !matches!((n, k), (512, 2048) | (2048, 512))
    {
        return None;
    }
    let nt = n.div_ceil(64);
    let small_items = 256u32.checked_mul(nt)?;
    let expanded = u32::try_from(rows.checked_mul(topk)?).ok()?;
    let large_items = expanded.div_ceil(128).checked_add(257)?.checked_mul(nt)?;
    Some(Plan {
        small_items,
        large_items,
        small_grid: small_items.min(sms.checked_mul(16)?).clamp(1, 16384),
        large_grid: large_items.min(sms.checked_mul(4)?).clamp(1, 16384),
        threshold: sms.checked_mul(4)?,
    })
}

impl MoeLayer {
    /// 2026-09-25: Returns false without launching when this exact route is unavailable.
    /// A pair shares the gate/up residency pattern, as in the existing PM4 arm.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn try_adaptive_fp8(
        &self,
        input: DevicePtr,
        scales: DevicePtr,
        projections: &[(&Fp8ExpertPtrTable, DevicePtr)],
        offsets: DevicePtr,
        sorted: DevicePtr,
        n: u32,
        k: u32,
        rows: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        let Some(p) = plan(
            rows,
            ctx.config.num_experts,
            ctx.config.num_experts_per_tok,
            ctx.decode_step,
            n,
            k,
            self.moe_adaptive_sms,
            self.moe_bucket_builder_k.0 != 0
                && self.moe_w8a8_m16_k.0 != 0
                && self.moe_w8a8_grouped_gemm_pm4_k.0 != 0,
        ) else {
            return Ok(false);
        };
        anyhow::ensure!(
            !projections.is_empty() && projections.len() <= 2,
            "adaptive FP8 requires one projection or a gate/up pair"
        );
        let scratch = ctx.buffers.moe_fp8_scratch(ctx.config, rows)?;
        let small_bytes = usize::try_from(p.small_items)?
            .checked_mul(8)
            .ok_or_else(|| anyhow::anyhow!("adaptive small worklist overflow"))?;
        let large_bytes = usize::try_from(p.large_items)?
            .checked_mul(8)
            .ok_or_else(|| anyhow::anyhow!("adaptive large worklist overflow"))?;
        anyhow::ensure!(
            small_bytes <= scratch.small_worklist_bytes
                && large_bytes <= scratch.worklist_bytes
                && scratch.small_worklist != DevicePtr::NULL
                && scratch.small_total_tiles != DevicePtr::NULL,
            "adaptive FP8 worklist exceeds persistent arena capacity"
        );
        if ctx.stats.once("log:moe_adaptive_fp8_prefill") {
            tracing::info!(
                "[metrale] Hopper adaptive W8A8 prefill: M16/M128, SMs={}, small-tile threshold={} (device decision), persistent worklists, K=2048 gate/up large bucket overlaps on a side stream",
                self.moe_adaptive_sms,
                p.threshold
            );
        }
        // 2026-09-25: K=2048 gate/up: the M16 grid fills every SM, then a short tail leaves
        // the rest idle. The large bucket writes other experts' rows, so it can
        // occupy those SMs. Down (K=512) stays on the caller stream.
        let side = if overlap_k2048(k) {
            Some(k2048_side(ctx.gpu)?)
        } else {
            None
        };
        let large_stream = side
            .as_ref()
            .map(|(side_stream, _, _)| *side_stream)
            .unwrap_or(stream);
        KernelLaunch::new(ctx.gpu, self.moe_bucket_builder_k)
            .grid([1, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(offsets)
            .arg_ptr(projections[0].0.weight_ptrs)
            .arg_ptr(scratch.worklist)
            .arg_ptr(scratch.total_tiles)
            .arg_ptr(scratch.small_worklist)
            .arg_ptr(scratch.small_total_tiles)
            .arg_u32(256)
            .arg_u32(n.div_ceil(64))
            .arg_u32(p.threshold)
            .launch(stream)?;
        let _join = if let Some((side_stream, ready, done)) = side {
            ctx.gpu.record_event(ready, stream)?;
            ctx.gpu.stream_wait_event(side_stream, ready)?;
            Some(SideJoin {
                gpu: ctx.gpu,
                main: stream,
                side: side_stream,
                done,
            })
        } else {
            None
        };
        for &(weights, output) in projections {
            KernelLaunch::new(ctx.gpu, self.moe_w8a8_m16_k)
                .grid([p.small_grid, 1, 1])
                .block([32, 1, 1])
                .arg_ptr(input)
                .arg_ptr(scales)
                .arg_ptr(weights.weight_ptrs)
                .arg_ptr(weights.scale_ptrs)
                .arg_ptr(output)
                .arg_ptr(offsets)
                .arg_ptr(sorted)
                .arg_u32(256)
                .arg_u32(n)
                .arg_u32(k)
                .arg_ptr(scratch.small_worklist)
                .arg_ptr(scratch.small_total_tiles)
                .launch(stream)?;
            ops::moe_w8a8_grouped_gemm_pm4(
                ctx.gpu,
                self.moe_w8a8_grouped_gemm_pm4_k,
                input,
                scales,
                weights.weight_ptrs,
                weights.scale_ptrs,
                output,
                offsets,
                sorted,
                256,
                n,
                k,
                scratch.worklist,
                scratch.total_tiles,
                p.large_grid,
                large_stream,
            )?;
        }
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::{overlap_k2048, overlap_shared_router, plan};

    #[test]
    fn k2048_gate_up_overlaps_and_down_does_not() {
        assert!(overlap_k2048(2048));
        assert!(!overlap_k2048(512));
        assert!(!overlap_k2048(0));
    }

    #[test]
    fn shared_router_overlap_is_the_short_c1_shape_only() {
        assert!(overlap_shared_router(128, 256, 8));
        assert!(overlap_shared_router(65, 256, 8));
        assert!(!overlap_shared_router(64, 256, 8));
        assert!(!overlap_shared_router(129, 256, 8));
        assert!(!overlap_shared_router(128, 257, 8));
        assert!(!overlap_shared_router(128, 256, 4));
    }

    #[test]
    fn qualified_shape_bounds_and_grid_capacity_are_independent() {
        for rows in [65, 96, 128] {
            for (n, k) in [(512, 2048), (2048, 512)] {
                let p = plan(rows, 256, 8, false, n, k, 132, true).unwrap();
                assert_eq!(p.small_items, 256 * n.div_ceil(64));
                assert!(p.small_grid <= p.small_items && p.large_grid <= p.large_items);
                assert_eq!(p.threshold, 528);
                assert_eq!(p.large_grid, p.large_items.min(528));
            }
        }
    }
    #[test]
    fn missing_module_decode_other_shapes_and_overflow_keep_original_path() {
        assert!(plan(128, 256, 8, false, 512, 2048, 132, true).is_some());
        for rows in [0, 1, 32, 64, 129, usize::MAX] {
            assert!(plan(rows, 256, 8, false, 512, 2048, 132, true).is_none());
        }
        for (e, topk, decode, n, k, sms, available) in [
            (257, 8, false, 512, 2048, 132, true),
            (256, 4, false, 512, 2048, 132, true),
            (256, 8, true, 512, 2048, 132, true),
            (256, 8, false, 4096, 512, 132, true),
            (256, 8, false, 512, 512, 132, true),
            (256, 8, false, 512, 2048, 0, true),
            (256, 8, false, 512, 2048, u32::MAX, true),
            (256, 8, false, 512, 2048, 132, false),
        ] {
            assert!(plan(128, e, topk, decode, n, k, sms, available).is_none());
        }
    }
}
