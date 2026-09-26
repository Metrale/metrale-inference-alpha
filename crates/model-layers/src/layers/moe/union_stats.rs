// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Sampled expert-union telemetry for the two- and three-row MoE
//! decode batches (`forward_k2`, `forward_k3`), enabled by the `moe_union_stats`
//! lever (`METRALE_MOE_UNION_STATS`). It logs the mean number of distinct experts
//! per sampled layer step next to the mean routed slots (`m * top_k`).
//!
//! Owner: model-layers (MoE).
//! Invariants:
//! - With the lever off, or while `stream` is capturing a graph, it returns
//!   before any synchronize or copy.

use std::sync::atomic::Ordering;

use metrale_gpu_runtime::gpu::DevicePtr;

const SAMPLE_EVERY: u64 = 64;
const LOG_EVERY: u64 = 64;

/// 2026-09-25: Sample the expert-id union for one MoE layer's batch. `indices_dev`
/// holds `m * top_k` u32 expert ids written on `stream`. One call in
/// [`SAMPLE_EVERY`] (counted in this model's `ModelStats::moe_union`) is sampled,
/// and the running means are logged every [`LOG_EVERY`] samples.
pub(super) fn maybe_sample_expert_union(
    ctx: &crate::layer::ForwardContext<'_>,
    indices_dev: DevicePtr,
    m: usize,
    top_k: usize,
    stream: u64,
) {
    if !ctx.levers.moe_union_stats {
        return;
    }
    // 2026-09-25: A synchronize inside a graph capture invalidates the capture.
    // Graph replays run no host code, so only eager steps are sampled.
    let gpu = ctx.gpu;
    if gpu.stream_is_capturing(stream) {
        return;
    }
    let call = ctx.stats.moe_union.calls.fetch_add(1, Ordering::Relaxed);
    if !call.is_multiple_of(SAMPLE_EVERY) {
        return;
    }
    // 2026-09-25: Order the copy after the kernel that wrote the indices.
    if gpu.synchronize(stream).is_err() {
        return;
    }
    let n = m * top_k;
    let mut buf = vec![0u8; n * 4];
    if gpu.copy_d2h(indices_dev, &mut buf).is_err() {
        return;
    }
    let mut ids: Vec<u32> = buf
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    ids.sort_unstable();
    ids.dedup();
    let unique = ids.len() as u64;
    ctx.stats
        .moe_union
        .unique_sum
        .fetch_add(unique, Ordering::Relaxed);
    ctx.stats
        .moe_union
        .slots_sum
        .fetch_add(n as u64, Ordering::Relaxed);
    let s = ctx.stats.moe_union.samples.fetch_add(1, Ordering::Relaxed) + 1;
    if s.is_multiple_of(LOG_EVERY) {
        let uniq = ctx.stats.moe_union.unique_sum.load(Ordering::Relaxed) as f64 / s as f64;
        let slots = ctx.stats.moe_union.slots_sum.load(Ordering::Relaxed) as f64 / s as f64;
        tracing::info!(
            "moe-union-stats: samples={s} mean_unique_experts={uniq:.1} \
             mean_routed_slots={slots:.1} overlap_saving={:.0}% (m={m} top_k={top_k})",
            (1.0 - uniq / slots.max(1.0)) * 100.0,
        );
    }
}
