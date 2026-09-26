// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Per-section decode timing for the GLM-5.3 layers, enabled by `METRALE_GLM_PROFILE`.
//!
//! Owner: model-arch (GLM-5.3).
//! Invariants:
//! - Unless `METRALE_GLM_PROFILE` is `1` or `2`, `start` and `start_hot` return `None` and no
//!   span synchronises.
//! - `end` and `end_us` synchronise the stream before reading the clock, so a profiled run is
//!   serialised and its throughput is not a serving figure.

use metrale_gpu_runtime::gpu::GpuBackend;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::Instant;

pub const MHC: usize = 0;
pub const NORM: usize = 1;
pub const KDA: usize = 2;
pub const DSA_PROJ: usize = 3;
pub const DSA_INDEXER: usize = 4;
pub const DSA_SELECT: usize = 5;
pub const DSA_ATTEND: usize = 6;
pub const REDUCE_ATTN: usize = 7;
pub const MLP_DENSE: usize = 8;
pub const MOE_ROUTER: usize = 9;
pub const MOE_HOSTSYNC: usize = 10;
pub const MOE_EXPERTS: usize = 11;
pub const MOE_SHARED: usize = 12;
pub const MOE_COMBINE: usize = 13;
pub const REDUCE_MLP: usize = 14;
/// 2026-09-25: Host time to enqueue the attention and MLP all-reduces, without a synchronise.
/// `REDUCE_ATTN` and `REDUCE_MLP` then time only the `synchronize` that follows the enqueue.
pub const REDUCE_ATTN_ENQ: usize = 15;
pub const REDUCE_MLP_ENQ: usize = 16;
/// 2026-09-25: A 2-byte all-reduce issued just before the real one while profiling
/// (`Glm5NextLayer::reduce_probe`), so that the ranks' arrival skew is charged here and not to
/// `REDUCE_ATTN` or `REDUCE_MLP`.
pub const REDUCE_ATTN_BAR: usize = 17;
pub const REDUCE_MLP_BAR: usize = 18;
/// 2026-09-25: `hc_post`, plus `hc_head_mean` on the last layer; `MHC` holds `hc_expand` and
/// `hc_pre`.
pub const MHC_POST: usize = 19;
/// 2026-09-25: Sub-spans of the KDA mixer: the projections before the recurrence, the
/// recurrence, and the output side, timed in `glm5next_kda`'s decode and prefill.
pub const KDA_FRONT: usize = 20;
pub const KDA_RECUR: usize = 21;
pub const KDA_BACK: usize = 22;
const N: usize = 23;

const NAMES: [&str; N] = [
    "mhc",
    "norm",
    "kda_mixer",
    "dsa_proj",
    "dsa_indexer",
    "dsa_select",
    "dsa_attend",
    "reduce_attn",
    "mlp_dense",
    "moe_router",
    "moe_hostsync",
    "moe_experts",
    "moe_shared",
    "moe_combine",
    "reduce_mlp",
    "reduce_attn_enq",
    "reduce_mlp_enq",
    "reduce_attn_bar",
    "reduce_mlp_bar",
    "mhc_post",
    "kda_front",
    "kda_recur",
    "kda_back",
];

static NANOS: [AtomicU64; N] = [const { AtomicU64::new(0) }; N];
static CALLS: [AtomicU64; N] = [const { AtomicU64::new(0) }; N];
static STEPS: AtomicU64 = AtomicU64::new(0);

/// 2026-09-25: `METRALE_GLM_PROFILE`: `1` times every span, `2` only the collective spans
/// (`start_hot`), anything else is off. Read once.
fn level() -> u8 {
    static L: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *L.get_or_init(|| match std::env::var("METRALE_GLM_PROFILE").as_deref() {
        Ok("1") => 1,
        Ok("2") => 2,
        _ => 0,
    })
}

pub fn on() -> bool {
    level() != 0
}

/// 2026-09-25: True only at level 1, the per-kernel spans.
pub fn full() -> bool {
    level() == 1
}

/// 2026-09-25: Open a span at level 1; `None` otherwise, which makes [`end`] a no-op.
pub fn start() -> Option<Instant> {
    full().then(Instant::now)
}

/// 2026-09-25: Open a span at level 1 or 2, for the collectives and their rendezvous probe.
pub fn start_hot() -> Option<Instant> {
    on().then(Instant::now)
}

pub fn end(bucket: usize, t0: Option<Instant>, gpu: &dyn GpuBackend, stream: u64) {
    let _ = end_us(bucket, t0, gpu, stream);
}

/// 2026-09-25: [`end`], returning the span in microseconds; 0.0 when `t0` is `None`.
pub fn end_us(bucket: usize, t0: Option<Instant>, gpu: &dyn GpuBackend, stream: u64) -> f64 {
    let Some(t0) = t0 else { return 0.0 };
    let _ = gpu.synchronize(stream);
    let ns = t0.elapsed().as_nanos() as u64;
    NANOS[bucket].fetch_add(ns, Relaxed);
    CALLS[bucket].fetch_add(1, Relaxed);
    ns as f64 / 1e3
}

/// 2026-09-25: `METRALE_GLM_ROUTE_TRACE=1`: `trace_bar` logs one line per reduce site per layer
/// with the rendezvous time and the expert ids `stash_route` last saved. Read once. It logs only
/// while profiling is on too, because `trace_bar` is called from `reduce_probe`.
pub fn trace_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("METRALE_GLM_ROUTE_TRACE").as_deref() == Ok("1"))
}

thread_local! {
    /// 2026-09-25: The ids `forward_moe` last read back to the host, for the next trace line.
    static ROUTE: std::cell::RefCell<Vec<i32>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// 2026-09-25: Called from `forward_moe` after the ids are read back. No-op unless tracing.
pub fn stash_route(ids: &[i32]) {
    if !trace_on() {
        return;
    }
    ROUTE.with(|r| {
        let mut v = r.borrow_mut();
        v.clear();
        v.extend_from_slice(ids);
    });
}

/// 2026-09-25: Log the trace line of one reduce site: `site` is `attn` or `mlp`, and `moe` says
/// whether this layer's MLP is routed. No-op unless tracing.
pub fn trace_bar(site: &str, layer: usize, moe: bool, us: f64) {
    if !trace_on() {
        return;
    }
    let step = STEPS.load(Relaxed);
    ROUTE.with(|r| {
        let v = r.borrow();
        let ids = v
            .iter()
            .map(|x| x.to_string())
            .collect::<Vec<_>>()
            .join(",");
        tracing::warn!(
            "GLMTRACE step={step} site={site} L={layer} moe={} bar_us={us:.1} ids={ids}",
            u8::from(moe)
        );
    });
}

/// 2026-09-25: 4-byte device buffer for the rendezvous probe, allocated on first use; 0 when the
/// allocation failed, and `reduce_probe` then skips the probe.
pub fn probe_buf(gpu: &dyn GpuBackend) -> u64 {
    static P: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *P.get_or_init(|| gpu.alloc(4).map(|p| p.0).unwrap_or(0))
}

/// 2026-09-25: Record a span without synchronising: host wall time only.
pub fn end_nosync(bucket: usize, t0: Option<Instant>) {
    let Some(t0) = t0 else { return };
    NANOS[bucket].fetch_add(t0.elapsed().as_nanos() as u64, Relaxed);
    CALLS[bucket].fetch_add(1, Relaxed);
}

/// 2026-09-25: Count one token. Every 8th token, log the cumulative time per token and calls per
/// token of each bucket.
pub fn step() {
    if !on() {
        return;
    }
    let s = STEPS.fetch_add(1, Relaxed) + 1;
    if !s.is_multiple_of(8) {
        return;
    }
    let total: u64 = NANOS.iter().map(|n| n.load(Relaxed)).sum();
    let mut rows: Vec<(usize, u64, u64)> = (0..N)
        .map(|i| (i, NANOS[i].load(Relaxed), CALLS[i].load(Relaxed)))
        .collect();
    rows.sort_by_key(|r| std::cmp::Reverse(r.1));
    let mut out = format!(
        "GLM decode profile after {s} steps — {:.2} ms/token measured under profiling\n",
        total as f64 / 1e6 / s as f64
    );
    for (i, ns, calls) in rows {
        if calls == 0 {
            continue;
        }
        out += &format!(
            "  {:<13} {:>8.2} ms/tok  {:>6.1}%  {:>5} calls/tok  {:>7.1} us/call\n",
            NAMES[i],
            ns as f64 / 1e6 / s as f64,
            100.0 * ns as f64 / total.max(1) as f64,
            calls / s,
            ns as f64 / 1e3 / calls.max(1) as f64,
        );
    }
    tracing::warn!("{out}");
}
