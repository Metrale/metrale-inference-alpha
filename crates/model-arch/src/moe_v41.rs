// SPDX-License-Identifier: MIT OR Apache-2.0
// provenance-id: 526f6e616c6420522e205374657369616b

//! 2026-09-25: DeepSeek-V4.1 Flash MoE on streamed, still-quantized experts.
//!
//! The routed experts stay in their GGUF blocks: gate and up run on Q2_K
//! kernels, down on Q3_K kernels. Per token the router picks `topk` of
//! `n_routed` experts, the [`ExpertLru`](metrale_model_weights::weights::expert_stream::ExpertLru) gathers those experts' raw
//! slices into its device-visible slots (misses are read from the
//! `ExpertSource`), and the kernels read the blocks directly with the
//! activation quantised to q8_1: GEMVs for groups of up to eight rows, and by
//! default MMQ above. The shared expert is resident, bf16 or K-quant (`ResidentMat`).
//!
//! Routing matches `deepseek_v41_ref::moe::gate`: f32-accumulated router
//! logits, scores `sqrt(softplus(logit / gate_temp))`, the top-k chosen by
//! `score + gate_bias` with a stable sort, and the weights the unbiased scores,
//! renormalised when `norm_topk_prob` and scaled by `route_scale`. By default
//! the selection runs on the CPU from the downloaded logits; `device_route`
//! holds the device version.
//!
//! Tested in `moe_v41_tests.rs` on synthetic Q2_K / Q3_K experts against the
//! reference `gate` and a CPU emulation of the numerics.
//!
//! Owner: model-arch (DeepSeek-V4.1 MoE).
//! Invariants: none beyond the types.

use metrale_gpu_runtime::gpu::{DevicePtr, KernelHandle};

use metrale_model_layers::layers::ops::ResidentMat;

mod device_route;
mod forward;

/// 2026-09-25: Layers the device slot table and the route headers cover;
/// `device_route_ok` refuses the device selection for a layer at or past it.
pub(crate) const SLOT_TABLE_LAYERS: usize = 64;
mod init;
mod route;
mod shared;
mod single;

const MODULE: &str = "moe_v41";
const GEMM_MODULE: &str = "gemm";

#[derive(Clone, Debug)]
pub struct MoeV41Cfg {
    pub dim: usize,
    pub inter: usize,
    pub n_routed: usize,
    pub topk: usize,
    pub gate_temp: f32,
    pub norm_topk_prob: bool,
    pub route_scale: f32,
    pub swiglu_limit: f32,
    pub max_tokens: usize,
}

/// 2026-09-25: One layer's resident MoE weights: the router and the shared expert.
pub struct MoeV41LayerWeights {
    pub layer: u32,
    /// 2026-09-25: bf16 `[n_routed, dim]`.
    pub gate_w: DevicePtr,
    /// 2026-09-25: f32 `[n_routed]` on the host, for the CPU selection.
    pub gate_bias: Vec<f32>,
    /// 2026-09-25: The same bias on the device, for the device-side selection.
    pub gate_bias_dev: DevicePtr,
    /// 2026-09-25: w1, w2, w3 as `[inter, dim]`, `[dim, inter]`, `[inter, dim]`:
    /// bf16, or the GGUF's K-quant blocks run on the routed experts' kernels.
    pub shared_w1: ResidentMat,
    pub shared_w2: ResidentMat,
    pub shared_w3: ResidentMat,
}

/// 2026-09-25: The router of one layer alone: what predicting a layer's
/// selection from an earlier layer's input needs (see `MoeV41::predict_launch`).
pub struct RouterWeights {
    pub layer: u32,
    /// 2026-09-25: bf16 `[n_routed, dim]`.
    pub gate_w: DevicePtr,
    pub gate_bias: Vec<f32>,
}

/// 2026-09-25: `METRALE_DS41_PREFETCH_K`, read once (default 0, off): how
/// many of the next layer's predicted experts to start reading while this
/// layer computes. It acts only at one token, with the host selection, and
/// with the reader pool (`METRALE_DS41_READER_POOL=1`).
pub fn prefetch_k() -> usize {
    static K: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *K.get_or_init(|| {
        std::env::var("METRALE_DS41_PREFETCH_K")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    })
}

struct Kernels {
    gemm: KernelHandle,
    /// 2026-09-25: The bf16 shared expert at one token.
    gemv: KernelHandle,
    gemm_f32out: KernelHandle,
    /// 2026-09-25: Router logits at up to eight tokens with
    /// `METRALE_DS41_ROUTER_STAGED=0`, and for the prediction.
    router_gemv: KernelHandle,
    /// 2026-09-25: Router logits at up to eight tokens by default: one block
    /// per expert row, the products staged in shared memory.
    router_gemv_staged: KernelHandle,
    q8_rows: KernelHandle,
    mmvq_q2k: KernelHandle,
    mmvq_q3k: KernelHandle,
    /// 2026-09-25: The single-token arm: all selected experts in one launch
    /// per projection.
    mmvq_q2k_experts: KernelHandle,
    mmvq_q3k_experts: KernelHandle,
    /// 2026-09-25: Warps per block of the expert batch: 2, 4 or 8
    /// (`METRALE_DS41_EXPERT_WARPS`, default 8).
    experts_warps: u32,
    swiglu: KernelHandle,
    /// 2026-09-25: The SwiGLU and the q8_1 rows of its output in one launch
    /// (up to eight rows).
    swiglu_q8: KernelHandle,
    accumulate: KernelHandle,
    finish: KernelHandle,
    gather: KernelHandle,
    scatter_add: KernelHandle,
    sum_rows: KernelHandle,
    quant_d2s6: KernelHandle,
    quant_d4: KernelHandle,
    mmq_q2k_nc: KernelHandle,
    mmq_q2k_wc: KernelHandle,
    mmq_q3k_nc: KernelHandle,
    mmq_q3k_wc: KernelHandle,
    route_select: KernelHandle,
    slot_table_set: KernelHandle,
}

/// 2026-09-25: Where one call's time went (wall clock, host side).
#[derive(Clone, Copy, Debug, Default)]
pub struct MoeV41Timing {
    pub route_ms: f64,
    pub fetch_ms: f64,
    pub compute_ms: f64,
    pub hits: u64,
    pub misses: u64,
    pub bytes_read: u64,
}

impl MoeV41Timing {
    pub fn add(&mut self, o: &MoeV41Timing) {
        self.route_ms += o.route_ms;
        self.fetch_ms += o.fetch_ms;
        self.compute_ms += o.compute_ms;
        self.hits += o.hits;
        self.misses += o.misses;
        self.bytes_read += o.bytes_read;
    }
}

/// 2026-09-25: What `stage_m1` leaves for `compute_m1`: the distinct expert
/// count and the routing (for callers and diagnostics), with the host-span
/// timing.
pub struct MoeV41Stage {
    pub ne: usize,
    pub weights: Vec<f32>,
    pub indices: Vec<usize>,
    pub timing: MoeV41Timing,
}

pub struct MoeV41 {
    /// 2026-09-25: The last forward's timing.
    pub last: std::cell::Cell<MoeV41Timing>,
    /// 2026-09-25: `METRALE_DS41_DIAG=1`: synchronise at the end of `forward`
    /// so `compute_ms` is GPU time.
    timing_sync: bool,
    pub cfg: MoeV41Cfg,
    k: Kernels,
    logits: DevicePtr,
    /// 2026-09-25: The next layer's router on this layer's input
    /// (`[n_routed]` f32).
    pred_logits: DevicePtr,
    /// 2026-09-25: Gathered rows of one expert group, `[m, dim]` bf16; also
    /// the staging buffer of `slot_table_update`.
    a_rows: DevicePtr,
    /// 2026-09-25: The group's q8_1 activations (plain rows or the MMQ layout).
    a_q8: DevicePtr,
    gate_out: DevicePtr,
    up_out: DevicePtr,
    h: DevicePtr,
    h_q8: DevicePtr,
    down_out: DevicePtr,
    /// 2026-09-25: `[m * topk]` i32 token rows and f32 routing weights,
    /// grouped by ascending expert id.
    rows_dev: DevicePtr,
    weight_dev: DevicePtr,
    /// 2026-09-25: The device selection's headers (`header_words` i32 words
    /// per layer: miss flag, picks, weight bits, plan slots) and the
    /// `[SLOT_TABLE_LAYERS][n_routed]` i32 slot table (-1 = not resident).
    route_hdr: DevicePtr,
    slot_table: DevicePtr,
    /// 2026-09-25: Gate, up and down block pointers of the token's experts,
    /// `3 * topk` u64.
    ptrs_dev: DevicePtr,
    sg: DevicePtr,
    su: DevicePtr,
    sh: DevicePtr,
    sd: DevicePtr,
    acc: DevicePtr,
    out: DevicePtr,
    /// 2026-09-25: The shared expert's own q8_1 scratch for its SwiGLU output
    /// (`[1, inter]`) on the side stream, so it never shares `h_q8` with the routed
    /// experts running at the same time; the token's input rows (`a_q8`) are
    /// written once on the forking stream and read by both.
    sh_q8: DevicePtr,
    /// 2026-09-25: The side stream the single-token shared expert runs on, and the two
    /// events that fence it: `ev_in` (the input is ready, recorded on the
    /// main stream) and `ev_out` (`sd` is ready, recorded on the side).
    side: u64,
    ev_in: u64,
    ev_out: u64,
}

/// 2026-09-25: `METRALE_DS41_SHARED_SIDE`, read once (default on; `0` turns it
/// off): at one token the shared expert runs on a side stream while the main
/// stream runs the router and the host picks the experts. Its output is added
/// into `acc` after the routed experts, the same point as on the main stream,
/// so the accumulation order does not change.
pub fn shared_side() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| !std::env::var("METRALE_DS41_SHARED_SIDE").is_ok_and(|v| v == "0"))
}

fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { x.exp().ln_1p() }
}

/// 2026-09-25: `deepseek_v41_ref::moe::gate` on f32 logits: returns
/// (`weights[tokens, topk]`, `indices[tokens, topk]`), each token's picks in
/// descending `score + bias` order, ties to the lower expert id. Panics on a
/// NaN key.
pub fn route_from_logits(
    logits: &[f32],
    tokens: usize,
    bias: &[f32],
    c: &MoeV41Cfg,
) -> (Vec<f32>, Vec<usize>) {
    let n = c.n_routed;
    let mut weights = Vec::with_capacity(tokens * c.topk);
    let mut indices = Vec::with_capacity(tokens * c.topk);
    for t in 0..tokens {
        let scores: Vec<f32> = (0..n)
            .map(|e| softplus(logits[t * n + e] / c.gate_temp).sqrt())
            .collect();
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| {
            (scores[b] + bias[b])
                .partial_cmp(&(scores[a] + bias[a]))
                .expect("finite scores")
        });
        let picked = &order[..c.topk];
        let mut wt: Vec<f32> = picked.iter().map(|&e| scores[e]).collect();
        if c.norm_topk_prob && c.topk > 1 {
            let sum: f32 = wt.iter().sum::<f32>() + 1e-20;
            for v in &mut wt {
                *v /= sum;
            }
        }
        for v in &mut wt {
            *v *= c.route_scale;
        }
        weights.extend(wt);
        indices.extend_from_slice(picked);
    }
    (weights, indices)
}

/// 2026-09-25: `v` rounded to bf16 (nearest, ties to even) as little-endian
/// bytes, for callers staging inputs.
pub fn bf16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter()
        .flat_map(|&x| {
            let b = x.to_bits();
            let lsb = (b >> 16) & 1;
            ((b.wrapping_add(0x7FFF + lsb) >> 16) as u16).to_le_bytes()
        })
        .collect()
}

#[cfg(test)]
#[path = "moe_v41_route_tests.rs"]
mod route_tests;
#[cfg(test)]
#[path = "moe_v41_tests.rs"]
mod tests;
