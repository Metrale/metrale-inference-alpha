// SPDX-License-Identifier: MIT OR Apache-2.0
// provenance-id: 526f6e616c6420522e205374657369616b

//! 2026-09-25: DeepSeek-V4.1 Flash: one transformer block on the generic model.
//!
//! The layer runs on the f32 mHC highway the generic model provides
//! (`ctx.buffers.hc_streams()`, `[T, hc, H]`), in the order of the CPU
//! reference `deepseek_v41_ref::model::forward`:
//!
//! ```text
//! layer 0 only:        hc_expand(embedding -> streams)
//! engram layers only:  streams += engram(token n-grams)          (EngramV41)
//! attention:           (pre_a, post_a, comb_a) = hc_mixes(streams, attn site)
//!                      x = rmsnorm(collapse(streams, pre_prev))   <- delayed pre
//!                      streams = hc_post(attn(x), streams, post_a, comb_a)
//! ffn:                 (pre_f, post_f, comb_f) = hc_mixes(streams, ffn site)
//!                      x = rmsnorm(collapse(streams, pre_a))
//!                      streams = hc_post(moe(x), streams, post_f, comb_f)
//!                      pre_prev = pre_f
//! last layer only:     hidden = collapse(streams, pre_prev)
//! ```
//!
//! `pre_prev` is set by layer 0 to the one-hot on stream 0 (the reference's
//! initial pre-mix) and lives in [`V41Runtime`], with the shared attention
//! slots, the engram hasher and the expert cache: attention state is shared
//! across layers, so one runtime is shared by every layer through an `Arc`,
//! its mutable parts behind mutexes. One sequence at a time: multi-sequence
//! decode and the model-level CUDA graph are declined through the layer
//! hooks. The layer can capture its own single-token step instead
//! (`METRALE_DS41_GRAPH=1`, see [`GraphMode`], or the multi-layer segments of
//! `METRALE_DS41_STEP_GRAPH=1`): the host work in the middle of every layer
//! (the routing download and the expert fetch; on kv and index source layers
//! also the compressor group and the index top-k) splits the step into graph
//! segments with host spans between them.
//!
//! Routed experts are read on demand into an expert cache (`ExpertLru` over
//! the `ExpertArena` slots, `ExpertSliceMap` locating each expert in the GGUF
//! shards) with positional file reads. The arena is device memory behind a
//! page-locked staging ring, or with `METRALE_DS41_ARENA_DEVICE=0` a
//! page-locked arena the GPU reads in place. The engram tables are read by
//! row (`EngramRowReader`).
//!
//! Owner: model-arch, DeepSeek-V4.1.
//! Invariants:
//! - Layer 0's `step` at position 0 resets the shared attention slots, the
//!   engram hasher and the cached step hashes after its argument checks and
//!   before any engram, attention or MoE work.

use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex, OnceLock};

use anyhow::Result;

use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, GraphHandle, KernelHandle};
use metrale_model_weights::weights::expert_stream::{
    EngramRowReader, ExpertArena, ExpertLru, ExpertSliceMap,
};

use crate::attn_v41::{
    AttnV41, AttnV41Cfg, AttnV41LayerState, AttnV41LayerWeights, LayerRole, SharedV41,
};
use crate::engram_v41::{EngramHashTables, EngramHasher, EngramV41};
use crate::moe_v41::{MoeV41, MoeV41Cfg, MoeV41LayerWeights, RouterWeights};
use metrale_model_layers::layer::LayerState;
use metrale_model_layers::layers::qwen3_attention::HcSiteWeights;
use metrale_model_layers::weight_map::DenseWeight;

mod graph;
mod hc_launch;
mod step;
mod step_ffn;
mod step_seg;
mod step_seg_run;
mod trait_impl;

pub use step_seg::{SegState, step_graph_on};

/// 2026-09-25: Everything the layers share for the one sequence in flight.
pub struct V41Runtime {
    pub attn_cfg: AttnV41Cfg,
    pub moe_cfg: MoeV41Cfg,
    pub attn: Mutex<AttnV41>,
    pub moe: Mutex<MoeV41>,
    pub engram: Mutex<EngramV41>,
    pub lru: Mutex<ExpertLru>,
    pub arena: ExpertArena,
    pub slices: Arc<ExpertSliceMap>,
    pub rows: EngramRowReader,
    pub tables: Arc<EngramHashTables>,
    pub hasher: Mutex<EngramHasher>,
    pub shared: Mutex<SharedV41>,
    /// 2026-09-25: The step's engram hashes, computed by the first engram layer
    /// (hash index 0) or the segment-graph preload, and reused by the others.
    pub step_hashes: Mutex<Option<Vec<i64>>>,
    /// 2026-09-25: The delayed `pre` mix, f32 `[max_tokens, hc]`.
    pub pre_prev: DevicePtr,
    /// 2026-09-25: f32 `[max_tokens, (2 + hc) * hc]` scratch between
    /// `hc_v41_mixes_dot` and `hc_v41_finish_collapse`.
    pub mixes_s: DevicePtr,
    pub reader_threads: usize,
    pub n_layers: usize,
    pub hc_mult: usize,
    pub hidden: usize,
    pub sinkhorn_iters: usize,
    pub hc_eps: f32,
    pub norm_eps: f32,
    pub pre_a: DevicePtr,
    pub pre_f: DevicePtr,
    pub post_s: DevicePtr,
    pub comb_s: DevicePtr,
    pub attn_in: DevicePtr,
    pub max_tokens: usize,
    /// 2026-09-25: Per-step totals across layers, logged by the last layer when
    /// `METRALE_DS41_DIAG=1`.
    pub step_moe: Mutex<crate::moe_v41::MoeV41Timing>,
    pub step_attn_ms: Mutex<f64>,
    pub step_engram_ms: Mutex<f64>,
    pub step_start: Mutex<Option<std::time::Instant>>,
    /// 2026-09-25: Set when a `METRALE_DS41_GRAPH` capture failed: segments
    /// without a graph run eagerly from then on; captured graphs keep replaying.
    pub graph_disabled: AtomicBool,
    /// 2026-09-25: The multi-layer segment graphs (`METRALE_DS41_STEP_GRAPH=1`,
    /// `step_seg.rs`).
    pub seg: Mutex<SegState>,
    /// 2026-09-25: Every layer's role, filled as the layers are built; a segment
    /// owner reads the roles of the layers its graph spans.
    pub roles: Mutex<Vec<Option<LayerRole>>>,
    /// 2026-09-25: `(layer, hash index)` of the engram layers.
    pub engram_layers: Mutex<Vec<(usize, usize)>>,
    /// 2026-09-25: `METRALE_DS41_PREDICT_TRACE`: the last three layers' MoE
    /// inputs (bf16 `[3, hidden]`, slot `layer % 3`) and the trace file.
    pub pred_x: DevicePtr,
    pub pred_trace: Mutex<Option<std::io::BufWriter<std::fs::File>>>,
}

/// 2026-09-25: How the single-token step runs, read once.
/// `METRALE_DS41_GRAPH_ORACLE=1` runs every layer both ways and logs whether
/// the highway, the delayed mix and the hidden match bit for bit;
/// `METRALE_DS41_GRAPH=1` replays the captured segments; anything else is the
/// eager step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GraphMode {
    Off,
    On,
    Oracle,
}

pub fn graph_mode() -> GraphMode {
    static MODE: OnceLock<GraphMode> = OnceLock::new();
    *MODE.get_or_init(|| {
        let is_one = |k: &str| std::env::var(k).is_ok_and(|v| v == "1");
        if is_one("METRALE_DS41_GRAPH_ORACLE") {
            GraphMode::Oracle
        } else if is_one("METRALE_DS41_GRAPH") {
            GraphMode::On
        } else {
            GraphMode::Off
        }
    })
}

/// 2026-09-25: The pointers a layer's captured segments bake that are not the
/// runtime's own. Compared on every step: a different set destroys the
/// graphs, which are then captured again.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Baked {
    hidden: DevicePtr,
    streams: DevicePtr,
    normed: DevicePtr,
    window: DevicePtr,
    rows_b: Option<DevicePtr>,
    attn_out: DevicePtr,
    moe_out: DevicePtr,
}

/// 2026-09-25: One layer's captured single-token step, held in the sequence's
/// layer state (it bakes the sequence's window ring and the kv source's cache).
///
/// Segment A = the attention site's mixes, collapse and norm, the attention
/// (capturable layers only), `hc_post`, the ffn site's mixes, collapse and
/// norm, the router GEMV. On kv and index source layers the attention runs
/// eagerly between `a[0]` (through the norm) and `a[1]` (from `hc_post`).
/// Host span: `MoeV41::stage_m1`. Segment B = the expert compute, `hc_post`,
/// the delayed-mix copy, and on the last layer the final collapse.
struct LayerGraphs {
    a: [Option<GraphHandle>; 2],
    b: Option<GraphHandle>,
    baked: Baked,
}

impl LayerGraphs {
    fn destroy(self, gpu: &dyn GpuBackend) -> Result<()> {
        for g in self.a.into_iter().chain([self.b]).flatten() {
            gpu.destroy_graph(g)?;
        }
        Ok(())
    }
}

// 2026-09-25: SAFETY: the raw device pointers are allocated at load and not
// reassigned behind the `Arc`; the mutable state is behind the mutexes, and
// one sequence runs at a time.
unsafe impl Send for V41Runtime {}
unsafe impl Sync for V41Runtime {}

pub struct V41LayerState {
    pub attn: AttnV41LayerState,
    graphs: Option<LayerGraphs>,
}

impl LayerState for V41LayerState {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

pub struct DeepSeekV41Layer {
    pub idx: usize,
    pub role: LayerRole,
    pub rt: Arc<V41Runtime>,
    pub attn_w: AttnV41LayerWeights,
    pub moe_w: MoeV41LayerWeights,
    /// 2026-09-25: The next layer's router. With expert prefetch armed
    /// (`prefetch_k() > 0` and the reader pool), `MoeV41::forward` /
    /// `stage_m1` run it on this layer's MoE input to start reading the
    /// experts the next layer is predicted to ask for.
    pub next_router: Option<RouterWeights>,
    /// 2026-09-25: `Some(hash index)` on engram layers (`EngramHashTables::hash_index`).
    pub engram_index: Option<usize>,
    pub hc_attn: HcSiteWeights,
    pub hc_ffn: HcSiteWeights,
    pub attn_norm: DenseWeight,
    pub ffn_norm: DenseWeight,
    pub k_hc_expand: KernelHandle,
    pub k_hc_post: KernelHandle,
    pub k_mixes_dot: KernelHandle,
    pub k_mixes_finish: KernelHandle,
    pub k_collapse: KernelHandle,
    /// 2026-09-25: `hc_v41_finish_collapse`: a site's mix finish and the
    /// block's collapse in one launch over ceil(H / 256) blocks a token.
    pub k_finish_collapse: KernelHandle,
    /// 2026-09-25: `hc_v41_collapse_wide` and `hc_v41_post_wide`: the collapse
    /// and `hc_post` over ceil(H / 256) blocks a token.
    pub k_collapse_wide: KernelHandle,
    pub k_post_wide: KernelHandle,
    pub k_rms_norm: KernelHandle,
}
