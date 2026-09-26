// SPDX-License-Identifier: MIT OR Apache-2.0
// provenance-id: 526f6e616c6420522e205374657369616b

//! 2026-09-25: The single-token step as a few CUDA graphs spanning layers
//! (`METRALE_DS41_STEP_GRAPH=1`, which needs `METRALE_DS41_DEVICE_ROUTE=1`).
//!
//! With the routing on the device and every engram layer's rows uploaded up
//! front, the host work left in a decode step is the attention of the kv and
//! index source layers (the index score read-back and the host top-k, plus
//! the compressor's position parity). The step is captured as segments
//! between those layers: layer 0's segment runs to the first source layer's
//! attention input; a source layer's segment runs from its ffn to the next
//! source layer's attention input; the last runs to the final collapse. Each
//! launch is followed by one read-back of the routing headers of the
//! segment's layers.
//!
//! The protocol. A segment's graph copies a snapshot before every layer's
//! ffn (`step_seg_run::SAVE_N` buffers), and a segment is launched only at
//! its owner's step, never at its close. The owner reads its layers' routing
//! headers once after the launch. Inside a replay nothing fetches: the
//! selection kernel points an absent expert at slot 0 and flags the layer,
//! so every layer after it routed on a wrong input. The first flagged layer
//! k (or the first whose fetch moved a cache slot) has real picks: its
//! misses are fetched, its snapshot restored, layer k runs from its ffn on
//! with the eager step's code (`ffn_eager`), the layers after it to the
//! segment's end run the whole eager step, and the layers before k stand.
//! The capture token records the bodies on a second stream (the capture is
//! relaxed) while every layer runs eagerly on the compute stream, so the
//! capture token's result is the eager path's and the first replay is the
//! next token. An owner runs its own attention input eagerly whenever the
//! segment before it did not replay through to it. A failed capture turns
//! the mode off for the process; the token in flight finishes eagerly.
//!
//! In this mode every layer's `step` lands here; the owner of a segment
//! (layer 0, or a kv/index source layer) prepares, launches and runs the
//! protocol, and the layers a replayed segment covered return at once
//! (`skip_until`).
//!
//! Owner: model-arch, DeepSeek-V4.1.
//! Invariants: none beyond the types.

use std::sync::OnceLock;

use anyhow::{Context, Result, ensure};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, GraphHandle};

use super::step::{trace_hash, trace_on};
use super::step_seg_run::SAVE_N;
use super::{DeepSeekV41Layer, V41LayerState};
use crate::attn_v41::LayerRole;
use crate::moe_v41::MoeV41;
use metrale_model_layers::layer::ForwardContext;

/// 2026-09-25: `METRALE_DS41_STEP_GRAPH=1`, read once.
pub fn step_graph_on() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var("METRALE_DS41_STEP_GRAPH").is_ok_and(|v| v == "1"))
}

/// 2026-09-25: One captured segment. `end` is the layer whose attention input
/// closes it (`n_layers` for the last one); layers below it are covered.
/// `moe_layers` ran their MoE inside it (their headers are read after a
/// launch). `classes` are the selection classes (ratio > 0, ratio 0) its
/// attention bodies read, and `rows_b` the compressed rows they bake.
pub struct SegGraph {
    pub(super) g: GraphHandle,
    pub(super) end: usize,
    pub(super) moe_layers: Vec<u32>,
    pub(super) classes: (bool, bool),
    pub(super) rows_b: Option<DevicePtr>,
}

/// 2026-09-25: An open capture: the owner and the MoE layers appended so far.
pub struct SegCapture {
    pub(super) owner: usize,
    pub(super) moe_layers: Vec<u32>,
    pub(super) classes: (bool, bool),
    pub(super) rows_b: Option<DevicePtr>,
}

/// 2026-09-25: The segment-graph state shared by every layer.
/// - `disabled`: a capture failed; the mode is off for the process.
/// - `skip_until`: layers below it were covered by the segment just launched.
/// - `eager_until`: layers below it run the whole eager step this token.
/// - `ffn_only_at`: the layer whose snapshot the owner restored; it runs from
///   its ffn on.
/// - `graphs`: by owner layer.
/// - `baked`: the buffers every graph bakes (hidden, streams, normed).
/// - `cap_stream`, `cap_side`, `ev_fork`, `ev_join`: the stream a capture
///   token records on, the side stream the recorded shared expert forks
///   onto, and the fork / join events.
/// - `save`: per layer, the snapshot before its ffn (`step_seg_run::SAVE_N`
///   buffers: streams, the attention output, `pre_a`, `post_s`, `comb_s`).
/// - `windows`: every layer's window ring as last seen (`note_window`), which
///   the captured attention bodies bake.
/// - `stale`: a window ring changed since the graphs were captured; layer 0
///   destroys them all.
/// - `launches`, `reruns`: graph launches and miss falls this process, for
///   the log.
#[derive(Default)]
pub struct SegState {
    pub disabled: bool,
    pub skip_until: usize,
    pub eager_until: usize,
    pub ffn_only_at: Option<usize>,
    pub capturing: Option<SegCapture>,
    pub graphs: Vec<Option<SegGraph>>,
    pub baked: Option<[DevicePtr; 3]>,
    pub cap_stream: Option<u64>,
    pub cap_side: u64,
    pub ev_fork: u64,
    pub ev_join: u64,
    pub save: Vec<[DevicePtr; SAVE_N]>,
    pub windows: Vec<DevicePtr>,
    pub stale: bool,
    pub launches: u64,
    pub reruns: u64,
}

impl DeepSeekV41Layer {
    fn role_of(&self, l: usize) -> Result<LayerRole> {
        self.rt.roles.lock().unwrap()[l].context("step graph: a layer without a role")
    }

    fn capturable(r: &LayerRole) -> bool {
        !r.is_kv_source && !r.is_index_source
    }

    /// 2026-09-25: The segment an owner at `s` spans: its end and the selection
    /// classes of the capturable layers in it.
    fn geometry(&self, s: usize) -> Result<(usize, (bool, bool))> {
        let n = self.rt.n_layers;
        let mut classes = (false, false);
        let mut end = n;
        for l in s..n {
            let r = self.role_of(l)?;
            if l != s && !Self::capturable(&r) {
                end = l;
                break;
            }
            if Self::capturable(&r) {
                if r.ratio > 0 {
                    classes.0 = true;
                } else {
                    classes.1 = true;
                }
            }
        }
        Ok((end, classes))
    }

    /// 2026-09-25: With `METRALE_DS41_STEP_GRAPH=1` every step (prefill too)
    /// notes this layer's window ring: the captured attention bodies bake it,
    /// and a new sequence brings new rings, so a change marks the graphs stale.
    pub(super) fn note_window(&self, window: DevicePtr) {
        let mut seg = self.rt.seg.lock().unwrap();
        let n = self.rt.n_layers;
        if seg.windows.len() != n {
            seg.windows = vec![DevicePtr(0); n];
        }
        if seg.windows[self.idx] != window {
            seg.windows[self.idx] = window;
            seg.stale = true;
        }
    }

    /// 2026-09-25: The step of one layer in segment-graph mode.
    pub(super) fn step_seg(
        &self,
        hidden: DevicePtr,
        start_pos: usize,
        st: &mut V41LayerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let rt = &self.rt;
        let gpu = ctx.gpu;
        let streams = ctx.buffers.hc_streams();
        let normed = ctx.buffers.norm_output();
        let n = rt.n_layers;
        let mut seg = rt.seg.lock().unwrap();
        if seg.graphs.len() != n {
            seg.graphs = (0..n).map(|_| None).collect();
        }
        if self.idx == 0 {
            // 2026-09-25: A new step: the baked buffers, the skip and fallback marks.
            seg.skip_until = 0;
            seg.eager_until = 0;
            seg.ffn_only_at = None;
            let baked = [hidden, streams, normed];
            if seg.baked != Some(baked) || seg.stale {
                if seg.baked.is_some() && seg.graphs.iter().any(Option::is_some) {
                    tracing::info!(
                        "DS41 step graph: {}; recapturing every segment",
                        if seg.stale {
                            "a sequence's window rings changed"
                        } else {
                            "buffers changed"
                        }
                    );
                }
                for g in seg.graphs.iter_mut().flat_map(Option::take) {
                    gpu.destroy_graph(g.g)?;
                }
                seg.baked = Some(baked);
                seg.stale = false;
            }
            if seg.cap_stream.is_none() {
                seg.cap_stream = Some(
                    gpu.create_stream()
                        .context("step graph: the recording stream")?,
                );
                seg.cap_side = gpu.create_stream().context("step graph: the side stream")?;
                seg.ev_fork = gpu.create_event()?;
                seg.ev_join = gpu.create_event()?;
            }
            if seg.save.len() != n {
                self.alloc_save(gpu, &mut seg)?;
            }
        }
        if self.idx < seg.skip_until {
            return Ok(());
        }
        if seg.ffn_only_at == Some(self.idx) {
            // 2026-09-25: The owner restored this layer's snapshot: from the ffn on.
            seg.ffn_only_at = None;
            drop(seg);
            return self.ffn_eager(hidden, 1, start_pos, ctx, stream);
        }
        if self.idx < seg.eager_until {
            drop(seg);
            return self.eager_layer(hidden, start_pos, st, ctx, stream);
        }
        let role = self.role_of(self.idx)?;
        let owner = self.idx == 0 || !Self::capturable(&role);
        let cs = seg.cap_stream.context("step graph: no recording stream")?;

        // 2026-09-25: A layer inside an open capture: its body is recorded on
        // the recording stream while the layer runs eagerly on the compute stream.
        if let Some(mut cap) = seg.capturing.take() {
            ensure!(
                cap.owner < self.idx,
                "step graph: capture owned by a later layer"
            );
            if owner {
                // 2026-09-25: The closer: its attention input ends the segment;
                // it runs that input eagerly below, as any owner a replay did
                // not reach.
                let r = (|| -> Result<()> {
                    self.engram_apply(gpu, streams, cs)?;
                    self.seg_attn_in(gpu, streams, hidden, normed, cs)
                })();
                self.close_or_disable(gpu, &mut seg, cap, self.idx, cs, r);
            } else {
                let arena = MoeV41::arena_of(&rt.lru.lock().unwrap());
                let save = seg.save[self.idx];
                let fork = (seg.cap_side, seg.ev_fork, seg.ev_join);
                let r = (|| -> Result<()> {
                    let moe = rt.moe.lock().unwrap();
                    self.body_attn(gpu, st, hidden, streams, normed, cap.rows_b, cs)?;
                    self.save_nodes(gpu, &save, hidden, streams, cs)?;
                    self.body_ffn(gpu, &moe, arena, fork, hidden, streams, normed, cs)
                })();
                cap.moe_layers.push(self.idx as u32);
                if r.is_err() || self.idx + 1 == n {
                    self.close_or_disable(gpu, &mut seg, cap, n, cs, r);
                } else {
                    seg.capturing = Some(cap);
                }
                drop(seg);
                return self.eager_layer(hidden, start_pos, st, ctx, stream);
            }
        }
        if seg.disabled {
            drop(seg);
            return self.eager_layer(hidden, start_pos, st, ctx, stream);
        }
        ensure!(
            owner,
            "step graph: layer {} reached outside any segment",
            self.idx
        );

        // 2026-09-25: The owner: its attention input if no replay delivered
        // it, then prepare, then replay or capture the segment from here.
        if self.idx > 0 && seg.skip_until != self.idx {
            self.engram_apply(gpu, streams, stream)?;
            self.seg_attn_in(gpu, streams, hidden, normed, stream)?;
        }
        let (end, classes) = self.geometry(self.idx)?;
        let rows_b = self.prepare(gpu, st, ctx, start_pos, classes, normed, stream)?;
        if let Some(sg) = seg.graphs[self.idx].as_ref()
            && (sg.end != end || sg.classes != classes || sg.rows_b != rows_b)
        {
            tracing::warn!(
                "DS41 step graph: segment {} bakes other rows; recapturing",
                self.idx
            );
            if let Some(sg) = seg.graphs[self.idx].take() {
                gpu.destroy_graph(sg.g)?;
            }
        }
        if let Some(sg) = seg.graphs[self.idx].as_ref() {
            let (gh, layers) = (sg.g, sg.moe_layers.clone());
            let first_miss = self.launch_once(gpu, gh, &layers, stream)?;
            seg.launches += 1;
            let Some(k) = first_miss else {
                seg.skip_until = end;
                if trace_on() {
                    // 2026-09-25: `METRALE_DS41_TRACE=1`: the highway after the
                    // segment, to diff against the eager trace's line for layer end-1.
                    gpu.synchronize(stream)?;
                    tracing::info!(
                        "DS41 trace pos={start_pos} seg={}..{end} hwy={:016x} hidden={:016x}",
                        self.idx,
                        trace_hash(gpu, streams, rt.hc_mult * rt.hidden * 4),
                        trace_hash(gpu, hidden, rt.hidden * 2)
                    );
                }
                if end == n {
                    self.log_step(&seg, start_pos);
                }
                return Ok(());
            };
            // 2026-09-25: Layer k's picks were real and are resident now: its
            // snapshot back, k from its ffn on and k+1..end whole, eagerly.
            seg.reruns += 1;
            let save = seg.save[k];
            self.restore(gpu, &save, hidden, streams, stream)?;
            seg.eager_until = end;
            if k == self.idx {
                drop(seg);
                return self.ffn_eager(hidden, 1, start_pos, ctx, stream);
            }
            seg.skip_until = k;
            seg.ffn_only_at = Some(k);
            return Ok(());
        }
        // 2026-09-25: No graph yet: record from here on the recording stream
        // while this token runs eagerly on the compute stream.
        let arena = MoeV41::arena_of(&rt.lru.lock().unwrap());
        let save = seg.save[self.idx];
        let fork = (seg.cap_side, seg.ev_fork, seg.ev_join);
        let r = gpu
            .begin_capture(cs)
            .context("step graph: begin_capture")
            .and_then(|_| {
                let moe = rt.moe.lock().unwrap();
                if self.idx == 0 {
                    self.body_attn(gpu, st, hidden, streams, normed, rows_b, cs)?;
                }
                self.save_nodes(gpu, &save, hidden, streams, cs)?;
                self.body_ffn(gpu, &moe, arena, fork, hidden, streams, normed, cs)
            });
        let cap = SegCapture {
            owner: self.idx,
            moe_layers: vec![self.idx as u32],
            classes,
            rows_b,
        };
        if r.is_err() || self.idx + 1 == n {
            self.close_or_disable(gpu, &mut seg, cap, n, cs, r);
        } else {
            seg.capturing = Some(cap);
        }
        drop(seg);
        if self.idx == 0 {
            self.eager_layer(hidden, start_pos, st, ctx, stream)
        } else {
            // 2026-09-25: A source-layer owner: its attention ran eagerly in
            // `prepare`.
            self.ffn_eager(hidden, 1, start_pos, ctx, stream)
        }
    }

    /// 2026-09-25: End the open capture as the segment of `cap.owner` ending at
    /// `end`, or, when the recording failed (`r`) or the capture cannot end,
    /// turn the mode off: the token in flight finishes eagerly.
    fn close_or_disable(
        &self,
        gpu: &dyn GpuBackend,
        seg: &mut SegState,
        cap: SegCapture,
        end: usize,
        cs: u64,
        r: Result<()>,
    ) {
        match r.and_then(|_| gpu.end_capture(cs).context("step graph: end_capture")) {
            Ok(g) => {
                tracing::info!(
                    "DS41 step graph: segment {} = layers {}..{} captured ({} MoE layers)",
                    cap.owner,
                    cap.owner,
                    end,
                    cap.moe_layers.len()
                );
                seg.graphs[cap.owner] = Some(SegGraph {
                    g,
                    end,
                    moe_layers: cap.moe_layers,
                    classes: cap.classes,
                    rows_b: cap.rows_b,
                });
            }
            Err(e) => {
                gpu.abort_capture_if_active(cs);
                seg.disabled = true;
                seg.capturing = None;
                seg.eager_until = self.rt.n_layers;
                tracing::warn!(
                    "DS41 L{}: step graph capture failed ({e:#}); the mode is off, the step runs eagerly",
                    self.idx
                );
            }
        }
    }

    fn log_step(&self, seg: &SegState, start_pos: usize) {
        if step_log_on() {
            tracing::info!(
                "DS41 step graph pos {start_pos}: {} launches, {} miss falls so far",
                seg.launches,
                seg.reruns
            );
        }
    }
}

/// 2026-09-25: `METRALE_DS41_STEP_GRAPH_LOG=1`, read once: the running counts
/// whenever the last segment replays without a miss, and a line per segment miss.
pub(super) fn step_log_on() -> bool {
    static V: OnceLock<bool> = OnceLock::new();
    *V.get_or_init(|| std::env::var("METRALE_DS41_STEP_GRAPH_LOG").is_ok_and(|v| v == "1"))
}
