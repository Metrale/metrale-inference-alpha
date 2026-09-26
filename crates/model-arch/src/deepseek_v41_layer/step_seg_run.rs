// SPDX-License-Identifier: MIT OR Apache-2.0
// provenance-id: 526f6e616c6420522e205374657369616b

//! 2026-09-25: The segment owner's work around a graph: the engram preload,
//! the device-only layer bodies recorded under capture, the per-layer
//! snapshot nodes and their restore, the launch with the header read and the
//! eager fallbacks.
//!
//! Owner: model-arch, DeepSeek-V4.1.
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, GraphHandle};

use super::step_seg::{SegState, step_log_on};
use super::{DeepSeekV41Layer, V41LayerState};
use crate::engram_v41::ENGRAM_ROW_BYTES;
use crate::moe_v41::MoeV41;
use metrale_model_layers::layer::ForwardContext;

/// 2026-09-25: The buffers a layer's snapshot holds: streams, the attention
/// output, `pre_a`, `post_s`, `comb_s` (what the ffn half reads before
/// writing; it writes `hidden` and `pre_prev` before reading them).
pub(super) const SAVE_N: usize = 5;

/// 2026-09-25: Their sizes in bytes for one token.
fn save_bytes(h: usize, hc: usize) -> [usize; SAVE_N] {
    [hc * h * 4, h * 2, hc * 4, hc * 4, hc * hc * 4]
}

impl DeepSeekV41Layer {
    /// 2026-09-25: Hash this token and upload every engram layer's rows before
    /// any graph (nothing with `METRALE_DS41_NO_ENGRAM=1`).
    pub(super) fn engram_preload(
        &self,
        ctx: &ForwardContext,
        start_pos: usize,
        stream: u64,
    ) -> Result<()> {
        let rt = &self.rt;
        if std::env::var("METRALE_DS41_NO_ENGRAM").is_ok_and(|v| v == "1") {
            return Ok(());
        }
        let ids = self.step_token_ids(ctx, 1)?;
        let mut hasher = rt.hasher.lock().unwrap();
        let hashes = hasher.hash(&ids, start_pos)?;
        let engram = rt.engram.lock().unwrap();
        for &(layer, hi) in rt.engram_layers.lock().unwrap().iter() {
            let row_ids = hasher.layer_row_ids(&hashes, 1, hi);
            let mut raw = vec![0u8; row_ids.len() * ENGRAM_ROW_BYTES];
            rt.rows.read_rows(layer, &row_ids, &mut raw)?;
            engram.rows_from_q2k(ctx.gpu, Some(layer), &raw, row_ids.len(), stream)?;
        }
        *rt.step_hashes.lock().unwrap() = Some(hashes);
        Ok(())
    }

    /// 2026-09-25: This layer's device-only attention body under a capture:
    /// the engram on the preloaded rows, the attention input, `decode_body`.
    pub(super) fn body_attn(
        &self,
        gpu: &dyn GpuBackend,
        st: &V41LayerState,
        hidden: DevicePtr,
        streams: DevicePtr,
        normed: DevicePtr,
        rows_b: Option<DevicePtr>,
        stream: u64,
    ) -> Result<()> {
        let rt = &self.rt;
        if self.engram_index.is_some()
            && !std::env::var("METRALE_DS41_NO_ENGRAM").is_ok_and(|v| v == "1")
        {
            rt.engram
                .lock()
                .unwrap()
                .apply(gpu, self.idx, streams, 1, stream)?;
        }
        self.seg_attn_in(gpu, streams, hidden, normed, stream)?;
        let attn = rt.attn.lock().unwrap();
        attn.decode_body(gpu, &self.attn_w, &st.attn, normed, rows_b, stream)?;
        Ok(())
    }

    /// 2026-09-25: This layer's device-only ffn body: the attention's
    /// `hc_post`, the ffn mixes and norm, the router, the shared expert
    /// forked onto the capture's side stream, the device selection (header
    /// deferred), the routed experts, the join, the post mix, the
    /// delayed-mix copy and on the last layer the final collapse.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn body_ffn(
        &self,
        gpu: &dyn GpuBackend,
        moe: &MoeV41,
        arena: (u64, u64, u64, u64, u64),
        (side, ev_fork, ev_join): (u64, u64, u64),
        hidden: DevicePtr,
        streams: DevicePtr,
        normed: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let rt = &self.rt;
        let attn_out = rt.attn.lock().unwrap().out_ptr();
        self.seg_ffn_in(gpu, moe, attn_out, streams, hidden, normed, stream)?;
        // 2026-09-25: The token's q8_1 rows once, read by the side and the
        // main stream.
        moe.ffn_input_q8_m1(gpu, normed, stream)?;
        gpu.record_event(ev_fork, stream)?;
        gpu.stream_wait_event(side, ev_fork)?;
        moe.shared_expert_on(gpu, &self.moe_w, normed, side)?;
        gpu.record_event(ev_join, side)?;
        moe.route_select_deferred(gpu, &self.moe_w, arena, stream)?;
        let moe_out = moe.compute_m1_joined(gpu, normed, moe.cfg.topk, ev_join, stream)?;
        self.hc_post(gpu, moe_out, streams, rt.post_s, rt.comb_s, 1, stream)?;
        gpu.copy_d2d_async(rt.pre_f, rt.pre_prev, rt.hc_mult * 4, stream)?;
        if self.idx + 1 == rt.n_layers {
            // 2026-09-25: The final collapse uses the last ffn `pre` (`pre_prev`).
            self.collapse(gpu, streams, rt.pre_prev, hidden, 1, stream)?;
        }
        Ok(())
    }

    /// 2026-09-25: The snapshot before a layer's ffn, into its slot: the copy
    /// nodes a graph carries for every layer it covers. The streams, the
    /// attention output, `pre_a`, and the attention site's `post_s` /
    /// `comb_s` (its `hc_post` mixes, overwritten by every later site).
    pub(super) fn save_nodes(
        &self,
        gpu: &dyn GpuBackend,
        save: &[DevicePtr; SAVE_N],
        hidden: DevicePtr,
        streams: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let rt = &self.rt;
        let (h, hc) = (rt.hidden, rt.hc_mult);
        let attn_out = rt.attn.lock().unwrap().out_ptr();
        let _ = hidden;
        let src = [streams, attn_out, rt.pre_a, rt.post_s, rt.comb_s];
        for i in 0..SAVE_N {
            gpu.copy_d2d_async(src[i], save[i], save_bytes(h, hc)[i], stream)?;
        }
        Ok(())
    }

    /// 2026-09-25: A layer's snapshot back: the state before its ffn.
    pub(super) fn restore(
        &self,
        gpu: &dyn GpuBackend,
        save: &[DevicePtr; SAVE_N],
        hidden: DevicePtr,
        streams: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let rt = &self.rt;
        let (h, hc) = (rt.hidden, rt.hc_mult);
        let attn_out = rt.attn.lock().unwrap().out_ptr();
        let _ = hidden;
        let dst = [streams, attn_out, rt.pre_a, rt.post_s, rt.comb_s];
        for i in 0..SAVE_N {
            gpu.copy_d2d_async(save[i], dst[i], save_bytes(h, hc)[i], stream)?;
        }
        Ok(())
    }

    /// 2026-09-25: One snapshot slot a layer, allocated when `SegState::save`
    /// does not have one per layer.
    pub(super) fn alloc_save(&self, gpu: &dyn GpuBackend, seg: &mut SegState) -> Result<()> {
        let rt = &self.rt;
        let (h, hc) = (rt.hidden, rt.hc_mult);
        let n = save_bytes(h, hc);
        let mut save = Vec::with_capacity(rt.n_layers);
        for _ in 0..rt.n_layers {
            let mut slot = [DevicePtr(0); SAVE_N];
            for (i, p) in slot.iter_mut().enumerate() {
                *p = gpu.alloc(n[i].max(16))?;
            }
            save.push(slot);
        }
        seg.save = save;
        Ok(())
    }

    /// 2026-09-25: This layer's engram, on the rows layer 0 uploaded for the step.
    pub(super) fn engram_apply(
        &self,
        gpu: &dyn GpuBackend,
        streams: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        if self.engram_index.is_some()
            && !std::env::var("METRALE_DS41_NO_ENGRAM").is_ok_and(|v| v == "1")
        {
            self.rt
                .engram
                .lock()
                .unwrap()
                .apply(gpu, self.idx, streams, 1, stream)?;
        }
        Ok(())
    }

    /// 2026-09-25: The whole eager step of this layer inside a segment-graph
    /// token: the engram on the preloaded rows, then the eager step.
    pub(super) fn eager_layer(
        &self,
        hidden: DevicePtr,
        start_pos: usize,
        st: &mut V41LayerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.engram_apply(ctx.gpu, ctx.buffers.hc_streams(), stream)?;
        self.step_eager(hidden, 1, start_pos, st, ctx, stream)
    }

    /// 2026-09-25: Launch a segment once, read its layers' headers in order,
    /// touch the picks in the cache (fetching the misses) and bring the slot
    /// table up to date. Returns the first layer that flagged an absent expert
    /// (or whose fetch moved a slot): its picks are real, its output and every
    /// later layer's are not (an absent expert read slot 0, the later layers
    /// routed on a wrong input), so the caller restores that layer's snapshot
    /// and runs from its ffn to the segment's end eagerly. The later layers'
    /// picks are not fetched.
    pub(super) fn launch_once(
        &self,
        gpu: &dyn GpuBackend,
        g: GraphHandle,
        moe_layers: &[u32],
        stream: u64,
    ) -> Result<Option<usize>> {
        let rt = &self.rt;
        let moe = rt.moe.lock().unwrap();
        gpu.launch_graph(g, stream)?;
        let hdr = moe.read_headers(gpu, stream)?;
        let mut lru = rt.lru.lock().unwrap();
        lru.begin_token();
        let misses0 = lru.stats().misses;
        let mut first: Option<usize> = None;
        let mut changes: Vec<(u32, u32, i32)> = Vec::new();
        for &l in moe_layers {
            let (flag, picks, _) = moe.parse_header(&hdr, l)?;
            let keys: Vec<(u32, u32)> = picks.iter().map(|&e| (l, e as u32)).collect();
            lru.fetch_many_on(gpu, stream, &*rt.slices, &keys, &[], rt.reader_threads)?;
            changes.extend(lru.drain_slot_changes());
            if flag || !changes.is_empty() {
                first = Some(l as usize);
                break;
            }
        }
        moe.slot_table_update(gpu, &changes, stream)?;
        let misses = lru.stats().misses - misses0;
        drop(lru);
        if step_log_on()
            && let Some(k) = first
        {
            tracing::info!(
                "DS41 step graph: segment {} missed at layer {k} ({misses} fetched): layers {k}.. run eagerly this token",
                self.idx
            );
        }
        Ok(first)
    }

    /// 2026-09-25: The owner's preparation before a launch or a capture: the
    /// engram rows (layer 0), the eager attention (a kv/index source layer),
    /// the selection uploads for the classes the segment reads, the slot table.
    pub(super) fn prepare(
        &self,
        gpu: &dyn GpuBackend,
        st: &mut V41LayerState,
        ctx: &ForwardContext,
        start_pos: usize,
        classes: (bool, bool),
        normed: DevicePtr,
        stream: u64,
    ) -> Result<Option<DevicePtr>> {
        let rt = &self.rt;
        if self.idx == 0 {
            self.engram_preload(ctx, start_pos, stream)?;
        } else {
            // 2026-09-25: A kv/index source: its attention is host work, run
            // eagerly on the normed input already in `normed`; it writes pos /
            // idx_dev for itself, so the captured uploads are stale after it.
            let mut attn = rt.attn.lock().unwrap();
            let mut shared = rt.shared.lock().unwrap();
            attn.forward(
                gpu,
                &self.attn_w,
                &mut st.attn,
                &mut shared,
                normed,
                1,
                start_pos,
                stream,
            )?;
            attn.invalidate_decode_uploads();
        }
        let mut attn = rt.attn.lock().unwrap();
        let shared = rt.shared.lock().unwrap();
        let mut rows_b = None;
        if classes.0 {
            rows_b = attn.decode_prep_role(true, &shared, gpu, start_pos, stream)?;
        }
        if classes.1 {
            attn.decode_prep_role(false, &shared, gpu, start_pos, stream)?;
        }
        drop(shared);
        drop(attn);
        let pending = rt.lru.lock().unwrap().drain_slot_changes();
        rt.moe
            .lock()
            .unwrap()
            .slot_table_update(gpu, &pending, stream)?;
        Ok(rows_b)
    }
}
