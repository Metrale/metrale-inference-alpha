// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The recording fake's device token feed: the `Model` methods the asynchronous router calls, recorded and answered like every other model call.
//!
//! Owner: scheduler.
//! Invariants: none beyond the types.
//!
//! The feed cells are a plain `Vec<u32>` in the shared state and the
//! "pinned" slot is heap memory. An event completes on its first query
//! unless `ModelCfg::event_lag` or `reversed_completion` holds it back, and
//! the n-th query can be scripted to fail (`event_query_fault_at`).

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;
use metrale_model_engine::traits::{FeedSource, RowMask, SequenceState};

use super::model::{RecordingModel, fmt_ids};
use super::model_forward::sid;

macro_rules! rec {
    ($m:expr, $($arg:tt)*) => { $m.rec(format!($($arg)*)) };
}

impl RecordingModel {
    fn feed_cell(&self, row: u32) -> u32 {
        let st = self.state.lock().unwrap();
        st.feed_cells.get(row as usize).copied().unwrap_or_else(|| {
            panic!(
                "feed cell {row} read, but the last feed argmax wrote {}",
                st.feed_cells.len()
            )
        })
    }

    /// 2026-09-25: Grow `seq.block_table` to what the sync decode of the same step would
    /// leave it at (`blocks_for(seq_len + 1)`), returning the blocks added.
    fn grow_for_next_position(&self, seq: &mut SequenceState) -> usize {
        let want = self.blocks_for(seq.seq_len + 1);
        let have = seq.block_table.len();
        if want > have {
            self.take_blocks(want - have);
            seq.block_table.extend((have..want).map(|b| b as u32));
        }
        want.saturating_sub(have)
    }

    pub(super) fn fx_decode_batch_fed(
        &self,
        sources: &[FeedSource],
        seqs: &mut [&mut SequenceState],
        st: u64,
    ) -> Result<DevicePtr> {
        let call = self.next_decode_batch_call();
        let inputs: Vec<u32> = sources
            .iter()
            .map(|s| match *s {
                FeedSource::Feed { from_row } => self.feed_cell(from_row),
                FeedSource::Host(id) => id,
            })
            .collect();
        let ids: Vec<String> = seqs.iter().map(|s| sid(s)).collect();
        let head = format!(
            "decode_batch_fed(inputs={}, seqs={ids:?}, stream={st})",
            fmt_ids(&inputs)
        );
        if self.cfg.kv_exhaust_at.contains(&call) {
            rec!(self, "{head} -> ERR KV cache exhausted");
            anyhow::bail!("KV cache exhausted (scripted at decode_batch #{call})");
        }
        let mut rows = Vec::with_capacity(seqs.len());
        for s in seqs.iter_mut() {
            self.maybe_cancel(s);
            // 2026-09-25: the input sits at position `seq_len`; the
            // script's next token follows it. No push: per the
            // `decode_batch_fed` contract the core applies that
            // bookkeeping.
            self.grow_for_next_position(s);
            rows.push(self.out_after(s, s.seq_len));
        }
        self.set_rows(rows);
        self.rec(head);
        Ok(self.row_ptr(0))
    }

    pub(super) fn fx_argmax_batch_to_feed(
        &self,
        logits_ptr: DevicePtr,
        masks: &[RowMask],
        dst: *mut u32,
        event: u64,
        st: u64,
    ) -> Result<()> {
        let (row, _) = self.locate(logits_ptr);
        let vocab = self.cfg.vocab as u32;
        let out: Vec<u32> = masks
            .iter()
            .enumerate()
            .map(|(i, mask)| {
                // 2026-09-25: the fake's logits are one-hot at the
                // scripted token: the pick is that token, and when a
                // row mask hits it the pick is the highest unmasked id.
                let hit = self.row(row + i);
                if hit == mask[0] || hit == mask[1] {
                    (0..vocab)
                        .rev()
                        .find(|t| *t != mask[0] && *t != mask[1])
                        .unwrap_or(0)
                } else {
                    hit
                }
            })
            .collect();
        self.state.lock().unwrap().feed_cells = out.clone();
        // 2026-09-25: per the trait contract, `event` is recorded after the D2H.
        self.fx_record_event(event);
        // 2026-09-25: SAFETY: `dst` is one of the router's readback ring
        // slots, `max_rows` cells wide, and the router refuses more than
        // `max_rows` masks; the ring lives until the router's teardown.
        let slot = unsafe { std::slice::from_raw_parts_mut(dst, out.len()) };
        slot.copy_from_slice(&out);
        rec!(
            self,
            "argmax_batch_to_feed(row={row}, n={}, masks={masks:?}, event={event}, stream={st}) -> {out:?}",
            masks.len()
        );
        Ok(())
    }

    pub(super) fn fx_reserve_decode_block(&self, seq: &mut SequenceState) -> Result<usize> {
        if self.free_blocks() == 0 && self.blocks_for(seq.seq_len + 1) > seq.block_table.len() {
            rec!(self, "reserve_decode_block({}) -> ERR exhausted", sid(seq));
            anyhow::bail!("KV cache exhausted: no free blocks (scripted reserve)");
        }
        let added = self.grow_for_next_position(seq);
        rec!(self, "reserve_decode_block({}) -> {added}", sid(seq));
        Ok(added)
    }

    pub(super) fn fx_release_decode_blocks(
        &self,
        seq: &mut SequenceState,
        blocks: usize,
    ) -> Result<()> {
        let n = blocks.min(seq.block_table.len());
        seq.block_table.truncate(seq.block_table.len() - n);
        self.give_blocks(n);
        rec!(self, "release_decode_blocks({}, n={blocks})", sid(seq));
        Ok(())
    }
}

// 2026-09-25: events, which the router polls before it reads a readback slot.

impl RecordingModel {
    pub(super) fn fx_create_event(&self) -> u64 {
        let mut st = self.state.lock().unwrap();
        st.events_created += 1;
        st.events_created + 1
    }

    pub(super) fn fx_record_event(&self, event: u64) {
        let mut st = self.state.lock().unwrap();
        st.pending_events.retain(|(e, _)| *e != event);
        st.pending_events.push((event, 0));
    }

    /// 2026-09-25: Complete `event` and everything recorded before it.
    pub(super) fn fx_event_synchronize(&self, event: u64) {
        let mut st = self.state.lock().unwrap();
        if let Some(pos) = st.pending_events.iter().position(|(e, _)| *e == event) {
            st.pending_events.drain(..=pos);
        }
    }

    pub(super) fn fx_event_query(&self, event: u64) -> Result<bool> {
        let mut st = self.state.lock().unwrap();
        st.event_queries += 1;
        if self.cfg.event_query_fault_at == Some(st.event_queries) {
            drop(st);
            rec!(
                self,
                "event_query(event={event}) -> ERR scripted device fault"
            );
            anyhow::bail!("scripted device fault");
        }
        let Some(pos) = st.pending_events.iter().position(|(e, _)| *e == event) else {
            return Ok(true);
        };
        st.pending_events[pos].1 += 1;
        let lagging = st.pending_events[pos].1 <= self.cfg.event_lag;
        let newer_pending = self.cfg.reversed_completion && pos + 1 < st.pending_events.len();
        if lagging || newer_pending {
            return Ok(false);
        }
        st.pending_events.remove(pos);
        Ok(true)
    }
}
