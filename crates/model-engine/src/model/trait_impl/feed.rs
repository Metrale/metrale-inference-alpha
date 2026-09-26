// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The device token feed: the `Model` half of the asynchronous scheduler router.
//!
//! A fed decode step is the plain decode step with two changes, both before the CUDA-graph
//! region. Its input ids are resolved on the device (`feed_resolve` over the previous step's
//! argmax cells) and gathered into `hidden` by `batched_embed`, not copied row by row from a
//! host id; the gather reads the same embedding rows and applies the same scaling. And the
//! host bookkeeping (`tokens.push`, `seq_len += 1`) is left to the core, which learns the id
//! only once the previous step is committed.
//!
//! The readback side is `argmax_batch_to_feed`: the masked batched argmax of
//! `argmax_feed.cu` writes the cells, an async D2H lands the ids in the router's pinned slot,
//! and an event marks completion. It does not synchronise a stream.
//!
//! Owner: model-engine decode.
//! Invariants:
//! - A fed decode step does not push to `seq.tokens` or advance `seq_len`.

use anyhow::{Result, bail};
use metrale_gpu_runtime::gpu::DevicePtr;
use metrale_model_layers::layers::ops;

use super::super::block_mgmt::ensure_blocks_through_decode;
use super::super::types::TransformerModel;
use crate::traits::{FeedSource, RowMask, SequenceState};

/// 2026-09-25: What a single-sequence decode row is fed. `Host`: the id is embedded from a
/// host value and pushed into `seq.tokens` by the decode step. `Fed`: the resolved id
/// already sits in the model's feed-id buffer, is gathered from there, and is not pushed.
#[derive(Clone, Copy, Debug)]
pub(super) enum DecodeInput {
    Host(u32),
    Fed,
}

impl DecodeInput {
    pub(super) fn host(self) -> Option<u32> {
        match self {
            Self::Host(t) => Some(t),
            Self::Fed => None,
        }
    }
}

/// 2026-09-25: The batched twin of [`DecodeInput`].
#[derive(Clone, Copy, Debug)]
pub(super) enum BatchInput<'a> {
    Host(&'a [u32]),
    Fed { n: usize },
}

impl BatchInput<'_> {
    pub(super) fn len(&self) -> usize {
        match self {
            Self::Host(t) => t.len(),
            Self::Fed { n } => *n,
        }
    }
    pub(super) fn host(&self) -> Option<&[u32]> {
        match self {
            Self::Host(t) => Some(t),
            Self::Fed { .. } => None,
        }
    }
    pub(super) fn is_fed(&self) -> bool {
        matches!(self, Self::Fed { .. })
    }
}

/// 2026-09-25: The source word `feed_resolve` reads for one row.
fn source_word(s: FeedSource) -> u32 {
    match s {
        FeedSource::Feed { from_row } => from_row,
        FeedSource::Host(id) => id | ops::FEED_HOST_BIT,
    }
}

impl TransformerModel {
    /// 2026-09-25: Whether this model can run fed steps: every path a fed step takes must be
    /// the one the synchronous step takes, minus the host id. Each clause names a path that
    /// would need the id on the host or would leave the plain batched route.
    pub(super) fn supports_device_token_feed_dispatch(&self) -> bool {
        let kernels = self.argmax_feed_kernel.0 != 0
            && self.feed_resolve_kernel.0 != 0
            && self.batched_embed_kernel.0 != 0
            && self.feed_cells != DevicePtr::NULL;
        let embed_needs_history = self.ngram_embed.is_some();
        let layers_need_host_id = self
            .layers
            .iter()
            .any(|l| l.decode_graph_unsupported() || l.decode_multi_seq_unsupported());
        let per_row_fallback =
            self.config.hc_mult > 0 || self.is_mla_dispatch() || self.config.index_topk > 0;
        let hss = self.kv_cache.lock().config().cache_blocks_per_seq.is_some();
        kernels
            && !embed_needs_history
            && !layers_need_host_id
            && !per_row_fallback
            && self.comm.is_none()
            && !self.use_fp32_logits
            && !self.profile
            && !hss
    }

    /// 2026-09-25: Resolve `sources` into the feed-id buffer on the device, then run the
    /// plain decode step over `seqs` with `DecodeInput::Fed`.
    pub(super) fn decode_batch_fed_dispatch(
        &self,
        sources: &[FeedSource],
        seqs: &mut [&mut SequenceState],
        stream: u64,
    ) -> Result<DevicePtr> {
        let n = sources.len();
        if n != seqs.len() {
            bail!("decode_batch_fed: {n} sources for {} rows", seqs.len());
        }
        if n == 0 || n > self.feed_rows {
            bail!(
                "decode_batch_fed: {n} rows, feed capacity {}",
                self.feed_rows
            );
        }
        if !self.supports_device_token_feed_dispatch() {
            bail!("decode_batch_fed: this model does not support the device token feed");
        }
        for s in sources {
            if let FeedSource::Feed { from_row } = s
                && *from_row as usize >= self.feed_rows
            {
                bail!("decode_batch_fed: feed row {from_row} beyond the cell capacity");
            }
        }
        let dev_stream = self.gpu.default_stream();
        let words: Vec<u8> = sources
            .iter()
            .flat_map(|s| source_word(*s).to_le_bytes())
            .collect();
        // 2026-09-25: Pageable source: the driver stages the bytes before returning, so
        // this enqueues without the stream wait a pinned source would cost.
        self.gpu
            .copy_h2d_async(&words, self.feed_sources, dev_stream)?;
        ops::feed_resolve(
            self.gpu.as_ref(),
            self.feed_resolve_kernel,
            self.feed_sources,
            self.feed_cells,
            self.feed_ids,
            n as u32,
            dev_stream,
        )?;
        for s in seqs.iter_mut() {
            self.ssm_h_to_f16_dispatch(s)?;
        }
        // 2026-09-25: A successful fed launch pushes exactly one pin, graphed or not, so each
        // `fed_step_settled` pops one entry.
        let mut graph_key: Option<Vec<u32>> = None;
        let logits = if n == 1 {
            self.decode_dispatch_with(DecodeInput::Fed, seqs[0], stream)?
        } else {
            self.decode_batch_compute_main_with(
                BatchInput::Fed { n },
                seqs,
                stream,
                &mut graph_key,
            )?
        };
        self.pin_fed_graph_key(graph_key.unwrap_or_default());
        Ok(logits)
    }

    /// 2026-09-25: Gather rows `0..n` of the feed-id buffer into `hidden`: the fed
    /// twin of `embed_ctx`'s plain row copy, same bytes from the same table.
    pub(super) fn feed_embed_rows(&self, hidden: DevicePtr, n: usize, stream: u64) -> Result<()> {
        ops::batched_embed(
            self.gpu.as_ref(),
            self.batched_embed_kernel,
            self.feed_ids,
            self.embed_tokens.weight,
            hidden,
            n as u32,
            self.config.hidden_size as u32,
            stream,
        )?;
        self.scale_embeddings(hidden, n, stream)
    }

    pub(super) fn argmax_batch_to_feed_dispatch(
        &self,
        logits: DevicePtr,
        masks: &[RowMask],
        dst: *mut u32,
        event: u64,
        _stream: u64,
    ) -> Result<()> {
        let n = masks.len();
        if n == 0 || n > self.feed_rows {
            bail!(
                "argmax_batch_to_feed: {n} rows, feed capacity {}",
                self.feed_rows
            );
        }
        if self.argmax_feed_kernel.0 == 0 || dst.is_null() {
            bail!("argmax_batch_to_feed: no feed kernel or no destination");
        }
        let stream = self.gpu.default_stream();
        let v = self.config.vocab_size as u32;
        let mask_bytes: Vec<u8> = masks
            .iter()
            .flat_map(|m| [m[0].to_le_bytes(), m[1].to_le_bytes()].concat())
            .collect();
        self.gpu
            .copy_h2d_async(&mask_bytes, self.feed_masks, stream)?;
        ops::argmax_bf16_batch_feed(
            self.gpu.as_ref(),
            self.argmax_feed_kernel,
            logits,
            self.feed_masks,
            self.feed_cells,
            v,
            n as u32,
            v,
            stream,
        )?;
        // 2026-09-25: SAFETY: `dst` is the router's pinned slot of at least `n` u32 cells,
        // which it keeps alive and unread until `event` completes.
        let out = unsafe { std::slice::from_raw_parts_mut(dst as *mut u8, n * 4) };
        self.gpu.copy_d2h_async(self.feed_cells, out, stream)?;
        self.gpu.record_event(event, stream)
    }

    /// 2026-09-25: The block for position `seq.seq_len`, allocated if missing, through
    /// `ensure_blocks_through_decode`. Returns how many blocks were added.
    pub(super) fn reserve_decode_block_dispatch(&self, seq: &mut SequenceState) -> Result<usize> {
        let mut kv_cache = self.kv_cache.lock();
        let bs = kv_cache.block_size();
        let before = seq.block_table.len();
        ensure_blocks_through_decode(
            seq,
            seq.seq_len / bs,
            &mut kv_cache,
            self.prefix_cache.as_ref(),
            self.gpu.as_ref(),
            self.gpu.default_stream(),
            self.levers.kv_poison,
        )?;
        Ok(seq.block_table.len() - before)
    }

    pub(super) fn release_decode_blocks_dispatch(
        &self,
        seq: &mut SequenceState,
        blocks: usize,
    ) -> Result<()> {
        if blocks > seq.block_table.len() {
            bail!(
                "release_decode_blocks: {blocks} requested, {} held",
                seq.block_table.len()
            );
        }
        let mut kv_cache = self.kv_cache.lock();
        for _ in 0..blocks {
            if let Some(b) = seq.block_table.pop() {
                kv_cache.free_block(b);
            }
        }
        Ok(())
    }

    /// 2026-09-25: Pin the batched graph a fed step replays until `fed_step_settled`.
    pub(super) fn pin_fed_graph_key(&self, key: Vec<u32>) {
        self.inflight_fed_graph_keys.lock().push_back(key);
    }

    pub(super) fn fed_step_settled_dispatch(&self) {
        self.inflight_fed_graph_keys.lock().pop_front();
    }

    pub(super) fn pinned_graph_keys(&self) -> Vec<Vec<u32>> {
        self.inflight_fed_graph_keys
            .lock()
            .iter()
            .cloned()
            .collect()
    }
}
