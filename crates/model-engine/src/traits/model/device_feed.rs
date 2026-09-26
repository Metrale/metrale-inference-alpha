// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: `ModelDeviceFeed`, one of the supertraits `Model` is made of. Its methods, default
//! bodies and docs are the ones `Model` declared before the split.
//!
//! Owner: model-engine.
//! Invariants: none beyond the types.

use super::{FeedSource, RowMask};
use crate::traits::SequenceState;
use anyhow::{Result, bail};
use metrale_gpu_runtime::gpu::DevicePtr;

/// 2026-09-26: The device token feed of the asynchronous scheduler router, with the event queries,
/// async copies and pinned host memory it uses.
pub trait ModelDeviceFeed {
    // 2026-09-25: Device token feed: what the asynchronous scheduler router needs to run a
    // plain decode step ahead of the host. `AsyncDeviceIo::new` refuses a model whose
    // `supports_device_token_feed` is `false`, the default.

    /// 2026-09-25: Whether the model can run a decode step whose input ids live on the device
    /// (`decode_batch_fed`) and answer it with a masked argmax written into the feed cells
    /// (`argmax_batch_to_feed`). Default `false`.
    fn supports_device_token_feed(&self) -> bool {
        false
    }

    /// 2026-09-25: The decode step over `seqs` with each row's input id read on the device from
    /// `sources`: the forward, KV writes and block allocation of `decode_batch`, but it does not
    /// push the token or advance `seq_len`, because the router does not know a fed row's id at
    /// launch; the scheduler core does that once the previous step is committed. Position
    /// `seq.seq_len` is written. Default: an error.
    fn decode_batch_fed(
        &self,
        _sources: &[FeedSource],
        _seqs: &mut [&mut SequenceState],
        _stream: u64,
    ) -> Result<DevicePtr> {
        bail!("decode_batch_fed: this model has no device token feed")
    }

    /// 2026-09-25: Argmax over `masks.len()` rows of `logits`, applying each row's two-id mask
    /// (`argmax_feed.cu`): writes the feed cells, copies the ids to the pinned `dst` with an
    /// async D2H on `stream`, then records `event`. It never synchronises; `dst` is valid once
    /// `event` completes. Default: an error.
    fn argmax_batch_to_feed(
        &self,
        _logits: DevicePtr,
        _masks: &[RowMask],
        _dst: *mut u32,
        _event: u64,
        _stream: u64,
    ) -> Result<()> {
        bail!("argmax_batch_to_feed: this model has no device token feed")
    }

    /// 2026-09-25: Allocate the KV block for the next decode position (`seq.seq_len`) if the
    /// block table lacks it; returns the blocks added (0 or 1). `Err` when the pool is empty.
    /// Default: an error.
    fn reserve_decode_block(&self, _seq: &mut SequenceState) -> Result<usize> {
        bail!("reserve_decode_block: this model has no device token feed")
    }

    /// 2026-09-25: Return the last `blocks` entries of the block table, added by
    /// `reserve_decode_block` for a discarded step, to the pool. Default: an error.
    fn release_decode_blocks(&self, _seq: &mut SequenceState, _blocks: usize) -> Result<()> {
        bail!("release_decode_blocks: this model has no device token feed")
    }

    /// 2026-09-25: A fed step has completed on the device, so the CUDA graph it replayed may be
    /// evicted. Default: no-op.
    fn fed_step_settled(&self) {}

    /// 2026-09-25: Whether the work recorded against `event` has completed; never blocks.
    /// Default `Ok(true)`.
    fn event_query(&self, _event: u64) -> Result<bool> {
        Ok(true)
    }

    /// 2026-09-25: Block the host until the work recorded against `event` has completed.
    /// Default: `Ok(())`.
    fn event_synchronize(&self, _event: u64) -> Result<()> {
        Ok(())
    }

    fn destroy_event(&self, _event: u64) -> Result<()> {
        Ok(())
    }

    /// 2026-09-25: Async device-to-host copy on `stream`; `dst` must stay valid and unread until
    /// `stream` is synchronised. Default: an error.
    fn copy_d2h_async(&self, _src: DevicePtr, _dst: &mut [u8], _stream: u64) -> Result<()> {
        bail!("copy_d2h_async: this model has no device token feed")
    }

    /// 2026-09-25: Zeroed page-locked host memory for the router's readback ring. Default: an
    /// error.
    fn alloc_host_pinned(&self, _bytes: usize) -> Result<*mut u8> {
        bail!("alloc_host_pinned: this model has no device token feed")
    }

    fn free_host_pinned(&self, _ptr: *mut u8, _bytes: usize) -> Result<()> {
        bail!("free_host_pinned: this model has no device token feed")
    }
}
