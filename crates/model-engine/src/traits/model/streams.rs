// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: `ModelStreams`, one of the supertraits `Model` is made of. Its methods, default
//! bodies and docs are the ones `Model` declared before the split.
//!
//! Owner: model-engine.
//! Invariants: none beyond the types.

use anyhow::Result;

/// 2026-09-26: Streams, events and host synchronisation.
pub trait ModelStreams {
    /// 2026-09-25: The default stream. Default `0`.
    fn default_stream(&self) -> u64 {
        0
    }

    /// 2026-09-25: A new stream. Default `Ok(0)`.
    fn create_stream(&self) -> Result<u64> {
        Ok(0)
    }

    /// 2026-09-25: A new event. Default `Ok(0)`.
    fn create_event(&self) -> Result<u64> {
        Ok(0)
    }

    /// 2026-09-25: Record `event` on `stream`. Default: `Ok(())`.
    fn record_event(&self, _event: u64, _stream: u64) -> Result<()> {
        Ok(())
    }

    /// 2026-09-25: Make `stream` wait for `event` on the device; the host does not block.
    /// Default: `Ok(())`.
    fn stream_wait_event(&self, _stream: u64, _event: u64) -> Result<()> {
        Ok(())
    }

    /// 2026-09-25: Block the host until `stream`'s work has completed. Default: `Ok(())`.
    fn synchronize(&self, _stream: u64) -> Result<()> {
        Ok(())
    }
}
