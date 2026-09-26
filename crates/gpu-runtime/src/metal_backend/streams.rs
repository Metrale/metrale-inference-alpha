// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: `MetalGpuBackend`'s buffer lookup and stream-slab helpers: the
//! allocation resolving a pointer, a stream handle's slab index, and a stream's
//! in-flight command buffer.
//!
//! Owner: gpu-runtime (Metal backend).
//! Invariants:
//! - Stream handle 0 is slab slot 0 and handle `h > 0` is slot `h - 1`.
//! - Each method that takes the streams mutex releases it before returning.

use std::collections::BTreeMap;

use anyhow::{Result, anyhow, bail};
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandQueue};

use super::{MetalGpuBackend, MetalStream, ObjBuffer, ObjCmdBuf};
use crate::gpu::DevicePtr;

impl MetalGpuBackend {
    /// 2026-09-25: The buffer with the largest base at or below `ptr`, and
    /// `ptr`'s offset in it. `None` when no base is at or below `ptr`, or the
    /// offset exceeds the buffer's length (an offset equal to the length is
    /// accepted).
    pub(super) fn find_buffer(
        allocs: &BTreeMap<u64, ObjBuffer>,
        ptr: DevicePtr,
    ) -> Option<(ObjBuffer, usize)> {
        let (base, buf) = allocs.range(..=ptr.0).next_back()?;
        let offset = (ptr.0 - *base) as usize;
        if offset > buf.length() {
            return None;
        }
        Some((buf.clone(), offset))
    }

    /// 2026-09-25: Slab index of a stream handle: 0 for handle 0, otherwise
    /// `handle - 1`; an error past the end of the slab.
    pub(super) fn stream_index(handle: u64, slab: &[MetalStream]) -> Result<usize> {
        let idx = if handle == 0 {
            0
        } else {
            (handle - 1) as usize
        };
        if idx >= slab.len() {
            bail!("Metal: invalid stream handle {handle}");
        }
        Ok(idx)
    }

    /// 2026-09-25: The stream's in-flight command buffer, opened if there is
    /// none. Returns a clone, so the caller encodes without holding the streams
    /// mutex.
    pub(super) fn current_cmd_buf(&self, stream_handle: u64) -> Result<ObjCmdBuf> {
        let mut slab = self.streams.lock();
        let idx = Self::stream_index(stream_handle, &slab)?;
        let s = &mut slab[idx];
        if let Some(ref cb) = s.in_flight {
            return Ok(cb.clone());
        }
        let cb = s
            .queue
            .commandBuffer()
            .ok_or_else(|| anyhow!("commandBuffer returned null on stream {stream_handle}"))?;
        s.in_flight = Some(cb.clone());
        Ok(cb)
    }

    /// 2026-09-25: Commit the stream's in-flight buffer without waiting, and
    /// return it so the caller can `waitUntilCompleted`.
    pub(super) fn commit_in_flight(&self, stream_handle: u64) -> Result<Option<ObjCmdBuf>> {
        let mut slab = self.streams.lock();
        let idx = Self::stream_index(stream_handle, &slab)?;
        let s = &mut slab[idx];
        let Some(cb) = s.in_flight.take() else {
            return Ok(None);
        };
        cb.commit();
        Ok(Some(cb))
    }
}
