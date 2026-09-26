// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The bodies of `MetalGpuBackend`'s event methods, over the
//! shared-event slab.
//!
//! Owner: gpu-runtime (Metal backend).
//! Invariants:
//! - Event handle `h` is slab slot `h - 1`; 0 is never returned, and a slot is
//!   never removed, so a handle is never reused.

use anyhow::{Result, anyhow};
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandBuffer, MTLDevice, MTLEvent};

use super::{EventSlot, MetalGpuBackend};

impl MetalGpuBackend {
    pub(super) fn create_event_mtl(&self) -> Result<u64> {
        let event = self
            .device
            .newSharedEvent()
            .ok_or_else(|| anyhow!("newSharedEvent returned null"))?;
        let mut slab = self.events.lock();
        slab.push(EventSlot { event, next: 1 });
        // 2026-09-25: Handle = slab index + 1; 0 is never returned.
        Ok(slab.len() as u64)
    }

    pub(super) fn record_event_mtl(&self, event: u64, stream: u64) -> Result<()> {
        let value = {
            let mut slab = self.events.lock();
            let idx = (event as usize)
                .checked_sub(1)
                .ok_or_else(|| anyhow!("record_event: invalid event handle {event}"))?;
            let slot = slab
                .get_mut(idx)
                .ok_or_else(|| anyhow!("record_event: event handle {event} out of range"))?;
            let v = slot.next;
            slot.next += 1;
            v
        };
        let cmd_buf = self.current_cmd_buf(stream)?;
        let event_obj = {
            let slab = self.events.lock();
            slab[(event - 1) as usize].event.clone()
        };
        // 2026-09-25: Encoded on the stream's in-flight command buffer.
        let proto: &ProtocolObject<dyn MTLEvent> = ProtocolObject::from_ref(&*event_obj);
        cmd_buf.encodeSignalEvent_value(proto, value);
        Ok(())
    }

    pub(super) fn stream_wait_event_mtl(&self, stream: u64, event: u64) -> Result<()> {
        let (event_obj, value) = {
            let slab = self.events.lock();
            let idx = (event as usize)
                .checked_sub(1)
                .ok_or_else(|| anyhow!("stream_wait_event: invalid event handle {event}"))?;
            let slot = slab
                .get(idx)
                .ok_or_else(|| anyhow!("stream_wait_event: event handle {event} out of range"))?;
            // 2026-09-25: The last recorded value, `next - 1`; 0 when nothing
            // was recorded.
            (slot.event.clone(), slot.next.saturating_sub(1))
        };
        let cmd_buf = self.current_cmd_buf(stream)?;
        let proto: &ProtocolObject<dyn MTLEvent> = ProtocolObject::from_ref(&*event_obj);
        cmd_buf.encodeWaitForEvent_value(proto, value);
        Ok(())
    }

    pub(super) fn destroy_event_mtl(&self, event: u64) -> Result<()> {
        if event == 0 {
            return Ok(());
        }
        let mut slab = self.events.lock();
        let idx = (event - 1) as usize;
        if let Some(slot) = slab.get_mut(idx) {
            // 2026-09-25: The slot stays, so indices are stable and a handle is
            // never reused; only its counter is reset.
            slot.next = 0;
        }
        Ok(())
    }
}
