// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Readback of the batched verify's argmax rows into host memory.
//!
//! Owner: model-engine (speculative verify).
//! Invariants:
//! - Every arm is blocking: the returned bytes are the finished argmax rows.

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::super::super::types::TransformerModel;

impl TransformerModel {
    /// 2026-09-26: The `r_total * 4` argmax bytes, read from the mapped blob or copied
    /// from scratch.
    pub(super) fn read_verify_argmax(
        &self,
        mapped_argmax: Option<(*mut u8, DevicePtr)>,
        r_total: usize,
        stream: u64,
    ) -> Result<Vec<u8>> {
        let mut buf = vec![0u8; r_total * 4];
        let mut filled = false;
        if let Some((host, _)) = mapped_argmax {
            // 2026-09-25: Mapped results: synchronise the stream, then read
            // host memory.
            self.gpu.synchronize(stream)?;
            // 2026-09-25: SAFETY: `host` is the 65_536-byte `alloc_host_pinned`
            // blob from `mapped_argmax_host_dev`, never freed. The
            // `ensure!(r_total <= VERIFY_ROW_CAP)` in
            // `decode_verify_batched_dispatch` bounds
            // `r_total * 4 <= 640`. The blob is zeroed at allocation, the
            // argmax wrote rows `0..r_total` through its device alias, and the
            // `synchronize` above ordered those writes before this read.
            let src = unsafe { std::slice::from_raw_parts(host, r_total * 4) };
            buf.copy_from_slice(src);
            filled = true;
        }
        // 2026-09-25: Copy arms when there is no mapped blob: the default
        // stream copy under `METRALE_VERIFY_D2H_DEFAULT_STREAM=1`, an
        // on-stream copy into `buf` under `METRALE_NO_PINNED_VERIFY_D2H=1`,
        // and otherwise an on-stream copy into a 64 KiB page-locked blob
        // allocated on first use and kept for the process. Every arm is a
        // blocking copy.
        if filled {
            // 2026-09-25: The mapped path already read the results.
        } else if super::super::verify_e2::verify_d2h_default_stream() {
            self.gpu.copy_d2h(self.buffers.scratch(), &mut buf)?;
        } else if super::super::verify_e2::verify_d2h_no_pinned() {
            self.gpu
                .copy_d2h_on_stream(self.buffers.scratch(), &mut buf, stream)?;
        } else {
            use std::sync::atomic::{AtomicPtr, Ordering};
            const PINNED_CAP: usize = 65_536;
            static PINNED: AtomicPtr<u8> = AtomicPtr::new(std::ptr::null_mut());
            let mut p = PINNED.load(Ordering::Acquire);
            if p.is_null() {
                p = self.gpu.alloc_host_pinned(PINNED_CAP)?;
                // 2026-09-25: Only the scheduler thread calls this, so a plain store suffices.
                PINNED.store(p, Ordering::Release);
            }
            if buf.len() <= PINNED_CAP {
                // 2026-09-25: SAFETY: PINNED points at a live
                // `alloc_host_pinned` blob of PINNED_CAP bytes, zeroed at
                // allocation, and the enclosing `if buf.len() <= PINNED_CAP`
                // keeps the slice inside it. The synchronous
                // `copy_d2h_on_stream` fills all `buf.len()` bytes before
                // `copy_from_slice` reads them. Only the scheduler thread uses
                // the blob, and the slice does not outlive this block.
                let dst = unsafe { std::slice::from_raw_parts_mut(p, buf.len()) };
                self.gpu
                    .copy_d2h_on_stream(self.buffers.scratch(), dst, stream)?;
                buf.copy_from_slice(dst);
            } else {
                self.gpu
                    .copy_d2h_on_stream(self.buffers.scratch(), &mut buf, stream)?;
            }
        }
        Ok(buf)
    }
}
