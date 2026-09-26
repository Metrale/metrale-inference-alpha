// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The portable file backend: one pinned bounce buffer, a
//! positional read through `crate::tier::pio` (`pread`/`pwrite` on unix,
//! `seek_read`/`seek_write` on Windows) and a host-to-device copy per group.
//! `HighSpeedSwap` uses it on every target but Linux.
//!
//! Owner: metrale-storage high-speed swap.
//! Invariants:
//! - The bounce buffer is refilled only after the stream sync that follows its
//!   previous copy.

use anyhow::{Context, Result, bail};
use std::ffi::c_void;

use super::{ReadRequest, StorageBackend};
use crate::cuda_min::{PinnedBuffer, copy_h_to_d_async, stream_sync};
use crate::group::{GroupKey, GroupLayout};
use crate::layout::Layout;

pub struct PosixBackend {
    layout: Layout,
    bounce: PinnedBuffer,
}

impl PosixBackend {
    pub fn new(layout: Layout) -> Result<Self> {
        let bounce = PinnedBuffer::new(layout.group_bytes() as usize)
            .context("alloc pinned bounce buffer")?;
        Ok(Self { layout, bounce })
    }
    pub fn layout(&self) -> &Layout {
        &self.layout
    }
}

impl StorageBackend for PosixBackend {
    fn read(&mut self, requests: &[ReadRequest], stream: u64) -> Result<()> {
        let bytes = self.layout.group_bytes() as usize;
        let bounce_ptr = self.bounce.ptr;
        for req in requests {
            let off = self.layout.offset(req.group);
            // 2026-09-25: SAFETY: `bounce_ptr` is a pinned allocation of
            // `group_bytes()` owned by `self.bounce`; the previous copy out of it
            // finished at the `stream_sync` below.
            let buf = unsafe { std::slice::from_raw_parts_mut(bounce_ptr as *mut u8, bytes) };
            crate::tier::pio::read_exact_at(self.layout.file(req.group.layer), buf, off)
                .with_context(|| format!("read {bytes}@{off}"))?;
            // 2026-09-25: Every request reuses the one bounce buffer, so the copy
            // completes before the next read overwrites it.
            copy_h_to_d_async(req.dst_dev_ptr, bounce_ptr as *const c_void, bytes, stream)?;
            stream_sync(stream)?;
        }
        Ok(())
    }

    fn write_from_host(&mut self, key: GroupKey, src: &[u8]) -> Result<()> {
        let bytes = self.layout.group_bytes() as usize;
        if src.len() != bytes {
            bail!(
                "write_from_host: src len {} != group bytes {bytes}",
                src.len()
            );
        }
        // 2026-09-25: The Linux layer files are O_DIRECT (layout.rs), so `src`
        // is staged through the pinned bounce buffer.
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), self.bounce.ptr as *mut u8, bytes);
        }
        let off = self.layout.offset(key);
        // 2026-09-25: SAFETY: pinned, `group_bytes()` long, owned by `self`.
        let buf = unsafe { std::slice::from_raw_parts(self.bounce.ptr as *const u8, bytes) };
        crate::tier::pio::write_all_at(self.layout.file(key.layer), buf, off)
            .with_context(|| format!("write {bytes}@{off}"))?;
        // 2026-09-25: No fsync: the write is not durable across a crash.
        Ok(())
    }

    fn group_layout(&self) -> GroupLayout {
        self.layout.spec
    }
}

impl PosixBackend {
    /// 2026-09-25: Advise the kernel to drop every layer file's page cache
    /// (`POSIX_FADV_DONTNEED`), so later reads go to the device. A no-op on
    /// non-unix targets, where reads may still come from the cache.
    #[cfg(unix)]
    pub fn drop_pagecache(&self) {
        for layer in 0..self.layout.spec.num_layers {
            let fd = self.layout.fd(layer);
            unsafe { libc::posix_fadvise(fd, 0, 0, libc::POSIX_FADV_DONTNEED) };
        }
    }

    #[cfg(not(unix))]
    pub fn drop_pagecache(&self) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::group::{GroupLayout, KvKind};

    fn tempdir(name: &str) -> std::path::PathBuf {
        let p =
            std::env::temp_dir().join(format!("metrale-storage-{}-{}", name, std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    #[ignore = "requires GPU"]
    fn write_then_read_round_trip() {
        // 2026-09-25: `PinnedBuffer::new` needs a current CUDA context.
        let _ctx = crate::cuda_min::CudaCtx::new(0).expect("cuda init");
        let dir = tempdir("rt");
        let spec = GroupLayout::new(1, 2, 1, 16, 128, 2, 4096);
        let layout = Layout::create(&dir, spec).unwrap();
        let mut backend = PosixBackend::new(layout).unwrap();
        let bytes = backend.layout().group_bytes() as usize;
        let pat: Vec<u8> = (0..bytes).map(|i| (i & 0xFF) as u8).collect();
        let key = GroupKey::new(0, 1, 0, KvKind::V);
        backend.write_from_host(key, &pat).unwrap();
        backend.drop_pagecache();

        let dev = crate::cuda_min::DeviceBuffer::new(bytes).unwrap();
        let req = ReadRequest {
            group: key,
            dst_dev_ptr: dev.ptr,
        };
        backend.read(&[req], _ctx.stream).unwrap();
        let mut host_back = vec![0_u8; bytes];
        crate::cuda_min::copy_d_to_h_async(
            host_back.as_mut_ptr() as *mut c_void,
            dev.ptr,
            bytes,
            _ctx.stream,
        )
        .unwrap();
        crate::cuda_min::stream_sync(_ctx.stream).unwrap();
        assert_eq!(host_back, pat);
        std::fs::remove_dir_all(&dir).ok();
    }
}
