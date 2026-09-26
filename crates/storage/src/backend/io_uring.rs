// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The Linux file backend: `io_uring` fixed-buffer reads (SQPOLL,
//! registered pinned buffers) into `qd` bounce buffers, each copied to the
//! device on the caller's stream.
//!
//! Owner: metrale-storage high-speed swap.
//! Invariants:
//! - A bounce buffer is refilled only after the event recorded behind its last
//!   host-to-device copy has completed (`wait_buffer_free`).

use anyhow::{Context, Result, bail};
use io_uring::{IoUring, opcode, types};
use std::ffi::c_void;
use std::os::fd::RawFd;

use super::{ReadRequest, StorageBackend};
use crate::cuda_min::{CudaEvent, PinnedBuffer, copy_h_to_d_async, stream_sync};
use crate::group::{GroupKey, GroupLayout};
use crate::layout::Layout;

pub struct IoUringBackend {
    layout: Layout,
    ring: IoUring,
    buffers: Vec<PinnedBuffer>,
    // 2026-09-25: The event behind each buffer's last copy; `None` when no copy
    // is pending.
    events: Vec<Option<CudaEvent>>,
    qd: usize,
}

impl IoUringBackend {
    pub fn new(layout: Layout, qd: usize) -> Result<Self> {
        if qd == 0 {
            bail!("queue depth must be ≥ 1");
        }
        // 2026-09-25: The SQPOLL kernel thread idles 2000 ms before it sleeps.
        let ring = IoUring::builder()
            .setup_sqpoll(2_000)
            .build(qd as u32)
            .context("io_uring build")?;

        let group_bytes = layout.group_bytes() as usize;
        let mut buffers = Vec::with_capacity(qd);
        for _ in 0..qd {
            buffers.push(PinnedBuffer::new(group_bytes)?);
        }
        // 2026-09-25: After registration, `ReadFixed` with index `i` lands in
        // `buffers[i]`.
        let iovecs: Vec<libc::iovec> = buffers
            .iter()
            .map(|b| libc::iovec {
                iov_base: b.ptr,
                iov_len: b.bytes,
            })
            .collect();
        unsafe {
            ring.submitter()
                .register_buffers(&iovecs)
                .context("register_buffers")?;
        }
        let events: Vec<Option<CudaEvent>> = (0..qd).map(|_| None).collect();
        Ok(Self {
            layout,
            ring,
            buffers,
            events,
            qd,
        })
    }

    pub fn layout(&self) -> &Layout {
        &self.layout
    }

    /// 2026-09-25: Advise the kernel to drop every layer file's page cache
    /// (`POSIX_FADV_DONTNEED`), so later reads go to the device.
    pub fn drop_pagecache(&self) {
        for layer in 0..self.layout.spec.num_layers {
            let fd = self.layout.fd(layer);
            unsafe { libc::posix_fadvise(fd, 0, 0, libc::POSIX_FADV_DONTNEED) };
        }
    }

    /// 2026-09-25: Wait for the last copy out of `buf_idx`, if one is pending,
    /// so the buffer can take a new read.
    fn wait_buffer_free(&mut self, buf_idx: usize) -> Result<()> {
        if let Some(ev) = self.events[buf_idx].take() {
            ev.sync()?;
        }
        Ok(())
    }

    /// 2026-09-25: Queue one fixed-buffer read of `bytes` at `offset` into
    /// `buf_idx`, tagged `user_data`. Fails when the submission queue is full.
    fn submit_read(
        &mut self,
        fd: RawFd,
        offset: u64,
        bytes: u32,
        buf_idx: u16,
        user_data: u64,
    ) -> Result<()> {
        let buf_ptr = self.buffers[buf_idx as usize].ptr as *mut u8;
        let read_e = opcode::ReadFixed::new(types::Fd(fd), buf_ptr, bytes, buf_idx)
            .offset(offset)
            .build()
            .user_data(user_data);
        unsafe {
            self.ring
                .submission()
                .push(&read_e)
                .map_err(|_| anyhow::anyhow!("io_uring SQ full"))?;
        }
        Ok(())
    }
}

impl StorageBackend for IoUringBackend {
    fn read(&mut self, requests: &[ReadRequest], stream: u64) -> Result<()> {
        let bytes = self.layout.group_bytes() as u32;
        // 2026-09-25: user_data is `req_index << 16 | buf_index`.
        if requests.len() > u16::MAX as usize {
            bail!("io_uring batch too large: {}", requests.len());
        }

        let mut next_submit = 0;
        let mut completed = 0;
        // 2026-09-25: Free buffers form a stack; a buffer leaves it at submit
        // and returns when its completion is reaped.
        let mut free_bufs: Vec<u16> = (0..self.qd as u16).rev().collect();

        while completed < requests.len() {
            while next_submit < requests.len() {
                let Some(&buf_idx) = free_bufs.last() else {
                    break;
                };
                self.wait_buffer_free(buf_idx as usize)?;
                free_bufs.pop();
                let req = &requests[next_submit];
                let fd = self.layout.fd(req.group.layer);
                let off = self.layout.offset(req.group);
                let user = ((next_submit as u64) << 16) | (buf_idx as u64);
                self.submit_read(fd, off, bytes, buf_idx, user)?;
                next_submit += 1;
            }
            self.ring
                .submit_and_wait(1)
                .context("io_uring submit_and_wait")?;
            let cq = self.ring.completion();
            for cqe in cq {
                let user = cqe.user_data();
                let buf_idx = (user & 0xFFFF) as u16;
                let req_idx = (user >> 16) as usize;
                let result = cqe.result();
                if result < 0 {
                    bail!("io_uring read failed for req {req_idx}: errno {}", -result);
                }
                if result as u32 != bytes {
                    bail!("io_uring short read: req {req_idx} got {result}, expected {bytes}");
                }
                let req = &requests[req_idx];
                let buf = &self.buffers[buf_idx as usize];
                copy_h_to_d_async(
                    req.dst_dev_ptr,
                    buf.ptr as *const c_void,
                    bytes as usize,
                    stream,
                )?;
                let ev = CudaEvent::new()?;
                ev.record(stream)?;
                self.events[buf_idx as usize] = Some(ev);
                free_bufs.push(buf_idx);
                completed += 1;
            }
        }
        // 2026-09-25: The `read` contract: the data is visible to later work
        // on `stream`.
        stream_sync(stream)?;
        // 2026-09-25: The sync above completed every recorded event.
        for slot in self.events.iter_mut() {
            *slot = None;
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
        // 2026-09-25: The layer files are O_DIRECT (layout.rs), so `src` is
        // staged through pinned buffer 0.
        self.wait_buffer_free(0)?;
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), self.buffers[0].ptr as *mut u8, bytes);
        }
        let fd = self.layout.fd(key.layer);
        let off = self.layout.offset(key) as i64;
        let n = unsafe { libc::pwrite(fd, self.buffers[0].ptr, bytes, off) };
        if n != bytes as isize {
            bail!(
                "pwrite {bytes}@{off} returned {n}, errno {}",
                std::io::Error::last_os_error()
            );
        }
        Ok(())
    }

    fn group_layout(&self) -> GroupLayout {
        self.layout.spec
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda_min::{CudaCtx, DeviceBuffer, copy_d_to_h_async};
    use crate::group::{GroupKey, GroupLayout, KvKind};
    use std::path::PathBuf;

    fn tempdir(name: &str) -> PathBuf {
        let p =
            std::env::temp_dir().join(format!("metrale-iouring-{}-{}", name, std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    #[ignore = "requires GPU"]
    fn write_then_read_round_trip() {
        let _ctx = CudaCtx::new(0).expect("cuda init");
        let dir = tempdir("rt");
        let spec = GroupLayout::new(1, 4, 1, 16, 128, 2, 4096);
        let layout = Layout::create(&dir, spec).unwrap();
        let mut backend = IoUringBackend::new(layout, 4).unwrap();
        let bytes = backend.layout().group_bytes() as usize;
        // 2026-09-25: Four blocks with distinct patterns, as many as the queue
        // depth.
        let patterns: Vec<(GroupKey, Vec<u8>)> = (0..4u32)
            .map(|b| {
                let k = GroupKey::new(0, b, 0, KvKind::K);
                let pat: Vec<u8> = (0..bytes)
                    .map(|i| ((i + b as usize) & 0xFF) as u8)
                    .collect();
                (k, pat)
            })
            .collect();
        for (k, p) in &patterns {
            backend.write_from_host(*k, p).unwrap();
        }
        backend.drop_pagecache();
        let dev: Vec<DeviceBuffer> = patterns
            .iter()
            .map(|_| DeviceBuffer::new(bytes).unwrap())
            .collect();
        let reqs: Vec<ReadRequest> = patterns
            .iter()
            .zip(&dev)
            .map(|((k, _), d)| ReadRequest {
                group: *k,
                dst_dev_ptr: d.ptr,
            })
            .collect();
        backend.read(&reqs, _ctx.stream).unwrap();
        for ((_, expected), d) in patterns.iter().zip(&dev) {
            let mut got = vec![0_u8; bytes];
            copy_d_to_h_async(got.as_mut_ptr() as *mut c_void, d.ptr, bytes, _ctx.stream).unwrap();
            stream_sync(_ctx.stream).unwrap();
            assert_eq!(&got, expected);
        }
        std::fs::remove_dir_all(&dir).ok();
    }
}
