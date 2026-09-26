// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The `StorageBackend` trait, which moves KV groups and blocks
//! between a storage tier and device memory, and the two file backends.
//!
//! Owner: metrale-storage high-speed swap.
//! Invariants:
//! - A backend that keeps the default `read_blocks` issues, for each block,
//!   the per-head reads of `expand_blocks_to_groups`, in that order (pinned
//!   by `mod_tests.rs`).

use anyhow::Result;

use crate::group::{GroupKey, GroupLayout, KvKind};

// 2026-09-25: io_uring is Linux-only; `posix` builds on every target.
#[cfg(target_os = "linux")]
pub mod io_uring;
pub mod posix;

#[cfg(target_os = "linux")]
pub use self::io_uring::IoUringBackend;
pub use posix::PosixBackend;

/// 2026-09-25: Read `group` from the tier into device memory at `dst_dev_ptr`.
#[derive(Clone, Copy, Debug)]
pub struct ReadRequest {
    pub group: GroupKey,
    pub dst_dev_ptr: u64,
}

/// 2026-09-25: Read one whole block (`block_bytes()`: every kv_head's K, then
/// every V) into device memory starting at `dst_dev_ptr`. Only `base_key`'s
/// `layer` and `block` are read.
#[derive(Clone, Copy, Debug)]
pub struct BlockReadRequest {
    pub base_key: GroupKey,
    pub dst_dev_ptr: u64,
}

/// 2026-09-25: Expand each block request into its `2·nkv` per-head requests,
/// ordered `K(kh), V(kh)` for `kh` in `0..nkv`, with device destinations
/// `dst + kh·gs` (K) and `dst + (nkv + kh)·gs` (V). The default `read_blocks`
/// and `read_blocks_async` fan out through it.
pub fn expand_blocks_to_groups(spec: &GroupLayout, reqs: &[BlockReadRequest]) -> Vec<ReadRequest> {
    let nkv = spec.num_kv_heads;
    let gs = spec.group_stride;
    let mut out = Vec::with_capacity(reqs.len() * 2 * nkv as usize);
    for r in reqs {
        let layer = r.base_key.layer;
        let block = r.base_key.block;
        for kh in 0..nkv {
            out.push(ReadRequest {
                group: GroupKey::new(layer, block, kh, KvKind::K),
                dst_dev_ptr: r.dst_dev_ptr + (kh as u64) * gs,
            });
            out.push(ReadRequest {
                group: GroupKey::new(layer, block, kh, KvKind::V),
                dst_dev_ptr: r.dst_dev_ptr + (nkv as u64 + kh as u64) * gs,
            });
        }
    }
    out
}

pub trait StorageBackend: Send + Sync {
    /// 2026-09-25: Fill every request's destination. Returns once the data is
    /// in place, so work the caller enqueues on `stream` afterwards reads it.
    fn read(&mut self, requests: &[ReadRequest], stream: u64) -> Result<()>;

    /// 2026-09-25: Like `read`, but an implementation may return before the
    /// copies on `stream` finish. The default calls `read`.
    fn read_async(&mut self, requests: &[ReadRequest], stream: u64) -> Result<()> {
        self.read(requests, stream)
    }

    /// 2026-09-25: Write one group from host memory; `src` is `group_bytes()`
    /// long.
    fn write_from_host(&mut self, key: GroupKey, src: &[u8]) -> Result<()>;

    /// 2026-09-25: The geometry the default block methods fan out with.
    fn group_layout(&self) -> GroupLayout;

    /// 2026-09-25: Read whole blocks, with the stream contract of `read`. The
    /// default fans out to `read` through `expand_blocks_to_groups`.
    fn read_blocks(&mut self, requests: &[BlockReadRequest], stream: u64) -> Result<()> {
        let groups = expand_blocks_to_groups(&self.group_layout(), requests);
        self.read(&groups, stream)
    }

    /// 2026-09-25: The block form of `read_async`. The default fans out to
    /// `read_async`.
    fn read_blocks_async(&mut self, requests: &[BlockReadRequest], stream: u64) -> Result<()> {
        let groups = expand_blocks_to_groups(&self.group_layout(), requests);
        self.read_async(&groups, stream)
    }

    /// 2026-09-25: Write one whole block. `src` is `block_bytes()` laid out
    /// `[K0, …, K(nkv-1), V0, …, V(nkv-1)]` at `group_stride` pitch; only
    /// `base_key`'s `layer` and `block` are read. The default rejects any other
    /// length, then calls `write_from_host` for K(kh) and V(kh) of each head.
    fn write_block_from_host(&mut self, base_key: GroupKey, src: &[u8]) -> Result<()> {
        let spec = self.group_layout();
        let nkv = spec.num_kv_heads as usize;
        let gs = spec.group_stride as usize;
        let expect = 2 * nkv * gs;
        if src.len() != expect {
            anyhow::bail!(
                "write_block_from_host: src len {} != block bytes {expect}",
                src.len()
            );
        }
        let layer = base_key.layer;
        let block = base_key.block;
        for kh in 0..nkv {
            let k_off = kh * gs;
            let v_off = (nkv + kh) * gs;
            self.write_from_host(
                GroupKey::new(layer, block, kh as u16, KvKind::K),
                &src[k_off..k_off + gs],
            )?;
            self.write_from_host(
                GroupKey::new(layer, block, kh as u16, KvKind::V),
                &src[v_off..v_off + gs],
            )?;
        }
        Ok(())
    }

    /// 2026-09-25: Write `run_len` consecutive blocks of one layer, starting at
    /// `base_key`'s block; `src` is `run_len · block_bytes()`. The default
    /// rejects any other length, then calls `write_block_from_host` once per
    /// block.
    fn write_blocks_run(&mut self, base_key: GroupKey, run_len: usize, src: &[u8]) -> Result<()> {
        let spec = self.group_layout();
        let block_bytes = spec.block_bytes() as usize;
        let expect = run_len * block_bytes;
        if src.len() != expect {
            anyhow::bail!(
                "write_blocks_run: src len {} != run bytes {expect} ({run_len} × {block_bytes})",
                src.len()
            );
        }
        for i in 0..run_len {
            let off = i * block_bytes;
            self.write_block_from_host(
                GroupKey::new(base_key.layer, base_key.block + i as u32, 0, KvKind::K),
                &src[off..off + block_bytes],
            )?;
        }
        Ok(())
    }

    /// 2026-09-25: Whether this backend serves `write_blocks_run` as one wide
    /// op. The default is `false`.
    fn supports_write_run_coalescing(&self) -> bool {
        false
    }

    /// 2026-09-25: Register `[base, base + len)` as the region reads land in.
    /// The RDMA backends register it as one memory region per rail; the default
    /// does nothing.
    fn register_landing_region(&mut self, base: u64, len: usize) -> Result<()> {
        let _ = (base, len);
        Ok(())
    }
}

#[cfg(test)]
#[path = "mod_tests.rs"]
mod coalesce_tests;
