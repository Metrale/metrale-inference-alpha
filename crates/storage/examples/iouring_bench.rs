// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Read-throughput bench for `IoUringBackend`: one batch of
//! `n_iters` pseudo-random group reads per queue depth, after a page-cache
//! drop, printed as MiB/s.
//!
//! Owner: metrale-storage.
//! Invariants: none beyond the types.

use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use metrale_storage::backend::{IoUringBackend, ReadRequest, StorageBackend};
use metrale_storage::cuda_min::{CudaCtx, DeviceBuffer};
use metrale_storage::group::{GroupKey, GroupLayout, KvKind};
use metrale_storage::layout::Layout;

fn parse_args() -> (PathBuf, u32, u32) {
    let mut dir: Option<PathBuf> = None;
    // 2026-09-25: With the default head_dim: 256 × 128 × 2 bytes = 64 KiB groups.
    let mut block_size: u32 = 256;
    let mut head_dim: u32 = 128;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--dir" => dir = Some(PathBuf::from(args.next().unwrap())),
            "--block-size" => block_size = args.next().unwrap().parse().unwrap(),
            "--head-dim" => head_dim = args.next().unwrap().parse().unwrap(),
            "--help" | "-h" => {
                eprintln!("usage: iouring-bench --dir <path> [--block-size N] [--head-dim N]");
                std::process::exit(0);
            }
            other => panic!("unknown arg: {other}"),
        }
    }
    (dir.expect("--dir required"), block_size, head_dim)
}

fn main() -> Result<()> {
    let (dir, block_size, head_dim) = parse_args();
    std::fs::create_dir_all(&dir)?;
    let _ctx = CudaCtx::new(0)?;
    let nkv: u16 = 1;
    let num_blocks: u32 = 1024;
    let spec = GroupLayout::new(1, num_blocks, nkv, block_size, head_dim, 2, 4096);
    let group_bytes = spec.group_bytes() as usize;
    eprintln!("group_bytes = {group_bytes}, blocks = {num_blocks}");
    let layout = Layout::create(&dir, spec)?;

    {
        let mut backend = IoUringBackend::new(layout, 1)?;
        let pat = vec![0xA5_u8; group_bytes];
        for blk in 0..num_blocks {
            backend.write_from_host(GroupKey::new(0, blk, 0, KvKind::K), &pat)?;
        }
    }

    // 2026-09-25: `IoUringBackend::new` takes the layout by value, so each
    // queue depth reopens it.
    for &qd in &[1usize, 2, 4, 8, 16, 32] {
        let layout = Layout::open(&dir, spec)?;
        let mut backend = IoUringBackend::new(layout, qd)?;
        backend.drop_pagecache();

        // 2026-09-25: Blocks are picked by a multiplicative hash of the read
        // index; every read lands in the same device buffer.
        let n_iters: usize = 1024;
        let dev = DeviceBuffer::new(group_bytes)?;
        let reqs: Vec<ReadRequest> = (0..n_iters)
            .map(|i| {
                let blk = ((i as u32).wrapping_mul(2_654_435_761)) % num_blocks;
                ReadRequest {
                    group: GroupKey::new(0, blk, 0, KvKind::K),
                    dst_dev_ptr: dev.ptr,
                }
            })
            .collect();

        let t = Instant::now();
        backend.read(&reqs, _ctx.stream)?;
        let dt = t.elapsed().as_secs_f64();
        let mib = (n_iters * group_bytes) as f64 / (1024.0 * 1024.0);
        eprintln!(
            "qd={qd:>3}: {:>8.1} MiB/s ({n_iters} reads of {group_bytes}B in {dt:.3}s)",
            mib / dt
        );
    }

    std::fs::remove_dir_all(&dir).ok();
    Ok(())
}
