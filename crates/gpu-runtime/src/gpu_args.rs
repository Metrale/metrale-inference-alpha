// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Packing typed kernel arguments into the driver's u64 parameter
//! slots.
//!
//! Owner: gpu-runtime.
//! Invariants:
//! - `starts` has one entry per argument, in order, pointing at its first slot.
//! - Every byte of every argument is copied; none is truncated.

use crate::gpu::KernelArg;

/// 2026-09-25: Pack typed args into u64 slots; returns the slots and each
/// argument's starting slot index.
///
/// A buffer takes one slot. A byte argument takes `ceil(len / 8)` consecutive
/// slots (one when empty), little-endian, the last zero-padded; a 128-byte
/// `CUtensorMap` takes 16. Each argument contributes one entry to `starts`.
pub fn pack_kernel_args(args: &[KernelArg<'_>]) -> (Vec<u64>, Vec<usize>) {
    let total_slots: usize = args
        .iter()
        .map(|a| match a {
            KernelArg::Buffer(_) => 1,
            KernelArg::Bytes(b) => b.len().div_ceil(8).max(1),
        })
        .sum();
    let mut storage: Vec<u64> = Vec::with_capacity(total_slots);
    let mut starts: Vec<usize> = Vec::with_capacity(args.len());
    for arg in args {
        starts.push(storage.len());
        match arg {
            KernelArg::Buffer(p) => storage.push(p.0),
            KernelArg::Bytes(b) => {
                for c in b.chunks(8) {
                    let mut slot = [0u8; 8];
                    slot[..c.len()].copy_from_slice(c);
                    storage.push(u64::from_le_bytes(slot));
                }
                if b.is_empty() {
                    storage.push(0);
                }
            }
        }
    }
    (storage, starts)
}
