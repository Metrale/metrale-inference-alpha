// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `KernelLaunch`, the builder for kernel launches on any
//! `GpuBackend`. Each argument is recorded as a buffer or as scalar bytes, so the
//! Metal backend can bind buffers with `setBuffer:offset:atIndex:` and scalars with
//! `setBytes:length:atIndex:`.
//!
//! Owner: gpu-runtime.
//! Invariants:
//! - Each `arg_*` call adds exactly one `KernelArg` to the launch; a
//!   `CUtensorMap` occupies 16 consecutive storage slots but is still one argument.
//! - Arguments reach `GpuBackend::launch_typed` in the order they were added.
//!
//! Usage:
//!
//! ```ignore
//! KernelLaunch::new(gpu, kernel)
//!     .grid([num_tokens, 1, 1])
//!     .block([256, 1, 1])
//!     .arg_ptr(input)
//!     .arg_u32(hidden_size)
//!     .arg_f32(eps)
//!     .launch(stream)?;
//! ```

use anyhow::Result;

use crate::gpu::{DevicePtr, GpuBackend, KernelArg, KernelHandle};

/// 2026-09-25: Per-arg metadata: the first `storage` slot it occupies, and for a
/// scalar its byte width. Buffer args set `is_buffer` and leave `byte_len` 0.
struct ArgKind {
    is_buffer: bool,
    /// 2026-09-25: Byte count for scalar args: 4 for u32/i32/f32, 8 for u64, 128
    /// for a `CUtensorMap`.
    byte_len: u16,
    /// 2026-09-25: Starting slot in `storage`. A `CUtensorMap` occupies 16
    /// consecutive slots, so an arg's slot is not its position in `kinds`.
    slot: u32,
}

/// 2026-09-25: Builder for a kernel launch: grid, block, dynamic shared memory
/// and typed arguments. `launch()` passes the arguments as `&[KernelArg]` to
/// `GpuBackend::launch_typed`. Grid and block default to `[1, 1, 1]`, shared
/// memory to 0.
pub struct KernelLaunch<'a> {
    gpu: &'a dyn GpuBackend,
    kernel: KernelHandle,
    grid: [u32; 3],
    block: [u32; 3],
    shared_mem: u32,
    /// 2026-09-25: Argument bytes in u64 slots: little-endian for scalars, the
    /// device address for buffers.
    storage: Vec<u64>,
    /// 2026-09-25: One entry per argument, in order.
    kinds: Vec<ArgKind>,
}

impl<'a> KernelLaunch<'a> {
    pub fn new(gpu: &'a dyn GpuBackend, kernel: KernelHandle) -> Self {
        Self {
            gpu,
            kernel,
            grid: [1, 1, 1],
            block: [1, 1, 1],
            shared_mem: 0,
            storage: Vec::with_capacity(16),
            kinds: Vec::with_capacity(16),
        }
    }

    pub fn grid(mut self, grid: [u32; 3]) -> Self {
        self.grid = grid;
        self
    }

    pub fn block(mut self, block: [u32; 3]) -> Self {
        self.block = block;
        self
    }

    pub fn shared_mem(mut self, bytes: u32) -> Self {
        self.shared_mem = bytes;
        self
    }

    /// 2026-09-25: Add a device buffer argument.
    pub fn arg_ptr(mut self, p: DevicePtr) -> Self {
        let slot = self.storage.len() as u32;
        self.storage.push(p.0);
        self.kinds.push(ArgKind {
            is_buffer: true,
            byte_len: 0,
            slot,
        });
        self
    }

    /// 2026-09-25: Add a 128-byte `CUtensorMap` by value, for a kernel parameter
    /// declared `__grid_constant__ const CUtensorMap`. The bytes fill 16
    /// consecutive slots and form one argument.
    pub fn arg_tensormap(mut self, map: &[u8; 128]) -> Self {
        let slot = self.storage.len() as u32;
        for c in map.chunks(8) {
            let mut w = [0u8; 8];
            w.copy_from_slice(c);
            self.storage.push(u64::from_le_bytes(w));
        }
        self.kinds.push(ArgKind {
            is_buffer: false,
            byte_len: 128,
            slot,
        });
        self
    }

    pub fn arg_u32(mut self, v: u32) -> Self {
        let slot = self.storage.len() as u32;
        self.storage.push(v as u64);
        self.kinds.push(ArgKind {
            is_buffer: false,
            byte_len: 4,
            slot,
        });
        self
    }

    pub fn arg_u64(mut self, v: u64) -> Self {
        let slot = self.storage.len() as u32;
        self.storage.push(v);
        self.kinds.push(ArgKind {
            is_buffer: false,
            byte_len: 8,
            slot,
        });
        self
    }

    pub fn arg_i32(mut self, v: i32) -> Self {
        let slot = self.storage.len() as u32;
        self.storage.push(v as u32 as u64);
        self.kinds.push(ArgKind {
            is_buffer: false,
            byte_len: 4,
            slot,
        });
        self
    }

    pub fn arg_f32(mut self, v: f32) -> Self {
        let slot = self.storage.len() as u32;
        self.storage.push(f32::to_bits(v) as u64);
        self.kinds.push(ArgKind {
            is_buffer: false,
            byte_len: 4,
            slot,
        });
        self
    }

    /// 2026-09-25: Launch on `stream` through `GpuBackend::launch_typed`.
    ///
    /// With `debug_sync_kernels` on, the stream is synchronised after a
    /// successful launch and a fault is returned with the launch geometry and a
    /// backtrace.
    pub fn launch(self, stream: u64) -> Result<()> {
        let mut args: Vec<KernelArg<'_>> = Vec::with_capacity(self.kinds.len());
        for kind in self.kinds.iter() {
            let slot = &self.storage[kind.slot as usize];
            if kind.is_buffer {
                args.push(KernelArg::Buffer(DevicePtr(*slot)));
            } else {
                // 2026-09-25: SAFETY: the arg's bytes start at `slot` and run
                // `byte_len` bytes through consecutive slots of
                // `self.storage`, which is not modified while `args` lives.
                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        slot as *const u64 as *const u8,
                        kind.byte_len as usize,
                    )
                };
                args.push(KernelArg::Bytes(bytes));
            }
        }
        let r = self.gpu.launch_typed(
            self.kernel,
            self.grid,
            self.block,
            self.shared_mem,
            stream,
            &args,
        );
        // 2026-09-25: `METRALE_DEBUG_SYNC_KERNELS=1` on the CUDA backend: an
        // asynchronous fault is reported at the launch that caused it.
        if r.is_ok() && self.gpu.debug_sync_kernels() {
            self.gpu.synchronize(stream).map_err(|e| {
                let bt = std::backtrace::Backtrace::force_capture();
                anyhow::anyhow!(
                    "METRALE_DEBUG_SYNC_KERNELS: async GPU fault immediately after kernel launch \
                     grid={:?} block={:?} shared_mem={}: {e}\nLAUNCH BACKTRACE:\n{bt}",
                    self.grid,
                    self.block,
                    self.shared_mem
                )
            })?;
        }
        r
    }
}

/// 2026-09-25: `a / b`, rounded up.
pub fn div_ceil(a: u32, b: u32) -> u32 {
    a.div_ceil(b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::mock::MockGpuBackend;

    #[test]
    fn test_kernel_launch_builder() {
        let gpu = MockGpuBackend::new();
        let kernel = gpu.kernel("test", "test_kernel").unwrap();

        let result = KernelLaunch::new(&gpu, kernel)
            .grid([4, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(DevicePtr(0x1000))
            .arg_u32(42)
            .arg_f32(1.5)
            .launch(0);

        assert!(result.is_ok());
        assert_eq!(gpu.launch_count(), 1);
    }

    /// 2026-09-25: A 128-byte by-value argument (a `CUtensorMap`) occupies 16
    /// consecutive slots with one param entry, and its bytes round-trip.
    #[test]
    fn a_128_byte_arg_is_not_truncated() {
        use crate::gpu::{KernelArg, pack_kernel_args};
        let map: Vec<u8> = (0..128u16).map(|i| (i * 7 % 251) as u8).collect();
        let args = [
            KernelArg::Buffer(DevicePtr(0xDEAD_BEEF)),
            KernelArg::Bytes(&map),
            KernelArg::Bytes(&42u32.to_le_bytes()),
        ];
        let (storage, starts) = pack_kernel_args(&args);

        assert_eq!(
            starts.len(),
            3,
            "one param entry per argument, not per slot"
        );
        assert_eq!(starts, vec![0, 1, 17], "the map occupies slots 1..=16");
        assert_eq!(storage.len(), 18);
        assert_eq!(storage[0], 0xDEAD_BEEF);

        let round_trip: Vec<u8> = storage[starts[1]..starts[1] + 16]
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect();
        assert_eq!(
            round_trip, map,
            "128-byte struct arg was corrupted or truncated"
        );
        assert_eq!(
            storage[starts[2]] as u32, 42,
            "the arg after it is still intact"
        );
    }

    /// 2026-09-25: Through the builder, a tensormap between two ordinary args
    /// leaves both intact and counts as one argument.
    #[test]
    fn a_tensormap_arg_does_not_shift_the_args_around_it() {
        let gpu = MockGpuBackend::new();
        let kernel = gpu.kernel("test", "tma_kernel").unwrap();
        let map = [0xABu8; 128];

        let b = KernelLaunch::new(&gpu, kernel)
            .arg_ptr(DevicePtr(0x1000))
            .arg_tensormap(&map)
            .arg_u32(7);

        assert_eq!(b.kinds.len(), 3, "one param entry per arg, not per slot");
        assert_eq!(b.storage.len(), 1 + 16 + 1);
        assert_eq!(b.kinds[0].slot, 0);
        assert_eq!(b.kinds[1].slot, 1);
        assert_eq!(b.kinds[1].byte_len, 128);
        assert_eq!(
            b.kinds[2].slot, 17,
            "the arg after the map must not be shifted"
        );
        assert_eq!(b.storage[b.kinds[2].slot as usize] as u32, 7);
        assert!(b.launch(0).is_ok());
    }

    /// 2026-09-25: A zero-length byte arg still gets a slot to point at.
    #[test]
    fn an_empty_byte_arg_still_gets_one_slot() {
        use crate::gpu::{KernelArg, pack_kernel_args};
        let (storage, starts) =
            pack_kernel_args(&[KernelArg::Bytes(&[]), KernelArg::Bytes(&[9u8])]);
        assert_eq!(starts, vec![0, 1]);
        assert_eq!(storage.len(), 2);
        assert_eq!(storage[1], 9);
    }

    #[test]
    fn test_div_ceil() {
        assert_eq!(div_ceil(10, 3), 4);
        assert_eq!(div_ceil(9, 3), 3);
        assert_eq!(div_ceil(1, 256), 1);
        assert_eq!(div_ceil(256, 256), 1);
        assert_eq!(div_ceil(257, 256), 2);
    }
}
