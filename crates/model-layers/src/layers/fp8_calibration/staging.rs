// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Device copies of the calibration window's BF16 K/V and slot
//! mappings, replayed through `ops::reshape_and_cache_fp8` at the frozen scale.
//!
//! Rewriting from the original BF16 quantizes each window entry once, at the
//! final scale, with the same write the live path uses. Replay runs in write
//! order, so when two staged batches wrote the same slot the later one wins,
//! as it did originally. The batch that reaches the window is written by the
//! caller after the replay, at the frozen scale, and is not staged.
//!
//! Cost: `capacity_tokens * elems_per_token * 2 B` for each of K and V, plus
//! `8 B` per token of slot mapping, per attention layer, allocated on the
//! first staged batch and freed at the freeze.
//!
//! Owner: model-layers (FP8 KV cache).
//! Invariants:
//! - `used_tokens <= capacity_tokens`; a batch that does not fit is declined.
//! - `batches` is in staging order and tiles `0..used_tokens` without gaps.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use crate::layers::ops;

/// 2026-09-25: Cap on the calibration window, and so on the staging size per
/// layer. `Fp8KvCalibration::new` clamps the window to it.
pub(super) const MAX_STAGED_TOKENS: usize = 4096;

const BF16: usize = 2;
/// 2026-09-25: Bytes per `slot_mapping` entry: the kernel reads `const long long*`
/// (`reshape_and_cache_flash_fp8` in `kernels/gb10/common/reshape_and_cache.cu`).
const SLOT: usize = 8;

/// 2026-09-25: Where the caller's live FP8 write goes, so the replay can
/// repeat it (`qwen3_attention/decode/write_kv_cache_fp8.rs` builds it).
#[derive(Debug, Clone, Copy)]
pub struct Fp8KvWriteTarget {
    /// 2026-09-25: The layer's `reshape_cache_k` handle.
    pub kernel: KernelHandle,
    pub k_pool: DevicePtr,
    pub v_pool: DevicePtr,
    pub block_size: u32,
    /// 2026-09-25: Pool stride per block, in elements (`PagedKvCache::cache_stride`).
    pub cache_stride: u64,
    /// 2026-09-25: Source K row stride in elements, as passed to the live write.
    pub key_stride: u32,
    /// 2026-09-25: Source V row stride in elements, as passed to the live write.
    pub value_stride: u32,
    /// 2026-09-25: This batch's `slot_mapping`, staged with the K/V so the
    /// replay writes the same cache entries.
    pub slot: DevicePtr,
}

#[derive(Debug, Clone, Copy)]
struct StagedBatch {
    offset_tokens: usize,
    num_tokens: u32,
}

#[derive(Debug)]
pub(super) struct KvStaging {
    k: DevicePtr,
    v: DevicePtr,
    slots: DevicePtr,
    capacity_tokens: usize,
    elems_per_token: usize,
    used_tokens: usize,
    batches: Vec<StagedBatch>,
}

impl Default for KvStaging {
    fn default() -> Self {
        Self {
            k: DevicePtr(0),
            v: DevicePtr(0),
            slots: DevicePtr(0),
            capacity_tokens: 0,
            elems_per_token: 0,
            used_tokens: 0,
            batches: Vec::new(),
        }
    }
}

impl KvStaging {
    pub(super) fn used_tokens(&self) -> usize {
        self.used_tokens
    }

    /// 2026-09-25: Copy one pre-freeze batch aside, packed. Allocates the
    /// buffers for `capacity_tokens` on the first call. Returns `Ok(false)`
    /// without copying when the batch's `elems_per_token` differs from the
    /// first batch's or it does not fit; the freeze then leaves that batch's
    /// entries at the provisional scale.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn stage(
        &mut self,
        gpu: &dyn GpuBackend,
        k: DevicePtr,
        v: DevicePtr,
        num_tokens: u32,
        elems_per_token: usize,
        target: &Fp8KvWriteTarget,
        capacity_tokens: usize,
        stream: u64,
    ) -> Result<bool> {
        if self.capacity_tokens == 0 {
            self.k = gpu.alloc(capacity_tokens * elems_per_token * BF16)?;
            self.v = gpu.alloc(capacity_tokens * elems_per_token * BF16)?;
            self.slots = gpu.alloc(capacity_tokens * SLOT)?;
            self.capacity_tokens = capacity_tokens;
            self.elems_per_token = elems_per_token;
        }
        let n = num_tokens as usize;
        if elems_per_token != self.elems_per_token || self.used_tokens + n > self.capacity_tokens {
            return Ok(false);
        }

        let row = elems_per_token * BF16;
        let dst_off = self.used_tokens * row;
        copy_rows(
            gpu,
            k,
            target.key_stride,
            self.k.offset(dst_off),
            row,
            n,
            stream,
        )?;
        copy_rows(
            gpu,
            v,
            target.value_stride,
            self.v.offset(dst_off),
            row,
            n,
            stream,
        )?;
        gpu.copy_d2d_async(
            target.slot,
            self.slots.offset(self.used_tokens * SLOT),
            n * SLOT,
            stream,
        )?;

        self.batches.push(StagedBatch {
            offset_tokens: self.used_tokens,
            num_tokens,
        });
        self.used_tokens += n;
        Ok(true)
    }

    /// 2026-09-25: Rewrite every staged batch at the frozen scale, in staging
    /// order, then release the buffers. Returns the number of tokens rewritten.
    /// On a launch error the buffers are not released.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn replay_and_release(
        &mut self,
        gpu: &dyn GpuBackend,
        target: &Fp8KvWriteTarget,
        num_kv_heads: u32,
        head_dim: u32,
        k_scale: f32,
        v_scale: f32,
        stream: u64,
    ) -> Result<usize> {
        let rewritten = self.used_tokens;
        let row = self.elems_per_token * BF16;
        for batch in std::mem::take(&mut self.batches) {
            let off = batch.offset_tokens;
            ops::reshape_and_cache_fp8(
                gpu,
                target.kernel,
                self.k.offset(off * row),
                self.v.offset(off * row),
                target.k_pool,
                target.v_pool,
                self.slots.offset(off * SLOT),
                batch.num_tokens,
                num_kv_heads,
                head_dim,
                target.block_size,
                k_scale,
                v_scale,
                // 2026-09-25: The staging copies are packed, whatever the live
                // source strides were.
                self.elems_per_token as u32,
                self.elems_per_token as u32,
                target.cache_stride,
                stream,
            )?;
        }
        self.release(gpu)?;
        Ok(rewritten)
    }

    /// 2026-09-25: Free the staging buffers. A no-op when nothing was allocated.
    pub(super) fn release(&mut self, gpu: &dyn GpuBackend) -> Result<()> {
        if self.capacity_tokens == 0 {
            return Ok(());
        }
        gpu.free(self.k)?;
        gpu.free(self.v)?;
        gpu.free(self.slots)?;
        *self = Self::default();
        Ok(())
    }
}

/// 2026-09-25: Copy `rows` rows of `row_bytes` from a source with `src_stride`
/// BF16 elements between rows into a packed destination.
fn copy_rows(
    gpu: &dyn GpuBackend,
    src: DevicePtr,
    src_stride: u32,
    dst: DevicePtr,
    row_bytes: usize,
    rows: usize,
    stream: u64,
) -> Result<()> {
    let src_pitch = src_stride as usize * BF16;
    if src_pitch == row_bytes {
        return gpu.copy_d2d_async(src, dst, row_bytes * rows, stream);
    }
    gpu.copy_d2d_2d_async(src, src_pitch, dst, row_bytes, row_bytes, rows, stream)
}
