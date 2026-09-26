// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: One n-gram lookup table on device: BF16 as shipped,
//! FP8-quantized at load, or NVMe-backed by a bounded row cache.
//!
//! Owner: model-layers (n-gram embedding).
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_gpu_runtime::gpu::GpuBackend;

use crate::weight_map::{DenseWeight, Fp8DenseWeight};

/// 2026-09-25: One n-gram lookup table. `Fp8` holds E4M3 rows with a
/// per-row f32 scale, which the gather (`batched_embed_fp8`) applies.
pub enum NgramTable {
    Bf16(DenseWeight),
    Fp8(Fp8DenseWeight),
    /// 2026-09-25: NVMe-backed: a bounded set of rows is resident in a
    /// GPU-addressable arena. The host maps row ids to arena slots
    /// (`NgramRowCache::resolve`) and the same gather kernels read the arena
    /// by slot.
    #[cfg(feature = "cuda")]
    Cached(Box<metrale_storage::NgramRowCache>),
}

impl NgramTable {
    /// 2026-09-25: Quantize a BF16 table to FP8 on the GPU with
    /// `quantize_bf16_to_fp8`. `w` is not freed.
    pub fn quantize_bf16(
        w: &DenseWeight,
        rows: usize,
        dim: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<Self> {
        let k = gpu.kernel("gemv_fp8w", "quantize_bf16_to_fp8")?;
        Ok(Self::Fp8(crate::weight_map::quantize_to_fp8(
            w, rows, dim, gpu, k, stream,
        )?))
    }
}
