// SPDX-License-Identifier: MIT OR Apache-2.0
// provenance-id: 526f6e616c6420522e205374657369616b

//! 2026-09-25: The engram `wkv` as raw Q2_K blocks on the device, for the
//! single-token projection (`engram_v41_wkv_q2k_gemv`). The multi-token
//! projection reads the store's bf16 `wkv`.
//!
//! Owner: model-arch weight loader.
//! Invariants: none beyond the types.

use anyhow::{Context, Result};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_model_weights::weights::expert_stream::{ShardFiles, pread};

/// 2026-09-25: `blk.{layer}.engram_wkv.weight` raw on the device, or null when
/// the tensor is not Q2_K in these shards or `METRALE_DS41_ENGRAM_Q2K=0`.
pub(super) fn engram_wkv_q2k(
    gpu: &dyn GpuBackend,
    files: &ShardFiles,
    layer: usize,
) -> Result<DevicePtr> {
    if std::env::var("METRALE_DS41_ENGRAM_Q2K").is_ok_and(|v| v == "0") {
        return Ok(DevicePtr(0));
    }
    let name = format!("blk.{layer}.engram_wkv.weight");
    let Some((shard, off, bytes)) = files.locate_q2k(&name) else {
        tracing::info!("engram layer {layer}: {name} is not Q2_K here; the projection reads bf16");
        return Ok(DevicePtr(0));
    };
    let mut host = vec![0u8; bytes];
    pread(files.file(shard), off, &mut host).with_context(|| name.clone())?;
    let dev = gpu.alloc(bytes)?;
    gpu.copy_h2d(&host, dev)?;
    tracing::info!(
        "engram layer {layer}: wkv raw Q2_K on the device ({:.1} MiB) for the single-token projection",
        bytes as f64 / 1048576.0
    );
    Ok(dev)
}
