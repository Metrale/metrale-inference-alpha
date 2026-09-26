// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tensor-parallel (head-parallel) sharding of the Gated-DeltaNet
//! (SSM / linear-attention) layers.
//!
//! Owner: model-arch (tensor parallelism).
//! Invariants:
//! - A segmented slice keeps the segment order and packs each segment's local
//!   rows back to back; every segment must divide by `tp_size`, else nothing
//!   is allocated and an error is returned.

use anyhow::{Result, ensure};
use metrale_config::ModelConfig;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

use super::{BF16_BYTES, TpShardKind, shard_dense_bf16};

// 2026-09-25: Each TP rank owns a contiguous range of key and value heads and
// runs the recurrence on its local heads; the layer all-reduces after
// `out_proj` (`ssm_tp_all_reduce` in model-layers' `qwen3_ssm`).
//
// The in-projection is segmented, not one flat matrix:
//
//   in_proj_qkv : [Q | K | V]        rows = nk·kd + nk·kd + nv·vd  (= conv_dim)
//   in_proj_z   : [Z]                rows = nv·vd
//   → gpu_concat_rows → QKVZ         [Q | K | V | Z]
//
// A plain "first out_dim/tp rows" slice would give rank 0 the whole Q block
// plus part of K, so each segment is sliced by the local head range on its own
// and the local slices are packed in the same order (`segment_copy_plan`).
//
// The depthwise `conv1d` weight `[conv_dim, d_conv]` uses the same segments as
// QKV (one filter per QKV channel). `a_log` / `dt_bias` (`[nv]` FP32) and
// `out_proj` (`[h, nv·vd]`, row-parallel) shard on the value-head axis. The
// Qwen3.5 loaders keep the gated-norm weight `norm` whole on every rank. The
// BA gate buffer is interleaved per key-head group, and a rank's rows start on
// a group boundary, so it slices as one contiguous block.

/// 2026-09-25: Pre-TP-shard GDN (linear-attention / SSM) dimensions
/// reconstructed from `config`.
///
/// As for [`super::TpAttentionDims`]: `topology.rs` divides
/// `linear_num_key_heads` / `linear_num_value_heads` by the TP size before any
/// loader runs, so `config` holds per-rank head counts. The `full_*` fields
/// multiply back up to the pre-shard sizes the segment slicers take. Head dims
/// (`kd`, `vd`) and the hidden size `h` are not divided.
#[derive(Debug, Clone, Copy)]
pub struct TpGdnDims {
    pub tp_rank: usize,
    /// 2026-09-25: `tp_world_size` clamped to `>= 1`.
    pub tp_size: usize,
    /// 2026-09-25: Hidden size, not divided per rank.
    pub h: usize,
    /// 2026-09-25: Key head dim (`linear_key_head_dim`).
    pub kd: usize,
    /// 2026-09-25: Value head dim (`linear_value_head_dim`).
    pub vd: usize,
    /// 2026-09-25: Per-rank key heads (Q and K share this count).
    pub local_nk: usize,
    /// 2026-09-25: Full pre-shard key heads = `local_nk * tp_size`.
    pub full_nk: usize,
    /// 2026-09-25: Per-rank value heads.
    pub local_nv: usize,
    /// 2026-09-25: Full pre-shard value heads = `local_nv * tp_size`.
    pub full_nv: usize,
}

impl TpGdnDims {
    pub fn from_config(config: &ModelConfig) -> Self {
        let tp_size = config.tp_world_size.max(1);
        let local_nk = config.linear_num_key_heads;
        let local_nv = config.linear_num_value_heads;
        Self {
            tp_rank: config.tp_rank,
            tp_size,
            h: config.hidden_size,
            kd: config.linear_key_head_dim,
            vd: config.linear_value_head_dim,
            local_nk,
            full_nk: local_nk * tp_size,
            local_nv,
            full_nv: local_nv * tp_size,
        }
    }

    /// 2026-09-25: Full (pre-shard) key projection width: `full_nk * kd`.
    pub fn full_key_dim(&self) -> usize {
        self.full_nk * self.kd
    }
    /// 2026-09-25: Local key projection width: `local_nk * kd`.
    pub fn local_key_dim(&self) -> usize {
        self.local_nk * self.kd
    }
    /// 2026-09-25: Full (pre-shard) value projection width: `full_nv * vd`.
    pub fn full_value_dim(&self) -> usize {
        self.full_nv * self.vd
    }
    /// 2026-09-25: Local value projection width: `local_nv * vd`.
    pub fn local_value_dim(&self) -> usize {
        self.local_nv * self.vd
    }
    /// 2026-09-25: Full conv / QKV width: `2*full_nk*kd + full_nv*vd`.
    pub fn full_conv_dim(&self) -> usize {
        2 * self.full_key_dim() + self.full_value_dim()
    }
    /// 2026-09-25: Local conv / QKV width: `2*local_nk*kd + local_nv*vd`.
    pub fn local_conv_dim(&self) -> usize {
        2 * self.local_key_dim() + self.local_value_dim()
    }
    /// 2026-09-25: Full QKVZ out dim: `2*full_nk*kd + 2*full_nv*vd`.
    pub fn full_qkvz_out(&self) -> usize {
        self.full_conv_dim() + self.full_value_dim()
    }
    /// 2026-09-25: Local QKVZ out dim: `2*local_nk*kd + 2*local_nv*vd`.
    pub fn local_qkvz_out(&self) -> usize {
        self.local_conv_dim() + self.local_value_dim()
    }

    /// 2026-09-25: Full-row segment list for the `[Q|K|V]` in-projection.
    pub(crate) fn qkv_segments(&self) -> [usize; 3] {
        [
            self.full_key_dim(),
            self.full_key_dim(),
            self.full_value_dim(),
        ]
    }
    /// 2026-09-25: Full-row segment list for the concatenated `[Q|K|V|Z]` in-projection.
    pub(crate) fn qkvz_segments(&self) -> [usize; 4] {
        [
            self.full_key_dim(),
            self.full_key_dim(),
            self.full_value_dim(),
            self.full_value_dim(),
        ]
    }
}

/// 2026-09-25: A single device-to-device copy in a segmented-slice plan, in bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CopyOp {
    pub(crate) src_off: usize,
    pub(crate) dst_off: usize,
    pub(crate) len: usize,
}

/// 2026-09-25: Build the copy plan for a segmented row-slice.
///
/// `segments` lists the full (pre-shard) row count of each contiguous block
/// (Q, K, V[, Z] for QKVZ). Each block is sliced on its own to the local
/// rank's range `[tp_rank * seg/tp, (tp_rank+1) * seg/tp)` and the local
/// slices are packed back-to-back into the output buffer, preserving segment
/// order. `row_bytes` is the byte width of one row (`in_dim * elem_bytes`).
///
/// Returns `(ops, local_total_rows)`. Errors when `tp_rank >= tp_size` or a
/// segment is not divisible by `tp_size`. `TpGdnDims` segments are multiples of
/// `tp_size` by construction (`full_* = local_* * tp_size`).
pub(crate) fn segment_copy_plan(
    segments: &[usize],
    row_bytes: usize,
    tp_rank: usize,
    tp_size: usize,
) -> Result<(Vec<CopyOp>, usize)> {
    ensure!(tp_rank < tp_size, "tp_rank {tp_rank} >= tp_size {tp_size}");
    let mut ops = Vec::with_capacity(segments.len());
    let mut src_rows = 0usize;
    let mut dst_rows = 0usize;
    for (i, &seg) in segments.iter().enumerate() {
        ensure!(
            seg.is_multiple_of(tp_size),
            "segment {i} ({seg} rows) not divisible by tp_size {tp_size}",
        );
        let local = seg / tp_size;
        ops.push(CopyOp {
            src_off: (src_rows + tp_rank * local) * row_bytes,
            dst_off: dst_rows * row_bytes,
            len: local * row_bytes,
        });
        src_rows += seg;
        dst_rows += local;
    }
    Ok((ops, dst_rows))
}

/// 2026-09-25: Execute a segmented row-slice on the GPU. `row_elems` is the number of
/// elements per row (`in_dim`); `elem_bytes` its size (2 = BF16, 4 = FP32).
/// Returns `(local_ptr, local_total_rows)`. For `tp_size <= 1` returns the
/// source untouched (no allocation, caller must not double-free).
fn slice_segments(
    src: DevicePtr,
    segments: &[usize],
    row_elems: usize,
    elem_bytes: usize,
    tp_rank: usize,
    tp_size: usize,
    gpu: &dyn GpuBackend,
) -> Result<(DevicePtr, usize)> {
    let full_rows: usize = segments.iter().sum();
    if tp_size <= 1 {
        return Ok((src, full_rows));
    }
    let row_bytes = row_elems * elem_bytes;
    let (ops, local_rows) = segment_copy_plan(segments, row_bytes, tp_rank, tp_size)?;
    let dst = gpu.alloc(local_rows * row_bytes)?;
    tracing::debug!(
        target: "metrale_model_arch::tp_shard",
        ?segments, full_rows, row_elems, elem_bytes, local_rows,
        tp_rank, tp_size, src = src.0,
        "gdn segmented row-slice"
    );
    for op in &ops {
        gpu.copy_d2d(src.offset(op.src_off), dst.offset(op.dst_off), op.len)?;
    }
    Ok((dst, local_rows))
}

/// 2026-09-25: Shard the `[Q|K|V]` (`in_proj_qkv`) BF16 weight `[full_conv_dim, h]` to the
/// local rank's `[local_conv_dim, h]`, slicing Q, K and V independently by the
/// local head range. Returns `(ptr, local_rows, h)`.
pub fn shard_gdn_qkv_rows(
    src: DevicePtr,
    dims: &TpGdnDims,
    gpu: &dyn GpuBackend,
) -> Result<(DevicePtr, usize, usize)> {
    let (ptr, rows) = slice_segments(
        src,
        &dims.qkv_segments(),
        dims.h,
        BF16_BYTES,
        dims.tp_rank,
        dims.tp_size,
        gpu,
    )?;
    Ok((ptr, rows, dims.h))
}

/// 2026-09-25: Shard the concatenated `[Q|K|V|Z]` (`in_proj_qkvz`) BF16 weight
/// `[full_qkvz_out, h]` to the local rank's `[local_qkvz_out, h]`, slicing all
/// four segments independently. Returns `(ptr, local_rows, h)`.
pub fn shard_gdn_qkvz_rows(
    src: DevicePtr,
    dims: &TpGdnDims,
    gpu: &dyn GpuBackend,
) -> Result<(DevicePtr, usize, usize)> {
    let (ptr, rows) = slice_segments(
        src,
        &dims.qkvz_segments(),
        dims.h,
        BF16_BYTES,
        dims.tp_rank,
        dims.tp_size,
        gpu,
    )?;
    Ok((ptr, rows, dims.h))
}

/// 2026-09-25: Shard the BA gate BF16 weight `[2*full_nv, h]` to `[2*local_nv, h]`.
///
/// `interleave_ba` lays it out per key-head group (`[β₀..β_{vpg-1},
/// α₀..α_{vpg-1}]` per group, `vpg = nv/nk`). Rank `r` owns key-head groups
/// `[r*local_nk, (r+1)*local_nk)`, which are the contiguous row range
/// `[r*2*local_nv, (r+1)*2*local_nv)`, so one contiguous slice keeps the
/// interleave.
pub fn shard_gdn_ba_rows(
    src: DevicePtr,
    dims: &TpGdnDims,
    gpu: &dyn GpuBackend,
) -> Result<(DevicePtr, usize, usize)> {
    // 2026-09-25: `full_nk` divisible by `tp_size` means each rank gets whole groups.
    ensure!(
        dims.full_nk.is_multiple_of(dims.tp_size),
        "BA: full_nk {} not divisible by tp_size {}",
        dims.full_nk,
        dims.tp_size,
    );
    let (ptr, rows) = slice_segments(
        src,
        &[2 * dims.full_nv],
        dims.h,
        BF16_BYTES,
        dims.tp_rank,
        dims.tp_size,
        gpu,
    )?;
    Ok((ptr, rows, dims.h))
}

/// 2026-09-25: Shard the depthwise `conv1d` BF16 weight `[full_conv_dim, d_conv]` to
/// `[local_conv_dim, d_conv]`. Its channels are the QKV channels (one filter
/// per channel), so it uses the same `[Q|K|V]` segments as the QKV
/// in-projection.
pub fn shard_gdn_conv_rows(
    src: DevicePtr,
    dims: &TpGdnDims,
    d_conv: usize,
    gpu: &dyn GpuBackend,
) -> Result<(DevicePtr, usize, usize)> {
    let (ptr, rows) = slice_segments(
        src,
        &dims.qkv_segments(),
        d_conv,
        BF16_BYTES,
        dims.tp_rank,
        dims.tp_size,
        gpu,
    )?;
    Ok((ptr, rows, d_conv))
}

/// 2026-09-25: Shard a per-value-head 1D vector on the value-head axis:
/// `[full_nv * unit]` → `[local_nv * unit]`, where `unit` is the number of
/// elements per value head and `elem_bytes` their size. The loaders call it for
/// `a_log` / `dt_bias` (`unit = 1`, `elem_bytes = 4`). Returns
/// `(ptr, local_len_elems)`.
pub fn shard_gdn_value_vector(
    src: DevicePtr,
    dims: &TpGdnDims,
    unit: usize,
    elem_bytes: usize,
    gpu: &dyn GpuBackend,
) -> Result<(DevicePtr, usize)> {
    let full_len = dims.full_nv * unit;
    if dims.tp_size <= 1 {
        return Ok((src, full_len));
    }
    ensure!(
        dims.tp_rank < dims.tp_size,
        "tp_rank {} >= tp_size {}",
        dims.tp_rank,
        dims.tp_size,
    );
    let local_len = dims.local_nv * unit;
    let local_bytes = local_len * elem_bytes;
    let dst = gpu.alloc(local_bytes)?;
    let src_off = dims.tp_rank * local_bytes;
    tracing::debug!(
        target: "metrale_model_arch::tp_shard",
        full_nv = dims.full_nv, local_nv = dims.local_nv, unit, elem_bytes,
        full_len, local_len, local_bytes, src_off, tp_rank = dims.tp_rank,
        src = src.0,
        "gdn value-vector shard (per-value-head axis)"
    );
    gpu.copy_d2d(src.offset(src_off), dst, local_bytes)?;
    Ok((dst, local_len))
}

/// 2026-09-25: Shard the `out_proj` BF16 weight `[h, full_value_dim]` row-parallel on its
/// input dim (value_dim). Rank `r` keeps columns
/// `[r*local_value_dim, (r+1)*local_value_dim)` of every output row; the layer
/// sums the partial products with an all-reduce after the GEMM. Returns
/// `(ptr, h, local_value_dim)`.
pub fn shard_gdn_out_proj_row_parallel(
    src: DevicePtr,
    dims: &TpGdnDims,
    gpu: &dyn GpuBackend,
) -> Result<(DevicePtr, usize, usize)> {
    shard_dense_bf16(
        src,
        dims.h,
        dims.full_value_dim(),
        TpShardKind::RowParallel,
        dims.tp_rank,
        dims.tp_size,
        gpu,
    )
}
