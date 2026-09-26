// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tensor-parallel weight sharding helpers.
//!
//! Each weight tensor is sliced along one of two axes:
//!
//! - **Column-parallel** (Q/K/V proj, gate_proj, up_proj): weight shape
//!   `[out, in]` becomes `[out / tp, in]`. Rank `r` keeps rows
//!   `[r * out / tp, (r + 1) * out / tp)`, one contiguous slice in row-major
//!   layout, so one `copy_d2d`.
//!
//! - **Row-parallel** (O proj, down_proj): weight `[out, in]` becomes
//!   `[out, in / tp]`. Rank `r` keeps cols `[r * in / tp, (r + 1) * in / tp)`,
//!   one strided copy per row.
//!
//! 1D per-output vectors shard along the output axis of their column-parallel
//! GEMM (`shard_dense_1d_bf16`).
//!
//! The BF16 primitives are here; `quant_shard.rs` slices weights that are
//! already NVFP4 or FP8 block-scaled, and `gdn.rs` the segmented Gated-DeltaNet
//! projections.
//!
//! Owner: model-arch (tensor parallelism).
//! Invariants:
//! - With `tp_size <= 1` every shard function returns its source unchanged
//!   and allocates nothing.

use anyhow::{Result, ensure};
use metrale_config::ModelConfig;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

use metrale_model_layers::weight_map::DenseWeight;

/// 2026-09-25: Bytes per BF16 element.
const BF16_BYTES: usize = 2;

/// 2026-09-25: TP shard kind for a 2D weight `[out_dim, in_dim]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TpShardKind {
    /// 2026-09-25: Replicated: every TP rank holds the full tensor.
    Replicated,
    /// 2026-09-25: Column-parallel: split `out_dim` evenly across ranks. Rank `r` keeps
    /// rows `[r * out_dim / tp, (r + 1) * out_dim / tp)`.
    ColumnParallel,
    /// 2026-09-25: Row-parallel: split `in_dim` evenly across ranks. Rank `r` keeps
    /// cols `[r * in_dim / tp, (r + 1) * in_dim / tp)`.
    RowParallel,
}

/// 2026-09-25: Shard a BF16 dense weight `[out_dim, in_dim]` according to `kind`.
///
/// Returns `(sharded_ptr, sharded_out, sharded_in)`. When `tp_size <= 1`
/// or `kind == Replicated`, returns the source pointer untouched; the caller
/// must not free the source separately (no shard happened).
///
/// Otherwise allocates a new device buffer holding the local rank's slice,
/// copies into it, and returns the new pointer. The caller owns the source
/// and must `gpu.free` it after the shard is built. Errors when
/// `tp_rank >= tp_size` or the split dim is not divisible by `tp_size`.
pub fn shard_dense_bf16(
    src: DevicePtr,
    out_dim: usize,
    in_dim: usize,
    kind: TpShardKind,
    tp_rank: usize,
    tp_size: usize,
    gpu: &dyn GpuBackend,
) -> Result<(DevicePtr, usize, usize)> {
    if tp_size <= 1 || kind == TpShardKind::Replicated {
        return Ok((src, out_dim, in_dim));
    }
    ensure!(tp_rank < tp_size, "tp_rank {tp_rank} >= tp_size {tp_size}");
    match kind {
        TpShardKind::Replicated => unreachable!("handled above"),
        TpShardKind::ColumnParallel => {
            ensure!(
                out_dim.is_multiple_of(tp_size),
                "ColumnParallel: out_dim {out_dim} not divisible by tp_size {tp_size}",
            );
            let local_out = out_dim / tp_size;
            let row_bytes = in_dim * BF16_BYTES;
            let local_bytes = local_out * row_bytes;
            let dst = gpu.alloc(local_bytes)?;
            let src_offset = tp_rank * local_out * row_bytes;
            let src_slice = DevicePtr(src.0 + src_offset as u64);
            gpu.copy_d2d(src_slice, dst, local_bytes)?;
            Ok((dst, local_out, in_dim))
        }
        TpShardKind::RowParallel => {
            ensure!(
                in_dim.is_multiple_of(tp_size),
                "RowParallel: in_dim {in_dim} not divisible by tp_size {tp_size}",
            );
            let local_in = in_dim / tp_size;
            let local_row_bytes = local_in * BF16_BYTES;
            let src_row_bytes = in_dim * BF16_BYTES;
            let local_bytes = out_dim * local_row_bytes;
            let dst = gpu.alloc(local_bytes)?;
            // 2026-09-25: Per-row strided copy: row r of dst comes from row r of src,
            // starting at column `tp_rank * local_in`.
            let col_offset_bytes = tp_rank * local_row_bytes;
            tracing::debug!(
                target: "metrale_model_arch::tp_shard",
                out_dim, in_dim, local_in, src_row_bytes, local_row_bytes,
                tp_rank, tp_size, src = src.0,
                "dense row-parallel shard (per-row strided)"
            );
            for r in 0..out_dim {
                let src_off = r * src_row_bytes + col_offset_bytes;
                let dst_off = r * local_row_bytes;
                gpu.copy_d2d(
                    DevicePtr(src.0 + src_off as u64),
                    DevicePtr(dst.0 + dst_off as u64),
                    local_row_bytes,
                )?;
            }
            Ok((dst, out_dim, local_in))
        }
    }
}

/// 2026-09-25: Shard a 1D BF16 vector `[dim]` on dim 0, for per-output
/// vectors of column-parallel GEMMs. Same ownership rule as `shard_dense_bf16`.
pub fn shard_dense_1d_bf16(
    src: DevicePtr,
    dim: usize,
    tp_rank: usize,
    tp_size: usize,
    gpu: &dyn GpuBackend,
) -> Result<(DevicePtr, usize)> {
    if tp_size <= 1 {
        return Ok((src, dim));
    }
    ensure!(tp_rank < tp_size, "tp_rank {tp_rank} >= tp_size {tp_size}");
    ensure!(
        dim.is_multiple_of(tp_size),
        "shard_dense_1d_bf16: dim {dim} not divisible by tp_size {tp_size}",
    );
    let local_dim = dim / tp_size;
    let local_bytes = local_dim * BF16_BYTES;
    let dst = gpu.alloc(local_bytes)?;
    let src_offset = tp_rank * local_bytes;
    gpu.copy_d2d(DevicePtr(src.0 + src_offset as u64), dst, local_bytes)?;
    Ok((dst, local_dim))
}

/// 2026-09-25: `shard_dense_bf16` on a `DenseWeight`, with the same ownership
/// rule: a new buffer only when a shard happened, and then the caller frees
/// the source.
pub fn shard_dense_weight(
    src: &DenseWeight,
    out_dim: usize,
    in_dim: usize,
    kind: TpShardKind,
    tp_rank: usize,
    tp_size: usize,
    gpu: &dyn GpuBackend,
) -> Result<(DenseWeight, usize, usize)> {
    let (ptr, n, k) = shard_dense_bf16(src.weight, out_dim, in_dim, kind, tp_rank, tp_size, gpu)?;
    Ok((DenseWeight { weight: ptr }, n, k))
}

// 2026-09-25: Dimension helpers shared by the loaders: `TpAttentionDims` and
// `TpMoeDims` rebuild the pre-shard sizes from `config`, and `load_qkvo_tp` /
// `load_qk_norms_tp` run the per-projection loads through a caller closure,
// which picks the format's slicing primitive.

/// 2026-09-25: Pre-TP-shard attention dimensions reconstructed from `config`.
///
/// The server's `serve_phases/topology.rs` divides `num_attention_heads` and
/// `num_key_value_heads` by the TP size before any loader runs, so `config`
/// holds per-rank head counts. The `full_*` fields multiply back up to the
/// pre-shard sizes the shard functions take.
///
/// When `config.attn_gated` is true the Q projection's output includes the
/// gate, doubling it: `full_q_n` includes the gate and `full_o_in` does not.
#[derive(Debug, Clone, Copy)]
pub struct TpAttentionDims {
    pub tp_rank: usize,
    /// 2026-09-25: `tp_world_size` clamped to `>= 1`.
    pub tp_size: usize,
    /// 2026-09-25: Hidden size, not divided per rank.
    pub h: usize,
    pub head_dim: usize,
    /// 2026-09-25: Q-projection output dim, doubled for gated attention.
    pub full_q_n: usize,
    /// 2026-09-25: O-projection input dim, the un-gated attention output:
    /// `num_attention_heads * tp_size * head_dim`, not doubled.
    pub full_o_in: usize,
    /// 2026-09-25: `num_key_value_heads_local * tp_size * head_dim`: full K/V pre-shard.
    pub full_kv_n: usize,
    /// 2026-09-25: `config.attn_gated`.
    pub gated: bool,
}

impl TpAttentionDims {
    pub fn from_config(config: &ModelConfig) -> Self {
        let tp_size = config.tp_world_size.max(1);
        let head_dim = config.head_dim;
        let num_heads_local = config.num_attention_heads;
        let num_kv_heads_local = config.num_key_value_heads;
        let gated = config.attn_gated;
        let attn_out = num_heads_local * tp_size * head_dim;
        let q_factor = if gated { 2 } else { 1 };
        Self {
            tp_rank: config.tp_rank,
            tp_size,
            h: config.hidden_size,
            head_dim,
            full_q_n: attn_out * q_factor,
            full_o_in: attn_out,
            full_kv_n: num_kv_heads_local * tp_size * head_dim,
            gated,
        }
    }

    /// 2026-09-25: `(out_dim, in_dim, kind)` for a given QKVO projection; `None` for
    /// any other name.
    pub fn proj_shape(&self, name: &str) -> Option<(usize, usize, TpShardKind)> {
        match name {
            "q_proj" => Some((self.full_q_n, self.h, TpShardKind::ColumnParallel)),
            "k_proj" | "v_proj" => Some((self.full_kv_n, self.h, TpShardKind::ColumnParallel)),
            "o_proj" => Some((self.h, self.full_o_in, TpShardKind::RowParallel)),
            _ => None,
        }
    }
}

/// 2026-09-25: Sequence the four Q/K/V/O loads via a loader-supplied closure. The
/// closure receives `(name, full_out, full_in, kind)` and returns the
/// loader's representation of that projection (BF16 dense, NVFP4
/// quantized, FP8 block-scaled — varies by format).
///
/// Returns `[Q, K, V, O]`; callers destructure with
/// `let [q, k, v, o] = load_qkvo_tp(config, |name, n, k, kind| { ... })?;`.
pub fn load_qkvo_tp<F, T>(config: &ModelConfig, mut proj_loader: F) -> Result<[T; 4]>
where
    F: FnMut(&str, usize, usize, TpShardKind) -> Result<T>,
{
    let dims = TpAttentionDims::from_config(config);
    let q = proj_loader("q_proj", dims.full_q_n, dims.h, TpShardKind::ColumnParallel)?;
    let k = proj_loader(
        "k_proj",
        dims.full_kv_n,
        dims.h,
        TpShardKind::ColumnParallel,
    )?;
    let v = proj_loader(
        "v_proj",
        dims.full_kv_n,
        dims.h,
        TpShardKind::ColumnParallel,
    )?;
    // 2026-09-25: O proj input dim is the un-gated attention output; for gated
    // attention it differs from `full_q_n`, which includes the gate.
    let o = proj_loader("o_proj", dims.h, dims.full_o_in, TpShardKind::RowParallel)?;
    Ok([q, k, v, o])
}

/// 2026-09-25: Q/K-norm 1D shard pair. The closure receives `(name, full_dim)` and
/// returns the loader's sharded norm — typically a `DenseWeight`.
/// `q_norm` is sharded against `full_q_n`; `k_norm` against `full_kv_n`.
/// Returns `(q_norm, k_norm)`.
pub fn load_qk_norms_tp<F, T>(config: &ModelConfig, mut norm_loader: F) -> Result<(T, T)>
where
    F: FnMut(&str, usize) -> Result<T>,
{
    let dims = TpAttentionDims::from_config(config);
    let q_norm = norm_loader("q_norm", dims.full_q_n)?;
    let k_norm = norm_loader("k_norm", dims.full_kv_n)?;
    Ok((q_norm, k_norm))
}

/// 2026-09-25: Pre-TP-shard dimensions for MoE expert projections. Unlike the
/// head counts, `serve_phases/topology.rs` does not divide
/// `moe_intermediate_size`, so `full_inter == config.moe_intermediate_size`
/// and the local size is computed here.
#[derive(Debug, Clone, Copy)]
pub struct TpMoeDims {
    pub tp_rank: usize,
    pub tp_size: usize,
    pub h: usize,
    /// 2026-09-25: Full MoE intermediate dim (not TP-divided).
    pub full_inter: usize,
    /// 2026-09-25: Local (post-shard) MoE intermediate dim, `full_inter / tp_size`.
    pub local_inter: usize,
}

impl TpMoeDims {
    pub fn from_config(config: &ModelConfig) -> Self {
        let tp_size = config.tp_world_size.max(1);
        let full_inter = config.moe_intermediate_size;
        Self {
            tp_rank: config.tp_rank,
            tp_size,
            h: config.hidden_size,
            full_inter,
            local_inter: full_inter / tp_size,
        }
    }

    /// 2026-09-25: `(out_dim, in_dim, kind)` for one of `gate_proj` / `up_proj` /
    /// `down_proj`. Gate/up are column-parallel on inter; down is
    /// row-parallel on inter (so `[h, inter]` rows truncate to `[h, inter/tp]`).
    pub fn proj_shape(&self, name: &str) -> Option<(usize, usize, TpShardKind)> {
        match name {
            "gate_proj" | "up_proj" => Some((self.full_inter, self.h, TpShardKind::ColumnParallel)),
            "down_proj" => Some((self.h, self.full_inter, TpShardKind::RowParallel)),
            _ => None,
        }
    }
}

mod gdn;
pub use gdn::*;

mod quant_shard;
pub use quant_shard::{shard_fp8_block_scaled, shard_quantized_nvfp4};

#[cfg(test)]
mod tests;
