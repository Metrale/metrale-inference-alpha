// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: GPU side of the high-speed-swap block predictor. Loads the
//! `q_lowrank_project`, `kv_lowrank_project` and `predictor_score` PTX from build.rs
//! and owns the projection `P` and the per-token low-rank keys `A_g`.
//!
//! Owner: storage, high-speed swap.
//! Invariants:
//! - `P` is written once, in the constructor, from `projection_seed`.
//!
//! `project_q` computes Q @ P; `project_kv_block` computes K @ P for every token of
//! one block; `score_blocks` gives each block the max, over query heads and tokens,
//! of the projected dot product. `HighSpeedSwap` uses the scores only to rank
//! eviction victims (`EvictionPolicy::record_score`); attention reads every block.

use anyhow::{Context, Result, bail};
use half::bf16;
use std::ffi::c_void;

use crate::cuda_min::{
    CudaCtx, CudaModule, DeviceBuffer, copy_d_to_h_async, copy_h_to_d_async, launch_kernel,
    stream_sync,
};
use crate::projection::{PredictorShape, build_projection};

include!(concat!(env!("OUT_DIR"), "/storage_ptx.rs"));

#[derive(Clone, Copy, Debug)]
pub struct PredictorDims {
    pub num_layers: usize,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub r: usize,
    pub block_size: usize,
    pub max_blocks: usize,
}

impl PredictorDims {
    pub fn validate(&self) -> Result<()> {
        if !self.num_q_heads.is_multiple_of(self.num_kv_heads) {
            bail!(
                "num_q_heads ({}) must divide num_kv_heads ({})",
                self.num_q_heads,
                self.num_kv_heads
            );
        }
        Ok(())
    }
    pub fn gqa_ratio(&self) -> i32 {
        (self.num_q_heads / self.num_kv_heads) as i32
    }
    pub fn a_g_bytes(&self) -> usize {
        // 2026-09-25: Per-token layout:
        // [num_layers, max_blocks, num_kv_heads, block_size, r], BF16.
        self.num_layers * self.max_blocks * self.num_kv_heads * self.block_size * self.r * 2
    }
    pub fn per_layer_block_floats(&self) -> usize {
        self.num_kv_heads * self.block_size * self.r
    }
    pub fn p_bytes(&self) -> usize {
        self.head_dim * self.r * 2
    }
}

pub struct Predictor {
    dims: PredictorDims,
    _modules: Vec<CudaModule>,
    f_q_proj: u64,
    f_kv_proj: u64,
    f_score: u64,
    // 2026-09-25: `[head_dim, r]` BF16.
    p_dev: DeviceBuffer,
    // 2026-09-25: `[num_layers, max_blocks, num_kv_heads, block_size, r]` BF16.
    a_g_dev: DeviceBuffer,
}

impl Predictor {
    pub fn new(ctx: &CudaCtx, dims: PredictorDims, projection_seed: u64) -> Result<Self> {
        Self::new_on_stream(ctx.stream, dims, projection_seed)
    }

    pub fn new_on_stream(stream: u64, dims: PredictorDims, projection_seed: u64) -> Result<Self> {
        dims.validate()?;
        // 2026-09-25: Only the predictor modules; `TiledAttention::new` loads the
        // attention ones.
        let mut modules: Vec<CudaModule> = Vec::new();
        let mut f_q_proj = 0u64;
        let mut f_kv_proj = 0u64;
        let mut f_score = 0u64;
        for entry in STORAGE_PTX.iter() {
            match entry.name {
                "q_lowrank_project" | "kv_lowrank_project" | "predictor_score" => {
                    let m = CudaModule::from_ptx(entry.ptx)
                        .with_context(|| format!("load PTX module {}", entry.name))?;
                    match entry.name {
                        "q_lowrank_project" => f_q_proj = m.function("q_lowrank_project")?,
                        "kv_lowrank_project" => f_kv_proj = m.function("kv_lowrank_project")?,
                        "predictor_score" => f_score = m.function("predictor_score")?,
                        _ => unreachable!(),
                    }
                    modules.push(m);
                }
                _ => {}
            }
        }
        if f_q_proj == 0 || f_kv_proj == 0 || f_score == 0 {
            bail!("missing predictor kernel function — PTX list incomplete");
        }
        let shape = PredictorShape::new(dims.head_dim, dims.r);
        let p_host = build_projection(shape, projection_seed);
        let p_dev = DeviceBuffer::new(dims.p_bytes())?;
        copy_h_to_d_async(
            p_dev.ptr,
            p_host.as_ptr() as *const c_void,
            dims.p_bytes(),
            stream,
        )?;
        // 2026-09-25: Refuse before allocating when A_g does not fit, with an error
        // that names the settings that shrink it; a failed `cuMemAlloc_v2` names
        // none.
        let a_g_need = dims.a_g_bytes();
        let (free_hbm, _total_hbm) = crate::cuda_min::mem_info()?;
        // 2026-09-25: 5% of free memory is left for the scratch pool, tiled
        // attention and the smaller buffers allocated after this.
        let a_g_budget = free_hbm.saturating_mul(95) / 100;
        if a_g_need > a_g_budget {
            bail!(
                "HSS predictor A_g would need {:.2} GB but only {:.2} GB of HBM is free \
                 (5% margin reserved for scratch + tiled-attention).\n\
                 Tune one of:\n  \
                 - --high-speed-swap-rank: current {} ; try {} (halves A_g)\n  \
                 - --max-seq-len: max_blocks={} → currently dominates A_g; halve --max-seq-len to halve A_g\n  \
                 - --kv-cache-dtype nvfp4: halves the KV pool, freeing room for A_g\n  \
                 - --gpu-memory-utilization: lower so weight-side allocations leave more HBM\n\
                 A_g sizing = num_layers ({}) × max_blocks ({}) × num_kv_heads ({}) × block_size ({}) × r ({}) × 2 bytes.",
                a_g_need as f64 / (1u64 << 30) as f64,
                a_g_budget as f64 / (1u64 << 30) as f64,
                dims.r,
                dims.r / 2,
                dims.max_blocks,
                dims.num_layers,
                dims.max_blocks,
                dims.num_kv_heads,
                dims.block_size,
                dims.r,
            );
        }
        // 2026-09-25: A_g is not zeroed. `score_blocks` also reads slots that
        // `project_kv_block` never wrote, such as blocks offloaded through
        // `offload_block_no_predict_on_stream`.
        let a_g_dev = DeviceBuffer::new(a_g_need)?;
        stream_sync(stream)?;
        Ok(Self {
            dims,
            _modules: modules,
            f_q_proj,
            f_kv_proj,
            f_score,
            p_dev,
            a_g_dev,
        })
    }

    /// 2026-09-25: `q_proj = q @ P`. `q` is a device pointer to
    /// `[num_q_heads, head_dim]` BF16, `q_proj` to the `[num_q_heads, r]` BF16 output.
    pub fn project_q(&self, ctx: &CudaCtx, q: u64, q_proj: u64) -> Result<()> {
        self.project_q_on_stream(ctx.stream, q, q_proj)
    }

    /// 2026-09-25: [`Self::project_q`] on an explicit stream.
    pub fn project_q_on_stream(&self, stream: u64, q: u64, q_proj: u64) -> Result<()> {
        let mut q_v = q;
        let mut p_v = self.p_dev.ptr;
        let mut o_v = q_proj;
        let mut nq = self.dims.num_q_heads as i32;
        let mut hd = self.dims.head_dim as i32;
        let mut r = self.dims.r as i32;
        let mut params = [
            &mut q_v as *mut _ as *mut c_void,
            &mut p_v as *mut _ as *mut c_void,
            &mut o_v as *mut _ as *mut c_void,
            &mut nq as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut r as *mut _ as *mut c_void,
        ];
        launch_kernel(
            self.f_q_proj,
            (self.dims.num_q_heads as u32, 1, 1),
            (self.dims.r as u32, 1, 1),
            0,
            stream,
            &mut params,
        )
    }

    /// 2026-09-25: Write the low-rank keys of block `block_id` at `layer` into A_g.
    /// `k_block` is a device pointer to `[block_size, num_kv_heads, head_dim]` BF16.
    /// An out-of-range layer or block is an error.
    pub fn project_kv_block(
        &self,
        ctx: &CudaCtx,
        layer: usize,
        block_id: usize,
        k_block: u64,
    ) -> Result<()> {
        self.project_kv_block_on_stream(ctx.stream, layer, block_id, k_block)
    }

    pub fn project_kv_block_on_stream(
        &self,
        stream: u64,
        layer: usize,
        block_id: usize,
        k_block: u64,
    ) -> Result<()> {
        if layer >= self.dims.num_layers || block_id >= self.dims.max_blocks {
            bail!("project_kv_block out of range: layer {layer}, block {block_id}");
        }
        let slot_floats = self.dims.per_layer_block_floats();
        let k_lr_slot = self.a_g_dev.ptr
            + (((layer * self.dims.max_blocks + block_id) * slot_floats) * 2) as u64;
        let mut k_v = k_block;
        let mut p_v = self.p_dev.ptr;
        let mut o_v = k_lr_slot;
        let mut bs = self.dims.block_size as i32;
        let mut nk = self.dims.num_kv_heads as i32;
        let mut hd = self.dims.head_dim as i32;
        let mut r = self.dims.r as i32;
        let mut params = [
            &mut k_v as *mut _ as *mut c_void,
            &mut p_v as *mut _ as *mut c_void,
            &mut o_v as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
            &mut nk as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
            &mut r as *mut _ as *mut c_void,
        ];
        launch_kernel(
            self.f_kv_proj,
            (
                self.dims.num_kv_heads as u32,
                self.dims.block_size as u32,
                1,
            ),
            (self.dims.r as u32, 1, 1),
            0,
            stream,
            &mut params,
        )
    }

    /// 2026-09-25: Score the first `num_active_blocks` blocks at `k_lr_seq`, a device
    /// pointer to `[num_active_blocks, num_kv_heads, block_size, r]` BF16. `q_proj` is
    /// `[num_q_heads, r]` BF16; `scores_out` receives `[num_active_blocks]` f32.
    pub fn score_blocks(
        &self,
        ctx: &CudaCtx,
        q_proj: u64,
        k_lr_seq: u64,
        scores_out: u64,
        num_active_blocks: usize,
    ) -> Result<()> {
        self.score_blocks_on_stream(ctx.stream, q_proj, k_lr_seq, scores_out, num_active_blocks)
    }

    pub fn score_blocks_on_stream(
        &self,
        stream: u64,
        q_proj: u64,
        k_lr_seq: u64,
        scores_out: u64,
        num_active_blocks: usize,
    ) -> Result<()> {
        let mut q_v = q_proj;
        let mut a_v = k_lr_seq;
        let mut s_v = scores_out;
        let mut nq = self.dims.num_q_heads as i32;
        let mut nk = self.dims.num_kv_heads as i32;
        let mut bs = self.dims.block_size as i32;
        let mut r = self.dims.r as i32;
        let mut gqa = self.dims.gqa_ratio();
        let mut params = [
            &mut q_v as *mut _ as *mut c_void,
            &mut a_v as *mut _ as *mut c_void,
            &mut s_v as *mut _ as *mut c_void,
            &mut nq as *mut _ as *mut c_void,
            &mut nk as *mut _ as *mut c_void,
            &mut bs as *mut _ as *mut c_void,
            &mut r as *mut _ as *mut c_void,
            &mut gqa as *mut _ as *mut c_void,
        ];
        launch_kernel(
            self.f_score,
            (num_active_blocks as u32, 1, 1),
            (128, 1, 1),
            0,
            stream,
            &mut params,
        )
    }

    pub fn dims(&self) -> PredictorDims {
        self.dims
    }
    pub fn a_g_dev_ptr(&self) -> u64 {
        self.a_g_dev.ptr
    }
}

/// 2026-09-25: Copy the A_g slot of `(layer, block)` to the host, as BF16
/// `[num_kv_heads, block_size, r]`, and synchronise the stream.
pub fn read_k_lr_slot(
    ctx: &CudaCtx,
    pred: &Predictor,
    layer: usize,
    block: usize,
) -> Result<Vec<bf16>> {
    let dims = pred.dims();
    let n = dims.per_layer_block_floats();
    let slot = pred.a_g_dev.ptr + (((layer * dims.max_blocks + block) * n) * 2) as u64;
    let mut host = vec![bf16::from_f32(0.0); n];
    copy_d_to_h_async(host.as_mut_ptr() as *mut c_void, slot, n * 2, ctx.stream)?;
    stream_sync(ctx.stream)?;
    Ok(host)
}
