// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `HighSpeedSwap`, the high-speed-swap orchestrator. It writes KV
//! blocks to per-layer files (`offload_block*`) and streams a sequence's blocks
//! back through a scratch pool into tiled attention (`attend_layer*`), using the
//! predictor, the file backend and the eviction policy. It also keeps the disk
//! block ids and the per-thread slot (`install_local`, `with_local`) through
//! which model code reaches it.
//!
//! Owner: metrale-storage high-speed swap.
//! Invariants:
//! - `alloc_disk_block_id` reuses ids from `free_list` before it issues a new
//!   id, and never issues one at or above `max_blocks_per_layer`.

use anyhow::{Context, Result};

// 2026-09-25: io_uring on Linux, the portable backend elsewhere; the alias
// keeps one orchestrator body for both.
#[cfg(target_os = "linux")]
use crate::backend::IoUringBackend as TierBackend;
#[cfg(not(target_os = "linux"))]
use crate::backend::PosixBackend as TierBackend;
use crate::config::HighSpeedSwapConfig;
use crate::cuda_min::{CudaCtx, DeviceBuffer};
use crate::eviction::EvictionPolicy;
use crate::group::GroupLayout;
use crate::layout::Layout;
use crate::predictor::{Predictor, PredictorDims};
use crate::scratch_pool::{ScratchDims, ScratchPool};
use crate::tiled_attention::{TiledAttention, TiledAttentionDims};

// 2026-09-25: `ModelDims` is in `crate::model_dims`, which builds without the
// `cuda` feature; this module does not.
pub use crate::model_dims::ModelDims;

pub struct HighSpeedSwap {
    cfg: HighSpeedSwapConfig,
    model: ModelDims,
    predictor: Predictor,
    pool: ScratchPool,
    backend: TierBackend,
    attn: TiledAttention,
    eviction: EvictionPolicy,
    q_proj: DeviceBuffer,
    // 2026-09-25: f32 scores `[max_blocks_per_layer]`, the tile's i32 block
    // table `[resident_blocks]`, and the i32 block count of the one sequence.
    block_scores_dev: DeviceBuffer,
    block_table_dev: DeviceBuffer,
    counts_dev: DeviceBuffer,
    score_host_buf: Vec<f32>,
    // 2026-09-25: One disk-block-id allocator for all layers: an id names the
    // same block position in every layer's file.
    disk_state: DiskState,
}

#[derive(Debug)]
struct DiskState {
    next_id: u32,
    free_list: Vec<u32>,
    refcount: Vec<u32>,
}

impl DiskState {
    fn new() -> Self {
        Self {
            next_id: 0,
            free_list: Vec::new(),
            refcount: Vec::new(),
        }
    }
}

impl HighSpeedSwap {
    pub fn new(ctx: &CudaCtx, cfg: HighSpeedSwapConfig, model: ModelDims) -> Result<Self> {
        Self::new_on_stream(ctx.stream, cfg, model)
    }

    /// 2026-09-25: Build on `stream`, which must belong to the current thread's
    /// CUDA context. The stream carries only the construction-time upload of the
    /// predictor's projection; per-step methods take their own stream.
    pub fn new_on_stream(stream: u64, cfg: HighSpeedSwapConfig, model: ModelDims) -> Result<Self> {
        cfg.validate_and_prepare()?;
        let group_layout = GroupLayout::new(
            model.num_layers,
            model.max_blocks_per_layer,
            model.num_kv_heads,
            model.block_size as u32,
            model.head_dim as u32,
            2,
            4096,
        );
        let layout = Layout::create(&cfg.dir, group_layout).context("create layout")?;
        // 2026-09-25: The portable backend has one bounce buffer and no queue
        // depth.
        #[cfg(target_os = "linux")]
        let backend = TierBackend::new(layout, cfg.qd as usize)?;
        #[cfg(not(target_os = "linux"))]
        let backend = TierBackend::new(layout)?;
        let pool = ScratchPool::new(ScratchDims {
            num_slots: cfg.resident_blocks,
            num_kv_heads: model.num_kv_heads,
            group_stride: group_layout.group_stride,
        })?;
        let predictor = Predictor::new_on_stream(
            stream,
            PredictorDims {
                num_layers: model.num_layers as usize,
                num_q_heads: model.num_q_heads as usize,
                num_kv_heads: model.num_kv_heads as usize,
                head_dim: model.head_dim as usize,
                r: cfg.rank as usize,
                block_size: model.block_size as usize,
                max_blocks: model.max_blocks_per_layer as usize,
            },
            cfg.projection_seed,
        )?;
        let attn = TiledAttention::new(TiledAttentionDims {
            max_seqs: 1,
            num_q_heads: model.num_q_heads as usize,
            num_kv_heads: model.num_kv_heads as usize,
            head_dim: model.head_dim as usize,
            block_size: model.block_size as usize,
            tile_capacity: cfg.resident_blocks as usize,
        })?;
        let eviction = EvictionPolicy::new(cfg.resident_blocks);
        let q_proj = DeviceBuffer::new(model.num_q_heads as usize * cfg.rank as usize * 2)?;
        let block_scores_dev = DeviceBuffer::new(model.max_blocks_per_layer as usize * 4)?;
        let block_table_dev = DeviceBuffer::new(cfg.resident_blocks as usize * 4)?;
        let counts_dev = DeviceBuffer::new(4)?;
        let score_host_buf = vec![0.0_f32; model.max_blocks_per_layer as usize];
        let disk_state = DiskState::new();
        Ok(Self {
            cfg,
            model,
            predictor,
            pool,
            backend,
            attn,
            eviction,
            q_proj,
            block_scores_dev,
            block_table_dev,
            counts_dev,
            score_host_buf,
            disk_state,
        })
    }

    // 2026-09-25: Disk block ids, shared by all layers, at most
    // `max_blocks_per_layer` of them. `alloc_disk_block_id` returns `None` when
    // none is free; `inc_disk_ref` panics on a freed id; `dec_disk_ref` returns
    // the new count and frees the id at 0.

    pub fn alloc_disk_block_id(&mut self) -> Option<u32> {
        let st = &mut self.disk_state;
        if let Some(id) = st.free_list.pop() {
            st.refcount[id as usize] = 1;
            return Some(id);
        }
        if st.next_id >= self.model.max_blocks_per_layer {
            return None;
        }
        let id = st.next_id;
        st.next_id += 1;
        st.refcount.push(1);
        Some(id)
    }

    pub fn inc_disk_ref(&mut self, id: u32) {
        let rc = &mut self.disk_state.refcount[id as usize];
        if *rc == 0 {
            panic!("inc_disk_ref on freed disk_block_id {id}; caller must hold a live ref");
        }
        *rc += 1;
    }

    pub fn dec_disk_ref(&mut self, id: u32) -> u32 {
        let st = &mut self.disk_state;
        let rc = &mut st.refcount[id as usize];
        debug_assert!(*rc > 0, "dec_disk_ref on already-freed id {id}");
        *rc = rc.saturating_sub(1);
        let new_rc = *rc;
        if new_rc == 0 {
            st.free_list.push(id);
        }
        new_rc
    }

    pub fn disk_refcount(&self, id: u32) -> u32 {
        self.disk_state.refcount[id as usize]
    }

    pub fn disk_free_count(&self) -> usize {
        let st = &self.disk_state;
        st.free_list.len() + (self.model.max_blocks_per_layer - st.next_id) as usize
    }

    /// 2026-09-25: Disk-id and scratch-pool occupancy.
    pub fn diagnostic_summary(&self) -> HighSpeedSwapDiagnostic {
        let st = &self.disk_state;
        let active = st.next_id.saturating_sub(st.free_list.len() as u32);
        HighSpeedSwapDiagnostic {
            num_layers: self.model.num_layers,
            active_disk_blocks: active,
            disk_block_capacity: self.model.max_blocks_per_layer,
            scratch_pool_resident: self.pool.dims().num_slots,
            scratch_pool_free: self.pool.free_count(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct HighSpeedSwapDiagnostic {
    pub num_layers: u32,
    pub active_disk_blocks: u32,
    pub disk_block_capacity: u32,
    pub scratch_pool_resident: u32,
    pub scratch_pool_free: u32,
}

#[cfg(test)]
mod disk_id_tests;

mod impl_more;

use std::cell::RefCell;
// 2026-09-25: Thread-local because the orchestrator's device memory and
// streams belong to one thread's CUDA context. The server installs it
// (`install_local`), and model-layer and model-engine code reach it with
// `with_local` instead of passing it through every layer signature. It is
// dropped when the thread exits or another install replaces it.
thread_local! {
    static LOCAL: RefCell<Option<HighSpeedSwap>> = const { RefCell::new(None) };
}

/// 2026-09-25: Build an orchestrator and install it on the current thread,
/// dropping any earlier one. On a build error the earlier one stays.
pub fn install_local(stream: u64, cfg: HighSpeedSwapConfig, model: ModelDims) -> Result<()> {
    let hss = HighSpeedSwap::new_on_stream(stream, cfg, model)?;
    LOCAL.with(|cell| {
        *cell.borrow_mut() = Some(hss);
    });
    Ok(())
}

/// 2026-09-25: Whether this thread has an installed orchestrator.
pub fn local_installed() -> bool {
    LOCAL.with(|cell| cell.borrow().is_some())
}

/// 2026-09-25: Run `f` on this thread's orchestrator; `None` when none is
/// installed.
pub fn with_local<R>(f: impl FnOnce(&mut HighSpeedSwap) -> Result<R>) -> Option<Result<R>> {
    LOCAL.with(|cell| cell.borrow_mut().as_mut().map(f))
}
