// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The high-speed-swap surface for builds without the `cuda` feature.
//!
//! Owner: storage, high-speed swap.
//! Invariants:
//! - `local_installed()` is `false`, and `with_local` returns `None` without calling
//!   its closure.
//! - `install_local` always returns an error.
//!
//! The names match the real `HighSpeedSwap` API, so the callers in metrale-model-engine
//! and metrale-model-layers compile unchanged. The offload and attend methods panic;
//! `with_local` never hands out a `HighSpeedSwap` to call them on.

use crate::config::HighSpeedSwapConfig;
use crate::model_dims::ModelDims;

pub struct HighSpeedSwap;

#[allow(unused_variables)]
impl HighSpeedSwap {
    pub fn alloc_disk_block_id(&mut self) -> Option<u32> {
        None
    }

    pub fn inc_disk_ref(&mut self, id: u32) {}

    pub fn dec_disk_ref(&mut self, id: u32) -> u32 {
        0
    }

    pub fn offload_block_on_stream(
        &mut self,
        stream: u64,
        layer: u32,
        block: u32,
        k_block_dev: u64,
        k_block_host: &[half::bf16],
        v_block_host: &[half::bf16],
    ) -> anyhow::Result<()> {
        unreachable!("HighSpeedSwap stub: cuda feature is off")
    }

    pub fn offload_block_no_predict_on_stream(
        &mut self,
        stream: u64,
        layer: u32,
        block: u32,
        k_block_host: &[half::bf16],
        v_block_host: &[half::bf16],
    ) -> anyhow::Result<()> {
        unreachable!()
    }

    pub fn attend_layer_on_stream(
        &mut self,
        stream: u64,
        layer: u32,
        seq_block_ids: &[u32],
        q_dev: u64,
        output_dev: u64,
    ) -> anyhow::Result<()> {
        unreachable!()
    }

    pub fn attend_layer_on_stream_with_q_pos(
        &mut self,
        stream: u64,
        layer: u32,
        seq_block_ids: &[u32],
        q_dev: u64,
        output_dev: u64,
        last_block_valid_slots: i32,
    ) -> anyhow::Result<()> {
        unreachable!()
    }
}

pub fn local_installed() -> bool {
    false
}

pub fn with_local<R>(
    _f: impl FnOnce(&mut HighSpeedSwap) -> anyhow::Result<R>,
) -> Option<anyhow::Result<R>> {
    None
}

pub fn install_local(
    _stream: u64,
    _cfg: HighSpeedSwapConfig,
    _model: ModelDims,
) -> anyhow::Result<()> {
    anyhow::bail!("HighSpeedSwap unavailable: metrale-storage built without cuda feature")
}
