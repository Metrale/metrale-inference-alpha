// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: One arena-owned slab that every grouped FP8 MoE layer reuses in turn.
//! Quantizers overwrite their live activation/scale ranges. The worklist builder
//! overwrites the live entries and `total_tiles` on EVERY invocation, including
//! empty routes; consumers never read the stale tail. No host synchronization
//! or allocation is needed inside CUDA capture, and addresses survive replay.
//!
//! Owner: gpu-runtime.
//! Invariants:
//! - The regions are disjoint and 256-byte aligned.
//! - `Layout::new(c, rows).bytes <= Layout::new(c, max_batch_tokens).bytes` for
//!   every `rows <= max_batch_tokens`.

use super::BufferArena;
use crate::gpu::DevicePtr;
use metrale_config::ModelConfig;

pub(super) struct Layout {
    pub(super) scale: usize,
    pub(super) worklist: usize,
    pub(super) counter: usize,
    pub(super) small_worklist: usize,
    pub(super) small_counter: usize,
    pub(super) small_bytes: usize,
    pub(super) bytes: usize,
}

impl Layout {
    pub(super) fn new(c: &ModelConfig, rows: usize) -> Self {
        if c.num_experts == 0 {
            return Self {
                scale: 0,
                worklist: 0,
                counter: 0,
                small_worklist: 0,
                small_counter: 0,
                small_bytes: 0,
                bytes: 0,
            };
        }
        let expanded = rows * c.num_experts_per_tok;
        let shared_k = c.hidden_size.max(c.shared_expert_intermediate_size);
        let activation = (rows * shared_k).max(expanded * c.moe_intermediate_size);
        let scales = (rows * shared_k.div_ceil(128))
            .max(expanded * c.moe_intermediate_size.div_ceil(128))
            * 4;
        // 2026-09-25: Same PM4 geometry as the grouped kernel: M=128, N=64. Per-expert
        // rounding is bounded by +num_experts; +1 retains the launcher's slack.
        let items = (expanded.div_ceil(128) + c.num_experts + 1)
            * c.hidden_size.max(c.moe_intermediate_size).div_ceil(64);
        let align = |n: usize| n.div_ceil(256) * 256;
        let scale = align(activation);
        let worklist = scale + align(scales);
        let counter = worklist + align(items * 8);
        // 2026-09-25: Only E256 is qualified for the adaptive Hopper dispatch. One small
        // tile per expert, N=64; the capacity comes from the dimensions, never a fixed KiB.
        let small_bytes = if c.num_experts == 256 {
            small_list_bytes(c.num_experts, c.hidden_size.max(c.moe_intermediate_size))
                .expect("adaptive FP8 small worklist size overflow")
        } else {
            0
        };
        let small_worklist = if small_bytes > 0 {
            align(counter + 4)
        } else {
            0
        };
        let small_counter = small_worklist + align(small_bytes);
        Self {
            scale,
            worklist,
            counter,
            small_worklist,
            small_counter,
            small_bytes,
            bytes: if small_bytes > 0 {
                small_counter + 4
            } else {
                counter + 4
            },
        }
    }
}

fn small_list_bytes(experts: usize, width: usize) -> Option<usize> {
    experts.checked_mul(width.div_ceil(64))?.checked_mul(8)
}

/// 2026-09-25: Disjoint regions; shared/gate/up/down phases reuse these on one stream.
pub struct MoeFp8Scratch {
    pub activation: DevicePtr,
    pub scales: DevicePtr,
    pub worklist: DevicePtr,
    pub total_tiles: DevicePtr,
    pub worklist_bytes: usize,
    pub small_worklist: DevicePtr,
    pub small_total_tiles: DevicePtr,
    pub small_worklist_bytes: usize,
}

impl BufferArena {
    /// 2026-09-25: Fixed offsets for a given shape, without allocating or mutating ownership.
    ///
    /// # Errors
    /// When `rows` exceeds the arena's batch, the model has no experts, the shape
    /// needs more than the slab holds, or the slab was released.
    pub fn moe_fp8_scratch(&self, c: &ModelConfig, rows: usize) -> anyhow::Result<MoeFp8Scratch> {
        let l = Layout::new(c, rows);
        anyhow::ensure!(
            rows <= self.max_batch_tokens
                && l.bytes > 0
                && l.bytes <= self.sizes.moe_fp8_scratch
                && self.moe_fp8_scratch != DevicePtr::NULL,
            "FP8 MoE scratch exceeds arena: rows={rows}, requested={} bytes, capacity={} bytes",
            l.bytes,
            self.sizes.moe_fp8_scratch
        );
        Ok(MoeFp8Scratch {
            activation: self.moe_fp8_scratch,
            scales: self.moe_fp8_scratch.offset(l.scale),
            worklist: self.moe_fp8_scratch.offset(l.worklist),
            total_tiles: self.moe_fp8_scratch.offset(l.counter),
            worklist_bytes: l.counter - l.worklist,
            small_worklist: if l.small_bytes > 0 {
                self.moe_fp8_scratch.offset(l.small_worklist)
            } else {
                DevicePtr::NULL
            },
            small_total_tiles: if l.small_bytes > 0 {
                self.moe_fp8_scratch.offset(l.small_counter)
            } else {
                DevicePtr::NULL
            },
            small_worklist_bytes: l.small_bytes,
        })
    }
}

#[cfg(test)]
#[path = "moe_fp8_scratch_tests.rs"]
mod tests;
