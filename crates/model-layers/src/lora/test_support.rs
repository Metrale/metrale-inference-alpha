// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Fixtures for the LoRA unit tests: the factory model config, a
//! `SlotView` builder, and a `LoraPair` that needs no GPU.
//!
//! Owner: model-layers (lora).
//! Invariants: none beyond the types.

use metrale_config::ModelConfig;
use metrale_gpu_runtime::gpu::DevicePtr;

use crate::layers::ops::lora_delta::LoraPair;
use crate::lora::SlotView;
use crate::weight_map::DenseWeight;

// 2026-09-25: `ModelConfig::qwen3_next_80b_nvfp4`: 48 layers, of which 3, 7,
// ..., 47 are full-attention and the rest linear-attention, with routed
// experts.
pub(crate) fn cfg() -> ModelConfig {
    ModelConfig::qwen3_next_80b_nvfp4()
}

pub(crate) fn view(filled: bool, ref_count: usize, last_used: u64) -> SlotView {
    SlotView {
        filled,
        ref_count,
        last_used,
    }
}

// 2026-09-25: `tag` is the A pointer (B is `tag + 1`), so a test can tell
// which pair it selected.
pub(crate) fn dummy_pair(tag: u64, k_in: u32, n_out: u32) -> LoraPair {
    LoraPair {
        a: DenseWeight {
            weight: DevicePtr(tag),
        },
        b: DenseWeight {
            weight: DevicePtr(tag + 1),
        },
        rank: 8,
        k_in,
        n_out,
        scale: 0.5,
        max_rank: 16,
    }
}
