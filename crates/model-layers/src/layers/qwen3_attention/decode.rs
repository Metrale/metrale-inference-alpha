// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The single-token decode path of [`super::Qwen3AttentionLayer`]: the submodules
//! under `decode/`, plus `effective_fp8_scales`.
//!
//! Owner: model-layers attention decode.
//! Invariants: none beyond the types.

use super::Qwen3AttentionLayer;

mod attention_forward;
mod attention_forward_kv;
// 2026-09-25: Visible to `trait_impl::multi_seq::mla`, which builds `DecodeMlaArgs`.
pub(in crate::layers::qwen3_attention) mod attention_forward_mla;
mod attention_forward_oproj;
mod attention_forward_v4;
mod high_speed_swap;
mod run_paged_decode;
// 2026-09-25: The paged-decode split-K rule, called from the NVFP4, FP8 and BF16 arms of
// `run_paged_decode.rs`.
mod splitk_dispatch;
#[cfg(test)]
#[path = "decode/splitk_route_tests.rs"]
mod splitk_route_tests;
// 2026-09-25: The `#[ignore]`d GPU equality test for the GQA-packed non-split kernels. It fails,
// rather than skipping, when its lever is not armed.
#[cfg(test)]
#[path = "decode/gqa_pack_fixture.rs"]
mod gqa_pack_fixture;
#[cfg(test)]
#[path = "decode/gqa_pack_gpu_tests.rs"]
mod gqa_pack_gpu_tests;
mod write_kv_cache;
mod write_kv_cache_fp8;
// 2026-09-25: CPU model of the fused FP8 KV decode write against the unfused chain.
#[cfg(test)]
#[path = "decode/write_kv_cache_fp8_tests.rs"]
mod write_kv_cache_fp8_tests;

impl Qwen3AttentionLayer {
    pub(super) fn effective_fp8_scales(&self) -> (f32, f32) {
        if let Some(ref cal) = self.fp8_calibration {
            cal.scales()
        } else {
            (self.attn.k_scale, self.attn.v_scale)
        }
    }
}
