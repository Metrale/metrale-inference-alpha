// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The full-attention transformer layer, `Qwen3AttentionLayer`:
//! its construction, kernel requirements, decode and prefill attention, and
//! the call into the layer's FFN.
//!
//! Owner: model-layers (attention).
//! Invariants: none beyond the types.

// 2026-09-25: The N-column-blocked W8A16 decode tier for the attention
// projections. `init` reads its lever once and caches it on the layer
// (`attn_ncol`); both multi-seq routes, strided QKV and contiguous o_proj, read
// that field.
mod attn_ncol_gemv;
// 2026-09-25: The log-once route lines of the `METRALE_ATTN_M16_TC` tier, one
// for the strided QKV call site and one for the o_proj call site.
pub(crate) mod attn_m16_tc_route;
mod decode;
// 2026-09-25: `pub` because the DeepSeek-V4 weight loader in
// metrale-model-arch calls `helpers::yarn_rope_mscale`.
pub mod helpers;
mod init;
mod init_arch_gates;
mod init_decode_kernels;
mod init_kernel_dispatch;
mod init_prefill_kernels;
mod init_proj_kernels;
mod kernel_requirements;
mod op_dump;
// 2026-09-25: `innerq_driver` uses `metrale_gpu_runtime::registry`, which
// exists only with the `cuda` feature, so the module carries the same gate.
#[cfg(feature = "cuda")]
pub mod innerq_driver;
mod prefill;
mod prefill_qkv_w8a8;
mod prefill_w8a8;
mod prefill_weights;
mod trait_impl;
pub mod types;
mod types_weights;

#[cfg(feature = "cuda")]
pub use innerq_driver::InnerQDriver;
pub(crate) use types::HeadGateActivation;
pub use types::Qwen3AttentionLayer;
pub use types_weights::{
    CompressorWeights, Fp8TwinSet, HcHeadWeights, HcLowRank, HcSiteWeights, HcWeights, MlaWeights,
    W8A8_PREFILL_KERNELS, w8a8_prefill_kernels_loaded,
};

/// 2026-09-25: Startup check for one `--kv-cache-dtype`: resolves every kernel
/// the dtype's dispatch needs (the reshape/decode pair, the chunked-prefill
/// kernel and, for a WHT-rotated side, the WHT bookends) and errors with the
/// full missing list. The server calls it before `build_model`. See
/// `kernel_requirements.rs`.
pub fn validate_required_kv_kernels(
    gpu: &dyn metrale_gpu_runtime::gpu::GpuBackend,
    kv_dtype: metrale_cache::kv_cache::KvCacheDtype,
    head_dim: usize,
) -> anyhow::Result<()> {
    kernel_requirements::validate_required_kernels(gpu, kv_dtype, head_dim)
}

/// 2026-09-25: The reference sequence count that the `Legacy` split-K rule
/// (`metrale_kernels::attn_splitk::legacy_splits`) divides the SM count by:
/// the configured max decode batch (`ModelLevers::max_decode_seqs`), raised to
/// `num_seqs` when the live batch is wider.
///
/// Using the max batch rather than the live count keeps a sequence's split
/// count, and so its split-merge reduction tree, the same whether it is decoded
/// alone or co-batched; the merge is not associative, so a different tree can
/// change the temp-0 argmax. The raise to `num_seqs` keeps
/// `num_seqs * num_q_heads * num_splits <= sm_count` whenever the rule picks
/// more than one split, which is the bound the `Legacy` workspace is sized to
/// (`attn_splitk::workspace_slots`).
pub(crate) fn split_ref_seqs(num_seqs: u32, max_decode_seqs: u32) -> u32 {
    max_decode_seqs.max(num_seqs)
}

/// 2026-09-25: Host time in µs spent in the FFN of prefill attention layers,
/// added when `METRALE_PREFILL_HOST_TIMING=1`. Summed across layers; the layer
/// loop takes it once per prefill and reports the attention remainder as the
/// in-layer time minus this.
pub static FFN_HOST_US: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub fn add_ffn_host_us(us: u64) {
    FFN_HOST_US.fetch_add(us, std::sync::atomic::Ordering::Relaxed);
}

pub fn take_ffn_host_us() -> u64 {
    FFN_HOST_US.swap(0, std::sync::atomic::Ordering::Relaxed)
}

/// 2026-09-25: Per-phase host time in µs for the prefill attention path, written
/// only by `prefill/cache_skip.rs`: 0 = the Q/K/V projections, 1 = from the
/// projections to the BR=64 attention call, 2 = that attention call. No code
/// writes slot 3. Summed across layers; the layer loop takes them once per
/// prefill when `METRALE_PREFILL_HOST_TIMING=1`.
pub static ATTN_PHASE_US: [std::sync::atomic::AtomicU64; 4] = [
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
];

pub fn add_attn_phase_us(i: usize, us: u64) {
    ATTN_PHASE_US[i].fetch_add(us, std::sync::atomic::Ordering::Relaxed);
}

pub fn take_attn_phase_us() -> [u64; 4] {
    let mut o = [0u64; 4];
    for (i, a) in ATTN_PHASE_US.iter().enumerate() {
        o[i] = a.swap(0, std::sync::atomic::Ordering::Relaxed);
    }
    o
}

#[cfg(test)]
mod split_ref_seqs_tests {
    use super::split_ref_seqs;

    #[test]
    fn the_split_count_does_not_move_with_co_batch_size() {
        // 2026-09-25: A sequence decoded alone and the same sequence co-batched
        // get the same reference count, so the `Legacy` rule picks the same
        // split count for both.
        let pin = 16;
        assert_eq!(split_ref_seqs(1, pin), split_ref_seqs(8, pin));
        assert_eq!(split_ref_seqs(1, pin), pin);
    }

    #[test]
    fn a_batch_larger_than_the_pin_clamps_up() {
        assert_eq!(split_ref_seqs(32, 16), 32);
    }

    #[test]
    fn two_models_can_pin_to_different_batches() {
        assert_ne!(split_ref_seqs(1, 4), split_ref_seqs(1, 16));
    }
}
