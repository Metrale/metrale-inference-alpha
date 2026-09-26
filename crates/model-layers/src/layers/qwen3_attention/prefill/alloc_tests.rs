// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Allocation tests for the two attention prefill projection
//! chains that the cuBLAS `attn` scope reaches: the cache-skip Q/K/V chain
//! (`cache_skip_qkv.rs`) and the paged O projection (`paged_oproj.rs`). With
//! block-scaled FP8 weights and every cuBLAS scope armed, neither may allocate
//! device memory; the arena already holds every buffer they need.
//!
//! The live allocation count is compared with the count taken before the
//! first call, so an allocation cached after the first call is caught too.
//!
//! Both fixtures zero the k-major scale adapter, so with the default K-major
//! scale layout each chain takes the in-tree kernel rather than cuBLASLt,
//! whose FFI a CPU test cannot enter. The cuBLASLt selectors are tested in
//! `prefill_qkv_w8a8_tests.rs`.
//!
//! Owner: model-layers (attention).
//! Invariants: none beyond the types.

use super::super::Qwen3AttentionLayer;
use crate::layer::{ForwardContext, MoeLoraRoute};
use crate::layers::FfnComponent;
use crate::layers::ops::{CublasScope, DerivedWeights, GemmDispatch, ModelLevers, ModelStats};
use crate::weight_map::{
    AttentionWeights, DenseWeight, Fp8Weight, QuantWeight, QuantizedWeight, WeightQuantFormat,
};
use metrale_cache::kv_cache::KvCacheDtype;
use metrale_config::ModelConfig;
use metrale_gpu_runtime::buffers::BufferArena;
use metrale_gpu_runtime::gpu::mock::MockGpuBackend;
use metrale_gpu_runtime::gpu::{GpuBackend, KernelHandle};

const M: u32 = 28;
const QUANT_K: KernelHandle = KernelHandle(0xA101);
const BLOCKSCALED_K: KernelHandle = KernelHandle(0xA102);
const W8A16_PIPELINED_K: KernelHandle = KernelHandle(0xA103);

fn fp8(gpu: &MockGpuBackend, n: u32, k: u32) -> Fp8Weight {
    Fp8Weight {
        weight: gpu.alloc(n as usize * k as usize).unwrap(),
        row_scale: gpu
            .alloc((n as usize).div_ceil(128) * (k as usize).div_ceil(128) * 4)
            .unwrap(),
        n,
        k,
        scale_format: WeightQuantFormat::Fp8BlockScaled,
    }
}

/// 2026-09-25: A full-attention layer with block-scaled FP8 Q/K/V/O weights.
fn native_fp8_attention_layer(gpu: &MockGpuBackend, config: &ModelConfig) -> Qwen3AttentionLayer {
    let dense = DenseWeight {
        weight: gpu
            .alloc(config.hidden_size * config.hidden_size * 2)
            .unwrap(),
    };
    let attn = AttentionWeights {
        q_proj: dense,
        k_proj: dense,
        v_proj: dense,
        o_proj: QuantizedWeight::null(),
        q_norm: dense,
        k_norm: dense,
        q_norm_full: None,
        k_norm_full: None,
        k_scale: 1.0,
        v_scale: 1.0,
    };
    let mut layer = Qwen3AttentionLayer::new(
        dense,
        attn,
        dense,
        FfnComponent::None,
        0,
        None,
        None,
        None,
        gpu,
        KvCacheDtype::Bf16,
        0,
        config,
    )
    .unwrap();
    let h = config.hidden_size as u32;
    let nq = config.num_attention_heads as u32;
    let nkv = config.num_key_value_heads as u32;
    let hd = config.head_dim as u32;
    let q_proj_dim = if layer.gated { 2 * nq * hd } else { nq * hd };
    layer.q_weight = Some(QuantWeight::Fp8(fp8(gpu, q_proj_dim, h)));
    layer.k_weight = Some(QuantWeight::Fp8(fp8(gpu, nkv * hd, h)));
    layer.v_weight = Some(QuantWeight::Fp8(fp8(gpu, nkv * hd, h)));
    layer.o_weight = Some(QuantWeight::Fp8(fp8(gpu, h, nq * hd)));
    // 2026-09-25: No transposed or FP8xFP8 copies, so each Q/K/V projection
    // reaches the `w8a16_gemm_pipelined` arm of `cache_skip_one_proj`.
    layer.q_fp8w_t = None;
    layer.k_fp8w_t = None;
    layer.v_fp8w_t = None;
    layer.o_fp8w_t = None;
    layer.q_fp8 = None;
    layer.k_fp8 = None;
    layer.v_fp8 = None;
    layer.o_fp8 = None;
    layer.per_token_group_quant_fp8_k = crate::layers::ops::Fp8ActQuant::shared_only(QUANT_K);
    layer.fp8_gemm_t_blockscaled_k = BLOCKSCALED_K;
    layer.w8a16_gemm_pipelined_k = W8A16_PIPELINED_K;
    // 2026-09-25: No k-major adapter: the chains take the in-tree kernel, which
    // the mock backend can run (see the module header).
    layer.fp8_act_scale_kmajor_k = KernelHandle(0);
    layer
}

/// 2026-09-25: Every cuBLAS scope armed, `attn` included.
fn armed_dispatch() -> GemmDispatch {
    GemmDispatch {
        cublas: CublasScope::ALL,
        ..GemmDispatch::defaults()
    }
}

macro_rules! fwd_ctx {
    ($buffers:expr, $gpu:expr, $config:expr, $dispatch:expr, $derived:expr, $levers:expr, $stats:expr) => {
        ForwardContext {
            buffers: $buffers,
            hc_row_offset: 0,
            gpu: $gpu,
            config: $config,
            dispatch: $dispatch,
            derived: $derived,
            levers: $levers,
            stats: $stats,
            attn_metadata: None,
            decode_step: false,
            profile: false,
            comm: None,
            graph_capture: false,
            gdn_exact_replay: false,
            gdn_write_on_accept: false,
            token_ids: None,
            host_token_ids: None,
            routed_lora_layers: None,
            midchunk_capture: None,
            moe_lora_route: MoeLoraRoute::Fold,
        }
    };
}

/// 2026-09-25: `paged_oproj.rs`: the O projection allocates nothing with the
/// `attn` scope armed, on the first call or the second.
#[test]
fn attention_o_projection_prefill_allocates_nothing_with_the_cublas_lever_armed() {
    let config = ModelConfig::qwen3_next_80b_nvfp4();
    let gpu = MockGpuBackend::new();
    let layer = native_fp8_attention_layer(&gpu, &config);
    let buffers = BufferArena::new(&config, 64, 4096, 16, 32, &gpu).unwrap();
    let (dispatch, derived) = (armed_dispatch(), DerivedWeights::new());
    let (levers, stats) = (ModelLevers::defaults(), ModelStats::new());
    let ctx = fwd_ctx!(
        &buffers, &gpu, &config, &dispatch, &derived, &levers, &stats
    );

    let h = config.hidden_size as u32;
    let nq = config.num_attention_heads as u32;
    let hd = config.head_dim as u32;
    let run = || layer.prefill_attention_paged_oproj(buffers.attn_output(), M, h, nq, hd, &ctx, 0);
    let (allocs, bytes) = (gpu.live_alloc_count(), gpu.live_bytes());
    let launches = gpu.launch_count();
    // 2026-09-25: The allocation is checked before the result is unwrapped, so
    // an arm that allocates and then fails inside cuBLASLt's FFI on the mock
    // still reports the allocation.
    let first = run();
    assert_eq!(
        (gpu.live_alloc_count(), gpu.live_bytes()),
        (allocs, bytes),
        "the O-projection prefill allocated on its FIRST call — that is the \
         off-ledger BF16 weight dequant from the H100 receipt (here `o_proj` \
         [h, nq*hd] x 2 B = 16777216 bytes, per layer)"
    );
    first.expect("first O-projection prefill");
    // 2026-09-25: Two launches per call, the per-token FP8 activation quant and
    // the block-scaled GEMM, so "allocated nothing" cannot be met by "did
    // nothing", and an extra dequant launch shows even if it reuses a buffer.
    assert_eq!(
        gpu.launch_count() - launches,
        2,
        "expected per_token_group_quant_fp8 + fp8_gemm_t_blockscaled"
    );
    let second = run();
    assert_eq!(
        (gpu.live_alloc_count(), gpu.live_bytes()),
        (allocs, bytes),
        "the O-projection prefill allocated per call"
    );
    second.expect("second O-projection prefill");
    assert_eq!(gpu.launch_count() - launches, 4);
}

/// 2026-09-25: `cache_skip_qkv.rs`: the Q/K/V chain allocates nothing with the
/// `attn` scope armed, on the first call or the second.
#[test]
fn attention_cache_skip_qkv_prefill_allocates_nothing_with_the_cublas_lever_armed() {
    let config = ModelConfig::qwen3_next_80b_nvfp4();
    let gpu = MockGpuBackend::new();
    let layer = native_fp8_attention_layer(&gpu, &config);
    let buffers = BufferArena::new(&config, 64, 4096, 16, 32, &gpu).unwrap();
    let (dispatch, derived) = (armed_dispatch(), DerivedWeights::new());
    let (levers, stats) = (ModelLevers::defaults(), ModelStats::new());
    let ctx = fwd_ctx!(
        &buffers, &gpu, &config, &dispatch, &derived, &levers, &stats
    );

    let h = config.hidden_size as u32;
    let nq = config.num_attention_heads as u32;
    let nkv = config.num_key_value_heads as u32;
    let hd = config.head_dim as u32;
    let q_dim = (nq * hd) as usize;
    let q_proj_dim = if layer.gated { q_dim * 2 } else { q_dim };
    let kv_dim = (nkv * hd) as usize;
    let run = || {
        layer.prefill_attention_cache_skip_qkv(
            buffers.norm_output(),
            metrale_gpu_runtime::gpu::DevicePtr::NULL,
            M,
            h,
            nkv,
            hd,
            q_proj_dim,
            kv_dim,
            M as usize,
            2,
            &ctx,
            0,
        )
    };
    let (allocs, bytes) = (gpu.live_alloc_count(), gpu.live_bytes());
    let launches = gpu.launch_count();
    // 2026-09-25: Allocation before unwrap, as in the O-projection test above.
    let first = run();
    assert_eq!(
        (gpu.live_alloc_count(), gpu.live_bytes()),
        (allocs, bytes),
        "the Q/K/V prefill chain allocated on its FIRST call — that is the \
         off-ledger BF16 weight dequant from the H100 receipt, once per projection"
    );
    first.expect("first Q/K/V prefill chain");
    assert_eq!(
        gpu.launch_count() - launches,
        3,
        "expected one w8a16_gemm_pipelined per projection"
    );
    let second = run();
    assert_eq!(
        (gpu.live_alloc_count(), gpu.live_bytes()),
        (allocs, bytes),
        "the Q/K/V prefill chain allocated per call"
    );
    second.expect("second Q/K/V prefill chain");
    assert_eq!(gpu.launch_count() - launches, 6);
}
