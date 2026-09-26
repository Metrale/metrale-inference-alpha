// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Allocation test for the GDN QKVZ prefill projection with every
//! cuBLAS family armed. The layer comes from `tests::native_fp8_gdn_layer`.
//!
//! Owner: model-layers (qwen3_ssm tests).
//! Invariants: none beyond the types.

use super::tests::native_fp8_gdn_layer;
use super::*;
use metrale_config::ModelConfig;
use metrale_gpu_runtime::buffers::BufferArena;
use metrale_gpu_runtime::gpu::mock::MockGpuBackend;

/// 2026-09-25: With `CublasScope::ALL` and the k-major adapter handle zeroed,
/// the W8A8 arm takes the in-tree kernel, which the mock backend can run (the
/// cuBLASLt arm's clauses are tested in `prefill_w8a8_tests.rs`). Two calls
/// allocate nothing. The baseline is read before the first call, so an
/// allocation made once and cached would still fail the test.
#[test]
fn ssm_qkvz_prefill_allocates_nothing_with_the_cublas_lever_armed() {
    use crate::layers::ops::CublasScope;

    let config = ModelConfig::qwen3_next_80b_nvfp4();
    let gpu = MockGpuBackend::new();
    let mut layer = native_fp8_gdn_layer(&gpu, &config, true, true);
    layer.fp8_act_scale_kmajor_k = metrale_gpu_runtime::gpu::KernelHandle(0);

    let buffers = BufferArena::new(&config, 64, 4096, 16, 32, &gpu).unwrap();
    let dispatch = crate::layers::ops::GemmDispatch {
        cublas: CublasScope::ALL,
        ..crate::layers::ops::GemmDispatch::defaults()
    };
    let derived = crate::layers::ops::DerivedWeights::new();
    let levers = crate::layers::ops::ModelLevers::defaults();
    let stats = crate::layers::ops::ModelStats::new();
    let ctx = ForwardContext {
        dispatch: &dispatch,
        derived: &derived,
        levers: &levers,
        stats: &stats,
        buffers: &buffers,
        hc_row_offset: 0,
        gpu: &gpu,
        config: &config,
        attn_metadata: None,
        profile: false,
        comm: None,
        graph_capture: false,
        decode_step: false,
        gdn_exact_replay: false,
        gdn_write_on_accept: false,
        token_ids: None,
        host_token_ids: None,
        routed_lora_layers: None,
        midchunk_capture: None,
        moe_lora_route: crate::layer::MoeLoraRoute::Fold,
    };

    let m = 28_u32;
    let qkvz_size = config.ssm_qkvz_size();
    let run = || {
        layer.prefill_qkvz_proj(
            buffers.norm_output(),
            buffers.ssm_deinterleaved(),
            m,
            qkvz_size,
            config.hidden_size,
            config.linear_num_key_heads,
            config.linear_key_head_dim,
            config.linear_num_value_heads / config.linear_num_key_heads.max(1),
            config.linear_value_head_dim,
            &ctx,
            0,
        )
    };
    let (allocs, bytes) = (gpu.live_alloc_count(), gpu.live_bytes());
    let launches = gpu.launch_count();
    run().expect("first QKVZ prefill projection");
    assert_eq!(
        (gpu.live_alloc_count(), gpu.live_bytes()),
        (allocs, bytes),
        "the QKVZ prefill projection allocated on its FIRST call — that is the \
         167772160-byte per-layer BF16 weight dequant from the H100 receipt"
    );
    // 2026-09-25: Exactly two launches per call (the activation quant and the
    // GEMM), so "allocated nothing" cannot pass by doing nothing, and an added
    // dequant launch fails the count.
    assert_eq!(
        gpu.launch_count() - launches,
        2,
        "expected per_token_group_quant_fp8 + fp8_gemm_t_blockscaled"
    );
    run().expect("second QKVZ prefill projection");
    assert_eq!(
        (gpu.live_alloc_count(), gpu.live_bytes()),
        (allocs, bytes),
        "the QKVZ prefill projection allocated per call"
    );
    assert_eq!(gpu.launch_count() - launches, 4);
}
