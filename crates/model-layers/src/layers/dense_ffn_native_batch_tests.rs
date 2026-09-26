// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `forward_km` launches the installed weight format: the FP8 overlay when present, the NVFP4 batch kernels otherwise.
//!
//! Owner: model-layers (dense FFN).
//! Invariants: none beyond the types.

use super::{DenseFfnLayer, DenseFfnWeights};
use crate::layer::{ForwardContext, MoeLoraRoute};
use crate::layers::ops::{DerivedWeights, GemmDispatch, ModelLevers, ModelStats};
use crate::weight_map::{Fp8Weight, QuantizedWeight, WeightQuantFormat};
use metrale_config::ModelConfig;
use metrale_gpu_runtime::buffers::BufferArena;
use metrale_gpu_runtime::gpu::mock::{MockArg, MockGpuBackend};
use metrale_gpu_runtime::gpu::{GpuBackend, KernelHandle};

fn run_batch(native_fp8: bool, rows: u32) {
    let gpu = MockGpuBackend::new();
    let mut config = ModelConfig::qwen3_next_80b_nvfp4();
    config.hidden_size = 128;
    config.intermediate_size = 128;
    config.num_experts = 1;
    config.num_experts_per_tok = 1;
    config.moe_intermediate_size = 128;
    config.vocab_size = 128;
    let buffers = BufferArena::new(&config, 8, 16, 16, 8, &gpu).unwrap();
    // 2026-09-25: Non-null NVFP4 fallback weights sit beside the FP8 overlay,
    // so a launch that reads them is caught.
    let mut fallback = QuantizedWeight::null();
    fallback.weight = gpu.alloc(128 * 64).unwrap();
    fallback.weight_scale = gpu.alloc(128 * 8).unwrap();
    let mut layer = DenseFfnLayer::new(
        DenseFfnWeights {
            gate_proj: fallback,
            up_proj: fallback,
            down_proj: fallback,
            gate_proj_t: None,
            up_proj_t: None,
            down_proj_t: None,
        },
        &gpu,
    )
    .unwrap();
    // 2026-09-25: The mock returns one placeholder handle for every kernel name;
    // distinct handles here identify the dispatch route.
    layer.w8a16_gemm_k = KernelHandle(0xF08);
    // 2026-09-25: With the optional FP8 kernels absent, the FP8 path lands on
    // the base `w8a16_gemm`.
    layer.w8a16_gemv_batch4_k = KernelHandle(0);
    layer.w8a16_gemv_batch16_k = KernelHandle(0);
    layer.w8a16_gemm_pipelined_k = KernelHandle(0);
    layer.act_mul = KernelHandle(0xAC7);
    let fp8 = Fp8Weight {
        weight: gpu.alloc(128 * 128).unwrap(),
        row_scale: gpu.alloc(4).unwrap(),
        n: 128,
        k: 128,
        scale_format: WeightQuantFormat::Fp8BlockScaled,
    };
    if native_fp8 {
        layer.set_fp8_weights(fp8, fp8, fp8);
    }
    let dispatch = GemmDispatch::defaults();
    let derived = DerivedWeights::new();
    let levers = ModelLevers::defaults();
    let stats = ModelStats::new();
    let ctx = ForwardContext {
        buffers: &buffers,
        hc_row_offset: 0,
        gpu: &gpu,
        config: &config,
        dispatch: &dispatch,
        derived: &derived,
        levers: &levers,
        stats: &stats,
        attn_metadata: None,
        profile: false,
        comm: None,
        graph_capture: false,
        // 2026-09-25: Prefill shape; the dense FFN never reads `decode_step`.
        decode_step: false,
        gdn_exact_replay: false,
        gdn_write_on_accept: false,
        token_ids: None,
        host_token_ids: None,
        routed_lora_layers: None,
        midchunk_capture: None,
        moe_lora_route: MoeLoraRoute::Fold,
    };
    assert!(layer.can_forward_km(rows));
    let start = gpu.launch_count();
    layer
        .forward_km(buffers.norm_output(), rows, &ctx, 0)
        .unwrap();
    let launches = gpu.launches_snapshot();
    let launches = &launches[start..];
    let expected = if native_fp8 {
        layer.w8a16_gemm_k
    } else {
        layer.batchm_kernel(rows)
    };
    assert_eq!(
        launches
            .iter()
            .filter(|launch| launch.func == expected.0)
            .count(),
        3,
        "M={rows}: gate/up/down must all use the installed weight format"
    );
    if native_fp8 {
        assert!(
            launches
                .iter()
                .all(|launch| { !launch.args.contains(&MockArg::Buffer(fallback.weight)) }),
            "M={rows}: native FP8 dispatch read the NVFP4 fallback"
        );
    }
}

#[test]
fn native_fp8_multi_row_dispatch_preserves_checkpoint_weights() {
    for rows in [4, 5, 8] {
        run_batch(true, rows);
    }
}

#[test]
fn nvfp4_multi_row_dispatch_keeps_existing_batch_kernels() {
    for rows in [4, 5, 8] {
        run_batch(false, rows);
    }
}
