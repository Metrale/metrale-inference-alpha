// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Which kernel each row count of a native-FP8 dense-FFN layer launches, with the NVFP4 fallback never read.
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

fn run_case(
    rows: u32,
    prefill: bool,
    expected: u64,
    grid: [u32; 3],
    block: [u32; 3],
    configure: impl FnOnce(&mut DenseFfnLayer),
) {
    let gpu = MockGpuBackend::new();
    let mut config = ModelConfig::qwen3_next_80b_nvfp4();
    config.hidden_size = 128;
    config.intermediate_size = 128;
    config.num_experts = 1;
    config.num_experts_per_tok = 1;
    config.moe_intermediate_size = 128;
    config.vocab_size = 128;
    let buffers = BufferArena::new(&config, 8, 256, 256, 8, &gpu).unwrap();
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
    layer.act_mul = KernelHandle(0xAC7);
    let fp8 = Fp8Weight {
        weight: gpu.alloc(128 * 128).unwrap(),
        row_scale: gpu.alloc(4).unwrap(),
        n: 128,
        k: 128,
        scale_format: WeightQuantFormat::Fp8BlockScaled,
    };
    layer.set_fp8_weights(fp8, fp8, fp8);
    configure(&mut layer);
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
    let start = gpu.launch_count();
    let allocs = gpu.alloc_count();
    if prefill {
        layer
            .forward_prefill(buffers.norm_output(), rows as usize, &ctx, 7)
            .unwrap();
    } else {
        assert!(layer.can_forward_km(rows));
        layer
            .forward_km(buffers.norm_output(), rows, &ctx, 7)
            .unwrap();
    }
    assert_eq!(
        gpu.alloc_count(),
        allocs,
        "dispatch must not allocate weight copies"
    );
    let launches = gpu.launches_snapshot();
    let launches = &launches[start..];
    let projections: Vec<_> = launches
        .iter()
        .filter(|launch| launch.func != layer.act_mul.0)
        .collect();
    assert_eq!(projections.len(), 3);
    for launch in projections {
        assert_eq!(
            launch.func, expected,
            "M={rows}: gate/up/down selected the wrong native kernel"
        );
        assert_eq!(launch.grid, grid);
        assert_eq!(launch.block, block);
        assert_eq!(launch.stream, 7);
        assert_eq!(launch.args[1], MockArg::Buffer(fp8.weight));
        assert_eq!(launch.args[2], MockArg::Buffer(fp8.row_scale));
        assert_eq!(launch.args[4], MockArg::Bytes(rows.to_ne_bytes().to_vec()));
        assert_eq!(
            launch.args[5],
            MockArg::Bytes(128_u32.to_ne_bytes().to_vec())
        );
        assert_eq!(
            launch.args[6],
            MockArg::Bytes(128_u32.to_ne_bytes().to_vec())
        );
        assert!(!launch.args.contains(&MockArg::Buffer(fallback.weight)));
    }
}

#[test]
fn small_native_ffn_uses_existing_batched_gemv() {
    for rows in [1, 2, 3, 4] {
        run_case(rows, true, 0xDEAD, [32, 1, 1], [256, 1, 1], |_| {});
    }
    run_case(4, false, 0xDEAD, [32, 1, 1], [256, 1, 1], |_| {});
}

#[test]
fn larger_native_ffn_uses_existing_pipelined_gemm() {
    // 2026-09-25: With the batch16 rung disarmed (`batch16_enabled` false, the
    // default) or its handle absent, 5 and 8 rows reach the pipelined GEMM.
    run_case(129, true, 0xDEAD, [4, 2, 1], [256, 1, 1], |_| {});
    for rows in [5, 8] {
        run_case(
            rows,
            true,
            0xDEAD,
            [4, rows.div_ceil(128), 1],
            [256, 1, 1],
            |layer| layer.w8a16_gemv_batch16_k = KernelHandle(0),
        );
        run_case(
            rows,
            true,
            0xDEAD,
            [4, rows.div_ceil(128), 1],
            [256, 1, 1],
            |layer| layer.batch16_enabled = false,
        );
    }
    run_case(5, false, 0xDEAD, [4, 1, 1], [256, 1, 1], |layer| {
        layer.w8a16_gemv_batch16_k = KernelHandle(0)
    });
}

/// 2026-09-25: With the batch16 rung armed (`METRALE_FFN_BATCH16=1` sets
/// `batch16_enabled`), 5..=16 rows take `w8a16_gemv_batch16`, the MAX_M=16
/// instantiation of the `w8a16_gemv_batch4` template: the same grid and block
/// as the batch4 rung and one launch per projection, told apart only by the
/// handle.
#[test]
fn five_to_sixteen_row_native_ffn_uses_the_batch16_gemv_when_armed() {
    for rows in [5, 8, 16] {
        run_case(rows, true, 0xB16, [32, 1, 1], [256, 1, 1], |layer| {
            layer.w8a16_gemv_batch16_k = KernelHandle(0xB16);
            layer.batch16_enabled = true;
        });
    }
    run_case(5, false, 0xB16, [32, 1, 1], [256, 1, 1], |layer| {
        layer.w8a16_gemv_batch16_k = KernelHandle(0xB16);
        layer.batch16_enabled = true;
    });
}

#[test]
fn missing_small_batch_kernel_uses_same_precision_pipelined_fallback() {
    run_case(4, false, 0xDEAD, [4, 1, 1], [256, 1, 1], |layer| {
        layer.w8a16_gemv_batch4_k = KernelHandle(0);
    });
}

#[test]
fn missing_optional_kernels_keep_base_native_gemm() {
    for rows in [4, 5, 129] {
        run_case(
            rows,
            true,
            0xF08,
            [2, rows.div_ceil(64), 1],
            [128, 1, 1],
            |layer| {
                layer.w8a16_gemv_batch4_k = KernelHandle(0);
                layer.w8a16_gemv_batch16_k = KernelHandle(0);
                layer.w8a16_gemm_pipelined_k = KernelHandle(0);
            },
        );
    }
}
