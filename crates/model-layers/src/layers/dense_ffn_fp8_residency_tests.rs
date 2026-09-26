// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: A native-FP8 dense-FFN layer with null NVFP4 fallback weights still dispatches, and the MMQ finalizers skip it.
//!
//! Owner: model-layers (dense FFN).
//! Invariants: none beyond the types.
//!
//! On the native-FP8 route `qwen35_dense.rs` installs `QuantizedWeight::null()`
//! for the NVFP4 gate/up/down unless the residency plan keeps them
//! (`fp8_residency.rs`, `METRALE_DENSE_FP8_KEEP_NVFP4`). `forward_k2`,
//! `forward_k3` and `forward_km` redirect an FP8 layer to `forward_prefill`
//! (`native_small_batch_uses_prefill`), and `w8_gemm!` binds its transposed
//! operands to `None`, so no dispatch reads the null weights.

use super::{DenseFfnLayer, DenseFfnWeights};
use crate::layer::{ForwardContext, MoeLoraRoute};
use crate::layers::ops::{DerivedWeights, GemmDispatch, ModelLevers, ModelStats};
use crate::weight_map::{Fp8Weight, QuantizedWeight, WeightQuantFormat};
use metrale_config::ModelConfig;
use metrale_gpu_runtime::buffers::BufferArena;
use metrale_gpu_runtime::gpu::mock::MockGpuBackend;
use metrale_gpu_runtime::gpu::{GpuBackend, KernelHandle};

fn config() -> ModelConfig {
    let mut config = ModelConfig::qwen3_next_80b_nvfp4();
    config.hidden_size = 128;
    config.intermediate_size = 128;
    config.num_experts = 1;
    config.num_experts_per_tok = 1;
    config.moe_intermediate_size = 128;
    config.vocab_size = 128;
    config
}

/// 2026-09-25: The layer `qwen35_dense.rs` builds on the native-FP8 route when
/// the NVFP4 fallback is dropped: null NVFP4 weights, no transposed twins, an
/// FP8 overlay.
fn native_fp8_layer(gpu: &MockGpuBackend) -> DenseFfnLayer {
    let mut layer = DenseFfnLayer::new(
        DenseFfnWeights {
            gate_proj: QuantizedWeight::null(),
            up_proj: QuantizedWeight::null(),
            down_proj: QuantizedWeight::null(),
            gate_proj_t: None,
            up_proj_t: None,
            down_proj_t: None,
        },
        gpu,
    )
    .unwrap();
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
    layer
}

/// 2026-09-25: Sets every kernel handle the two load-time MMQ finalizers check,
/// so a no-op cannot be explained by a zero handle.
fn arm_the_mmq_finalizers(layer: &mut DenseFfnLayer) {
    for h in [
        &mut layer.q4k_mmq_nc_k,
        &mut layer.q4k_quant_act_k,
        &mut layer.q4k_quant_w_k,
        &mut layer.dequant_nvfp4_bf16_k,
        &mut layer.nvfp4_mmq_nc_k,
        &mut layer.nvfp4_quant_act_k,
        &mut layer.nvfp4_repack_k,
        &mut layer.nvfp4_silu_scaled_k,
    ] {
        *h = KernelHandle(0xBEEF);
    }
}

fn with_ctx(gpu: &MockGpuBackend, f: impl FnOnce(&ForwardContext, &BufferArena)) {
    let config = config();
    let buffers = BufferArena::new(&config, 8, 256, 256, 8, gpu).unwrap();
    let dispatch = GemmDispatch::defaults();
    let derived = DerivedWeights::new();
    let levers = ModelLevers::defaults();
    let stats = ModelStats::new();
    let ctx = ForwardContext {
        buffers: &buffers,
        hc_row_offset: 0,
        gpu,
        config: &config,
        dispatch: &dispatch,
        derived: &derived,
        levers: &levers,
        stats: &stats,
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
        moe_lora_route: MoeLoraRoute::Fold,
    };
    f(&ctx, &buffers);
}

#[test]
fn the_verify_arm_still_selects_this_layer_without_nvfp4_weights() {
    // 2026-09-25: `can_forward_km` gates the multi-row decode branches in
    // `qwen3_attention/trait_impl/multi_seq/ffn.rs` and
    // `qwen3_ssm/trait_decode_multi_seq.rs`; a native-FP8 layer stays eligible
    // without NVFP4 weights.
    let gpu = MockGpuBackend::new();
    let layer = native_fp8_layer(&gpu);
    for rows in 1..=8 {
        assert!(
            layer.can_forward_km(rows),
            "native FP8 layer must stay eligible at m={rows}"
        );
    }
}

#[test]
fn a_layer_with_neither_nvfp4_nor_fp8_weights_is_not_eligible() {
    // 2026-09-25: Null NVFP4 weights and no FP8 overlay: not eligible.
    let gpu = MockGpuBackend::new();
    let mut layer = DenseFfnLayer::new(
        DenseFfnWeights {
            gate_proj: QuantizedWeight::null(),
            up_proj: QuantizedWeight::null(),
            down_proj: QuantizedWeight::null(),
            gate_proj_t: None,
            up_proj_t: None,
            down_proj_t: None,
        },
        &gpu,
    )
    .unwrap();
    layer.act_mul = KernelHandle(0xAC7);
    assert!(!layer.can_forward_km(4));
}

#[test]
fn the_mmq_finalizers_are_no_ops_under_a_native_fp8_overlay() {
    // 2026-09-25: `finalize_nvfp4_mmq_load` needs no opt-in variable, and over
    // null NVFP4 sources it would repack null pointers. Both finalizers return
    // early on a layer with an FP8 overlay.
    let gpu = MockGpuBackend::new();
    let mut layer = native_fp8_layer(&gpu);
    arm_the_mmq_finalizers(&mut layer);
    let allocs = gpu.alloc_count();
    let launches = gpu.launch_count();
    layer.finalize_q4k_load(&gpu, 128, 128, 7).unwrap();
    layer.finalize_nvfp4_mmq_load(&gpu, 128, 128, 7).unwrap();
    assert_eq!(gpu.alloc_count(), allocs, "no MMQ repack may be allocated");
    assert_eq!(gpu.launch_count(), launches, "no kernel may be launched");
}

#[test]
fn every_small_batch_entry_point_runs_without_nvfp4_weights() {
    // 2026-09-25: `forward_k2`, `forward_k3` and `forward_km` redirect an FP8
    // layer to `forward_prefill`. A launch that read a null NVFP4 weight would
    // carry a null buffer argument on the mock backend.
    let gpu = MockGpuBackend::new();
    let layer = native_fp8_layer(&gpu);
    with_ctx(&gpu, |ctx, buffers| {
        let start = gpu.launch_count();
        let allocs = gpu.alloc_count();
        let input = buffers.norm_output();
        layer.forward(input, ctx, 7).unwrap();
        layer.forward_k2(input, ctx, 7).unwrap();
        layer.forward_k3(input, ctx, 7).unwrap();
        layer.forward_km(input, 4, ctx, 7).unwrap();
        layer.forward_prefill(input, 64, ctx, 7).unwrap();
        assert_eq!(
            gpu.alloc_count(),
            allocs,
            "dispatch must not allocate weight copies"
        );
        let launches = gpu.launches_snapshot();
        assert!(launches.len() > start, "the forwards must have dispatched");
        for launch in &launches[start..] {
            for arg in &launch.args {
                if let metrale_gpu_runtime::gpu::mock::MockArg::Buffer(p) = arg {
                    assert!(
                        !p.is_null(),
                        "a NULL NVFP4 weight reached a kernel launch: {launch:?}"
                    );
                }
            }
        }
    });
}
