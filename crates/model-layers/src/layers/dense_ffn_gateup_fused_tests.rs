// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: CPU tests for the fused gate+up arm: which widths it claims, its launch count, and that it allocates nothing.
//!
//! Owner: model-layers (dense FFN).
//! Invariants: none beyond the types.
//!
//! Numerics are graded on the GPU by
//! `examples/native_fp8_ffn_gateup_fused_microtest.rs`, which requires byte
//! equality with the un-fused pair.

use super::{fused_out_bytes, gateup_fused_selected};
use crate::layer::{ForwardContext, MoeLoraRoute};
use crate::layers::dense_ffn::{DenseFfnLayer, DenseFfnWeights};
use crate::layers::ops::{self, DerivedWeights, GemmDispatch, ModelLevers, ModelStats};
use crate::weight_map::{Fp8Weight, QuantizedWeight, WeightQuantFormat};
use metrale_config::ModelConfig;
use metrale_gpu_runtime::buffers::{BufferArena, GATEUP_FUSED_MAX_M};
use metrale_gpu_runtime::gpu::mock::MockGpuBackend;
use metrale_gpu_runtime::gpu::{GpuBackend, KernelHandle};

/// 2026-09-25: Qwen3.8-27B dense-FFN widths (`kernels/gb10/qwen3.8-27b/MODEL.toml`).
const H: u32 = 5120;
const INTER: u32 = 17408;

/// 2026-09-25: The rule with every clause but `m` at its selecting value, so
/// each case below perturbs one thing.
fn selected(m: u32) -> bool {
    gateup_fused_selected(m, INTER, true, true, true, true, usize::MAX)
}

/// 2026-09-25: The band is 5..=`GATEUP_FUSED_MAX_M`. Below 5 the W8A8 rule this
/// arm rides on does not select; the upper edge is also the row extent of the
/// arena's fused output buffer.
#[test]
fn the_fused_arm_claims_exactly_the_five_to_sixteen_row_decode_band() {
    for m in [5_u32, 6, 8, 12, 15, 16] {
        assert!(selected(m), "m={m} is inside the decode band");
    }
    for m in [1_u32, 2, 4] {
        assert!(!selected(m), "m={m} belongs to the batch4 GEMV tier");
    }
    for m in [17_u32, 25, 32, 1168, 4576] {
        assert!(
            !selected(m),
            "m={m} is a prefill width, where the arm is inert"
        );
    }
    assert_eq!(
        GATEUP_FUSED_MAX_M, 16,
        "the band's upper edge and the arena buffer's row extent are ONE \
         constant; changing it here without the arena is a cross-buffer write"
    );
}

/// 2026-09-25: Lever off (`METRALE_FFN_GATEUP_FUSED=0`, or a target that
/// declares `ffn_gateup_fused = false`).
#[test]
fn the_lever_off_declines_at_every_width() {
    for m in 1_u32..=32 {
        assert!(!gateup_fused_selected(
            m,
            INTER,
            false,
            true,
            true,
            true,
            usize::MAX
        ));
    }
}

/// 2026-09-25: The fused GEMM is the W8A8 arm at twice the N. If the ladder
/// would have put either half on a W8A16 rung, fusing would change that half's
/// arithmetic, so the fused arm declines.
#[test]
fn the_w8a8_arm_must_have_claimed_both_halves() {
    assert!(!gateup_fused_selected(
        8,
        INTER,
        true,
        false,
        true,
        true,
        usize::MAX
    ));
}

/// 2026-09-25: No fused weight, or no strided SiLU consumer: the arm declines
/// rather than launch with a null pointer or the wrong stride.
#[test]
fn a_missing_weight_or_a_missing_consumer_declines() {
    assert!(!gateup_fused_selected(
        8,
        INTER,
        true,
        true,
        false,
        true,
        usize::MAX
    ));
    assert!(!gateup_fused_selected(
        8,
        INTER,
        true,
        true,
        true,
        false,
        usize::MAX
    ));
}

/// 2026-09-25: An output buffer one byte short of `fused_out_bytes` declines:
/// the cuBLASLt arm writes `ceil16(m)` rows.
#[test]
fn an_output_buffer_short_of_the_padded_extent_declines() {
    let need = fused_out_bytes(16, INTER);
    assert_eq!(need, 16 * 2 * INTER as usize * 2, "ceil16(16) == 16");
    assert!(gateup_fused_selected(
        16, INTER, true, true, true, true, need
    ));
    assert!(!gateup_fused_selected(
        16,
        INTER,
        true,
        true,
        true,
        true,
        need - 1
    ));
    // 2026-09-25: m=5 is sized for 16 rows, the padded M cuBLASLt is handed.
    assert_eq!(fused_out_bytes(5, INTER), need);
}

struct Harness {
    gpu: MockGpuBackend,
    layer: DenseFfnLayer,
    buffers: BufferArena,
    config: ModelConfig,
    fp8: Fp8Weight,
}

/// 2026-09-25: A dense config at the real FFN widths. With `armed`, the lever
/// field, the strided SiLU handle and the fused weight are set on the layer
/// directly, because the production lever is a process-global `OnceLock`.
fn harness(armed: bool) -> Harness {
    let gpu = MockGpuBackend::new();
    let mut config = ModelConfig::qwen3_next_80b_nvfp4();
    config.hidden_size = H as usize;
    config.intermediate_size = INTER as usize;
    config.num_experts = 0;
    config.num_experts_per_tok = 0;
    config.moe_intermediate_size = INTER as usize;
    config.vocab_size = 256;
    let buffers = BufferArena::new(&config, 64, 256, 256, 8, &gpu).unwrap();
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
    layer.per_token_group_quant_fp8_k =
        crate::layers::ops::Fp8ActQuant::shared_only(KernelHandle(0xA8A));
    layer.fp8_gemm_t_blockscaled_k = KernelHandle(0xA88);
    // 2026-09-25: `GemmDispatch::defaults()` leaves `cublas.ffn` off, so
    // `w8a8_gemm` takes the in-tree kernel; the k-major adapter handle is zeroed
    // as well. The fused arm's rule does not depend on which GEMM runs.
    layer.fp8_act_scale_kmajor_k = KernelHandle(0);
    layer.gateup_fused = armed;
    layer.silu_mul_strided_k = if armed {
        KernelHandle(0x5171)
    } else {
        KernelHandle(0)
    };
    // 2026-09-25: One fused allocation with gate and up as views inside it, as
    // the loader builds it, so `set_fp8_gate_up_fused`'s debug assert holds.
    let fused_w = gpu.alloc(2 * INTER as usize * H as usize).unwrap();
    let grid = (INTER as usize / 128) * (H as usize / 128) * 4;
    let fused_s = gpu.alloc(2 * grid).unwrap();
    let view = |off_w: usize, off_s: usize| Fp8Weight {
        weight: fused_w.offset(off_w),
        row_scale: fused_s.offset(off_s),
        n: INTER,
        k: H,
        scale_format: WeightQuantFormat::Fp8BlockScaled,
    };
    let fp8 = view(0, 0);
    layer.set_fp8_weights(
        fp8,
        view(INTER as usize * H as usize, grid),
        Fp8Weight {
            weight: fused_w,
            row_scale: fused_s,
            n: H,
            k: INTER,
            scale_format: WeightQuantFormat::Fp8BlockScaled,
        },
    );
    if armed {
        layer.set_fp8_gate_up_fused(Fp8Weight {
            weight: fused_w,
            row_scale: fused_s,
            n: 2 * INTER,
            k: H,
            scale_format: WeightQuantFormat::Fp8BlockScaled,
        });
    }
    Harness {
        gpu,
        layer,
        buffers,
        config,
        fp8,
    }
}

fn ctx<'a>(
    h: &'a Harness,
    dispatch: &'a GemmDispatch,
    derived: &'a DerivedWeights,
    levers: &'a ModelLevers,
    stats: &'a ModelStats,
) -> ForwardContext<'a> {
    ForwardContext {
        dispatch,
        derived,
        levers,
        stats,
        buffers: &h.buffers,
        hc_row_offset: 0,
        gpu: &h.gpu,
        config: &h.config,
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
    }
}

/// 2026-09-25: A GeLU layer never takes the fused arm: its consumer is the
/// SiLU-mul. The activation is a property of the layer, so the pure rule
/// cannot show this.
#[test]
fn a_gelu_layer_never_takes_the_fused_arm() {
    let mut h = harness(true);
    h.layer.activation = crate::layers::dense_ffn::FfnActivation::GeLU;
    let (d, w, l, s) = (
        GemmDispatch::defaults(),
        DerivedWeights::new(),
        ModelLevers::defaults(),
        ModelStats::new(),
    );
    let c = ctx(&h, &d, &w, &l, &s);
    for m in [5_u32, 8, 16] {
        assert!(h.layer.gateup_fused_plan(&c, m, INTER, true).is_none());
    }
}

/// 2026-09-25: The layer plans the arm at the decode widths and nowhere else,
/// which also pins its wiring to the installed weight, the lever field and the
/// arena's capacity.
#[test]
fn the_layer_plans_the_fused_arm_only_inside_the_band() {
    let h = harness(true);
    let (d, w, l, s) = (
        GemmDispatch::defaults(),
        DerivedWeights::new(),
        ModelLevers::defaults(),
        ModelStats::new(),
    );
    let c = ctx(&h, &d, &w, &l, &s);
    for m in [5_u32, 8, 16] {
        assert!(h.layer.gateup_fused_plan(&c, m, INTER, true).is_some());
    }
    for m in [1_u32, 4, 17, 64] {
        assert!(h.layer.gateup_fused_plan(&c, m, INTER, true).is_none());
    }
    // 2026-09-25: Unarmed (lever off, no SiLU handle, no fused weight): never
    // planned.
    let off = harness(false);
    let c_off = ctx(&off, &d, &w, &l, &s);
    for m in [5_u32, 8, 16] {
        assert!(
            off.layer
                .gateup_fused_plan(&c_off, m, INTER, true)
                .is_none()
        );
    }
}

/// 2026-09-25: The fused arm allocates nothing. The live count is taken before
/// the first call, not between two calls, so an allocation cached across
/// calls is caught too.
#[test]
fn the_fused_arm_allocates_nothing_per_call() {
    let h = harness(true);
    let (d, w, l, s) = (
        GemmDispatch::defaults(),
        DerivedWeights::new(),
        ModelLevers::defaults(),
        ModelStats::new(),
    );
    let c = ctx(&h, &d, &w, &l, &s);
    let fused = h
        .layer
        .gateup_fused_plan(&c, 16, INTER, true)
        .expect("armed at m=16");
    let (allocs, bytes) = (h.gpu.live_alloc_count(), h.gpu.live_bytes());
    let (a8, sc) = h
        .layer
        .w8a8_quant_act(&c, h.buffers.norm_output(), 16, H, 0)
        .expect("activation quant");
    for _ in 0..3 {
        h.layer
            .w8a8_gate_up_fused(
                &c,
                a8,
                sc,
                fused,
                h.buffers.expert_gate_out(),
                16,
                INTER,
                H,
                0,
            )
            .expect("fused gate+up");
    }
    assert_eq!(
        (h.gpu.live_alloc_count(), h.gpu.live_bytes()),
        (allocs, bytes),
        "the fused gate+up arm must allocate nothing — every operand is an \
         arena buffer or a weight view"
    );
}

/// 2026-09-25: The fused arm issues one GEMM and one SiLU launch where the
/// un-fused pair issues two GEMMs and one SiLU.
#[test]
fn the_fused_arm_issues_one_gemm_where_the_pair_issues_two() {
    let h = harness(true);
    let (d, w, l, s) = (
        GemmDispatch::defaults(),
        DerivedWeights::new(),
        ModelLevers::defaults(),
        ModelStats::new(),
    );
    let c = ctx(&h, &d, &w, &l, &s);
    let fused = h.layer.gateup_fused_plan(&c, 16, INTER, true).unwrap();
    let (a8, sc) = h
        .layer
        .w8a8_quant_act(&c, h.buffers.norm_output(), 16, H, 0)
        .unwrap();
    let before = h.gpu.launch_count();
    h.layer
        .w8a8_gate_up_fused(
            &c,
            a8,
            sc,
            fused,
            h.buffers.expert_gate_out(),
            16,
            INTER,
            H,
            0,
        )
        .unwrap();
    assert_eq!(
        h.gpu.launch_count() - before,
        2,
        "one block-scaled GEMM + one strided SiLU"
    );

    let before = h.gpu.launch_count();
    for out in [h.buffers.expert_gate_out(), h.buffers.expert_up_out()] {
        h.layer
            .w8a8_gemm(
                &c,
                a8,
                sc,
                &h.fp8,
                out,
                h.buffers.expert_gate_out_bytes(),
                16,
                INTER,
                H,
                0,
            )
            .unwrap();
    }
    ops::silu_mul(
        &h.gpu,
        h.layer.act_mul,
        h.buffers.expert_gate_out(),
        h.buffers.expert_up_out(),
        h.buffers.expert_gate_out(),
        16 * INTER,
        0,
    )
    .unwrap();
    assert_eq!(h.gpu.launch_count() - before, 3);
}
