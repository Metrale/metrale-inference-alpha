// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: CPU tests for the W8A8 dense-FFN selection rule and the buffer extents its cuBLASLt arm writes.
//!
//! Owner: model-layers (dense FFN).
//! Invariants: none beyond the types.
//!
//! Numerics are graded on the GPU by `examples/native_fp8_ffn_w8a8_microtest.rs`.

use super::{max_m_for, w8a8_prefill_selected};
use crate::layer::{ForwardContext, MoeLoraRoute};
use crate::layers::dense_ffn::{DenseFfnLayer, DenseFfnWeights};
use crate::layers::ops::{
    self, DerivedWeights, GemmDispatch, ModelLevers, ModelStats, cublas_fp8_m_pad,
};
use crate::weight_map::{Fp8Weight, QuantizedWeight, WeightQuantFormat};
use metrale_config::ModelConfig;
use metrale_gpu_runtime::buffers::BufferArena;
use metrale_gpu_runtime::gpu::mock::MockGpuBackend;
use metrale_gpu_runtime::gpu::{GpuBackend, KernelHandle};

/// 2026-09-25: Qwen3.8-27B dense-FFN widths (`kernels/gb10/qwen3.8-27b/MODEL.toml`):
/// gate/up are `[INTER, H]`, down is `[H, INTER]`.
const H: u32 = 5120;
const INTER: u32 = 17408;
const PROMPT_TOKENS: u32 = 1193;
/// 2026-09-25: A quantizer with only the shared kernel. The rule checks only
/// `Fp8ActQuant::available`, so the twin handle does not change these cases.
const QUANT_K: ops::Fp8ActQuant = ops::Fp8ActQuant {
    shared: KernelHandle(0xA8A),
    hopper: KernelHandle(0),
};
const NO_QUANT: ops::Fp8ActQuant = ops::Fp8ActQuant {
    shared: KernelHandle(0),
    hopper: KernelHandle(0),
};
const GEMM_K: KernelHandle = KernelHandle(0xA88);

/// 2026-09-25: The rule with no M cap, for the cases that perturb another
/// clause; the ceiling has its own cases below.
#[allow(clippy::too_many_arguments)]
fn selected(
    m: u32,
    n: u32,
    k: u32,
    fmt: WeightQuantFormat,
    blockscaled_lever: bool,
    quant_k: ops::Fp8ActQuant,
    gemm_k: KernelHandle,
    w8a16_only: bool,
) -> bool {
    selected_capped(
        m,
        n,
        k,
        fmt,
        blockscaled_lever,
        quant_k,
        gemm_k,
        w8a16_only,
        u32::MAX,
    )
}

#[allow(clippy::too_many_arguments)]
fn selected_capped(
    m: u32,
    n: u32,
    k: u32,
    fmt: WeightQuantFormat,
    blockscaled_lever: bool,
    quant_k: ops::Fp8ActQuant,
    gemm_k: KernelHandle,
    w8a16_only: bool,
    max_m: u32,
) -> bool {
    w8a8_prefill_selected(
        m,
        n,
        k,
        fmt,
        blockscaled_lever,
        quant_k,
        gemm_k,
        w8a16_only,
        max_m,
    )
}

/// 2026-09-25: The ceiling is inclusive: `m <= max_m` selects and
/// `max_m + 1` does not.
#[test]
fn the_ceiling_is_inclusive_and_cuts_above_it() {
    let case = |m: u32, max_m: u32| {
        selected_capped(
            m,
            INTER,
            H,
            WeightQuantFormat::Fp8BlockScaled,
            true,
            QUANT_K,
            KernelHandle(1),
            false,
            max_m,
        )
    };
    assert!(case(63, 64), "below the ceiling stays on W8A8");
    assert!(
        case(64, 64),
        "AT the ceiling stays on W8A8 — the bound is <="
    );
    assert!(
        !case(65, 64),
        "one row above the ceiling falls back to W8A16"
    );
    assert!(
        !case(949, 64),
        "the served M that measured -23.4% falls back"
    );
}

/// 2026-09-25: `u32::MAX` caps nothing, including `m = u32::MAX` itself.
#[test]
fn the_baseline_ceiling_caps_nothing() {
    let case = |m: u32| {
        selected_capped(
            m,
            INTER,
            H,
            WeightQuantFormat::Fp8BlockScaled,
            true,
            QUANT_K,
            KernelHandle(1),
            false,
            u32::MAX,
        )
    };
    assert!(case(949));
    assert!(case(u32::MAX));
}

/// 2026-09-25: A ceiling of 0 selects nothing. `ops::target_defaults::resolve_max_m`
/// passes a parsed 0 through, so an operator's 0 turns the arm off for that
/// shape.
#[test]
fn a_zero_ceiling_selects_nothing() {
    for m in [5, 64, 949] {
        assert!(!selected_capped(
            m,
            INTER,
            H,
            WeightQuantFormat::Fp8BlockScaled,
            true,
            QUANT_K,
            KernelHandle(1),
            false,
            0,
        ));
    }
}

fn gate_up(m: u32) -> bool {
    selected(
        m,
        INTER,
        H,
        WeightQuantFormat::Fp8BlockScaled,
        true,
        QUANT_K,
        GEMM_K,
        false,
    )
}

fn down(m: u32) -> bool {
    selected(
        m,
        H,
        INTER,
        WeightQuantFormat::Fp8BlockScaled,
        true,
        QUANT_K,
        GEMM_K,
        false,
    )
}

#[test]
fn selected_for_prefill_batches_at_the_real_ffn_shapes() {
    for m in [64, PROMPT_TOKENS] {
        assert!(gate_up(m), "gate/up must take W8A8 at M={m}");
        assert!(down(m), "down must take W8A8 at M={m}");
    }
}

#[test]
fn not_selected_for_small_batches_that_belong_to_the_gemv() {
    // 2026-09-25: m <= 4 belongs to the `w8a16_gemv_batch4` rung.
    for m in [1, 2, 3, 4] {
        assert!(!gate_up(m), "M={m} must stay on the batch4 GEMV");
    }
    assert!(gate_up(5), "M=5 is the first W8A8 row count");
}

#[test]
fn not_selected_for_per_row_scales() {
    // 2026-09-25: Per-row and single-scale weights do not carry the
    // `[N/128, K/128]` grid the block-scaled GEMM indexes.
    for fmt in [
        WeightQuantFormat::Fp8PerRow,
        WeightQuantFormat::Fp8SingleScale,
    ] {
        assert!(
            !selected(64, INTER, H, fmt, true, QUANT_K, GEMM_K, false),
            "{fmt:?} must not take the block-scaled GEMM"
        );
    }
}

#[test]
fn not_selected_for_unaligned_shapes() {
    let f = WeightQuantFormat::Fp8BlockScaled;
    // 2026-09-25: K not a multiple of 128: the activation quantizer emits one
    // FP32 scale per 128-wide K group, so a ragged tail has no scale.
    assert!(!selected(64, INTER, H + 1, f, true, QUANT_K, GEMM_K, false));
    assert!(!selected(64, INTER, 127, f, true, QUANT_K, GEMM_K, false));
    // 2026-09-25: N not a multiple of 128: the weight scale grid is `[N/128, K/128]`.
    assert!(!selected(64, INTER + 1, H, f, true, QUANT_K, GEMM_K, false));
    assert!(!selected(64, 64, H, f, true, QUANT_K, GEMM_K, false));
}

#[test]
fn not_selected_when_the_kill_switch_is_set() {
    // 2026-09-25: `METRALE_FFN_W8A16_ONLY` is injected, not read from the
    // environment: the real accessor is a process-global `OnceLock`, and a test
    // that set the variable would leak into every other test in the binary.
    assert!(!selected(
        PROMPT_TOKENS,
        INTER,
        H,
        WeightQuantFormat::Fp8BlockScaled,
        true,
        QUANT_K,
        GEMM_K,
        true,
    ));
}

#[test]
fn not_selected_when_the_blockscaled_prefill_lever_is_off() {
    // 2026-09-25: `METRALE_FP8_SINGLE_SCALE=1` clears
    // `dispatch.fp8_blockscaled_prefill`, which the attention W8A8 paths read
    // too.
    assert!(!selected(
        PROMPT_TOKENS,
        INTER,
        H,
        WeightQuantFormat::Fp8BlockScaled,
        false,
        QUANT_K,
        GEMM_K,
        false,
    ));
}

#[test]
fn not_selected_when_either_kernel_is_missing() {
    let f = WeightQuantFormat::Fp8BlockScaled;
    assert!(!selected(64, INTER, H, f, true, NO_QUANT, GEMM_K, false));
    assert!(!selected(
        64,
        INTER,
        H,
        f,
        true,
        QUANT_K,
        KernelHandle(0),
        false
    ));
}

struct Harness {
    gpu: MockGpuBackend,
    layer: DenseFfnLayer,
    buffers: BufferArena,
    config: ModelConfig,
    fp8: Fp8Weight,
}

/// 2026-09-25: A config at the real FFN widths with `num_experts` experts (0 is
/// dense), and an arena sized for `max_batch_tokens` rows.
fn harness(num_experts: usize, max_batch_tokens: usize) -> Harness {
    let gpu = MockGpuBackend::new();
    let mut config = ModelConfig::qwen3_next_80b_nvfp4();
    config.hidden_size = H as usize;
    config.intermediate_size = INTER as usize;
    config.num_experts = num_experts;
    config.num_experts_per_tok = num_experts.min(1);
    config.moe_intermediate_size = INTER as usize;
    config.vocab_size = 256;
    let buffers = BufferArena::new(&config, max_batch_tokens, 256, 256, 8, &gpu).unwrap();
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
    layer.per_token_group_quant_fp8_k = QUANT_K;
    layer.fp8_gemm_t_blockscaled_k = GEMM_K;
    let fp8 = Fp8Weight {
        weight: gpu.alloc(1024).unwrap(),
        row_scale: gpu.alloc(1024).unwrap(),
        n: INTER,
        k: H,
        scale_format: WeightQuantFormat::Fp8BlockScaled,
    };
    Harness {
        gpu,
        layer,
        buffers,
        config,
        fp8,
    }
}

/// 2026-09-25: Runs `f` with a prefill-shaped `ForwardContext` over the harness.
fn with_ctx<R>(h: &Harness, dispatch: GemmDispatch, f: impl FnOnce(&ForwardContext) -> R) -> R {
    let derived = DerivedWeights::new();
    let levers = ModelLevers::defaults();
    let stats = ModelStats::new();
    let ctx = ForwardContext {
        buffers: &h.buffers,
        hc_row_offset: 0,
        gpu: &h.gpu,
        config: &h.config,
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
    f(&ctx)
}

/// 2026-09-25: The layer reads the ceiling the compiled target declares, so the
/// test asserts the relationship (selected at the ceiling or at
/// `PROMPT_TOKENS`, declined one row above the ceiling) rather than a value.
/// The per-target values are pinned by `crates/kernels/tests/target_defaults.rs`.
#[test]
fn layer_selects_w8a8_on_a_dense_config() {
    let h = harness(0, 2048);
    let max_m = max_m_for(INTER, H);
    let inside = max_m.min(PROMPT_TOKENS);
    assert!(
        inside > 4,
        "the m > 4 clause must not be what this test is measuring"
    );
    with_ctx(&h, GemmDispatch::defaults(), |ctx| {
        assert!(
            h.layer.prefill_w8a8_selected(ctx, inside, INTER, H, &h.fp8),
            "inside the target's ceiling ({max_m}) the dense config takes W8A8"
        );
        // 2026-09-25: A target with no cap (`u32::MAX`) has no row above it.
        if max_m < u32::MAX {
            assert!(
                !h.layer
                    .prefill_w8a8_selected(ctx, max_m + 1, INTER, H, &h.fp8),
                "one row above the ceiling ({max_m}) must fall back to W8A16"
            );
        }
    });
}

#[test]
fn layer_declines_w8a8_without_the_shared_ffn_scratch() {
    // 2026-09-25: `ffn_act_a` / `ffn_act_scale` are 0 for MoE configs. A null
    // scratch pointer would be a launch writing to address 0, so its absence
    // gates the arm.
    let h = harness(2, 2048);
    assert_eq!(
        h.buffers.ffn_act_a().0,
        0,
        "MoE arena must have no FFN scratch"
    );
    with_ctx(&h, GemmDispatch::defaults(), |ctx| {
        assert!(
            !h.layer
                .prefill_w8a8_selected(ctx, PROMPT_TOKENS, INTER, H, &h.fp8)
        );
    });
}

#[test]
fn cublas_m_pad_rounds_up_to_sixteen() {
    // 2026-09-25: `cublas_fp8_proj_prequant` pads M to a multiple of 16 and
    // writes the padded rows.
    assert_eq!(cublas_fp8_m_pad(PROMPT_TOKENS), 1200);
    assert_eq!(cublas_fp8_m_pad(64), 64);
    assert_eq!(cublas_fp8_m_pad(1), 16);
    for m in [1_u32, 5, 63, 64, 65, 1193, 8192] {
        assert!(cublas_fp8_m_pad(m) >= m);
        assert_eq!(cublas_fp8_m_pad(m) % 16, 0);
    }
}

#[test]
fn ffn_output_buffers_hold_the_padded_m_the_cublas_gemm_writes() {
    // 2026-09-25: The worst case is a prefill chunk as wide as the arena. The
    // arena is sized in `metrale-gpu-runtime` and the writer is in this crate,
    // so the padded extent is checked here.
    for max_batch_tokens in [256_usize, 1193, 2048] {
        let h = harness(0, max_batch_tokens);
        let m = cublas_fp8_m_pad(max_batch_tokens as u32) as usize;
        assert!(
            h.buffers.expert_gate_out_bytes() >= m * INTER as usize * 2,
            "gate/up output too small for padded M at max_batch_tokens={max_batch_tokens}"
        );
        assert!(
            h.buffers.moe_output_bytes() >= m * H as usize * 2,
            "down output too small for padded M at max_batch_tokens={max_batch_tokens}"
        );
    }
}

#[test]
fn activation_scratch_holds_the_widest_ffn_projection() {
    // 2026-09-25: gate/up reduce over K=hidden and down over K=intermediate,
    // and one scratch set serves both, so it is sized for the wider.
    // `ffn_act_scale` holds the `[M, K/128]` scales `per_token_group_quant_fp8`
    // writes; `ffn_act_scale_kmajor` holds the `[K/128, ceil16(M)]` transpose
    // the cuBLASLt arm reads.
    let max_batch_tokens = 1193_usize;
    let h = harness(0, max_batch_tokens);
    let kmax = H.max(INTER) as usize;
    let padded = cublas_fp8_m_pad(max_batch_tokens as u32) as usize;
    assert!(h.buffers.ffn_act_a_bytes() >= padded * kmax);
    assert!(h.buffers.ffn_act_scale_bytes() >= padded * (kmax / 128) * 4);
    assert!(h.buffers.ffn_act_scale_kmajor_bytes() >= padded * (kmax / 128) * 4);
}

#[test]
fn cublas_arm_requires_a_multiple_of_four_weight_scale_column_stride() {
    // 2026-09-25: The weight scales go to cuBLASLt as the checkpoint's
    // `[N/128, K/128]` grid, and `blk128x128_stride_ok` requires `K/128` to be a
    // multiple of 4. Both FFN reduction depths pass.
    use metrale_gpu_runtime::cublaslt::scale_layout::blk128x128_stride_ok;
    assert!(blk128x128_stride_ok(H as usize));
    assert!(blk128x128_stride_ok(INTER as usize));
    assert!(!blk128x128_stride_ok(128 * 3));
    assert!(!blk128x128_stride_ok(128 * 6));
}
