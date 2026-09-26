// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Dispatch tests for the multi-sequence FP8 O projection: for each
//! row count and kernel set, which kernel runs, how many launches, and each
//! launch's row offsets, recorded by the mock backend.
//!
//! Owner: model-layers (qwen3 attention).
//! Invariants: none beyond the types.

use super::super::super::ctx::MultiSeqCtx;
use crate::layer::{ForwardContext, MoeLoraRoute};
use crate::layers::ops::{DerivedWeights, GemmDispatch, ModelLevers, ModelStats};
use crate::layers::qwen3_attention::attn_ncol_gemv::NcolWidth;
use crate::layers::{FfnComponent, qwen3_attention::Qwen3AttentionLayer};
use crate::weight_map::{
    AttentionWeights, DenseWeight, Fp8Weight, QuantWeight, QuantizedWeight, WeightQuantFormat,
};
use metrale_cache::kv_cache::KvCacheDtype;
use metrale_config::ModelConfig;
use metrale_gpu_runtime::buffers::BufferArena;
use metrale_gpu_runtime::gpu::mock::{MockArg, MockGpuBackend};
use metrale_gpu_runtime::gpu::{GpuBackend, KernelHandle};

/// 2026-09-25: Which tier the FP8 o_proj is expected to take, and the row stride
/// the group loop must walk with. `Scalar` is one `w8a16_gemv` launch per row.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Tier {
    Scalar,
    Batch4,
    Batch16,
    /// 2026-09-25: The N-column-blocked rung: same 16-row group as `Batch16`, so
    /// only the kernel differs.
    Ncol2,
    Ncol4,
    /// 2026-09-25: The tensor-core rung (`w8a16_gemm_m16`): same 16-row group as
    /// `Batch16`.
    M16Tc,
    /// 2026-09-25: The 32-row M-tile kernel: one launch over all rows, `grid.y`
    /// tiling M inside the kernel, so the group is the whole batch.
    M32Tile,
}

impl Tier {
    /// 2026-09-25: Rows per launch of this tier; the group loop issues
    /// `ceil(rows / step)` launches.
    fn step(self, rows: usize) -> usize {
        match self {
            Tier::Scalar => 1,
            Tier::Batch4 => 4,
            Tier::Batch16 | Tier::Ncol2 | Tier::Ncol4 | Tier::M16Tc => 16,
            Tier::M32Tile => rows,
        }
    }

    fn kernel(self) -> u64 {
        match self {
            Tier::Scalar => SCALAR_K,
            Tier::Batch4 => BATCH4_K,
            Tier::Batch16 => BATCH16_K,
            Tier::Ncol2 => NCOL2_K,
            Tier::Ncol4 => NCOL4_K,
            Tier::M16Tc => M16TC_K,
            Tier::M32Tile => M32_K,
        }
    }
}

const SCALAR_K: u64 = 0xF081;
const BATCH4_K: u64 = 0xF084;
const BATCH16_K: u64 = 0xF08C;
const NCOL2_K: u64 = 0xF0C2;
const NCOL4_K: u64 = 0xF0C4;
const M16TC_K: u64 = 0xF08E;
const M32_K: u64 = 0xF032;

#[test]
fn native_fp8_attention_o_projection_batches_four_real_rows() {
    check_dispatch(
        4,
        128,
        true,
        WeightQuantFormat::Fp8BlockScaled,
        Tier::Batch4,
    );
}

/// 2026-09-25: 5..=16 rows take one `w8a16_gemv_batch16` launch, one pass over
/// the weight.
#[test]
fn native_fp8_attention_o_projection_batches_up_to_sixteen_rows_in_one_pass() {
    for rows in [5, 8, 12, 16] {
        check_dispatch(
            rows,
            128,
            true,
            WeightQuantFormat::Fp8BlockScaled,
            Tier::Batch16,
        );
    }
}

/// 2026-09-25: Above 16 rows the 32-row M-tile kernel takes the whole batch in
/// one launch, n = 64 included (the kernel tiles M).
#[test]
fn native_fp8_attention_o_projection_takes_the_m32_tile_above_sixteen_rows() {
    for rows in [17, 20, 32, 64] {
        check_dispatch(
            rows,
            128,
            true,
            WeightQuantFormat::Fp8BlockScaled,
            Tier::M32Tile,
        );
    }
}

/// 2026-09-25: Negative control: without `w8a16_gemm_pipelined_m32`, rows above
/// 16 walk batch16 in 16-row groups and stay on the batched tier.
#[test]
fn native_fp8_attention_o_projection_without_the_m32_tile_keeps_sixteen_row_groups() {
    for rows in [20, 32] {
        check_dispatch_with(
            rows,
            128,
            true,
            true,
            WeightQuantFormat::Fp8BlockScaled,
            Tier::Batch16,
            None,
            None,
            false,
        );
    }
}

/// 2026-09-25: The M32 tile starts above 16 rows: 16 rows stay on batch16.
#[test]
fn native_fp8_attention_o_projection_m32_tile_leaves_sixteen_rows_on_batch16() {
    check_dispatch(
        16,
        128,
        true,
        WeightQuantFormat::Fp8BlockScaled,
        Tier::Batch16,
    );
}

fn check_dispatch(
    rows: usize,
    width: usize,
    available: bool,
    format: WeightQuantFormat,
    tier: Tier,
) {
    check_dispatch_with(
        rows, width, available, available, format, tier, None, None, true,
    )
}

/// 2026-09-25: `check_dispatch` with the N-column tier on, set through the layer
/// field `attn_ncol` that the lever resolves to at construction.
fn check_dispatch_ncol(rows: usize, tier: Tier, ncol: NcolWidth) {
    check_dispatch_with(
        rows,
        128,
        true,
        true,
        WeightQuantFormat::Fp8BlockScaled,
        tier,
        Some(ncol),
        None,
        true,
    )
}

/// 2026-09-25: `check_dispatch` with the tensor-core tier on, set through the
/// layer field `m16_tc`. `handle` is whether the kernel set carries
/// `w8a16_gemm_m16`, the other half of the tier's predicate.
fn check_dispatch_m16_tc(rows: usize, tier: Tier, handle: bool) {
    check_dispatch_with(
        rows,
        128,
        true,
        true,
        WeightQuantFormat::Fp8BlockScaled,
        tier,
        None,
        Some(handle),
        true,
    )
}

/// 2026-09-25: `available` is the presence of the batch4 handle and `wide` of
/// the batch16 handle, so a kernel set without batch16 is reachable. `m16_tc` is
/// `None` with the tensor-core lever off and `Some(handle)` with it on, where
/// `handle` is the presence of `w8a16_gemm_m16`. `m32` is the presence of
/// `w8a16_gemm_pipelined_m32`, set explicitly so the test does not depend on
/// `ModelLevers::fp8_attn_m32`, which decides that handle at construction.
#[allow(clippy::too_many_arguments)]
fn check_dispatch_with(
    rows: usize,
    width: usize,
    available: bool,
    wide: bool,
    format: WeightQuantFormat,
    tier: Tier,
    ncol: Option<NcolWidth>,
    m16_tc: Option<bool>,
    m32: bool,
) {
    let gpu = MockGpuBackend::new();
    let mut config = ModelConfig::qwen3_next_80b_nvfp4();
    config.hidden_size = width;
    config.intermediate_size = 128;
    config.num_attention_heads = 1;
    config.num_key_value_heads = 1;
    config.head_dim = width;
    config.num_experts = 1;
    config.num_experts_per_tok = 1;
    config.moe_intermediate_size = 128;
    config.vocab_size = 128;
    let buffers = BufferArena::new(&config, 8, 16, 16, 8, &gpu).unwrap();
    let dense = DenseWeight {
        weight: gpu.alloc(128 * 128 * 2).unwrap(),
    };
    let fallback = QuantizedWeight::null();
    let attn = AttentionWeights {
        q_proj: dense,
        k_proj: dense,
        v_proj: dense,
        o_proj: fallback,
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
        &gpu,
        KvCacheDtype::Bf16,
        0,
        &config,
    )
    .unwrap();
    layer.w8a16_gemv_k = KernelHandle(SCALAR_K);
    layer.w8a16_gemv_batch4_k = KernelHandle(if available { BATCH4_K } else { 0 });
    layer.w8a16_gemv_batch16_k = KernelHandle(if wide { BATCH16_K } else { 0 });
    layer.w8a16_gemv_ncol2_k = KernelHandle(NCOL2_K);
    layer.w8a16_gemv_ncol4_k = KernelHandle(NCOL4_K);
    layer.attn_ncol = ncol;
    layer.m16_tc = m16_tc.is_some();
    layer.w8a16_gemm_m16_k = KernelHandle(if m16_tc == Some(true) { M16TC_K } else { 0 });
    layer.w8a16_gemm_pipelined_m32_k = KernelHandle(if m32 { M32_K } else { 0 });
    let fp8 = Fp8Weight {
        weight: gpu.alloc(128 * 128).unwrap(),
        row_scale: gpu.alloc(4).unwrap(),
        n: 128,
        k: 128,
        scale_format: format,
    };
    layer.o_weight = Some(QuantWeight::Fp8(fp8));
    let dispatch = GemmDispatch::defaults();
    let derived = DerivedWeights::new();
    let levers = ModelLevers::defaults();
    let stats = ModelStats::new();
    let fwd = ForwardContext {
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
        decode_step: false,
        gdn_exact_replay: false,
        gdn_write_on_accept: false,
        token_ids: None,
        host_token_ids: None,
        routed_lora_layers: None,
        midchunk_capture: None,
        moe_lora_route: MoeLoraRoute::Fold,
    };
    let c = MultiSeqCtx::new(
        &layer,
        &fwd,
        buffers.hidden_states(),
        buffers.residual(),
        rows,
        16,
        0,
    );
    let first = gpu.launch_count();
    let allocations = gpu.alloc_count();
    let output = layer.ms_phase_o_proj(&c, buffers.attn_output()).unwrap();
    let all = gpu.launches_snapshot();
    let launches: Vec<_> = all[first..]
        .iter()
        .filter(|l| l.args.contains(&MockArg::Buffer(fp8.weight)))
        .collect();
    let step = tier.step(rows);
    assert_eq!(
        launches.len(),
        rows.div_ceil(step),
        "production O-projection dispatch (tier {tier:?}, rows {rows})"
    );
    assert_eq!(
        gpu.alloc_count(),
        allocations,
        "projection must reuse existing buffers"
    );
    for (group, launch) in launches.iter().enumerate() {
        let row = group * step;
        assert_eq!(launch.func, tier.kernel());
        assert_eq!(
            launch.args[0],
            MockArg::Buffer(buffers.attn_output().offset(row * width * 2))
        );
        assert_eq!(launch.args[1], MockArg::Buffer(fp8.weight));
        assert_eq!(launch.args[2], MockArg::Buffer(fp8.row_scale));
        assert_eq!(
            launch.args[3],
            MockArg::Buffer(output.offset(row * width * 2))
        );
        if tier != Tier::Scalar {
            assert_eq!(
                launch.args[4],
                MockArg::Bytes(((rows - row).min(step) as u32).to_ne_bytes().to_vec())
            );
        }
    }
}

#[test]
fn native_fp8_attention_o_projection_chunks_preserve_offsets() {
    check_dispatch(
        2,
        128,
        true,
        WeightQuantFormat::Fp8BlockScaled,
        Tier::Batch4,
    );
    for rows in [5, 16] {
        check_dispatch(
            rows,
            128,
            true,
            WeightQuantFormat::Fp8BlockScaled,
            Tier::Batch16,
        );
    }
}

#[test]
fn native_fp8_attention_o_projection_retains_scalar_fallbacks() {
    let bs = WeightQuantFormat::Fp8BlockScaled;
    check_dispatch(1, 128, true, bs, Tier::Scalar);
    check_dispatch(4, 128, false, bs, Tier::Scalar);
    check_dispatch(4, 64, true, bs, Tier::Scalar);
    check_dispatch(4, 128, true, WeightQuantFormat::Fp8PerRow, Tier::Scalar);
    // 2026-09-25: Per-row scales and unaligned dims disqualify the batch16 tier
    // too: every batched rung sits behind the one `block_scaled` guard.
    check_dispatch(8, 64, true, bs, Tier::Scalar);
    check_dispatch(8, 128, true, WeightQuantFormat::Fp8PerRow, Tier::Scalar);
}

/// 2026-09-25: A kernel set without batch16 walks batch4 in 4-row groups and
/// stays on the batched tier.
#[test]
fn native_fp8_attention_o_projection_without_batch16_keeps_four_row_groups() {
    for rows in [8, 16] {
        check_dispatch_with(
            rows,
            128,
            true,
            false,
            WeightQuantFormat::Fp8BlockScaled,
            Tier::Batch4,
            None,
            None,
            true,
        );
    }
}

/// 2026-09-25: The N-column tier serves 5..=16 rows in the same single 16-row
/// group as batch16: the kernel changes, the launch count does not.
#[test]
fn native_fp8_attention_o_projection_takes_the_ncol_tier() {
    for rows in [5, 8, 12, 16] {
        check_dispatch_ncol(rows, Tier::Ncol2, NcolWidth::Two);
        check_dispatch_ncol(rows, Tier::Ncol4, NcolWidth::Four);
    }
}

/// 2026-09-25: `ncol_plan` claims only 5..=16 rows, so 1 row stays scalar and
/// 2..=4 rows stay on batch4.
#[test]
fn native_fp8_attention_o_projection_ncol_leaves_small_batches_alone() {
    for rows in [1, 2, 4] {
        let tier = if rows == 1 {
            Tier::Scalar
        } else {
            Tier::Batch4
        };
        check_dispatch_ncol(rows, tier, NcolWidth::Two);
    }
}

/// 2026-09-25: Above 16 rows the N-column tier declines: the M32 tile takes
/// the batch, and without it the loop walks batch16 in 16-row groups.
#[test]
fn native_fp8_attention_o_projection_ncol_declines_above_max_m() {
    check_dispatch_ncol(20, Tier::M32Tile, NcolWidth::Two);
    check_dispatch_with(
        20,
        128,
        true,
        true,
        WeightQuantFormat::Fp8BlockScaled,
        Tier::Batch16,
        Some(NcolWidth::Two),
        None,
        false,
    );
}

/// 2026-09-25: With the attention tensor-core lever on, 5..=16 rows take
/// `w8a16_gemm_m16` in the same 16-row group.
#[test]
fn native_fp8_o_projection_attn_m16_tc_takes_the_sixteen_row_group() {
    for rows in [5, 8, 12, 16] {
        check_dispatch_m16_tc(rows, Tier::M16Tc, true);
    }
}

/// 2026-09-25: Above 16 rows the M32 tile wins over the tensor-core lever;
/// without the M32 tile, `w8a16_gemm_m16` serves the rows in 16-row groups.
#[test]
fn native_fp8_o_projection_attn_m16_tc_yields_to_the_m32_tile_above_max_m() {
    check_dispatch_m16_tc(20, Tier::M32Tile, true);
    check_dispatch_with(
        20,
        128,
        true,
        true,
        WeightQuantFormat::Fp8BlockScaled,
        Tier::M16Tc,
        None,
        Some(true),
        false,
    );
}

/// 2026-09-25: 2..=4 rows keep `w8a16_gemv_batch4` with the lever on: the tier
/// requires `wide` (n > 4).
#[test]
fn native_fp8_o_projection_attn_m16_tc_leaves_small_batches_on_batch4() {
    check_dispatch_m16_tc(4, Tier::Batch4, true);
}

/// 2026-09-25: A kernel set without `w8a16_gemm_m16` stays on batch16 with the
/// lever on, rather than launching a zero handle.
#[test]
fn native_fp8_o_projection_attn_m16_tc_declines_without_its_entry_point() {
    check_dispatch_m16_tc(16, Tier::Batch16, false);
}

/// 2026-09-25: With the lever off, 16 rows stay on batch16.
#[test]
fn native_fp8_o_projection_without_the_attn_lever_stays_on_batch16() {
    check_dispatch(
        16,
        128,
        true,
        WeightQuantFormat::Fp8BlockScaled,
        Tier::Batch16,
    );
}
