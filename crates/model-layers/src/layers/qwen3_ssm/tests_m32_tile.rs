// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Mock-GPU tests of the tile choice for the block-scaled FP8 `in_proj_qkvz`
//! and `out_proj` on the batched GDN verify: the 32-row M-tile twin
//! `w8a16_gemm_pipelined_m32` (nine arguments) or the 128-row `w8a16_gemm_pipelined`
//! (seven).
//!
//! Owner: model-layers (Qwen3 SSM layer).
//! Invariants: none beyond the types.
//!
//! A child of `tests.rs`, using its native-FP8 layer and verify helpers. The GPU oracle
//! `native_fp8_gdn_proj_m32_microtest` (metrale-model-arch example) compares the twin's
//! output byte for byte with the 128-row tile.

use super::{native_fp8_gdn_layer, run_batched_verify, scalar_u32};
use crate::weight_map::Fp8Weight;
use metrale_config::ModelConfig;
use metrale_gpu_runtime::gpu::KernelHandle;
use metrale_gpu_runtime::gpu::mock::{MockArg, MockGpuBackend, MockLaunch};

const QKVZ_N: u32 = 12_288;
const H: u32 = 2_048;
const OUT_N: u32 = 2_048;
const VALUE_DIM: u32 = 4_096;

/// 2026-09-25: Every launch that used `weight`'s block-scaled pair at `m` rows. The
/// argument count tells the two tiles apart, because the mock gives every kernel the
/// same handle.
fn projection_launches(gpu: &MockGpuBackend, weight: &Fp8Weight, m: u32) -> Vec<MockLaunch> {
    gpu.launches_snapshot()
        .into_iter()
        .filter(|l| {
            l.args.len() >= 7
                && l.args[1] == MockArg::Buffer(weight.weight)
                && l.args[2] == MockArg::Buffer(weight.row_scale)
                && scalar_u32(&l.args[4]) == Some(m)
        })
        .collect()
}

/// 2026-09-25: Assert a launch of the 32-row twin: grid `(ceil(N/32), ceil(M/32))`, and
/// nine arguments ending in the contiguous pitches `lda = K`, `ldc = N`.
fn assert_m32_tile(l: &MockLaunch, m: u32, n: u32, k: u32) {
    assert_eq!(l.args.len(), 9, "the twin takes lda/ldc");
    assert_eq!(l.grid, [n.div_ceil(32), m.div_ceil(32), 1]);
    assert_eq!(l.block, [256, 1, 1]);
    assert_eq!(scalar_u32(&l.args[5]), Some(n));
    assert_eq!(scalar_u32(&l.args[6]), Some(k));
    assert_eq!(scalar_u32(&l.args[7]), Some(k), "lda = K (contiguous)");
    assert_eq!(scalar_u32(&l.args[8]), Some(n), "ldc = N (contiguous)");
}

/// 2026-09-25: R = 8 × 4 = 32 rows: one 32-row twin launch per projection and no other
/// launch at 32 rows on either weight.
#[test]
fn native_fp8_gdn_batched_verify_r32_takes_the_m32_tile() {
    let config = ModelConfig::qwen3_next_80b_nvfp4();
    let gpu = MockGpuBackend::new();
    let layer = native_fp8_gdn_layer(&gpu, &config, true, true);
    run_batched_verify(&gpu, &config, &layer, &[4; 8]).unwrap();
    let qkvz = layer.qkvz_fp8w.as_ref().unwrap();
    let out = layer.out_proj_fp8w.as_ref().unwrap();

    let q = projection_launches(&gpu, qkvz, 32);
    assert_eq!(q.len(), 1, "one QKVZ launch at R=32: {q:?}");
    assert_m32_tile(&q[0], 32, QKVZ_N, H);
    let o = projection_launches(&gpu, out, 32);
    assert_eq!(o.len(), 1, "one out_proj launch at R=32: {o:?}");
    assert_m32_tile(&o[0], 32, OUT_N, VALUE_DIM);
}

/// 2026-09-25: R = 16 runs `w8a16_gemv_batch16` (grid `ceil(N/4)`, seven arguments), which
/// comes before the tile arm for 5..=16 rows.
#[test]
fn native_fp8_gdn_batched_verify_r16_stays_on_batch16() {
    let config = ModelConfig::qwen3_next_80b_nvfp4();
    let gpu = MockGpuBackend::new();
    let layer = native_fp8_gdn_layer(&gpu, &config, true, true);
    run_batched_verify(&gpu, &config, &layer, &[4; 4]).unwrap();
    let qkvz = layer.qkvz_fp8w.as_ref().unwrap();
    let out = layer.out_proj_fp8w.as_ref().unwrap();
    for (w, n) in [(qkvz, QKVZ_N), (out, OUT_N)] {
        let l = projection_launches(&gpu, w, 16);
        assert_eq!(l.len(), 1);
        assert_eq!(l[0].args.len(), 7, "batch16 GEMV, not the twin");
        assert_eq!(l[0].grid, [n.div_ceil(4), 1, 1]);
    }
}

/// 2026-09-25: Without `w8a16_gemv_batch16`, R = 7 reaches the tile arm and runs the
/// 32-row twin (one row tile). With neither kernel it runs the 128-row tile, tested in
/// `tests.rs`.
#[test]
fn native_fp8_gdn_batched_verify_r7_without_batch16_takes_the_m32_tile() {
    let config = ModelConfig::qwen3_next_80b_nvfp4();
    let gpu = MockGpuBackend::new();
    let mut layer = native_fp8_gdn_layer(&gpu, &config, true, true);
    layer.w8a16_gemv_batch16_k = KernelHandle(0);
    run_batched_verify(&gpu, &config, &layer, &[4, 3]).unwrap();
    let qkvz = layer.qkvz_fp8w.as_ref().unwrap();
    let out = layer.out_proj_fp8w.as_ref().unwrap();
    let q = projection_launches(&gpu, qkvz, 7);
    assert_eq!(q.len(), 1);
    assert_m32_tile(&q[0], 7, QKVZ_N, H);
    let o = projection_launches(&gpu, out, 7);
    assert_eq!(o.len(), 1);
    assert_m32_tile(&o[0], 7, OUT_N, VALUE_DIM);
}

/// 2026-09-25: Without `w8a16_gemm_pipelined_m32`, R = 32 runs the 128-row tile
/// (`grid.y = 1`, seven arguments) rather than the zero handle.
#[test]
fn native_fp8_gdn_batched_verify_r32_without_the_twin_keeps_the_128_tile() {
    let config = ModelConfig::qwen3_next_80b_nvfp4();
    let gpu = MockGpuBackend::new();
    let mut layer = native_fp8_gdn_layer(&gpu, &config, true, true);
    layer.w8a16_gemm_pipelined_m32_k = KernelHandle(0);
    run_batched_verify(&gpu, &config, &layer, &[4; 8]).unwrap();
    let qkvz = layer.qkvz_fp8w.as_ref().unwrap();
    let out = layer.out_proj_fp8w.as_ref().unwrap();
    for (w, n) in [(qkvz, QKVZ_N), (out, OUT_N)] {
        let l = projection_launches(&gpu, w, 32);
        assert_eq!(l.len(), 1);
        assert_eq!(l[0].args.len(), 7, "the 128-tile kernel takes no pitches");
        assert_eq!(l[0].grid, [n.div_ceil(32), 1, 1]);
    }
}
