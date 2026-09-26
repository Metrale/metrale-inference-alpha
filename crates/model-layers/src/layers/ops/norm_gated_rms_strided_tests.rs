// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Launch-shape tests for `gated_rms_norm_strided` against the
//! per-sequence `gated_rms_norm` loop it stands in for: one launch at any row
//! count, a `(heads, sequences)` grid, and sequence strides equal to the loop's
//! pointer deltas.
//!
//! Owner: model-layers ops.
//! Invariants: none beyond the types.

use super::*;
use crate::weight_map::DenseWeight;
use metrale_gpu_runtime::gpu::mock::{MockArg, MockGpuBackend};
use metrale_gpu_runtime::gpu::{GpuBackend, KernelHandle};

const NORM_K: u64 = 0xF17E;
const STRIDED_K: u64 = 0xF17F;
/// 2026-09-25: Test shape: 32 heads of 128 per sequence, and a 16384-wide QKVZ
/// row, so the gate stride differs from the value stride.
const NV: u32 = 32;
const VD: u32 = 128;
const QKVZ: u32 = 16384;
const VALUE_DIM: u32 = NV * VD;
/// 2026-09-25: Linear-attention layers of Qwen3.8-27B (48 of the 64 `layer_types`
/// in the unsloth/Qwen3.8-27B-NVFP4 `config.json`): launches per layer times this
/// is launches per step.
const SSM_LAYERS: usize = 48;

struct Bufs {
    gdn_out: metrale_gpu_runtime::gpu::DevicePtr,
    z_base: metrale_gpu_runtime::gpu::DevicePtr,
    normed_out: metrale_gpu_runtime::gpu::DevicePtr,
    weight: DenseWeight,
}

fn bufs(gpu: &MockGpuBackend) -> Bufs {
    Bufs {
        gdn_out: gpu.alloc(16 * VALUE_DIM as usize * 4).unwrap(),
        z_base: gpu.alloc(16 * QKVZ as usize * 2).unwrap(),
        normed_out: gpu.alloc(16 * VALUE_DIM as usize * 2).unwrap(),
        weight: DenseWeight {
            weight: gpu.alloc(VD as usize * 2).unwrap(),
        },
    }
}

/// 2026-09-25: The per-sequence fallback shape: one `gated_rms_norm` launch per
/// sequence at that sequence's base pointers.
fn per_seq_loop(gpu: &MockGpuBackend, b: &Bufs, n: usize) {
    for i in 0..n {
        gated_rms_norm(
            gpu,
            KernelHandle(NORM_K),
            b.gdn_out.offset(i * VALUE_DIM as usize * 4),
            b.z_base.offset(i * QKVZ as usize * 2),
            &b.weight,
            b.normed_out.offset(i * VALUE_DIM as usize * 2),
            NV,
            VD,
            VD,
            1e-6,
            VD,
            0,
        )
        .unwrap();
    }
}

fn strided_once(gpu: &MockGpuBackend, b: &Bufs, n: usize) {
    gated_rms_norm_strided(
        gpu,
        KernelHandle(STRIDED_K),
        b.gdn_out,
        b.z_base,
        &b.weight,
        b.normed_out,
        NV,
        n as u32,
        VD,
        VD,
        1e-6,
        VD,
        VALUE_DIM,
        QKVZ,
        VALUE_DIM,
        0,
    )
    .unwrap();
}

/// 2026-09-25: One launch per layer for every row count tried.
#[test]
fn the_strided_gated_norm_costs_one_launch_per_layer_at_any_row_count() {
    for n in [2usize, 5, 8, 12, 16] {
        let gpu = MockGpuBackend::new();
        let b = bufs(&gpu);
        let before = gpu.launch_count();
        strided_once(&gpu, &b, n);
        let per_layer = gpu.launch_count() - before;
        assert_eq!(per_layer, 1, "n={n}: one launch per layer");
        assert_eq!(
            per_layer * SSM_LAYERS,
            48,
            "n={n}: {SSM_LAYERS} SSM layers => 48 launches per step, not 768"
        );
    }
}

/// 2026-09-25: Control for the test above: the per-sequence loop launches once
/// per sequence.
#[test]
fn the_per_sequence_loop_is_the_768_launch_shape() {
    let gpu = MockGpuBackend::new();
    let b = bufs(&gpu);
    per_seq_loop(&gpu, &b, 16);
    assert_eq!(gpu.launch_count(), 16);
    assert_eq!(gpu.launch_count() * SSM_LAYERS, 768);
}

/// 2026-09-25: The grid is `(heads, sequences)`, so every sequence's rows are
/// normalised; a `(heads, 1)` grid would cover sequence 0 only.
#[test]
fn the_strided_grid_covers_every_sequence_and_head() {
    let gpu = MockGpuBackend::new();
    let b = bufs(&gpu);
    strided_once(&gpu, &b, 16);
    let l = &gpu.launches_snapshot()[0];
    assert_eq!(l.grid, [NV, 16, 1], "grid is (heads_per_seq, num_seqs, 1)");
    assert_eq!(l.block, [VD, 1, 1], "one block per row, VD threads wide");
}

/// 2026-09-25: The three sequence strides equal the pointer deltas of the loop.
/// `gdn_out` and `normed_out` step by `value_dim` (f32 and BF16), and the gate
/// steps by the whole QKVZ row, because z sits inside each sequence's QKVZ row.
/// A wrong gate stride would not crash; it would read another sequence's gates.
#[test]
fn the_strided_arguments_reproduce_the_loops_pointer_deltas() {
    let gpu = MockGpuBackend::new();
    let b = bufs(&gpu);
    per_seq_loop(&gpu, &b, 2);
    let loops = gpu.launches_snapshot();
    let base = |l: &metrale_gpu_runtime::gpu::mock::MockLaunch, i: usize| match &l.args[i] {
        MockArg::Buffer(p) => p.0,
        other => panic!("arg {i} is not a buffer: {other:?}"),
    };
    let d_input = base(&loops[1], 0) - base(&loops[0], 0);
    let d_gate = base(&loops[1], 1) - base(&loops[0], 1);
    let d_output = base(&loops[1], 3) - base(&loops[0], 3);
    assert_eq!(d_input, VALUE_DIM as u64 * 4, "gdn_out rows are f32");
    assert_eq!(d_gate, QKVZ as u64 * 2, "z rows are a whole QKVZ block");
    assert_eq!(d_output, VALUE_DIM as u64 * 2, "normed_out rows are bf16");

    let gpu2 = MockGpuBackend::new();
    let b2 = bufs(&gpu2);
    strided_once(&gpu2, &b2, 2);
    let l = &gpu2.launches_snapshot()[0];
    let u32_arg = |v: u32| MockArg::Bytes(v.to_ne_bytes().to_vec());
    // 2026-09-25: Bases are sequence 0's, and each stride counts elements of its
    // own buffer's type.
    assert_eq!(l.args[0], MockArg::Buffer(b2.gdn_out));
    assert_eq!(l.args[1], MockArg::Buffer(b2.z_base));
    assert_eq!(l.args[3], MockArg::Buffer(b2.normed_out));
    assert_eq!(
        l.args[8],
        u32_arg(d_input as u32 / 4),
        "input_seq_stride, f32"
    );
    assert_eq!(
        l.args[9],
        u32_arg(d_gate as u32 / 2),
        "gate_seq_stride, bf16"
    );
    assert_eq!(
        l.args[10],
        u32_arg(d_output as u32 / 2),
        "output_seq_stride, bf16"
    );
    // 2026-09-25: The per-head arguments equal the per-sequence launch's.
    assert_eq!(l.args[4], u32_arg(VD), "hidden_size");
    assert_eq!(l.args[6], u32_arg(VD), "gate_stride between HEADS");
}
