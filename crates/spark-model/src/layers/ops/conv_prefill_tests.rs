// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use spark_runtime::gpu::mock::{MockArg, MockGpuBackend};

fn run(n: u32, taps: u32, tp: u64, commit: u64, stride: u32) -> (bool, MockGpuBackend) {
    let gpu = MockGpuBackend::new();
    let result = conv1d_update_prefill(
        &gpu,
        KernelHandle(1),
        KernelHandle(tp),
        KernelHandle(commit),
        DevicePtr(0x1000),
        DevicePtr(0x2000),
        &DenseWeight {
            weight: DevicePtr(0x3000),
        },
        DevicePtr(0x4000),
        DevicePtr(0x5000),
        33,
        taps,
        n,
        stride,
        41,
        7,
    );
    (result.is_ok(), gpu)
}

#[test]
fn compute_then_commit_preserves_stream_and_input_layout() {
    for n in [4, 7, 8, 9, 54, 64, 65, 127, 128, 129] {
        let (ok, gpu) = run(n, 4, 2, 3, 39);
        assert!(ok);
        let launches = gpu.launches_snapshot();
        let enabled = std::env::var("METRALE_CONV1D_TP").ok().as_deref() != Some("0");
        assert_eq!(launches.len(), if enabled { 2 } else { 1 });
        if !enabled {
            continue;
        }
        assert_eq!(launches[0].func, 2);
        assert_eq!(launches[0].grid, [2, n.div_ceil(64), 1]);
        assert_eq!(launches[0].block, [32, 8, 1]);
        assert_eq!(launches[1].func, 3);
        assert_eq!(launches[1].grid, [1, 1, 1]);
        for l in &launches {
            assert_eq!(l.stream, 7);
        }
        assert_eq!(
            launches[1].args,
            vec![
                MockArg::Buffer(DevicePtr(0x1000)),
                MockArg::Buffer(DevicePtr(0x2000)),
                MockArg::Bytes(33u32.to_ne_bytes().to_vec()),
                MockArg::Bytes(4u32.to_ne_bytes().to_vec()),
                MockArg::Bytes(n.to_ne_bytes().to_vec()),
                MockArg::Bytes(39u32.to_ne_bytes().to_vec()),
            ]
        );
    }
}

#[test]
fn missing_pair_and_short_sequences_use_serial() {
    for (n, tp, commit) in [
        (54, 0, 3),
        (54, 2, 0),
        (54, 0, 0),
        (1, 2, 3),
        (2, 2, 3),
        (3, 2, 3),
    ] {
        let (ok, gpu) = run(n, 4, tp, commit, 39);
        assert!(ok);
        let l = gpu.launches_snapshot();
        assert_eq!(l.len(), 1);
        assert_eq!(l[0].func, 1);
    }
}

#[test]
fn empty_is_noop_and_unsupported_layouts_fail_before_launch() {
    let (ok, gpu) = run(0, 4, 2, 3, 39);
    assert!(ok && gpu.launches_snapshot().is_empty());
    for taps in [0, 1, 2, 3, 5, 8] {
        let (ok, gpu) = run(54, taps, 2, 3, 39);
        assert!(!ok && gpu.launches_snapshot().is_empty());
    }
    let (ok, gpu) = run(54, 4, 2, 3, 32);
    assert!(!ok && gpu.launches_snapshot().is_empty());
}

#[test]
fn zero_channels_and_short_output_stride() {
    for (dim, output_stride, expected_ok) in [(0, 0, true), (33, 32, false)] {
        let gpu = MockGpuBackend::new();
        let result = conv1d_update_prefill(
            &gpu,
            KernelHandle(1),
            KernelHandle(2),
            KernelHandle(3),
            DevicePtr(0x1000),
            DevicePtr(0x2000),
            &DenseWeight {
                weight: DevicePtr(0x3000),
            },
            DevicePtr(0x4000),
            DevicePtr(0x5000),
            dim,
            4,
            54,
            39,
            output_stride,
            7,
        );
        assert_eq!(result.is_ok(), expected_ok);
        assert!(gpu.launches_snapshot().is_empty());
    }
}
