// SPDX-License-Identifier: AGPL-3.0-only
//! NoPE skips only rotation; positive-dimension launches preserve Q/K layout.

use super::*;
use spark_runtime::gpu::mock::{MockArg, MockGpuBackend};

fn call(gpu: &MockGpuBackend, rotary: u32, strided: bool) {
    let (q, k, positions) = (DevicePtr(0x1000), DevicePtr(0x2000), DevicePtr(0x3000));
    if strided {
        rope_strided(
            gpu,
            KernelHandle(17),
            q,
            k,
            positions,
            3,
            32,
            2,
            128,
            rotary,
            10000.0,
            4352,
            4352,
            7,
        )
        .unwrap();
    } else {
        rope(
            gpu,
            KernelHandle(17),
            q,
            k,
            positions,
            3,
            32,
            2,
            128,
            rotary,
            10000.0,
            7,
        )
        .unwrap();
    }
}

#[test]
fn zero_rotary_dimension_never_launches_packed() {
    let gpu = MockGpuBackend::new();
    call(&gpu, 0, false);
    assert!(gpu.launches_snapshot().is_empty());
}

#[test]
fn zero_rotary_dimension_never_launches_strided() {
    let gpu = MockGpuBackend::new();
    call(&gpu, 0, true);
    assert!(gpu.launches_snapshot().is_empty());
}

#[test]
fn positive_rotary_dimension_keeps_launch_and_pointer_shapes() {
    for rotary in [64, 128] {
        let gpu = MockGpuBackend::new();
        call(&gpu, rotary, false);
        call(&gpu, rotary, true);
        let launches = gpu.launches_snapshot();
        assert_eq!(launches.len(), 2);
        for (index, launch) in launches.iter().enumerate() {
            assert_eq!(launch.func, 17);
            assert_eq!(launch.grid, [34, 3u32.div_ceil(128 / (rotary / 2)), 1]);
            assert_eq!(launch.block, [128, 1, 1]);
            assert_eq!(launch.stream, 7);
            assert_eq!(launch.args[0], MockArg::Buffer(DevicePtr(0x1000)));
            assert_eq!(launch.args[1], MockArg::Buffer(DevicePtr(0x2000)));
            assert_eq!(launch.args[2], MockArg::Buffer(DevicePtr(0x3000)));
            let dims = [3u32, 32, 2, 128, rotary];
            for (arg, expected) in launch.args[3..8].iter().zip(dims) {
                assert_eq!(*arg, MockArg::Bytes(expected.to_ne_bytes().to_vec()));
            }
            assert_eq!(launch.args.len(), if index == 0 { 9 } else { 11 });
            if index == 1 {
                for arg in &launch.args[9..] {
                    assert_eq!(*arg, MockArg::Bytes(4352u32.to_ne_bytes().to_vec()));
                }
            }
        }
    }
}
