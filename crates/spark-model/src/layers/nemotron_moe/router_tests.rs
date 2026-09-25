// SPDX-License-Identifier: AGPL-3.0-only
use super::*;
use spark_runtime::gpu::mock::{MockArg, MockGpuBackend};

#[test]
fn single_and_batched_routing_preserve_arguments() {
    for n in [1, 3, 54] {
        for normalize in [false, true] {
            let gpu = MockGpuBackend::new();
            launch_topk(
                &gpu,
                KernelHandle(17),
                DevicePtr(100),
                DevicePtr(200),
                DevicePtr(300),
                DevicePtr(400),
                128,
                6,
                normalize,
                2.5,
                n,
                9,
            )
            .unwrap();
            let calls = gpu.launches_snapshot();
            assert_eq!(calls.len(), 1);
            let call = &calls[0];
            assert_eq!(call.func, 17);
            assert_eq!(call.grid, [1, n, 1]);
            assert_eq!(call.block, [256, 1, 1]);
            assert_eq!(call.stream, 9);
            assert_eq!(call.args.len(), 9);
            for (arg, ptr) in call.args[..4].iter().zip([100, 200, 300, 400]) {
                assert_eq!(*arg, MockArg::Buffer(DevicePtr(ptr)));
            }
            for (arg, bits) in
                call.args[4..]
                    .iter()
                    .zip([128, 6, u32::from(normalize), 2.5f32.to_bits(), n])
            {
                assert_eq!(*arg, MockArg::Bytes(bits.to_ne_bytes().to_vec()));
            }
        }
    }
}
