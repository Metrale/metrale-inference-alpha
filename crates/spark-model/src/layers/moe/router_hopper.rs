// SPDX-License-Identifier: AGPL-3.0-only
//! Hopper M128 router: exact BF16 logits, strict separate FP32 multiply/add.

use super::*;
use spark_runtime::kernel_args::KernelLaunch;

pub(super) fn kernel(gpu: &dyn GpuBackend) -> KernelHandle {
    crate::layers::try_target_kernel(
        gpu,
        "dense_gemm_router_hopper",
        "dense_gemm_router_hopper_8x32",
    )
}

impl MoeLayer {
    /// Caller checks the measured M128/N256/K2048 prefill shape and kernel handle.
    pub(super) fn router_gate_gemm_hopper(
        &self,
        input: DevicePtr,
        output: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        KernelLaunch::new(ctx.gpu, self.dense_gemm_router_hopper)
            .grid([8, 16, 1])
            .block([16, 8, 1])
            .arg_ptr(input)
            .arg_ptr(self.weights.gate.weight)
            .arg_ptr(output)
            .arg_u32(128)
            .arg_u32(256)
            .arg_u32(2048)
            .launch(stream)
    }
}
