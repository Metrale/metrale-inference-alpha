// SPDX-License-Identifier: AGPL-3.0-only
//! Optional Hopper tile for short shared-expert FP8 projections.

use super::*;
use spark_runtime::kernel_args::KernelLaunch;

pub(super) fn kernel(gpu: &dyn GpuBackend) -> KernelHandle {
    crate::layers::try_target_kernel(gpu, "shared_fp8_m16n32", "shared_fp8_m16n32")
}

impl MoeLayer {
    /// Reuses the existing FP8 bytes, scales and output buffers without changing
    /// reduction order. Unsupported shapes and targets retain the original path.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn shared_fp8_projection(
        &self,
        ctx: &ForwardContext,
        input: DevicePtr,
        input_scale: DevicePtr,
        weight: DevicePtr,
        weight_scale: DevicePtr,
        output: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<()> {
        if !shared_fp8_hopper_shape::eligible(
            m,
            n,
            k,
            ctx.decode_step,
            self.shared_fp8_hopper_k.0 != 0,
            ctx.config.num_experts,
            ctx.config.num_experts_per_tok,
        ) {
            return ops::fp8_gemm_t_blockscaled(
                ctx.gpu,
                self.fp8_gemm_t_blockscaled_k,
                input,
                input_scale,
                weight,
                weight_scale,
                output,
                m,
                n,
                k,
                stream,
            );
        }
        KernelLaunch::new(ctx.gpu, self.shared_fp8_hopper_k)
            .grid([n.div_ceil(32), m.div_ceil(16), 1])
            .block([32, 1, 1])
            .arg_ptr(input)
            .arg_ptr(input_scale)
            .arg_ptr(weight)
            .arg_ptr(weight_scale)
            .arg_ptr(output)
            .arg_u32(m)
            .arg_u32(n)
            .arg_u32(k)
            .launch(stream)
    }
}
