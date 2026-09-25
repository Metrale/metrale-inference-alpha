// SPDX-License-Identifier: AGPL-3.0-only
//! Preserve FP32 gate accumulators through Nano sigmoid routing. The pinned
//! checkpoint's F32 router weights are all BF16-exact (7,913,472 values checked);
//! this diagnostic does not generalize arbitrary F32 weight loading.

use super::*;
use spark_runtime::kernel_args::KernelLaunch;

pub(super) struct Fp32Router {
    gemv: KernelHandle,
    gemm: KernelHandle,
    topk: KernelHandle,
}

impl Fp32Router {
    pub(super) fn load(gpu: &dyn GpuBackend, config: &ModelConfig) -> Result<Option<Self>> {
        if !super::router_dispatch::eligible(
            &config.model_type,
            config.hidden_size,
            config.num_experts,
        ) || !gpu.has_module("nemotron_router_f32")
        {
            return Ok(None);
        }
        let result = Self {
            gemv: gpu.kernel("gemv", "dense_gemv_bf16_fp32out")?,
            gemm: gpu.kernel("gemm", "dense_gemm_bf16_f32out")?,
            topk: gpu.kernel("nemotron_router_f32", "nemotron_router_topk_f32")?,
        };
        static LOG: std::sync::Once = std::sync::Once::new();
        LOG.call_once(|| {
            tracing::info!("Nemotron Nano router: FP32 logits through sigmoid/top-k selected")
        });
        Ok(Some(result))
    }
}

impl NemotronMoeLayer {
    pub(super) fn router_logits(
        &self,
        input: DevicePtr,
        n: u32,
        decode: bool,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        let e = ctx.config.num_experts as u32;
        let h = ctx.config.hidden_size as u32;
        if let Some(router) = &self.fp32_router {
            let out = ctx.buffers.gate_logits_f32();
            if decode {
                ops::dense_gemv(
                    ctx.gpu,
                    router.gemv,
                    input,
                    &self.weights.gate,
                    out,
                    e,
                    h,
                    stream,
                )?;
            } else {
                ops::dense_gemm(
                    ctx.gpu,
                    router.gemm,
                    input,
                    &self.weights.gate,
                    out,
                    n,
                    e,
                    h,
                    stream,
                )?;
            }
            return Ok(out);
        }
        let out = ctx.buffers.gate_logits();
        if decode {
            ops::dense_gemv(
                ctx.gpu,
                self.dense_gemv_k,
                input,
                &self.weights.gate,
                out,
                e,
                h,
                stream,
            )?;
        } else {
            self.dense_gemm_prefill(ctx.gpu, input, &self.weights.gate, out, n, e, h, stream)?;
        }
        Ok(out)
    }

    pub(super) fn router_topk(
        &self,
        logits: DevicePtr,
        indices: DevicePtr,
        weights: DevicePtr,
        n: u32,
        batched: bool,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let e = ctx.config.num_experts as u32;
        let k = self.top_k as u32;
        let normalize = ctx.config.norm_topk_prob;
        let scale = ctx.config.routed_scaling_factor as f32;
        if self.fp32_router.is_some() || batched {
            let kernel = self
                .fp32_router
                .as_ref()
                .map_or(self.topk_sigmoid_batched_k, |r| r.topk);
            return launch_topk(
                ctx.gpu,
                kernel,
                logits,
                self.weights.e_score_correction_bias.weight,
                indices,
                weights,
                e,
                k,
                normalize,
                scale,
                n,
                stream,
            );
        }
        ops::moe_topk_sigmoid(
            ctx.gpu,
            self.topk_sigmoid_k,
            logits,
            self.weights.e_score_correction_bias.weight,
            indices,
            weights,
            e,
            k,
            normalize,
            scale,
            stream,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn launch_topk(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    logits: DevicePtr,
    bias: DevicePtr,
    indices: DevicePtr,
    weights: DevicePtr,
    experts: u32,
    top_k: u32,
    normalize: bool,
    scale: f32,
    n: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([1, n, 1])
        .block([256, 1, 1])
        .arg_ptr(logits)
        .arg_ptr(bias)
        .arg_ptr(indices)
        .arg_ptr(weights)
        .arg_u32(experts)
        .arg_u32(top_k)
        .arg_u32(u32::from(normalize))
        .arg_f32(scale)
        .arg_u32(n)
        .launch(stream)
}

#[cfg(test)]
#[path = "router_tests.rs"]
mod tests;
