// SPDX-License-Identifier: AGPL-3.0-only

//! Optional Hopper E256/K8 single-token top-k. The CUDA oracle compares exact
//! indices and FP32 weights, including ties/non-finite inputs and both normalize
//! settings. The warp kernel preserves the original softmax reduction tree.

use super::*;
use spark_runtime::kernel_args::KernelLaunch;

fn warp_eligible(
    experts: usize,
    topk: usize,
    single_decode: bool,
    native_fp8: bool,
    bf16_gate: bool,
    kernel: bool,
) -> bool {
    experts == 256 && topk == 8 && single_decode && native_fp8 && bf16_gate && kernel
}

impl MoeLayer {
    pub(super) fn topk_decode(
        &self,
        logits: DevicePtr,
        indices: DevicePtr,
        weights: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let single_decode =
            ctx.decode_step && ctx.attn_metadata.as_ref().map_or(1, |m| m.num_seqs) == 1;
        let use_warp = warp_eligible(
            ctx.config.num_experts,
            ctx.config.num_experts_per_tok,
            single_decode,
            self.fp8_gate_weight_ptrs.is_some(),
            self.gate_nvfp4.is_none()
                && !ctx.levers.fp32_gate
                && !self.fp32_routing_active(ctx.levers),
            self.moe_topk_warp.0 != 0,
        );
        if use_warp {
            if ctx.stats.once("log:moe_topk_warp_decode") {
                tracing::info!(
                    "[metrale] MoE single-sequence decode: Hopper warp top-k selected \
                     (experts=256, top_k=8, BF16 gate, native FP8 experts)"
                );
            }
            return KernelLaunch::new(ctx.gpu, self.moe_topk_warp)
                .grid([1, 1, 1])
                .block([32, 1, 1])
                .arg_ptr(logits)
                .arg_ptr(indices)
                .arg_ptr(weights)
                .arg_u32(256)
                .arg_u32(8)
                .arg_u32(u32::from(ctx.config.norm_topk_prob))
                .launch(stream);
        }
        ops::moe_topk_softmax(
            ctx.gpu,
            self.moe_topk,
            logits,
            indices,
            weights,
            ctx.config.num_experts as u32,
            ctx.config.num_experts_per_tok as u32,
            ctx.config.norm_topk_prob,
            stream,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::warp_eligible;

    #[test]
    fn specialized_single_token_native_fp8_route_only() {
        assert!(warp_eligible(256, 8, true, true, true, true));
        for (e, k) in [(128, 8), (512, 8), (256, 4), (256, 10)] {
            assert!(!warp_eligible(e, k, true, true, true, true));
        }
        for disabled in 0..4 {
            let mut flags = [true; 4];
            flags[disabled] = false;
            assert!(!warp_eligible(
                256, 8, flags[0], flags[1], flags[2], flags[3]
            ));
        }
    }
}
