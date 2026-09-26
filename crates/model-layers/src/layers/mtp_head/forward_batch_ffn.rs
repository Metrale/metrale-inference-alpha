// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The FFN of the batched propose (step 9 of
//! `forward_batch_position`) and the decision that names which FFN a head
//! can batch. The dense arm (`dense_ffn_generic`) runs gate/up/down as n-row
//! projections through `proj_rows` with SiLU over the `[n, inter]` block;
//! the native-FP8 MoE arm (`moe_fp8`) runs one
//! `MoeLayer::forward_fp8_grouped_decode` for the n rows.
//!
//! Owner: model-layers (MTP head).
//! Invariants:
//! - `MtpHead::propose_ffn_arm` is the one reader of the arm decision: the
//!   width scope (`batch_caps`), `ffn_rows` and the log line all call it.
//! - A head that qualifies for both arms, or for neither, is not batchable.
//! - The FFN of either arm writes neither `norm_output` (its input) nor
//!   `hidden_states` (the residual `ffn_rows` adds its result into). The
//!   grouped decode writes `gate_logits`, `scratch[0..2*n*top_k*4)`,
//!   `expert_{gate,up,down}_out`, `logits`, `ssm_qkvz`, `attn_output` and
//!   `moe_output` (the drafter's MoE has no router pre-norm).

use anyhow::Result;
use metrale_gpu_runtime::gpu::DevicePtr;

use super::batch_caps::is_row_proj;
use super::{MtpHead, ProjectionWeight};
use crate::layer::ForwardContext;
use crate::layers::ops;

/// 2026-09-25: Which FFN the batched propose runs for a head.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ProposeFfnArm {
    Dense,
    MoeFp8Grouped,
}

impl ProposeFfnArm {
    /// 2026-09-25: Name for the "propose_batch active" log line.
    fn name(self) -> &'static str {
        match self {
            Self::Dense => "DENSE",
            Self::MoeFp8Grouped => "MOE-FP8-GROUPED",
        }
    }
}

/// 2026-09-25: Arm selection from three facts. The dense arm needs a
/// row-dispatchable triple ([`dense_rows_ok`]) and its SiLU kernel; the MoE
/// arm needs the native-FP8 layer. Both, or neither, gives `None`.
pub(super) fn propose_ffn_arm(
    dense_rows: bool,
    silu_mul_k: bool,
    moe_fp8: bool,
) -> Option<ProposeFfnArm> {
    match (dense_rows && silu_mul_k, moe_fp8) {
        (true, false) => Some(ProposeFfnArm::Dense),
        (false, true) => Some(ProposeFfnArm::MoeFp8Grouped),
        _ => None,
    }
}

/// 2026-09-25: Whether a dense FFN triple can run through `proj_rows`: each
/// of gate/up/down BF16 or NVFP4 (`batch_caps::is_row_proj`).
pub(super) fn dense_rows_ok(
    ffn: Option<&(ProjectionWeight, ProjectionWeight, ProjectionWeight)>,
) -> bool {
    ffn.is_some_and(|(g, u, d)| is_row_proj(g) && is_row_proj(u) && is_row_proj(d))
}

impl MtpHead {
    /// 2026-09-25: The arm for this head.
    pub(super) fn propose_ffn_arm(&self) -> Option<ProposeFfnArm> {
        propose_ffn_arm(
            dense_rows_ok(self.dense_ffn_generic.as_ref()),
            self.moe_silu_mul_k.is_some(),
            self.moe_fp8.is_some(),
        )
    }

    /// 2026-09-25: Arm name for the log line; `NONE` when not batchable.
    pub(super) fn propose_ffn_arm_name(&self) -> &'static str {
        self.propose_ffn_arm().map_or("NONE", ProposeFfnArm::name)
    }

    /// 2026-09-25: `hidden[0..n] += FFN(normed2[0..n])` for the n stacked
    /// rows. Errors when the head has no batchable FFN.
    pub(super) fn ffn_rows(
        &self,
        normed2: DevicePtr,
        hidden: DevicePtr,
        n: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let gpu = ctx.gpu;
        let h = ctx.config.hidden_size;
        let ffn_out = match self.propose_ffn_arm() {
            Some(ProposeFfnArm::Dense) => {
                let inter = if ctx.config.intermediate_size > 0 {
                    ctx.config.intermediate_size as u32
                } else {
                    ctx.config.moe_intermediate_size as u32
                };
                let Some((gate_w, up_w, down_w)) = self.dense_ffn_generic.as_ref() else {
                    anyhow::bail!("propose_batch: no dense FFN (propose_ffn_arm lied)");
                };
                let Some(silu_k) = self.moe_silu_mul_k else {
                    anyhow::bail!("propose_batch: no SiLU kernel (propose_ffn_arm lied)");
                };
                let gate_out = ctx.buffers.expert_gate_out();
                let up_out = ctx.buffers.expert_up_out();
                self.proj_rows(gpu, normed2, gate_w, gate_out, n, inter, h as u32, stream)?;
                self.proj_rows(gpu, normed2, up_w, up_out, n, inter, h as u32, stream)?;
                ops::moe_silu_mul(
                    gpu,
                    silu_k,
                    gate_out,
                    up_out,
                    gate_out,
                    n as u32 * inter,
                    stream,
                )?;
                let ffn_out = ctx.buffers.moe_output();
                self.proj_rows(gpu, gate_out, down_w, ffn_out, n, h as u32, inter, stream)?;
                ffn_out
            }
            Some(ProposeFfnArm::MoeFp8Grouped) => {
                let Some(moe) = self.moe_fp8.as_ref() else {
                    anyhow::bail!("propose_batch: no FP8 MoE layer (propose_ffn_arm lied)");
                };
                // 2026-09-25: Errors unless `fp8_grouped_decode_ok` holds
                // (`propose_batch` checks it first); writes `moe_output`.
                moe.forward_fp8_grouped_decode(normed2, n, ctx, stream)?;
                ctx.buffers.moe_output()
            }
            None => anyhow::bail!("propose_batch: no batchable FFN (can_propose_batch lied)"),
        };
        ops::residual_add(
            gpu,
            self.residual_add_k,
            hidden,
            ffn_out,
            n as u32 * h as u32,
            stream,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::{ProposeFfnArm, propose_ffn_arm};

    #[test]
    fn dense_bf16_head_batches_through_the_dense_arm() {
        assert_eq!(
            propose_ffn_arm(true, true, false),
            Some(ProposeFfnArm::Dense)
        );
    }

    #[test]
    fn native_fp8_moe_head_batches_through_the_grouped_arm() {
        assert_eq!(
            propose_ffn_arm(false, false, true),
            Some(ProposeFfnArm::MoeFp8Grouped)
        );
        // 2026-09-25: The dense SiLU kernel does not affect the MoE arm.
        assert_eq!(
            propose_ffn_arm(false, true, true),
            Some(ProposeFfnArm::MoeFp8Grouped)
        );
    }

    #[test]
    fn per_expert_moe_head_cannot_batch() {
        // 2026-09-25: Per-expert MTP weights (no FP8 tables) have no
        // cross-sequence form (`moe_forward_generic` runs one row).
        assert_eq!(propose_ffn_arm(false, true, false), None);
        assert_eq!(propose_ffn_arm(false, false, false), None);
    }

    #[test]
    fn dense_arm_needs_its_silu_kernel() {
        assert_eq!(propose_ffn_arm(true, false, false), None);
    }

    #[test]
    fn a_head_with_both_identities_is_refused() {
        // 2026-09-25: `MtpHead::new` builds the MoE only when `dense_ffn` is
        // absent; the arm decision refuses the combination anyway.
        assert_eq!(propose_ffn_arm(true, true, true), None);
    }

    #[test]
    fn arm_names_are_distinct_for_the_log_line() {
        assert_ne!(
            ProposeFfnArm::Dense.name(),
            ProposeFfnArm::MoeFp8Grouped.name()
        );
    }
}
