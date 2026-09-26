// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: KDA block geometry (`Glm5NextKdaConfig`) and device weights (`Glm5NextKdaWeights`).
//!
//! Owner: model-arch (GLM-5.3-Flash KDA).
//! Invariants: none beyond the types; `Glm5NextKdaConfig::validate` states what the kernels need.

use super::*;

/// 2026-09-25: KDA geometry and the two numeric config values the block uses. The loader fills
/// every field from `ModelConfig` except `l2_eps` and `chunk`; there is no `Default`.
#[derive(Clone, Copy, Debug)]
pub struct Glm5NextKdaConfig {
    pub hidden: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub conv_kernel: usize,
    /// 2026-09-25: `linear_attn_config.gate_lower_bound`.
    pub gate_lower_bound: f32,
    /// 2026-09-25: `rms_norm_eps`, read by `o_norm`.
    pub rms_norm_eps: f32,
    /// 2026-09-25: The L2 epsilon, applied as `1/sqrt(sum + eps)` rather than `max(norm, eps)`.
    /// Not a config key.
    pub l2_eps: f32,
    /// 2026-09-25: Prefill chunk width, bounded by [`SMEM_CEILING`] through `validate`.
    pub chunk: usize,
}

impl Glm5NextKdaConfig {
    pub fn qkv_dim(&self) -> usize {
        self.heads * self.head_dim
    }
    /// 2026-09-25: q | k | v concatenated: the channel count the depthwise conv sees.
    pub fn conv_dim(&self) -> usize {
        3 * self.qkv_dim()
    }
    /// 2026-09-25: The q | k channels, the only ones L2-normalised.
    pub fn qk_channels(&self) -> usize {
        2 * self.qkv_dim()
    }
    /// 2026-09-25: Elements of the FP32 `[heads, head_dim, head_dim]` K-major recurrent state.
    pub fn recurrent_state_elems(&self) -> usize {
        self.heads * self.head_dim * self.head_dim
    }
    /// 2026-09-25: Elements of the FP32 `[conv_dim, conv_kernel]` conv state (see `KdaSeqState`).
    pub fn conv_state_elems(&self) -> usize {
        self.conv_dim() * self.conv_kernel
    }
    pub fn smem_prepare(&self) -> usize {
        (self.chunk * self.head_dim + self.chunk * self.chunk + self.chunk) * 4
    }
    pub fn smem_scan(&self) -> usize {
        (2 * self.chunk * self.head_dim + self.chunk * self.chunk) * 4
    }

    pub fn validate(&self) -> Result<()> {
        if !self.qk_channels().is_multiple_of(256) {
            bail!("causal_conv1d_update_l2norm requires qk_channels % 256 == 0");
        }
        if self.head_dim != 128 {
            bail!("the fused conv+L2 kernel hardcodes 2 heads per 256-thread block");
        }
        if self.conv_kernel > 4 {
            bail!("the conv kernels keep the sliding window in 4 registers");
        }
        let (p, s) = (self.smem_prepare(), self.smem_scan());
        if p > SMEM_CEILING || s > SMEM_CEILING {
            bail!(
                "chunk={} needs {p}/{s} B shared, ceiling {SMEM_CEILING}",
                self.chunk
            );
        }
        Ok(())
    }
}

/// 2026-09-25: One KDA block's device weights. Torch `Linear` layout `[out, in]`, BF16, except the
/// two F32 gate parameters. There is no `Z` tensor; the output gate is low-rank `g_a`/`g_b`.
pub struct Glm5NextKdaWeights {
    pub q_proj: DenseWeight,
    pub k_proj: DenseWeight,
    pub v_proj: DenseWeight,
    /// 2026-09-25: `[conv_dim, conv_kernel]` BF16: the q, k and v conv tensors concatenated, with the
    /// singleton middle dimension dropped.
    pub conv: DenseWeight,
    pub f_a: DenseWeight,
    pub f_b: DenseWeight,
    /// 2026-09-25: `[heads * head_dim]` F32, one value per channel.
    pub dt_bias: DevicePtr,
    /// 2026-09-25: `[heads]` F32, one value per head (unlike `dt_bias`).
    pub a_log: DevicePtr,
    pub b_proj: DenseWeight,
    pub g_a: DenseWeight,
    pub g_b: DenseWeight,
    /// 2026-09-25: `[head_dim]` BF16.
    pub o_norm: DenseWeight,
    pub o_proj: DenseWeight,
}
