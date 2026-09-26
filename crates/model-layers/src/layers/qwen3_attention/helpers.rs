// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `Qwen3AttentionLayer` setters that model loaders call after
//! construction, the YaRN mscale helper, and small per-layer helpers
//! (`apply_layer_scalar`, `effective_attn_scale`, `nvfp4_decode_gemv`,
//! `qsa_seq_state`).
//!
//! Owner: model-layers (attention).
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

use super::types::{HcWeights, HeadGateActivation, MlaWeights, Qwen3AttentionLayer};
use crate::layers::FfnComponent;
use crate::layers::ops;
use crate::weight_map::{DenseWeight, QuantizedWeight};

/// 2026-09-25: YaRN attention-temperature factor for one `mscale`:
/// `0.1 * mscale * ln(scale) + 1.0` for `scale > 1`, else `1.0`.
fn yarn_get_mscale(scale: f32, mscale: f32) -> f32 {
    if scale <= 1.0 {
        1.0
    } else {
        0.1 * mscale * scale.ln() + 1.0
    }
}

/// 2026-09-25: The YaRN mscale ratio
/// `get_mscale(factor, yarn_mscale) / get_mscale(factor, yarn_mscale_all_dim)`,
/// or 1.0 when `yarn_factor <= 1`. The DeepSeek-V4 RoPE call sites pass it to
/// their kernels.
pub fn yarn_rope_mscale(config: &metrale_config::ModelConfig) -> f32 {
    let factor = config.yarn_factor;
    if factor <= 1.0 {
        return 1.0;
    }
    let num = yarn_get_mscale(factor, config.yarn_mscale);
    let den = yarn_get_mscale(factor, config.yarn_mscale_all_dim);
    num / den
}

impl Qwen3AttentionLayer {
    /// 2026-09-25: Installs the MLA weights; the decode and prefill paths take
    /// their MLA arms when `mla` is set.
    pub fn set_mla_weights(&mut self, mla: MlaWeights) {
        self.mla = Some(mla);
    }

    /// 2026-09-25: Installs the hyper-connection weights. When they are set,
    /// decode and prefill take the hyper-connection paths, which work on the
    /// model-level `hc_streams` buffer.
    pub fn set_hc_weights(&mut self, hc: HcWeights) {
        self.hc = Some(hc);
    }

    pub fn set_qsa(&mut self, qsa: crate::layers::qsa::QsaIndexer) {
        self.qsa = Some(qsa);
    }

    /// 2026-09-25: Per-layer head_dim and Q/KV head counts, for models whose
    /// attention layers differ in shape.
    pub fn set_dimension_overrides(
        &mut self,
        head_dim: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
    ) {
        self.head_dim_override = Some(head_dim);
        self.num_q_heads_override = Some(num_q_heads);
        self.num_kv_heads_override = Some(num_kv_heads);
    }

    /// 2026-09-25: Per-layer sliding-window size: `Some(window)` on
    /// sliding-window layers, `None` on full-attention layers.
    pub fn set_sliding_window(&mut self, window: Option<u32>) {
        self.sliding_window = window;
    }

    /// 2026-09-25: Per-head attention-output gate: a BF16
    /// `[num_q_heads, hidden_size]` weight and its activation.
    pub fn set_head_gate_weight(&mut self, w: DenseWeight, activation: HeadGateActivation) {
        self.head_gate_weight = Some(w);
        self.head_gate_activation = activation;
    }

    pub fn set_yarn_rope(&mut self, inv_freq: DevicePtr, attention_factor: f32) {
        self.yarn_inv_freq = inv_freq;
        self.yarn_attention_factor = attention_factor;
    }

    pub fn set_rope_overrides(&mut self, theta: f32, rotary_dim: u32) {
        self.rope_theta_override = Some(theta);
        self.rotary_dim_override = Some(rotary_dim);
    }

    /// 2026-09-25: Proportional RoPE: rotation pairs `(i, i + head_dim/2)` for
    /// `i` below the `rotary_dim` given to `set_rope_overrides`, which this
    /// mode reads as the number of rotated pairs (`ops::rope_proportional`).
    /// Both setters only store fields, so their order does not matter.
    pub fn set_rope_proportional(&mut self, enable: bool) {
        self.rope_proportional = enable;
    }

    /// 2026-09-25: Replaces the default `1/sqrt(head_dim)` attention scale
    /// (`effective_attn_scale`).
    pub fn set_attn_scale_override(&mut self, scale: f32) {
        self.attn_scale_override = Some(scale);
    }

    /// 2026-09-25: NVFP4 M=1 decode GEMV: the single-warp-per-output
    /// `w4a16_gemv_sw` when `use_sw` is set and that kernel resolved, otherwise
    /// `w4a16_gemv` (`ops::w4a16_decode_gemv`).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn nvfp4_decode_gemv(
        &self,
        gpu: &dyn GpuBackend,
        use_sw: bool,
        input: DevicePtr,
        weight: &QuantizedWeight,
        output: DevicePtr,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<()> {
        ops::w4a16_decode_gemv(
            gpu,
            self.w4a16_gemv_k,
            self.w4a16_gemv_sw_k,
            use_sw,
            input,
            weight,
            output,
            n,
            k,
            stream,
        )
    }

    /// 2026-09-25: Sets `k_eq_v` and installs `v_norm_weight`, a BF16
    /// `[head_dim]` buffer. The Gemma-4 loader calls this, with a ones-filled
    /// weight, when the checkpoint has no V projection.
    pub fn set_k_eq_v(&mut self, v_norm_weight: DenseWeight) {
        self.k_eq_v = true;
        self.v_norm_weight = Some(v_norm_weight);
    }

    /// 2026-09-25: Installs `v_norm_weight` without setting `k_eq_v`. The
    /// Gemma-4 loader calls this, with a ones-filled weight, when the checkpoint
    /// has a V projection.
    pub fn set_v_norm(&mut self, v_norm_weight: DenseWeight) {
        self.v_norm_weight = Some(v_norm_weight);
    }

    /// 2026-09-25: Installs a BF16 output-projection weight. The decode and
    /// prefill o_proj dispatch (`decode/attention_forward_oproj.rs`,
    /// `prefill/paged_oproj.rs`, `trait_impl/multi_seq/attn/o_proj.rs`) uses it
    /// when it is set and no earlier arm (MLA, a transposed quantized copy)
    /// matched.
    pub fn set_o_dense_bf16(&mut self, o_dense: DenseWeight) {
        self.o_dense_bf16 = Some(o_dense);
    }

    pub fn set_post_sublayer_norms(
        &mut self,
        post_attn_out: DenseWeight,
        post_ffn_out: DenseWeight,
    ) {
        self.post_attn_out_norm = Some(post_attn_out);
        self.post_ffn_out_norm = Some(post_ffn_out);
    }

    pub fn set_layer_scalar(&mut self, scalar: f32) {
        self.layer_scalar = Some(scalar);
    }

    /// 2026-09-25: Installs a second FFN (an MoE) with the pre-MoE,
    /// post-MoE-output and post-dense-FFN norms; the dual-FFN path runs only
    /// when all four are set.
    pub fn set_moe_ffn(
        &mut self,
        ffn: FfnComponent,
        pre_norm: DenseWeight,
        post_norm: DenseWeight,
        post_dense_norm: DenseWeight,
    ) {
        self.moe_ffn = Some(ffn);
        self.pre_moe_norm = Some(pre_norm);
        self.post_moe_out_norm = Some(post_norm);
        self.post_dense_ffn_norm = Some(post_dense_norm);
    }

    /// 2026-09-25: Installs the shortcut MoE on the first sublayer of a
    /// dual-sublayer block. The MoE runs on this sublayer's post-attention
    /// normed input, before its dense FFN, and its output is stored in `carry`
    /// (room for `carry_tokens` tokens; a larger chunk is an error). The second
    /// sublayer adds it back, see [`Self::set_shortcut_carry_in`].
    pub fn set_shortcut_moe(
        &mut self,
        moe: FfnComponent,
        carry: metrale_gpu_runtime::gpu::DevicePtr,
        carry_tokens: usize,
    ) {
        self.moe_ffn = Some(moe);
        self.shortcut_carry_out = Some((carry, carry_tokens));
    }

    /// 2026-09-25: On the second sublayer of a dual-sublayer block: adds the
    /// paired first sublayer's stored shortcut-MoE output at the end of this
    /// sublayer.
    pub fn set_shortcut_carry_in(
        &mut self,
        carry: metrale_gpu_runtime::gpu::DevicePtr,
        carry_tokens: usize,
    ) {
        self.shortcut_carry_in = Some((carry, carry_tokens));
    }

    /// 2026-09-25: Multiplies the `hidden_size` BF16 values at `hidden` by
    /// `scalar` in place (`embed_scale::bf16_scale_inplace`).
    pub(crate) fn apply_layer_scalar(
        &self,
        gpu: &dyn metrale_gpu_runtime::gpu::GpuBackend,
        hidden: metrale_gpu_runtime::gpu::DevicePtr,
        hidden_size: usize,
        scalar: f32,
        stream: u64,
    ) -> anyhow::Result<()> {
        use metrale_gpu_runtime::kernel_args::KernelLaunch;
        let scale_k = gpu.kernel("embed_scale", "bf16_scale_inplace")?;
        let n = hidden_size as u32;
        KernelLaunch::new(gpu, scale_k)
            .grid([n.div_ceil(256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(hidden)
            .arg_u32(n)
            .arg_f32(scalar)
            .launch(stream)
    }

    pub(crate) fn effective_attn_scale(&self, head_dim: u32) -> f32 {
        self.attn_scale_override
            .unwrap_or_else(|| 1.0f32 / (head_dim as f32).sqrt())
    }
}

/// 2026-09-25: The sequence's QSA state inside its
/// [`crate::layer::AttnLayerState`], created on first use. Errors when the
/// layer state is not an `AttnLayerState`.
pub(in crate::layers::qwen3_attention) fn qsa_seq_state<'a>(
    qsa: &crate::layers::qsa::QsaIndexer,
    state: &'a mut dyn crate::layer::LayerState,
    gpu: &dyn metrale_gpu_runtime::gpu::GpuBackend,
) -> anyhow::Result<&'a mut crate::layers::qsa::QsaSeqState> {
    let attn = state
        .as_any_mut()
        .downcast_mut::<crate::layer::AttnLayerState>()
        .ok_or_else(|| anyhow::anyhow!("QSA host layer state is not AttnLayerState"))?;
    if attn.qsa.is_none() {
        attn.qsa = Some(qsa.new_seq_state(gpu)?);
    }
    Ok(attn.qsa.as_mut().expect("just created"))
}

#[cfg(test)]
mod yarn_mscale_tests {
    use super::yarn_rope_mscale;
    use metrale_config::ModelConfig;

    // 2026-09-25: `yarn_mscale == yarn_mscale_all_dim == 0.0` at factor 16
    // gives exactly 1.0.
    #[test]
    fn ds4f_forced_config_yields_mscale_one() {
        let mut c = ModelConfig::qwen3_next_80b_nvfp4();
        c.yarn_factor = 16.0;
        c.yarn_mscale = 0.0;
        c.yarn_mscale_all_dim = 0.0;
        assert_eq!(yarn_rope_mscale(&c), 1.0);
    }

    // 2026-09-25: mscale 1.0 and mscale_all_dim 0.0 at factor 16 give
    // `0.1 * ln(16) + 1`, about 1.2772589.
    #[test]
    fn generic_yarn_default_unchanged_1277() {
        let mut c = ModelConfig::qwen3_next_80b_nvfp4();
        c.yarn_factor = 16.0;
        c.yarn_mscale = 1.0;
        c.yarn_mscale_all_dim = 0.0;
        let m = yarn_rope_mscale(&c);
        assert!((m - 1.2772589).abs() < 1e-5, "expected ~1.2772589, got {m}");
    }

    // 2026-09-25: `yarn_factor <= 1` returns 1.0 without reading the mscales.
    #[test]
    fn yarn_disabled_factor_one_is_mscale_one() {
        let mut c = ModelConfig::qwen3_next_80b_nvfp4();
        c.yarn_factor = 1.0;
        c.yarn_mscale = 1.0;
        c.yarn_mscale_all_dim = 0.0;
        assert_eq!(yarn_rope_mscale(&c), 1.0);
    }
}
