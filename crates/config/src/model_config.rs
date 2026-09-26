// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: `ModelConfig`, the parsed checkpoint config: one field per model dimension or
//! switch the engine reads, plus the fields serve sets at startup.
//!
//! Owner: config.
//! Invariants: none beyond the types.

use super::*;

#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    pub hidden_size: usize,
    #[serde(default)]
    pub num_hidden_layers: usize,
    #[serde(default)]
    pub intermediate_size: usize,
    #[serde(default)]
    pub vocab_size: usize,

    #[serde(default)]
    pub num_attention_heads: usize,
    /// 2026-09-26: Per-layer Q-head counts (the laguna parser fills it). When non-empty,
    /// `validate_config` requires one entry per layer.
    #[serde(default)]
    pub num_attention_heads_per_layer: Vec<usize>,
    #[serde(default)]
    pub num_key_value_heads: usize,
    #[serde(default)]
    pub head_dim: usize,
    /// 2026-09-26: Fraction of `head_dim` that `rotary_dim()` rotates when `rotary_dim` is 0;
    /// 1.0 when absent.
    #[serde(default = "default_partial_rotary")]
    pub partial_rotary_factor: f64,

    #[serde(default)]
    pub linear_num_key_heads: usize,
    #[serde(default)]
    pub linear_key_head_dim: usize,
    #[serde(default)]
    pub linear_num_value_heads: usize,
    #[serde(default)]
    pub linear_value_head_dim: usize,
    #[serde(default = "default_conv_kernel")]
    pub linear_conv_kernel_dim: usize,

    #[serde(default)]
    pub num_experts: usize,
    #[serde(default)]
    pub zero_expert_num: usize,
    #[serde(default = "default_one")]
    pub num_experts_per_tok: usize,
    #[serde(default)]
    pub moe_intermediate_size: usize,
    #[serde(default)]
    pub shared_expert_intermediate_size: usize,
    #[serde(default)]
    pub norm_topk_prob: bool,
    #[serde(default = "default_one")]
    pub decoder_sparse_step: usize,

    /// 2026-09-26: Per-layer kind; when empty, `layer_type()` falls back to
    /// `full_attention_interval`.
    #[serde(default)]
    pub layer_types: Vec<LayerType>,
    /// 2026-09-26: Per-layer kind of the MTP / NextN layers past `num_hidden_layers`; empty
    /// when there are none. They stay out of `layer_types`, whose length `validate_config`
    /// checks against `num_hidden_layers`; [`ModelConfig::layer_type_at`] reads both.
    #[serde(default)]
    pub mtp_layer_types: Vec<LayerType>,
    /// 2026-09-26: With `layer_types` empty, every `full_attention_interval`-th layer is
    /// `FullAttention` and the rest `LinearAttention`.
    #[serde(default = "default_one")]
    pub full_attention_interval: usize,
    /// 2026-09-26: 0 when absent or JSON null.
    #[serde(default, deserialize_with = "nullable_u32")]
    pub sliding_window: u32,

    #[serde(default)]
    pub max_position_embeddings: usize,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,

    #[serde(default = "default_rms_eps")]
    pub rms_norm_eps: f64,

    /// 2026-09-26: JSON null gives 0.
    #[serde(default, deserialize_with = "nullable_u32")]
    pub bos_token_id: u32,
    /// 2026-09-26: See [`Glm5NextRouterMode`]; `HfFp32` when absent.
    #[serde(default)]
    pub glm5next_router_mode: Glm5NextRouterMode,
    /// 2026-09-26: The primary stop token; [`ModelConfig::eos_ids`] gives the full set.
    #[serde(default, deserialize_with = "eos_token_id_field")]
    pub eos_token_id: u32,
    /// 2026-09-26: Every declared stop id, primary first. `parse_config` fills it for every
    /// family. Empty means not populated (a hand-built `ModelConfig`), not "no stop tokens",
    /// so read it through [`ModelConfig::eos_ids`].
    #[serde(default)]
    pub eos_token_ids: Vec<u32>,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    /// 2026-09-26: Set at serve startup from `--lm-head-dtype`.
    /// `Some(true)` forces a BF16 LM head, `Some(false)` a quantized one, and `None` leaves
    /// the choice to `skip_lm_head_quantization()`'s model rules.
    #[serde(default)]
    pub lm_head_bf16_override: Option<bool>,
    /// 2026-09-26: Set by `--lm-head-dtype fp8`. When `skip_lm_head_quantization()` is false
    /// the LM head is FP8 (the checkpoint's own FP8 head, else a quantized copy) instead of
    /// NVFP4; a pre-packed NVFP4 head is used whatever this says.
    #[serde(default)]
    pub lm_head_fp8: bool,

    #[serde(default)]
    pub model_type: String,

    #[serde(default)]
    pub mtp_num_hidden_layers: usize,

    #[serde(default)]
    pub dspark_block_size: usize,
    #[serde(default)]
    pub dspark_noise_token_id: u32,
    #[serde(default)]
    pub dspark_target_layer_ids: Vec<usize>,
    #[serde(default)]
    pub dspark_markov_rank: usize,

    #[serde(default)]
    pub hybrid_override_pattern: String,
    #[serde(default)]
    pub mamba_num_heads: usize,
    #[serde(default)]
    pub mamba_head_dim: usize,
    #[serde(default)]
    pub ssm_state_size: usize,
    #[serde(default)]
    pub n_groups: usize,
    #[serde(default)]
    pub expand: usize,
    /// 2026-09-26: Nemotron-H name; the nemotron_h arm of `parse_config` copies it into
    /// `num_experts` when that is 0.
    #[serde(default)]
    pub n_routed_experts: usize,
    /// 2026-09-26: Nemotron-H name; copied into `rms_norm_eps` when that is still the default.
    #[serde(default)]
    pub norm_eps: f64,
    /// 2026-09-26: Nemotron-H name; copied into `linear_conv_kernel_dim` when that is still the
    /// default.
    #[serde(default)]
    pub conv_kernel: usize,
    /// 2026-09-26: Nemotron-H name; copied into `shared_expert_intermediate_size` when that is 0.
    #[serde(default)]
    pub moe_shared_expert_intermediate_size: usize,
    #[serde(default = "default_one_f64")]
    pub routed_scaling_factor: f64,
    /// 2026-09-26: KDA forget-gate lower bound (`linear_attn_config.gate_lower_bound`). The
    /// glm5_next parser refuses a config without it; for Kimi K3, 0.0 (absent) makes the host
    /// KDA path run unbounded.
    #[serde(default)]
    pub linear_gate_lower_bound: f32,
    /// 2026-09-26: SwiGLU clamp bound (`swiglu_limit`); 0.0 means no clamp. The glm5_next
    /// parser refuses a config without a positive one.
    #[serde(default)]
    pub swiglu_limit: f32,
    #[serde(default)]
    pub mlp_only_layers: Vec<usize>,
    #[serde(default)]
    pub moe_latent_size: usize,
    /// 2026-09-26: Per-layer routed-expert intermediate sizes, set by the Puzzle parser (one
    /// entry per layer, 0 for non-MoE layers) and never read from JSON. Empty means the scalar
    /// `moe_intermediate_size` for every layer.
    #[serde(default, skip_deserializing, skip_serializing)]
    pub moe_intermediate_sizes: Vec<usize>,
    /// 2026-09-26: Per-layer top-k expert counts, same layout as `moe_intermediate_sizes`.
    /// Empty means the scalar `num_experts_per_tok`.
    #[serde(default, skip_deserializing, skip_serializing)]
    pub num_experts_per_toks: Vec<usize>,

    /// 2026-09-26: 0 means standard attention; non-zero makes `capabilities()` report MLA.
    #[serde(default)]
    pub kv_lora_rank: usize,
    /// 2026-09-26: Per-layer KV cache `(num_kv_heads, head_dim)`, set by the model factory from
    /// the weight loader; never read from JSON.
    #[serde(default, skip_deserializing, skip_serializing)]
    pub kv_layer_dims: Vec<(usize, usize)>,
    #[serde(default)]
    pub q_lora_rank: usize,
    #[serde(default)]
    pub qk_nope_head_dim: usize,
    #[serde(default)]
    pub qk_rope_head_dim: usize,
    #[serde(default)]
    pub v_head_dim: usize,

    #[serde(default)]
    pub ngram_vocab_size_ratio: usize,
    #[serde(default)]
    pub emb_neighbor_num: usize,
    #[serde(default)]
    pub emb_split_num: usize,
    #[serde(default)]
    pub ngram_vocab_size_base: usize,
    #[serde(default)]
    pub ngram_split_parts: usize,
    #[serde(default)]
    pub ple_layer_ids: Vec<usize>,
    #[serde(default)]
    pub ple_conv_kernel_size: usize,

    #[serde(default)]
    pub o_lora_rank: usize,
    #[serde(default)]
    pub o_groups: usize,
    #[serde(default)]
    pub yarn_mscale: f32,
    #[serde(default)]
    pub yarn_mscale_all_dim: f32,
    #[serde(default)]
    pub hc_mult: usize,
    #[serde(default)]
    pub hc_sinkhorn_iters: usize,
    #[serde(default)]
    pub hc_eps: f32,
    #[serde(default)]
    pub hc_lowrank: usize,
    #[serde(default)]
    pub final_norm_identity: bool,
    /// 2026-09-26: Per-layer attention compression ratios; 0 means uncompressed. The GGUF
    /// builder accepts more entries than `num_hidden_layers`, never fewer.
    #[serde(default)]
    pub compress_ratios: Vec<usize>,
    #[serde(default)]
    pub index_n_heads: usize,
    #[serde(default)]
    pub index_head_dim: usize,
    #[serde(default)]
    pub index_topk: usize,
    /// 2026-09-26: The QSA indexer's compression ratio. The qwen4_exp parser sets it and
    /// leaves `compress_ratios` empty; 0 means no indexer.
    #[serde(default)]
    pub index_compress_ratio: usize,
    #[serde(default)]
    pub index_kpool: usize,
    #[serde(default)]
    pub index_kpool_always_select_tail: bool,
    #[serde(default)]
    pub num_hash_layers: usize,

    /// 2026-09-26: The compressor's RoPE base, separate from `rope_theta`; the GGUF builder
    /// reads `attention.compress_rope_freq_base`.
    #[serde(default)]
    pub compress_rope_theta: f32,
    #[serde(default)]
    pub kv_source_layer_ids: Vec<usize>,
    #[serde(default)]
    pub index_source_layer_ids: Vec<usize>,
    #[serde(default)]
    pub candidate_source_layer_id: Option<usize>,
    #[serde(default)]
    pub candidate_topk_blocks: usize,
    #[serde(default)]
    pub candidate_block_size: usize,

    #[serde(default)]
    pub engram_layer_ids: Vec<usize>,
    #[serde(default)]
    pub engram_num_embeddings: Vec<u64>,
    #[serde(default)]
    pub engram_max_ngram_size: usize,
    #[serde(default)]
    pub engram_vocab_size: usize,
    #[serde(default)]
    pub engram_n_heads: usize,
    #[serde(default)]
    pub engram_head_dim: usize,
    #[serde(default)]
    pub engram_pad_token_id: u32,
    #[serde(default)]
    pub engram_compressed_vocab_size: usize,
    #[serde(default)]
    pub engram_multipliers: Vec<u64>,
    #[serde(default)]
    pub engram_primes: Vec<u64>,
    #[serde(default)]
    pub engram_offsets: Vec<u64>,

    #[serde(default)]
    pub yarn_factor: f32,
    #[serde(default)]
    pub yarn_beta_slow: f32,
    #[serde(default)]
    pub yarn_beta_fast: f32,
    #[serde(default)]
    pub yarn_original_max_position_embeddings: usize,
    #[serde(default = "default_one_f32")]
    pub yarn_attention_factor: f32,
    #[serde(default)]
    pub llama_4_scaling_beta: f32,
    #[serde(default)]
    pub llama_4_scaling_original_max_position_embeddings: usize,

    /// 2026-09-26: `None` for text-only models, and for GLM-5.3 unless `glm_vision_enabled()`.
    #[serde(skip)]
    pub vision: Option<VisionConfig>,

    /// 2026-09-26: Set by `finalize_config` from `quantization_config`. When that leaves it
    /// `None`, serve fills it from a sibling `hf_quant_config.json`
    /// (`merge_sidecar_quant_config`).
    #[serde(skip)]
    pub quantization_config: Option<QuantizationConfig>,

    /// 2026-09-26: Whether the Q projection carries an output gate. The parsers set it: false
    /// for `qwen3_vl_moe`, Nemotron-H, and the GGUF llama/qwen2 mapping.
    #[serde(skip)]
    pub attn_gated: bool,
    /// 2026-09-26: The GDN gated norm uses a sigmoid gate instead of SiLU; the qwen4_exp
    /// parser sets it for `output_gate_type: "sigmoid"`.
    #[serde(default)]
    pub gdn_norm_sigmoid: bool,
    /// 2026-09-26: The text model's fields sit under a nested key such as `text_config`.
    /// Weight-prefix detection reads it when `weight_prefix` is empty.
    #[serde(skip)]
    pub nested_config: bool,
    /// 2026-09-26: MRoPE section sizes; `[0, 0, 0]` when the config declares none.
    #[serde(skip)]
    pub mrope_section: [usize; 3],
    #[serde(skip)]
    pub mrope_interleaved: bool,

    #[serde(skip)]
    pub weight_prefix: String,

    /// 2026-09-26: Set from serve's `--profile`.
    #[serde(skip)]
    pub profile: bool,

    #[serde(skip)]
    pub ep_rank: usize,
    #[serde(skip)]
    pub ep_world_size: usize,

    #[serde(skip)]
    pub tp_rank: usize,
    #[serde(skip)]
    pub tp_world_size: usize,

    /// 2026-09-26: The serve's `--max-seq-len`, set at startup. It is 0 when nothing set it (a
    /// unit test, an offline tool), which readers must treat as unknown, not as zero context.
    #[serde(skip)]
    pub serve_max_seq_len: usize,

    #[serde(skip)]
    pub fp8_kv_calibration_tokens: usize,
    #[serde(skip)]
    pub fp8_kv_headroom: f32,

    #[serde(skip)]
    pub final_logit_softcapping: f32,
    #[serde(skip)]
    pub embed_scale: f32,

    #[serde(default)]
    pub scoring_func: String,
    #[serde(default)]
    pub use_routing_bias: bool,
    #[serde(default)]
    pub qk_norm_type: String,
    #[serde(default)]
    pub num_mtp_modules: usize,
    #[serde(default)]
    pub mtp_transformer_layers: usize,
    /// 2026-09-26: When non-zero, `rotary_dim()` returns it instead of
    /// `partial_rotary_factor * head_dim`.
    #[serde(default)]
    pub rotary_dim: usize,

    #[serde(default)]
    pub dflash_capture_layers: Vec<usize>,
    /// 2026-09-26: The DFlash drafter γ, set by the model factory with `dflash_capture_layers`;
    /// `None` when DFlash is inactive.
    pub dflash_gamma: Option<usize>,

    /// 2026-09-26: `--max-lora-rank`, set by the model factory. `BufferSizes` sizes the LoRA
    /// scratch from it and allocates none when it is 0.
    #[serde(default)]
    pub adapter_max_rank: usize,

    /// 2026-09-26: `attn_res_block_size`; the kimi_k3 parser requires it.
    #[serde(default)]
    pub attn_res_block_size: usize,
    /// 2026-09-26: `linear_attn_config.use_full_rank_gate`.
    #[serde(default)]
    pub use_full_rank_gate: bool,
    #[serde(default)]
    pub mla_use_nope: bool,
    #[serde(default)]
    pub mla_use_output_gate: bool,
    #[serde(default)]
    pub latent_moe_use_norm: bool,
    /// 2026-09-26: HF `hidden_act`; empty when absent.
    #[serde(default)]
    pub hidden_act: String,
    #[serde(default)]
    pub activation_situ_beta: f32,
    #[serde(default)]
    pub activation_situ_linear_beta: f32,
    /// 2026-09-26: `num_shared_experts`, else `n_shared_experts` (kimi_k3 parser).
    #[serde(default)]
    pub n_shared_experts: usize,
}
