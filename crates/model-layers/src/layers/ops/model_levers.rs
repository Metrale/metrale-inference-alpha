// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Model-side kernel-path levers: one `Copy` struct, resolved from the environment and carried on `ForwardContext`.
//!
//! Owner: model-layers ops.
//! Invariants:
//! - [`ModelLevers::get`] resolves the environment once per process (a `OnceLock` in
//!   `model_levers_resolve.rs`).
//! - `from_values` reads nothing but its two closures and its arguments, so the tests drive the
//!   production resolution without touching the process environment.
//! - Every field but `max_decode_seqs` comes from `from_values`. It sets `max_decode_seqs` to 1,
//!   and `TransformerModel::new` overwrites it with the configured max batch.
//!
//! Read a lever here once and pass the resolved field down; do not read `METRALE_*` in code that
//! runs per token, per layer or per request. The `hot_path_env_guards` test fails when a guarded
//! hot-path file reads the environment outside its allow list.
//!
//! [`super::GemmDispatch`], the other lever struct on [`crate::layer::ForwardContext`], chooses
//! the GEMM implementation of each projection. [`ModelLevers`] holds every other kernel-path
//! switch.

/// 2026-09-25: Kernel-path levers for one loaded model. `TransformerModel::new` resolves its own
/// copy with [`ModelLevers::from_env`]; other readers use the process-wide [`ModelLevers::get`].
// 2026-09-25: `Eq` is not derived because `draft_conf_tau` is an `f32`.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct ModelLevers {
    /// 2026-09-25: On unless `METRALE_NO_GDN_REGRESIDENT=1`. Allows the register-resident GDN
    /// prefill recurrence on the replay path when the K and V head dims are 128 and the kernel
    /// resolved (`qwen3_ssm/trait_prefill_recur.rs`).
    pub gdn_regresident: bool,
    /// 2026-09-25: `METRALE_GDN_BATCHED_FLA=1`: the batched FLA kernels for multi-stream GDN
    /// prefill when the K and V head dims are 128 (`qwen3_ssm/trait_prefill_gdn/batched.rs`).
    pub gdn_batched_fla: bool,
    /// 2026-09-25: On unless `METRALE_GDN_WY17=0`. Allows the WY17 kernel for a 17-token batched
    /// GDN decode, except under the FP16 SSM state (`trait_decode_batched_conv_gdn.rs`).
    pub gdn_wy17: bool,
    /// 2026-09-25: On unless `METRALE_GDN_WYN=0`. Passed to `wyn_kernel` / `wyn_table_kernel`,
    /// which choose the WY-N GDN decode kernels.
    pub gdn_wyn: bool,

    /// 2026-09-25: On unless `METRALE_NO_GEMV_SW=1` (`gemv_sw::gemv_sw_from`). Selects the
    /// single-warp decode GEMVs (`w4a16_gemv_sw`, `w4a16_gemv_dual_sw`) where their kernel
    /// resolved (`gemv_sw::use_gemv_sw`).
    pub gemv_sw: bool,
    /// 2026-09-25: `METRALE_DECODE_FFN_VIA_GEMM=1`: a SiLU dense-FFN decode runs gate, up and down
    /// through `w4a16_prefill_gemm` at M=1 when the transposed `gate_proj_t` / `up_proj_t` weights
    /// are present; without them it warns once and keeps the GEMV path (`dense_ffn.rs`).
    pub decode_ffn_via_gemm: bool,
    /// 2026-09-25: On unless `METRALE_FFN_SMALLM=0`. Allows the small-M GEMM tiles for
    /// `m <= 64` with K a multiple of 32 (`dense_ffn.rs`, `multi_seq/qkv.rs`).
    pub ffn_small_m: bool,
    /// 2026-09-25: `METRALE_HOLO_MOE_DOWN_FP4=1` or `true` (any case). Only the weight loader's
    /// warning reads this field (`weight_loader/qwen35/load_layers.rs`); `MoeLayer` reads the
    /// variable itself into `down_fp4` (`moe/init.rs`).
    pub holo_moe_down_fp4: bool,
    /// 2026-09-25: `METRALE_HOLO_MOE_GATEUP_FP4=1` or `true` (any case). Only the weight loader's
    /// warning reads this field; `MoeLayer` reads the variable itself into `gateup_fp4`.
    pub holo_moe_gateup_fp4: bool,
    /// 2026-09-25: `METRALE_MOE_UNION_STATS=1`: sample per-layer MoE expert-union statistics
    /// into `ModelStats::moe_union` (`moe/union_stats.rs`).
    pub moe_union_stats: bool,
    /// 2026-09-25: `METRALE_FP32_ROUTING=1`: the MoE-input norm writes an FP32 router input and
    /// the gate GEMM reads it at full precision. The last term of
    /// `MoeLayer::fp32_routing_active`, whose other four terms are properties of the layer's
    /// weights and kernels.
    pub fp32_routing: bool,
    /// 2026-09-25: `METRALE_FP32_GATE=1`: the batched gate writes FP32 logits from the BF16
    /// router input (`moe/forward_batched_gate.rs`). Implied there by an active
    /// [`Self::fp32_routing`].
    pub fp32_gate: bool,
    /// 2026-09-25: `METRALE_FRANKENSTEIN_DECODE_VIA_PREFILL=1`: DFlash capture layers
    /// (`is_dflash_capture_layer`) run their MoE decode through `forward_prefill` with one row;
    /// other layers are unchanged (`moe/forward.rs`). Diagnostic.
    pub frankenstein_decode_via_prefill: bool,
    /// 2026-09-25: `METRALE_K2_DIAG=1`: synchronising checkpoints in `forward_k2`, each labelled
    /// with its stage, so the first failing stage names itself.
    pub k2_diag: bool,

    /// 2026-09-25: On when the compiled target's `[defaults] decode_split_silu` is on (a target
    /// that does not declare it inherits on, `crates/kernels/build_defaults.rs`), unless
    /// `METRALE_NO_DECODE_SPLIT_SILU` is present with any value. Decode runs `silu_mul` and then
    /// a separate down GEMV instead of the fused SiLU+down kernel. An installed LoRA adapter takes
    /// the split path whatever this says (`levers.decode_split_silu || self.lora.is_some()` in
    /// `dense_ffn.rs`).
    pub decode_split_silu: bool,
    /// 2026-09-25: `METRALE_BF16_TC_PREFILL` (presence): the BF16 tensor-core dense-FFN prefill
    /// GEMM. The call site honours it only when the BF16 kernel it selected (v2, or v1 without
    /// [`Self::prefill_v2`]) is loaded.
    pub bf16_tc_prefill: bool,
    /// 2026-09-25: `METRALE_FP8_M64_PREFILL` (presence): the FP8 M64 dense-FFN prefill arm, when
    /// `w4a16_gemm_t` is loaded.
    pub fp8_m64_prefill: bool,
    /// 2026-09-25: `METRALE_INT8_PREFILL` (presence): dense-FFN prefill through an int8 requant
    /// and `int8_gemm_faith2`, when that kernel is loaded.
    pub int8_prefill: bool,
    /// 2026-09-25: `METRALE_INT8_FAITH5` (presence): the int8 arm launches the faith5 kernel
    /// instead of faith2, when faith5 is loaded.
    pub int8_faith5: bool,
    /// 2026-09-25: On unless `METRALE_NO_FFN_NVFP4_MMQ` is present. NVFP4 W4A4 MMQ for the
    /// gate/up prefill GEMMs of a SiLU dense FFN, when its kernels are loaded and no LoRA adapter
    /// is installed (`dense_ffn.rs`).
    pub ffn_nvfp4_mmq: bool,
    /// 2026-09-25: On unless `METRALE_NO_FFN_NVFP4_MMQ_DOWN` is present. The same MMQ arm for the
    /// down projection. It applies only when the gate/up arm ([`Self::ffn_nvfp4_mmq`]) is active,
    /// but the two variables are independent.
    pub ffn_nvfp4_mmq_down: bool,
    /// 2026-09-25: `METRALE_FFN_MMQ` (presence): the Q4_K MMQ dense-FFN prefill arm, when its
    /// kernels are loaded and the NVFP4 MMQ arm is not active.
    pub ffn_mmq: bool,
    /// 2026-09-25: `METRALE_FFN_MMQ_DOWN_Q4K` (presence): under [`Self::ffn_mmq`], keep the down
    /// projection on Q4_K. Unset, down goes through the int8 faith2 path (`down_faith2` in
    /// `dense_ffn.rs`, which reads `!levers.ffn_mmq_down_q4k`).
    pub ffn_mmq_down_q4k: bool,
    /// 2026-09-25: `METRALE_FP4_PREFILL` (presence): dense-FFN prefill through `w4a4_gemm`, with
    /// the activations quantized to NVFP4 per GEMM, when both kernels are loaded.
    pub fp4_prefill: bool,
    /// 2026-09-25: On unless `METRALE_DISABLE_PREFILL_V2` is present. Prefer the v2 BF16 t_m128
    /// prefill kernel over v1 when v2 is loaded.
    pub prefill_v2: bool,

    /// 2026-09-25: `METRALE_HOLO_MOE_GROUPED_CUTLASS=1`: the routed MoE prefill runs gate/up
    /// through the single-launch CUTLASS grouped NVFP4 GEMM when the layer holds the load-time
    /// host tables (`cutlass_grouped_host`, `moe/forward_prefill_routed.rs`).
    pub moe_grouped_cutlass: bool,
    /// 2026-09-25: `METRALE_HOLO_MOE_GROUPED_DOWN=1`: the down projection too. It also needs
    /// [`Self::moe_grouped_cutlass`] and the host down tables.
    pub moe_grouped_down: bool,
    /// 2026-09-25: `METRALE_MOE_PREFILL_EXACT_TILES=1` or `=0`; any other value, or unset, is
    /// `None`, which the call site resolves to on for NVFP4 experts and off otherwise
    /// (`forward_prefill_routed.rs`). When on, the per-expert tile bound is read back from device
    /// memory, so graph capture forces it off.
    pub moe_prefill_exact_tiles: Option<bool>,
    /// 2026-09-25: `METRALE_MOE_PREFILL_MAX_LOAD_FACTOR=<n>`: with exact tiles off, cap the
    /// per-expert tile bound at n times the average rows per expert. Unset, unparseable or `0` is
    /// `None`, the worst-case bound.
    pub moe_prefill_max_load_factor: Option<usize>,
    /// 2026-09-25: `METRALE_MOE_PREFILL_ZERO=1`: zero the routed-prefill gate/up/down scratch
    /// before the grouped GEMMs. Expert parallelism (`ctx.comm`) zeroes it regardless.
    pub moe_prefill_zero: bool,
    /// 2026-09-25: `METRALE_MOE_PREFILL_FP8_DOWN=1`: the FP8 grouped GEMM for the routed-prefill
    /// down projection, when `moe_fp8_grouped_gemm_t` and `bf16_to_fp8` are loaded.
    pub moe_prefill_fp8_down: bool,

    // 2026-09-25: The Nemotron prefill levers (`nemotron_mamba2/prefill.rs`,
    // `nemotron_moe/prefill_sorted.rs`, `prefill_shared_up.rs`). All eight are presence-gated:
    // any value, `0` included, counts as set.
    /// 2026-09-25: On unless `METRALE_NO_SSM_W4A4` is present. The W4A4 native-FP4 GEMM for the
    /// Mamba2 SSM prefill projections at n >= 512, when the kernels are loaded and the `fp8_act`
    /// buffer is large enough.
    pub ssm_w4a4: bool,
    /// 2026-09-25: On unless `METRALE_NO_SSD` is present. The chunked SSD scan for Mamba2
    /// prefill, when `ops::ssd_scan_fits(state_size)` also holds.
    pub ssd: bool,
    /// 2026-09-25: On unless `METRALE_NO_SSM_PERSISTENT` is present. The persistent SSM prefill
    /// kernel, tried only when the SSD scan is not taken; otherwise the sequential scan runs.
    pub ssm_persistent: bool,
    /// 2026-09-25: On unless `METRALE_MOE_NO_ZERO_INTERMEDIATES` is present. Zero
    /// `expert_up_out` and `expert_down_out` before the sorted-prefill grouped GEMMs.
    pub moe_zero_intermediates: bool,
    /// 2026-09-25: `METRALE_MOE_MAX_M_TILES_ESTIMATE` (presence): size the grouped GEMM's
    /// `max_m_tiles` (its grid.y) from twice the average rows per expert instead of the worst
    /// case, every routed row on one expert. Rows of an expert beyond that bound get no block and
    /// are not computed.
    pub moe_max_m_tiles_estimate: bool,
    /// 2026-09-25: `METRALE_MOE_W4A4` (presence): the W4A4 grouped up projection for a latent
    /// MoE input at n >= 512.
    pub moe_w4a4: bool,
    /// 2026-09-25: On unless `METRALE_NO_SHARED_W4A4` is present. W4A4 for the shared-expert up
    /// projection at n >= 512, when no native FP8 shared-up weight is used.
    pub shared_w4a4: bool,
    /// 2026-09-25: `METRALE_SHARED_W4A4_DOWN` (presence): W4A4 for the shared-expert down
    /// projection at n >= 512. Independent of [`Self::shared_w4a4`].
    pub shared_w4a4_down: bool,

    /// 2026-09-25: `METRALE_DFLASH_CONTIG_ATTN=1`: the DFlash head's paged block layer takes
    /// `forward_block_layer_attention_contig`.
    pub dflash_contig_attn: bool,

    /// 2026-09-25: `METRALE_LORA_EAGER=1` or `true` (any case): decode and verify run without
    /// CUDA graphs while a LoRA pool is loaded (`model/trait_impl/decode_a.rs`, `decode_a2.rs`,
    /// `verify_b.rs`, `verify_c.rs`).
    pub lora_eager: bool,
    /// 2026-09-25: `METRALE_LORA_ROTATE=1` or `true` (any case): permits runtime adapter
    /// rotation and swap (`TransformerModel::lora_rotatable`, which `METRALE_LORA_PEER` also
    /// sets).
    pub lora_rotate: bool,

    /// 2026-09-25: `METRALE_K4_DIAG=1`: synchronise after each named phase of the batched GDN
    /// decode so an illegal access is attributed to its phase; skipped under graph capture
    /// (`qwen3_ssm/trait_decode_batched.rs`).
    pub k4_diag: bool,
    /// 2026-09-25: `METRALE_DIAG_GEMMA4=1` or `true` (any case): per-layer hidden-state norm dumps
    /// on the Gemma-4 attention decode path; off under graph capture (`decode_inner.rs`).
    pub gemma4_diag: bool,
    /// 2026-09-25: `METRALE_DFLASH_DEBUG_DUMP_FULL=1`: the model-side half of the DFlash full
    /// dump, which writes the sequence's tokens once per model (`model-engine` `impl_b3.rs`).
    ///
    /// The drafter half is `DFlashLevers::debug_dump_full` in metrale-model-arch's
    /// `dflash_head/levers.rs`, which resolves the same variable: `TransformerModel` holds the
    /// drafter only as a `dyn DraftProposer`. The test
    /// `the_two_halves_of_the_dflash_dump_name_the_same_flag` checks that both resolutions name
    /// one variable.
    pub dflash_debug_dump_full: bool,
    /// 2026-09-25: `METRALE_MTP_DEBUG_NORMS=1`: per-stage norm dumps in the MTP drafter's
    /// `forward_one`.
    pub mtp_debug_norms: bool,
    /// 2026-09-25: `METRALE_MTP_CHAIN_POSTNORM=1`: draft `j > 0` of an MTP propose chain reads the
    /// drafter's final-normed hidden (`norm_output`) instead of its pre-norm residual stream
    /// (`hidden_states`). Both chains read it through `MtpHead::chain_hidden`.
    pub mtp_chain_postnorm: bool,
    /// 2026-09-25: `METRALE_MTP_TARGET_POSTNORM=1`: drafter rows built from a target hidden (the
    /// first draft, drafter prefill, the exact-KV catch-up) go through the target's final norm
    /// before `pre_fc_norm_hidden` (`MtpHead::target_postnorm_rows`). No effect when the head holds
    /// no target final-norm weight.
    pub mtp_target_postnorm: bool,
    /// 2026-09-25: `METRALE_MTP_KV_EXACT=1`: after a verify, the MTP drafter keeps the KV row built
    /// from the verified input token and trims only the chain rows (`draft_proposer.rs`); the
    /// model then appends one catch-up row per accepted draft from the verify forward's stashed
    /// hidden (`model/trait_impl/speculative_mtp.rs`).
    pub mtp_kv_exact: bool,
    /// 2026-09-25: `METRALE_FP8_MOE_GROUPED_DECODE=1`: the target model's multi-row FP8 MoE decode
    /// runs through the cross-row grouped kernels (`FfnComponent::fp8_grouped_decode_ok`). The MTP
    /// drafter calls `MoeLayer::fp8_grouped_decode_ok` directly and does not read this lever;
    /// `METRALE_NO_FP8_MOE_GROUPED_DECODE` (presence) disables both
    /// (`moe/forward_fp8_grouped_decode.rs`).
    pub moe_fp8_grouped_decode_target: bool,
    /// 2026-09-26: `METRALE_FP8_ATTN_M32=1`: attention layers load the 32-row tensor-core twin
    /// `w8a16_gemm_pipelined_m32` at construction (`qwen3_attention/init_proj_kernels.rs`),
    /// which batches of more than 16 rows can then take. Without it the handle is zero.
    pub fp8_attn_m32: bool,
    /// 2026-09-25: `METRALE_MTP_DRAFT_CONF=<t>` (`speculative::draft_conf_tau`): the confidence
    /// floor for submitting MTP drafts to verification, clamped to `[0.0, 0.99]`. Unset or
    /// unparseable is `0.0`, which disables it. Below the floor the drafts are dropped and the
    /// drafter rows trimmed (`model-engine` `impl_b3.rs`). `MtpHead::last_confidence` reads the
    /// variable itself (`draft_proposer.rs`).
    pub draft_conf_tau: f32,
    /// 2026-09-25: `METRALE_SSM_SAVE_DUMP` (presence): the scratch/SSM-state fingerprint probe on a
    /// sequence's first decode step, which also runs that step without graphs
    /// (`model/trait_impl/decode_a.rs`).
    pub ssm_save_dump: bool,

    // 2026-09-25: The batched-decode levers, read in `model/trait_impl/decode_a2.rs`.
    // `mla_perseq_fallback` and `conc_hsd` accept `1` or `true` (case-sensitive); the other
    // three accept only `1`.
    /// 2026-09-25: `METRALE_MLA_PERSEQ_FALLBACK`: on an MLA model, decode the batch one sequence
    /// at a time.
    pub mla_perseq_fallback: bool,
    /// 2026-09-25: `METRALE_HC_PERSEQ_DECODE=1`: on a model with `hc_mult > 0`, decode the batch
    /// one sequence at a time, as an active QSA indexer already does. Decided before the
    /// expert-parallel branch, so an EP batch takes the per-sequence path too.
    pub hc_perseq_decode: bool,
    /// 2026-09-25: `METRALE_DECODE_BATCH_LOG=1`: log the batch's SSM slot vector each step.
    pub decode_batch_log: bool,
    /// 2026-09-25: `METRALE_MS_PROFILE=1`: multi-sequence decode profiling. It disables the
    /// batched decode graph so the profiling syncs are legal. A different variable from
    /// [`Self::ssm_ms_profile`] (`METRALE_SSM_MS_PROFILE`).
    pub ms_profile: bool,
    /// 2026-09-25: `METRALE_CONC_HSD`: per-row hidden-state dumps after the embed and after each
    /// layer, for batches of two or more rows without a communicator.
    pub conc_hsd: bool,

    /// 2026-09-25: `METRALE_EP_GRAPHS=1` or `true` (case-sensitive): allow decode CUDA-graph
    /// capture when a communicator is present (`model/trait_impl/decode_a.rs`).
    pub ep_graphs: bool,
    /// 2026-09-25: `METRALE_GDN_DECODE_GRAPH=1` or `true` (case-sensitive): the same permission as
    /// [`Self::ep_graphs`]; `decode_a.rs` ORs the two.
    pub gdn_decode_graph: bool,

    /// 2026-09-25: `METRALE_BF16_TC_PROJ` (presence): attention prefill projections use
    /// `w4a16_gemm_n128_m128_bf16` when that kernel is loaded (`qwen3_attention/prefill_weights.rs`).
    pub bf16_tc_proj: bool,
    /// 2026-09-25: `TQ_PLUS_WEIGHT_ROTATION=1` or `true` (any case): the weight loader rotates the
    /// Q/K/V projection weights at load (head_dim 128 only, `load_layers/attention_arms.rs`), and
    /// the attention paths skip their runtime rotation (`if !weight_pre_rotated`, e.g.
    /// `decode/write_kv_cache.rs`).
    pub weight_pre_rotated: bool,

    /// 2026-09-25: `METRALE_SSM_MS_PROFILE=1`: per-phase timing of the multi-sequence SSM decode;
    /// off under graph capture (`qwen3_ssm/trait_decode_multi_seq.rs`).
    pub ssm_ms_profile: bool,
    /// 2026-09-25: `METRALE_SSM_DETAIL_PROFILE=1`: per-sub-step timing inside the batched SSM
    /// recurrence; off under graph capture (`trait_decode_multi_seq/ssm_batched.rs`).
    pub ssm_detail_profile: bool,
    /// 2026-09-25: On unless `METRALE_SSM_GEMV_BATCH4=0`. For n <= 16 the SSM projections use the
    /// contiguous-batch GEMV tier (batch4 for n <= 4, batch16 above) when its kernel resolved
    /// (`trait_decode_multi_seq/ssm_batched_proj.rs`).
    pub ssm_gemv_batch4: bool,
    /// 2026-09-25: `METRALE_GDN_FUSED_CONV=1`: the fused conv+GDN+norm decode kernel when
    /// `nv == 2 * nk`, the K and V head dims are 128 and the kernel resolved
    /// (`trait_decode_multi_seq/ssm_batched_recurrent.rs`).
    pub gdn_fused_conv: bool,
    /// 2026-09-25: `METRALE_MOE_LEGACY_PERTOKEN_DECODE=1`: the multi-sequence MoE decode takes the
    /// per-token path instead of the token-major one. The call site branches on
    /// `!moe_legacy_pertoken_decode` (`qwen3_ssm/trait_decode_multi_seq.rs`).
    pub moe_legacy_pertoken_decode: bool,
    /// 2026-09-25: The configured max decode batch. Not from the environment: `from_values` sets
    /// 1, and `TransformerModel::new` overwrites it with `max_batch_size` (at least 1). The paged
    /// decode attention pins its split-K split count to it (`decode/attention_forward.rs`), so a
    /// sequence's attention reduction does not depend on how many sequences share its batch. It
    /// also bounds the verify WY-table stash (`model/trait_impl/gdn_woa.rs`).
    pub max_decode_seqs: u32,
    /// 2026-09-25: `METRALE_MTP_SHADOW_TOPK=k` (`speculative::shadow_topk`): unset or unparseable
    /// is 0 (off), and it is capped at 8. The MTP drafter copies its logits to the host and logs
    /// its top-k candidates (`mtp_head/forward.rs`); the server's `verify_k3_step.rs` and
    /// `verify_k4_step.rs` also read it.
    pub shadow_topk: usize,
    /// 2026-09-25: `METRALE_KV_POISON=1`: fresh KV blocks are filled with `0xFF`
    /// (`PagedKvCache::poison_block`) instead of zeroed (`model/block_mgmt.rs`). Diagnostic.
    pub kv_poison: bool,
    /// 2026-09-25: The MTP drafter context policy from `drafter_context::resolve_from_env`
    /// (`METRALE_NO_MTP_DRAFTER_CONTEXT`, `METRALE_MTP_DRAFTER_CONTEXT_PREFILL_ONLY_UNSAFE`). One
    /// value, because its `carry` implies its `prefill`.
    pub drafter: crate::drafter_context::DrafterContext,
}

/// 2026-09-25: How the levers are read: `from_values` and the `get`, `from_env` and `defaults`
/// constructors.
#[path = "model_levers_resolve.rs"]
mod resolve;

#[cfg(test)]
#[path = "model_levers_tests.rs"]
mod tests;

/// 2026-09-25: The source-level test of which hot-path files may read the environment. Declared
/// here rather than in `tests` because it scans files outside this module.
#[cfg(test)]
#[path = "hot_path_env_guards.rs"]
mod hot_path_env_guards;
