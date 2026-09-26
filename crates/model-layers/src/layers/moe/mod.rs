// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The MoE feed-forward block `MoeLayer`: router, routed experts, optional
//! shared expert, and the kernel handles and pointer tables its forward paths use.
//!
//! Owner: model-layers (MoE).
//! Invariants:
//! - Construction (`new_with_hash`) fails unless `num_experts > 0`,
//!   `num_experts_per_tok <= num_experts`, and both are within the sigmoid
//!   routing kernels' fixed bounds.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use crate::layer::ForwardContext;
use crate::layers::ops;
use crate::weight_map::{DenseWeight, Fp8ExpertWeight, MoeWeights, QuantizedWeight};

/// 2026-09-25: The MoE feed-forward block. It is a component that layers call for
/// their FFN step; it does not implement `TransformerLayer`.
#[allow(dead_code)]
pub struct MoeLayer {
    pub weights: MoeWeights,
    /// 2026-09-25: Quant format of the routed experts. `new_with_hash` sets `Nvfp4`;
    /// the DeepSeek-V4 loader sets the detected format, and `Mxfp4E8m0` makes
    /// `e8m0_or` and the prefill dispatch pick the `_e8m0` kernels.
    #[allow(dead_code)]
    pub experts_scale_kind: crate::weight_map::WeightQuantFormat,
    /// 2026-09-25: Quant format of the shared expert, detected separately from the
    /// routed experts. `new_with_hash` sets `Nvfp4`. With E8M0 routed experts,
    /// the dual-format decode paths check it with `WeightQuantFormat::expect`.
    #[allow(dead_code)]
    pub shared_experts_scale_kind: crate::weight_map::WeightQuantFormat,
    // 2026-09-25: NVFP4 copy of the router gate. When set, the router GEMM runs on
    // it instead of the BF16 `weights.gate`.
    gate_nvfp4: Option<QuantizedWeight>,
    /// 2026-09-25: RMS norm applied to the expert input after routing; the router reads
    /// its own input (`router_input`). Only the Gemma-4 loader sets it.
    pub pre_expert_norm: Option<crate::weight_map::DenseWeight>,
    pre_expert_norm_k: metrale_gpu_runtime::gpu::KernelHandle,
    dense_gemv: KernelHandle,
    w4a16_gemv: KernelHandle,
    w4a16_gemv_sw: KernelHandle,
    w4a16_gemm: KernelHandle,
    dense_gemm: KernelHandle,
    /// 2026-09-25: `dense_gemm_bf16_router`, read only by `router_gate_gemm_dense`,
    /// which uses `dense_gemm` instead when this handle is 0.
    dense_gemm_router: KernelHandle,
    dense_gemm_pipelined: KernelHandle,
    /// 2026-09-25: FP32-output router GEMM for the `fp32_gate` lever
    /// (`METRALE_FP32_GATE`). 0 when the target lacks it; the gate then stays BF16.
    dense_gemm_f32out: KernelHandle,
    /// 2026-09-25: FP32-in/FP32-out router GEMM for the `fp32_routing` lever
    /// (`METRALE_FP32_ROUTING`); `fp32_routing_active` is false while it is 0.
    dense_gemm_f32in: KernelHandle,
    moe_topk_f32: KernelHandle,
    moe_expert_gate_up_shared: KernelHandle,
    moe_expert_silu_down_shared: KernelHandle,
    moe_topk: KernelHandle,
    moe_weighted_sum_blend: KernelHandle,
    residual_add: KernelHandle,
    moe_topk_batched: KernelHandle,
    moe_expert_gate_up_shared_batch2: KernelHandle,
    moe_expert_silu_down_shared_batch2: KernelHandle,
    moe_weighted_sum_blend_batch2: KernelHandle,
    w4a16_gemv_batch2: KernelHandle,
    moe_expert_gate_up_shared_batch3: KernelHandle,
    moe_expert_silu_down_shared_batch3: KernelHandle,
    moe_weighted_sum_blend_batch3: KernelHandle,
    w4a16_gemv_batch3: KernelHandle,
    // 2026-09-25: Token-major kernels for `forward_token_major_decode`. The
    // `atomic_c4` pair serves `forward_atomic_c4_decode`
    // (`METRALE_MOE_ATOMIC_C4_DECODE=1`).
    moe_expert_gate_up_shared_token_major: KernelHandle,
    moe_expert_silu_down_shared_token_major: KernelHandle,
    moe_weighted_sum_blend_token_major: KernelHandle,
    moe_decode_atomic_c4_silu_down_accum_k: KernelHandle,
    moe_decode_atomic_c4_finalize_k: KernelHandle,
    moe_sort_by_expert: KernelHandle,
    moe_sorted_gate_up: KernelHandle,
    moe_sorted_silu_down: KernelHandle,
    moe_grouped_gemm: KernelHandle,
    /// 2026-09-25: Wider-K twin of `moe_grouped_gemm`. Loaded only when
    /// `METRALE_MOE_GROUPED_K32=1` and the target ships it; `grouped_gemm_kernel`
    /// prefers it whenever it is non-zero.
    moe_grouped_gemm_k32: KernelHandle,
    /// 2026-09-25: M_TILE=256 twin of `moe_grouped_gemm`. Loaded only when
    /// `METRALE_MOE_GROUPED_M256=1` and the target ships it; used whenever it is
    /// non-zero.
    moe_grouped_gemm_m256: KernelHandle,
    moe_silu_mul: KernelHandle,
    /// 2026-09-25: Activation kernel for the sorted/unfused paths: `moe_silu_mul`, or
    /// `gelu_mul` after `set_gelu_activation`.
    moe_act_mul: KernelHandle,
    /// 2026-09-25: Set by `set_gelu_activation`. The fused SiLU+FP8-quant kernel is used
    /// only while it is false (`fused_silu_quant_ok`).
    gelu_activation: bool,
    moe_unpermute_reduce: KernelHandle,
    moe_batched_blend: KernelHandle,
    gate_ptrs: ExpertPtrTable,
    up_ptrs: ExpertPtrTable,
    down_ptrs: ExpertPtrTable,
    /// 2026-09-25: Transposed-layout pointer tables; `None` until a transpose pass or
    /// `set_down_transpose_scratch` fills them.
    gate_ptrs_t: Option<ExpertPtrTable>,
    up_ptrs_t: Option<ExpertPtrTable>,
    down_ptrs_t: Option<ExpertPtrTable>,
    /// 2026-09-25: Host-side per-expert packed/SFB pointers and scale2 for the CUTLASS
    /// grouped NVFP4 path, built once by `build_cutlass_grouped_sfb`. The grouped
    /// entry reads them on the host; they live on the layer so they are dropped
    /// with the model. `None` means that path is unavailable.
    cutlass_grouped_host: Option<ops::MoeCutlassHostTables>,
    /// 2026-09-25: Owns the per-expert SFB buffers that `cutlass_grouped_host` points at.
    _cutlass_sfb_owned: Vec<DevicePtr>,
    /// 2026-09-25: Down-projection transpose scratch shared by all MoE layers, wired by
    /// `set_down_transpose_scratch`. `transpose_down_into_scratch` refills it at
    /// the start of each layer's prefill, and `down_ptrs_t` then points into it.
    /// `None` unless that scratch is wired.
    down_t_scratch_packed: Option<DevicePtr>,
    down_t_scratch_scale: Option<DevicePtr>,
    moe_transpose_u8_batched_k: KernelHandle,
    moe_expert_gate_up_shared_t_k: KernelHandle,
    moe_expert_silu_down_shared_t_k: KernelHandle,
    // 2026-09-25: Decode variants for E8M0-scaled routed experts with an NVFP4 shared
    // expert. 0 on targets that do not ship them.
    moe_expert_gate_up_shared_t_e8m0_k: KernelHandle,
    moe_expert_silu_down_shared_t_e8m0_k: KernelHandle,
    moe_topk_sqrtsoftplus_k: KernelHandle,
    moe_topk_sqrtsoftplus_batched_k: KernelHandle,
    moe_hash_route_k: KernelHandle,
    moe_hash_route_batched_k: KernelHandle,
    /// 2026-09-25: Router logit width, `num_experts + zero_expert_num`. It equals
    /// `num_experts` unless the model has zero-computation experts.
    pub(crate) router_logits_n: u32,
    moe_topk_softmax_bias_k: KernelHandle,
    moe_topk_softmax_bias_batched_k: KernelHandle,
    moe_zero_expert_add_k: KernelHandle,
    /// 2026-09-25: Per-token folded zero-expert weight (f32), written by the
    /// softmax+bias router kernels. Allocated once, for 16384 tokens.
    zero_accum_dev: DevicePtr,
    /// 2026-09-25: Static hash-routing table `[vocab_size, top_k]` i64. `Some` marks a
    /// hash-routed layer: expert selection reads this table instead of the
    /// gate's top-k.
    tid2eid_dev: Option<DevicePtr>,
    moe_expert_gate_up_shared_batch2_t_k: KernelHandle,
    moe_expert_silu_down_shared_batch2_t_k: KernelHandle,
    moe_expert_gate_up_shared_batch3_t_k: KernelHandle,
    moe_expert_silu_down_shared_batch3_t_k: KernelHandle,
    moe_expert_gate_up_shared_fp8_t_k: KernelHandle,
    moe_expert_silu_down_shared_fp8_t_k: KernelHandle,
    moe_expert_gate_up_shared_fp8_batch2_t_k: KernelHandle,
    moe_expert_silu_down_shared_fp8_batch2_t_k: KernelHandle,
    moe_expert_gate_up_shared_fp8_batch3_t_k: KernelHandle,
    moe_expert_silu_down_shared_fp8_batch3_t_k: KernelHandle,
    /// 2026-09-25: `METRALE_UNIFIED_MOE_LAYOUT` (`1` or `true`), read at construction.
    /// With all three transposed tables built and no down scratch, decode runs
    /// the `moe_expert_*_shared_t` kernels unless `hybrid_layout` is also set.
    unified_layout: bool,
    /// 2026-09-25: `METRALE_NVFP4_GATE_UP_M128` (`1` or `true`). Prefill's fused
    /// gate+up uses the M=128 kernel when this is set and
    /// `moe_fused_gate_up_t_k64_m128` is non-zero, else the M=64 kernel.
    nvfp4_gate_up_m128: bool,
    /// 2026-09-25: `METRALE_HOLO_MOE_GATEUP_FP4` (`1` or `true`). Prefill's fused
    /// gate+up runs `moe_fused_gate_up_t_k64_fp4` over the existing
    /// `gate_ptrs_t`/`up_ptrs_t` tables when this is set and the handle is non-zero.
    gateup_fp4: bool,
    /// 2026-09-25: `METRALE_HOLO_MOE_DOWN_FP4` (`1` or `true`): the same choice for the
    /// prefill down projection, with `moe_down_t_k64_fp4` over `down_ptrs_t`.
    down_fp4: bool,
    /// 2026-09-25: `METRALE_HYBRID_MOE_LAYOUT` (`1` or `true`), read at construction.
    /// Prefill uses the transposed tables (`use_t_layout_for_prefill`) while
    /// decode keeps the original `[N, K/2]` tables, even when `unified_layout`
    /// is also set.
    hybrid_layout: bool,
    shared_gate_t: Option<QuantizedWeight>,
    shared_up_t: Option<QuantizedWeight>,
    shared_down_t: Option<QuantizedWeight>,
    moe_grouped_gemm_t: KernelHandle,
    moe_grouped_gemm_t_k64: KernelHandle,
    moe_fused_gate_up_t: KernelHandle,
    moe_fused_gate_up_t_k64: KernelHandle,
    // 2026-09-25: E8M0-scale prefill variants of the routed-expert W4A16 GEMMs. 0 on
    // targets that do not ship them.
    moe_grouped_gemm_e8m0: KernelHandle,
    moe_grouped_gemm_t_e8m0: KernelHandle,
    moe_grouped_gemm_t_k64_e8m0: KernelHandle,
    moe_fused_gate_up_t_e8m0: KernelHandle,
    moe_fused_gate_up_t_k64_e8m0: KernelHandle,
    /// 2026-09-25: M=128 variant of the K64 fused gate+up kernel; 0 on targets that do
    /// not ship it.
    moe_fused_gate_up_t_k64_m128: KernelHandle,
    /// 2026-09-25: Block-scaled FP4 variant of the K64 fused gate+up kernel, launched
    /// through the same wrapper as `moe_fused_gate_up_t_k64`; 0 on targets that
    /// do not ship it.
    moe_fused_gate_up_t_k64_fp4: KernelHandle,
    moe_fp8_grouped_gemm_t: KernelHandle,
    w4a16_gemm_t: KernelHandle,
    bf16_to_fp8_k: KernelHandle,
    /// 2026-09-25: FP8 copies of the NVFP4 router gate and shared expert, made by
    /// `predequant_for_prefill`. The shared-expert copies are skipped when a BF16
    /// shared expert is installed.
    gate_fp8: Option<DevicePtr>,
    shared_gate_fp8: Option<DevicePtr>,
    shared_up_fp8: Option<DevicePtr>,
    shared_down_fp8: Option<DevicePtr>,
    fp8_gemm_k: KernelHandle,
    /// 2026-09-25: Auxiliary stream. Nothing launches on it today: the prefill
    /// shared-expert overlap is switched off (`use_overlap = false` in
    /// `forward_prefill`), and `kick_off_lazy_transpose`, the down transpose that
    /// would run on it, has no caller.
    prefill_stream: u64,
    /// 2026-09-25: `event_a` marks the input ready for the auxiliary stream; `event_b`
    /// marks the auxiliary work (the overlapped shared expert, or
    /// `kick_off_lazy_transpose`'s transpose) done. Neither is recorded while
    /// both of those are off.
    event_a: u64,
    event_b: u64,
    /// 2026-09-25: `[num_experts]` routing correction bias, from
    /// `MoeWeights.correction_bias`. `Some` selects bias-corrected routing
    /// (sigmoid, or sqrtsoftplus when the config says so); `None` selects softmax.
    correction_bias_dev: Option<DevicePtr>,
    moe_topk_sigmoid_k: KernelHandle,
    moe_topk_sigmoid_batched_k: KernelHandle,
    moe_expert_gate_up_shared_fp8: KernelHandle,
    moe_expert_silu_down_shared_fp8: KernelHandle,
    moe_expert_gate_up_shared_fp8_batch2: KernelHandle,
    moe_expert_silu_down_shared_fp8_batch2: KernelHandle,
    moe_weighted_sum_blend_fp8_batch2: KernelHandle,
    moe_expert_gate_up_shared_fp8_batch3: KernelHandle,
    moe_expert_silu_down_shared_fp8_batch3: KernelHandle,
    moe_weighted_sum_blend_fp8_batch3: KernelHandle,
    // 2026-09-25: Grouped FP8 decode kernels from `GroupedKernels::resolve`; 0 on
    // targets that do not ship them.
    moe_expert_gate_up_shared_fp8_grouped_k: KernelHandle,
    moe_expert_silu_down_shared_fp8_grouped_k: KernelHandle,
    moe_weighted_sum_blend_fp8_grouped_k: KernelHandle,
    moe_fp8_grouped_compact_k: KernelHandle,
    // 2026-09-25: Routed-expert FP8 grouped GEMM over the tile work-list that
    // `moe_build_tile_worklist` builds; 0 on targets that do not ship it.
    moe_fp8_grouped_gemm_k: KernelHandle,
    moe_build_tile_worklist_k: KernelHandle,
    // 2026-09-25: The Hopper short-prefill M16 expert GEMM and bucket builder
    // (`adaptive_fp8.rs`), and the SM count their grids are sized from; 0 when the
    // target lacks either module.
    moe_w8a8_m16_k: KernelHandle,
    moe_bucket_builder_k: KernelHandle,
    moe_adaptive_sms: u32,
    // 2026-09-25: W8A8 grouped GEMM with an FP32 epilogue over per-token-quantized
    // activations. The FP8 prefill uses it when `fp8_blockscaled_prefill` is on
    // (the default; `METRALE_FP8_SINGLE_SCALE=1` turns it off) and this handle
    // and the per-token quant kernel are present.
    moe_w8a8_grouped_gemm_k: KernelHandle,
    // 2026-09-25: Work-list variant of the W8A8 grouped GEMM. Preferred when it and
    // `moe_build_tile_worklist_k` are non-zero; otherwise the W8A8 path runs
    // `moe_w8a8_grouped_gemm_k`.
    moe_w8a8_grouped_gemm_pm4_k: KernelHandle,
    per_token_group_quant_fp8_k: ops::Fp8ActQuant,
    /// 2026-09-25: Fused SiLU·mul + per-token-group FP8 quant for the W8A8 prefill
    /// down inputs. When `fused_silu_quant_ok` is false (handle 0, GeGLU, or an
    /// unsupported K) the path runs `silu_mul` then `per_token_group_quant_fp8`.
    silu_mul_quant_fp8_k: KernelHandle,
    // 2026-09-25: Dense W8A8 GEMM for the shared expert on the W8A8 FP8 prefill path.
    fp8_gemm_t_blockscaled_k: KernelHandle,
    // 2026-09-25: BF16 grouped GEMM for experts installed with `set_bf16_experts`.
    // Prefill of more than 64 tokens uses it when it is non-zero, except on HIP
    // builds; otherwise BF16 experts run `forward_batched`.
    moe_bf16_grouped_gemm_k: KernelHandle,
    moe_expert_gate_up_shared_bf16_k: KernelHandle,
    moe_expert_silu_down_shared_bf16_k: KernelHandle,
    // 2026-09-25: Two-row BF16 decode kernels. `forward_k2` uses them only when both
    // are non-zero, BF16 experts are installed and expert parallelism is off.
    moe_expert_gate_up_shared_bf16_batch2_k: KernelHandle,
    moe_expert_silu_down_shared_bf16_batch2_k: KernelHandle,
    w8a16_gemm_k: KernelHandle,
    w8a16_gemm_pipelined_k: KernelHandle,
    moe_gate_topk_fused_k: KernelHandle,
    // 2026-09-25: FP8 expert pointer tables; `None` until `set_fp8_experts`.
    fp8_gate_weight_ptrs: Option<Fp8ExpertPtrTable>,
    fp8_up_weight_ptrs: Option<Fp8ExpertPtrTable>,
    fp8_down_weight_ptrs: Option<Fp8ExpertPtrTable>,
    // 2026-09-25: BF16 expert pointer tables; `None` until `set_bf16_experts`. When
    // set, prefill takes the BF16 expert path before any FP8 or NVFP4 one.
    bf16_gate_weight_ptrs: Option<DevicePtr>,
    bf16_up_weight_ptrs: Option<DevicePtr>,
    bf16_down_weight_ptrs: Option<DevicePtr>,
    // 2026-09-25: BF16 shared expert, installed independently of the routed experts'
    // precision (`set_bf16_shared_expert`).
    bf16_shared_expert: Option<Bf16SharedExpert>,
    // 2026-09-25: FP8 shared expert; `None` until `set_fp8_experts`.
    fp8_shared_expert: Option<Fp8ExpertWeight>,
    /// 2026-09-25: `moe_w4a16_down_t_k64_fp4`; 0 on targets that do not ship it.
    pub(crate) moe_down_t_k64_fp4: KernelHandle,
    /// 2026-09-25: `moe_permute_tokens`. Loaded, but no forward path reads it.
    #[allow(dead_code)]
    pub(crate) moe_permute_tokens_k: KernelHandle,
    // 2026-09-25: True when this layer's index is in `config.dflash_capture_layers`
    // (set by the qwen35 loader). With the `frankenstein_decode_via_prefill`
    // lever (`METRALE_FRANKENSTEIN_DECODE_VIA_PREFILL`), `forward` runs this
    // layer's single-token decode through `forward_prefill` with one row.
    pub is_dflash_capture_layer: bool,
    /// 2026-09-25: This layer's installed MoE LoRA (router and routed-expert deltas plus
    /// apply scratch), set by [`MoeLayer::set_lora_weights`]; `None` means no
    /// adapter. Decode paths that call `reject_decode_lora` refuse a batch
    /// routed to the adapter.
    pub(crate) lora: Option<MoeLoraWeights>,
}

impl MoeLayer {
    /// 2026-09-25: Returns `e8m0` when the routed experts are `Mxfp4E8m0`, else `nvfp4`.
    /// Panics when `e8m0` is selected but its handle is 0, so E8M0 weights never
    /// run through an NVFP4 kernel.
    #[inline]
    fn e8m0_or(
        &self,
        nvfp4: metrale_gpu_runtime::gpu::KernelHandle,
        e8m0: metrale_gpu_runtime::gpu::KernelHandle,
        site: &str,
    ) -> metrale_gpu_runtime::gpu::KernelHandle {
        if self.experts_scale_kind == crate::weight_map::WeightQuantFormat::Mxfp4E8m0 {
            assert!(
                e8m0.0 != 0,
                "ARM-2 Phase-K: routed experts tagged Mxfp4E8m0 at {site}, but the \
                 _e8m0 kernel handle is unresolved (not compiled into this target)."
            );
            e8m0
        } else {
            nvfp4
        }
    }
}

mod adaptive_fp8;
mod tables;
pub(crate) use tables::{Bf16SharedExpert, ExpertPtrTable, Fp8ExpertPtrTable};

mod dump;
mod forward;
mod lora;
mod lora_gateup;
mod lora_router;
pub(crate) use lora::MoeLoraWeights;
mod forward_atomic_c4;
mod forward_batched;
mod forward_batched_gate;
mod forward_ep;
mod forward_fp8_grouped_decode;
pub use forward_fp8_grouped_decode::fp8_grouped_decode_shape_ok;
mod forward_k2;
mod forward_k3;
mod forward_phase;
mod forward_prefill;
mod forward_prefill_bf16;
mod forward_prefill_fp8;
mod forward_prefill_phase;
mod forward_prefill_routed;
mod forward_prefill_router;
mod forward_token_major;
mod helpers_a;
mod helpers_b;
mod helpers_c;
mod init;
#[cfg(test)]
mod mod_tests;
mod ptr_table_build;
mod union_stats;
pub(crate) use ptr_table_build::*;
