// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Dense gated FFN layer, `down(act(gate(x)) * up(x))` with SiLU or GELU, for
//! single-token decode, small verify batches and prefill.
//!
//! Owner: model-layers (dense FFN).
//! Invariants:
//! - On success, every forward entry point leaves its `[rows, hidden]` BF16 output in
//!   `ctx.buffers.moe_output()`; decode `forward` also returns that pointer.
//! - `forward`, `forward_k2`, `forward_k3` and `forward_prefill` serve an installed
//!   packed-Q2 overlay first, then native FP8, then BF16, and use the NVFP4 weights only
//!   when none is installed.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

// 2026-09-26: Imported so the moved bodies in `dense_ffn_init.rs` and `dense_ffn_overlays.rs` keep
// calling these as `super::`.
use super::{k64_kernel, tgemm_kernel, try_kernel, try_target_kernel, w4a16_v2_kernel};
use crate::layer::ForwardContext;
use crate::layers::ops;
use crate::layers::w4a16_gemv_tiers::W4a16BatchmTiers;
use crate::weight_map::Fp8Weight;

/// 2026-09-26: The public weight sets and `FfnActivation`.
#[path = "dense_ffn_weights.rs"]
mod weights;
pub use weights::{
    DenseFfnWeights, DenseFfnWeightsBf16, DenseFfnWeightsFp8, DenseFfnWeightsQ2, FfnActivation,
};

/// 2026-09-25: Int8 copy of one NVFP4 projection for the `int8_gemm_faith2` prefill arms: `w_i8`
/// is `[N, K]` int8 and `w_scale` is `[N, K/32]` F32. Built once by `ensure_int8_weight`.
#[derive(Debug, Clone, Copy)]
struct Int8Weight {
    w_i8: DevicePtr,
    w_scale: DevicePtr,
}

/// 2026-09-25: GGML `block_q4_K` copy of one NVFP4 projection for the `METRALE_FFN_MMQ` prefill
/// arm, built once by `ensure_q4k_weight`.
#[derive(Debug, Clone, Copy)]
struct Q4kWeight {
    w_q4k: DevicePtr,
}

/// 2026-09-25: `block_nvfp4` repack of one NVFP4 projection for the NVFP4 MMQ prefill arm, built
/// once by `ensure_nvfp4_mmq_weight`.
#[derive(Debug, Clone, Copy)]
struct Fp4MmqWeight {
    w: DevicePtr,
}

pub struct DenseFfnLayer {
    pub weights: DenseFfnWeights,
    activation: FfnActivation,
    w4a16_gemv: KernelHandle,
    /// 2026-09-25: Single-warp `w4a16_gemv_sw`. `KernelHandle(0)` on miss, which selects the base
    /// GEMV. `examples/w4a16_gemv_sw_microtest.rs` requires it to match `w4a16_gemv` bit for bit.
    w4a16_gemv_sw: KernelHandle,
    w4a16_gemv_dual: KernelHandle,
    w4a16_gemv_silu_input: KernelHandle,
    // 2026-09-25: Single-warp-per-output decode variants, used when `ModelLevers::gemv_sw` is on
    // (unless `METRALE_NO_GEMV_SW=1`) and the handle resolved; otherwise the 64-thread
    // kernels run.
    w4a16_gemv_dual_sw: KernelHandle,
    w4a16_gemv_silu_input_sw: KernelHandle,
    w4a16_gemv_dual_batch2: KernelHandle,
    w4a16_gemv_dual_batch3: KernelHandle,
    w4a16_gemv_batch2: KernelHandle,
    w4a16_gemv_batch3: KernelHandle,
    /// 2026-09-25: The `w4a16_gemv_batch{M}` tiers `forward_km` launches.
    /// `W4a16BatchmTiers::kernel` picks the tier for `m` and returns a zero handle when none serves
    /// it.
    w4a16_batchm: W4a16BatchmTiers,
    w4a16_gemm: KernelHandle,
    w4a16_gemm_t_m128_k: KernelHandle,
    // 2026-09-25: 8-warp variant of `w4a16_gemm_t_m128`, preferred over it when resolved.
    // `w4a16_v2_kernel` resolves it only under `METRALE_W4A16_VARIANT=v2` or `v3`.
    w4a16_gemm_t_m128_v2_k: KernelHandle,
    // 2026-09-25: BF16 tensor-core variant of `w4a16_gemm_t_m128`, used for prefill only when
    // `METRALE_BF16_TC_PREFILL` is set. `examples/w4a16_bf16_v2_microtest.rs` compares it with
    // the base `w4a16_gemm` by cosine, not bit for bit.
    w4a16_gemm_t_m128_bf16_k: KernelHandle,
    // 2026-09-25: Second BF16 tensor-core variant, chosen over `w4a16_gemm_t_m128_bf16` when it
    // resolved unless `METRALE_DISABLE_PREFILL_V2` is set. `examples/w4a16_bf16_v2_microtest.rs`
    // requires its output to equal that kernel's bit for bit.
    w4a16_gemm_t_m128_bf16_v2_k: KernelHandle,
    // 2026-09-25: `w4a16_gemm_t` (M tile 64; `tgemm_kernel` prefers the `_p3` variant).
    // `w4a16_prefill_gemm` uses it for m <= 64, and `METRALE_FP8_M64_PREFILL` sends every
    // prefill projection that has a transposed copy through it.
    w4a16_gemm_t_k: KernelHandle,
    // 2026-09-25: int8 prefill (`METRALE_INT8_PREFILL`, and the down projection under
    // `METRALE_FFN_MMQ`): `requant_w_nvfp4_int8` converts each NVFP4 weight once into the
    // `OnceLock`s below, and `requant_a_bf16_int8` converts the BF16 activation on every call
    // into the arena's `ffn_act_a` / `ffn_act_scale`. A zero `int8_faith2_k` disables both.
    int8_faith2_k: KernelHandle,
    // 2026-09-25: `int8_gemm_i32acc`, which replaces `int8_gemm_faith2` in the int8 arms when
    // `METRALE_INT8_FAITH5` is set.
    int8_faith5_k: KernelHandle,
    requant_w_int8_k: KernelHandle,
    requant_a_int8_k: KernelHandle,
    int8_gate: std::sync::OnceLock<Int8Weight>,
    int8_up: std::sync::OnceLock<Int8Weight>,
    int8_down: std::sync::OnceLock<Int8Weight>,
    // 2026-09-25: W4A4 prefill (`METRALE_FP4_PREFILL`): `w4a4_gemm` reads the NVFP4 weights as they
    // are, and `quantize_bf16_to_nvfp4` writes the activation into the arena's `ffn_act_a` /
    // `ffn_act_scale`. Taken only when both handles resolved.
    w4a4_gemm_k: KernelHandle,
    quantize_nvfp4_k: KernelHandle,
    // 2026-09-25: Q4_K MMQ prefill (`METRALE_FFN_MMQ`): each NVFP4 weight is dequantized to BF16
    // and quantized to `block_q4_K` once (`ensure_q4k_weight`), and the activation is
    // quantized to q8_1 on every call into the arena's `ffn_act_q8`. All four handles must
    // resolve.
    q4k_mmq_nc_k: KernelHandle,
    q4k_mmq_wc_k: KernelHandle,
    q4k_quant_act_k: KernelHandle,
    q4k_quant_w_k: KernelHandle,
    dequant_nvfp4_bf16_k: KernelHandle,
    q4k_gate: std::sync::OnceLock<Q4kWeight>,
    q4k_up: std::sync::OnceLock<Q4kWeight>,
    q4k_down: std::sync::OnceLock<Q4kWeight>,
    // 2026-09-25: NVFP4 MMQ prefill, on by default (`ModelLevers::ffn_nvfp4_mmq`, off with
    // `METRALE_NO_FFN_NVFP4_MMQ`) for SiLU layers without a LoRA adapter.
    // `finalize_nvfp4_mmq_load` repacks the weights to `block_nvfp4` at load, and the
    // activation is quantized on every call into the arena's `ffn_act_q8`. Each projection's
    // `weight_scale_2` is applied after the GEMM (`nvfp4_silu_mul_scaled`,
    // `nvfp4_silu_mul_quant` or `nvfp4_scale_bf16`).
    nvfp4_mmq_nc_k: KernelHandle,
    nvfp4_mmq_wc_k: KernelHandle,
    /// 2026-09-25: The 16-, 32- and 64-row MMQ tiles, chosen for m <= 16, 32 or 64
    /// (`mmq_small_tile_enabled`, `mmq_tile64_enabled`). A zero handle keeps the 128-row tile.
    nvfp4_mmq16_nc_k: KernelHandle,
    nvfp4_mmq16_wc_k: KernelHandle,
    nvfp4_mmq32_nc_k: KernelHandle,
    nvfp4_mmq32_wc_k: KernelHandle,
    nvfp4_mmq64_nc_k: KernelHandle,
    nvfp4_mmq64_wc_k: KernelHandle,
    nvfp4_quant_act_k: KernelHandle,
    nvfp4_repack_k: KernelHandle,
    nvfp4_silu_scaled_k: KernelHandle,
    nvfp4_silu_quant_k: KernelHandle,
    nvfp4_scale_k: KernelHandle,
    fp4mmq_gate: std::sync::OnceLock<Fp4MmqWeight>,
    fp4mmq_up: std::sync::OnceLock<Fp4MmqWeight>,
    fp4mmq_down: std::sync::OnceLock<Fp4MmqWeight>,
    // 2026-09-25: Deep-K variant of `w4a16_gemm_t` (`k64_kernel` prefers `_p3`).
    // `w4a16_prefill_gemm` uses it for m <= 64 when k >= `w4a16_k64_min_k()` and k % 64 == 0.
    w4a16_gemm_t_k64_k: KernelHandle,
    /// 2026-09-25: SiLU(gate)*up or GELU(gate)*up, depending on the activation.
    act_mul: KernelHandle,
    /// 2026-09-25: Set by `set_bf16_weights`. When present and no packed-Q2 or FP8 overlay is
    /// installed, every forward path uses the BF16 kernels.
    bf16_weights: Option<DenseFfnWeightsBf16>,
    dense_gemv_bf16_k: KernelHandle,
    dense_gemm_bf16_k: KernelHandle,
    // 2026-09-25: Tensor-core BF16 GEMM for BF16 and packed-Q2 prefill; a zero handle falls back to
    // `dense_gemm_bf16`. For BF16 prefill, cuBLASLt comes first when `METRALE_CUBLAS_GEMM`
    // arms the `ffn` family.
    dense_gemm_tc_k: KernelHandle,
    /// 2026-09-25: Set by `set_fp8_weights`. When present (and no packed-Q2 overlay), every forward
    /// path runs FP8 kernels: `forward_k2`, `forward_k3` and `forward_km` hand the layer to
    /// `forward_prefill`.
    fp8_weights: Option<DenseFfnWeightsFp8>,
    w8a16_gemv_k: KernelHandle,
    w8a16_gemm_k: KernelHandle,
    w8a16_gemv_batch4_k: KernelHandle,
    /// 2026-09-25: `w8a16_gemv_batch16` for the opt-in 5..=32-row tier; zero when the target lacks
    /// it. Rule: `batch16_decode::batch16_plan`.
    w8a16_gemv_batch16_k: KernelHandle,
    /// 2026-09-25: Whether the batch16 tier is armed (`METRALE_FFN_BATCH16=1`), copied from
    /// `ffn_batch16_enabled()` at construction. A field, so the dispatch tests can set either
    /// value without touching the process-wide `OnceLock`.
    batch16_enabled: bool,
    /// 2026-09-25: `w8a16_gemm_m16`, the tensor-core tier `w8_gemm!` tries ahead of batch16 at
    /// 5..=32 rows when `m16_tc` is on; zero when the target lacks the module. Rule:
    /// `m16_tc::m16_tc_plan`.
    w8a16_gemm_m16_k: KernelHandle,
    /// 2026-09-25: The 64-wide N tile twin, used when `METRALE_FFN_M16_TC_NTILE=64` and it
    /// resolved; otherwise the 32-wide kernel runs (`m16_tc::m16_tc_kernel`).
    w8a16_gemm_m16_n64_k: KernelHandle,
    /// 2026-09-25: `m16_tc::m16_tc_levers().ffn`, copied at construction: the target's `[defaults]
    /// ffn_m16_tc`, overridden by `METRALE_FFN_M16_TC`, or by `METRALE_M16_TC` when that is
    /// unset. A field for the same reason as `batch16_enabled`.
    m16_tc: bool,
    /// 2026-09-25: The CTA N width this layer asks for, 32 or 64.
    m16_tc_n_tile: u32,
    w8a16_gemm_pipelined_k: KernelHandle,
    // 2026-09-25: Fused FP8 decode GEMVs: gate+up in one launch, and `silu(gate)*up` computed
    // inside the down GEMV. `fp8_down::fp8_down_arm` decides which run.
    w8a16_gemv_dual_k: KernelHandle,
    w8a16_gemv_silu_input_k: KernelHandle,
    // 2026-09-25: Transposed-weight FP8 prefill GEMM. `forward_prefill_inner` passes no transposed
    // FP8 copy (`gate_t` / `up_t` / `down_t` are `None`), so its `w8_gemm!` arm never matches.
    w8a16_gemm_t_m128_k: KernelHandle,
    // 2026-09-25: The W8A8 prefill pair: per-token, 128-group FP8 activation quantization and the
    // block-scaled FP8 x FP8 GEMM. Rule: `w8a8_prefill::w8a8_prefill_selected`.
    per_token_group_quant_fp8_k: ops::Fp8ActQuant,
    fp8_gemm_t_blockscaled_k: KernelHandle,
    // 2026-09-25: Activation-scale layout adapter for the cuBLASLt arm of `w8a8_gemm`. When
    // `ops::cublas_scale_layout_kmajor()` holds, a zero handle makes that arm decline and the
    // in-tree GEMM run.
    fp8_act_scale_kmajor_k: KernelHandle,
    /// 2026-09-25: The `[2*inter, hidden]` FP8 gate+up weight installed by `set_fp8_gate_up_fused`;
    /// `None` unless the loader built it. The installer's contract is that `gate_proj` and
    /// `up_proj` are views into it.
    fp8_gate_up_fused: Option<Fp8Weight>,
    /// 2026-09-25: `gateup_fused::ffn_gateup_fused()` at construction: the target's `[defaults]
    /// ffn_gateup_fused`, overridden by `METRALE_FFN_GATEUP_FUSED`. It also decides whether
    /// `silu_mul_strided_k` is looked up.
    gateup_fused: bool,
    /// 2026-09-25: The strided SiLU·mul that reads the fused `[m, 2*inter]` output. Its source
    /// exists only under `kernels/hopper/common/`, so the handle is zero on other targets, and it
    /// is looked up only when `gateup_fused` is on.
    silu_mul_strided_k: KernelHandle,
    /// 2026-09-25: LoRA overlay for gate/up/down, set by `set_lora_weights` (NVFP4 layers only).
    /// `apply_lora_gate_up` runs after the gate/up projections and before the activation;
    /// `apply_lora_down` runs after the down projection. `forward` (split-SiLU path),
    /// `forward_k2`, `forward_k3`, `forward_km` and the NVFP4 part of `forward_prefill` call
    /// both. The `ModelLevers::decode_ffn_via_gemm` branch of `forward` calls neither.
    lora: Option<ops::lora_delta::LoraFfnWeights>,

    /// 2026-09-25: Set by `set_q2_weights`. Checked first by `forward`, `forward_k2`, `forward_k3`
    /// and `forward_prefill`.
    q2_weights: Option<DenseFfnWeightsQ2>,
    q2_0_gemv_k: KernelHandle,
    // 2026-09-25: Batched packed-Q2 decode GEMV, used by `forward_km_q2`.
    #[allow(dead_code)]
    q2_0_gemv_batchm_k: KernelHandle,
    // 2026-09-25: Packed-Q2 to BF16 dequant for packed-Q2 prefill when the Q2_0 MMQ arm is not
    // taken: each projection is dequantized into the arena's `q2_dequant_scratch` and run
    // through the BF16 GEMM.
    dequant_q2_0_gn_k: KernelHandle,
    // 2026-09-25: Q2_0 MMQ prefill (`METRALE_GGUF_NATIVE_Q2_MMQ=1`, group-128 weights) against a
    // q8_1 activation from `q4k_quant_act_k`. Looked up by `set_q2_weights`; a zero handle leaves
    // prefill on the dequant path.
    q2_0_mmq_nc_k: KernelHandle,
    q2_0_mmq_wc_k: KernelHandle,
}

/// 2026-09-25: The 16/32/64-row MMQ tiles: on unless `METRALE_NO_MMQ_SMALL_TILE` is exactly `1`.
/// Read once per process.
fn mmq_small_tile_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("METRALE_NO_MMQ_SMALL_TILE").as_deref() != Ok("1"))
}

/// 2026-09-25: The 64-row MMQ tile: on unless `METRALE_NO_MMQ_TILE64` is exactly `1`;
/// `METRALE_NO_MMQ_SMALL_TILE=1` also turns it off. Read once per process.
fn mmq_tile64_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("METRALE_NO_MMQ_TILE64").as_deref() != Ok("1"))
}
impl DenseFfnLayer {
    pub fn new(weights: DenseFfnWeights, gpu: &dyn GpuBackend) -> Result<Self> {
        Self::new_with_activation(weights, FfnActivation::SiLU, gpu)
    }

    /// 2026-09-25: Load-time setup for the Q4_K MMQ prefill arm. Returns at once for packed-Q2 or
    /// FP8 layers, when one of the four Q4_K kernels is missing, or when `METRALE_FFN_MMQ` is
    /// unset. Otherwise it builds the gate/up Q4_K copies and the down copy (int8 when
    /// `int8_gemm_faith2` and `requant_a_bf16_int8` resolved and `METRALE_FFN_MMQ_DOWN_Q4K` is
    /// unset, Q4_K otherwise), synchronizes `stream`, and frees the transposed `_t` copies.
    pub fn finalize_q4k_load(
        &mut self,
        gpu: &dyn GpuBackend,
        h: u32,
        inter: u32,
        stream: u64,
    ) -> Result<()> {
        // 2026-09-25: Packed-Q2 prefill never reaches the Q4_K arm (`forward_prefill_inner`), so
        // the copies would be unused.
        if self.q2_weights.is_some() {
            return Ok(());
        }
        // 2026-09-25: FP8 prefill returns before the Q4_K arm, so the copies would be unused.
        if self.fp8_weights.is_some() {
            return Ok(());
        }
        let q4k_active = self.q4k_mmq_nc_k.0 != 0
            && self.q4k_quant_act_k.0 != 0
            && self.q4k_quant_w_k.0 != 0
            && self.dequant_nvfp4_bf16_k.0 != 0
            && std::env::var_os("METRALE_FFN_MMQ").is_some();
        if !q4k_active {
            return Ok(());
        }
        // 2026-09-25: Build the copies before freeing the `_t` weights.
        self.build_q4k_gate_up(gpu, h, inter, stream)?;
        let down_faith2 = self.int8_faith2_k.0 != 0
            && self.requant_a_int8_k.0 != 0
            && std::env::var_os("METRALE_FFN_MMQ_DOWN_Q4K").is_none();
        self.finish_q4k_load(gpu, h, inter, stream, down_faith2)
    }

    /// 2026-09-25: Load-time setup for the NVFP4 MMQ prefill arm. Returns at once for packed-Q2 or
    /// FP8 layers, GELU layers, when a kernel it needs is missing, or when
    /// `METRALE_NO_FFN_NVFP4_MMQ` is set. Otherwise it repacks gate and up, and down unless
    /// `METRALE_NO_FFN_NVFP4_MMQ_DOWN` is set, synchronizes `stream`, and frees the transposed `_t`
    /// copies of the repacked projections.
    pub fn finalize_nvfp4_mmq_load(
        &mut self,
        gpu: &dyn GpuBackend,
        h: u32,
        inter: u32,
        stream: u64,
    ) -> Result<()> {
        // 2026-09-25: Packed-Q2 prefill never reaches the MMQ arm, so a repack would be unused.
        if self.q2_weights.is_some() {
            return Ok(());
        }
        // 2026-09-25: FP8 prefill returns before the MMQ arm, so a repack would be unused.
        if self.fp8_weights.is_some() {
            return Ok(());
        }
        let active = self.nvfp4_mmq_nc_k.0 != 0
            && self.nvfp4_quant_act_k.0 != 0
            && self.nvfp4_repack_k.0 != 0
            && self.nvfp4_silu_scaled_k.0 != 0
            && matches!(self.activation, FfnActivation::SiLU)
            && std::env::var_os("METRALE_NO_FFN_NVFP4_MMQ").is_none();
        if !active {
            return Ok(());
        }
        self.build_nvfp4_mmq_gate_up(gpu, h, inter, stream)?;
        let down_mmq = std::env::var_os("METRALE_NO_FFN_NVFP4_MMQ_DOWN").is_none();
        self.finish_nvfp4_mmq_load(gpu, h, inter, stream, down_mmq)
    }

    /// 2026-09-26: The BF16 branch of `forward_prefill_inner` (`dense_ffn_prefill.rs`). It reads
    /// `dispatch.cublas.ffn`, which `ops/dispatch_config_routing_tests.rs` allows only in this
    /// file, `dense_ffn_w8a8_prefill.rs` and `moe/`.
    fn prefill_bf16(
        &self,
        bf16w: &DenseFfnWeightsBf16,
        input: DevicePtr,
        ctx: &ForwardContext,
        m: u32,
        h: u32,
        inter: u32,
        gate_out: DevicePtr,
        up_out: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let tc = self.dense_gemm_tc_k.0 != 0;
        // 2026-09-25: BF16 prefill GEMM: cuBLASLt when `METRALE_CUBLAS_GEMM` arms the `ffn`
        // family, else `dense_gemm_tc` when it resolved, else `dense_gemm_bf16`.
        macro_rules! ffn_gemm {
            ($a:expr, $b:expr, $c:expr, $n:expr, $k:expr) => {
                if ctx.dispatch.cublas.ffn {
                    ops::cublas_bf16_proj_dense($a, $b.weight, $c, m, $n, $k, stream)?;
                } else if tc {
                    ops::dense_gemm_tc(
                        ctx.gpu,
                        self.dense_gemm_tc_k,
                        $a,
                        $b,
                        $c,
                        m,
                        $n,
                        $k,
                        stream,
                    )?;
                } else {
                    ops::dense_gemm(
                        ctx.gpu,
                        self.dense_gemm_bf16_k,
                        $a,
                        $b,
                        $c,
                        m,
                        $n,
                        $k,
                        stream,
                    )?;
                }
            };
        }
        ffn_gemm!(input, &bf16w.gate_proj, gate_out, inter, h);
        ffn_gemm!(input, &bf16w.up_proj, up_out, inter, h);
        ops::silu_mul(
            ctx.gpu,
            self.act_mul,
            gate_out,
            up_out,
            gate_out,
            m * inter,
            stream,
        )?;
        let output = ctx.buffers.moe_output();
        ffn_gemm!(gate_out, &bf16w.down_proj, output, h, inter);
        Ok(())
    }

    /// 2026-09-25: Same as `forward_prefill`.
    pub fn forward_batched(
        &self,
        input: DevicePtr,
        num_tokens: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.forward_prefill(input, num_tokens, ctx, stream)
    }
}

/// 2026-09-25: Arm choice for the native-FP8 decode FFN.
#[path = "dense_ffn_fp8_down.rs"]
pub mod fp8_down;
/// 2026-09-25: The W8A8 block-scaled FP8 prefill arm.
#[path = "dense_ffn_w8a8_prefill.rs"]
pub mod w8a8_prefill;

/// 2026-09-25: The opt-in 5..=32-row native-FP8 decode tier.
#[path = "dense_ffn_batch16_decode.rs"]
pub mod batch16_decode;

/// 2026-09-25: The tensor-core 5..=32-row FP8 tier (`m16_tc`).
#[path = "dense_ffn_m16_tc.rs"]
pub mod m16_tc;

/// 2026-09-25: The fused FP8 gate+up GEMM (`ffn_gateup_fused`).
#[path = "dense_ffn_gateup_fused.rs"]
pub mod gateup_fused;

/// 2026-09-26: `new_with_activation`, which resolves every kernel handle.
#[path = "dense_ffn_init.rs"]
mod init;

/// 2026-09-26: The `ensure_*_weight` builders and the steps of the `finalize_*_load` functions.
#[path = "dense_ffn_load.rs"]
mod load;

/// 2026-09-26: The weight-overlay installers and the LoRA delta launches.
#[path = "dense_ffn_overlays.rs"]
mod overlays;

/// 2026-09-26: Single-token decode (`forward`).
#[path = "dense_ffn_decode.rs"]
mod decode;

/// 2026-09-26: Small-batch decode (`forward_k2`, `forward_k3`, `forward_km`).
#[path = "dense_ffn_decode_batch.rs"]
mod decode_batch;

/// 2026-09-26: Prefill entry point, its packed-Q2 branch, and `w4a16_prefill_gemm`.
#[path = "dense_ffn_prefill.rs"]
mod prefill;

/// 2026-09-26: The native-FP8 prefill branch (the `w8_gemm!` ladder).
#[path = "dense_ffn_prefill_fp8.rs"]
mod prefill_fp8;

/// 2026-09-26: The NVFP4 prefill branch (the `w4_gemm!` ladder).
#[path = "dense_ffn_prefill_nvfp4.rs"]
mod prefill_nvfp4;

/// 2026-09-26: The per-call arm choices of the NVFP4 prefill branch.
#[path = "dense_ffn_nvfp4_plan.rs"]
mod nvfp4_plan;

/// 2026-09-25: Whether `forward_k2`, `forward_k3` or `forward_km` must hand the layer to
/// `forward_prefill`: true when a BF16 or FP8 overlay is installed.
fn native_small_batch_uses_prefill(has_bf16: bool, has_fp8: bool) -> bool {
    has_bf16 || has_fp8
}

#[cfg(test)]
#[path = "dense_ffn_mmq_tests.rs"]
mod mmq_tests;

#[cfg(test)]
#[path = "dense_ffn_native_batch_tests.rs"]
mod native_batch_tests;

#[cfg(test)]
#[path = "dense_ffn_kernel_tests.rs"]
mod kernel_tests;

#[cfg(test)]
#[path = "dense_ffn_fp8_residency_tests.rs"]
mod fp8_residency_tests;

#[cfg(test)]
#[path = "dense_ffn_fp8_down_tests.rs"]
mod fp8_down_tests;

#[cfg(test)]
mod tests {
    use super::native_small_batch_uses_prefill;

    #[test]
    fn native_weight_presence_requires_prefill_dispatch() {
        assert!(native_small_batch_uses_prefill(true, false));
        assert!(native_small_batch_uses_prefill(false, true));
        assert!(native_small_batch_uses_prefill(true, true));
        assert!(!native_small_batch_uses_prefill(false, false));
    }
}
