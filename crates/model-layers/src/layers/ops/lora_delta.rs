// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Runtime LoRA delta `y += scale * (x @ A^T) @ B^T` on BF16 activations: the single-adapter path and the per-request routed path.
//!
//! Owner: model-layers ops.
//! Invariants:
//! - Both stages contract at the pool's padded rank `max_rank`, never at the adapter's rank;
//!   the pool is zeroed before packing, so the pad rows of A and pad columns of B are zero
//!   (`lora/loading.rs`).
//! - The launchers here only launch kernels: they allocate nothing and do not synchronize.
//! - `apply_lora_delta` returns without launching when `METRALE_LORA_NO_APPLY=1`.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use crate::layers::ops;
use crate::weight_map::DenseWeight;

/// 2026-09-25: The kernel handles the LoRA launchers use, resolved when an adapter is installed.
/// Module names follow the `[modules]` table of kernels/gb10/common/KERNEL.toml
/// (`dense_gemv_bf16.cu` is `gemv`, `dense_gemm_tc.cu` is `gemm_tc`, `dense_gemm_bf16.cu` is
/// `gemm`); `residual_add`, `lora_bgmv`, `moe_lora_grouped_down` and `moe_lora_gather_bgmv` are
/// file stems.
#[derive(Clone, Copy)]
pub struct LoraKernels {
    pub gemv_k: KernelHandle,
    pub gemm_tc_k: KernelHandle, // 2026-09-25: KernelHandle(0) when missing; `gemm_k` is used instead.
    /// 2026-09-25: Fused expand and fold (`C += scale * A @ B^T`). `KernelHandle(0)` when missing;
    /// the unfused `gemm_tc` + `scaled_add` pair is used instead.
    pub gemm_tc_acc_k: KernelHandle,
    pub gemm_k: KernelHandle,
    pub scaled_add_k: KernelHandle,
    /// 2026-09-25: The per-request routed kernels of [`apply_lora_bgmv`]: shrink, then expand and
    /// fold.
    pub bgmv_shrink_k: KernelHandle,
    pub bgmv_expand_fold_k: KernelHandle,
    /// 2026-09-25: MoE expert LoRA over expert-sorted rows, keyed on expert through the device
    /// `expert_offsets`: shrink, then expand and fold. `KernelHandle(0)` when missing; the launcher
    /// in `moe_lora_grouped.rs` then returns an error rather than launching.
    pub moe_down_shrink_k: KernelHandle,
    pub moe_down_expand_fold_k: KernelHandle,
    /// 2026-09-25: MoE expert LoRA over unsorted, slot-major rows, keyed on expert through the
    /// per-slot `indices` array: shrink, then expand and fold. `KernelHandle(0)` when missing; the
    /// launcher in `moe_lora_grouped.rs` then returns an error rather than launching.
    pub moe_gather_shrink_k: KernelHandle,
    pub moe_gather_expand_fold_k: KernelHandle,
}

impl LoraKernels {
    pub fn new(gpu: &dyn GpuBackend) -> Result<Self> {
        Ok(Self {
            gemv_k: gpu.kernel("gemv", "dense_gemv_bf16")?,
            gemm_tc_k: crate::layers::try_kernel(gpu, "gemm_tc", "dense_gemm_tc"),
            gemm_tc_acc_k: crate::layers::try_kernel(gpu, "gemm_tc", "dense_gemm_tc_scaled_acc"),
            gemm_k: gpu.kernel("gemm", "dense_gemm_bf16")?,
            scaled_add_k: gpu.kernel("residual_add", "bf16_scaled_add")?,
            bgmv_shrink_k: gpu.kernel("lora_bgmv", "lora_bgmv_shrink")?,
            bgmv_expand_fold_k: gpu.kernel("lora_bgmv", "lora_bgmv_expand_fold")?,
            moe_down_shrink_k: crate::layers::try_kernel(
                gpu,
                "moe_lora_grouped_down",
                "moe_lora_grouped_down_shrink",
            ),
            moe_down_expand_fold_k: crate::layers::try_kernel(
                gpu,
                "moe_lora_grouped_down",
                "moe_lora_grouped_down_expand_fold",
            ),
            moe_gather_shrink_k: crate::layers::try_kernel(
                gpu,
                "moe_lora_gather_bgmv",
                "moe_lora_gather_bgmv_shrink",
            ),
            moe_gather_expand_fold_k: crate::layers::try_kernel(
                gpu,
                "moe_lora_gather_bgmv",
                "moe_lora_gather_bgmv_expand_fold",
            ),
        })
    }
}

/// 2026-09-25: Per-(layer, module) routing tables that [`apply_lora_bgmv`] reads: the
/// `[max_loras]` device tables of A and B pool addresses (`a_table` / `b_table`, 0 = the slot
/// does not adapt this module), the `[max_loras]` f32 `scale_table`, and the projection dims.
/// The addresses are fixed when the pool is packed, so the adapter for each row is chosen only
/// by the per-step `seq_slot` buffer. Installed by copy onto the layer next to the active
/// [`LoraPair`], which the unrouted path uses.
#[derive(Debug, Clone, Copy)]
pub struct LoraRoute {
    pub a_table: DevicePtr,
    pub b_table: DevicePtr,
    pub scale_table: DevicePtr,
    pub k_in: u32,
    pub n_out: u32,
    pub max_rank: u32,
}

/// 2026-09-25: One adapted module, with the PEFT tensors in their own layouts, which are the
/// `[N, K]` weight layout the dense kernels expect:
///   a: `[max_rank, k_in]` row-major BF16 (PEFT `lora_A` `[r, in_features]` in the first `rank`
///      rows, zero rows after);
///   b: `[n_out, max_rank]` row-major BF16 (PEFT `lora_B` `[out_features, r]` in the first
///      `rank` columns of each row, zero columns after).
/// `scale` is `lora_alpha / r`, or `lora_alpha / sqrt(r)` with `use_rslora`, from the adapter's
/// config, where `use_rslora` is required (config/src/parsers/lora.rs). It is applied in the fold,
/// not folded into B.
#[derive(Debug, Clone, Copy)]
pub struct LoraPair {
    pub a: DenseWeight,
    pub b: DenseWeight,
    pub rank: u32,
    pub k_in: u32,
    pub n_out: u32,
    pub scale: f32,
    /// 2026-09-25: The pool's padded rank: the row stride of `b` and the row count of `a`. The
    /// shrink produces `max_rank` columns and the expand contracts over `max_rank`, because B rows
    /// are `max_rank` elements apart: an expand at `k = rank` would misread every row after the
    /// first when `rank < max_rank`. The pad entries are zero.
    pub max_rank: u32,
}

/// 2026-09-25: Per-layer attention LoRA weights, installed by copy onto `Qwen3AttentionLayer`.
///
/// The `q` pair folds its delta into the raw q_proj output at offset 0 over the full q_proj
/// width (on a gated model the interleaved `[Q|gate]` row), before `deinterleave_qg` splits it
/// (`apply_q_lora` in `qwen3_attention/decode/attention_forward.rs`).
#[derive(Clone, Copy)]
pub struct LoraAttnWeights {
    /// 2026-09-25: The global layer index (`0..num_hidden_layers`), set at install. Prefill uses it
    /// to index a request slot's per-layer pairs; `attn_layer_idx` counts attention layers only and
    /// differs from it on hybrid GDN/attention models.
    pub layer_idx: usize,
    pub q: Option<LoraPair>,
    pub k: Option<LoraPair>,
    pub v: Option<LoraPair>,
    pub o: Option<LoraPair>,
    pub kernels: LoraKernels,
    /// 2026-09-25: Per-request routing tables, per module. `None` means no routing, and the
    /// module's pair above is applied. With `Some`, a step that carries a `seq_slot` buffer applies
    /// the routed delta through [`apply_lora_bgmv`].
    pub q_route: Option<LoraRoute>,
    pub k_route: Option<LoraRoute>,
    pub v_route: Option<LoraRoute>,
    pub o_route: Option<LoraRoute>,
}

/// 2026-09-25: Per-layer dense-FFN LoRA weights, installed by copy onto `DenseFfnLayer`.
#[derive(Clone, Copy)]
pub struct LoraFfnWeights {
    pub gate: Option<LoraPair>,
    pub up: Option<LoraPair>,
    pub down: Option<LoraPair>,
    pub kernels: LoraKernels,
}

/// 2026-09-25: `METRALE_LORA_NO_APPLY` equal to `"1"`, read once per process: keep the adapter
/// resident but skip every [`apply_lora_delta`]. A measurement lever that separates the cost of
/// applying an adapter from the cost of having one loaded; the output is the base model's while
/// it is set.
pub fn lora_no_apply() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("METRALE_LORA_NO_APPLY").as_deref() == Ok("1"))
}

/// 2026-09-25: Row count at or below which [`apply_lora_delta`] runs `m` single-row GEMV deltas
/// instead of one GEMM: `METRALE_LORA_GEMV_MAX_M`, read once per process, default 48 when unset
/// or unparsable. `dense_gemm_tc` tiles 16 rows (`TC_TM` in dense_gemm_tc.cu) and reads all of B
/// whatever `m` is, so a small `m` pays full B traffic for a partly empty tile.
pub fn lora_gemv_max_m() -> u32 {
    static V: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("METRALE_LORA_GEMV_MAX_M")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(48)
    })
}

/// 2026-09-25: `METRALE_LORA_NO_FFN` equal to `"1"`, read once per process: the dense-FFN
/// (`dense_ffn.rs`) and GDN out_proj (`qwen3_ssm/lora.rs`) skip their deltas, and the attention
/// deltas still apply. A measurement lever; the output is wrong while it is set.
pub fn lora_no_ffn() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("METRALE_LORA_NO_FFN").as_deref() == Ok("1"))
}

/// 2026-09-25: `base_out[m, n_out] += scale * (x[m, k_in] @ a^T) @ b^T` for one [`LoraPair`].
///
/// `x` rows must be contiguous with stride `k_in * 2` bytes and `base_out` rows with stride
/// `n_out * 2` bytes; a strided caller loops with `m = 1` on offset pointers. `lora_xa` must hold
/// `m * max_rank` BF16 and `lora_delta` `m * n_out` BF16.
///
/// Paths: `m == 1` runs two GEMVs; `1 < m <= lora_gemv_max_m()` runs one single-row delta per
/// row; larger `m` runs the tensor-core GEMMs, with the fused expand-and-fold when both
/// `gemm_tc` handles resolved, and `dense_gemm` when `gemm_tc` is missing. Every unfused path
/// ends with `bf16_scaled_add`.
#[allow(clippy::too_many_arguments)]
pub fn apply_lora_delta(
    gpu: &dyn GpuBackend,
    kernels: &LoraKernels,
    pair: &LoraPair,
    x: DevicePtr,        // 2026-09-25: [m, pair.k_in] BF16
    base_out: DevicePtr, // 2026-09-25: [m, pair.n_out] BF16, modified in place
    m: u32,
    lora_xa: DevicePtr,    // 2026-09-25: scratch of at least m * max_rank BF16
    lora_delta: DevicePtr, // 2026-09-25: scratch of at least m * n_out BF16
    stream: u64,
) -> Result<()> {
    if lora_no_apply() {
        return Ok(());
    }
    // 2026-09-25: Small m: one single-row delta per row. Rows are contiguous by this function's
    // contract, so a row is a pointer offset.
    if m > 1 && m <= lora_gemv_max_m() {
        for row in 0..m {
            let x_row = DevicePtr(x.0 + (row as u64) * (pair.k_in as u64) * 2);
            let out_row = DevicePtr(base_out.0 + (row as u64) * (pair.n_out as u64) * 2);
            apply_lora_delta(
                gpu, kernels, pair, x_row, out_row, 1, lora_xa, lora_delta, stream,
            )?;
        }
        return Ok(());
    }
    if m == 1 {
        // 2026-09-25: shrink: [1,k_in] @ A[max_rank,k_in]^T -> xa[1,max_rank]
        ops::dense_gemv(
            gpu,
            kernels.gemv_k,
            x,
            &pair.a,
            lora_xa,
            pair.max_rank,
            pair.k_in,
            stream,
        )?;
        // 2026-09-25: expand: [1,max_rank] @ B[n_out,max_rank]^T -> delta[1,n_out]
        ops::dense_gemv(
            gpu,
            kernels.gemv_k,
            lora_xa,
            &pair.b,
            lora_delta,
            pair.n_out,
            pair.max_rank,
            stream,
        )?;
    } else if kernels.gemm_tc_acc_k.0 != 0 && kernels.gemm_tc_k.0 != 0 {
        // 2026-09-25: Fused path: the shrink, then expand and fold in one launch, without the
        // `[m, n_out]` scratch or a separate scaled_add pass.
        ops::dense_gemm_tc(
            gpu,
            kernels.gemm_tc_k,
            x,
            &pair.a,
            lora_xa,
            m,
            pair.max_rank,
            pair.k_in,
            stream,
        )?;
        return ops::dense_gemm_tc_scaled_acc(
            gpu,
            kernels.gemm_tc_acc_k,
            lora_xa,
            &pair.b,
            base_out,
            m,
            pair.n_out,
            pair.max_rank,
            pair.scale,
            stream,
        );
    } else if kernels.gemm_tc_k.0 != 0 {
        ops::dense_gemm_tc(
            gpu,
            kernels.gemm_tc_k,
            x,
            &pair.a,
            lora_xa,
            m,
            pair.max_rank,
            pair.k_in,
            stream,
        )?;
        ops::dense_gemm_tc(
            gpu,
            kernels.gemm_tc_k,
            lora_xa,
            &pair.b,
            lora_delta,
            m,
            pair.n_out,
            pair.max_rank,
            stream,
        )?;
    } else {
        ops::dense_gemm(
            gpu,
            kernels.gemm_k,
            x,
            &pair.a,
            lora_xa,
            m,
            pair.max_rank,
            pair.k_in,
            stream,
        )?;
        ops::dense_gemm(
            gpu,
            kernels.gemm_k,
            lora_xa,
            &pair.b,
            lora_delta,
            m,
            pair.n_out,
            pair.max_rank,
            stream,
        )?;
    }
    // 2026-09-25: fold: base_out += scale * delta (`bf16_scaled_add` in residual_add.cu)
    ops::scaled_add(
        gpu,
        kernels.scaled_add_k,
        base_out,
        lora_delta,
        pair.scale,
        m * pair.n_out,
        stream,
    )
}

/// 2026-09-25: Per-request routed LoRA delta over `n` rows, each naming its adapter slot in
/// `seq_slot[n]` (i32; a negative slot gets no delta):
/// `out[i, :] += scale_s * (x[i, :] @ A_s^T) @ B_s^T` with `s = seq_slot[i]`.
///
/// Two launches (kernels/gb10/common/lora_bgmv.cu): the shrink writes BF16 `xa`, and the expand
/// rounds each delta to BF16 before adding `scale_table[s] * delta` in FP32, the order of
/// `bf16_scaled_add`. A slot whose table entry is 0 does not adapt this module. Both stages run
/// at `route.max_rank`.
///
/// Strides are in elements: `x_row_stride` between `x` rows (at least `k_in`) and
/// `out_row_stride` between `base_out` rows (at least `n_out`), so the fold can land inside an
/// interleaved buffer. `lora_xa` must hold `n * max_rank` BF16. The launch arguments follow the
/// kernels' parameter order; the launch does not check their types.
#[allow(clippy::too_many_arguments)]
pub fn apply_lora_bgmv(
    gpu: &dyn GpuBackend,
    kernels: &LoraKernels,
    route: &LoraRoute,
    x: DevicePtr,        // 2026-09-25: [n, x_row_stride] BF16
    base_out: DevicePtr, // 2026-09-25: [n, out_row_stride] BF16, folded in place
    seq_slot: DevicePtr,
    n: u32,
    x_row_stride: u32,
    out_row_stride: u32,
    lora_xa: DevicePtr,
    stream: u64,
) -> Result<()> {
    use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};

    // 2026-09-25: Kernel 1, shrink: xa[n, max_rank] = x @ A_s^T, 4 outputs per block.
    KernelLaunch::new(gpu, kernels.bgmv_shrink_k)
        .grid([div_ceil(route.max_rank, 4), n, 1])
        .block([256, 1, 1])
        .arg_ptr(x)
        .arg_ptr(seq_slot)
        .arg_ptr(route.a_table)
        .arg_ptr(lora_xa)
        .arg_u32(n)
        .arg_u32(route.max_rank)
        .arg_u32(route.k_in)
        .arg_u32(x_row_stride)
        .launch(stream)?;

    // 2026-09-25: Kernel 2, expand and fold: base_out += scale_s * (xa @ B_s^T), 4 outputs per
    // block.
    KernelLaunch::new(gpu, kernels.bgmv_expand_fold_k)
        .grid([div_ceil(route.n_out, 4), n, 1])
        .block([256, 1, 1])
        .arg_ptr(lora_xa)
        .arg_ptr(seq_slot)
        .arg_ptr(route.b_table)
        .arg_ptr(route.scale_table)
        .arg_ptr(base_out)
        .arg_u32(n)
        .arg_u32(route.n_out)
        .arg_u32(route.max_rank)
        .arg_u32(out_row_stride)
        .launch(stream)
}
