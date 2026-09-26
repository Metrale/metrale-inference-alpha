// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Paged-decode attention split-K: how many splits, which kernel pair runs them, the
//! GQA-packed non-split route, and the once-per-dtype route log line. `run_paged_decode.rs` calls
//! it from its NVFP4, FP8 and BF16 arms.
//!
//! Owner: model-layers attention decode.
//! Invariants:
//! - [`num_splits`] depends only on the resolved `attn_decode_splitk` policy,
//!   `TARGET_SM_COUNT`, the head count and head dim, and, under `legacy` only,
//!   `split_ref_seqs(num_seqs, max_decode_seqs)` = `max(max_decode_seqs, num_seqs)`. Under
//!   `auto` and a pinned count it does not move with the number of co-batched sequences.
//! - The GQA-packed kernel is returned only when the lever is armed, the shape passes
//!   `gqa_pack_shape_ok`, and the handle is present.
//!
//! The `legacy` rule (`attn_splitk::legacy_splits`) has this shape, with the compiled target's
//! `sm_count` in place of the `NUM_SMS` constant written here:
//!
//! ```text
//! use metrale_core::device::sm121::NUM_SMS;            // 48 — the GB10 constant
//! let current_ctas = num_q_heads * split_ref_seqs(num_seqs, max_decode_seqs);
//! let num_splits = if current_ctas >= NUM_SMS { 1 } else { NUM_SMS / current_ctas };
//! ```

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_kernels::attn_splitk;

use super::super::Qwen3AttentionLayer;
use crate::layers::ops;

/// 2026-09-25: Everything a split-K launch needs that is not dtype-specific.
#[derive(Clone, Copy)]
pub(super) struct SplitkPlan {
    pub num_splits: u32,
    pub num_q_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub block_size: u32,
    pub max_blocks_per_seq: u32,
    pub num_seqs: u32,
    pub inv_sqrt_d: f32,
    pub q_stride: u32,
    pub sliding_window: u32,
}

/// 2026-09-25: The head dim the split-K kernels are compiled for. `paged_decode_attn_fp8.cu`,
/// `paged_decode_attn_nvfp4.cu` and the Hopper twins' `paged_decode_splitk_hopper.cuh` take `HDIM`
/// from a `#define` that defaults to 256 and derive each lane's element count from it; the
/// `head_dim` argument feeds only pointer arithmetic.
pub(super) const SPLITK_HEAD_DIM: u32 = 256;

/// 2026-09-25: The split count for this launch.
///
/// `num_seqs` is read only by the `legacy` policy, through `split_ref_seqs`; `auto` and a pinned
/// count ignore it (`the_auto_split_count_does_not_move_with_the_co_batched_count` in
/// `attn_splitk_tests.rs`). `legacy` is the declaration on the b200, b300 and gb10 targets and the
/// value baked for a target without one; hopper declares `auto`.
///
/// A head dim other than [`SPLITK_HEAD_DIM`] gets one split under every policy but `legacy`,
/// which keeps its own answer.
pub(super) fn num_splits(
    num_q_heads: u32,
    head_dim: u32,
    num_seqs: u32,
    max_decode_seqs: u32,
) -> u32 {
    let policy = ops::target_defaults::resolved().attn_decode_splitk.value;
    if policy != attn_splitk::SplitkPolicy::Legacy && head_dim != SPLITK_HEAD_DIM {
        return 1;
    }
    attn_splitk::num_splits(
        policy,
        metrale_kernels::TARGET_SM_COUNT,
        num_q_heads,
        super::super::split_ref_seqs(num_seqs, max_decode_seqs),
    )
}

/// 2026-09-25: With `METRALE_ATTN_DBG` set to any value, logs at debug level the split count this
/// layer resolved, on every call with more than one sequence.
pub(super) fn trace_splits(layer_idx: usize, num_seqs: u32, num_q_heads: u32, num_splits: u32) {
    if num_seqs != 1 && std::env::var("METRALE_ATTN_DBG").is_ok() {
        tracing::debug!(
            "ATTN_DBG L{layer_idx} num_seqs={num_seqs} num_splits={num_splits} \
             (policy={} sm_count={} nq={num_q_heads})",
            ops::target_defaults::resolved()
                .attn_decode_splitk
                .value
                .label(),
            metrale_kernels::TARGET_SM_COUNT,
        );
    }
}

/// 2026-09-25: The kernel entry names the route line reports. Strings rather than handles, because
/// a `KernelHandle` is an opaque index and the line has to name the kernel.
pub(super) const ROUTE_SPLITK_FP8: &str = "paged_decode_attn_splitk_fp8_hopper";
pub(super) const ROUTE_SPLITK_BF16: &str = "paged_decode_attn_splitk_bf16_hopper";
pub(super) const ROUTE_SPLITK_GB10_FP8: &str = "paged_decode_attn_splitk_fp8";
pub(super) const ROUTE_SPLITK_NVFP4: &str = "paged_decode_attn_splitk_nvfp4";
pub(super) const ROUTE_NONSPLIT_FP8: &str = "paged_decode_attn_fp8";
pub(super) const ROUTE_NONSPLIT_BF16: &str = "paged_decode_attn";
pub(super) const ROUTE_NONSPLIT_NVFP4: &str = "paged_decode_attn_nvfp4";

/// 2026-09-25: The dispatch-side route line, as text.
///
/// ```text
/// paged decode attention: paged_decode_attn_splitk_fp8_hopper num_splits=11 \
///   sm_count=132 policy=auto (METRALE_ATTN_DECODE_SPLITK)
/// ```
///
/// `num_splits` is the count the caller passes, `policy` is
/// [`attn_splitk::SplitkPolicy::label`], the spelling the boot line also prints, and `sm_count`
/// is the compiled target's. The policy and the split count can differ: under `auto` on hopper
/// (132 SMs) with 24 q heads the line reads `policy=auto` and `num_splits=11`.
pub(super) fn route_line(
    kernel: &str,
    num_splits: u32,
    policy: attn_splitk::SplitkPolicy,
) -> String {
    format!(
        "paged decode attention: {kernel} num_splits={num_splits} sm_count={} policy={} \
         (METRALE_ATTN_DECODE_SPLITK)",
        metrale_kernels::TARGET_SM_COUNT,
        policy.label(),
    )
}

/// 2026-09-25: The GQA-packed non-split entry points, for the route line.
pub(super) const ROUTE_GQA_FP8: &str = "paged_decode_attn_fp8_gqa";
pub(super) const ROUTE_GQA_BF16: &str = "paged_decode_attn_bf16_gqa";

/// 2026-09-25: The packed non-split kernel for this launch, or `None` to keep the unpacked one.
/// All three must hold:
/// - the lever is armed ([`attn_splitk::gqa_pack_enabled`], off unless
///   `METRALE_ATTN_DECODE_GQA_PACK` arms it);
/// - the shape is the one the kernels are compiled for ([`attn_splitk::gqa_pack_shape_ok`]); any
///   other GQA ratio would index the wrong query heads;
/// - the handle is present (`init_arch_gates::present` stores a zero handle as `None`).
///
/// Called only from the non-split arms. The packed grid is `(num_kv_heads, num_seqs)`, so a split
/// version would partition the KV range by `num_kv_heads` and change the online-softmax merge
/// tree.
pub(super) fn gqa_pack_kernel(
    handle: Option<KernelHandle>,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
) -> Option<KernelHandle> {
    gqa_pack_route(
        attn_splitk::gqa_pack_enabled(),
        handle,
        num_q_heads,
        num_kv_heads,
        head_dim,
    )
}

/// 2026-09-25: The conjunction behind [`gqa_pack_kernel`], with the lever's value passed in, so
/// tests can reach the shape and handle checks at `armed = true`; `gqa_pack_enabled` is resolved
/// once per process from the environment.
pub(super) fn gqa_pack_route(
    armed: bool,
    handle: Option<KernelHandle>,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
) -> Option<KernelHandle> {
    if !armed {
        return None;
    }
    if !attn_splitk::gqa_pack_shape_ok(num_q_heads, num_kv_heads, head_dim) {
        return None;
    }
    handle
}

/// 2026-09-25: Which once-flag a route line belongs to. One per KV dtype: the arms dispatch
/// independently, and one model can run more than one of them, so a shared flag would report only
/// the arm reached first.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum RouteArm {
    Fp8,
    Bf16,
    Nvfp4,
}

/// 2026-09-25: Log the route line once per process per KV dtype, on that arm's first decode
/// dispatch.
pub(super) fn log_decode_route(arm: RouteArm, kernel: &str, num_splits: u32) {
    static SAID: [std::sync::Once; 3] = [const { std::sync::Once::new() }; 3];
    let idx = match arm {
        RouteArm::Fp8 => 0,
        RouteArm::Bf16 => 1,
        RouteArm::Nvfp4 => 2,
    };
    SAID[idx].call_once(|| {
        let policy = ops::target_defaults::resolved().attn_decode_splitk.value;
        tracing::info!("{}", route_line(kernel, num_splits, policy));
    });
}

/// 2026-09-25: A split-K kernel and its reduce kernel. For FP8 KV,
/// [`Qwen3AttentionLayer::fp8_splitk_pair`] picks the Hopper twins
/// (`kernels/hopper/common/paged_decode_fp8_splitk_hopper.cu`) when the build carries them and the
/// head dim is [`SPLITK_HEAD_DIM`], and the gb10 pair otherwise. The choice follows kernel
/// presence; whether split-K runs at all is the `attn_decode_splitk` policy.
pub(super) struct SplitkPair {
    pub splitk: KernelHandle,
    pub reduce: KernelHandle,
    /// 2026-09-25: The split kernel's entry name, for the route line. Set where the pair is chosen,
    /// so the log names the kernel that was resolved.
    pub name: &'static str,
}

impl Qwen3AttentionLayer {
    pub(super) fn fp8_splitk_pair(&self, head_dim: u32) -> Option<SplitkPair> {
        if head_dim == SPLITK_HEAD_DIM
            && let (Some(splitk), Some(reduce)) = (
                self.paged_decode_splitk_hopper_k,
                self.paged_decode_reduce_hopper_k,
            )
        {
            return Some(SplitkPair {
                splitk,
                reduce,
                name: ROUTE_SPLITK_FP8,
            });
        }
        Some(SplitkPair {
            splitk: self.paged_decode_splitk_k?,
            reduce: self.paged_decode_reduce_k?,
            name: ROUTE_SPLITK_GB10_FP8,
        })
    }

    /// 2026-09-25: The BF16-KV split-K pair: the Hopper twins only, so `None` on a build without
    /// them, and `None` for a head dim other than [`SPLITK_HEAD_DIM`]. On `None` the caller runs a
    /// non-split kernel (`paged_decode_512_k` for heads wider than 256 when it is loaded).
    pub(super) fn bf16_splitk_pair(&self, head_dim: u32) -> Option<SplitkPair> {
        if head_dim != SPLITK_HEAD_DIM {
            return None;
        }
        Some(SplitkPair {
            splitk: self.paged_decode_splitk_bf16_hopper_k?,
            reduce: self.paged_decode_reduce_bf16_hopper_k?,
            name: ROUTE_SPLITK_BF16,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn launch_splitk_fp8(
        &self,
        gpu: &dyn GpuBackend,
        pair: &SplitkPair,
        plan: SplitkPlan,
        q: DevicePtr,
        k_pool: DevicePtr,
        v_pool: DevicePtr,
        workspace: DevicePtr,
        output: DevicePtr,
        block_table: DevicePtr,
        seq_lens: DevicePtr,
        k_scale: f32,
        v_scale: f32,
        cache_stride: u64,
        stream: u64,
    ) -> Result<()> {
        ops::paged_decode_attn_splitk_fp8(
            gpu,
            pair.splitk,
            q,
            k_pool,
            v_pool,
            workspace,
            block_table,
            seq_lens,
            plan.max_blocks_per_seq,
            plan.num_q_heads,
            plan.num_kv_heads,
            plan.head_dim,
            plan.block_size,
            plan.inv_sqrt_d,
            plan.num_splits,
            k_scale,
            v_scale,
            plan.q_stride,
            cache_stride,
            plan.num_seqs,
            plan.sliding_window,
            stream,
        )?;
        ops::paged_decode_attn_reduce_fp8(
            gpu,
            pair.reduce,
            workspace,
            output,
            seq_lens,
            plan.num_q_heads,
            plan.head_dim,
            plan.num_splits,
            plan.num_seqs,
            stream,
        )
    }

    /// 2026-09-25: BF16 split-K, then the shared `paged_decode_attn_reduce_fp8` reduce.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn launch_splitk_bf16(
        &self,
        gpu: &dyn GpuBackend,
        pair: &SplitkPair,
        plan: SplitkPlan,
        q: DevicePtr,
        k_pool: DevicePtr,
        v_pool: DevicePtr,
        workspace: DevicePtr,
        output: DevicePtr,
        block_table: DevicePtr,
        seq_lens: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        ops::paged_decode_attn_splitk_bf16(
            gpu,
            pair.splitk,
            q,
            k_pool,
            v_pool,
            workspace,
            block_table,
            seq_lens,
            plan.max_blocks_per_seq,
            plan.num_q_heads,
            plan.num_kv_heads,
            plan.head_dim,
            plan.block_size,
            plan.inv_sqrt_d,
            plan.num_splits,
            plan.q_stride,
            plan.num_seqs,
            plan.sliding_window,
            stream,
        )?;
        ops::paged_decode_attn_reduce_fp8(
            gpu,
            pair.reduce,
            workspace,
            output,
            seq_lens,
            plan.num_q_heads,
            plan.head_dim,
            plan.num_splits,
            plan.num_seqs,
            stream,
        )
    }
}

/// 2026-09-25: True when this launch should take the split-K path at all. One split is no split-K:
/// the caller runs the non-split kernel.
pub(super) fn splits_are_worth_it(num_splits: u32) -> bool {
    num_splits > 1
}
