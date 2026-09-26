// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The Hopper BA-gates twin `dense_gemm_ba_gates_prefill_hopper`
//! (`kernels/hopper/common/ssm_ba_gates_hopper.cu`): its launcher, the choice
//! between it and the gb10 parent `dense_gemm_ba_gates_prefill`
//! (`ssm_preprocess.cu`), the log of that choice, and the index mappings both
//! kernels must share.
//!
//! The parent runs `ceil(N/4)` blocks per token; each block's 256 threads are
//! four 64-lane groups, one per BA output, and each group reads the token's
//! whole activation row, so the row is read `N` times per token. The twin runs
//! one block per token, and each thread accumulates [`BA_GATES_GROUPS`] output
//! groups per read of the row: 12 reads at `N = 96`.
//!
//! The twin keeps the parent's reduction order (lane-strided `kv` sweep, the
//! 16/8/4/2/1 shuffle, `warp_even + warp_odd`), and
//! `native_ssm_ba_gates_hopper_microtest` checks its gate and beta output
//! bytes against the parent's.
//!
//! Only `kernels/hopper` ships the twin, so elsewhere its handle is
//! `KernelHandle(0)` and [`ba_gates_pick`] keeps the parent even when the lever
//! is on. The lever is declared on in `kernels/hopper/HARDWARE.toml` and off in
//! the gb10, b200 and b300 ones.
//!
//! One block per token makes the grid the token count, and the parent's
//! callers include the batched decode (`qwen3_ssm/trait_decode_batched.rs`).
//! [`ssm_ba_gates_hopper_reject`] therefore declines below
//! `MIN_CTAS_PER_SM * sm_count` tokens, and the parent runs.
//!
//! Owner: model-layers ops.
//! Invariants:
//! - [`ba_gates_pick`] returns the twin only when [`ssm_ba_gates_hopper_reject`]
//!   returns `None`.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

use crate::weight_map::DenseWeight;

/// 2026-09-25: Threads per block, as in the parent (`BAH_BLOCK` in the twin). The
/// reduction order depends on it, [`BA_GATES_LANES`] and [`BA_GATES_OUTS`].
pub const BA_GATES_BLOCK: u32 = 256;
/// 2026-09-25: Threads that reduce one BA output: the parent's `threads_per_out`.
pub const BA_GATES_LANES: u32 = 64;
pub const BA_GATES_OUTS: u32 = BA_GATES_BLOCK / BA_GATES_LANES;
/// 2026-09-25: Warps per block: the width of one row of the twin's cross-warp
/// scratch `red`.
pub const BA_GATES_WARPS: u32 = BA_GATES_BLOCK / 32;
/// 2026-09-25: `BAH_GROUPS` in the twin: output groups a thread accumulates per
/// read of the activation row. The row is read `ceil(ceil(N/4)/8) * 4` times per
/// token, 12 at `N = 96`. The kernel source records the register and occupancy
/// table behind the choice of 8.
pub const BA_GATES_GROUPS: u32 = 8;

/// 2026-09-25: Tokens per SM the twin requires before it takes a launch. It is a
/// judgement, not a contract: set high, it only sends more launches to the
/// parent, which computes the same bits.
pub const MIN_CTAS_PER_SM: u32 = 2;

/// 2026-09-25: SM count used when `GpuBackend::sm_count()` fails: the
/// `sm_count` in `kernels/hopper/HARDWARE.toml`. A wrong value moves only the
/// guard's threshold.
pub const BA_GATES_FALLBACK_SM_COUNT: u32 = 132;

/// 2026-09-25: The smallest token count the twin accepts with `sm_count` SMs
/// (0 is treated as 1).
pub fn ba_gates_min_tokens(sm_count: u32) -> u32 {
    MIN_CTAS_PER_SM.saturating_mul(sm_count.max(1))
}

/// 2026-09-25: The lever: the target's `[defaults] ssm_ba_gates_hopper`, which
/// `METRALE_SSM_BA_GATES_HOPPER` overrides (`0`, `false`, `off` or `no` turn it
/// off, any other value on; [`super::target_defaults`]). There is no
/// `METRALE_NO_*` spelling for it.
pub fn ssm_ba_gates_hopper_enabled() -> bool {
    super::target_defaults::resolved().ssm_ba_gates_hopper.value
}

/// 2026-09-25: Why the twin does not run, as one of [`BA_GATES_REJECTS`]; `None`
/// means it runs. Pure, so tests cover it without a GPU. The shape guards:
/// - `K % 8 != 0`: the uint4 sweep would skip the tail (the parent has the same
///   limit and does not check it);
/// - `K_stride < K`: the row is shorter than the reduction;
/// - `N == 0` or `K == 0`: nothing to compute;
/// - fewer than [`ba_gates_min_tokens`] tokens: the decode guard in the module
///   header.
pub fn ssm_ba_gates_hopper_reject(
    requested: bool,
    kernel_present: bool,
    m: u32,
    n: u32,
    k: u32,
    k_stride: u32,
    sm_count: u32,
) -> Option<&'static str> {
    if !requested {
        Some("not requested")
    } else if !kernel_present {
        Some("kernel absent from this image (kernels/hopper only)")
    } else if n == 0 || k == 0 {
        Some("empty BA projection")
    } else if !k.is_multiple_of(8) {
        Some("K is not a multiple of 8 (the uint4 K sweep would drop a tail)")
    } else if k_stride < k {
        Some("K_stride < K: the activation row is shorter than the reduction")
    } else if m < ba_gates_min_tokens(sm_count) {
        Some(BA_GATES_TOO_FEW_TOKENS)
    } else {
        None
    }
}

/// 2026-09-25: The kernel a launch runs, and why the other did not.
pub struct BaGatesPick {
    pub kernel: KernelHandle,
    /// 2026-09-25: `true` when `kernel` is the Hopper twin.
    pub twin: bool,
    /// 2026-09-25: `None` when the twin runs; the refusing guard when the parent does.
    pub reject: Option<&'static str>,
}

/// 2026-09-25: Choose between the gb10 parent and the Hopper twin. Its one
/// caller is `ssm_preproc::dense_gemm_ba_gates_prefill`, which the five BA-gate
/// call sites share, so the rule exists once.
#[allow(clippy::too_many_arguments)]
pub fn ba_gates_pick(
    requested: bool,
    parent: KernelHandle,
    twin: KernelHandle,
    m: u32,
    n: u32,
    k: u32,
    k_stride: u32,
    sm_count: u32,
) -> BaGatesPick {
    let reject = ssm_ba_gates_hopper_reject(requested, twin.0 != 0, m, n, k, k_stride, sm_count);
    match reject {
        None => BaGatesPick {
            kernel: twin,
            twin: true,
            reject,
        },
        Some(_) => BaGatesPick {
            kernel: parent,
            twin: false,
            reject,
        },
    }
}

/// 2026-09-25: Every string [`ssm_ba_gates_hopper_reject`] can return, in the
/// order it tests them. [`ba_gates_log`] gives each its own once-flag by index
/// here; the test `every_reject_reason_has_its_own_log_slot` checks that every
/// returned reason has an index.
pub const BA_GATES_REJECTS: [&str; 6] = [
    "not requested",
    "kernel absent from this image (kernels/hopper only)",
    "empty BA projection",
    "K is not a multiple of 8 (the uint4 K sweep would drop a tail)",
    "K_stride < K: the activation row is shorter than the reduction",
    BA_GATES_TOO_FEW_TOKENS,
];

/// 2026-09-25: The token-count guard's string, named so tests can match it.
pub const BA_GATES_TOO_FEW_TOKENS: &str = "too few tokens to fill the device at one CTA per token";

/// 2026-09-25: The once-flag slot a verdict logs under in [`ba_gates_log`]; one
/// slot per outcome, so an early small request cannot hide a later line about
/// the other outcome.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BaGatesLogSlot {
    /// 2026-09-25: The twin took the launch.
    Twin,
    /// 2026-09-25: The parent took it, for the guard at this index of
    /// [`BA_GATES_REJECTS`].
    Reject(usize),
}

/// 2026-09-25: Once-flags [`ba_gates_log`] keeps: one per guard, plus the twin's.
pub const BA_GATES_LOG_SLOTS: usize = BA_GATES_REJECTS.len() + 1;

/// 2026-09-25: The slot a verdict belongs to, or `None` when the lever was not
/// requested. Pure, so tests can replay a call sequence on the CPU.
pub fn ba_gates_log_slot(pick: &BaGatesPick, requested: bool) -> Option<BaGatesLogSlot> {
    match pick.reject {
        None => Some(BaGatesLogSlot::Twin),
        // 2026-09-25: With the lever off the parent is expected, so nothing is logged.
        Some(_) if !requested => None,
        Some(why) => BA_GATES_REJECTS
            .iter()
            .position(|r| *r == why)
            .map(BaGatesLogSlot::Reject),
    }
}

/// 2026-09-25: Log which kernel runs and, when the lever asked for the twin and
/// a guard refused, which guard, once per process per slot. The call runs once
/// per layer per step and its verdict depends on `m`, so each outcome keeps its
/// own flag and names the `m` it was first reached at.
pub fn ba_gates_log(pick: &BaGatesPick, requested: bool, m: u32) {
    static SAID: [std::sync::Once; BA_GATES_LOG_SLOTS] =
        [const { std::sync::Once::new() }; BA_GATES_LOG_SLOTS];
    let Some(slot) = ba_gates_log_slot(pick, requested) else {
        return;
    };
    let (idx, why) = match slot {
        BaGatesLogSlot::Twin => (BA_GATES_LOG_SLOTS - 1, None),
        BaGatesLogSlot::Reject(i) => (i, pick.reject),
    };
    SAID[idx].call_once(|| match why {
        Some(why) => tracing::info!(
            "SSM ba_gates: the Hopper twin is NOT running at M={m}: {why} \
             (METRALE_SSM_BA_GATES_HOPPER)"
        ),
        None => tracing::info!(
            "SSM ba_gates: dense_gemm_ba_gates_prefill_hopper \
             (METRALE_SSM_BA_GATES_HOPPER) M={m} block={BA_GATES_BLOCK} grid=(M,1,1)"
        ),
    });
}

/// 2026-09-25: The device's SM count, or [`BA_GATES_FALLBACK_SM_COUNT`] when the
/// query fails; at least 1.
pub fn ba_gates_sm_count(gpu: &dyn GpuBackend) -> u32 {
    gpu.sm_count().unwrap_or(BA_GATES_FALLBACK_SM_COUNT).max(1)
}

/// 2026-09-25: Launch `dense_gemm_ba_gates_prefill_hopper`, one block per token.
/// It takes the parent's arguments in the parent's order; only the grid differs.
#[allow(clippy::too_many_arguments)]
pub fn dense_gemm_ba_gates_prefill_hopper(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    ba_weight: &DenseWeight,
    a_log: DevicePtr,
    dt_bias: DevicePtr,
    gate_out: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    k_stride: u32,
    gate_stride: u32,
    nv: u32,
    vheads_per_group: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([m, 1, 1])
        .block([BA_GATES_BLOCK, 1, 1])
        .arg_ptr(input)
        .arg_ptr(ba_weight.weight)
        .arg_ptr(a_log)
        .arg_ptr(dt_bias)
        .arg_ptr(gate_out)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(k_stride)
        .arg_u32(gate_stride)
        .arg_u32(nv)
        .arg_u32(vheads_per_group)
        .launch(stream)
}

// 2026-09-25: The index mappings the parent and the twin must share, as pure
// functions that `ssm_ba_gates_hopper_tests` checks on the CPU.

/// 2026-09-25: The `kv` indices lane `lane` accumulates, in order: the parent's
/// `for (kv = lane; kv < K_VEC; kv += threads_per_out)`. The order is the
/// reduction order.
pub fn ba_gates_lane_kv(lane: u32, k_vec: u32) -> Vec<u32> {
    let mut out = Vec::new();
    let mut kv = lane;
    while kv < k_vec {
        out.push(kv);
        kv += BA_GATES_LANES;
    }
    out
}

/// 2026-09-25: The BA output a thread owns: `group * BA_GATES_OUTS + local_out`.
/// The parent's group is `blockIdx.x`; the twin's is `g0 + g`, walking
/// `0..ceil(N/4)` in tiles of [`BA_GATES_GROUPS`].
pub fn ba_gates_output(group: u32, local_out: u32) -> u32 {
    group * BA_GATES_OUTS + local_out
}

/// 2026-09-25: The block-wide warp that holds lane `lane` of output `local_out`.
/// The parent writes its warp partial to `smem[local_out * 2 + lane / 32]` and
/// the twin to `red[g * BA_GATES_WARPS + threadIdx.x / 32]`; they match because
/// `threadIdx.x / 32 == local_out * 2 + lane / 32`.
pub fn ba_gates_warp(local_out: u32, lane: u32) -> u32 {
    local_out * 2 + lane / 32
}

/// 2026-09-25: The two warp partials summed for output `local_out`, in order: the
/// parent's `smem[local_out * 2] + smem[local_out * 2 + 1]`.
pub fn ba_gates_cross_warp_pair(local_out: u32) -> (u32, u32) {
    (local_out * 2, local_out * 2 + 1)
}

/// 2026-09-25: Where output `n` lands in the `[gate(nv), beta(nv)]` row: `Ok(vh)`
/// for a gate (alpha) element, `Err(vh)` for a beta element, split as the
/// parent splits them on `within_group < vheads_per_group`.
pub fn ba_gates_slot(n: u32, vheads_per_group: u32) -> Result<u32, u32> {
    let group_dim_ba = 2 * vheads_per_group;
    let within_group = n % group_dim_ba;
    let group = n / group_dim_ba;
    if within_group < vheads_per_group {
        Err(group * vheads_per_group + within_group)
    } else {
        Ok(group * vheads_per_group + (within_group - vheads_per_group))
    }
}

#[cfg(test)]
#[path = "ssm_ba_gates_hopper_tests.rs"]
mod ssm_ba_gates_hopper_tests;
