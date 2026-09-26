// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Routes the MTP drafter's BF16 small-M GEMM rows to the
//! tensor-core GEMV entries of `dense_gemv_bf16_tc.cu`.
//!
//! Owner: model-layers ops.
//! Invariants:
//! - [`try_dense_gemv_tc`] launches only when `METRALE_NO_MTP_TC` is unset or
//!   empty, `MIN_M <= m <= TC32_MAX_M`, `n > 0`, `k % K_STEP == 0` and a
//!   covering entry resolved; otherwise it returns `Ok(false)` and launches
//!   nothing.
//! - `METRALE_NO_MTP_TC` is read once per process. Any non-empty value, `0`
//!   included, turns the path off.
//!
//! The entries take `dense_gemv_bf16_batchm`'s arguments in the same order:
//! A `[M,K]` contiguous, W `[N,K]`, output rows `out_stride` elements apart.
//! The warps of a CTA reduce through shared memory in a fixed order with no
//! atomics, so the output is the same run to run.

use std::sync::{Mutex, OnceLock};

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::weight_map::DenseWeight;

/// 2026-09-25: Widest M the `tc8` entry covers: one 8-row token tile. The
/// `tc16` and `tc32` entries run 2 and 4 tiles.
pub const TC8_MAX_M: u32 = 8;
pub const TC16_MAX_M: u32 = 16;
pub const TC32_MAX_M: u32 = 32;
/// 2026-09-25: Weight rows per CTA: `16 * NT`, and every entry uses `NT = 1`.
pub const ROWS_PER_CTA: u32 = 16;
/// 2026-09-25: Threads per CTA: `DTC_WARPS` (8) warps that split K.
pub const BLOCK: u32 = 256;
/// 2026-09-25: K per warp step (`DTC_KB`); a K that is not a multiple of it
/// does not route.
pub const K_STEP: u32 = 64;
/// 2026-09-25: Narrowest M that routes; at M=1 the caller keeps its own kernel.
pub const MIN_M: u32 = 2;

/// 2026-09-25: Which tensor-core entry serves a launch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DtcKind {
    M8,
    M16,
    M32,
}

impl DtcKind {
    pub fn entry(self) -> &'static str {
        match self {
            DtcKind::M8 => "dense_gemv_bf16_tc8",
            DtcKind::M16 => "dense_gemv_bf16_tc16",
            DtcKind::M32 => "dense_gemv_bf16_tc32",
        }
    }
}

/// 2026-09-25: Pure routing decision; `None` keeps the caller's kernel.
///
/// `have` says which of `[tc8, tc16, tc32]` resolved. The narrowest resolved
/// entry that covers `m` wins; a wider one is also correct, because a token
/// tile past M is skipped. Declines when `enabled` is false, `m < MIN_M`,
/// `n == 0`, `k == 0` or `k` is not a multiple of `K_STEP`. Any other N
/// routes: the kernel loads zeros for weight rows past N and never stores
/// them.
pub fn route(m: u32, n: u32, k: u32, enabled: bool, have: [bool; 3]) -> Option<DtcKind> {
    if !enabled || m < MIN_M || n == 0 || k == 0 || !k.is_multiple_of(K_STEP) {
        return None;
    }
    [
        (TC8_MAX_M, DtcKind::M8),
        (TC16_MAX_M, DtcKind::M16),
        (TC32_MAX_M, DtcKind::M32),
    ]
    .into_iter()
    .zip(have)
    .find(|&((cap, _), ok)| ok && m <= cap)
    .map(|((_, kind), _)| kind)
}

/// 2026-09-25: The `METRALE_NO_MTP_TC` rule over the looked-up value: the
/// path is on unless the variable holds a non-empty value.
pub fn mtp_tc_from(kill: Option<&std::ffi::OsStr>) -> bool {
    kill.is_none_or(|v| v.is_empty())
}

/// 2026-09-25: Whether the path is on (`METRALE_NO_MTP_TC` unset or empty).
/// Read once, so every launch in a process sees the same choice.
pub fn mtp_tc_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| mtp_tc_from(std::env::var_os("METRALE_NO_MTP_TC").as_deref()))
}

/// 2026-09-25: Handles `[tc8, tc16, tc32]`, cached per backend, since a
/// `KernelHandle` belongs to one backend's loaded module. A missing entry is
/// the zero handle.
fn handles(gpu: &dyn GpuBackend) -> [KernelHandle; 3] {
    static CACHE: OnceLock<Mutex<Vec<(usize, [KernelHandle; 3])>>> = OnceLock::new();
    let key = gpu as *const dyn GpuBackend as *const () as usize;
    let cache = CACHE.get_or_init(|| Mutex::new(Vec::new()));
    let mut guard = cache.lock().unwrap_or_else(|p| p.into_inner());
    if let Some((_, h)) = guard.iter().find(|(k, _)| *k == key) {
        return *h;
    }
    let h = [DtcKind::M8, DtcKind::M16, DtcKind::M32]
        .map(|kind| crate::layers::try_kernel(gpu, "dense_gemv_bf16_tc", kind.entry()));
    guard.push((key, h));
    h
}

/// 2026-09-25: The tensor-core kernel and grid-x for this launch when the path
/// is on and an entry routes, else `None`.
pub fn kernel_for(gpu: &dyn GpuBackend, m: u32, n: u32, k: u32) -> Option<(KernelHandle, u32)> {
    if !mtp_tc_enabled() {
        return None;
    }
    let h = handles(gpu);
    let kind = route(m, n, k, true, h.map(|x| x.0 != 0))?;
    let handle = match kind {
        DtcKind::M8 => h[0],
        DtcKind::M16 => h[1],
        DtcKind::M32 => h[2],
    };
    Some((handle, div_ceil(n, ROWS_PER_CTA)))
}

/// 2026-09-25: Launch the given entry with the given grid-x, without routing.
/// [`try_dense_gemv_tc`] is the routed entry point.
#[allow(clippy::too_many_arguments)]
pub fn launch(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    grid_x: u32,
    input: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    out_stride: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([grid_x, 1, 1])
        .block([BLOCK, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight.weight)
        .arg_ptr(output)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .arg_u32(out_stride)
        .launch(stream)
}

/// 2026-09-25: Run `C[t] = A[t] @ W^T` on a tensor-core entry if the path is
/// on and the shape routes. `Ok(false)` means nothing was launched and the
/// caller must run its own kernel.
#[allow(clippy::too_many_arguments)]
pub fn try_dense_gemv_tc(
    gpu: &dyn GpuBackend,
    input: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    out_stride: u32,
    stream: u64,
) -> Result<bool> {
    let Some((kernel, grid_x)) = kernel_for(gpu, m, n, k) else {
        return Ok(false);
    };
    launch(
        gpu, kernel, grid_x, input, weight, output, m, n, k, out_stride, stream,
    )?;
    Ok(true)
}

#[cfg(test)]
#[path = "dense_gemv_tc_tests.rs"]
mod dense_gemv_tc_tests;
