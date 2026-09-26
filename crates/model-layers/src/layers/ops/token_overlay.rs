// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Launchers for the LoRA token-overlay kernels in
//! `kernels/gb10/common/token_overlay.cu`:
//! - [`embed_rowdiff`]: at adapter build, which adapter base rows differ from
//!   the served embedding table;
//! - [`embed_overlay_routed`]: in the forward, replace overridden vocab rows
//!   after the embed gather;
//! - [`lmhead_overlay_routed`]: in the forward, recompute overridden logit
//!   columns.
//!
//! Arguments are passed by position, so each launcher's argument order must
//! match its `.cu` signature.
//!
//! Owner: model-layers ops (LoRA token overlays).
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

use crate::layers::try_kernel;

/// 2026-09-25: The four token-overlay kernels, resolved once at model
/// construction with [`try_kernel`]. A missing kernel gives `KernelHandle(0)`:
/// the forward hooks then return without a launch, and `build_overlay` refuses
/// an adapter that ships overlay tensors.
#[derive(Clone, Copy)]
pub struct OverlayKernels {
    pub rowdiff: KernelHandle,
    pub embed_overlay: KernelHandle,
    pub lmhead_overlay_bf16: KernelHandle,
    pub lmhead_overlay_f32: KernelHandle,
}

impl Default for OverlayKernels {
    /// 2026-09-25: All handles zero, as if no overlay kernel were loaded.
    /// `KernelHandle` does not derive `Default`, so this is spelled out.
    fn default() -> Self {
        Self {
            rowdiff: KernelHandle(0),
            embed_overlay: KernelHandle(0),
            lmhead_overlay_bf16: KernelHandle(0),
            lmhead_overlay_f32: KernelHandle(0),
        }
    }
}

impl OverlayKernels {
    pub fn new(gpu: &dyn GpuBackend) -> Self {
        Self {
            rowdiff: try_kernel(gpu, "token_overlay", "embed_rowdiff_bf16"),
            embed_overlay: try_kernel(gpu, "token_overlay", "embed_overlay_routed_bf16"),
            lmhead_overlay_bf16: try_kernel(gpu, "token_overlay", "lmhead_overlay_routed_bf16"),
            lmhead_overlay_f32: try_kernel(gpu, "token_overlay", "lmhead_overlay_routed_f32"),
        }
    }
}

/// 2026-09-25: `flags[r] = (max_i |base[r,i] - served[r,i]| > thresh)`, one
/// thread per row. `base` and `served` are `[rows, h]` BF16; `flags` is
/// `[rows]` u8.
#[allow(clippy::too_many_arguments)]
pub fn embed_rowdiff(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    base: DevicePtr,
    served: DevicePtr,
    flags: DevicePtr,
    rows: u32,
    h: u32,
    thresh: f32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([rows.div_ceil(256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(base)
        .arg_ptr(served)
        .arg_ptr(flags)
        .arg_u32(rows)
        .arg_u32(h)
        .arg_f32(thresh)
        .launch(stream)
}

/// 2026-09-25: In-place replacement of overridden vocab rows in `out`
/// (`[num_tokens, h]` BF16), after the embed gather.
///
/// Per row `r`: `s = seq_slot[r]`, or `active` when `seq_slot` is null; skip
/// when `s < 0`, when `slot_map_tab[s]` is null, or when `ids[r] >= vocab`;
/// `slot = slot_map_tab[s][ids[r]]`; skip when `slot < 0` or
/// `slot >= n_tab[s]`; otherwise copy `rows_tab[s][slot]` over `out[r]`.
/// `ids` is `[num_tokens]` u32; the three tables hold one entry per adapter
/// slot; `vocab` is the length of each slot map.
#[allow(clippy::too_many_arguments)]
pub fn embed_overlay_routed(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    ids: DevicePtr,
    seq_slot: DevicePtr,
    active: i32,
    slot_map_tab: DevicePtr,
    rows_tab: DevicePtr,
    n_tab: DevicePtr,
    out: DevicePtr,
    num_tokens: u32,
    h: u32,
    vocab: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([num_tokens, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(ids)
        .arg_ptr(seq_slot)
        .arg_i32(active)
        .arg_ptr(slot_map_tab)
        .arg_ptr(rows_tab)
        .arg_ptr(n_tab)
        .arg_ptr(out)
        .arg_u32(h)
        .arg_u32(vocab)
        .launch(stream)
}

/// 2026-09-25: In-place recompute of overridden logit columns in `logits`
/// (`[m, vocab]`, BF16 or FP32 to match `kernel`). One warp per `(row, j)`,
/// where `j < max_n_override` indexes the overridden ids of that row's adapter
/// slot; the kernel skips `j >= n_tab[s]` and ids `>= vocab`. `hidden` is
/// `[m, h]` BF16; `seq_slot` is `[m]` i32, or null for the uniform `active`
/// slot.
#[allow(clippy::too_many_arguments)]
pub fn lmhead_overlay_routed(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    hidden: DevicePtr,
    seq_slot: DevicePtr,
    active: i32,
    rows_tab: DevicePtr,
    ids_tab: DevicePtr,
    n_tab: DevicePtr,
    logits: DevicePtr,
    m: u32,
    max_n_override: u32,
    h: u32,
    vocab: u32,
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, kernel)
        .grid([m, max_n_override, 1])
        .block([32, 1, 1])
        .arg_ptr(hidden)
        .arg_ptr(seq_slot)
        .arg_i32(active)
        .arg_ptr(rows_tab)
        .arg_ptr(ids_tab)
        .arg_ptr(n_tab)
        .arg_ptr(logits)
        .arg_u32(h)
        .arg_u32(vocab)
        .launch(stream)
}
