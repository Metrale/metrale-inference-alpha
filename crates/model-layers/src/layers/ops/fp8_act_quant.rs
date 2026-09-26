// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The per-token FP8 activation quantizers a layer can launch, and
//! the grid each one needs.
//!
//! Owner: model-layers ops.
//! Invariants:
//! - [`Fp8ActQuant::pick_with`] takes the kernel handle and the grid from the
//!   same `twin` flag, so a pick never pairs one kernel with the other's grid.
//!
//! The shared kernel (`kernels/gb10/common/per_token_group_quant_fp8.cu`) runs
//! one 128-thread CTA per 128-element K-group. The Hopper twin
//! (`kernels/hopper/common/fp8_act_quant_hopper.cu`) gives each group 16
//! threads that load one `uint4` each, and packs 8 groups into a CTA. Only
//! `kernels/hopper` carries the twin; on other targets its handle is 0 and the
//! shared kernel runs. `fp8_act_quant_floor.rs` decides when the twin runs.

use metrale_gpu_runtime::gpu::{GpuBackend, KernelHandle};

use super::Fp8QuantPick;

/// 2026-09-25: Module of the shared quantizer; the file and the kernel share
/// this name.
pub const FP8_QUANT_MODULE: &str = "per_token_group_quant_fp8";
/// 2026-09-25: Entry point of the shared quantizer.
pub const FP8_QUANT_ENTRY: &str = "per_token_group_quant_fp8";
/// 2026-09-25: Module (file stem) of the Hopper twin.
pub const FP8_QUANT_HOPPER_MODULE: &str = "fp8_act_quant_hopper";
/// 2026-09-25: Entry point of the Hopper twin. It differs from the shared
/// entry point, so a Hopper image carries both kernels.
pub const FP8_QUANT_HOPPER_ENTRY: &str = "per_token_group_quant_fp8_hopper";

/// 2026-09-25: K-groups one Hopper CTA covers: 128 threads / 16 threads per
/// group, as `HQ_GROUPS_PER_CTA` in the `.cu`. The kernel derives its group
/// span from `gridDim.y`, so this value sets only the grid's Y extent.
pub const FP8_QUANT_HOPPER_GROUPS_PER_CTA: u32 = 8;

/// 2026-09-25: The FP8 activation quantizer a layer launches: the shared
/// kernel, plus the Hopper twin when the image has one.
#[derive(Clone, Copy, Debug)]
pub struct Fp8ActQuant {
    /// 2026-09-25: `per_token_group_quant_fp8`, or `KernelHandle(0)` when the
    /// image lacks it.
    pub shared: KernelHandle,
    /// 2026-09-25: `per_token_group_quant_fp8_hopper`, or `KernelHandle(0)`
    /// when the image lacks it.
    pub hopper: KernelHandle,
}

impl Default for Fp8ActQuant {
    /// 2026-09-25: No quantizer: both handles are 0.
    fn default() -> Self {
        Self {
            shared: KernelHandle(0),
            hopper: KernelHandle(0),
        }
    }
}

impl Fp8ActQuant {
    /// 2026-09-25: Probe both kernels; a missing one resolves to
    /// `KernelHandle(0)`.
    pub fn resolve(gpu: &dyn GpuBackend) -> Self {
        Self {
            shared: crate::layers::try_kernel(gpu, FP8_QUANT_MODULE, FP8_QUANT_ENTRY),
            hopper: crate::layers::try_target_kernel(
                gpu,
                FP8_QUANT_HOPPER_MODULE,
                FP8_QUANT_HOPPER_ENTRY,
            ),
        }
    }

    /// 2026-09-25: A quantizer with only the shared kernel.
    pub fn shared_only(shared: KernelHandle) -> Self {
        Self {
            shared,
            hopper: KernelHandle(0),
        }
    }

    /// 2026-09-25: Is there a quantizer to launch at all? True when either
    /// handle is loaded, whatever the width; [`Self::pick`] is the per-launch
    /// choice.
    pub fn available(&self) -> bool {
        self.shared.0 != 0 || self.hopper.0 != 0
    }

    /// 2026-09-25: Is the twin in this image? Whether a launch uses it is
    /// [`Fp8QuantPick::twin`].
    pub fn twin_present(&self) -> bool {
        self.hopper.0 != 0
    }

    /// 2026-09-25: Which kernel this `(m, k)` launches, on which grid, and why
    /// not the other, from the resolved `fp8_act_quant_hopper` lever and the
    /// compiled target's SM count. [`Self::pick_with`] is the pure form.
    pub fn pick(&self, m: u32, k: u32) -> Fp8QuantPick {
        self.pick_with(
            super::fp8_act_quant_hopper_enabled(),
            m,
            k,
            metrale_kernels::TARGET_SM_COUNT,
        )
    }

    /// 2026-09-25: [`Self::pick`] over an explicit lever value and SM count.
    /// Pure, so CPU tests can grade any `(m, k)`.
    pub fn pick_with(&self, requested: bool, m: u32, k: u32, sm_count: u32) -> Fp8QuantPick {
        let reject = super::fp8_act_quant_hopper_reject(
            requested,
            self.twin_present(),
            self.shared.0 != 0,
            m,
            k,
            sm_count,
        );
        let twin = reject.is_none();
        Fp8QuantPick {
            kernel: if twin { self.hopper } else { self.shared },
            grid: fp8_quant_grid(twin, m, k),
            twin,
            reject,
            requested,
        }
    }

    /// 2026-09-25: The handle this `(m, k)` launches.
    pub fn kernel(&self, m: u32, k: u32) -> KernelHandle {
        self.pick(m, k).kernel
    }

    /// 2026-09-25: The grid for `(m, k)`, from the same pick as
    /// [`Self::kernel`].
    pub fn grid(&self, m: u32, k: u32) -> [u32; 3] {
        self.pick(m, k).grid
    }
}

/// 2026-09-25: Grid for one quantizer launch. Pure, so
/// `fp8_act_quant_tests.rs` checks both arms without a GPU.
///
/// `M` is on grid X in both arms: X allows 2^31-1 blocks, Y only 65535.
///
/// Shared arm: `(M, K/128, 1)`, one CTA per K-group; the kernel reads the
/// group from `blockIdx.y`.
///
/// Hopper arm: `(M, max(1, ceil(K/128 / 8)), 1)`. The kernel derives its group
/// span as `ceil(L / gridDim.y)`, so any Y in `1..=L` covers every group
/// exactly once; the Y extent is a speed choice.
pub fn fp8_quant_grid(hopper: bool, m: u32, k: u32) -> [u32; 3] {
    let groups = k / 128;
    if hopper {
        [
            m,
            groups.div_ceil(FP8_QUANT_HOPPER_GROUPS_PER_CTA).max(1),
            1,
        ]
    } else {
        [m, groups, 1]
    }
}

/// 2026-09-25: The half-open K-group range one Hopper CTA owns, computed as
/// the `.cu` computes `gpc`/`g0`/`g_end`. `fp8_act_quant_tests.rs` uses it to
/// check on the host that the grid's Y extent and the kernel's span partition
/// the groups.
pub fn fp8_quant_hopper_span(groups: u32, grid_y: u32, block_y: u32) -> (u32, u32) {
    let gpc = groups.div_ceil(grid_y);
    let g0 = block_y * gpc;
    if g0 >= groups {
        return (groups, groups);
    }
    (g0, (g0 + gpc).min(groups))
}

/// 2026-09-25: The group slot thread `tid` of a Hopper CTA works on, and the
/// half-open range `[start, end)` of 8 elements it owns within that
/// 128-element group, as `sub`/`lane` in the `.cu`. `None` when the slot is
/// past the CTA's span.
pub fn fp8_quant_hopper_lane(tid: u32, span: u32) -> Option<(u32, u32, u32)> {
    const LANES: u32 = 16;
    const ELEMS: u32 = 8;
    let sub = tid / LANES;
    let lane = tid % LANES;
    if sub >= span {
        return None;
    }
    Some((sub, lane * ELEMS, lane * ELEMS + ELEMS))
}

#[cfg(test)]
#[path = "fp8_act_quant_tests.rs"]
mod fp8_act_quant_tests;
