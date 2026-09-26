// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: When the Hopper FP8 activation-quant twin takes a launch: the
//! CTA-count floor, the lever that requests it, and one log line per branch.
//!
//! Owner: model-layers ops.
//! Invariants:
//! - With the twin present and no shared kernel in the pair,
//!   [`fp8_act_quant_hopper_reject`] returns `None` (the twin runs), whatever
//!   the lever and the floor say.
//!
//! The twin packs 8 K-groups into a CTA, so its grid (`M x ceil(K/128 / 8)`)
//! has about an eighth of the shared kernel's CTAs. It takes a launch only
//! when that grid has at least [`FP8_QUANT_MIN_CTAS_PER_SM`] x `sm_count`
//! CTAs. The floor counts CTAs, not tokens, because the same M makes a
//! different grid at a different K. With `sm_count = 132`
//! (`kernels/hopper/HARDWARE.toml`) the floor is 264 CTAs, and the smallest M
//! the twin accepts is:
//!
//! | K | K/128 | grid Y | min M |
//! |---:|---:|---:|---:|
//! | 5120 | 40 | 5 | 53 |
//! | 6144 | 48 | 6 | 44 |
//! | 17408 | 136 | 17 | 16 |
//!
//! `fp8_act_quant_tests.rs` pins these thresholds.

use metrale_gpu_runtime::gpu::KernelHandle;

/// 2026-09-25: CTAs per SM the twin's grid must reach before it takes a
/// launch. The same value as `ssm_ba_gates_hopper::MIN_CTAS_PER_SM`.
pub const FP8_QUANT_MIN_CTAS_PER_SM: u32 = 2;

/// 2026-09-25: The smallest twin grid, in CTAs, the floor accepts for this SM
/// count; an `sm_count` of 0 counts as 1.
pub fn fp8_quant_min_ctas(sm_count: u32) -> u32 {
    FP8_QUANT_MIN_CTAS_PER_SM.saturating_mul(sm_count.max(1))
}

/// 2026-09-25: CTAs the twin's grid launches for `(m, k)`: the product of
/// [`super::fp8_quant_grid`]'s Hopper arm, so the floor measures the grid the
/// launch uses.
pub fn fp8_quant_hopper_ctas(m: u32, k: u32) -> u32 {
    let [x, y, z] = super::fp8_quant_grid(true, m, k);
    x.saturating_mul(y).saturating_mul(z)
}

/// 2026-09-25: The smallest `m` the twin accepts at this `k`: the table in
/// this module's header, as a function.
pub fn fp8_quant_hopper_min_m(k: u32, sm_count: u32) -> u32 {
    let [_, y, _] = super::fp8_quant_grid(true, 1, k);
    fp8_quant_min_ctas(sm_count).div_ceil(y.max(1))
}

/// 2026-09-25: Is the twin requested? The compiled target's `[defaults]
/// fp8_act_quant_hopper`, overridden by `METRALE_FP8_ACT_QUANT_HOPPER` when
/// it is set ([`super::target_defaults`]).
pub fn fp8_act_quant_hopper_enabled() -> bool {
    super::target_defaults::resolved()
        .fp8_act_quant_hopper
        .value
}

/// 2026-09-25: The floor's refusal reason.
pub const FP8_QUANT_TOO_FEW_CTAS: &str = "too few CTAs to fill the device at 8 K-groups per CTA";

/// 2026-09-25: Every reason [`fp8_act_quant_hopper_reject`] can return.
/// [`fp8_quant_log`] keeps one once-flag per entry, by index.
pub const FP8_QUANT_REJECTS: [&str; 3] = [
    "not requested",
    "kernel absent from this image (kernels/hopper only)",
    FP8_QUANT_TOO_FEW_CTAS,
];

/// 2026-09-25: Why the twin is not running; `None` means it runs.
///
/// Pure, so the rule is testable without a GPU or the process environment.
///
/// When the twin is present and the pair has no shared kernel, it returns
/// `None` before it reads the lever or the floor: a refusal hands the launch
/// to `Fp8ActQuant::shared`, and there is none to hand it to.
/// `native_fp8_act_quant_hopper_microtest` builds such a pair
/// (`shared: KernelHandle(0)`) to run the twin at every M.
pub fn fp8_act_quant_hopper_reject(
    requested: bool,
    twin_present: bool,
    parent_present: bool,
    m: u32,
    k: u32,
    sm_count: u32,
) -> Option<&'static str> {
    if !twin_present {
        Some(FP8_QUANT_REJECTS[1])
    } else if !parent_present {
        None
    } else if !requested {
        Some(FP8_QUANT_REJECTS[0])
    } else if fp8_quant_hopper_ctas(m, k) < fp8_quant_min_ctas(sm_count) {
        Some(FP8_QUANT_TOO_FEW_CTAS)
    } else {
        None
    }
}

/// 2026-09-25: Which kernel a launch runs, on what grid, and why the other did
/// not. [`super::Fp8ActQuant::pick_with`] sets the handle and the grid from one
/// decision.
///
/// No `PartialEq`, because `KernelHandle` does not implement it; tests compare
/// `twin`, `reject` and `grid`.
#[derive(Debug, Clone, Copy)]
pub struct Fp8QuantPick {
    pub kernel: KernelHandle,
    pub grid: [u32; 3],
    /// 2026-09-25: `true` when `kernel` is the Hopper twin.
    pub twin: bool,
    /// 2026-09-25: `None` when the twin runs; otherwise the refusal reason.
    pub reject: Option<&'static str>,
    /// 2026-09-25: The lever value the pick was made with.
    pub requested: bool,
}

/// 2026-09-25: Which once-flag [`fp8_quant_log`] uses for a verdict. There is
/// one per branch because the verdict changes with M, so one process can log
/// both the twin line and a refusal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fp8QuantLogSlot {
    /// 2026-09-25: The twin took the launch.
    Twin,
    /// 2026-09-25: The shared kernel took it, for the reason at this index of
    /// [`FP8_QUANT_REJECTS`].
    Reject(usize),
}

/// 2026-09-25: Once-flags [`fp8_quant_log`] keeps: one per refusal reason,
/// plus the twin's.
pub const FP8_QUANT_LOG_SLOTS: usize = FP8_QUANT_REJECTS.len() + 1;

/// 2026-09-25: The slot a verdict belongs to, or `None` when it is not logged.
/// Pure, so tests replay a call order on the CPU.
pub fn fp8_quant_log_slot(pick: &Fp8QuantPick) -> Option<Fp8QuantLogSlot> {
    match pick.reject {
        None => Some(Fp8QuantLogSlot::Twin),
        // 2026-09-25: The twin was not requested, so the shared kernel is the
        // expected outcome and nothing is logged.
        Some(_) if !pick.requested => None,
        Some(why) => FP8_QUANT_REJECTS
            .iter()
            .position(|r| *r == why)
            .map(Fp8QuantLogSlot::Reject),
    }
}

/// 2026-09-25: Log which quantizer runs and, when the lever requested the twin
/// and did not get it, which reason refused it: once per process per slot.
/// `per_token_group_quant_fp8` calls this on every launch.
pub fn fp8_quant_log(pick: &Fp8QuantPick, m: u32, k: u32) {
    static SAID: [std::sync::Once; FP8_QUANT_LOG_SLOTS] =
        [const { std::sync::Once::new() }; FP8_QUANT_LOG_SLOTS];
    let Some(slot) = fp8_quant_log_slot(pick) else {
        return;
    };
    let idx = match slot {
        Fp8QuantLogSlot::Twin => FP8_QUANT_LOG_SLOTS - 1,
        Fp8QuantLogSlot::Reject(i) => i,
    };
    let [gx, gy, _] = pick.grid;
    SAID[idx].call_once(|| match pick.reject {
        Some(why) => tracing::info!(
            "FP8 act-quant: the Hopper twin is NOT running at M={m} K={k}: {why} \
             (METRALE_FP8_ACT_QUANT_HOPPER)"
        ),
        None => tracing::info!(
            "FP8 act-quant: per_token_group_quant_fp8_hopper \
             (METRALE_FP8_ACT_QUANT_HOPPER) M={m} K={k} grid=({gx},{gy},1) block=128"
        ),
    });
}
