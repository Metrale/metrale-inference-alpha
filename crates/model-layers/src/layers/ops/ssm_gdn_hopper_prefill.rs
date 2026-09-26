// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The Hopper twins of chunked-prefill kernels 1
//! (`recompute_wu`) and 3 (`chunk_fwd_o`): their shared-memory sizes and the
//! rule that selects them.
//!
//! The kernels are `kernels/hopper/common/gdn_fwd_o_hopper.cu` and
//! `gdn_recompute_wu_hopper.cu`, which only `kernels/hopper` compiles. On every
//! other target their handles are `KernelHandle(0)` and the parents run.
//! The twins follow the tensor-core prefill lever that also selects the state
//! spine (`[defaults] gdn_prefill_tc`, `METRALE_GDN_PREFILL_TC` overriding).
//! `METRALE_NO_GDN_PREFILL_TC_REMNANTS=1` keeps the spine and leaves both on
//! their parents.
//!
//! Owner: model-layers ops (GDN).
//! Invariants:
//! - A refused pick is the parent's handle, block size and shared memory,
//!   unchanged.

use metrale_gpu_runtime::gpu::KernelHandle;

/// 2026-09-25: Compile-time tile of both twins: `GDNH_K_DIM == GDNH_V_DIM` in
/// `kernels/hopper/common/gdn_prefill_hopper.cuh`.
pub(crate) const GDN_HOPPER_DIM: u32 = 128;
/// 2026-09-25: That header's `GDNH_CHUNK`.
pub(crate) const GDN_HOPPER_CHUNK: u32 = 64;

/// 2026-09-25: Mirror of `FOH_SMEM` in `gdn_fwd_o_hopper.cu`:
///
/// ```text
/// sq[64][136] + sk[64][136] + Sb[128][136] + ucT[128][72] + kqh[64][72]
///   + gc[64] f32 = 17408 + 17408 + 34816 + 18432 + 9216 + 256 = 97 536 B
/// ```
///
/// Smaller than the parent's 98 816 B, because the `kq` lo limb aliases `sk`.
/// The padded 136/72 row strides make the MMA fragment reads
/// bank-conflict-free. Under-sizing this reads a tile out of bounds; the kernel
/// `static_assert`s the same value.
pub(crate) const GDN_FWD_O_HOPPER_SMEM: u32 = 2 * (GDN_HOPPER_CHUNK * 136 * 2)
    + GDN_HOPPER_DIM * 136 * 2
    + GDN_HOPPER_DIM * 72 * 2
    + GDN_HOPPER_CHUNK * 72 * 2
    + GDN_HOPPER_CHUNK * 4;

/// 2026-09-25: Mirror of `WUH_SMEM` in `gdn_recompute_wu_hopper.cu`, which
/// `static_assert`s the same value:
///
/// ```text
/// sk[64][136] + Ld/Tf[64][24] f32 + Lh/Ll[64][72] + Th/Tl[64][24]
///   + Xh/Xl[16][16][24] + gc[64] f32
///   = 17408 + 2*6144 + 2*9216 + 2*3072 + 2*12288 + 256 = 79 104 B
/// ```
pub(crate) const GDN_WU_HOPPER_SMEM: u32 = GDN_HOPPER_CHUNK * 136 * 2
    + 2 * (GDN_HOPPER_CHUNK * 24 * 4)
    + 2 * (GDN_HOPPER_CHUNK * 72 * 2)
    + 2 * (GDN_HOPPER_CHUNK * 24 * 2)
    + 2 * (16 * 16 * 24 * 2)
    + GDN_HOPPER_CHUNK * 4;

/// 2026-09-25: Both twins are built for 512 threads (`__launch_bounds__(512, _)`).
pub(crate) const GDN_HOPPER_REMNANT_BLOCK: u32 = 512;

/// 2026-09-25: Why a Hopper twin is not running; `None` means it is.
///
/// Pure, so the grammar is testable without a GPU or the process environment.
/// Each refusal names its guard. The tile guards are required: both kernels'
/// fragment maps, padded strides and warp splits are compile-time
/// 128/128/64, so a narrower head or another chunk would read the wrong
/// columns.
pub(crate) fn gdn_hopper_remnant_reject(
    requested: bool,
    killed: bool,
    kernel_present: bool,
    k_dim: u32,
    v_dim: u32,
    chunk: u32,
) -> Option<&'static str> {
    if !requested {
        Some("not requested")
    } else if killed {
        Some("METRALE_NO_GDN_PREFILL_TC_REMNANTS=1 pins wu/fwd_o to their parents")
    } else if !kernel_present {
        Some("kernel absent from this image (kernels/hopper only)")
    } else if k_dim != GDN_HOPPER_DIM || v_dim != GDN_HOPPER_DIM || chunk != GDN_HOPPER_CHUNK {
        Some("head/chunk differs from the compile-time tile (K_DIM=V_DIM=128, CHUNK=64)")
    } else {
        None
    }
}

/// 2026-09-25: One remnant's launch: the handle, the thread count and the
/// shared memory. Both remnants are picked by [`gdn_hopper_remnant_pick`], so
/// they follow one selection rule.
pub(crate) struct RemnantPick {
    pub kernel: KernelHandle,
    pub block: u32,
    pub smem: u32,
    /// 2026-09-25: `None` when the twin runs; the refusing guard when the
    /// parent does.
    pub reject: Option<&'static str>,
}

/// 2026-09-25: Choose between a parent kernel and its Hopper twin. When the
/// twin is refused, the parent's handle, `parent_block` and `parent_smem` are
/// returned unchanged.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gdn_hopper_remnant_pick(
    requested: bool,
    killed: bool,
    parent: KernelHandle,
    parent_block: u32,
    parent_smem: u32,
    twin: KernelHandle,
    twin_smem: u32,
    k_dim: u32,
    v_dim: u32,
    chunk: u32,
) -> RemnantPick {
    let reject = gdn_hopper_remnant_reject(requested, killed, twin.0 != 0, k_dim, v_dim, chunk);
    match reject {
        None => RemnantPick {
            kernel: twin,
            block: GDN_HOPPER_REMNANT_BLOCK,
            smem: twin_smem,
            reject,
        },
        Some(_) => RemnantPick {
            kernel: parent,
            block: parent_block,
            smem: parent_smem,
            reject,
        },
    }
}

/// 2026-09-25: Log which kernel ran and, when the lever asked for a twin it did
/// not get, which guard refused. Nothing is logged when the lever is off.
pub(crate) fn gdn_hopper_remnant_log(name: &str, pick: &RemnantPick, requested: bool) {
    match pick.reject {
        Some(why) if requested => {
            tracing::warn!("GDN {name}: the Hopper twin is NOT running: {why}");
        }
        None => tracing::info!(
            "GDN {name}: gated_delta_rule_{name}_hopper (METRALE_GDN_PREFILL_TC) \
             block={} smem={}B",
            pick.block,
            pick.smem
        ),
        _ => {}
    }
}

/// 2026-09-25: Pick both remnants and log the verdicts.
///
/// `requested` is the resolved tensor-core prefill lever. The caller reads it
/// once and uses the same value for the state spine, so the twins and the
/// spine cannot disagree about it. The parents launch with 256
/// (`recompute_wu`) and 512 (`chunk_fwd_o`) threads.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gdn_hopper_remnants(
    requested: bool,
    k_wu: KernelHandle,
    smem_wu: u32,
    k_fo: KernelHandle,
    smem_fo: u32,
    k_wu_hopper: KernelHandle,
    k_fo_hopper: KernelHandle,
    k_dim: u32,
    v_dim: u32,
    chunk: u32,
) -> (RemnantPick, RemnantPick) {
    let killed = gdn_hopper_remnants_killed();
    let pick = |parent, pblock, psmem, twin, tsmem| {
        gdn_hopper_remnant_pick(
            requested, killed, parent, pblock, psmem, twin, tsmem, k_dim, v_dim, chunk,
        )
    };
    let wu = pick(k_wu, 256, smem_wu, k_wu_hopper, GDN_WU_HOPPER_SMEM);
    let fo = pick(k_fo, 512, smem_fo, k_fo_hopper, GDN_FWD_O_HOPPER_SMEM);
    gdn_hopper_remnant_log("recompute_wu", &wu, requested);
    gdn_hopper_remnant_log("chunk_fwd_o", &fo, requested);
    (wu, fo)
}

/// 2026-09-25: `METRALE_NO_GDN_PREFILL_TC_REMNANTS=1` pins both remnants to
/// their parents. Only the value `1` counts. Read once per
/// [`gdn_hopper_remnants`] call, for both remnants.
pub(crate) fn gdn_hopper_remnants_killed() -> bool {
    std::env::var("METRALE_NO_GDN_PREFILL_TC_REMNANTS")
        .ok()
        .as_deref()
        == Some("1")
}

#[cfg(test)]
#[path = "ssm_gdn_remnants_tests.rs"]
mod ssm_gdn_remnants_tests;

#[cfg(test)]
#[path = "ssm_gdn_remnants_numerics_tests.rs"]
mod ssm_gdn_remnants_numerics_tests;
