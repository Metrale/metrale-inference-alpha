// SPDX-License-Identifier: AGPL-3.0-only

//! Which `w4a4_gemv_mx*` entry serves an `m`-row projection, and with what
//! launch shape: the `METRALE_W4A4_MX_NT` / `METRALE_W4A4_MX_PS` levers and
//! the pure planning behind `mx_plan`. Split out of `w4a4_proj.rs` for the
//! 500-line cap; the code is that file's, moved verbatim.

use std::sync::OnceLock;

use spark_runtime::gpu::KernelHandle;

use super::{W4A4_MAX_M, W4a4State};

/// `METRALE_W4A4_MX_NT` = 4 (default) | 1. At 4 the 9..=64-row W4A4 GEMV
/// runs the activation-reuse twins: 17..=32 rows take 4 16-row weight tiles
/// per CTA, 9..=16 and 33..=64 rows take 2, and each warp feeds its
/// activation fragments to every tile from registers, so the activation
/// matrix is re-read from L2 that many times less. Bit-identical to the
/// one-tile kernels (same per-warp chunk order, warp-order reduction). 1 is
/// the kill switch: the historical one-tile kernels, unchanged code.
/// dgx1 ABBA vs 178a1246 (GPU-rail J/token): C=8 -6.1% at +0.8% tok/s,
/// C=16 -9.1% at -2.6% tok/s.
pub(super) fn mx_nt() -> u32 {
    static NT: OnceLock<u32> = OnceLock::new();
    *NT.get_or_init(
        || match std::env::var("METRALE_W4A4_MX_NT").ok().as_deref() {
            None | Some("") | Some("4") => 4,
            Some("1") => 1,
            Some(v) => panic!("METRALE_W4A4_MX_NT={v}: expected 1 or 4"),
        },
    )
}

/// `METRALE_W4A4_MX_PS` = 1 (default) | 0. At 1, a 9..=32-row launch whose
/// whole activation stripe fits in shared memory and whose weight has at
/// least [`PS_MIN_TILES_PER_SM`] 16-row tiles per SM runs the persistent
/// activation-staged entry (`w4a4_gemv_mx{16,32}_ps`): one CTA per SM pulls
/// 16-row tiles from a counter and reads the activations from shared memory,
/// so they leave L2 once per SM instead of once per tile. Bit-identical to
/// the one-tile kernels. 0 routes those launches back to
/// [`METRALE_W4A4_MX_NT`](mx_nt)'s kernels. It only acts when the tile
/// factor is not 1, so `METRALE_W4A4_MX_NT=1` stays a full revert.
pub(super) fn mx_ps() -> bool {
    static PS: OnceLock<bool> = OnceLock::new();
    *PS.get_or_init(
        || match std::env::var("METRALE_W4A4_MX_PS").ok().as_deref() {
            None | Some("") | Some("1") => true,
            Some("0") => false,
            Some(v) => panic!("METRALE_W4A4_MX_PS={v}: expected 0 or 1"),
        },
    )
}

/// Largest dynamic shared memory one CTA may opt in to on GB10 (sm_121).
pub const PS_SMEM_MAX: u32 = 101_376;
/// The persistent entries need this many 16-row tiles per SM. Below it the
/// once-per-launch staging is not amortised (dgx1, M=32: k/v 1024x5120 runs
/// +28% time, the 6144-row z projection -16% energy at +2% time).
pub const PS_MIN_TILES_PER_SM: u32 = 8;

/// PURE: 8-token column blocks of the persistent entry serving `m` rows
/// (`w4a4_gemv_mx16_ps` or `w4a4_gemv_mx32_ps`).
pub fn ps_column_blocks(m: u32) -> u32 {
    if m <= 16 { 2 } else { 4 }
}

/// PURE: the host side of the persistent entries' launch contract
/// (`w4a4_gemv_mx_ps.cuh`): dynamic shared memory for `mb` column blocks with
/// `sst` k128 chunks staged per warp. 8 warps x sst x mb x (512 B of
/// fragments + 64 B of scales), plus the 2-block reduction buffer.
pub fn ps_smem_bytes(mb: u32, sst: u32) -> u32 {
    8 * sst * mb * 576 + 2 * 4096
}

/// PURE: k128 chunks in one warp's stripe (chunks c = warp mod 8).
pub fn ps_stripe_chunks(k: u32) -> u32 {
    (k / 128).div_ceil(8)
}

/// How one W4A4 GEMV launch is shaped.
#[derive(Clone, Copy, Debug)]
pub(super) enum MxLaunch {
    /// Grid ceil(N / rows_per_cta).
    Tiles {
        kernel: KernelHandle,
        rows_per_cta: u32,
    },
    /// Grid #SMs, `smem` bytes of dynamic shared memory, `sst` staged chunks.
    Persistent {
        kernel: KernelHandle,
        sst: u32,
        smem: u32,
    },
}

/// PURE: the launch for an `m`-row [`n`, `k`] projection.
pub(super) fn mx_plan(s: &W4a4State, m: u32, n: u32, k: u32, nt: u32, ps: bool) -> MxLaunch {
    if ps && nt != 1 && m > 8 && m <= W4A4_MAX_M {
        let kernel = if m <= 16 { s.mx16_ps } else { s.mx32_ps };
        let sst = ps_stripe_chunks(k);
        let smem = ps_smem_bytes(ps_column_blocks(m), sst);
        if smem <= PS_SMEM_MAX && n.div_ceil(16) >= PS_MIN_TILES_PER_SM * s.sms {
            return MxLaunch::Persistent { kernel, sst, smem };
        }
    }
    let (kernel, rows_per_cta) = mx_pick(s, m, nt);
    MxLaunch::Tiles {
        kernel,
        rows_per_cta,
    }
}

/// PURE: (kernel, rows per CTA) for an `m`-row launch at tile factor `nt`.
pub(super) fn mx_pick(s: &W4a4State, m: u32, nt: u32) -> (KernelHandle, u32) {
    match (m, nt) {
        (0..=8, _) => (s.mx8, 16),
        (9..=16, 1) => (s.mx16, 16),
        (9..=16, _) => (s.mx16_nt2, 32),
        (17..=32, 1) => (s.mx32, 16),
        (17..=32, _) => (s.mx32_nt4, 64),
        (_, 1) => (s.mx64, 16),
        _ => (s.mx64_nt2, 32),
    }
}
