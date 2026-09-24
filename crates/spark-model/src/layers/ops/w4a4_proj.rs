// SPDX-License-Identifier: AGPL-3.0-only

//! OPT-IN W4A4 small-M projection on FP4 tensor cores (`w4a4_gemv_mx.cu`,
//! module `w4a4_gemv_mx`), under `--w4a4-downcast true` (default false).
//!
//! # Why
//!
//! bf16 tensor-core MMA power on GB10 scales with LIVE rows (TEP 2026-09-23:
//! `w4a16_gemv_tc8` draws 46/51/59 W at 1/4/8 rows). The block-scale FP4 MMA
//! (`kind::mxf4nvf4`) consumes the checkpoint's NVFP4 weight bytes and E4M3
//! group scales directly, with no dequant, and moves 4x fewer operand bits per
//! MAC. vLLM runs its NVFP4 layers this way at every M. The activations are
//! quantised per row to NVFP4 first (dynamic per-row global scale; see the
//! .cu header), so the numerics become W4A4. That is a real accuracy change,
//! which is why this is opt-in.
//!
//! # Scope
//!
//! Projection sites only, via [`nvfp4_proj_small_m`]: GDN qkvz/out_proj,
//! attention q/k/v/o, dense-FFN gate/up/down at 1..=32 rows. The lm_head keeps
//! `w4a16_gemv_batchm`, because logits are the most quantisation-sensitive
//! output.
//!
//! # Scratch
//!
//! The quantised activations need `[M, K/2] + [M, K/16] + [M] f32` of device
//! scratch. One buffer is allocated per backend by [`prepare`] at model build
//! (called from `W4a16BatchmTiers::resolve`), never on the decode path, so a
//! CUDA-graph capture never sees an allocation. Launches are stream-ordered
//! on the caller's stream. Two projections on DIFFERENT streams at once would
//! race on it, and no dense-27B site does that.

use std::sync::{Mutex, OnceLock};

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::weight_map::QuantizedWeight;

/// Rows the narrow entries (`w4a4_gemv_mx32`) cover.
pub const W4A4_MAX_M: u32 = 32;
/// Rows the wide entries (`w4a4_gemv_mx64*`, `--w4a4-downcast-wide`) cover.
pub const W4A4_WIDE_MAX_M: u32 = 64;
/// Largest K the scratch is sized for (every 27B projection is <= 17408).
pub const W4A4_MAX_K: u32 = 32768;

/// `--w4a4-downcast`: THE one reader of the W4A4 activation-downcast choice.
/// It covers both halves: the projection path in this module, and the
/// dense-FFN small-M steps on the W4A4 MMQ (`DenseFfn::forward_k2/k3/km`).
/// There is NO environment fallback. The serve publishes it before the model
/// is built, and anything that reads it first (a test, an example) sees the
/// default, false.
static W4A4_DOWNCAST: OnceLock<bool> = OnceLock::new();

/// Publish `--w4a4-downcast`. Returns the value in force: the first
/// publication wins, and a caller seeing a different value must warn.
pub fn set_w4a4_downcast_from_cli(on: bool) -> bool {
    let _ = W4A4_DOWNCAST.set(on);
    *W4A4_DOWNCAST.get().expect("just set")
}

/// `--w4a4-downcast` in force? False unless the serve published true.
pub fn w4a4_downcast_enabled() -> bool {
    *W4A4_DOWNCAST.get_or_init(|| false)
}

/// `--w4a4-downcast-wide`: THE one reader. Same publication rules as
/// [`W4A4_DOWNCAST`]; the serve publishes `downcast && wide`.
static W4A4_WIDE: OnceLock<bool> = OnceLock::new();

/// Publish `--w4a4-downcast-wide`. Returns the value in force.
pub fn set_w4a4_wide_from_cli(on: bool) -> bool {
    let _ = W4A4_WIDE.set(on);
    *W4A4_WIDE.get().expect("just set")
}

/// `--w4a4-downcast-wide` in force (and `--w4a4-downcast` with it)?
pub fn w4a4_wide_enabled() -> bool {
    w4a4_downcast_enabled() && *W4A4_WIDE.get_or_init(|| false)
}

/// Widest row count the W4A4 projection path serves right now.
pub fn w4a4_max_m() -> u32 {
    if w4a4_wide_enabled() {
        W4A4_WIDE_MAX_M
    } else {
        W4A4_MAX_M
    }
}

#[derive(Clone, Copy)]
struct W4a4State {
    quant: KernelHandle,
    mx8: KernelHandle,
    mx16: KernelHandle,
    mx32: KernelHandle,
    /// Activation-reuse twins (`METRALE_W4A4_MX_NT`), bit-identical to mx16/mx32.
    mx16_nt2: KernelHandle,
    mx32_nt4: KernelHandle,
    /// Persistent activation-staged entries (`METRALE_W4A4_MX_PS`),
    /// bit-identical to mx16/mx32.
    mx16_ps: KernelHandle,
    mx32_ps: KernelHandle,
    /// Streaming multiprocessors: the persistent entries' grid.
    sms: u32,
    /// 33..=64 rows under `--w4a4-downcast-wide` (zero handles otherwise).
    mx64: KernelHandle,
    mx64_nt2: KernelHandle,
    aq: DevicePtr,
    a_scale: DevicePtr,
    a_gs: DevicePtr,
    /// W4A16 reference output for [`audit_enabled`] (null otherwise).
    audit_ref: DevicePtr,
}

/// Largest projection N the audit buffer holds (the FFN gate+up, 34816).
const AUDIT_MAX_N: usize = 65536;

/// `METRALE_W4A4_PROJ_AUDIT` (non-empty): DIAGNOSTIC ONLY. Every eager (not
/// graph-captured) W4A4 projection also runs the W4A16 path, synchronises,
/// and accumulates `||y_w4a4 - y_w4a16|| / ||y_w4a16||` per CALL SITE, logged
/// every 64 samples. It costs a sync per launch, so run it only with graphs
/// off (`METRALE_NO_MTP_VERIFY_GRAPHS=1 METRALE_NO_DECODE_GRAPHS_MULTISEQ=1`).
fn audit_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("METRALE_W4A4_PROJ_AUDIT").is_some_and(|v| !v.is_empty()))
}

/// `METRALE_W4A4_MX_NT` = 4 (default) | 1. At 4 the 9..=64-row W4A4 GEMV
/// runs the activation-reuse twins: 17..=32 rows take 4 16-row weight tiles
/// per CTA, 9..=16 and 33..=64 rows take 2, and each warp feeds its
/// activation fragments to every tile from registers, so the activation
/// matrix is re-read from L2 that many times less. Bit-identical to the
/// one-tile kernels (same per-warp chunk order, warp-order reduction). 1 is
/// the kill switch: the historical one-tile kernels, unchanged code.
/// dgx1 ABBA vs 178a1246 (GPU-rail J/token): C=8 -6.1% at +0.8% tok/s,
/// C=16 -9.1% at -2.6% tok/s.
fn mx_nt() -> u32 {
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
fn mx_ps() -> bool {
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
enum MxLaunch {
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
fn mx_plan(s: &W4a4State, m: u32, n: u32, k: u32, nt: u32, ps: bool) -> MxLaunch {
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
fn mx_pick(s: &W4a4State, m: u32, nt: u32) -> (KernelHandle, u32) {
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

fn cache() -> &'static Mutex<Vec<(usize, Option<W4a4State>)>> {
    static CACHE: OnceLock<Mutex<Vec<(usize, Option<W4a4State>)>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(Vec::new()))
}

fn key(gpu: &dyn GpuBackend) -> usize {
    gpu as *const dyn GpuBackend as *const () as usize
}

/// Resolve the kernels and allocate the scratch ONCE per backend, when the
/// lever is on. Call at model build, never inside a graph capture.
pub fn prepare(gpu: &dyn GpuBackend) -> Result<()> {
    if !w4a4_downcast_enabled() {
        return Ok(());
    }
    let mut guard = cache().lock().unwrap_or_else(|p| p.into_inner());
    if guard.iter().any(|(k, _)| *k == key(gpu)) {
        return Ok(());
    }
    let h = |f: &str| crate::layers::try_kernel(gpu, "w4a4_gemv_mx", f);
    let (quant, mx8, mx16, mx32) = (
        h("w4a4_quant_rows"),
        h("w4a4_gemv_mx8"),
        h("w4a4_gemv_mx16"),
        h("w4a4_gemv_mx32"),
    );
    let (mx16_nt2, mx32_nt4) = (h("w4a4_gemv_mx16_nt2"), h("w4a4_gemv_mx32_nt4"));
    let (mx16_ps, mx32_ps) = (h("w4a4_gemv_mx16_ps"), h("w4a4_gemv_mx32_ps"));
    let state = if [quant, mx8, mx16, mx32, mx16_nt2, mx32_nt4, mx16_ps, mx32_ps]
        .iter()
        .all(|k| k.0 != 0)
    {
        let sms = gpu.sm_count()?;
        tracing::info!(
            "w4a4 projection: METRALE_W4A4_MX_NT={} METRALE_W4A4_MX_PS={} ({sms} SMs) wide={} (max rows {}, ffn {})",
            mx_nt(),
            u8::from(mx_ps()),
            w4a4_wide_enabled(),
            w4a4_max_m(),
            ffn_proj_max_rows()
        );
        let (m, k) = (w4a4_max_m() as usize, W4A4_MAX_K as usize);
        let wide = |f: &str| {
            if w4a4_wide_enabled() {
                h(f)
            } else {
                KernelHandle(0)
            }
        };
        Some(W4a4State {
            quant,
            mx8,
            mx16,
            mx32,
            mx16_nt2,
            mx32_nt4,
            mx16_ps,
            mx32_ps,
            sms,
            mx64: wide("w4a4_gemv_mx64"),
            mx64_nt2: wide("w4a4_gemv_mx64_nt2"),
            aq: gpu.alloc(m * k / 2)?,
            a_scale: gpu.alloc(m * k / 16)?,
            a_gs: gpu.alloc(m * 4)?,
            audit_ref: if audit_enabled() {
                gpu.alloc(m * AUDIT_MAX_N * 2)?
            } else {
                DevicePtr::NULL
            },
        })
    } else {
        tracing::warn!(
            "--w4a4-downcast is on but the w4a4_gemv_mx kernels are not in this target; projections stay W4A16"
        );
        None
    };
    guard.push((key(gpu), state));
    Ok(())
}

fn state(gpu: &dyn GpuBackend) -> Option<W4a4State> {
    let guard = cache().lock().unwrap_or_else(|p| p.into_inner());
    guard
        .iter()
        .find(|(k, _)| *k == key(gpu))
        .and_then(|(_, s)| *s)
}

/// Row edge of the projection sites' narrow arms (GDN qkvz/out_proj, both
/// dense-FFN arms): 32 under `--w4a4-downcast` (the `w4a4_gemv_mx32` reach,
/// with `w4a16_gemv_batch16/32` as the W4A16 fallback handles), else the
/// W4A16 edge [`super::gemv_tc::narrow_gemv_max_rows`]. The lm_head verify
/// arm deliberately keeps the W4A16 edge.
pub fn proj_max_rows() -> u32 {
    if w4a4_downcast_enabled() {
        w4a4_max_m()
    } else {
        super::gemv_tc::narrow_gemv_max_rows()
    }
}

/// Row edge of the two dense-FFN narrow arms: [`proj_max_rows`], but never
/// past 32. `--w4a4-downcast-wide` leaves the 33..=64-row dense FFN on its
/// W4A4 MMQ (a real tiled GEMM with shared-memory activation reuse), which
/// measured cheaper there than the GEMV twin: dgx1 C=32, GPU-rail J/token
/// 0.192 with the FFN on the MMQ vs 0.203 with it on `mx64_nt2`.
pub fn ffn_proj_max_rows() -> u32 {
    proj_max_rows().min(W4A4_MAX_M)
}

/// PURE: may the W4A4 path serve this launch? `wide` = `--w4a4-downcast-wide`.
pub fn w4a4_route(m: u32, n: u32, k: u32, enabled: bool, wide: bool) -> bool {
    let max_m = if wide { W4A4_WIDE_MAX_M } else { W4A4_MAX_M };
    enabled && (1..=max_m).contains(&m) && n > 0 && k > 0 && k.is_multiple_of(64) && k <= W4A4_MAX_K
}

/// The projection launcher: W4A4 FP4 MMA when opted in and prepared, else
/// the W4A16 `w4a16_gemv_batchm` (which itself prefers the tensor-core GEMV).
#[allow(clippy::too_many_arguments)]
#[track_caller]
pub fn nvfp4_proj_small_m(
    gpu: &dyn GpuBackend,
    batch_kernel: KernelHandle,
    input: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    proj(
        gpu,
        batch_kernel,
        input,
        weight,
        output,
        m,
        n,
        k,
        stream,
        false,
    )
}

/// [`nvfp4_proj_small_m`] for a projection whose `input` is BYTE-FOR-BYTE the
/// input of the immediately preceding projection on this stream (attention
/// k/v after q, FFN up after gate). It skips re-quantising when the previous
/// W4A4 launch quantised exactly `(input, m, k)` on this stream, and otherwise
/// quantises as usual, so a wrong claim about the ADDRESS can never read a
/// stale quantisation. The CALLER guarantees the CONTENTS did not change in
/// between.
#[allow(clippy::too_many_arguments)]
#[track_caller]
pub fn nvfp4_proj_small_m_same_input(
    gpu: &dyn GpuBackend,
    batch_kernel: KernelHandle,
    input: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
) -> Result<()> {
    proj(
        gpu,
        batch_kernel,
        input,
        weight,
        output,
        m,
        n,
        k,
        stream,
        true,
    )
}

/// What the scratch currently holds: (backend, input address, m, k, stream).
type QuantKey = (usize, u64, u32, u32, u64);

fn last_quant() -> &'static Mutex<Option<QuantKey>> {
    static LAST: OnceLock<Mutex<Option<QuantKey>>> = OnceLock::new();
    LAST.get_or_init(|| Mutex::new(None))
}

#[allow(clippy::too_many_arguments)]
#[track_caller]
fn proj(
    gpu: &dyn GpuBackend,
    batch_kernel: KernelHandle,
    input: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
    stream: u64,
    same_input: bool,
) -> Result<()> {
    if w4a4_route(m, n, k, w4a4_downcast_enabled(), w4a4_wide_enabled())
        && let Some(s) = state(gpu)
    {
        let want: QuantKey = (key(gpu), input.0, m, k, stream);
        let mut last = last_quant().lock().unwrap_or_else(|p| p.into_inner());
        if !(same_input && *last == Some(want)) {
            KernelLaunch::new(gpu, s.quant)
                .grid([m, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(input)
                .arg_ptr(s.aq)
                .arg_ptr(s.a_scale)
                .arg_ptr(s.a_gs)
                .arg_u32(k)
                .launch(stream)?;
            *last = Some(want);
        }
        drop(last);
        let (mx, grid, smem, sst) = match mx_plan(&s, m, n, k, mx_nt(), mx_ps()) {
            MxLaunch::Tiles {
                kernel,
                rows_per_cta,
            } => (kernel, div_ceil(n, rows_per_cta), 0, None),
            MxLaunch::Persistent { kernel, sst, smem } => (kernel, s.sms, smem, Some(sst)),
        };
        anyhow::ensure!(
            mx.0 != 0,
            "w4a4: no kernel for {m} rows (--w4a4-downcast-wide kernels missing)"
        );
        let launch = KernelLaunch::new(gpu, mx)
            .grid([grid, 1, 1])
            .block([256, 1, 1])
            .shared_mem(smem)
            .arg_ptr(s.aq)
            .arg_ptr(s.a_scale)
            .arg_ptr(s.a_gs)
            .arg_ptr(weight.weight)
            .arg_ptr(weight.weight_scale)
            .arg_f32(weight.weight_scale_2)
            .arg_ptr(output)
            .arg_u32(m)
            .arg_u32(n)
            .arg_u32(k);
        match sst {
            Some(sst) => launch.arg_u32(sst).launch(stream)?,
            None => launch.launch(stream)?,
        }
        if audit_enabled()
            && !s.audit_ref.is_null()
            && (n as usize) <= AUDIT_MAX_N
            && !gpu.stream_is_capturing(stream)
        {
            let site = std::panic::Location::caller();
            super::w4a16_gemv_batchm(
                gpu,
                batch_kernel,
                input,
                weight,
                s.audit_ref,
                m,
                n,
                k,
                stream,
            )?;
            audit(
                gpu,
                output,
                s.audit_ref,
                (m * n) as usize,
                stream,
                site,
                n,
                k,
            )?;
        }
        return Ok(());
    }
    anyhow::ensure!(
        m <= W4A4_MAX_M,
        "w4a4: {m} rows reached the W4A16 fallback, which covers at most {W4A4_MAX_M}"
    );
    super::w4a16_gemv_batchm(gpu, batch_kernel, input, weight, output, m, n, k, stream)
}

/// Per-site relative-error accumulator for [`audit_enabled`].
#[allow(clippy::too_many_arguments)]
fn audit(
    gpu: &dyn GpuBackend,
    got: DevicePtr,
    reference: DevicePtr,
    elems: usize,
    stream: u64,
    site: &'static std::panic::Location<'static>,
    n: u32,
    k: u32,
) -> Result<()> {
    type Acc = std::collections::HashMap<String, (u64, f64, f64)>;
    static ACC: OnceLock<Mutex<Acc>> = OnceLock::new();
    gpu.synchronize(stream)?;
    let mut a = vec![0u8; elems * 2];
    let mut b = vec![0u8; elems * 2];
    gpu.copy_d2h(got, &mut a)?;
    gpu.copy_d2h(reference, &mut b)?;
    let bf = |x: &[u8]| f32::from_bits((u16::from_le_bytes([x[0], x[1]]) as u32) << 16) as f64;
    let (mut num, mut den) = (0f64, 0f64);
    for (x, y) in a.chunks_exact(2).zip(b.chunks_exact(2)) {
        let (x, y) = (bf(x), bf(y));
        num += (x - y) * (x - y);
        den += y * y;
    }
    let rel = if den > 0.0 { (num / den).sqrt() } else { 0.0 };
    let key = format!("{}:{} N={n} K={k}", site.file(), site.line());
    let mut acc = ACC
        .get_or_init(|| Mutex::new(Acc::new()))
        .lock()
        .unwrap_or_else(|p| p.into_inner());
    let e = acc.entry(key.clone()).or_insert((0, 0.0, 0.0));
    e.0 += 1;
    e.1 += rel;
    e.2 = e.2.max(rel);
    if e.0.is_multiple_of(64) {
        tracing::info!(
            "W4A4_AUDIT site={key} samples={} mean_rel={:.5} max_rel={:.5}",
            e.0,
            e.1 / e.0 as f64,
            e.2
        );
    }
    Ok(())
}

#[cfg(test)]
#[path = "w4a4_proj_tests.rs"]
mod w4a4_proj_tests;
