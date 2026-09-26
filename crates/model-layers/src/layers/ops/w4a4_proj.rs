// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Opt-in W4A4 small-M projection on FP4 tensor cores
//! (`w4a4_gemv_mx.cu`, module `w4a4_gemv_mx`), under `--w4a4-downcast`.
//!
//! The block-scale FP4 MMA (`kind::mxf4nvf4`) takes the checkpoint's NVFP4
//! weight bytes and E4M3 group scales as operands, with no dequant. The
//! activations are first quantised per row to NVFP4 with a per-row FP32 global
//! scale (`w4a4_quant_rows`), so the numerics become W4A4. That is an accuracy
//! change, which is why the path is opt-in.
//!
//! Scope: the projection sites that call [`nvfp4_proj_small_m`]: GDN
//! qkvz/out_proj, attention q/k/v/o and dense-FFN gate/up/down, up to
//! [`w4a4_max_m`] rows. The lm_head does not call it.
//!
//! Scratch: the quantised activations need `[M, K/2] + [M, K/16] + [M] f32` of
//! device memory. [`prepare`] allocates it once per backend at model build
//! (from `W4a16BatchmTiers::resolve`). Launches run on the caller's stream,
//! and the scratch and the last-quantisation record are shared, so two
//! projections on different streams at once would race on them.
//!
//! Owner: model-layers ops.
//! Invariants:
//! - Device scratch is allocated only in [`prepare`], never by a launch.

use std::sync::{Mutex, OnceLock};

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};

use crate::weight_map::QuantizedWeight;

/// 2026-09-25: Rows the narrow entries (`w4a4_gemv_mx32`) cover.
pub const W4A4_MAX_M: u32 = 32;
/// 2026-09-25: Rows the wide entries (`w4a4_gemv_mx64*`, `--w4a4-downcast-wide`)
/// cover.
pub const W4A4_WIDE_MAX_M: u32 = 64;
/// 2026-09-25: Largest K the scratch is sized for.
pub const W4A4_MAX_K: u32 = 32768;

/// 2026-09-25: `--w4a4-downcast`, as published by the serve. It is read by this
/// module's projection path and by `DenseFfn::forward_k2` / `forward_k3`, which
/// under it take the batched `forward_km` arm. There is no environment
/// fallback. The serve publishes it before the model is built; anything that
/// reads it first (a test, an example) fixes it at false.
static W4A4_DOWNCAST: OnceLock<bool> = OnceLock::new();

/// 2026-09-25: Publish `--w4a4-downcast`. Returns the value in force: the first
/// publication or read wins, and a caller that gets a different value should
/// warn.
pub fn set_w4a4_downcast_from_cli(on: bool) -> bool {
    let _ = W4A4_DOWNCAST.set(on);
    *W4A4_DOWNCAST.get().expect("just set")
}

/// 2026-09-25: `--w4a4-downcast` in force? False unless the serve published true.
pub fn w4a4_downcast_enabled() -> bool {
    *W4A4_DOWNCAST.get_or_init(|| false)
}

/// 2026-09-25: `--w4a4-downcast-wide`, as published by the serve. Same rules as
/// [`W4A4_DOWNCAST`]; the serve publishes `downcast && wide`.
static W4A4_WIDE: OnceLock<bool> = OnceLock::new();

/// 2026-09-25: Publish `--w4a4-downcast-wide`. Returns the value in force.
pub fn set_w4a4_wide_from_cli(on: bool) -> bool {
    let _ = W4A4_WIDE.set(on);
    *W4A4_WIDE.get().expect("just set")
}

/// 2026-09-25: `--w4a4-downcast-wide` in force (and `--w4a4-downcast` with it)?
pub fn w4a4_wide_enabled() -> bool {
    w4a4_downcast_enabled() && *W4A4_WIDE.get_or_init(|| false)
}

/// 2026-09-25: Widest row count the W4A4 projection path serves in this process.
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
    /// 2026-09-25: Activation-reuse twins (`METRALE_W4A4_MX_NT`). The
    /// `w4a4_gemv_nt_oracle` example compares them bit for bit with mx16/mx32.
    mx16_nt2: KernelHandle,
    mx32_nt4: KernelHandle,
    /// 2026-09-25: Persistent activation-staged entries (`METRALE_W4A4_MX_PS`),
    /// compared bit for bit with mx16/mx32 by the same example.
    mx16_ps: KernelHandle,
    mx32_ps: KernelHandle,
    /// 2026-09-25: Streaming multiprocessors: the persistent entries' grid.
    sms: u32,
    /// 2026-09-25: 33..=64 rows under `--w4a4-downcast-wide` (zero handles
    /// otherwise).
    mx64: KernelHandle,
    mx64_nt2: KernelHandle,
    aq: DevicePtr,
    a_scale: DevicePtr,
    a_gs: DevicePtr,
    /// 2026-09-25: W4A16 reference output for [`audit_enabled`] (null
    /// otherwise).
    audit_ref: DevicePtr,
}

/// 2026-09-25: Largest projection N the audit reference buffer holds; a wider
/// launch is not audited.
const AUDIT_MAX_N: usize = 65536;

/// 2026-09-25: `METRALE_W4A4_PROJ_AUDIT` (non-empty), a diagnostic. Every W4A4
/// projection launched outside a graph capture also runs the W4A16 path,
/// synchronises, and accumulates `||y_w4a4 - y_w4a16|| / ||y_w4a16||` per call
/// site, logged every 64 samples. Captured launches are not audited, so it
/// needs graphs off (`METRALE_NO_MTP_VERIFY_GRAPHS=1
/// METRALE_NO_DECODE_GRAPHS_MULTISEQ=1`) to see the decode and verify steps.
fn audit_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("METRALE_W4A4_PROJ_AUDIT").is_some_and(|v| !v.is_empty()))
}

// 2026-09-25: `w4a4_proj.rs` is loaded via `#[path = "ops/w4a4_proj.rs"]`, so
// the explicit `#[path]` is required to nest the submodule under this file.
#[path = "w4a4_proj/mx_plan.rs"]
mod mx_plan;
pub use mx_plan::*;

fn cache() -> &'static Mutex<Vec<(usize, Option<W4a4State>)>> {
    static CACHE: OnceLock<Mutex<Vec<(usize, Option<W4a4State>)>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(Vec::new()))
}

fn key(gpu: &dyn GpuBackend) -> usize {
    gpu as *const dyn GpuBackend as *const () as usize
}

/// 2026-09-25: Resolve the kernels and allocate the scratch once per backend,
/// when `--w4a4-downcast` is on. Call at model build, never inside a graph
/// capture. When a kernel is missing it records `None` for the backend and
/// logs a warning, and every projection stays W4A16.
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

/// 2026-09-25: Row edge of the narrow projection arms (GDN qkvz/out_proj, the
/// dense FFN through [`ffn_proj_max_rows`], and `W4a16BatchmTiers::kernel`):
/// [`w4a4_max_m`] under `--w4a4-downcast`, else the W4A16 edge
/// [`super::gemv_tc::narrow_gemv_max_rows`]. The lm_head reads
/// `narrow_gemv_max_rows` directly.
pub fn proj_max_rows() -> u32 {
    if w4a4_downcast_enabled() {
        w4a4_max_m()
    } else {
        super::gemv_tc::narrow_gemv_max_rows()
    }
}

/// 2026-09-25: Row edge of the two dense-FFN narrow arms: [`proj_max_rows`],
/// capped at [`W4A4_MAX_M`], so under `--w4a4-downcast-wide` the dense FFN's
/// 33..=64-row steps do not take the narrow arm.
pub fn ffn_proj_max_rows() -> u32 {
    proj_max_rows().min(W4A4_MAX_M)
}

/// 2026-09-25: Pure: may the W4A4 path serve this launch? `wide` is
/// `--w4a4-downcast-wide`.
pub fn w4a4_route(m: u32, n: u32, k: u32, enabled: bool, wide: bool) -> bool {
    let max_m = if wide { W4A4_WIDE_MAX_M } else { W4A4_MAX_M };
    enabled && (1..=max_m).contains(&m) && n > 0 && k > 0 && k.is_multiple_of(64) && k <= W4A4_MAX_K
}

/// 2026-09-25: The projection launcher: the W4A4 FP4 MMA when opted in,
/// prepared and [`w4a4_route`] admits the shape, else the W4A16
/// `w4a16_gemv_batchm` (which tries the tensor-core GEMV first). The W4A16
/// fallback returns an error above [`W4A4_MAX_M`] rows.
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

/// 2026-09-25: [`nvfp4_proj_small_m`] for a projection whose `input` is byte for
/// byte the input of the preceding projection on this stream (attention k/v
/// after q, FFN up after gate). It skips re-quantising when the previous W4A4
/// quantisation was of exactly `(backend, input, m, k, stream)`, and otherwise
/// quantises as usual, so a wrong claim about the address cannot read a stale
/// quantisation. The caller guarantees the contents did not change in between.
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

/// 2026-09-25: What the scratch holds: (backend, input address, m, k, stream).
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

/// 2026-09-25: Per-site relative-error accumulator for [`audit_enabled`].
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
