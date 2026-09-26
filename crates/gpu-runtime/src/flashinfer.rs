// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Host-side FFI to the FlashInfer ragged (varlen) BF16 prefill
//! attention wrapper in `cuda/flashinfer_ragged_prefill.cu`: several requests'
//! attention in one launch, delimited by `qo_indptr`/`kv_indptr`. `hd128`
//! holds the head_dim-128 entry.
//!
//! Owner: gpu-runtime.
//! Invariants:
//! - The FFI and workspaces exist only under `cfg(metrale_flashinfer)`, which
//!   `build.rs` sets only when `FLASHINFER_HOME` is set at build time. Without
//!   it `available()` is false and every entry point returns an error.
//! - The only caller, the batched first-chunk prefill in metrale-model-layers
//!   (`qwen3_attention/prefill/paged.rs`), also requires
//!   `METRALE_FLASHINFER_PREFILL=1`.

use anyhow::{Result, bail};

mod hd128;
pub use hd128::ragged_prefill_bf16_hd128;

#[cfg(metrale_flashinfer)]
use std::ffi::c_void;
#[cfg(metrale_flashinfer)]
use std::sync::OnceLock;

#[cfg(metrale_flashinfer)]
unsafe extern "C" {
    // 2026-09-25: Declared but not called: the workspaces use fixed budgets.
    #[allow(dead_code)]
    fn metrale_fi_ragged_prefill_workspace_sizes(
        max_batch: u32,
        max_total_qo_rows: u32,
        num_qo_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        float_ws_bytes_out: *mut usize,
        int_ws_bytes_out: *mut usize,
        pinned_int_ws_bytes_out: *mut usize,
    ) -> i32;

    #[allow(clippy::too_many_arguments)]
    fn metrale_fi_ragged_prefill_bf16_hd256(
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        o: *mut c_void,
        qo_indptr_h: *const i32,
        kv_indptr_h: *const i32,
        qo_indptr_d: *const i32,
        kv_indptr_d: *const i32,
        batch: u32,
        total_qo_rows: u32,
        total_kv_rows: u32,
        num_qo_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        sm_scale: f32,
        causal: i32,
        float_ws: *mut c_void,
        float_ws_bytes: usize,
        int_ws: *mut c_void,
        int_ws_bytes: usize,
        pinned_int_ws: *mut c_void,
        pinned_int_ws_bytes: usize,
        stream: *mut c_void,
    ) -> i32;

    #[cfg(metrale_flashinfer)]
    fn cuMemAlloc_v2(dptr: *mut u64, bytesize: usize) -> i32;
    #[cfg(metrale_flashinfer)]
    fn cudaHostAlloc(ptr: *mut *mut c_void, size: usize, flags: u32) -> i32;
}

/// 2026-09-25: Whether the FlashInfer wrapper was compiled in (`FLASHINFER_HOME`
/// was set at build time).
pub fn available() -> bool {
    cfg!(metrale_flashinfer)
}

// 2026-09-25: Workspaces allocated on first use and reused; each call plans
// into them afresh (`PrefillPlan` in the wrapper), so no state carries over.
#[cfg(metrale_flashinfer)]
struct Workspaces {
    float_ws: u64,
    int_ws: u64,
    pinned_int_ws: u64,
    float_sz: usize,
    int_sz: usize,
    pinned_sz: usize,
}
#[cfg(metrale_flashinfer)]
unsafe impl Send for Workspaces {}
#[cfg(metrale_flashinfer)]
unsafe impl Sync for Workspaces {}
#[cfg(metrale_flashinfer)]
/// 2026-09-25: Never freed. Static like the process CUDA context
/// (`crate::cuda_host`): the sizes are the fixed `*_WS_BYTES` budgets, not taken
/// from any model, so the workspaces are kept across model loads.
static WS: OnceLock<Workspaces> = OnceLock::new();

#[cfg(metrale_flashinfer)]
const MAX_BATCH: u32 = 16;
#[cfg(metrale_flashinfer)]
const MAX_TOTAL_QO_ROWS: u32 = 16 * 16384;
#[cfg(metrale_flashinfer)]
const N_QO_HEADS: u32 = 16;
#[cfg(metrale_flashinfer)]
const N_KV_HEADS: u32 = 2;
#[cfg(metrale_flashinfer)]
const HEAD_DIM: u32 = 256;

// 2026-09-25: Workspace budgets passed to `PrefillPlan`: the float workspace
// holds split-KV partial results, the int and pinned ones its scheduler metadata.
#[cfg(metrale_flashinfer)]
const FLOAT_WS_BYTES: usize = 256 << 20;
#[cfg(metrale_flashinfer)]
const INT_WS_BYTES: usize = 64 << 20;
#[cfg(metrale_flashinfer)]
const PINNED_WS_BYTES: usize = 64 << 20;

#[cfg(metrale_flashinfer)]
fn workspaces() -> Result<&'static Workspaces> {
    if let Some(w) = WS.get() {
        return Ok(w);
    }
    let _ = (
        MAX_BATCH,
        MAX_TOTAL_QO_ROWS,
        N_QO_HEADS,
        N_KV_HEADS,
        HEAD_DIM,
    );
    let (fsz, isz, psz) = (FLOAT_WS_BYTES, INT_WS_BYTES, PINNED_WS_BYTES);
    let mut float_ws = 0u64;
    let mut int_ws = 0u64;
    let mut pinned = std::ptr::null_mut::<c_void>();
    unsafe {
        let s1 = cuMemAlloc_v2(&mut float_ws, fsz.max(1));
        if s1 != 0 {
            bail!("cuMemAlloc FlashInfer float ws ({fsz}B) failed: {s1}");
        }
        let s2 = cuMemAlloc_v2(&mut int_ws, isz.max(1));
        if s2 != 0 {
            bail!("cuMemAlloc FlashInfer int ws ({isz}B) failed: {s2}");
        }
        let s3 = cudaHostAlloc(&mut pinned, psz.max(1), 0);
        if s3 != 0 {
            bail!("cudaHostAlloc FlashInfer pinned int ws ({psz}B) failed: {s3}");
        }
    }
    let _ = WS.set(Workspaces {
        float_ws,
        int_ws,
        pinned_int_ws: pinned as u64,
        float_sz: fsz,
        int_sz: isz,
        pinned_sz: psz,
    });
    Ok(WS.get().unwrap())
}

/// 2026-09-25: Ragged batched prefill attention, BF16, head_dim 256, GQA,
/// optionally causal.
///
/// `q`/`o`: `[total_qo_rows, num_qo_heads, 256]` BF16 on the device; `k`/`v`:
/// `[total_kv_rows, num_kv_heads, 256]`. `qo_indptr`/`kv_indptr` are `[batch+1]`
/// int32 prefix sums, on the host (`*_h`, read by `PrefillPlan`) and on the
/// device (`*_d`, read by the kernel). Errors when `head_dim != 256`, when a host
/// indptr slice is not `batch + 1` long, or on a non-zero status.
#[allow(clippy::too_many_arguments)]
pub fn ragged_prefill_bf16_hd256(
    q: u64,
    k: u64,
    v: u64,
    o: u64,
    qo_indptr_h: &[i32],
    kv_indptr_h: &[i32],
    qo_indptr_d: u64,
    kv_indptr_d: u64,
    batch: u32,
    total_qo_rows: u32,
    total_kv_rows: u32,
    num_qo_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    sm_scale: f32,
    causal: bool,
    stream: u64,
) -> Result<()> {
    #[cfg(metrale_flashinfer)]
    {
        if head_dim != HEAD_DIM {
            bail!("FlashInfer wrapper is head_dim=256 only (got {head_dim})");
        }
        if qo_indptr_h.len() != (batch + 1) as usize || kv_indptr_h.len() != (batch + 1) as usize {
            bail!("indptr host slices must be batch+1 long");
        }
        let ws = workspaces()?;
        let st = unsafe {
            metrale_fi_ragged_prefill_bf16_hd256(
                q as *const c_void,
                k as *const c_void,
                v as *const c_void,
                o as *mut c_void,
                qo_indptr_h.as_ptr(),
                kv_indptr_h.as_ptr(),
                qo_indptr_d as *const i32,
                kv_indptr_d as *const i32,
                batch,
                total_qo_rows,
                total_kv_rows,
                num_qo_heads,
                num_kv_heads,
                head_dim,
                sm_scale,
                if causal { 1 } else { 0 },
                ws.float_ws as *mut c_void,
                ws.float_sz,
                ws.int_ws as *mut c_void,
                ws.int_sz,
                ws.pinned_int_ws as *mut c_void,
                ws.pinned_sz,
                stream as *mut c_void,
            )
        };
        if st != 0 {
            bail!(
                "FlashInfer ragged prefill failed: status {st} (batch={batch}, qo={total_qo_rows})"
            );
        }
        Ok(())
    }
    #[cfg(not(metrale_flashinfer))]
    {
        let _ = (
            q,
            k,
            v,
            o,
            qo_indptr_h,
            kv_indptr_h,
            qo_indptr_d,
            kv_indptr_d,
            batch,
            total_qo_rows,
            total_kv_rows,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            sm_scale,
            causal,
            stream,
        );
        bail!("FlashInfer support was not built; set FLASHINFER_HOME when building")
    }
}

#[cfg(all(test, metrale_flashinfer))]
mod tests {
    use super::*;
    use std::ffi::c_void;

    const H2D: i32 = 1;
    const D2H: i32 = 2;
    unsafe extern "C" {
        fn cudaMalloc(p: *mut *mut c_void, n: usize) -> i32;
        fn cudaFree(p: *mut c_void) -> i32;
        fn cudaMemcpy(d: *mut c_void, s: *const c_void, n: usize, k: i32) -> i32;
        fn cudaDeviceSynchronize() -> i32;
    }
    fn f32_to_bf16(x: f32) -> u16 {
        let b = x.to_bits();
        ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
    }
    fn bf16_to_f32(x: u16) -> f32 {
        f32::from_bits((x as u32) << 16)
    }
    unsafe fn dev<T>(data: &[T]) -> u64 {
        let bytes = std::mem::size_of_val(data);
        let mut p = std::ptr::null_mut();
        assert_eq!(unsafe { cudaMalloc(&mut p, bytes.max(1)) }, 0);
        assert_eq!(
            unsafe { cudaMemcpy(p, data.as_ptr() as *const c_void, bytes, H2D) },
            0
        );
        p as u64
    }

    #[test]
    #[ignore = "requires a free CUDA device + FLASHINFER_HOME build"]
    #[allow(clippy::needless_range_loop)] // 2026-09-25: index loops mirror the reference math.
    fn flashinfer_ragged_prefill_matches_cpu_reference() {
        const HD: usize = 256;
        const NQO: usize = 4;
        const NKV: usize = 2;
        let lens = [6usize, 10usize];
        let qo_indptr: Vec<i32> = {
            let mut v = vec![0i32];
            for &l in &lens {
                v.push(v.last().unwrap() + l as i32);
            }
            v
        };
        let kv_indptr = qo_indptr.clone();
        let total: usize = lens.iter().sum();
        let sm_scale = 1.0f32 / (HD as f32).sqrt();

        let rnd = |seed: u64| -> f32 {
            let mut x = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
            x ^= x >> 31;
            x = x.wrapping_mul(0xBF58476D1CE4E5B9);
            ((x >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.5
        };
        let q: Vec<u16> = (0..total * NQO * HD)
            .map(|i| f32_to_bf16(rnd(i as u64)))
            .collect();
        let k: Vec<u16> = (0..total * NKV * HD)
            .map(|i| f32_to_bf16(rnd(i as u64 ^ 0x1111)))
            .collect();
        let v: Vec<u16> = (0..total * NKV * HD)
            .map(|i| f32_to_bf16(rnd(i as u64 ^ 0x2222)))
            .collect();
        let mut o = vec![0u16; total * NQO * HD];

        let (q_d, k_d, v_d, o_d, qo_d, kv_d);
        unsafe {
            q_d = dev(&q);
            k_d = dev(&k);
            v_d = dev(&v);
            o_d = dev(&o);
            qo_d = dev(&qo_indptr);
            kv_d = dev(&kv_indptr);
        }

        ragged_prefill_bf16_hd256(
            q_d,
            k_d,
            v_d,
            o_d,
            &qo_indptr,
            &kv_indptr,
            qo_d,
            kv_d,
            lens.len() as u32,
            total as u32,
            total as u32,
            NQO as u32,
            NKV as u32,
            HD as u32,
            sm_scale,
            true,
            0,
        )
        .unwrap();
        unsafe {
            assert_eq!(cudaDeviceSynchronize(), 0);
            assert_eq!(
                cudaMemcpy(
                    o.as_mut_ptr() as *mut c_void,
                    o_d as *const c_void,
                    o.len() * 2,
                    D2H
                ),
                0
            );
        }

        // 2026-09-25: CPU reference: per-request causal GQA attention.
        let group = NQO / NKV;
        let qf = |r: usize, h: usize, d: usize| bf16_to_f32(q[(r * NQO + h) * HD + d]);
        let kf = |r: usize, kh: usize, d: usize| bf16_to_f32(k[(r * NKV + kh) * HD + d]);
        let vf = |r: usize, kh: usize, d: usize| bf16_to_f32(v[(r * NKV + kh) * HD + d]);
        let mut max_rel = 0.0f64;
        let mut worst_cos = 1.0f64;
        for (b, &len) in lens.iter().enumerate() {
            let start = qo_indptr[b] as usize;
            for qi in 0..len {
                for h in 0..NQO {
                    let kh = h / group;
                    let mut scores = vec![0f32; qi + 1];
                    for j in 0..=qi {
                        let mut s = 0.0f32;
                        for d in 0..HD {
                            s += qf(start + qi, h, d) * kf(start + j, kh, d);
                        }
                        scores[j] = s * sm_scale;
                    }
                    let mx = scores.iter().cloned().fold(f32::MIN, f32::max);
                    let mut den = 0.0f32;
                    for s in &mut scores {
                        *s = (*s - mx).exp();
                        den += *s;
                    }
                    let mut out_ref = vec![0f32; HD];
                    for (j, &p) in scores.iter().enumerate() {
                        let w = p / den;
                        for d in 0..HD {
                            out_ref[d] += w * vf(start + j, kh, d);
                        }
                    }
                    let mut dot = 0.0f64;
                    let mut na = 0.0f64;
                    let mut nb = 0.0f64;
                    for d in 0..HD {
                        let g = bf16_to_f32(o[(start + qi) * NQO * HD + h * HD + d]) as f64;
                        let r = out_ref[d] as f64;
                        dot += g * r;
                        na += g * g;
                        nb += r * r;
                        max_rel = max_rel.max((g - r).abs() / (r.abs() + 1e-3));
                    }
                    let cos = dot / (na.sqrt() * nb.sqrt() + 1e-12);
                    worst_cos = worst_cos.min(cos);
                }
            }
        }
        unsafe {
            for p in [q_d, k_d, v_d, o_d, qo_d, kv_d] {
                cudaFree(p as *mut c_void);
            }
        }
        tracing::debug!("FLASHINFER_RAGGED worst_cos={worst_cos:.6} max_rel={max_rel:.4}");
        assert!(
            worst_cos > 0.99,
            "FlashInfer ragged prefill diverges from CPU ref: cos {worst_cos}"
        );
    }
}
