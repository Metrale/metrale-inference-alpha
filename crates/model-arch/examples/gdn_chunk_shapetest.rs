// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Accuracy and timing of a GDN state-spine challenger (default
//! `gated_delta_rule_chunk_delta_h_tc_vblock`) against `gated_delta_rule_chunk_delta_h_ksplit`.
//!
//! Owner: model-arch examples.
//! Invariants:
//! - Exit 1 unless, for every t, the challenger's S_c, uc and final state each reach
//!   cosine >= `COS_GATE` and norm-ratio deviation < `NRDEV_GATE` against ksplit.
//!
//! The comparison is by cosine and norm ratio rather than bytes: tc_vblock computes
//! W S_c on tensor cores from a BF16 copy of the state. The norm ratio |‖new‖/‖ref‖ - 1| is
//! scale-sensitive where the cosine is not.
//!
//! Per t (2048, 8192, 16384; batch 1):
//!   1. recompute_wu computes W, U and gc once per leg.
//!   2. ksplit: grid [NV, batch, 1], block 256, `KSPLIT_SMEM` (99,336 B).
//!   3. the challenger: tc_vblock at grid [NV, NUM_DV_BLK * batch, 1], block 256,
//!      `TC_VBLOCK_SMEM` (82,952 B).
//!   4. S_c, uc and the final state compared by cosine and norm-ratio deviation.
//!   5. kernel-only CUDA-event timing, 8 warmup and 50 timed launches per leg.
//!
//! Run on a GB10 host:
//!   cargo run -p metrale-model-arch --release --features cuda,gpu-examples \
//!       --example gdn_chunk_shapetest

use anyhow::{Result, bail};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

const KD: usize = 128;
const VD: usize = 128;
const NK: usize = 16;
const NV: usize = 32;
const C: usize = 64;

// 2026-09-25: tc_vblock splits the 128 value columns into blocks of 64, folded into
// blockIdx.y: grid.y = NUM_DV_BLK * batch.
const DV_BLK: usize = 64;
const NUM_DV_BLK: usize = VD / DV_BLK;

/// 2026-09-25: The challenger entry: `METRALE_GDN_CHALLENGER=dvsplit` selects
/// `_dvsplit`, any value starting with `v` selects `gated_delta_rule_chunk_delta_h_<value>`,
/// and anything else `_tc_vblock`.
fn challenger_name() -> String {
    match std::env::var("METRALE_GDN_CHALLENGER").ok().as_deref() {
        Some("dvsplit") => "gated_delta_rule_chunk_delta_h_dvsplit".to_string(),
        Some(v) if v.starts_with('v') => format!("gated_delta_rule_chunk_delta_h_{v}"),
        _ => "gated_delta_rule_chunk_delta_h_tc_vblock".to_string(),
    }
}

/// 2026-09-25: Dynamic shared memory for the selected challenger.
fn challenger_smem() -> u32 {
    match std::env::var("METRALE_GDN_CHALLENGER").ok().as_deref() {
        Some("dvsplit") => (C * KD * 2 + C * KD * 2 + C * DV_BLK * 2 + (C + 1) * 4) as u32,
        Some(v) if v.starts_with('v') => (C * (KD * 4 + VD * 2) + (C + 1) * 4) as u32,
        _ => TC_VBLOCK_SMEM,
    }
}

const KSPLIT_SMEM: u32 = (2 * (C * (2 * KD + VD) * 2) + 2 * C * 4 + 2 * (C + 1) * 4) as u32;
// 2026-09-25: tc_vblock shared memory (82,952 B): St[DV_BLK*KD] BF16 (K reuses it),
// ws[CHUNK*DV_BLK] FP32, buf[2][CHUNK*KD + CHUNK*DV_BLK] BF16, gcb[2][CHUNK] FP32 and
// decb[2][CHUNK+1] FP32, the terms below in that order.
const TC_VBLOCK_SMEM: u32 = (DV_BLK * KD * 2
    + C * DV_BLK * 4
    + 2 * (C * KD + C * DV_BLK) * 2
    + 2 * C * 4
    + 2 * (C + 1) * 4) as u32;

const COS_GATE: f64 = 0.99;
const NRDEV_GATE: f64 = 0.05;

// 2026-09-25: CUDA driver event API, for kernel-only timing.
unsafe extern "C" {
    fn cuEventCreate(event: *mut u64, flags: u32) -> i32;
    fn cuEventRecord(event: u64, stream: u64) -> i32;
    fn cuEventSynchronize(event: u64) -> i32;
    fn cuEventElapsedTime(ms: *mut f32, start: u64, end: u64) -> i32;
    fn cuEventDestroy_v2(event: u64) -> i32;
}

struct Lcg(u64);
impl Lcg {
    fn f(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
    fn r(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.f()
    }
}

fn up_bf16(g: &dyn GpuBackend, d: &[bf16]) -> Result<DevicePtr> {
    let b: Vec<u8> = d.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect();
    let p = g.alloc(b.len())?;
    g.copy_h2d(&b, p)?;
    Ok(p)
}
fn up_f32(g: &dyn GpuBackend, d: &[f32]) -> Result<DevicePtr> {
    let b: Vec<u8> = d.iter().flat_map(|x| x.to_le_bytes()).collect();
    let p = g.alloc(b.len())?;
    g.copy_h2d(&b, p)?;
    Ok(p)
}
fn dn(g: &dyn GpuBackend, p: DevicePtr, n_bytes: usize) -> Result<Vec<u8>> {
    let mut b = vec![0u8; n_bytes];
    g.copy_d2h(p, &mut b)?;
    Ok(b)
}

fn bf16_bits_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// 2026-09-25: Decode a raw little-endian BF16 byte buffer to f64.
fn dn_bf16(bytes: &[u8]) -> Vec<f64> {
    bytes
        .chunks_exact(2)
        .map(|c| bf16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])) as f64)
        .collect()
}
/// 2026-09-25: Decode a raw little-endian f32 byte buffer to f64.
fn dn_f32(bytes: &[u8]) -> Vec<f64> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64)
        .collect()
}

/// 2026-09-25: cosine = dot / (‖a‖‖b‖); 0 when either norm is 0.
fn cosine(a: &[f64], b: &[f64]) -> f64 {
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if na > 0.0 && nb > 0.0 {
        dot / (na.sqrt() * nb.sqrt())
    } else {
        0.0
    }
}
/// 2026-09-25: Norm-ratio deviation |‖a‖/‖b‖ - 1|; 0 when ‖b‖ is 0.
fn norm_ratio_dev(a: &[f64], b: &[f64]) -> f64 {
    let (mut na, mut nb) = (0f64, 0f64);
    for i in 0..a.len() {
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if nb > 0.0 {
        (na.sqrt() / nb.sqrt() - 1.0).abs()
    } else {
        0.0
    }
}

struct Case {
    t: usize,
    nt: usize,
    batch: usize,
    key: Vec<bf16>,
    val: Vec<bf16>,
    gate: Vec<f32>,
    beta: Vec<f32>,
    h0: Vec<f32>,
}

fn gen_case(t: usize, batch: usize) -> Case {
    let nt = t.div_ceil(C);
    let (mut key, mut val, mut gate, mut beta, mut h0) = (vec![], vec![], vec![], vec![], vec![]);
    for bi in 0..batch {
        let mut r = Lcg(0xDE17A ^ ((t as u64) ^ (bi as u64).wrapping_mul(0x9E3779B9)));
        for _ in 0..t * NK * KD {
            key.push(bf16::from_f64(r.r(-0.5, 0.5)));
        }
        for _ in 0..t * NV * VD {
            val.push(bf16::from_f64(r.r(-0.5, 0.5)));
        }
        for _ in 0..t * NV {
            gate.push(r.r(0.80, 0.999) as f32);
        }
        for _ in 0..t * NV {
            beta.push(r.r(0.0, 1.0) as f32);
        }
        for _ in 0..NV * KD * VD {
            h0.push(r.r(-0.1, 0.1) as f32);
        }
    }
    Case {
        t,
        nt,
        batch,
        key,
        val,
        gate,
        beta,
        h0,
    }
}

// 2026-09-25: recompute_wu: W and U (BF16) and gc_out (FP32, read by the scan as gc_in),
// grid (nt, NV, batch), with the block size and shared memory of `smem_wu` in
// ops/ssm_gdn_a3.rs.
#[allow(clippy::too_many_arguments)]
fn run_wu(
    g: &dyn GpuBackend,
    k_wu: KernelHandle,
    c: &Case,
    kp: DevicePtr,
    vp: DevicePtr,
    gp: DevicePtr,
    bp: DevicePtr,
    wp: DevicePtr,
    up: DevicePtr,
    gcp: DevicePtr,
) -> Result<()> {
    let smem1 = (C * KD * 2 + C * C * 4 + C * 4) as u32;
    KernelLaunch::new(g, k_wu)
        .grid([c.nt as u32, NV as u32, c.batch as u32])
        .block([256, 1, 1])
        .shared_mem(smem1)
        .arg_ptr(kp)
        .arg_ptr(vp)
        .arg_ptr(gp)
        .arg_ptr(bp)
        .arg_ptr(wp)
        .arg_ptr(up)
        .arg_ptr(gcp)
        .arg_u32(c.batch as u32)
        .arg_u32(c.t as u32)
        .arg_u32(c.nt as u32)
        .arg_u32(NK as u32)
        .arg_u32(NV as u32)
        .arg_u32(KD as u32)
        .arg_u32(VD as u32)
        .arg_u32((NK * KD) as u32)
        .arg_u32((NV * VD) as u32)
        .arg_u32(NV as u32)
        .arg_ptr(DevicePtr::NULL)
        .arg_ptr(DevicePtr::NULL)
        .arg_u32(0)
        .launch(0)?;
    Ok(())
}

// 2026-09-25: The scan: `tc` = false launches ksplit, `tc` = true the challenger, both with
// the same arguments.
#[allow(clippy::too_many_arguments)]
fn launch_scan(
    g: &dyn GpuBackend,
    k: KernelHandle,
    c: &Case,
    hp: DevicePtr,
    wp: DevicePtr,
    up: DevicePtr,
    kp: DevicePtr,
    gp: DevicePtr,
    gcp: DevicePtr,
    scp: DevicePtr,
    ucp: DevicePtr,
    tc: bool,
    stream: u64,
) -> Result<()> {
    let (grid, smem) = if tc && challenger_name().contains("_delta_h_v") {
        // 2026-09-25: Challengers named `_delta_h_v*` (`_vfused`, `_vtile`) do not split
        // the value columns: grid y is batch.
        ([NV as u32, c.batch as u32, 1], challenger_smem())
    } else if tc {
        (
            [NV as u32, (NUM_DV_BLK * c.batch) as u32, 1],
            challenger_smem(),
        )
    } else {
        ([NV as u32, c.batch as u32, 1], KSPLIT_SMEM)
    };
    // 2026-09-25: `_vtile` runs 512 threads; the other spines run 256.
    let wide = challenger_name().ends_with("vtile");
    let block = if tc && wide { 512u32 } else { 256u32 };
    KernelLaunch::new(g, k)
        .grid(grid)
        .block([block, 1, 1])
        .shared_mem(smem)
        .arg_ptr(hp)
        .arg_ptr(wp)
        .arg_ptr(up)
        .arg_ptr(kp)
        .arg_ptr(gp)
        .arg_ptr(gcp)
        .arg_ptr(scp)
        .arg_ptr(ucp)
        .arg_u32(c.batch as u32)
        .arg_u32(c.t as u32)
        .arg_u32(c.nt as u32)
        .arg_u32(NK as u32)
        .arg_u32(NV as u32)
        .arg_u32(KD as u32)
        .arg_u32(VD as u32)
        .arg_u32((NK * KD) as u32)
        .arg_u32(NV as u32)
        .arg_u32(0)
        .arg_ptr(DevicePtr::NULL)
        .arg_ptr(DevicePtr::NULL)
        .arg_u32(0)
        .launch(stream)?;
    Ok(())
}

// 2026-09-25: One full run -> (S_c, uc, final state) bytes. h0 is uploaded fresh each call
// because the scan writes the final state over h_state.
fn run_full(
    g: &dyn GpuBackend,
    k_wu: KernelHandle,
    k_scan: KernelHandle,
    c: &Case,
    tc: bool,
) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    let kp = up_bf16(g, &c.key)?;
    let vp = up_bf16(g, &c.val)?;
    let gp = up_f32(g, &c.gate)?;
    let bp = up_f32(g, &c.beta)?;
    let gcp = g.alloc(c.batch * c.nt * NV * C * 4)?;
    let wp = g.alloc(c.batch * c.nt * NV * C * KD * 2)?;
    let up = g.alloc(c.batch * c.nt * NV * C * VD * 2)?;
    run_wu(g, k_wu, c, kp, vp, gp, bp, wp, up, gcp)?;
    let hp = up_f32(g, &c.h0)?;
    let scp = g.alloc(c.batch * c.nt * NV * KD * VD * 2)?;
    let ucp = g.alloc(c.batch * c.nt * NV * C * VD * 2)?;
    launch_scan(g, k_scan, c, hp, wp, up, kp, gp, gcp, scp, ucp, tc, 0)?;
    g.synchronize(0)?;
    let sc = dn(g, scp, c.batch * c.nt * NV * KD * VD * 2)?;
    let uc = dn(g, ucp, c.batch * c.nt * NV * C * VD * 2)?;
    let sf = dn(g, hp, c.batch * NV * KD * VD * 4)?;
    for p in [kp, vp, gp, bp, gcp, wp, up, hp, scp, ucp] {
        let _ = g.free(p);
    }
    Ok((sc, uc, sf))
}

// 2026-09-25: Kernel-only timing of the scan: W and U computed once, h0 not re-uploaded
// between launches (nothing is read back), 8 warmup launches, then `iters` timed.
fn time_scan(
    g: &dyn GpuBackend,
    k_wu: KernelHandle,
    k_scan: KernelHandle,
    c: &Case,
    tc: bool,
    iters: u32,
) -> Result<f64> {
    let kp = up_bf16(g, &c.key)?;
    let vp = up_bf16(g, &c.val)?;
    let gp = up_f32(g, &c.gate)?;
    let bp = up_f32(g, &c.beta)?;
    let gcp = g.alloc(c.batch * c.nt * NV * C * 4)?;
    let wp = g.alloc(c.batch * c.nt * NV * C * KD * 2)?;
    let up = g.alloc(c.batch * c.nt * NV * C * VD * 2)?;
    run_wu(g, k_wu, c, kp, vp, gp, bp, wp, up, gcp)?;
    let hp = up_f32(g, &c.h0)?;
    let scp = g.alloc(c.batch * c.nt * NV * KD * VD * 2)?;
    let ucp = g.alloc(c.batch * c.nt * NV * C * VD * 2)?;
    let s = g.create_stream()?;
    for _ in 0..8 {
        launch_scan(g, k_scan, c, hp, wp, up, kp, gp, gcp, scp, ucp, tc, s)?;
    }
    g.synchronize(s)?;
    let (mut e0, mut e1): (u64, u64) = (0, 0);
    let mut ms: f32 = 0.0;
    unsafe {
        if cuEventCreate(&mut e0, 0) != 0 || cuEventCreate(&mut e1, 0) != 0 {
            bail!("cuEventCreate");
        }
        if cuEventRecord(e0, s) != 0 {
            bail!("record start");
        }
    }
    for _ in 0..iters {
        launch_scan(g, k_scan, c, hp, wp, up, kp, gp, gcp, scp, ucp, tc, s)?;
    }
    unsafe {
        if cuEventRecord(e1, s) != 0 {
            bail!("record end");
        }
        if cuEventSynchronize(e1) != 0 {
            bail!("sync");
        }
        if cuEventElapsedTime(&mut ms, e0, e1) != 0 {
            bail!("elapsed");
        }
        cuEventDestroy_v2(e0);
        cuEventDestroy_v2(e1);
    }
    for p in [kp, vp, gp, bp, gcp, wp, up, hp, scp, ucp] {
        let _ = g.free(p);
    }
    Ok(ms as f64 / iters as f64)
}

/// 2026-09-25: (cos, nrdev) for a BF16 output (S_c, uc) decoded from raw bytes.
fn cmp_bf16(new: &[u8], reference: &[u8]) -> (f64, f64) {
    let a = dn_bf16(new);
    let b = dn_bf16(reference);
    (cosine(&a, &b), norm_ratio_dev(&a, &b))
}
/// 2026-09-25: (cos, nrdev) for an f32 output (final state) decoded from raw bytes.
fn cmp_f32(new: &[u8], reference: &[u8]) -> (f64, f64) {
    let a = dn_f32(new);
    let b = dn_f32(reference);
    (cosine(&a, &b), norm_ratio_dev(&a, &b))
}

fn main() -> Result<()> {
    let backend = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let k_wu = g.kernel("gated_delta_rule_fla", "gated_delta_rule_recompute_wu")?;
    let k_ref = g.kernel(
        "gated_delta_rule_fla",
        "gated_delta_rule_chunk_delta_h_ksplit",
    )?;
    let k_tc = g.kernel("gated_delta_rule_fla", &challenger_name())?;

    let iters = 50u32;
    let mut all_ok = true;

    println!(
        "=== GDN chunk_delta_h tc_vblock SHAPE TEST (cos>={COS_GATE} && nrdev<{NRDEV_GATE}) ==="
    );
    println!(
        "Holo GDN: KD={KD} VD={VD} NK={NK} NV={NV} C={C}; DV_BLK={DV_BLK} NUM_DV_BLK={NUM_DV_BLK}"
    );
    println!(
        "ksplit smem={KSPLIT_SMEM}  challenger={} smem={}",
        challenger_name(),
        challenger_smem()
    );
    println!(
        "{:>5} {:>5} | {:>16} | {:>16} | {:>16} | {:>10} | {}",
        "t", "batch", "S_c: cos nrdev", "uc: cos nrdev", "S_final: cos nrdev", "speedup", "result"
    );
    println!("{}", "-".repeat(96));

    for &t in &[2048usize, 8192, 16384] {
        for &batch in &[1usize] {
            let case = gen_case(t, batch);

            let (sc0, uc0, sf0) = run_full(g, k_wu, k_ref, &case, false)?;
            let (sc1, uc1, sf1) = run_full(g, k_wu, k_tc, &case, true)?;

            let (sc_cos, sc_nr) = cmp_bf16(&sc1, &sc0);
            let (uc_cos, uc_nr) = cmp_bf16(&uc1, &uc0);
            let (sf_cos, sf_nr) = cmp_f32(&sf1, &sf0);

            let pass = sc_cos >= COS_GATE
                && sc_nr < NRDEV_GATE
                && uc_cos >= COS_GATE
                && uc_nr < NRDEV_GATE
                && sf_cos >= COS_GATE
                && sf_nr < NRDEV_GATE;
            all_ok &= pass;

            let t_ref = time_scan(g, k_wu, k_ref, &case, false, iters)?;
            let t_new = time_scan(g, k_wu, k_tc, &case, true, iters)?;
            let speedup = if t_new > 0.0 { t_ref / t_new } else { 0.0 };

            println!(
                "{t:>5} {batch:>5} | ksplit {t_ref:>8.4}ms  tc_vblock {t_new:>8.4}ms | {speedup:>6.2}x | Sf_cos {sf_cos:>7.4} | {}",
                if pass { "PASS" } else { "FAIL" }
            );
        }
    }

    println!("{}", "-".repeat(96));
    if all_ok {
        println!("GDN-CHUNK RESULT: PASS (all cos>={COS_GATE} && nrdev<{NRDEV_GATE})");
        Ok(())
    } else {
        eprintln!("GDN-CHUNK RESULT: FAIL (some cos<{COS_GATE} or nrdev>={NRDEV_GATE})");
        std::process::exit(1);
    }
}
