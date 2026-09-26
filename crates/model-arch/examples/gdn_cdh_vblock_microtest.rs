// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Bit-parity and timing of `gated_delta_rule_chunk_delta_h_ksplit_vblock{2,4,8}`
//! against `gated_delta_rule_chunk_delta_h_ksplit`.
//!
//! Owner: model-arch examples.
//! Invariants:
//! - Exit 1 unless, for every (t, batch, VTILES), the vblock kernel's S_c, uc and final
//!   state bytes equal ksplit's on the same inputs.
//!
//! The vblock kernels add a value-column grid axis: grid (nv, VTILES, batch), each block
//! owning V_DIM / VTILES columns, against ksplit's (nv, batch). Timing is kernel-only
//! (CUDA events), printed per case as ms per iteration.
//!
//! Run on a GB10 host:
//!   cargo run -p metrale-model-arch --release --features cuda,gpu-examples \
//!       --example gdn_cdh_vblock_microtest

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

// 2026-09-25: VTILES -> (kernel entry, block_x = (VD / VTILES) * SPLIT), with SPLIT = 2.
fn vblock_kernel(vt: u32) -> (&'static str, u32) {
    match vt {
        2 => (
            "gated_delta_rule_chunk_delta_h_ksplit_vblock2",
            (VD as u32 / 2) * 2,
        ),
        4 => (
            "gated_delta_rule_chunk_delta_h_ksplit_vblock4",
            (VD as u32 / 4) * 2,
        ),
        8 => (
            "gated_delta_rule_chunk_delta_h_ksplit_vblock8",
            (VD as u32 / 8) * 2,
        ),
        _ => panic!("vtiles must be 2/4/8"),
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
        let _ = bi;
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
// grid (nt, NV, batch).
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
    let smem1 = (C * KD * 2 + C * C * 4 + C * C * 4 + C * 4) as u32;
    KernelLaunch::new(g, k_wu)
        .grid([c.nt as u32, NV as u32, c.batch as u32])
        .block([128, 1, 1])
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

// 2026-09-25: The scan: `vt` = 0 launches ksplit (grid [NV, batch, 1], block 256), any
// other `vt` a vblock kernel (grid [NV, vt, batch], block (VD / vt) * 2).
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
    vt: u32,
    stream: u64,
) -> Result<()> {
    // 2026-09-25: ksplit's dynamic shared memory (99,336 B at these dims); the vblock
    // kernels take the same.
    let smem = (2 * (C * (2 * KD + VD) * 2) + 2 * C * 4 + 2 * (C + 1) * 4) as u32;
    let (grid, block_x) = if vt == 0 {
        ([NV as u32, c.batch as u32, 1], 256u32)
    } else {
        ([NV as u32, vt, c.batch as u32], (VD as u32 / vt) * 2)
    };
    KernelLaunch::new(g, k)
        .grid(grid)
        .block([block_x, 1, 1])
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
    vt: u32,
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
    launch_scan(g, k_scan, c, hp, wp, up, kp, gp, gcp, scp, ucp, vt, 0)?;
    g.synchronize(0)?;
    let sc = dn(g, scp, c.batch * c.nt * NV * KD * VD * 2)?;
    let uc = dn(g, ucp, c.batch * c.nt * NV * C * VD * 2)?;
    let sf = dn(g, hp, c.batch * NV * KD * VD * 4)?;
    for p in [kp, vp, gp, bp, gcp, wp, up, hp, scp, ucp] {
        let _ = g.free(p);
    }
    Ok((sc, uc, sf))
}

// 2026-09-25: Kernel-only timing of the scan. W and U are computed once, and h0 is not
// re-uploaded between iterations since no output is read back.
#[allow(clippy::too_many_arguments)]
fn time_scan(
    g: &dyn GpuBackend,
    k_wu: KernelHandle,
    k_scan: KernelHandle,
    c: &Case,
    vt: u32,
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
        launch_scan(g, k_scan, c, hp, wp, up, kp, gp, gcp, scp, ucp, vt, s)?;
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
        launch_scan(g, k_scan, c, hp, wp, up, kp, gp, gcp, scp, ucp, vt, s)?;
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

fn main() -> Result<()> {
    let backend = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let k_wu = g.kernel("gated_delta_rule_fla", "gated_delta_rule_recompute_wu")?;
    let k_ksplit = g.kernel(
        "gated_delta_rule_fla",
        "gated_delta_rule_chunk_delta_h_ksplit",
    )?;

    let iters = 50u32;
    let mut all_ok = true;
    println!("=== GDN chunk_delta_h V-block A/B (bit-parity vs ksplit + perf) ===");
    for &t in &[128usize, 256] {
        for &batch in &[1usize, 2, 4] {
            let case = gen_case(t, batch);
            let (sc0, uc0, sf0) = run_full(g, k_wu, k_ksplit, &case, 0)?;
            let t_cur = time_scan(g, k_wu, k_ksplit, &case, 0, iters)?;
            for &vt in &[2u32, 4, 8] {
                let (kname, _) = vblock_kernel(vt);
                let k_new = g.kernel("gated_delta_rule_fla", kname)?;
                let (sc1, uc1, sf1) = run_full(g, k_wu, k_new, &case, vt)?;
                let parity = sc0 == sc1 && uc0 == uc1 && sf0 == sf1;
                let t_new = time_scan(g, k_wu, k_new, &case, vt, iters)?;
                all_ok &= parity;
                println!(
                    "t={t:4} batch={batch} VT={vt}: bit-parity={}  current={t_cur:.4}ms  vblock={t_new:.4}ms  speedup={:.2}x",
                    if parity { "PASS" } else { "FAIL ❌" },
                    t_cur / t_new
                );
            }
        }
    }
    println!(
        "\n{}",
        if all_ok {
            "ALL BIT-PARITY GATES PASS ✅"
        } else {
            "BIT-PARITY FAILED ❌"
        }
    );
    if !all_ok {
        std::process::exit(1);
    }
    Ok(())
}
