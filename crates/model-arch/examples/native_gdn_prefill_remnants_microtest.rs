// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Oracle for the two Hopper GDN chunked-prefill remnant twins:
//! `gated_delta_rule_recompute_wu` against `..._hopper` and
//! `gated_delta_rule_chunk_fwd_o` against `..._hopper`, on identical inputs at
//! T in {256, 1193, 4593}, then both twins end to end.
//!
//! Owner: model-arch examples (GPU microtests).
//! Invariants:
//! - The run exits 1 unless, at every T, every gate below passes, no guard tail
//!   changed, and the KNOWN_BAD control moved `max_abs`. An image without the
//!   twins prints SKIPPED and exits 0.
//!
//! Every arm is scored against an f64 CPU reference, not against the other:
//! the twins round new operands to BF16 and reassociate k-sums into the MMA
//! tree (`kernels/hopper/common/gdn_recompute_wu_hopper.cu`,
//! `gdn_fwd_o_hopper.cu`). Gates:
//!   * `W`, `U` and the `fwd_o` output are BF16 outputs: each twin's rel_rms
//!     <= 1.25x its parent's.
//!   * `gc` is f32 and must be bit-identical to the parent's: the twin keeps
//!     the parent's serial scan.
//!   * Both twins end to end (wu twin -> spine -> fwd_o twin), scored against
//!     the parent-fed reference: rel_rms <= 1.5x the parents'.
//!   * Every kernel output has a sentinel tail that must come back untouched.
//!   * The KNOWN_BAD perturbation is sized from the arm's own clean extreme
//!     (`known_bad_probe`), so it trips at every T.
//!
//! The twins exist only in `kernels/hopper/common`. The GPU-free half of their
//! contract is `crates/model-layers/src/layers/ops/ssm_gdn_remnants*_tests.rs`.
//!
//!   cargo run -p metrale-model-arch --release --features cuda,gpu-examples \
//!       --example native_gdn_prefill_remnants_microtest

use anyhow::{Result, bail};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

#[path = "common/gdn_remnants.rs"]
mod gdn_remnants;
use gdn_remnants::{
    C, Case, KD, NK, NV, VD, alloc_guarded, dn_bf16, dn_f32, gen_case, guard_intact,
    known_bad_probe, metrics, ref_fwd_o, ref_wu, report, selfcheck_known_bad, selfcheck_take, take,
    up_bf16, up_f32,
};

const SMEM_WU: u32 = (C * KD * 2 + C * C * 4 + C * 4) as u32;
const SMEM_WU_H: u32 = (C * 136 * 2
    + 2 * (C * 24 * 4)
    + 2 * (C * 72 * 2)
    + 2 * (C * 24 * 2)
    + 2 * (16 * 16 * 24 * 2)
    + C * 4) as u32;
const SMEM_FO: u32 =
    (C * KD * 2 + C * KD * 2 + C * C * 4 + C * VD * 2 + KD * VD * 2 + 2 * C * 4) as u32;
const SMEM_FO_H: u32 = (2 * (C * 136 * 2) + VD * 136 * 2 + VD * 72 * 2 + C * 72 * 2 + C * 4) as u32;
const SMEM_SPINE: u32 = (C * KD * 2 + C * KD * 2 + C * VD * 2 + (C + 1) * 4) as u32;

unsafe extern "C" {
    fn cuEventCreate(event: *mut u64, flags: u32) -> i32;
    fn cuEventRecord(event: u64, stream: u64) -> i32;
    fn cuEventSynchronize(event: u64) -> i32;
    fn cuEventElapsedTime(ms: *mut f32, start: u64, end: u64) -> i32;
    fn cuEventDestroy_v2(event: u64) -> i32;
}

struct In {
    q: DevicePtr,
    k: DevicePtr,
    v: DevicePtr,
    g: DevicePtr,
    b: DevicePtr,
}
struct Wu {
    w: DevicePtr,
    u: DevicePtr,
    gc: DevicePtr,
}

#[allow(clippy::too_many_arguments)]
fn launch_wu(
    g: &dyn GpuBackend,
    k: KernelHandle,
    blk: u32,
    smem: u32,
    c: &Case,
    i: &In,
    o: &Wu,
    s: u64,
) -> Result<()> {
    KernelLaunch::new(g, k)
        .grid([c.nt as u32, NV as u32, 1])
        .block([blk, 1, 1])
        .shared_mem(smem)
        .arg_ptr(i.k)
        .arg_ptr(i.v)
        .arg_ptr(i.g)
        .arg_ptr(i.b)
        .arg_ptr(o.w)
        .arg_ptr(o.u)
        .arg_ptr(o.gc)
        .arg_u32(1)
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
        .launch(s)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn launch_fo(
    g: &dyn GpuBackend,
    k: KernelHandle,
    blk: u32,
    smem: u32,
    c: &Case,
    i: &In,
    gc: DevicePtr,
    sc: DevicePtr,
    uc: DevicePtr,
    out: DevicePtr,
    s: u64,
) -> Result<()> {
    KernelLaunch::new(g, k)
        .grid([c.nt as u32, NV as u32, 1])
        .block([blk, 1, 1])
        .shared_mem(smem)
        .arg_ptr(i.q)
        .arg_ptr(i.k)
        .arg_ptr(i.g)
        .arg_ptr(gc)
        .arg_ptr(sc)
        .arg_ptr(uc)
        .arg_ptr(out)
        .arg_u32(1)
        .arg_u32(c.t as u32)
        .arg_u32(c.nt as u32)
        .arg_u32(NK as u32)
        .arg_u32(NV as u32)
        .arg_u32(KD as u32)
        .arg_u32(VD as u32)
        .arg_u32((NK * KD) as u32)
        .arg_u32(NV as u32)
        .arg_ptr(DevicePtr::NULL)
        .arg_ptr(DevicePtr::NULL)
        .arg_u32(0)
        .launch(s)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn launch_spine(
    g: &dyn GpuBackend,
    k: KernelHandle,
    c: &Case,
    i: &In,
    w: &Wu,
    h: DevicePtr,
    sc: DevicePtr,
    uc: DevicePtr,
) -> Result<()> {
    KernelLaunch::new(g, k)
        .grid([NV as u32, 1, 1])
        .block([256, 1, 1])
        .shared_mem(SMEM_SPINE)
        .arg_ptr(h)
        .arg_ptr(w.w)
        .arg_ptr(w.u)
        .arg_ptr(i.k)
        .arg_ptr(i.g)
        .arg_ptr(w.gc)
        .arg_ptr(sc)
        .arg_ptr(uc)
        .arg_u32(1)
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
        .launch(0)?;
    Ok(())
}

/// 2026-09-25: Time `f` over `iters` launches on a private stream, after 3
/// untimed launches, in ms per launch.
fn time_ms(g: &dyn GpuBackend, iters: u32, mut f: impl FnMut(u64) -> Result<()>) -> Result<f64> {
    let s = g.create_stream()?;
    for _ in 0..3 {
        f(s)?;
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
        f(s)?;
    }
    unsafe {
        if cuEventRecord(e1, s) != 0
            || cuEventSynchronize(e1) != 0
            || cuEventElapsedTime(&mut ms, e0, e1) != 0
        {
            bail!("event timing");
        }
        cuEventDestroy_v2(e0);
        cuEventDestroy_v2(e1);
    }
    Ok(ms as f64 / iters as f64)
}

fn main() -> Result<()> {
    // 2026-09-25: Host-side checks before the device is touched: every `take`
    // call's arguments at this example's geometry, and the KNOWN_BAD
    // arithmetic at all three T, on synthetic buffers.
    selfcheck_take();
    selfcheck_known_bad();

    let backend = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let fla = "gated_delta_rule_fla";
    let k_wu = g.kernel(fla, "gated_delta_rule_recompute_wu")?;
    let k_fo = g.kernel(fla, "gated_delta_rule_chunk_fwd_o")?;
    let k_sp = g.kernel(fla, "gated_delta_rule_chunk_delta_h_vfused")?;
    let (Ok(k_wu_h), Ok(k_fo_h)) = (
        g.kernel(
            "gdn_recompute_wu_hopper",
            "gated_delta_rule_recompute_wu_hopper",
        ),
        g.kernel("gdn_fwd_o_hopper", "gated_delta_rule_chunk_fwd_o_hopper"),
    ) else {
        println!(
            "SKIPPED: the Hopper prefill remnant twins are not in this image. They \
             exist only under kernels/hopper (sm_90a); build with that hardware set \
             to run this oracle. The GPU-free half is ssm_gdn_remnants*_tests.rs."
        );
        return Ok(());
    };

    println!("=== GDN prefill remnants: gb10 parents vs Hopper twins (#928) ===");
    println!("nk={NK} nv={NV} kd={KD} vd={VD} chunk={C}");
    println!("smem  wu {SMEM_WU} -> {SMEM_WU_H}   fwd_o {SMEM_FO} -> {SMEM_FO_H}");
    println!("gate: bf16 outputs <= 1.25x the parent's rel_rms; gc bit-identical; guards intact\n");

    let mut all_ok = true;
    for &t in &[256usize, 1193, 4593] {
        let c = gen_case(t);
        let iters = if t > 2048 { 10 } else { 30 };
        // 2026-09-25: `take(full, rows, per)` applies NV itself: `rows` is the
        // outer dimension only, the chunk count for W/U/gc and the token count
        // for O. The buffer lengths below are the [rows][NV][per] products.
        let (wb, ub, gcb) = (c.nt * NV * C * KD, c.nt * NV * C * VD, c.nt * NV * C);
        let inp = In {
            q: up_bf16(g, &c.query)?,
            k: up_bf16(g, &c.key)?,
            v: up_bf16(g, &c.val)?,
            g: up_f32(g, &c.gate)?,
            b: up_f32(g, &c.beta)?,
        };
        let (rw, ru, rgc) = ref_wu(&c);
        println!("T={t} chunks={}", c.nt);

        // 2026-09-25: recompute_wu.
        let mut arms: Vec<(Vec<f32>, Vec<f32>, Vec<f32>)> = Vec::new();
        let mut wus: Vec<Wu> = Vec::new();
        let mut base = (0.0f64, 0.0f64, 0.0f64);
        for (name, k, blk, smem, gated) in [
            ("wu parent", k_wu, 256u32, SMEM_WU, false),
            ("wu hopper twin", k_wu_h, 512u32, SMEM_WU_H, true),
        ] {
            let o = Wu {
                w: alloc_guarded(g, wb * 2)?,
                u: alloc_guarded(g, ub * 2)?,
                gc: alloc_guarded(g, gcb * 4)?,
            };
            launch_wu(g, k, blk, smem, &c, &inp, &o, 0)?;
            g.synchronize(0)?;
            for (tag, p, by) in [
                ("W", o.w, wb * 2),
                ("U", o.u, ub * 2),
                ("gc", o.gc, gcb * 4),
            ] {
                if !guard_intact(g, p, by)? {
                    println!("    GUARD VIOLATED: {name} overwrote {tag}'s sentinel tail");
                    all_ok = false;
                }
            }
            let (w, u, gc) = (
                dn_bf16(g, o.w, wb)?,
                dn_bf16(g, o.u, ub)?,
                dn_f32(g, o.gc, gcb)?,
            );
            let ms = time_ms(g, iters, |s| launch_wu(g, k, blk, smem, &c, &inp, &o, s))?;
            // 2026-09-25: The parent's algorithmic FLOP count on both arms. The
            // twin's blocked solve does more MACs (327,680 against 258,048 per
            // solve, gdn_recompute_wu_hopper.cu), and rating it on its own
            // count would flatter it.
            let fl = (c.nt * NV * 2 * (C * C * KD + 2 * (C * (C - 1) / 2) * KD)) as f64;
            println!("  {name:<16} {ms:.4} ms / {:.2} TFLOP/s", fl / (ms * 1e9));
            let rwo = report("W (bf16 out)", &take(&w, c.nt, C * KD), &rw);
            let ruo = report("U (bf16 out)", &take(&u, c.nt, C * VD), &ru);
            let (_, rgo) = metrics(&take(&gc, c.nt, C), &rgc);
            if !gated {
                base = (rwo, ruo, rgo);
                println!("    gc (f32)               rel_rms={rgo:.4e}  <- the parent's scan");
            } else {
                let bit = take(&gc, c.nt, C) == take(&arms[0].2, c.nt, C);
                let ok = rwo <= 1.25 * base.0 && ruo <= 1.25 * base.1 && bit;
                println!(
                    "    gc bit-identical to the parent: {}",
                    if bit { "YES" } else { "NO" }
                );
                println!(
                    "  VERDICT T={t} wu: W {} U {} gc {} => {}",
                    if rwo <= 1.25 * base.0 { "PASS" } else { "FAIL" },
                    if ruo <= 1.25 * base.1 { "PASS" } else { "FAIL" },
                    if bit { "PASS" } else { "FAIL" },
                    if ok { "PASS" } else { "FAIL" }
                );
                all_ok &= ok;
            }
            arms.push((w, u, gc));
            wus.push(o);
        }

        // 2026-09-25: The spine, once, from the parent's W/U/gc, so the fwd_o
        // comparison differs in one variable only.
        let hp = up_f32(g, &c.h0)?;
        let scp = alloc_guarded(g, c.nt * NV * KD * VD * 2)?;
        let ucp = alloc_guarded(g, ub * 2)?;
        launch_spine(g, k_sp, &c, &inp, &wus[0], hp, scp, ucp)?;
        g.synchronize(0)?;
        let (scv, ucv) = (dn_bf16(g, scp, c.nt * NV * KD * VD)?, dn_bf16(g, ucp, ub)?);
        let ro = ref_fwd_o(&c, &scv, &ucv, &arms[0].2);

        // 2026-09-25: chunk_fwd_o.
        let outs = t * NV * VD;
        let mut fo_base = 0.0f64;
        let mut clean: Vec<f32> = Vec::new();
        for (name, k, smem, gated) in [
            ("fwd_o parent", k_fo, SMEM_FO, false),
            ("fwd_o hopper twin", k_fo_h, SMEM_FO_H, true),
        ] {
            let op = alloc_guarded(g, outs * 2)?;
            let gc = wus[0].gc;
            launch_fo(
                g,
                k,
                if gated { 512 } else { 512 },
                smem,
                &c,
                &inp,
                gc,
                scp,
                ucp,
                op,
                0,
            )?;
            g.synchronize(0)?;
            if !guard_intact(g, op, outs * 2)? {
                println!("    GUARD VIOLATED: {name} overwrote the output sentinel tail");
                all_ok = false;
            }
            let o = dn_bf16(g, op, outs)?;
            let ms = time_ms(g, iters, |s| {
                launch_fo(g, k, 512, smem, &c, &inp, gc, scp, ucp, op, s)
            })?;
            let fl = (c.nt * NV * 2 * (C * C * KD + C * VD * KD + (C * (C + 1) / 2) * VD)) as f64;
            println!("  {name:<16} {ms:.4} ms / {:.2} TFLOP/s", fl / (ms * 1e9));
            let r = report("O (bf16 out)", &take(&o, t, VD), &ro);
            if !gated {
                fo_base = r;
                clean = take(&o, t, VD);
            } else {
                let ok = r <= 1.25 * fo_base;
                println!(
                    "  VERDICT T={t} fwd_o: O {} (parent {fo_base:.4e})",
                    if ok { "PASS" } else { "FAIL" }
                );
                all_ok &= ok;
                // 2026-09-25: KNOWN_BAD: perturb one reference element and
                // require max_abs to move; the rel_rms gate barely sees a single
                // element. `known_bad_probe` sizes the perturbation from the
                // clean max_abs, so it moves max_abs at every T.
                let (mb, mo) = known_bad_probe(&take(&o, t, VD), &ro);
                if mb <= mo {
                    println!("  KNOWN_BAD control DID NOT trip ({mb:.3e} <= {mo:.3e})");
                    all_ok = false;
                } else {
                    println!("  KNOWN_BAD control refused: {mb:.3e} > clean {mo:.3e}");
                }
            }
            let _ = g.free(op);
        }

        // 2026-09-25: Both twins, end to end: wu_hopper -> spine ->
        // fwd_o_hopper.
        {
            let hp2 = up_f32(g, &c.h0)?;
            let sc2 = alloc_guarded(g, c.nt * NV * KD * VD * 2)?;
            let uc2 = alloc_guarded(g, ub * 2)?;
            let op = alloc_guarded(g, outs * 2)?;
            launch_spine(g, k_sp, &c, &inp, &wus[1], hp2, sc2, uc2)?;
            launch_fo(
                g, k_fo_h, 512, SMEM_FO_H, &c, &inp, wus[1].gc, sc2, uc2, op, 0,
            )?;
            g.synchronize(0)?;
            let o = take(&dn_bf16(g, op, outs)?, t, VD);
            // 2026-09-25: Scored against the parent-fed reference, so this is
            // the compounded end-to-end deviation, not one kernel's.
            let r = report("e2e both twins", &o, &ro);
            let (_, rc) = metrics(&clean, &ro);
            let ok = r <= 1.5 * rc.max(1e-12);
            println!(
                "  VERDICT T={t} e2e: {} (parent trio {rc:.4e}, budget 1.5x)",
                if ok { "PASS" } else { "FAIL" }
            );
            all_ok &= ok;
            for p in [hp2, sc2, uc2, op] {
                let _ = g.free(p);
            }
        }
        println!();
        for w in &wus {
            for p in [w.w, w.u, w.gc] {
                let _ = g.free(p);
            }
        }
        for p in [inp.q, inp.k, inp.v, inp.g, inp.b, hp, scp, ucp] {
            let _ = g.free(p);
        }
    }

    println!(
        "{}",
        if all_ok {
            "ALL GATES PASS"
        } else {
            "GATES FAILED"
        }
    );
    if !all_ok {
        std::process::exit(1);
    }
    Ok(())
}
