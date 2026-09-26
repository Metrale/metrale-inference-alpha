// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The speed section for one prefill shape: int8_gemm_t_m128 through `run`,
//! every tile and faith variant, then split-K at 2, 4, 8 and 16 slices, printed as
//! one TFLOP/s line.
//!
//! Owner: model-arch examples (int8 GEMM kernels).
//! Invariants: every buffer allocated here is freed before `speed_shape` returns `Ok`.

mod faith;
mod tiles;

use anyhow::Result;
use metrale_gpu_runtime::gpu::{GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use std::time::Instant;

use crate::{Rng, bytemuck_f32, bytemuck_i8, run, up};
use faith::time_faiths;
use tiles::time_tiles;

pub(super) fn speed_shape(
    gpu: &dyn GpuBackend,
    stream: u64,
    h: KernelHandle,
    h64: KernelHandle,
    hk64: KernelHandle,
    h8w: KernelHandle,
    h8w3: KernelHandle,
    h8wl: KernelHandle,
    h8wi: KernelHandle,
    hmmq: KernelHandle,
    h8wab: KernelHandle,
    hpipe: KernelHandle,
    hpada: KernelHandle,
    hfaith: KernelHandle,
    hfaith2: KernelHandle,
    hfaith3: KernelHandle,
    hfaith4: KernelHandle,
    hfaith5: KernelHandle,
    hfaith6: KernelHandle,
    hfaith7: KernelHandle,
    hfaith8: KernelHandle,
    hfaith9: KernelHandle,
    hfaith10: KernelHandle,
    hmmqf: KernelHandle,
    hmmqf2: KernelHandle,
    hmmqf3: KernelHandle,
    hmmq2: KernelHandle,
    hsk: KernelHandle,
    hred: KernelHandle,
    label: &str,
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    let nb = k / 32;
    let mut rng = Rng(7);
    let a_i8: Vec<i8> = (0..m * k).map(|_| rng.i8()).collect();
    let b_i8: Vec<i8> = (0..n * k).map(|_| rng.i8()).collect();
    let a_sc: Vec<f32> = (0..m * nb).map(|_| rng.pos_scale()).collect();
    let b_sc: Vec<f32> = (0..n * nb).map(|_| rng.pos_scale()).collect();
    let a_p = up(gpu, bytemuck_i8(&a_i8))?;
    let b_p = up(gpu, bytemuck_i8(&b_i8))?;
    let as_p = up(gpu, bytemuck_f32(&a_sc))?;
    let bs_p = up(gpu, bytemuck_f32(&b_sc))?;
    let c_p = gpu.alloc(m * n * 2)?;
    for _ in 0..3 {
        run(gpu, stream, h, a_p, b_p, as_p, bs_p, c_p, m, n, k)?;
    }
    gpu.synchronize(stream)?;
    let iters = 30;
    let flops = 2.0 * m as f64 * n as f64 * k as f64;
    let t0 = Instant::now();
    for _ in 0..iters {
        run(gpu, stream, h, a_p, b_p, as_p, bs_p, c_p, m, n, k)?;
    }
    gpu.synchronize(stream)?;
    let tf128 = flops / (t0.elapsed().as_secs_f64() / iters as f64) / 1e12;
    let (tf64, tfk64, tf8w, tf8w3, tf8wl, tf8wi, tfmmq, tf8wab, tfpipe, tfpada, tffaith, tffaith2) =
        time_tiles(
            gpu, stream, h64, hk64, h8w, h8w3, h8wl, h8wi, hmmq, h8wab, hpipe, hpada, hfaith,
            hfaith2, a_p, b_p, as_p, bs_p, c_p, m, n, k, iters, flops,
        )?;
    let (
        tffaith3,
        tffaith4,
        tffaith5,
        tffaith6,
        tffaith7,
        tffaith8,
        tffaith9,
        tffaith10,
        tfmmqf,
        tfmmqf2,
        tfmmqf3,
        tfmmq2,
    ) = time_faiths(
        gpu, stream, hfaith3, hfaith4, hfaith5, hfaith6, hfaith7, hfaith8, hfaith9, hfaith10,
        hmmqf, hmmqf2, hmmqf3, hmmq2, a_p, b_p, as_p, bs_p, c_p, m, n, k, iters, flops,
    )?;
    let _ = (
        tf64, tfk64, tf8w3, tf8w, tf8wl, tf8wi, tfpipe, tfpada, tf8wab,
    );
    print!(
        "{label}: M128 {tf128:.2} | padA {tfpada:.2} | FAITH {tffaith:.2} | FAITH2 {tffaith2:.2} | FAITH3 {tffaith3:.2} | FAITH4 {tffaith4:.2} | FAITH5 {tffaith5:.2} | FAITH6 {tffaith6:.2} | FAITH7 {tffaith7:.2} | FAITH8 {tffaith8:.2} | FAITH9 {tffaith9:.2} | FAITH10 {tffaith10:.2} | MMQF {tfmmqf:.2} | MMQF2 {tfmmqf2:.2} | MMQF3 {tfmmqf3:.2} | MMQ {tfmmq:.2} | MMQ2 {tfmmq2:.2}  (bf16=30, llama Q4K=65/Q6K=41)"
    );
    for &ks in &[2u32, 4, 8, 16] {
        let cp = gpu.alloc(ks as usize * m * n * 4)?;
        let mn = (m * n) as u32;
        let go = || -> Result<()> {
            KernelLaunch::new(gpu, hsk)
                .grid([n.div_ceil(128) as u32, m.div_ceil(128) as u32, ks])
                .block([128, 1, 1])
                .arg_ptr(a_p)
                .arg_ptr(b_p)
                .arg_ptr(as_p)
                .arg_ptr(bs_p)
                .arg_ptr(cp)
                .arg_u32(m as u32)
                .arg_u32(n as u32)
                .arg_u32(k as u32)
                .arg_u32(ks)
                .launch(stream)?;
            KernelLaunch::new(gpu, hred)
                .grid([mn.div_ceil(256), 1, 1])
                .block([256, 1, 1])
                .arg_ptr(cp)
                .arg_ptr(c_p)
                .arg_u32(m as u32)
                .arg_u32(n as u32)
                .arg_u32(ks)
                .launch(stream)
        };
        for _ in 0..3 {
            go()?;
        }
        gpu.synchronize(stream)?;
        let t = Instant::now();
        for _ in 0..iters {
            go()?;
        }
        gpu.synchronize(stream)?;
        let tf = flops / (t.elapsed().as_secs_f64() / iters as f64) / 1e12;
        print!(" | sk{ks}={tf:.1}");
        let _ = gpu.free(cp);
    }
    println!("   (v2 bf16=30)");
    for p in [a_p, b_p, as_p, bs_p, c_p] {
        let _ = gpu.free(p);
    }
    Ok(())
}
