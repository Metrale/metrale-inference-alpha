// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Speed runs for the tile variants (m64, m128_k64, 8w, 8w3, 8w_ldm, 8w_ilp,
//! mmq, 8w_ldmab, 8w_pipe, padA) and faith, faith2 on one prefill shape: 3 warm-up
//! launches, then `iters` timed launches, returned as TFLOP/s.
//!
//! Owner: model-arch examples (int8 GEMM kernels).
//! Invariants: none beyond the types; the buffers belong to the caller.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use std::time::Instant;

pub(super) fn time_tiles(
    gpu: &dyn GpuBackend,
    stream: u64,
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
    a_p: DevicePtr,
    b_p: DevicePtr,
    as_p: DevicePtr,
    bs_p: DevicePtr,
    c_p: DevicePtr,
    m: usize,
    n: usize,
    k: usize,
    iters: i32,
    flops: f64,
) -> Result<(f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64)> {
    let launch64 = || -> Result<()> {
        KernelLaunch::new(gpu, h64)
            .grid([n.div_ceil(128) as u32, m.div_ceil(64) as u32, 1])
            .block([128, 1, 1])
            .arg_ptr(a_p)
            .arg_ptr(b_p)
            .arg_ptr(as_p)
            .arg_ptr(bs_p)
            .arg_ptr(c_p)
            .arg_u32(m as u32)
            .arg_u32(n as u32)
            .arg_u32(k as u32)
            .launch(stream)
    };
    for _ in 0..3 {
        launch64()?;
    }
    gpu.synchronize(stream)?;
    let t1 = Instant::now();
    for _ in 0..iters {
        launch64()?;
    }
    gpu.synchronize(stream)?;
    let tf64 = flops / (t1.elapsed().as_secs_f64() / iters as f64) / 1e12;
    let launchk64 = || -> Result<()> {
        KernelLaunch::new(gpu, hk64)
            .grid([n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1])
            .block([128, 1, 1])
            .arg_ptr(a_p)
            .arg_ptr(b_p)
            .arg_ptr(as_p)
            .arg_ptr(bs_p)
            .arg_ptr(c_p)
            .arg_u32(m as u32)
            .arg_u32(n as u32)
            .arg_u32(k as u32)
            .launch(stream)
    };
    for _ in 0..3 {
        launchk64()?;
    }
    gpu.synchronize(stream)?;
    let tk = Instant::now();
    for _ in 0..iters {
        launchk64()?;
    }
    gpu.synchronize(stream)?;
    let tfk64 = flops / (tk.elapsed().as_secs_f64() / iters as f64) / 1e12;
    let launch8w = || -> Result<()> {
        KernelLaunch::new(gpu, h8w)
            .grid([n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1])
            .block([256, 1, 1])
            .arg_ptr(a_p)
            .arg_ptr(b_p)
            .arg_ptr(as_p)
            .arg_ptr(bs_p)
            .arg_ptr(c_p)
            .arg_u32(m as u32)
            .arg_u32(n as u32)
            .arg_u32(k as u32)
            .launch(stream)
    };
    for _ in 0..3 {
        launch8w()?;
    }
    gpu.synchronize(stream)?;
    let t8 = Instant::now();
    for _ in 0..iters {
        launch8w()?;
    }
    gpu.synchronize(stream)?;
    let tf8w = flops / (t8.elapsed().as_secs_f64() / iters as f64) / 1e12;
    let launch8w3 = || -> Result<()> {
        KernelLaunch::new(gpu, h8w3)
            .grid([n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1])
            .block([256, 1, 1])
            .arg_ptr(a_p)
            .arg_ptr(b_p)
            .arg_ptr(as_p)
            .arg_ptr(bs_p)
            .arg_ptr(c_p)
            .arg_u32(m as u32)
            .arg_u32(n as u32)
            .arg_u32(k as u32)
            .launch(stream)
    };
    for _ in 0..3 {
        launch8w3()?;
    }
    gpu.synchronize(stream)?;
    let t83 = Instant::now();
    for _ in 0..iters {
        launch8w3()?;
    }
    gpu.synchronize(stream)?;
    let tf8w3 = flops / (t83.elapsed().as_secs_f64() / iters as f64) / 1e12;
    let launch8wl = || -> Result<()> {
        KernelLaunch::new(gpu, h8wl)
            .grid([n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1])
            .block([256, 1, 1])
            .arg_ptr(a_p)
            .arg_ptr(b_p)
            .arg_ptr(as_p)
            .arg_ptr(bs_p)
            .arg_ptr(c_p)
            .arg_u32(m as u32)
            .arg_u32(n as u32)
            .arg_u32(k as u32)
            .launch(stream)
    };
    for _ in 0..3 {
        launch8wl()?;
    }
    gpu.synchronize(stream)?;
    let tl = Instant::now();
    for _ in 0..iters {
        launch8wl()?;
    }
    gpu.synchronize(stream)?;
    let tf8wl = flops / (tl.elapsed().as_secs_f64() / iters as f64) / 1e12;
    let launch8wi = || -> Result<()> {
        KernelLaunch::new(gpu, h8wi)
            .grid([n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1])
            .block([256, 1, 1])
            .arg_ptr(a_p)
            .arg_ptr(b_p)
            .arg_ptr(as_p)
            .arg_ptr(bs_p)
            .arg_ptr(c_p)
            .arg_u32(m as u32)
            .arg_u32(n as u32)
            .arg_u32(k as u32)
            .launch(stream)
    };
    for _ in 0..3 {
        launch8wi()?;
    }
    gpu.synchronize(stream)?;
    let ti = Instant::now();
    for _ in 0..iters {
        launch8wi()?;
    }
    gpu.synchronize(stream)?;
    let tf8wi = flops / (ti.elapsed().as_secs_f64() / iters as f64) / 1e12;
    let launchmmq = || -> Result<()> {
        KernelLaunch::new(gpu, hmmq)
            .grid([n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1])
            .block([256, 1, 1])
            .arg_ptr(a_p)
            .arg_ptr(b_p)
            .arg_ptr(as_p)
            .arg_ptr(bs_p)
            .arg_ptr(c_p)
            .arg_u32(m as u32)
            .arg_u32(n as u32)
            .arg_u32(k as u32)
            .launch(stream)
    };
    for _ in 0..3 {
        launchmmq()?;
    }
    gpu.synchronize(stream)?;
    let tmq = Instant::now();
    for _ in 0..iters {
        launchmmq()?;
    }
    gpu.synchronize(stream)?;
    let tfmmq = flops / (tmq.elapsed().as_secs_f64() / iters as f64) / 1e12;
    let launch8wab = || -> Result<()> {
        KernelLaunch::new(gpu, h8wab)
            .grid([n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1])
            .block([256, 1, 1])
            .arg_ptr(a_p)
            .arg_ptr(b_p)
            .arg_ptr(as_p)
            .arg_ptr(bs_p)
            .arg_ptr(c_p)
            .arg_u32(m as u32)
            .arg_u32(n as u32)
            .arg_u32(k as u32)
            .launch(stream)
    };
    for _ in 0..3 {
        launch8wab()?;
    }
    gpu.synchronize(stream)?;
    let tab = Instant::now();
    for _ in 0..iters {
        launch8wab()?;
    }
    gpu.synchronize(stream)?;
    let tf8wab = flops / (tab.elapsed().as_secs_f64() / iters as f64) / 1e12;
    let launchpipe = || -> Result<()> {
        KernelLaunch::new(gpu, hpipe)
            .grid([n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1])
            .block([512, 1, 1])
            .arg_ptr(a_p)
            .arg_ptr(b_p)
            .arg_ptr(as_p)
            .arg_ptr(bs_p)
            .arg_ptr(c_p)
            .arg_u32(m as u32)
            .arg_u32(n as u32)
            .arg_u32(k as u32)
            .launch(stream)
    };
    for _ in 0..3 {
        launchpipe()?;
    }
    gpu.synchronize(stream)?;
    let tp = Instant::now();
    for _ in 0..iters {
        launchpipe()?;
    }
    gpu.synchronize(stream)?;
    let tfpipe = flops / (tp.elapsed().as_secs_f64() / iters as f64) / 1e12;
    let launchpada = || -> Result<()> {
        KernelLaunch::new(gpu, hpada)
            .grid([n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1])
            .block([256, 1, 1])
            .arg_ptr(a_p)
            .arg_ptr(b_p)
            .arg_ptr(as_p)
            .arg_ptr(bs_p)
            .arg_ptr(c_p)
            .arg_u32(m as u32)
            .arg_u32(n as u32)
            .arg_u32(k as u32)
            .launch(stream)
    };
    for _ in 0..3 {
        launchpada()?;
    }
    gpu.synchronize(stream)?;
    let tpa = Instant::now();
    for _ in 0..iters {
        launchpada()?;
    }
    gpu.synchronize(stream)?;
    let tfpada = flops / (tpa.elapsed().as_secs_f64() / iters as f64) / 1e12;
    let launchfaith = || -> Result<()> {
        KernelLaunch::new(gpu, hfaith)
            .grid([n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1])
            .block([256, 1, 1])
            .arg_ptr(a_p)
            .arg_ptr(b_p)
            .arg_ptr(as_p)
            .arg_ptr(bs_p)
            .arg_ptr(c_p)
            .arg_u32(m as u32)
            .arg_u32(n as u32)
            .arg_u32(k as u32)
            .launch(stream)
    };
    for _ in 0..3 {
        launchfaith()?;
    }
    gpu.synchronize(stream)?;
    let tfa = Instant::now();
    for _ in 0..iters {
        launchfaith()?;
    }
    gpu.synchronize(stream)?;
    let tffaith = flops / (tfa.elapsed().as_secs_f64() / iters as f64) / 1e12;
    let launchfaith2 = || -> Result<()> {
        KernelLaunch::new(gpu, hfaith2)
            .grid([n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1])
            .block([256, 1, 1])
            .arg_ptr(a_p)
            .arg_ptr(b_p)
            .arg_ptr(as_p)
            .arg_ptr(bs_p)
            .arg_ptr(c_p)
            .arg_u32(m as u32)
            .arg_u32(n as u32)
            .arg_u32(k as u32)
            .launch(stream)
    };
    for _ in 0..3 {
        launchfaith2()?;
    }
    gpu.synchronize(stream)?;
    let tf2 = Instant::now();
    for _ in 0..iters {
        launchfaith2()?;
    }
    gpu.synchronize(stream)?;
    let tffaith2 = flops / (tf2.elapsed().as_secs_f64() / iters as f64) / 1e12;
    Ok((
        tf64, tfk64, tf8w, tf8w3, tf8wl, tf8wi, tfmmq, tf8wab, tfpipe, tfpada, tffaith, tffaith2,
    ))
}
