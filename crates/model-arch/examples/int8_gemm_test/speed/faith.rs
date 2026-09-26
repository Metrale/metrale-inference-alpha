// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Speed runs for faith3 to faith10, mmqf, mmqf2, mmqf3 and mmq2 on one prefill
//! shape: 3 warm-up launches, then `iters` timed launches, returned as TFLOP/s.
//!
//! Owner: model-arch examples (int8 GEMM kernels).
//! Invariants: none beyond the types; the buffers belong to the caller.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use std::time::Instant;

pub(super) fn time_faiths(
    gpu: &dyn GpuBackend,
    stream: u64,
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
    let launchfaith3 = || -> Result<()> {
        KernelLaunch::new(gpu, hfaith3)
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
        launchfaith3()?;
    }
    gpu.synchronize(stream)?;
    let tf3 = Instant::now();
    for _ in 0..iters {
        launchfaith3()?;
    }
    gpu.synchronize(stream)?;
    let tffaith3 = flops / (tf3.elapsed().as_secs_f64() / iters as f64) / 1e12;
    let launchfaith4 = || -> Result<()> {
        KernelLaunch::new(gpu, hfaith4)
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
        launchfaith4()?;
    }
    gpu.synchronize(stream)?;
    let tf4 = Instant::now();
    for _ in 0..iters {
        launchfaith4()?;
    }
    gpu.synchronize(stream)?;
    let tffaith4 = flops / (tf4.elapsed().as_secs_f64() / iters as f64) / 1e12;
    let launchfaith5 = || -> Result<()> {
        KernelLaunch::new(gpu, hfaith5)
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
        launchfaith5()?;
    }
    gpu.synchronize(stream)?;
    let tf5 = Instant::now();
    for _ in 0..iters {
        launchfaith5()?;
    }
    gpu.synchronize(stream)?;
    let tffaith5 = flops / (tf5.elapsed().as_secs_f64() / iters as f64) / 1e12;
    let launchfaith6 = || -> Result<()> {
        KernelLaunch::new(gpu, hfaith6)
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
        launchfaith6()?;
    }
    gpu.synchronize(stream)?;
    let tf6 = Instant::now();
    for _ in 0..iters {
        launchfaith6()?;
    }
    gpu.synchronize(stream)?;
    let tffaith6 = flops / (tf6.elapsed().as_secs_f64() / iters as f64) / 1e12;
    let launchfaith7 = || -> Result<()> {
        KernelLaunch::new(gpu, hfaith7)
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
        launchfaith7()?;
    }
    gpu.synchronize(stream)?;
    let tf7 = Instant::now();
    for _ in 0..iters {
        launchfaith7()?;
    }
    gpu.synchronize(stream)?;
    let tffaith7 = flops / (tf7.elapsed().as_secs_f64() / iters as f64) / 1e12;
    let launchfaith8 = || -> Result<()> {
        KernelLaunch::new(gpu, hfaith8)
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
        launchfaith8()?;
    }
    gpu.synchronize(stream)?;
    let tf8 = Instant::now();
    for _ in 0..iters {
        launchfaith8()?;
    }
    gpu.synchronize(stream)?;
    let tffaith8 = flops / (tf8.elapsed().as_secs_f64() / iters as f64) / 1e12;
    let launchfaith9 = || -> Result<()> {
        KernelLaunch::new(gpu, hfaith9)
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
        launchfaith9()?;
    }
    gpu.synchronize(stream)?;
    let tf9 = Instant::now();
    for _ in 0..iters {
        launchfaith9()?;
    }
    gpu.synchronize(stream)?;
    let tffaith9 = flops / (tf9.elapsed().as_secs_f64() / iters as f64) / 1e12;
    let launchfaith10 = || -> Result<()> {
        KernelLaunch::new(gpu, hfaith10)
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
        launchfaith10()?;
    }
    gpu.synchronize(stream)?;
    let tf10 = Instant::now();
    for _ in 0..iters {
        launchfaith10()?;
    }
    gpu.synchronize(stream)?;
    let tffaith10 = flops / (tf10.elapsed().as_secs_f64() / iters as f64) / 1e12;
    let launchmmqf = || -> Result<()> {
        KernelLaunch::new(gpu, hmmqf)
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
        launchmmqf()?;
    }
    gpu.synchronize(stream)?;
    let tfmf = Instant::now();
    for _ in 0..iters {
        launchmmqf()?;
    }
    gpu.synchronize(stream)?;
    let tfmmqf = flops / (tfmf.elapsed().as_secs_f64() / iters as f64) / 1e12;
    let launchmmqf2 = || -> Result<()> {
        KernelLaunch::new(gpu, hmmqf2)
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
        launchmmqf2()?;
    }
    gpu.synchronize(stream)?;
    let tfmf2 = Instant::now();
    for _ in 0..iters {
        launchmmqf2()?;
    }
    gpu.synchronize(stream)?;
    let tfmmqf2 = flops / (tfmf2.elapsed().as_secs_f64() / iters as f64) / 1e12;
    let launchmmqf3 = || -> Result<()> {
        KernelLaunch::new(gpu, hmmqf3)
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
        launchmmqf3()?;
    }
    gpu.synchronize(stream)?;
    let tfmf3 = Instant::now();
    for _ in 0..iters {
        launchmmqf3()?;
    }
    gpu.synchronize(stream)?;
    let tfmmqf3 = flops / (tfmf3.elapsed().as_secs_f64() / iters as f64) / 1e12;
    let launchmmq2 = || -> Result<()> {
        KernelLaunch::new(gpu, hmmq2)
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
        launchmmq2()?;
    }
    gpu.synchronize(stream)?;
    let tm2 = Instant::now();
    for _ in 0..iters {
        launchmmq2()?;
    }
    gpu.synchronize(stream)?;
    let tfmmq2 = flops / (tm2.elapsed().as_secs_f64() / iters as f64) / 1e12;
    Ok((
        tffaith3, tffaith4, tffaith5, tffaith6, tffaith7, tffaith8, tffaith9, tffaith10, tfmmqf,
        tfmmqf2, tfmmqf3, tfmmq2,
    ))
}
