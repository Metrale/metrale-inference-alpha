// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Correctness arms for `int8_gemm_8w_ldm`, `int8_gemm_mmq`,
//! `int8_gemm_8w_ldmab`, `int8_gemm_8w_pipe` and `int8_gemm_padA` on the
//! 128x256x512 problem `main` builds, each scored by cosine against its host reference.
//!
//! Owner: model-arch examples (int8 GEMM kernels).
//! Invariants: each arm frees its own output buffer; the inputs belong to `main`.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

use crate::bf16_bits_to_f32;

pub(super) fn check_8w_ldm(
    gpu: &dyn GpuBackend,
    stream: u64,
    h8wl: KernelHandle,
    a_p: DevicePtr,
    b_p: DevicePtr,
    as_p: DevicePtr,
    bs_p: DevicePtr,
    c_ref: &[f32],
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    let c2 = gpu.alloc(m * n * 2)?;
    KernelLaunch::new(gpu, h8wl)
        .grid([n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1])
        .block([256, 1, 1])
        .arg_ptr(a_p)
        .arg_ptr(b_p)
        .arg_ptr(as_p)
        .arg_ptr(bs_p)
        .arg_ptr(c2)
        .arg_u32(m as u32)
        .arg_u32(n as u32)
        .arg_u32(k as u32)
        .launch(stream)?;
    gpu.synchronize(stream)?;
    let mut raw2 = vec![0u8; m * n * 2];
    gpu.copy_d2h(c2, &mut raw2)?;
    let cg: Vec<f32> = raw2
        .chunks_exact(2)
        .map(|c| bf16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect();
    let (mut d, mut nr, mut ng) = (0f64, 0f64, 0f64);
    for i in 0..m * n {
        let (x, y) = (c_ref[i] as f64, cg[i] as f64);
        d += x * y;
        nr += x * x;
        ng += y * y;
    }
    let cl = d / (nr.sqrt() * ng.sqrt());
    println!(
        "8w_ldm (ldmatrix.x4) correctness: cosine={cl:.6}  RESULT: {}",
        if cl > 0.999 { "PASS" } else { "FAIL" }
    );
    let _ = gpu.free(c2);
    Ok(())
}

pub(super) fn check_mmq(
    gpu: &dyn GpuBackend,
    stream: u64,
    hmmq: KernelHandle,
    a_p: DevicePtr,
    b_p: DevicePtr,
    as_p: DevicePtr,
    bs_p: DevicePtr,
    c_ref: &[f32],
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    let c3 = gpu.alloc(m * n * 2)?;
    KernelLaunch::new(gpu, hmmq)
        .grid([n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1])
        .block([256, 1, 1])
        .arg_ptr(a_p)
        .arg_ptr(b_p)
        .arg_ptr(as_p)
        .arg_ptr(bs_p)
        .arg_ptr(c3)
        .arg_u32(m as u32)
        .arg_u32(n as u32)
        .arg_u32(k as u32)
        .launch(stream)?;
    gpu.synchronize(stream)?;
    let mut r3 = vec![0u8; m * n * 2];
    gpu.copy_d2h(c3, &mut r3)?;
    let cg: Vec<f32> = r3
        .chunks_exact(2)
        .map(|c| bf16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect();
    let (mut d, mut nr, mut ng) = (0f64, 0f64, 0f64);
    for i in 0..m * n {
        let (x, y) = (c_ref[i] as f64, cg[i] as f64);
        d += x * y;
        nr += x * x;
        ng += y * y;
    }
    println!(
        "MMQ-tile correctness: cosine={:.6}  RESULT: {}",
        d / (nr.sqrt() * ng.sqrt()),
        if d / (nr.sqrt() * ng.sqrt()) > 0.999 {
            "PASS"
        } else {
            "FAIL"
        }
    );
    let _ = gpu.free(c3);
    Ok(())
}

pub(super) fn check_8w_ldmab(
    gpu: &dyn GpuBackend,
    stream: u64,
    h8wab: KernelHandle,
    a_p: DevicePtr,
    b_p: DevicePtr,
    as_p: DevicePtr,
    bs_p: DevicePtr,
    c_ref: &[f32],
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    let c4 = gpu.alloc(m * n * 2)?;
    KernelLaunch::new(gpu, h8wab)
        .grid([n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1])
        .block([256, 1, 1])
        .arg_ptr(a_p)
        .arg_ptr(b_p)
        .arg_ptr(as_p)
        .arg_ptr(bs_p)
        .arg_ptr(c4)
        .arg_u32(m as u32)
        .arg_u32(n as u32)
        .arg_u32(k as u32)
        .launch(stream)?;
    gpu.synchronize(stream)?;
    let mut r4 = vec![0u8; m * n * 2];
    gpu.copy_d2h(c4, &mut r4)?;
    let cg: Vec<f32> = r4
        .chunks_exact(2)
        .map(|c| bf16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect();
    let (mut d, mut nr, mut ng) = (0f64, 0f64, 0f64);
    for i in 0..m * n {
        let (x, y) = (c_ref[i] as f64, cg[i] as f64);
        d += x * y;
        nr += x * x;
        ng += y * y;
    }
    println!(
        "8w_ldmAB (ldmatrix A+B) correctness: cosine={:.6}  RESULT: {}",
        d / (nr.sqrt() * ng.sqrt()),
        if d / (nr.sqrt() * ng.sqrt()) > 0.999 {
            "PASS"
        } else {
            "FAIL"
        }
    );
    let _ = gpu.free(c4);
    Ok(())
}

pub(super) fn check_8w_pipe(
    gpu: &dyn GpuBackend,
    stream: u64,
    hpipe: KernelHandle,
    a_p: DevicePtr,
    b_p: DevicePtr,
    as_p: DevicePtr,
    bs_p: DevicePtr,
    c_ref: &[f32],
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    let c5 = gpu.alloc(m * n * 2)?;
    KernelLaunch::new(gpu, hpipe)
        .grid([n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1])
        .block([512, 1, 1])
        .arg_ptr(a_p)
        .arg_ptr(b_p)
        .arg_ptr(as_p)
        .arg_ptr(bs_p)
        .arg_ptr(c5)
        .arg_u32(m as u32)
        .arg_u32(n as u32)
        .arg_u32(k as u32)
        .launch(stream)?;
    gpu.synchronize(stream)?;
    let mut r5 = vec![0u8; m * n * 2];
    gpu.copy_d2h(c5, &mut r5)?;
    let cg: Vec<f32> = r5
        .chunks_exact(2)
        .map(|c| bf16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect();
    let (mut d, mut nr, mut ng) = (0f64, 0f64, 0f64);
    for i in 0..m * n {
        let (x, y) = (c_ref[i] as f64, cg[i] as f64);
        d += x * y;
        nr += x * x;
        ng += y * y;
    }
    println!(
        "8w_pipe (occ 512) correctness: cosine={:.6}  RESULT: {}",
        d / (nr.sqrt() * ng.sqrt()),
        if d / (nr.sqrt() * ng.sqrt()) > 0.999 {
            "PASS"
        } else {
            "FAIL"
        }
    );
    let _ = gpu.free(c5);
    Ok(())
}

pub(super) fn check_pada(
    gpu: &dyn GpuBackend,
    stream: u64,
    hpada: KernelHandle,
    a_p: DevicePtr,
    b_p: DevicePtr,
    as_p: DevicePtr,
    bs_p: DevicePtr,
    c_ref: &[f32],
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    let c6 = gpu.alloc(m * n * 2)?;
    KernelLaunch::new(gpu, hpada)
        .grid([n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1])
        .block([256, 1, 1])
        .arg_ptr(a_p)
        .arg_ptr(b_p)
        .arg_ptr(as_p)
        .arg_ptr(bs_p)
        .arg_ptr(c6)
        .arg_u32(m as u32)
        .arg_u32(n as u32)
        .arg_u32(k as u32)
        .launch(stream)?;
    gpu.synchronize(stream)?;
    let mut r6 = vec![0u8; m * n * 2];
    gpu.copy_d2h(c6, &mut r6)?;
    let cg: Vec<f32> = r6
        .chunks_exact(2)
        .map(|c| bf16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
        .collect();
    let (mut d, mut nr, mut ng) = (0f64, 0f64, 0f64);
    for i in 0..m * n {
        let (x, y) = (c_ref[i] as f64, cg[i] as f64);
        d += x * y;
        nr += x * x;
        ng += y * y;
    }
    println!(
        "padA (bank-fix ldmatrix) correctness: cosine={:.6}  RESULT: {}",
        d / (nr.sqrt() * ng.sqrt()),
        if d / (nr.sqrt() * ng.sqrt()) > 0.999 {
            "PASS"
        } else {
            "FAIL"
        }
    );
    let _ = gpu.free(c6);
    Ok(())
}
