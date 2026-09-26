// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Correctness arms for `int8_gemm_faith` to `int8_gemm_faith4` and
//! `int8_gemm_mmq2` on the 128x256x512 problem `main` builds, each scored by cosine
//! against its host reference.
//!
//! Owner: model-arch examples (int8 GEMM kernels).
//! Invariants: each arm frees its own output buffer; the inputs belong to `main`.

use anyhow::Result;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

use crate::bf16_bits_to_f32;

pub(super) fn check_faith(
    gpu: &dyn GpuBackend,
    stream: u64,
    hfaith: KernelHandle,
    a_p: DevicePtr,
    b_p: DevicePtr,
    as_p: DevicePtr,
    bs_p: DevicePtr,
    c_ref: &[f32],
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    let c7 = gpu.alloc(m * n * 2)?;
    KernelLaunch::new(gpu, hfaith)
        .grid([n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1])
        .block([256, 1, 1])
        .arg_ptr(a_p)
        .arg_ptr(b_p)
        .arg_ptr(as_p)
        .arg_ptr(bs_p)
        .arg_ptr(c7)
        .arg_u32(m as u32)
        .arg_u32(n as u32)
        .arg_u32(k as u32)
        .launch(stream)?;
    gpu.synchronize(stream)?;
    let mut r7 = vec![0u8; m * n * 2];
    gpu.copy_d2h(c7, &mut r7)?;
    let cg: Vec<f32> = r7
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
        "FAITH (llama-MMQ port) correctness: cosine={:.6}  RESULT: {}",
        d / (nr.sqrt() * ng.sqrt()),
        if d / (nr.sqrt() * ng.sqrt()) > 0.999 {
            "PASS"
        } else {
            "FAIL"
        }
    );
    let _ = gpu.free(c7);
    Ok(())
}

pub(super) fn check_faith2(
    gpu: &dyn GpuBackend,
    stream: u64,
    hfaith2: KernelHandle,
    a_p: DevicePtr,
    b_p: DevicePtr,
    as_p: DevicePtr,
    bs_p: DevicePtr,
    c_ref: &[f32],
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    let c8 = gpu.alloc(m * n * 2)?;
    KernelLaunch::new(gpu, hfaith2)
        .grid([n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1])
        .block([256, 1, 1])
        .arg_ptr(a_p)
        .arg_ptr(b_p)
        .arg_ptr(as_p)
        .arg_ptr(bs_p)
        .arg_ptr(c8)
        .arg_u32(m as u32)
        .arg_u32(n as u32)
        .arg_u32(k as u32)
        .launch(stream)?;
    gpu.synchronize(stream)?;
    let mut r8 = vec![0u8; m * n * 2];
    gpu.copy_d2h(c8, &mut r8)?;
    let cg: Vec<f32> = r8
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
        "FAITH2 (big-K rolling) correctness: cosine={:.6}  RESULT: {}",
        d / (nr.sqrt() * ng.sqrt()),
        if d / (nr.sqrt() * ng.sqrt()) > 0.999 {
            "PASS"
        } else {
            "FAIL"
        }
    );
    let _ = gpu.free(c8);
    Ok(())
}

pub(super) fn check_faith3(
    gpu: &dyn GpuBackend,
    stream: u64,
    hfaith3: KernelHandle,
    a_p: DevicePtr,
    b_p: DevicePtr,
    as_p: DevicePtr,
    bs_p: DevicePtr,
    c_ref: &[f32],
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    let c9 = gpu.alloc(m * n * 2)?;
    KernelLaunch::new(gpu, hfaith3)
        .grid([n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1])
        .block([256, 1, 1])
        .arg_ptr(a_p)
        .arg_ptr(b_p)
        .arg_ptr(as_p)
        .arg_ptr(bs_p)
        .arg_ptr(c9)
        .arg_u32(m as u32)
        .arg_u32(n as u32)
        .arg_u32(k as u32)
        .launch(stream)?;
    gpu.synchronize(stream)?;
    let mut r9 = vec![0u8; m * n * 2];
    gpu.copy_d2h(c9, &mut r9)?;
    let cg: Vec<f32> = r9
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
        "FAITH3 (B-frag ILP) correctness: cosine={:.6}  RESULT: {}",
        d / (nr.sqrt() * ng.sqrt()),
        if d / (nr.sqrt() * ng.sqrt()) > 0.999 {
            "PASS"
        } else {
            "FAIL"
        }
    );
    let _ = gpu.free(c9);
    Ok(())
}

pub(super) fn check_faith4(
    gpu: &dyn GpuBackend,
    stream: u64,
    hfaith4: KernelHandle,
    a_p: DevicePtr,
    b_p: DevicePtr,
    as_p: DevicePtr,
    bs_p: DevicePtr,
    c_ref: &[f32],
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    let c10 = gpu.alloc(m * n * 2)?;
    KernelLaunch::new(gpu, hfaith4)
        .grid([n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1])
        .block([512, 1, 1])
        .arg_ptr(a_p)
        .arg_ptr(b_p)
        .arg_ptr(as_p)
        .arg_ptr(bs_p)
        .arg_ptr(c10)
        .arg_u32(m as u32)
        .arg_u32(n as u32)
        .arg_u32(k as u32)
        .launch(stream)?;
    gpu.synchronize(stream)?;
    let mut r10 = vec![0u8; m * n * 2];
    gpu.copy_d2h(c10, &mut r10)?;
    let cg: Vec<f32> = r10
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
        "FAITH4 (512-thread occ) correctness: cosine={:.6}  RESULT: {}",
        d / (nr.sqrt() * ng.sqrt()),
        if d / (nr.sqrt() * ng.sqrt()) > 0.999 {
            "PASS"
        } else {
            "FAIL"
        }
    );
    let _ = gpu.free(c10);
    Ok(())
}

pub(super) fn check_mmq2(
    gpu: &dyn GpuBackend,
    stream: u64,
    hmmq2: KernelHandle,
    a_p: DevicePtr,
    b_p: DevicePtr,
    as_p: DevicePtr,
    bs_p: DevicePtr,
    c_ref: &[f32],
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    let c11 = gpu.alloc(m * n * 2)?;
    KernelLaunch::new(gpu, hmmq2)
        .grid([n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1])
        .block([256, 1, 1])
        .arg_ptr(a_p)
        .arg_ptr(b_p)
        .arg_ptr(as_p)
        .arg_ptr(bs_p)
        .arg_ptr(c11)
        .arg_u32(m as u32)
        .arg_u32(n as u32)
        .arg_u32(k as u32)
        .launch(stream)?;
    gpu.synchronize(stream)?;
    let mut r11 = vec![0u8; m * n * 2];
    gpu.copy_d2h(c11, &mut r11)?;
    let cg: Vec<f32> = r11
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
        "MMQ2 (faith2 + double-buffer) correctness: cosine={:.6}  RESULT: {}",
        d / (nr.sqrt() * ng.sqrt()),
        if d / (nr.sqrt() * ng.sqrt()) > 0.999 {
            "PASS"
        } else {
            "FAIL"
        }
    );
    let _ = gpu.free(c11);
    Ok(())
}
