// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `gated_delta_rule_recompute_wu` against an f64 CPU reference, the forward
//! substitutions (I+L)U = βV and (I+L)W = β exp(gc) K, at Qwen3.6-35B-A3B GDN dimensions.
//!
//! Owner: model-arch examples.
//! Invariants:
//! - Exit 1 unless U and W reach cosine >= 0.999 against the reference for every t.
//!
//! t defaults to 64, 128 and 200; `METRALE_WU_SHAPES` takes a comma-separated list.
//! `WU_BENCH_ITERS` > 0 also times the launch. Run:
//!   cargo run -p metrale-model-arch --release --example gdn_recompute_wu_gateb

use anyhow::Result;
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

const KD: usize = 128;
const VD: usize = 128;
const NK: usize = 16;
const NV: usize = 32;
const HR: usize = NV / NK;
const C: usize = 64;

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

fn up_bf16(gpu: &dyn GpuBackend, d: &[bf16]) -> Result<DevicePtr> {
    let bytes: Vec<u8> = d.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect();
    let p = gpu.alloc(bytes.len())?;
    gpu.copy_h2d(&bytes, p)?;
    Ok(p)
}
fn up_f32(gpu: &dyn GpuBackend, d: &[f32]) -> Result<DevicePtr> {
    let bytes: Vec<u8> = d.iter().flat_map(|x| x.to_le_bytes()).collect();
    let p = gpu.alloc(bytes.len())?;
    gpu.copy_h2d(&bytes, p)?;
    Ok(p)
}
fn down_bf16(gpu: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<f32>> {
    let mut bytes = vec![0u8; n * 2];
    gpu.copy_d2h(p, &mut bytes)?;
    Ok(bytes
        .chunks_exact(2)
        .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect())
}
fn compare(a: &[f32], b: &[f32]) -> (f32, f64) {
    let (mut md, mut dot, mut na, mut nb) = (0.0f32, 0.0f64, 0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        md = md.max((x - y).abs());
        dot += (*x as f64) * (*y as f64);
        na += (*x as f64).powi(2);
        nb += (*y as f64).powi(2);
    }
    (md, dot / (na.sqrt() * nb.sqrt() + 1e-12))
}

fn main() -> Result<()> {
    let set = metrale_kernels::ptx_for_config("qwen3_6_moe", 2048, &[], None)
        .expect("unambiguous")
        .expect("no ptx set");
    let backend = MetraleCudaBackend::new(0, &set.modules)?;
    let gpu: &dyn GpuBackend = &backend;
    let k = gpu.kernel("gated_delta_rule_fla", "gated_delta_rule_recompute_wu")?;
    eprintln!("recompute_wu handle={}", k.0);

    let mut all_ok = true;
    let shapes: Vec<usize> = match std::env::var("METRALE_WU_SHAPES") {
        Ok(v) => v.split(',').filter_map(|x| x.trim().parse().ok()).collect(),
        Err(_) => vec![64usize, 128, 200],
    };
    for &t in &shapes {
        let nt = t.div_ceil(C);
        let mut r = Lcg(0xC0FFEE ^ t as u64);
        let key: Vec<bf16> = (0..t * NK * KD)
            .map(|_| bf16::from_f64(r.r(-0.5, 0.5)))
            .collect();
        let val: Vec<bf16> = (0..t * NV * VD)
            .map(|_| bf16::from_f64(r.r(-0.5, 0.5)))
            .collect();
        let gate: Vec<f32> = (0..t * NV).map(|_| r.r(0.80, 0.999) as f32).collect();
        let beta: Vec<f32> = (0..t * NV).map(|_| r.r(0.0, 1.0) as f32).collect();

        // 2026-09-25: CPU reference, per chunk and head, by forward substitution.
        let mut u_ref = vec![0.0f32; nt * NV * C * VD];
        let mut w_ref = vec![0.0f32; nt * NV * C * KD];
        for vh in 0..NV {
            let kh = vh / HR;
            for c in 0..nt {
                let cs = c * C;
                let ce = C.min(t - cs);
                let mut gc = vec![0.0f64; ce];
                let mut acc = 0.0;
                for i in 0..ce {
                    acc += (gate[(cs + i) * NV + vh] as f64).ln();
                    gc[i] = acc;
                }
                let mut l_mat = vec![0.0f64; ce * ce];
                for i in 0..ce {
                    let bi = beta[(cs + i) * NV + vh] as f64;
                    for l in 0..i {
                        let mut kk = 0.0f64;
                        for j in 0..KD {
                            kk += key[((cs + l) * NK + kh) * KD + j].to_f64()
                                * key[((cs + i) * NK + kh) * KD + j].to_f64();
                        }
                        l_mat[i * ce + l] = bi * (gc[i] - gc[l]).exp() * kk;
                    }
                }
                let base = (c * NV + vh) * C;
                for v in 0..VD {
                    let mut u = vec![0.0f64; ce];
                    for i in 0..ce {
                        let bi = beta[(cs + i) * NV + vh] as f64;
                        let mut ui = bi * val[((cs + i) * NV + vh) * VD + v].to_f64();
                        for l in 0..i {
                            ui -= l_mat[i * ce + l] * u[l];
                        }
                        u[i] = ui;
                        u_ref[(base + i) * VD + v] = ui as f32;
                    }
                }
                for kk_ in 0..KD {
                    let mut w = vec![0.0f64; ce];
                    for i in 0..ce {
                        let bi = beta[(cs + i) * NV + vh] as f64;
                        let mut wi =
                            bi * gc[i].exp() * key[((cs + i) * NK + kh) * KD + kk_].to_f64();
                        for l in 0..i {
                            wi -= l_mat[i * ce + l] * w[l];
                        }
                        w[i] = wi;
                        w_ref[(base + i) * KD + kk_] = wi as f32;
                    }
                }
            }
        }

        let kp = up_bf16(gpu, &key)?;
        let vp = up_bf16(gpu, &val)?;
        let gp = up_f32(gpu, &gate)?;
        let bp = up_f32(gpu, &beta)?;
        let wp = gpu.alloc(nt * NV * C * KD * 2)?;
        let up = gpu.alloc(nt * NV * C * VD * 2)?;
        let gcp = gpu.alloc(nt * NV * C * 4)?;
        // 2026-09-25: Equal to `smem_wu` in ops/ssm_gdn_a3.rs.
        let smem = (C * KD * 2 + C * C * 4 + C * 4) as u32;
        let launch_wu = || -> Result<()> {
            KernelLaunch::new(gpu, k)
                .grid([nt as u32, NV as u32, 1])
                // 2026-09-25: 256 threads: the kernel runs the W solve on threads 128-255,
                // so a 128-thread launch would leave W_out unwritten.
                .block([256, 1, 1])
                .shared_mem(smem)
                .arg_ptr(kp)
                .arg_ptr(vp)
                .arg_ptr(gp)
                .arg_ptr(bp)
                .arg_ptr(wp)
                .arg_ptr(up)
                .arg_ptr(gcp)
                .arg_u32(1)
                .arg_u32(t as u32)
                .arg_u32(nt as u32)
                .arg_u32(NK as u32)
                .arg_u32(NV as u32)
                .arg_u32(KD as u32)
                .arg_u32(VD as u32)
                .arg_u32((NK * KD) as u32)
                .arg_u32((NV * VD) as u32)
                .arg_u32(NV as u32)
                .arg_ptr(DevicePtr(0))
                .arg_ptr(DevicePtr(0))
                .arg_u32(0)
                .launch(0)?;
            Ok(())
        };
        launch_wu()?;
        gpu.synchronize(0)?;
        // 2026-09-25: With WU_BENCH_ITERS > 0, time the same launch: 8 warm-up launches,
        // then the mean over the timed ones. The numbers are for one isolated launch on
        // synthetic inputs, for ranking variants against each other.
        let iters = std::env::var("WU_BENCH_ITERS")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(0);
        if iters > 0 {
            for _ in 0..8 {
                launch_wu()?;
            }
            gpu.synchronize(0)?;
            let t_start = std::time::Instant::now();
            for _ in 0..iters {
                launch_wu()?;
            }
            gpu.synchronize(0)?;
            let us = t_start.elapsed().as_secs_f64() * 1e6 / f64::from(iters);
            eprintln!("  recompute_wu t={t} nt={nt}: {us:.1} us/launch ({iters} iters)");
        }
        let u_gpu = down_bf16(gpu, up, nt * NV * C * VD)?;
        let w_gpu = down_bf16(gpu, wp, nt * NV * C * KD)?;
        for p in [kp, vp, gp, bp, wp, up, gcp] {
            let _ = gpu.free(p);
        }

        let (md_u, cos_u) = compare(&u_gpu, &u_ref);
        let (md_w, cos_w) = compare(&w_gpu, &w_ref);
        let ok = cos_u >= 0.999 && cos_w >= 0.999;
        all_ok &= ok;
        eprintln!(
            "t={t:4} nt={nt}  U: max={md_u:.4} cos={cos_u:.6} | W: max={md_w:.4} cos={cos_w:.6}  {}",
            if ok { "PASS" } else { "FAIL" }
        );
    }
    eprintln!(
        "\nrecompute_wu GATE B: {}",
        if all_ok { "PASS ✅" } else { "FAIL ❌" }
    );
    if !all_ok {
        std::process::exit(1);
    }
    Ok(())
}
