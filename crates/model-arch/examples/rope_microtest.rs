// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Oracle for `rope_forward_mrope_interleaved` with text-only
//! positions (the t, h and w streams all equal the token index), against a CPU
//! rotate-half RoPE on the first `ROT` dims of each head.
//!
//! Owner: model-arch examples (GPU microtests).
//! Invariants:
//! - The run exits 0 only if the cosine of both q and k against the CPU
//!   reference is at least 0.999.
//!
//!   cargo run -p metrale-model-arch --release --example rope_microtest --features cuda,gpu-examples -- [seq] [seed]
use anyhow::Result;
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};
const HD: usize = 256;
const ROT: usize = 64;
const NQ: usize = 16;
const NKV: usize = 2;
const THETA: f64 = 1.0e7;
struct Lcg(u64);
impl Lcg {
    fn f(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
    fn r(&mut self, a: f64, b: f64) -> f64 {
        a + (b - a) * self.f()
    }
}
fn ub(g: &dyn GpuBackend, d: &[bf16]) -> Result<DevicePtr> {
    let b: Vec<u8> = d.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect();
    let p = g.alloc(b.len())?;
    g.copy_h2d(&b, p)?;
    Ok(p)
}
fn uu(g: &dyn GpuBackend, d: &[u32]) -> Result<DevicePtr> {
    let b: Vec<u8> = d.iter().flat_map(|x| x.to_le_bytes()).collect();
    let p = g.alloc(b.len())?;
    g.copy_h2d(&b, p)?;
    Ok(p)
}
fn db(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<f32>> {
    let mut b = vec![0u8; n * 2];
    g.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(2)
        .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect())
}
fn cosf2(a: &[f32], b: &[f32]) -> f64 {
    let (mut d, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        d += (*x as f64) * (*y as f64);
        na += (*x as f64).powi(2);
        nb += (*y as f64).powi(2);
    }
    d / (na.sqrt() * nb.sqrt() + 1e-12)
}
fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let seq: usize = a.get(1).map_or(40, |s| s.parse().unwrap());
    let seed: u64 = a.get(2).map_or(0x51A7, |s| {
        u64::from_str_radix(s.trim_start_matches("0x"), 16).unwrap_or(0x51A7)
    });
    println!("=== rope microtest: seq={seq} nq={NQ} nkv={NKV} hd={HD} rot={ROT} ===");
    let mut r = Lcg(seed);
    let q: Vec<bf16> = (0..seq * NQ * HD)
        .map(|_| bf16::from_f64(r.r(-1.0, 1.0)))
        .collect();
    let k: Vec<bf16> = (0..seq * NKV * HD)
        .map(|_| bf16::from_f64(r.r(-1.0, 1.0)))
        .collect();
    let q0 = q.clone();
    let k0 = k.clone();
    let pos: Vec<u32> = (0..seq as u32).collect();
    // 2026-09-25: CPU reference; with all three position streams equal, the
    // section interleave does not change the angles.
    let half = ROT / 2;
    let rope = |buf: &mut [bf16], nh: usize| {
        for p in 0..seq {
            for h in 0..nh {
                let base = (p * nh + h) * HD;
                for pair in 0..half {
                    let fe = (2 * pair) as f64 / ROT as f64;
                    let freq = 1.0 / THETA.powf(fe);
                    let ang = (p as f64) * freq;
                    let (c, s) = (ang.cos() as f32, ang.sin() as f32);
                    let x0 = buf[base + pair].to_f32();
                    let x1 = buf[base + pair + half].to_f32();
                    buf[base + pair] = bf16::from_f32(x0 * c - x1 * s);
                    buf[base + pair + half] = bf16::from_f32(x1 * c + x0 * s);
                }
            }
        }
    };
    let mut qr = q0.clone();
    let mut kr = k0.clone();
    rope(&mut qr, NQ);
    rope(&mut kr, NKV);
    let qc: Vec<f32> = qr.iter().map(|x| x.to_f32()).collect();
    let kc: Vec<f32> = kr.iter().map(|x| x.to_f32()).collect();
    let g0 = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &g0;
    let st = g.create_stream()?;
    let qp = ub(g, &q)?;
    let kp = ub(g, &k)?;
    let pp = uu(g, &pos)?;
    let kern = g.kernel("rope_mrope_interleaved", "rope_forward_mrope_interleaved")?;
    let half_rot = (ROT / 2).max(1);
    let ppb = (128 / half_rot).max(1);
    let sb = div_ceil(seq as u32, ppb as u32);
    KernelLaunch::new(g, kern)
        .grid([(NQ + NKV) as u32, sb, 1])
        .block([128, 1, 1])
        .arg_ptr(qp)
        .arg_ptr(kp)
        .arg_ptr(pp)
        .arg_ptr(pp)
        .arg_ptr(pp)
        .arg_u32(seq as u32)
        .arg_u32(NQ as u32)
        .arg_u32(NKV as u32)
        .arg_u32(HD as u32)
        .arg_u32(ROT as u32)
        .arg_f32(THETA as f32)
        .launch(st)?;
    g.synchronize(st)?;
    let qg = db(g, qp, seq * NQ * HD)?;
    let kg = db(g, kp, seq * NKV * HD)?;
    let _ = (&q, &k, &q0, &k0);
    let cq = cosf2(&qg, &qc);
    let ck = cosf2(&kg, &kc);
    println!("q_cos={cq:.6}  k_cos={ck:.6}");
    for p in [qp, kp, pp] {
        let _ = g.free(p);
    }
    if cq >= 0.999 && ck >= 0.999 {
        println!("RESULT: PASS");
        Ok(())
    } else {
        println!("RESULT: FAIL");
        std::process::exit(1);
    }
}
