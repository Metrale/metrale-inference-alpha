// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Oracle for the FP16 h-state twins of the K = 2, 3, 4 WY verify GDN
//! kernels, at Qwen3.6-35B-A3B GDN dimensions.
//!
//! Owner: model-arch examples.
//! Invariants:
//! - Exit 1 unless every leg passes checks 1 and 2.
//!
//!   1. Bitwise, for K = 2 and 3: `gated_delta_rule_wy{2,3}_resident_f16` equals
//!      `gated_delta_rule_wy{2,3}_f16` on the output, every intermediate state and the
//!      final state.
//!   2. Cosine >= `PASS_COS` of the FP16 twin's outputs and states against a host f64
//!      token-by-token recurrence that rounds each state element to FP16 as it is
//!      written.
//!   3. Printed only: the cosine of the FP16 twin against its FP32 parent from the same
//!      initial state.
//!
//!   cargo run -p metrale-model-arch --release --example gdn_wy_f16_microtest \
//!       --features cuda,gpu-examples
use anyhow::Result;
use half::{bf16, f16};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

const KD: usize = 128;
const VD: usize = 128;
const NK: usize = 16;
const NV: usize = 32;
const HR: usize = NV / NK;

const PASS_COS: f64 = 0.99999;

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
    let p = g.alloc(b.len().max(1))?;
    g.copy_h2d(&b, p)?;
    Ok(p)
}
fn up_f32(g: &dyn GpuBackend, d: &[f32]) -> Result<DevicePtr> {
    let b: Vec<u8> = d.iter().flat_map(|x| x.to_le_bytes()).collect();
    let p = g.alloc(b.len().max(1))?;
    g.copy_h2d(&b, p)?;
    Ok(p)
}
/// 2026-09-25: Upload as FP16, the twins' state dtype.
fn up_f16(g: &dyn GpuBackend, d: &[f32]) -> Result<DevicePtr> {
    let b: Vec<u8> = d
        .iter()
        .flat_map(|x| f16::from_f32(*x).to_bits().to_le_bytes())
        .collect();
    let p = g.alloc(b.len().max(1))?;
    g.copy_h2d(&b, p)?;
    Ok(p)
}
fn dn_bf16(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<f32>> {
    let mut b = vec![0u8; n * 2];
    g.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(2)
        .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect())
}
fn dn_f16(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<f32>> {
    let mut b = vec![0u8; n * 2];
    g.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(2)
        .map(|c| f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect())
}
fn dn_f32(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<f32>> {
    let mut b = vec![0u8; n * 4];
    g.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}
fn bits_eq(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}
fn cos(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        dot += (*x as f64) * (*y as f64);
        na += (*x as f64).powi(2);
        nb += (*y as f64).powi(2);
    }
    dot / (na.sqrt() * nb.sqrt() + 1e-12)
}
/// 2026-09-25: Round a value through FP16.
fn r16(x: f64) -> f64 {
    f16::from_f32(x as f32).to_f32() as f64
}

/// 2026-09-25: Token-by-token f64 GDN recurrence in which every state element is rounded
/// to FP16 as it is written, so the q dot product and the next token read the rounded
/// value. Returns the per-token outputs and the state after each token.
#[allow(clippy::type_complexity)]
fn sequential_ref_f16(
    h0: &[f32],
    q: &[Vec<bf16>],
    key: &[Vec<bf16>],
    val: &[Vec<bf16>],
    gate: &[Vec<f32>],
    beta: &[Vec<f32>],
    k: usize,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let scale = (KD as f64).powf(-0.5);
    let mut s: Vec<f64> = h0.iter().map(|&x| r16(x as f64)).collect();
    let mut outs = Vec::with_capacity(k);
    let mut h_after = Vec::with_capacity(k);
    for t in 0..k {
        let mut o_t = vec![0f32; NV * VD];
        for vh in 0..NV {
            let kh = vh / HR;
            let gg = (gate[t][vh] as f64).clamp(1e-6, 1.0 - 1e-6);
            let bt = beta[t][vh] as f64;
            for v in 0..VD {
                let mut hk = 0.0;
                for kk in 0..KD {
                    hk += s[(vh * KD + kk) * VD + v] * key[t][kh * KD + kk].to_f64();
                }
                let vnew = (val[t][vh * VD + v].to_f64() - gg * hk) * bt;
                let mut qd = 0.0;
                for kk in 0..KD {
                    let idx = (vh * KD + kk) * VD + v;
                    let hn = r16(gg * s[idx] + key[t][kh * KD + kk].to_f64() * vnew);
                    s[idx] = hn;
                    qd += hn * q[t][kh * KD + kk].to_f64();
                }
                o_t[vh * VD + v] = (qd * scale) as f32;
            }
        }
        outs.push(o_t);
        h_after.push(s.iter().map(|&x| x as f32).collect());
    }
    (outs, h_after)
}

/// 2026-09-25: Run one WY kernel at batch 1. `f16_state` selects FP16 or FP32 storage for
/// the state and its intermediates; nothing else differs between parent and twin launches.
#[allow(clippy::type_complexity)]
fn run_wy(
    g: &dyn GpuBackend,
    kernel: metrale_gpu_runtime::gpu::KernelHandle,
    h0: &[f32],
    q: &[Vec<bf16>],
    key: &[Vec<bf16>],
    val: &[Vec<bf16>],
    gate: &[Vec<f32>],
    beta: &[Vec<f32>],
    k: usize,
    f16_state: bool,
) -> Result<(Vec<f32>, Vec<Vec<f32>>, Vec<f32>)> {
    use metrale_gpu_runtime::kernel_args::KernelLaunch;
    let mut q_flat = Vec::with_capacity(k * NK * KD);
    let mut k_flat = Vec::with_capacity(k * NK * KD);
    let mut v_flat = Vec::with_capacity(k * NV * VD);
    let mut g_flat = Vec::with_capacity(k * NV);
    let mut b_flat = Vec::with_capacity(k * NV);
    for t in 0..k {
        q_flat.extend_from_slice(&q[t]);
        k_flat.extend_from_slice(&key[t]);
        v_flat.extend_from_slice(&val[t]);
        g_flat.extend_from_slice(&gate[t]);
        b_flat.extend_from_slice(&beta[t]);
    }
    let h_numel = NV * KD * VD;
    let elt = if f16_state { 2 } else { 4 };
    let hp = if f16_state {
        up_f16(g, h0)?
    } else {
        up_f32(g, h0)?
    };
    let qp = up_bf16(g, &q_flat)?;
    let kp = up_bf16(g, &k_flat)?;
    let vp = up_bf16(g, &v_flat)?;
    let gp = up_f32(g, &g_flat)?;
    let bp = up_f32(g, &b_flat)?;
    let op = g.alloc(k * NV * VD * 2)?;
    let inters: Vec<DevicePtr> = (0..k - 1)
        .map(|_| g.alloc(h_numel * elt))
        .collect::<Result<_>>()?;

    let mut launch = KernelLaunch::new(g, kernel)
        .grid([NV as u32, 1, 1])
        .block([128, 1, 1])
        .arg_ptr(hp)
        .arg_ptr(qp)
        .arg_ptr(kp)
        .arg_ptr(vp)
        .arg_ptr(gp)
        .arg_ptr(bp)
        .arg_ptr(op);
    for &ip in &inters {
        launch = launch.arg_ptr(ip);
    }
    launch
        .arg_u32(1)
        .arg_u32(NK as u32)
        .arg_u32(NV as u32)
        .arg_u32(KD as u32)
        .arg_u32(VD as u32)
        .arg_u32((NK * KD) as u32)
        .arg_u32((NV * VD) as u32)
        .arg_u32(NV as u32)
        .arg_u32(0)
        .launch(0)?;
    g.synchronize(0)?;

    let out = dn_bf16(g, op, k * NV * VD)?;
    let mut inter_h = Vec::with_capacity(k - 1);
    for &ip in &inters {
        inter_h.push(if f16_state {
            dn_f16(g, ip, h_numel)?
        } else {
            dn_f32(g, ip, h_numel)?
        });
    }
    let final_h = if f16_state {
        dn_f16(g, hp, h_numel)?
    } else {
        dn_f32(g, hp, h_numel)?
    };
    for p in [hp, qp, kp, vp, gp, bp, op] {
        let _ = g.free(p);
    }
    for ip in inters {
        let _ = g.free(ip);
    }
    Ok((out, inter_h, final_h))
}

fn main() -> Result<()> {
    let g0 = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &g0;

    // 2026-09-25: (K, FP32 parent, FP16 twin, FP16 resident twin or None).
    let legs = [
        (
            2usize,
            g.kernel("gated_delta_rule_wy", "gated_delta_rule_wy2")?,
            g.kernel("gated_delta_rule_wy_f16", "gated_delta_rule_wy2_f16")?,
            Some(g.kernel(
                "gated_delta_rule_wy2_resident_f16",
                "gated_delta_rule_wy2_resident_f16",
            )?),
        ),
        (
            3usize,
            g.kernel("gated_delta_rule_wy3", "gated_delta_rule_wy3")?,
            g.kernel("gated_delta_rule_wy3_f16", "gated_delta_rule_wy3_f16")?,
            Some(g.kernel(
                "gated_delta_rule_wy3_resident_f16",
                "gated_delta_rule_wy3_resident_f16",
            )?),
        ),
        (
            4usize,
            g.kernel("gated_delta_rule_wy4", "gated_delta_rule_wy4")?,
            g.kernel("gated_delta_rule_wy4_f16", "gated_delta_rule_wy4_f16")?,
            None,
        ),
    ];

    let mut rng = Lcg(0x51ED_2701_ABCD_0F17);
    let h_numel = NV * KD * VD;
    // 2026-09-25: The initial state is FP16-representable, so the FP16 and FP32 launches
    // start from the same values.
    let h0: Vec<f32> = (0..h_numel)
        .map(|_| f16::from_f32(rng.r(-0.25, 0.25) as f32).to_f32())
        .collect();

    let mut all_ok = true;
    for (k, k32, k16, kres) in legs {
        let q: Vec<Vec<bf16>> = (0..k)
            .map(|_| {
                (0..NK * KD)
                    .map(|_| bf16::from_f64(rng.r(-1.0, 1.0)))
                    .collect()
            })
            .collect();
        let key: Vec<Vec<bf16>> = (0..k)
            .map(|_| {
                (0..NK * KD)
                    .map(|_| bf16::from_f64(rng.r(-1.0, 1.0)))
                    .collect()
            })
            .collect();
        let val: Vec<Vec<bf16>> = (0..k)
            .map(|_| {
                (0..NV * VD)
                    .map(|_| bf16::from_f64(rng.r(-1.0, 1.0)))
                    .collect()
            })
            .collect();
        let gate: Vec<Vec<f32>> = (0..k)
            .map(|_| (0..NV).map(|_| rng.r(0.80, 0.999) as f32).collect())
            .collect();
        let beta: Vec<Vec<f32>> = (0..k)
            .map(|_| (0..NV).map(|_| rng.r(0.0, 1.0) as f32).collect())
            .collect();

        let (ref_out, ref_h) = sequential_ref_f16(&h0, &q, &key, &val, &gate, &beta, k);
        let (o16, i16v, f16v) = run_wy(g, k16, &h0, &q, &key, &val, &gate, &beta, k, true)?;
        let (o32, i32v, f32v) = run_wy(g, k32, &h0, &q, &key, &val, &gate, &beta, k, false)?;

        // 2026-09-25: Check 2.
        let mut min_out = 1.0f64;
        for t in 0..k {
            let s = t * NV * VD;
            min_out = min_out.min(cos(&o16[s..s + NV * VD], &ref_out[t]));
        }
        let mut min_state = cos(&f16v, &ref_h[k - 1]);
        for (n, iv) in i16v.iter().enumerate() {
            min_state = min_state.min(cos(iv, &ref_h[n]));
        }
        let ref_ok = min_out >= PASS_COS && min_state >= PASS_COS;

        // 2026-09-25: Check 3, printed only.
        let mut div_out = 1.0f64;
        for t in 0..k {
            let s = t * NV * VD;
            div_out = div_out.min(cos(&o16[s..s + NV * VD], &o32[s..s + NV * VD]));
        }
        let mut div_state = cos(&f16v, &f32v);
        for (n, iv) in i16v.iter().enumerate() {
            div_state = div_state.min(cos(iv, &i32v[n]));
        }

        println!(
            "K={k} FP16-vs-FP16REF out_cos={min_out:.8} state_cos={min_state:.8} -> {}",
            if ref_ok { "PASS" } else { "FAIL" }
        );
        println!("K={k} FP16-vs-FP32 divergence out_cos={div_out:.8} state_cos={div_state:.8}");
        all_ok &= ref_ok;

        // 2026-09-25: Check 1.
        if let Some(kr) = kres {
            let (o_r, i_r, f_r) = run_wy(g, kr, &h0, &q, &key, &val, &gate, &beta, k, true)?;
            let mut bit_ok = bits_eq(&o_r, &o16) && bits_eq(&f_r, &f16v);
            for (a, b) in i_r.iter().zip(&i16v) {
                bit_ok &= bits_eq(a, b);
            }
            println!(
                "K={k} FP16 resident-vs-base BITWISE -> {}",
                if bit_ok { "PASS" } else { "FAIL" }
            );
            all_ok &= bit_ok;
        }
    }

    eprintln!(
        "\nWY-verify FP16 twin GATE: {}",
        if all_ok { "PASS" } else { "FAIL" }
    );
    if !all_ok {
        std::process::exit(1);
    }
    Ok(())
}
