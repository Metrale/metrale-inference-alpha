// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The K = 2..=8 WY verify GDN kernels against an f64 token-by-token GDN
//! recurrence, at Qwen3.6-35B-A3B GDN dimensions: `gated_delta_rule_wy2/wy3/wy4` with
//! per-intermediate pointers, `gated_delta_rule_wy5..wy8` (module `gated_delta_rule_wyn`)
//! with a contiguous intermediate pool.
//!
//! Owner: model-arch examples.
//! Invariants:
//! - Exit 1 unless every check below passes for every K and each of 6 seeds.
//!
//! Per K and seed, with random H_0 / q / k / v / gate / beta:
//!   (1) the reference captures each token's output and the state after each token;
//!   (2) the matching kernel runs once;
//!   (3) per-token output cosine >= 0.99999 and state cosine >= 0.99999 (the final state
//!       against the state after token K-1, intermediate n-1 against the state after
//!       token n-1).
//!
//! For K = 2 and 3, `gated_delta_rule_wy2_resident` / `gated_delta_rule_wy3_resident` must
//! also equal the base kernels bit for bit on output, intermediates and final state.
//!
//!   cargo run -p metrale-model-arch --release --example gdn_wy_verify_microtest \
//!       --features cuda,gpu-examples
use anyhow::Result;
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

// 2026-09-25: GDN dimensions of kernels/gb10/qwen3.6-35b-a3b/MODEL.toml.
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
fn dn_bf16(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<f32>> {
    let mut b = vec![0u8; n * 2];
    g.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(2)
        .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect())
}
fn dn_f32(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<f32>> {
    let mut b = vec![0u8; n * 4];
    g.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}
/// 2026-09-25: Bitwise equality over f32 slices, for the resident-twin check.
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

/// 2026-09-25: Token-by-token f64 GDN recurrence: BF16 inputs, gate clamped to
/// [1e-6, 1 - 1e-6], output scaled by 1/sqrt(k_dim). Returns (per-token outputs
/// `[k][NV*VD]`, the state after each token `[k][NV*KD*VD]` in `[vh][kd][vd]` layout).
fn sequential_ref(
    h0: &[f32],
    q: &[Vec<bf16>],
    key: &[Vec<bf16>],
    val: &[Vec<bf16>],
    gate: &[Vec<f32>],
    beta: &[Vec<f32>],
    k: usize,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let scale = (KD as f64).powf(-0.5);
    let mut s: Vec<f64> = h0.iter().map(|&x| x as f64).collect();
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
                    let hn = gg * s[idx] + key[t][kh * KD + kk].to_f64() * vnew;
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

/// 2026-09-25: Run a WY{K} kernel (K <= 4) once at batch 1, tokens packed with
/// qk_stride = NK*KD, v_stride = NV*VD and gb_stride = NV. Returns (output rows by token,
/// intermediates[0..K-1], final H).
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
    let hp = up_f32(g, h0)?;
    let qp = up_bf16(g, &q_flat)?;
    let kp = up_bf16(g, &k_flat)?;
    let vp = up_bf16(g, &v_flat)?;
    let gp = up_f32(g, &g_flat)?;
    let bp = up_f32(g, &b_flat)?;
    let op = g.alloc(k * NV * VD * 2)?;
    // 2026-09-25: K-1 intermediates: inter[i] is the state after token i.
    let inters: Vec<DevicePtr> = (0..k - 1)
        .map(|_| g.alloc(NV * KD * VD * 4))
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
        inter_h.push(dn_f32(g, ip, NV * KD * VD)?);
    }
    let final_h = dn_f32(g, hp, NV * KD * VD)?;
    for p in [hp, qp, kp, vp, gp, bp, op] {
        let _ = g.free(p);
    }
    for ip in inters {
        let _ = g.free(ip);
    }
    Ok((out, inter_h, final_h))
}

/// 2026-09-25: Run a wy5..wy8 kernel once: one contiguous pool of K-1 intermediates,
/// `inter_stride_floats` apart. Same input packing and return as `run_wy`.
fn run_wyn(
    g: &dyn GpuBackend,
    kernel: metrale_gpu_runtime::gpu::KernelHandle,
    h0: &[f32],
    q: &[Vec<bf16>],
    key: &[Vec<bf16>],
    val: &[Vec<bf16>],
    gate: &[Vec<f32>],
    beta: &[Vec<f32>],
    k: usize,
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
    let hp = up_f32(g, h0)?;
    let qp = up_bf16(g, &q_flat)?;
    let kp = up_bf16(g, &k_flat)?;
    let vp = up_bf16(g, &v_flat)?;
    let gp = up_f32(g, &g_flat)?;
    let bp = up_f32(g, &b_flat)?;
    let op = g.alloc(k * NV * VD * 2)?;
    let inter_pool = g.alloc((k - 1) * h_numel * 4)?;

    KernelLaunch::new(g, kernel)
        .grid([NV as u32, 1, 1])
        .block([128, 1, 1])
        .arg_ptr(hp)
        .arg_ptr(qp)
        .arg_ptr(kp)
        .arg_ptr(vp)
        .arg_ptr(gp)
        .arg_ptr(bp)
        .arg_ptr(op)
        .arg_ptr(inter_pool)
        .arg_u32(h_numel as u32)
        .arg_u32(1)
        .arg_u32(NK as u32)
        .arg_u32(NV as u32)
        .arg_u32(KD as u32)
        .arg_u32(VD as u32)
        .arg_u32((NK * KD) as u32)
        .arg_u32((NV * VD) as u32)
        .arg_u32(NV as u32)
        .launch(0)?;
    g.synchronize(0)?;

    let out = dn_bf16(g, op, k * NV * VD)?;
    let mut inter_h = Vec::with_capacity(k - 1);
    for t in 0..k - 1 {
        inter_h.push(dn_f32(g, inter_pool.offset(t * h_numel * 4), h_numel)?);
    }
    let final_h = dn_f32(g, hp, h_numel)?;
    for p in [hp, qp, kp, vp, gp, bp, op, inter_pool] {
        let _ = g.free(p);
    }
    Ok((out, inter_h, final_h))
}

fn main() -> Result<()> {
    let g0 = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &g0;
    let wy = [
        (
            2usize,
            g.kernel("gated_delta_rule_wy", "gated_delta_rule_wy2")?,
        ),
        (
            3usize,
            g.kernel("gated_delta_rule_wy3", "gated_delta_rule_wy3")?,
        ),
        (
            4usize,
            g.kernel("gated_delta_rule_wy4", "gated_delta_rule_wy4")?,
        ),
        (
            5usize,
            g.kernel("gated_delta_rule_wyn", "gated_delta_rule_wy5")?,
        ),
        (
            6usize,
            g.kernel("gated_delta_rule_wyn", "gated_delta_rule_wy6")?,
        ),
        (
            7usize,
            g.kernel("gated_delta_rule_wyn", "gated_delta_rule_wy7")?,
        ),
        (
            8usize,
            g.kernel("gated_delta_rule_wyn", "gated_delta_rule_wy8")?,
        ),
    ];

    let mut all_ok = true;
    for &k in &[2usize, 3, 4, 5, 6, 7, 8] {
        let wk = wy.iter().find(|(kk, _)| *kk == k).unwrap().1;
        for layer in 0..6u64 {
            let mut r = Lcg(0xD17A ^ (k as u64) << 8 ^ layer);
            let h0: Vec<f32> = (0..NV * KD * VD).map(|_| r.r(-0.1, 0.1) as f32).collect();
            let q: Vec<Vec<bf16>> = (0..k)
                .map(|_| {
                    (0..NK * KD)
                        .map(|_| bf16::from_f64(r.r(-0.5, 0.5)))
                        .collect()
                })
                .collect();
            let key: Vec<Vec<bf16>> = (0..k)
                .map(|_| {
                    (0..NK * KD)
                        .map(|_| bf16::from_f64(r.r(-0.5, 0.5)))
                        .collect()
                })
                .collect();
            let val: Vec<Vec<bf16>> = (0..k)
                .map(|_| {
                    (0..NV * VD)
                        .map(|_| bf16::from_f64(r.r(-0.5, 0.5)))
                        .collect()
                })
                .collect();
            let gate: Vec<Vec<f32>> = (0..k)
                .map(|_| (0..NV).map(|_| r.r(0.80, 0.999) as f32).collect())
                .collect();
            let beta: Vec<Vec<f32>> = (0..k)
                .map(|_| (0..NV).map(|_| r.r(0.0, 1.0) as f32).collect())
                .collect();

            let (ref_out, ref_h) = sequential_ref(&h0, &q, &key, &val, &gate, &beta, k);
            let (wy_out, wy_inter, wy_final) = if k <= 4 {
                run_wy(g, wk, &h0, &q, &key, &val, &gate, &beta, k)?
            } else {
                run_wyn(g, wk, &h0, &q, &key, &val, &gate, &beta, k)?
            };

            // 2026-09-25: wy_out is [token, vh, vd].
            let mut min_out_cos = 1.0f64;
            for t in 0..k {
                let wy_t = &wy_out[t * NV * VD..(t + 1) * NV * VD];
                min_out_cos = min_out_cos.min(cos(wy_t, &ref_out[t]));
            }

            // 2026-09-25: Final state against ref_h[k-1]; intermediate n-1 against ref_h[n-1].
            let mut min_state_cos = cos(&wy_final, &ref_h[k - 1]);
            for n in 1..k {
                min_state_cos = min_state_cos.min(cos(&wy_inter[n - 1], &ref_h[n - 1]));
            }

            let ok = min_out_cos >= PASS_COS && min_state_cos >= PASS_COS;
            all_ok &= ok;
            eprintln!(
                "K={k} layer={layer}  out_cos={min_out_cos:.7} state_cos={min_state_cos:.7}  {}",
                if ok { "PASS" } else { "FAIL" }
            );

            // 2026-09-25: The resident twins keep each value's expression and accumulation
            // order, so they are compared bit for bit, not by cosine.
            if k == 2 || k == 3 {
                let name = if k == 2 {
                    "gated_delta_rule_wy2_resident"
                } else {
                    "gated_delta_rule_wy3_resident"
                };
                let wk_res = g.kernel(name, name)?;
                let (res_out, res_inter, res_final) =
                    run_wy(g, wk_res, &h0, &q, &key, &val, &gate, &beta, k)?;
                let pok = bits_eq(&res_out, &wy_out)
                    && res_inter.len() == wy_inter.len()
                    && res_inter.iter().zip(&wy_inter).all(|(a, b)| bits_eq(a, b))
                    && bits_eq(&res_final, &wy_final);
                all_ok &= pok;
                eprintln!(
                    "K={k} layer={layer}  wy{k}_resident BITWISE parity: {}",
                    if pok { "PASS" } else { "FAIL" }
                );
            }
        }
    }
    eprintln!(
        "\nWY-verify equivalence GATE: {}",
        if all_ok { "PASS" } else { "FAIL" }
    );
    if !all_ok {
        std::process::exit(1);
    }
    Ok(())
}
