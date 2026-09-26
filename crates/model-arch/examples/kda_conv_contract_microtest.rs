// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Gate for the conv kernels at GLM-5.3 KDA geometry, against the golden
//! `gen_kda_conv_golden.py` writes.
//!
//! Owner: model-arch examples (GLM-5.3 KDA kernels).
//! Invariants:
//! - The run panics unless `qk_channels % 256 == 0` and `head_dim == 128`, the shape
//!   `causal_conv1d_update_l2norm` requires, and fails unless every gated check passes.
//!
//! Geometry comes from the golden's `fixture`: at GLM-5.3's 64 KDA heads of 128,
//! `conv_dim = 3 * 64 * 128` and `qk_channels = 2 * 64 * 128`, with a kernel of 4.
//!
//! Two paths:
//! * decode: `causal_conv1d_update_l2norm` fuses conv, SiLU and the L2 norm of the q|k heads;
//! * prefill: `causal_conv1d_update_prefill` does conv and SiLU, then `l2_norm_bf16` normalises
//!   the q|k channels only. The run checks that L2 ran exactly once: q|k rows have unit norm and
//!   V rows do not.
//!
//! State width: the golden's state is `hf_state_width` (3) slots per channel. The kernels keep
//! 4 and shift left before convolving, so slot 0 never participates: golden `[0..3]` is
//! kernel `[1..4]`. The run widens the golden state and checks that poisoning slot 0 changes
//! nothing.
//!
//!   cargo run -p metrale-model-arch --release --example kda_conv_contract_microtest \
//!       --features cuda,gpu-examples

use anyhow::{Result, bail};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use serde_json::Value;

#[path = "common/golden.rs"]
mod golden;

static GOLDEN: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
    golden::load(
        "crates/model-arch/src/glm5next_kda_ref/kda_conv_golden.json",
        "gen_kda_conv_golden.py",
    )
});

/// 2026-09-25: Inputs, weights and outputs are BF16, so the achievable error is set by that
/// rounding. The decode figures are printed beside a CPU reference fed the same rounded values,
/// which separates kernel error from input rounding.
const MAX_ABS_BF16: f64 = 6.0e-3;

fn up_f32(g: &dyn GpuBackend, d: &[f32]) -> Result<DevicePtr> {
    let b: Vec<u8> = d.iter().flat_map(|x| x.to_le_bytes()).collect();
    let p = g.alloc(b.len().max(1))?;
    g.copy_h2d(&b, p)?;
    Ok(p)
}
fn up_bf16(g: &dyn GpuBackend, d: &[f32]) -> Result<DevicePtr> {
    let b: Vec<u8> = d
        .iter()
        .flat_map(|x| bf16::from_f32(*x).to_bits().to_le_bytes())
        .collect();
    let p = g.alloc(b.len().max(1))?;
    g.copy_h2d(&b, p)?;
    Ok(p)
}
fn down_bf16(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<f32>> {
    let mut b = vec![0u8; n * 2];
    g.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(2)
        .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect())
}
fn down_f32(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<f32>> {
    let mut b = vec![0u8; n * 4];
    g.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}
fn arr(v: &Value, name: &str) -> Vec<f32> {
    v["outputs"][name]["data"]
        .as_array()
        .unwrap_or_else(|| panic!("missing {name}"))
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect()
}
fn maxabs(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len(), "len {} vs {}", a.len(), b.len());
    a.iter()
        .zip(b)
        .fold(0.0f64, |m, (x, y)| m.max((*x as f64 - *y as f64).abs()))
}
fn checksum(s: &[f32]) -> f64 {
    s.iter()
        .enumerate()
        .map(|(i, v)| *v as f64 * (i as f64 + 1.0))
        .sum()
}
fn sample(s: &[f32], stride: usize) -> Vec<f32> {
    s.iter().step_by(stride).copied().collect()
}

struct Lcg(u64);
impl Lcg {
    fn u(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (((self.0 >> 40) as f32) / ((1u32 << 24) as f32)) * 2.0 - 1.0
    }
    fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.u()).collect()
    }
}

/// 2026-09-25: CPU conv + SiLU on the same BF16-rounded token and weights the GPU reads.
fn cpu_conv_silu(
    state4: &[f32],
    tok: &[f32],
    w: &[f32],
    dim: usize,
    ks: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut st = state4.to_vec();
    let mut out = vec![0.0f32; dim];
    for ch in 0..dim {
        let s = &mut st[ch * ks..(ch + 1) * ks];
        for i in 0..ks - 1 {
            s[i] = s[i + 1];
        }
        s[ks - 1] = bf16::from_f32(tok[ch]).to_f32();
        let mut acc = 0.0f32;
        for k in 0..ks {
            acc += s[k] * bf16::from_f32(w[ch * ks + k]).to_f32();
        }
        out[ch] = acc * (1.0 / (1.0 + (-acc).exp()));
    }
    (out, st)
}

fn l2_rows(x: &mut [f32], d: usize, upto: usize) {
    for r in x[..upto].chunks_exact_mut(d) {
        let inv = 1.0 / (r.iter().map(|a| a * a).sum::<f32>() + 1e-6).sqrt();
        for a in r.iter_mut() {
            *a *= inv;
        }
    }
}

fn main() -> Result<()> {
    let g = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let gpu: &dyn GpuBackend = &g;
    let v: Value = serde_json::from_str(&GOLDEN)?;
    let f = &v["fixture"];
    let dim = f["conv_dim"].as_u64().unwrap() as usize;
    let qk = f["qk_channels"].as_u64().unwrap() as usize;
    let hd = f["head_dim"].as_u64().unwrap() as usize;
    let ks = f["kernel"].as_u64().unwrap() as usize;
    let tpre = f["t_prefill"].as_u64().unwrap() as usize;
    let stride = f["sample_stride"].as_u64().unwrap() as usize;
    let hf_w = f["hf_state_width"].as_u64().unwrap() as usize;

    println!("conv contract at production geometry");
    println!(
        "  conv_dim={dim} qk_channels={qk} head_dim={hd} kernel={ks} act={}",
        f["activation"]
    );
    println!("  qk_channels % 256 = {}  (kernel requires 0)", qk % 256);
    assert_eq!(qk % 256, 0);
    assert_eq!(
        hd, 128,
        "the fused kernel hardcodes 2 heads per 256-thread block"
    );

    let k_dec = gpu.kernel("causal_conv1d", "causal_conv1d_update_l2norm")?;
    let k_pre = gpu.kernel("causal_conv1d", "causal_conv1d_update_prefill")?;
    let k_l2 = gpu.kernel("norm", "l2_norm_bf16")?;
    println!("  all three conv/L2 entry points resolved (no fallback)\n");

    // 2026-09-25: The LCG must reproduce the generator's probe values, or the inputs differ
    // from the ones the golden was made from.
    let probe: Vec<f32> = v["lcg_probe"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap() as f32)
        .collect();
    let mut pr = Lcg(0x5EED_C0F0);
    if pr
        .vec(probe.len())
        .iter()
        .zip(&probe)
        .any(|(a, b)| a.to_bits() != b.to_bits())
    {
        bail!("LCG mismatch with the generator");
    }

    let mut rng = Lcg(0x5EED_C0F0);
    let weight: Vec<f32> = rng.vec(dim * ks).iter().map(|x| x * 0.5).collect();
    let state3: Vec<f32> = rng.vec(dim * hf_w).iter().map(|x| x * 0.5).collect();
    let tok = rng.vec(dim);
    let pre = rng.vec(tpre * dim);

    // 2026-09-25: Widen the golden's 3-wide state to the kernels' 4 slots; slot 0 is shifted
    // out before the conv, and `poison` fills it to show that.
    let widen = |poison: f32| -> Vec<f32> {
        let mut s = vec![0.0f32; dim * ks];
        for ch in 0..dim {
            s[ch * ks] = poison;
            for i in 0..hf_w {
                s[ch * ks + 1 + i] = state3[ch * hf_w + i];
            }
        }
        s
    };

    let mut ok = true;

    let dw = up_bf16(gpu, &weight)?;
    let dtok = up_bf16(gpu, &tok)?;
    let run_decode = |st: &[f32]| -> Result<(Vec<f32>, Vec<f32>)> {
        let dstate = up_f32(gpu, st)?;
        let dout = gpu.alloc(dim * 2)?;
        KernelLaunch::new(gpu, k_dec)
            .grid([(dim as u32).div_ceil(256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(dstate)
            .arg_ptr(dtok)
            .arg_ptr(dw)
            .arg_ptr(DevicePtr::NULL)
            .arg_ptr(dout)
            .arg_u32(1)
            .arg_u32(dim as u32)
            .arg_u32(ks as u32)
            .arg_u32(qk as u32)
            .arg_u32(hd as u32)
            .arg_f32(1e-6)
            .launch(0)?;
        gpu.synchronize(0)?;
        Ok((down_bf16(gpu, dout, dim)?, down_f32(gpu, dstate, dim * ks)?))
    };

    let (d_out, d_state) = run_decode(&widen(0.0))?;
    let want_out = arr(&v, "decode_out");
    let (mut cpu_out, cpu_state) = cpu_conv_silu(&widen(0.0), &tok, &weight, dim, ks);
    let cpu_state_tail: Vec<f32> = (0..dim)
        .flat_map(|ch| (1..ks).map(move |i| (ch, i)))
        .map(|(ch, i)| cpu_state[ch * ks + i])
        .collect();
    l2_rows(&mut cpu_out, hd, qk);
    let cpu_out_b: Vec<f32> = cpu_out
        .iter()
        .map(|x| bf16::from_f32(*x).to_f32())
        .collect();

    let e_hf = maxabs(&d_out, &want_out);
    let e_floor = maxabs(&cpu_out_b, &want_out);
    let e_kern = maxabs(&d_out, &cpu_out_b);
    println!("DECODE (conv + SiLU + L2 fused)");
    println!("  CPU-ref(bf16) vs HF   max_abs={e_floor:.3e}   <- input-rounding floor");
    println!("  GPU vs HF             max_abs={e_hf:.3e}");
    println!("  GPU vs CPU-ref        max_abs={e_kern:.3e}   <- kernel only");
    // 2026-09-25: Compare the overlapping window: kernel slots [1..4] against golden [0..3].
    let d_state_tail: Vec<f32> = (0..dim)
        .flat_map(|ch| (1..ks).map(move |i| (ch, i)))
        .map(|(ch, i)| d_state[ch * ks + i])
        .collect();
    let s_hf = maxabs(
        &sample(&d_state_tail, stride),
        &arr(&v, "decode_state_sample"),
    );
    let s_cpu = maxabs(&d_state_tail, &cpu_state_tail);
    println!(
        "  state Metrale Engine[1..4] vs HF[0..3] max_abs={s_hf:.3e}   vs CPU-ref {s_cpu:.3e}"
    );
    ok &= s_hf <= MAX_ABS_BF16;
    ok &= e_hf <= MAX_ABS_BF16 && e_kern <= MAX_ABS_BF16;

    let (p_out, _) = run_decode(&widen(1.0e6))?;
    let poison = maxabs(&p_out, &d_out);
    println!(
        "  poisoned state slot 0 -> output delta {poison:.3e}  [{}]",
        if poison == 0.0 {
            "ok, slot 0 is shifted out"
        } else {
            "FAIL"
        }
    );
    ok &= poison == 0.0;

    // 2026-09-25: L2 applied exactly once: every q|k head row has unit norm, and V rows do not.
    let qk_norms: Vec<f32> = d_out[..qk]
        .chunks_exact(hd)
        .map(|r| r.iter().map(|a| a * a).sum::<f32>().sqrt())
        .collect();
    let v_norms: Vec<f32> = d_out[qk..]
        .chunks_exact(hd)
        .map(|r| r.iter().map(|a| a * a).sum::<f32>().sqrt())
        .collect();
    let (qmin, qmax) = (
        qk_norms.iter().cloned().fold(f32::MAX, f32::min),
        qk_norms.iter().cloned().fold(0.0, f32::max),
    );
    let vmax = v_norms.iter().cloned().fold(0.0, f32::max);
    println!(
        "  q|k row norms in [{qmin:.6}, {qmax:.6}] (must be ~1); V max norm {vmax:.4} (must NOT be 1)"
    );
    ok &= (qmin - 1.0).abs() < 0.02 && (qmax - 1.0).abs() < 0.02 && (vmax - 1.0).abs() > 0.05;

    // 2026-09-25: Prefill: conv + SiLU over `t_prefill` tokens from a zero state, then L2 on
    // the q|k channels only.
    let dpre = up_bf16(gpu, &pre)?;
    let dstate = up_f32(gpu, &vec![0.0f32; dim * ks])?;
    let dpout = gpu.alloc(tpre * dim * 2)?;
    KernelLaunch::new(gpu, k_pre)
        .grid([(dim as u32).div_ceil(256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(dstate)
        .arg_ptr(dpre)
        .arg_ptr(dw)
        .arg_ptr(DevicePtr::NULL)
        .arg_ptr(dpout)
        .arg_u32(dim as u32)
        .arg_u32(ks as u32)
        .arg_u32(tpre as u32)
        .arg_u32(dim as u32)
        .arg_u32(dim as u32)
        .launch(0)?;
    // 2026-09-25: Grid x is `qk_channels / head_dim` heads, so the V channels are not touched.
    KernelLaunch::new(gpu, k_l2)
        .grid([(qk / hd) as u32, tpre as u32, 1])
        .block([hd as u32, 1, 1])
        .arg_ptr(dpout)
        .arg_u32(hd as u32)
        .arg_f32(1e-6)
        .arg_u32(dim as u32)
        .launch(0)?;
    gpu.synchronize(0)?;
    let p_out = down_bf16(gpu, dpout, tpre * dim)?;
    let p_state = down_f32(gpu, dstate, dim * ks)?;

    println!("\nPREFILL (conv + SiLU, then a SEPARATE L2 on q|k only)");
    let po_hf = maxabs(&sample(&p_out, stride), &arr(&v, "prefill_out_sample"));
    let po_ck = checksum(&p_out);
    let want_ck = v["checksums"]["prefill_out"].as_f64().unwrap();
    println!("  out sample vs HF      max_abs={po_hf:.3e}");
    println!(
        "  full-out fp64 checksum rel={:.3e}",
        (po_ck - want_ck).abs() / want_ck.abs().max(1.0)
    );
    ok &= po_hf <= MAX_ABS_BF16;

    let metrale_tail: Vec<f32> = (0..dim)
        .flat_map(|ch| (1..ks).map(move |i| (ch, i)))
        .map(|(ch, i)| p_state[ch * ks + i])
        .collect();
    let ps_hf = maxabs(
        &sample(&metrale_tail, stride),
        &arr(&v, "prefill_state_sample"),
    );
    println!("  final conv state (Metrale Engine[1..4] vs HF[0..3]) max_abs={ps_hf:.3e}");
    ok &= ps_hf <= MAX_ABS_BF16;

    let pqk: Vec<f32> = p_out[..qk]
        .chunks_exact(hd)
        .map(|r| r.iter().map(|a| a * a).sum::<f32>().sqrt())
        .collect();
    let (pmin, pmax) = (
        pqk.iter().cloned().fold(f32::MAX, f32::min),
        pqk.iter().cloned().fold(0.0, f32::max),
    );
    println!(
        "  token-0 q|k row norms in [{pmin:.6}, {pmax:.6}] (must be ~1, i.e. L2 ran exactly once)"
    );
    ok &= (pmin - 1.0).abs() < 0.02 && (pmax - 1.0).abs() < 0.02;

    // 2026-09-25: Report how far a second L2 moves the output; the exactly-once check is
    // meaningful only if that is visible. This line is printed, not gated.
    let mut twice = p_out.clone();
    for t in 0..tpre {
        l2_rows(&mut twice[t * dim..(t + 1) * dim], hd, qk);
    }
    let dbl = maxabs(&twice, &p_out);
    println!(
        "  applying L2 a SECOND time moves the result by {dbl:.3e}  [{}]",
        if dbl > 1e-4 {
            "ok, detectable"
        } else {
            "not detectable at bf16 -- see report"
        }
    );

    if ok {
        println!("\nCONV CONTRACT: PASS");
        Ok(())
    } else {
        bail!("CONV CONTRACT: FAIL")
    }
}
