// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Attribution bench for the divergence between the per-token GDN
//! decode chain and the chunked WY verify, on the GDN shape in the constants
//! below (NK=16, NV=48, KD=VD=128, D_CONV=4).
//!
//! Three arms start from one shared state and read identical BF16 qkvz inputs:
//!   D: causal_conv1d_update_l2norm_f32 -> gated_delta_rule_decode_f32, per token.
//!   H: arm V's BF16 conv output, widened to FP32 on the host ->
//!      gated_delta_rule_decode_f32, per token.
//!   V: causal_conv1d_update_l2norm (BF16) -> gated_delta_rule_wy4, one launch
//!      per K=4 window.
//!
//!   |D-H| isolates the BF16 rounding of the conv output;
//!   |H-V| isolates the WY chunked form against per-token decode, same inputs;
//!   |D-V| is the total conv+GDN divergence of one window.
//!
//! Phase 1 runs one window from the initial state. Phase 2 resets the states
//! and runs 400 windows (1600 tokens) with persistent per-arm state and fresh
//! shared inputs per window, printing the h-state relative L2 every 25
//! windows. Inputs never depend on outputs, so there is no feedback through
//! projections, attention or logits.
//!
//! Owner: model-arch examples.
//! Invariants: none beyond the types.
//!
//! Needs the qwen3.6-27b kernel set (`METRALE_TARGET_MODEL='*'` or that model).
//!   cargo run -p metrale-model-arch --release --example x435_ssm_attrib \
//!       --features cuda,gpu-examples

use anyhow::Result;
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};

const NK: usize = 16;
const NV: usize = 48;
const KD: usize = 128;
const VD: usize = 128;
const KEY_DIM: usize = NK * KD;
const VALUE_DIM: usize = NV * VD;
const CONV_DIM: usize = 2 * KEY_DIM + VALUE_DIM;
const D_CONV: usize = 4;
const H_NUMEL: usize = NV * KD * VD;
const K: usize = 4;

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

fn up_f32(g: &dyn GpuBackend, d: &[f32]) -> Result<DevicePtr> {
    let b: Vec<u8> = d.iter().flat_map(|x| x.to_le_bytes()).collect();
    let p = g.alloc(b.len().max(1))?;
    g.copy_h2d(&b, p)?;
    Ok(p)
}
fn dn_f32(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<f32>> {
    let mut b = vec![0u8; n * 4];
    g.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}
fn dn_bf16(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<f32>> {
    let mut b = vec![0u8; n * 2];
    g.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(2)
        .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect())
}

fn rel_l2(a: &[f32], b: &[f32]) -> f64 {
    let (mut d2, mut n2) = (0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        let d = (*x as f64) - (*y as f64);
        d2 += d * d;
        n2 += (*x as f64) * (*x as f64);
    }
    (d2 / (n2 + 1e-30)).sqrt()
}
fn max_abs_diff(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| ((*x as f64) - (*y as f64)).abs())
        .fold(0.0, f64::max)
}

#[allow(clippy::too_many_arguments)]
fn conv_launch(
    g: &dyn GpuBackend,
    k: KernelHandle,
    state: DevicePtr,
    input: DevicePtr,
    weight: DevicePtr,
    out: DevicePtr,
) -> Result<()> {
    KernelLaunch::new(g, k)
        .grid([div_ceil(CONV_DIM as u32, 256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(state)
        .arg_ptr(input)
        .arg_ptr(weight)
        .arg_ptr(DevicePtr::NULL)
        .arg_ptr(out)
        .arg_u32(1)
        .arg_u32(CONV_DIM as u32)
        .arg_u32(D_CONV as u32)
        .arg_u32((2 * KEY_DIM) as u32)
        .arg_u32(KD as u32)
        .arg_f32(1e-6)
        .launch(0)
}

#[allow(clippy::too_many_arguments)]
fn gdn_f32_launch(
    g: &dyn GpuBackend,
    k: KernelHandle,
    h: DevicePtr,
    q: DevicePtr,
    kk: DevicePtr,
    v: DevicePtr,
    gate: DevicePtr,
    beta: DevicePtr,
    out: DevicePtr,
) -> Result<()> {
    KernelLaunch::new(g, k)
        .grid([NV as u32, 1, 1])
        .block([128, 1, 1])
        .arg_ptr(h)
        .arg_ptr(q)
        .arg_ptr(kk)
        .arg_ptr(v)
        .arg_ptr(gate)
        .arg_ptr(beta)
        .arg_ptr(out)
        .arg_u32(1)
        .arg_u32(NK as u32)
        .arg_u32(NV as u32)
        .arg_u32(KD as u32)
        .arg_u32(VD as u32)
        .launch(0)
}

fn main() -> Result<()> {
    let set = metrale_kernels::ptx_for_model("qwen3.6-27b")
        .or_else(|| {
            metrale_kernels::ptx_for_config("qwen3_5_text", 5120, &[], None)
                .ok()
                .flatten()
        })
        .expect("no qwen3.6-27b ptx set — build with METRALE_TARGET_MODEL='*'");
    eprintln!("kernel set: {}", set.target.model);
    let g0 = MetraleCudaBackend::new(0, &set.modules)?;
    let g: &dyn GpuBackend = &g0;

    let conv_b = g.kernel("causal_conv1d", "causal_conv1d_update_l2norm")?;
    let conv_f = g.kernel("causal_conv1d", "causal_conv1d_update_l2norm_f32")?;
    let gdn_f32 = g.kernel("gated_delta_rule", "gated_delta_rule_decode_f32")?;
    let wy4 = g.kernel("gated_delta_rule_wy4", "gated_delta_rule_wy4")?;

    let mut r = Lcg(0x435435);

    // 2026-09-25: Shared conv weight `[CONV_DIM, D_CONV]` BF16.
    let wconv: Vec<u8> = (0..CONV_DIM * D_CONV)
        .flat_map(|_| bf16::from_f64(r.r(-0.5, 0.5)).to_bits().to_le_bytes())
        .collect();
    let wp = g.alloc(wconv.len())?;
    g.copy_h2d(&wconv, wp)?;

    // 2026-09-25: Shared initial states: h0 uniform in [-0.1, 0.1), conv window
    // uniform in [-1, 1).
    let h0: Vec<f32> = (0..H_NUMEL).map(|_| r.r(-0.1, 0.1) as f32).collect();
    let cs0: Vec<f32> = (0..CONV_DIM * D_CONV)
        .map(|_| r.r(-1.0, 1.0) as f32)
        .collect();

    let h_d = up_f32(g, &h0)?;
    let h_h = up_f32(g, &h0)?;
    let h_v = up_f32(g, &h0)?;
    let cs_d = up_f32(g, &cs0)?;
    let cs_v = up_f32(g, &cs0)?;

    let conv_out_f = g.alloc(K * CONV_DIM * 4)?;
    let conv_out_b = g.alloc(K * CONV_DIM * 2)?;
    let conv_out_h = g.alloc(K * CONV_DIM * 4)?;
    let out_d = g.alloc(K * VALUE_DIM * 4)?;
    let out_h = g.alloc(K * VALUE_DIM * 4)?;
    let out_v = g.alloc(K * VALUE_DIM * 2)?;
    let inter: Vec<DevicePtr> = (0..3)
        .map(|_| g.alloc(H_NUMEL * 4))
        .collect::<Result<_>>()?;
    let in_buf = g.alloc(K * CONV_DIM * 2)?;
    // 2026-09-25: `[K][gate(NV) | beta(NV)]` FP32.
    let gates_buf = g.alloc(K * 2 * NV * 4)?;

    // 2026-09-25: Per-head decay from 0.80 to 0.999 with +-0.005 jitter per token,
    // clamped to [0, 0.9995]; beta uniform in [0, 1). The measured drift depends
    // on this choice.
    let run_window = |r: &mut Lcg,
                      g: &dyn GpuBackend,
                      detail: bool|
     -> Result<(Vec<f64>, Vec<f64>, Vec<f64>)> {
        let inp: Vec<u8> = (0..K * CONV_DIM)
            .flat_map(|_| bf16::from_f64(r.r(-1.0, 1.0)).to_bits().to_le_bytes())
            .collect();
        g.copy_h2d(&inp, in_buf)?;
        let mut gb = vec![0f32; K * 2 * NV];
        for t in 0..K {
            for vh in 0..NV {
                let base = 0.80 + 0.199 * (vh as f64) / ((NV - 1) as f64);
                gb[t * 2 * NV + vh] = (base + r.r(-0.005, 0.005)).clamp(0.0, 0.9995) as f32;
                gb[t * 2 * NV + NV + vh] = r.r(0.0, 1.0) as f32;
            }
        }
        let gbb: Vec<u8> = gb.iter().flat_map(|x| x.to_le_bytes()).collect();
        g.copy_h2d(&gbb, gates_buf)?;

        for t in 0..K {
            let it = in_buf.offset(t * CONV_DIM * 2);
            let ot = conv_out_f.offset(t * CONV_DIM * 4);
            conv_launch(g, conv_f, cs_d, it, wp, ot)?;
            gdn_f32_launch(
                g,
                gdn_f32,
                h_d,
                ot,
                ot.offset(KEY_DIM * 4),
                ot.offset(2 * KEY_DIM * 4),
                gates_buf.offset(t * 2 * NV * 4),
                gates_buf.offset((t * 2 * NV + NV) * 4),
                out_d.offset(t * VALUE_DIM * 4),
            )?;
        }
        for t in 0..K {
            let it = in_buf.offset(t * CONV_DIM * 2);
            conv_launch(g, conv_b, cs_v, it, wp, conv_out_b.offset(t * CONV_DIM * 2))?;
        }
        KernelLaunch::new(g, wy4)
            .grid([NV as u32, 1, 1])
            .block([128, 1, 1])
            .arg_ptr(h_v)
            .arg_ptr(conv_out_b)
            .arg_ptr(conv_out_b.offset(KEY_DIM * 2))
            .arg_ptr(conv_out_b.offset(2 * KEY_DIM * 2))
            .arg_ptr(gates_buf)
            .arg_ptr(gates_buf.offset(NV * 4))
            .arg_ptr(out_v)
            .arg_ptr(inter[0])
            .arg_ptr(inter[1])
            .arg_ptr(inter[2])
            .arg_u32(1)
            .arg_u32(NK as u32)
            .arg_u32(NV as u32)
            .arg_u32(KD as u32)
            .arg_u32(VD as u32)
            .arg_u32(CONV_DIM as u32)
            .arg_u32(CONV_DIM as u32)
            .arg_u32((2 * NV) as u32)
            .arg_u32(0)
            .launch(0)?;
        g.synchronize(0)?;

        let cb = dn_bf16(g, conv_out_b, K * CONV_DIM)?;
        let cbb: Vec<u8> = cb.iter().flat_map(|x| x.to_le_bytes()).collect();
        g.copy_h2d(&cbb, conv_out_h)?;
        for t in 0..K {
            let ot = conv_out_h.offset(t * CONV_DIM * 4);
            gdn_f32_launch(
                g,
                gdn_f32,
                h_h,
                ot,
                ot.offset(KEY_DIM * 4),
                ot.offset(2 * KEY_DIM * 4),
                gates_buf.offset(t * 2 * NV * 4),
                gates_buf.offset((t * 2 * NV + NV) * 4),
                out_h.offset(t * VALUE_DIM * 4),
            )?;
        }
        g.synchronize(0)?;

        if detail {
            // 2026-09-25: Print how many elements of bf16(arm D conv output) differ
            // from arm V's BF16 conv output, and how many conv-state words differ.
            let cf = dn_f32(g, conv_out_f, K * CONV_DIM)?;
            let mism = cf
                .iter()
                .zip(&cb)
                .filter(|(a, b)| bf16::from_f32(**a).to_bits() != bf16::from_f32(**b).to_bits())
                .count();
            eprintln!(
                "conv twin check: bf16(f32_out) vs bf16_out mismatches = {mism}/{}",
                K * CONV_DIM
            );
            let sd = dn_f32(g, cs_d, CONV_DIM * D_CONV)?;
            let sv = dn_f32(g, cs_v, CONV_DIM * D_CONV)?;
            let smism = sd
                .iter()
                .zip(&sv)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            eprintln!(
                "conv state parity D vs V: mismatches = {smism}/{}",
                CONV_DIM * D_CONV
            );
            let od = dn_f32(g, out_d, K * VALUE_DIM)?;
            let oh = dn_f32(g, out_h, K * VALUE_DIM)?;
            let ov = dn_bf16(g, out_v, K * VALUE_DIM)?;
            for t in 0..K {
                let s = t * VALUE_DIM..(t + 1) * VALUE_DIM;
                eprintln!(
                    "  out tok{t}: relL2 D-H(conv prec)={:.3e}  H-V(WY form+bf16out)={:.3e}  D-V(total)={:.3e}  maxabs D-V={:.3e}",
                    rel_l2(&od[s.clone()], &oh[s.clone()]),
                    rel_l2(&oh[s.clone()], &ov[s.clone()]),
                    rel_l2(&od[s.clone()], &ov[s.clone()]),
                    max_abs_diff(&od[s.clone()], &ov[s])
                );
            }
        }

        let hd = dn_f32(g, h_d, H_NUMEL)?;
        let hh = dn_f32(g, h_h, H_NUMEL)?;
        let hv = dn_f32(g, h_v, H_NUMEL)?;
        Ok((
            vec![rel_l2(&hd, &hh)],
            vec![rel_l2(&hh, &hv)],
            vec![rel_l2(&hd, &hv)],
        ))
    };

    eprintln!("== PHASE 1: single K=4 verify window, fresh state ==");
    let (dh, hv, dv) = run_window(&mut r, g, true)?;
    eprintln!(
        "h_state after 4 tokens: relL2 D-H(conv prec)={:.3e}  H-V(WY form)={:.3e}  D-V(total)={:.3e}",
        dh[0], hv[0], dv[0]
    );

    eprintln!("== PHASE 2: closed-loop drift, 400 windows (1600 tokens), shared inputs ==");
    let hb: Vec<u8> = h0.iter().flat_map(|x| x.to_le_bytes()).collect();
    let cb0: Vec<u8> = cs0.iter().flat_map(|x| x.to_le_bytes()).collect();
    for p in [h_d, h_h, h_v] {
        g.copy_h2d(&hb, p)?;
    }
    for p in [cs_d, cs_v] {
        g.copy_h2d(&cb0, p)?;
    }
    eprintln!("tokens, relL2_D-H_convprec, relL2_H-V_wyform, relL2_D-V_total");
    for w in 1..=400usize {
        let detail = false;
        let sample = w == 1 || w % 25 == 0;
        if sample {
            let (dh, hv, dv) = run_window(&mut r, g, detail)?;
            eprintln!("{}, {:.4e}, {:.4e}, {:.4e}", 4 * w, dh[0], hv[0], dv[0]);
        } else {
            let _ = run_window(&mut r, g, detail)?;
        }
    }
    eprintln!("done");
    Ok(())
}
