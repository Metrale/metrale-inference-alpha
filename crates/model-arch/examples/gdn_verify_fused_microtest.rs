// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The fused K=2 GDN verify kernels (`gdn_verify_fused_k2.cu`) against the
//! per-token path, at Qwen3.6-35B-A3B GDN dimensions.
//!
//! Owner: model-arch examples.
//! Invariants:
//! - Exit 1 unless, for each of 6 seeds, stage 0 reproduces bit for bit and every stage-1
//!   cosine is >= `PASS_COS`.
//!
//!   Stage 0 (golden): `causal_conv1d_update_l2norm` per token, `gated_delta_rule_wy2`,
//!     then `gated_rms_norm` per token. It captures the gated-norm outputs, the state after
//!     token 1 (committed) and after token 0 (h_inter), and the conv state after each
//!     token. Two runs on the same seed must match bytewise.
//!
//!   Stage 1 (fused): `gdn_verify_fused_conv_k2` for both positions in one launch, the
//!     same `gated_delta_rule_wy2` launch, then `gdn_verify_fused_norm_k2` for both
//!     positions in one launch. The per-token gated-norm outputs and both conv states
//!     must reach cosine >= `PASS_COS` against stage 0.
//!
//!   cargo run -p metrale-model-arch --release --example gdn_verify_fused_microtest \
//!       --features cuda,gpu-examples
use anyhow::Result;
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

// 2026-09-25: GDN dimensions and conv width of kernels/gb10/qwen3.6-35b-a3b/MODEL.toml.
const KD: usize = 128;
const VD: usize = 128;
const NK: usize = 16;
const NV: usize = 32;
const D_CONV: usize = 4;

const KEY_DIM: usize = NK * KD;
const VALUE_DIM: usize = NV * VD;
const CONV_DIM: usize = KEY_DIM * 2 + VALUE_DIM;
const QK_CH: usize = KEY_DIM * 2;
const QKVZ_SIZE: usize = CONV_DIM + VALUE_DIM;

const L2_EPS: f32 = 1e-6;
const RMS_EPS: f32 = 1e-6;
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
fn cos(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        dot += (*x as f64) * (*y as f64);
        na += (*x as f64).powi(2);
        nb += (*y as f64).powi(2);
    }
    dot / (na.sqrt() * nb.sqrt() + 1e-12)
}

/// 2026-09-25: Random K=2 layer inputs:
///   - `deinterleaved`: [K, QKVZ_SIZE] BF16, each row Q | K | V | Z
///   - `conv_state0`:   [CONV_DIM, D_CONV] FP32, the initial conv window
///   - `conv_weight`:   [CONV_DIM, D_CONV] BF16
///   - `h0`:            [NV, KD, VD] FP32, the initial GDN state
///   - `gates`:         [K, 2 * NV] FP32, each row gate then beta
///   - `norm_weight`:   [VD] BF16
struct Inputs {
    deinterleaved: Vec<bf16>,
    conv_state0: Vec<f32>,
    conv_weight: Vec<bf16>,
    h0: Vec<f32>,
    gates: Vec<f32>,
    norm_weight: Vec<bf16>,
}

fn gen_inputs(seed: u64) -> Inputs {
    let mut r = Lcg(seed);
    let deinterleaved: Vec<bf16> = (0..2 * QKVZ_SIZE)
        .map(|_| bf16::from_f64(r.r(-0.5, 0.5)))
        .collect();
    let conv_state0: Vec<f32> = (0..CONV_DIM * D_CONV)
        .map(|_| r.r(-0.3, 0.3) as f32)
        .collect();
    let conv_weight: Vec<bf16> = (0..CONV_DIM * D_CONV)
        .map(|_| bf16::from_f64(r.r(-0.3, 0.3)))
        .collect();
    let h0: Vec<f32> = (0..NV * KD * VD).map(|_| r.r(-0.1, 0.1) as f32).collect();
    let mut gates = Vec::with_capacity(2 * 2 * NV);
    for _ in 0..2 {
        for _ in 0..NV {
            gates.push(r.r(0.80, 0.999) as f32);
        }
        for _ in 0..NV {
            gates.push(r.r(0.0, 1.0) as f32);
        }
    }
    let norm_weight: Vec<bf16> = (0..VD).map(|_| bf16::from_f64(r.r(0.5, 1.5))).collect();
    Inputs {
        deinterleaved,
        conv_state0,
        conv_weight,
        h0,
        gates,
        norm_weight,
    }
}

/// 2026-09-25: Stage-0 captures: the per-token gated-norm outputs (K x VALUE_DIM, read as
/// BF16), the state after token 1 and after token 0, and the conv state after token 1
/// and after token 0.
struct Golden {
    norm_out: Vec<f32>,
    h_committed: Vec<f32>,
    h_inter: Vec<f32>,
    conv_committed: Vec<f32>,
    conv_inter: Vec<f32>,
}

/// 2026-09-25: `causal_conv1d_update_l2norm` for one token, batch 1, no bias.
fn launch_conv1d(
    g: &dyn GpuBackend,
    k: KernelHandle,
    conv_state: DevicePtr,
    input: DevicePtr,
    weight: DevicePtr,
    output: DevicePtr,
) -> Result<()> {
    let bias = DevicePtr::NULL;
    KernelLaunch::new(g, k)
        .grid([CONV_DIM as u32 / 256, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(conv_state)
        .arg_ptr(input)
        .arg_ptr(weight)
        .arg_ptr(bias)
        .arg_ptr(output)
        .arg_u32(1)
        .arg_u32(CONV_DIM as u32)
        .arg_u32(D_CONV as u32)
        .arg_u32(QK_CH as u32)
        .arg_u32(KD as u32)
        .arg_f32(L2_EPS)
        .launch(0)
}

/// 2026-09-25: `gated_delta_rule_wy2`, the K=2 recurrence: the state after token 0 goes to
/// `h_inter`, after token 1 to `h_state`.
#[allow(clippy::too_many_arguments)]
fn launch_wy2(
    g: &dyn GpuBackend,
    k: KernelHandle,
    h_state: DevicePtr,
    q: DevicePtr,
    key: DevicePtr,
    val: DevicePtr,
    gate: DevicePtr,
    beta: DevicePtr,
    out: DevicePtr,
    h_inter: DevicePtr,
) -> Result<()> {
    KernelLaunch::new(g, k)
        .grid([NV as u32, 1, 1])
        .block([128, 1, 1])
        .arg_ptr(h_state)
        .arg_ptr(q)
        .arg_ptr(key)
        .arg_ptr(val)
        .arg_ptr(gate)
        .arg_ptr(beta)
        .arg_ptr(out)
        .arg_ptr(h_inter)
        .arg_u32(1)
        .arg_u32(NK as u32)
        .arg_u32(NV as u32)
        .arg_u32(KD as u32)
        .arg_u32(VD as u32)
        .arg_u32(CONV_DIM as u32)
        .arg_u32(CONV_DIM as u32)
        .arg_u32((NV * 2) as u32)
        .arg_u32(0)
        .launch(0)
}

/// 2026-09-25: `gated_rms_norm` for one token: one block per value head, normalizing VD
/// elements.
fn launch_norm(
    g: &dyn GpuBackend,
    k: KernelHandle,
    input: DevicePtr,
    gate: DevicePtr,
    weight: DevicePtr,
    output: DevicePtr,
) -> Result<()> {
    KernelLaunch::new(g, k)
        .grid([NV as u32, 1, 1])
        .block([VD as u32, 1, 1])
        .arg_ptr(input)
        .arg_ptr(gate)
        .arg_ptr(weight)
        .arg_ptr(output)
        .arg_u32(VD as u32)
        .arg_f32(RMS_EPS)
        .arg_u32(VD as u32)
        .arg_u32(VD as u32)
        .launch(0)
}

/// 2026-09-25: Stage 0, in order:
///   conv(t0), then copy the conv state to conv_inter;
///   conv(t1), then copy the conv state to conv_committed;
///   wy2 (h_inter after t0, h_state after t1);
///   norm(t0), norm(t1).
fn run_golden(g: &dyn GpuBackend, ins: &Inputs) -> Result<Golden> {
    let conv_k = g.kernel("causal_conv1d", "causal_conv1d_update_l2norm")?;
    let wy2_k = g.kernel("gated_delta_rule_wy", "gated_delta_rule_wy2")?;
    let norm_k = g.kernel("norm", "gated_rms_norm")?;

    let conv_state = up_f32(g, &ins.conv_state0)?;
    let conv_weight = up_bf16(g, &ins.conv_weight)?;
    let deint = up_bf16(g, &ins.deinterleaved)?;
    let h_state = up_f32(g, &ins.h0)?;
    let h_inter = g.alloc(NV * KD * VD * 4)?;
    let gates = up_f32(g, &ins.gates)?;
    let norm_w = up_bf16(g, &ins.norm_weight)?;

    let conv_out = g.alloc(2 * CONV_DIM * 2)?;
    let gdn_out = g.alloc(2 * VALUE_DIM * 2)?;
    let norm_out = g.alloc(2 * VALUE_DIM * 2)?;
    let conv_inter = g.alloc(CONV_DIM * D_CONV * 4)?;
    let conv_committed = g.alloc(CONV_DIM * D_CONV * 4)?;

    launch_conv1d(g, conv_k, conv_state, deint, conv_weight, conv_out)?;
    g.copy_d2d_async(conv_state, conv_inter, CONV_DIM * D_CONV * 4, 0)?;
    let deint_1 = deint.offset(QKVZ_SIZE * 2);
    let conv_out_1 = conv_out.offset(CONV_DIM * 2);
    launch_conv1d(g, conv_k, conv_state, deint_1, conv_weight, conv_out_1)?;
    g.copy_d2d_async(conv_state, conv_committed, CONV_DIM * D_CONV * 4, 0)?;

    let q_ptr = conv_out;
    let k_ptr = conv_out.offset(KEY_DIM * 2);
    let v_ptr = conv_out.offset(KEY_DIM * 2 * 2);
    let gate_ptr = gates;
    let beta_ptr = gates.offset(NV * 4);
    launch_wy2(
        g, wy2_k, h_state, q_ptr, k_ptr, v_ptr, gate_ptr, beta_ptr, gdn_out, h_inter,
    )?;

    // 2026-09-25: Z starts CONV_DIM elements into each deinterleaved row.
    for t in 0..2usize {
        let gdn_t = gdn_out.offset(t * VALUE_DIM * 2);
        let z_t = deint.offset(t * QKVZ_SIZE * 2 + CONV_DIM * 2);
        let norm_t = norm_out.offset(t * VALUE_DIM * 2);
        launch_norm(g, norm_k, gdn_t, z_t, norm_w, norm_t)?;
    }
    g.synchronize(0)?;

    let golden = Golden {
        norm_out: dn_bf16(g, norm_out, 2 * VALUE_DIM)?,
        h_committed: dn_f32(g, h_state, NV * KD * VD)?,
        h_inter: dn_f32(g, h_inter, NV * KD * VD)?,
        conv_committed: dn_f32(g, conv_committed, CONV_DIM * D_CONV)?,
        conv_inter: dn_f32(g, conv_inter, CONV_DIM * D_CONV)?,
    };

    for p in [
        conv_state,
        conv_weight,
        deint,
        h_state,
        h_inter,
        gates,
        norm_w,
        conv_out,
        gdn_out,
        norm_out,
        conv_inter,
        conv_committed,
    ] {
        let _ = g.free(p);
    }
    Ok(golden)
}

fn main() -> Result<()> {
    let g0 = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &g0;

    let mut all_ok = true;
    for layer in 0..6u64 {
        let ins = gen_inputs(0xF02E_D000 ^ layer);

        // 2026-09-25: Stage 0 twice on the same seed; every capture must match bytewise.
        let a = run_golden(g, &ins)?;
        let b = run_golden(g, &ins)?;
        let repro = a.norm_out == b.norm_out
            && a.h_committed == b.h_committed
            && a.h_inter == b.h_inter
            && a.conv_committed == b.conv_committed
            && a.conv_inter == b.conv_inter;
        all_ok &= repro;
        eprintln!(
            "STAGE0 layer={layer}  golden reproducible={}  \
             norm_out[0..4]={:?}  h_committed[0]={:.6}  conv_inter[0]={:.6}",
            if repro { "YES" } else { "NO" },
            &a.norm_out[0..4],
            a.h_committed[0],
            a.conv_inter[0],
        );

        let fused = run_fused_stage1(g, &ins)?;
        let mut min_out = 1.0f64;
        for t in 0..2 {
            let fa = &fused.norm_out[t * VALUE_DIM..(t + 1) * VALUE_DIM];
            let ga = &a.norm_out[t * VALUE_DIM..(t + 1) * VALUE_DIM];
            min_out = min_out.min(cos(fa, ga));
        }
        let conv_committed_cos = cos(&fused.conv_committed, &a.conv_committed);
        let conv_inter_cos = cos(&fused.conv_inter, &a.conv_inter);
        let state_cos = conv_committed_cos.min(conv_inter_cos);
        let ok = min_out >= PASS_COS && state_cos >= PASS_COS;
        all_ok &= ok;
        eprintln!(
            "STAGE1 layer={layer}  norm_out_cos={min_out:.7} \
             conv_committed_cos={conv_committed_cos:.7} \
             conv_inter_cos={conv_inter_cos:.7}  {}",
            if ok { "PASS" } else { "FAIL" }
        );
    }

    eprintln!(
        "\nFused-verify GATE (STAGE0 repro + STAGE1 cos≥{PASS_COS}): {}",
        if all_ok { "PASS" } else { "FAIL" }
    );
    if !all_ok {
        std::process::exit(1);
    }
    Ok(())
}

/// 2026-09-25: Stage-1 captures: the gated-norm outputs (K x VALUE_DIM), and the conv
/// state after token 1 and after token 0.
struct FusedStage1 {
    norm_out: Vec<f32>,
    conv_committed: Vec<f32>,
    conv_inter: Vec<f32>,
}

/// 2026-09-25: Stage 1: `gdn_verify_fused_conv_k2` for both positions (the window after
/// position 0 goes to conv_inter, after position 1 stays in conv_state), then
/// `gated_delta_rule_wy2`, then `gdn_verify_fused_norm_k2` for both positions.
fn run_fused_stage1(g: &dyn GpuBackend, ins: &Inputs) -> Result<FusedStage1> {
    let conv_k = g.kernel("gdn_verify_fused_k2", "gdn_verify_fused_conv_k2")?;
    let norm_k = g.kernel("gdn_verify_fused_k2", "gdn_verify_fused_norm_k2")?;
    let wy2_k = g.kernel("gated_delta_rule_wy", "gated_delta_rule_wy2")?;

    let conv_state = up_f32(g, &ins.conv_state0)?;
    let conv_weight = up_bf16(g, &ins.conv_weight)?;
    let deint = up_bf16(g, &ins.deinterleaved)?;
    let h_state = up_f32(g, &ins.h0)?;
    let h_inter = g.alloc(NV * KD * VD * 4)?;
    let gates = up_f32(g, &ins.gates)?;
    let norm_w = up_bf16(g, &ins.norm_weight)?;

    let conv_out = g.alloc(2 * CONV_DIM * 2)?;
    let gdn_out = g.alloc(2 * VALUE_DIM * 2)?;
    let norm_out = g.alloc(2 * VALUE_DIM * 2)?;
    let conv_inter = g.alloc(CONV_DIM * D_CONV * 4)?;

    KernelLaunch::new(g, conv_k)
        .grid([CONV_DIM as u32 / 256, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(conv_state)
        .arg_ptr(deint)
        .arg_ptr(conv_weight)
        .arg_ptr(conv_out)
        .arg_ptr(conv_inter)
        .arg_u32(CONV_DIM as u32)
        .arg_u32(D_CONV as u32)
        .arg_u32(QK_CH as u32)
        .arg_u32(KD as u32)
        .arg_u32(QKVZ_SIZE as u32)
        .arg_u32(CONV_DIM as u32)
        .arg_f32(L2_EPS)
        .launch(0)?;

    let q_ptr = conv_out;
    let k_ptr = conv_out.offset(KEY_DIM * 2);
    let v_ptr = conv_out.offset(KEY_DIM * 2 * 2);
    launch_wy2(
        g,
        wy2_k,
        h_state,
        q_ptr,
        k_ptr,
        v_ptr,
        gates,
        gates.offset(NV * 4),
        gdn_out,
        h_inter,
    )?;

    KernelLaunch::new(g, norm_k)
        .grid([NV as u32, 2, 1])
        .block([VD as u32, 1, 1])
        .arg_ptr(gdn_out)
        .arg_ptr(deint)
        .arg_ptr(norm_w)
        .arg_ptr(norm_out)
        .arg_u32(VD as u32)
        .arg_f32(RMS_EPS)
        .arg_u32(QKVZ_SIZE as u32)
        .arg_u32(CONV_DIM as u32)
        .arg_u32(VALUE_DIM as u32)
        .launch(0)?;

    g.synchronize(0)?;

    let out = FusedStage1 {
        norm_out: dn_bf16(g, norm_out, 2 * VALUE_DIM)?,
        conv_committed: dn_f32(g, conv_state, CONV_DIM * D_CONV)?,
        conv_inter: dn_f32(g, conv_inter, CONV_DIM * D_CONV)?,
    };

    for p in [
        conv_state,
        conv_weight,
        deint,
        h_state,
        h_inter,
        gates,
        norm_w,
        conv_out,
        gdn_out,
        norm_out,
        conv_inter,
    ] {
        let _ = g.free(p);
    }
    Ok(out)
}
