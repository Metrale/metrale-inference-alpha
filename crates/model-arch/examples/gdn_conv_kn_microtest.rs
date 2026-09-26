// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Byte-identity of the fused verify conv kernel `gdn_verify_fused_conv_kn`
//! against a per-token loop, for K = 17 tokens.
//!
//! Owner: model-arch examples.
//! Invariants:
//! - Exit 1 unless, for each of 6 seeds, the fused conv outputs, per-token conv-state
//!   snapshots and committed conv state are byte-identical to the loop's and each cosine
//!   is >= `PASS_COS`.
//!
//!   Loop: `causal_conv1d_update_l2norm` once per token, each followed by a copy of the
//!     conv state into that token's snapshot slot.
//!   Fused: one `gdn_verify_fused_conv_kn` launch writing the 17 outputs, the 17 snapshots
//!     and the committed conv state.
//!
//!   cargo run -p metrale-model-arch --release --example gdn_conv_kn_microtest \
//!       --features cuda,gpu-examples
use anyhow::Result;
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

// 2026-09-25: GDN dimensions and conv width of kernels/gb10/qwen3.6-35b-a3b/MODEL.toml.
const KD: usize = 128;
const NK: usize = 16;
const NV: usize = 32;
const VD: usize = 128;
const D_CONV: usize = 4;

const KEY_DIM: usize = NK * KD;
const VALUE_DIM: usize = NV * VD;
const CONV_DIM: usize = KEY_DIM * 2 + VALUE_DIM;
const QK_CH: usize = KEY_DIM * 2;
const QKVZ_SIZE: usize = CONV_DIM + VALUE_DIM;

const K: usize = 17;
const L2_EPS: f32 = 1e-6;
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
fn dn_bytes(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u8>> {
    let mut b = vec![0u8; n];
    g.copy_d2h(p, &mut b)?;
    Ok(b)
}
fn as_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}
fn bf16_as_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(2)
        .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect()
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

struct Inputs {
    deinterleaved: Vec<bf16>,
    conv_state0: Vec<f32>,
    conv_weight: Vec<bf16>,
}

fn gen_inputs(seed: u64) -> Inputs {
    let mut r = Lcg(seed);
    Inputs {
        deinterleaved: (0..K * QKVZ_SIZE)
            .map(|_| bf16::from_f64(r.r(-0.5, 0.5)))
            .collect(),
        conv_state0: (0..CONV_DIM * D_CONV)
            .map(|_| r.r(-0.3, 0.3) as f32)
            .collect(),
        conv_weight: (0..CONV_DIM * D_CONV)
            .map(|_| bf16::from_f64(r.r(-0.3, 0.3)))
            .collect(),
    }
}

/// 2026-09-25: One leg's outputs as raw bytes, compared bytewise: conv_out is K x CONV_DIM
/// BF16, snapshots K x CONV_DIM x D_CONV FP32 (token-major), conv_committed
/// CONV_DIM x D_CONV FP32.
struct Captured {
    conv_out: Vec<u8>,
    snapshots: Vec<u8>,
    conv_committed: Vec<u8>,
}

/// 2026-09-25: One token of `causal_conv1d_update_l2norm`.
fn launch_conv1d(
    g: &dyn GpuBackend,
    k: KernelHandle,
    conv_state: DevicePtr,
    input: DevicePtr,
    weight: DevicePtr,
    output: DevicePtr,
) -> Result<()> {
    KernelLaunch::new(g, k)
        .grid([CONV_DIM as u32 / 256, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(conv_state)
        .arg_ptr(input)
        .arg_ptr(weight)
        .arg_ptr(DevicePtr::NULL)
        .arg_ptr(output)
        .arg_u32(1)
        .arg_u32(CONV_DIM as u32)
        .arg_u32(D_CONV as u32)
        .arg_u32(QK_CH as u32)
        .arg_u32(KD as u32)
        .arg_f32(L2_EPS)
        .launch(0)
}

fn run_golden(g: &dyn GpuBackend, ins: &Inputs) -> Result<Captured> {
    let conv_k = g.kernel("causal_conv1d", "causal_conv1d_update_l2norm")?;

    let conv_state = up_f32(g, &ins.conv_state0)?;
    let conv_weight = up_bf16(g, &ins.conv_weight)?;
    let deint = up_bf16(g, &ins.deinterleaved)?;
    let conv_out = g.alloc(K * CONV_DIM * 2)?;
    let conv_bytes = CONV_DIM * D_CONV * 4;
    let snapshots = g.alloc(K * conv_bytes)?;

    for t in 0..K {
        launch_conv1d(
            g,
            conv_k,
            conv_state,
            deint.offset(t * QKVZ_SIZE * 2),
            conv_weight,
            conv_out.offset(t * CONV_DIM * 2),
        )?;
        g.copy_d2d_async(conv_state, snapshots.offset(t * conv_bytes), conv_bytes, 0)?;
    }
    g.synchronize(0)?;

    let cap = Captured {
        conv_out: dn_bytes(g, conv_out, K * CONV_DIM * 2)?,
        snapshots: dn_bytes(g, snapshots, K * conv_bytes)?,
        conv_committed: dn_bytes(g, conv_state, conv_bytes)?,
    };
    for p in [conv_state, conv_weight, deint, conv_out, snapshots] {
        let _ = g.free(p);
    }
    Ok(cap)
}

fn run_fused(g: &dyn GpuBackend, ins: &Inputs) -> Result<Captured> {
    let conv_k = g.kernel("gdn_verify_fused_conv_kn", "gdn_verify_fused_conv_kn")?;

    let conv_state = up_f32(g, &ins.conv_state0)?;
    let conv_weight = up_bf16(g, &ins.conv_weight)?;
    let deint = up_bf16(g, &ins.deinterleaved)?;
    let conv_out = g.alloc(K * CONV_DIM * 2)?;
    let conv_bytes = CONV_DIM * D_CONV * 4;
    let snapshots = g.alloc(K * conv_bytes)?;

    KernelLaunch::new(g, conv_k)
        .grid([CONV_DIM as u32 / 256, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(conv_state)
        .arg_ptr(deint)
        .arg_ptr(conv_weight)
        .arg_ptr(conv_out)
        .arg_ptr(snapshots)
        .arg_u32(K as u32)
        .arg_u32(CONV_DIM as u32)
        .arg_u32(D_CONV as u32)
        .arg_u32(QK_CH as u32)
        .arg_u32(KD as u32)
        .arg_u32(QKVZ_SIZE as u32)
        .arg_u32(CONV_DIM as u32)
        .arg_u32((conv_bytes / 4) as u32)
        .arg_f32(L2_EPS)
        .launch(0)?;
    g.synchronize(0)?;

    let cap = Captured {
        conv_out: dn_bytes(g, conv_out, K * CONV_DIM * 2)?,
        snapshots: dn_bytes(g, snapshots, K * conv_bytes)?,
        conv_committed: dn_bytes(g, conv_state, conv_bytes)?,
    };
    for p in [conv_state, conv_weight, deint, conv_out, snapshots] {
        let _ = g.free(p);
    }
    Ok(cap)
}

fn main() -> Result<()> {
    let g0 = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &g0;

    let mut all_ok = true;
    for layer in 0..6u64 {
        let ins = gen_inputs(0xC0D4_17AA ^ layer);
        let golden = run_golden(g, &ins)?;
        let fused = run_fused(g, &ins)?;

        let out_eq = golden.conv_out == fused.conv_out;
        let snap_eq = golden.snapshots == fused.snapshots;
        let commit_eq = golden.conv_committed == fused.conv_committed;

        // 2026-09-25: The worst per-token output cosine, so a mismatch names a token.
        let g_out = bf16_as_f32(&golden.conv_out);
        let f_out = bf16_as_f32(&fused.conv_out);
        let mut min_out_cos = 1.0f64;
        for t in 0..K {
            min_out_cos = min_out_cos.min(cos(
                &g_out[t * CONV_DIM..(t + 1) * CONV_DIM],
                &f_out[t * CONV_DIM..(t + 1) * CONV_DIM],
            ));
        }
        let snap_cos = cos(&as_f32(&golden.snapshots), &as_f32(&fused.snapshots));
        let commit_cos = cos(
            &as_f32(&golden.conv_committed),
            &as_f32(&fused.conv_committed),
        );

        let ok = out_eq
            && snap_eq
            && commit_eq
            && min_out_cos >= PASS_COS
            && snap_cos >= PASS_COS
            && commit_cos >= PASS_COS;
        all_ok &= ok;
        eprintln!(
            "layer={layer}  bytes(out/snap/commit)={}/{}/{}  \
             min_out_cos={min_out_cos:.7} snap_cos={snap_cos:.7} \
             commit_cos={commit_cos:.7}  {}",
            out_eq as u8,
            snap_eq as u8,
            commit_eq as u8,
            if ok { "PASS" } else { "FAIL" }
        );
    }

    eprintln!(
        "\nFused K=17 conv GATE (byte-identical to per-token loop): {}",
        if all_ok { "PASS" } else { "FAIL" }
    );
    if !all_ok {
        std::process::exit(1);
    }
    Ok(())
}
