// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Bit-exactness gate for `causal_conv1d_update_l2norm_f32_strided`
//! against one `causal_conv1d_update_l2norm_f32` launch per sequence.
//!
//! The input rows are `QKVZ_SIZE` apart. The plain kernel reads sequence `b` at
//! `b * dim`, so the reference passes it pre-offset pointers at batch 1; the
//! strided kernel takes the input and output row strides and runs all `N`
//! sequences in one launch. For each of three seeds the conv outputs and the
//! committed conv state must be byte-identical; the cosine is printed only. The
//! negative control, the plain kernel at batch `N`, must give a different output.
//!
//! Owner: model-arch examples.
//! Invariants: none beyond the types.
//!
//!   cargo run -p metrale-model-arch --release --example conv1d_strided_microtest \
//!       --features cuda,gpu-examples
use anyhow::Result;
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

const KD: usize = 128;
const NK: usize = 16;
const NV: usize = 32;
const VD: usize = 128;
const D_CONV: usize = 4;

const KEY_DIM: usize = NK * KD;
const VALUE_DIM: usize = NV * VD;
const CONV_DIM: usize = KEY_DIM * 2 + VALUE_DIM; // 2026-09-25: q | k | v channels.
const QK_CH: usize = KEY_DIM * 2; // 2026-09-25: the q and k channels, which get the L2 norm.
const QKVZ_SIZE: usize = CONV_DIM + VALUE_DIM; // 2026-09-25: one q | k | v | z input row.

const N: usize = 5;
const L2_EPS: f32 = 1e-6;

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
    /// 2026-09-25: `N` sequences at a row stride of `QKVZ_SIZE`, so random data
    /// sits between one sequence's conv channels and the next's.
    deinterleaved: Vec<bf16>,
    conv_state0: Vec<f32>,
    conv_weight: Vec<bf16>,
}

fn gen_inputs(seed: u64) -> Inputs {
    let mut r = Lcg(seed);
    Inputs {
        deinterleaved: (0..N * QKVZ_SIZE)
            .map(|_| bf16::from_f64(r.r(-0.5, 0.5)))
            .collect(),
        conv_state0: (0..N * CONV_DIM * D_CONV)
            .map(|_| r.r(-0.3, 0.3) as f32)
            .collect(),
        conv_weight: (0..CONV_DIM * D_CONV)
            .map(|_| bf16::from_f64(r.r(-0.3, 0.3)))
            .collect(),
    }
}

struct Captured {
    conv_out: Vec<u8>,
    conv_committed: Vec<u8>,
}

/// 2026-09-25: One launch of the plain kernel: at batch 1 with pre-offset
/// pointers for the reference loop, at batch `N` for the negative control.
#[allow(clippy::too_many_arguments)]
fn launch_plain(
    g: &dyn GpuBackend,
    k: KernelHandle,
    conv_state: DevicePtr,
    input: DevicePtr,
    weight: DevicePtr,
    out: DevicePtr,
    batch: u32,
) -> Result<()> {
    KernelLaunch::new(g, k)
        .grid([CONV_DIM.div_ceil(256) as u32, batch, 1])
        .block([256, 1, 1])
        .arg_ptr(conv_state)
        .arg_ptr(input)
        .arg_ptr(weight)
        .arg_ptr(DevicePtr::NULL)
        .arg_ptr(out)
        .arg_u32(batch)
        .arg_u32(CONV_DIM as u32)
        .arg_u32(D_CONV as u32)
        .arg_u32(QK_CH as u32)
        .arg_u32(KD as u32)
        .arg_f32(L2_EPS)
        .launch(0)
}

fn run_golden(g: &dyn GpuBackend, k: KernelHandle, inp: &Inputs) -> Result<Captured> {
    let state = up_f32(g, &inp.conv_state0)?;
    let input = up_bf16(g, &inp.deinterleaved)?;
    let weight = up_bf16(g, &inp.conv_weight)?;
    let out = g.alloc(N * CONV_DIM * 4)?;
    for i in 0..N {
        launch_plain(
            g,
            k,
            state.offset(i * CONV_DIM * D_CONV * 4),
            input.offset(i * QKVZ_SIZE * 2),
            weight,
            out.offset(i * CONV_DIM * 4),
            1,
        )?;
    }
    g.synchronize(0)?;
    Ok(Captured {
        conv_out: dn_bytes(g, out, N * CONV_DIM * 4)?,
        conv_committed: dn_bytes(g, state, N * CONV_DIM * D_CONV * 4)?,
    })
}

fn run_strided(g: &dyn GpuBackend, k: KernelHandle, inp: &Inputs) -> Result<Captured> {
    let state = up_f32(g, &inp.conv_state0)?;
    let input = up_bf16(g, &inp.deinterleaved)?;
    let weight = up_bf16(g, &inp.conv_weight)?;
    let out = g.alloc(N * CONV_DIM * 4)?;
    KernelLaunch::new(g, k)
        .grid([CONV_DIM.div_ceil(256) as u32, N as u32, 1])
        .block([256, 1, 1])
        .arg_ptr(state)
        .arg_ptr(input)
        .arg_ptr(weight)
        .arg_ptr(DevicePtr::NULL)
        .arg_ptr(out)
        .arg_u32(N as u32)
        .arg_u32(CONV_DIM as u32)
        .arg_u32(D_CONV as u32)
        .arg_u32(QK_CH as u32)
        .arg_u32(KD as u32)
        .arg_f32(L2_EPS)
        .arg_u32(QKVZ_SIZE as u32)
        .arg_u32(CONV_DIM as u32)
        .launch(0)?;
    g.synchronize(0)?;
    Ok(Captured {
        conv_out: dn_bytes(g, out, N * CONV_DIM * 4)?,
        conv_committed: dn_bytes(g, state, N * CONV_DIM * D_CONV * 4)?,
    })
}

/// 2026-09-25: Negative control: the plain kernel at batch `N` reads sequence `b`
/// at `b * CONV_DIM` instead of `b * QKVZ_SIZE`, so its output must differ from
/// the reference.
fn run_plain_batched(g: &dyn GpuBackend, k: KernelHandle, inp: &Inputs) -> Result<Captured> {
    let state = up_f32(g, &inp.conv_state0)?;
    let input = up_bf16(g, &inp.deinterleaved)?;
    let weight = up_bf16(g, &inp.conv_weight)?;
    let out = g.alloc(N * CONV_DIM * 4)?;
    launch_plain(g, k, state, input, weight, out, N as u32)?;
    g.synchronize(0)?;
    Ok(Captured {
        conv_out: dn_bytes(g, out, N * CONV_DIM * 4)?,
        conv_committed: dn_bytes(g, state, N * CONV_DIM * D_CONV * 4)?,
    })
}

fn main() -> Result<()> {
    let gpu = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &gpu;

    let k_plain = g.kernel("causal_conv1d", "causal_conv1d_update_l2norm_f32")?;
    let k_strided = g.kernel("causal_conv1d", "causal_conv1d_update_l2norm_f32_strided")?;

    let mut all_ok = true;
    for seed in [1u64, 99, 12345] {
        let inp = gen_inputs(seed);
        let golden = run_golden(g, k_plain, &inp)?;
        let strided = run_strided(g, k_strided, &inp)?;
        let bad = run_plain_batched(g, k_plain, &inp)?;

        let out_id = golden.conv_out == strided.conv_out;
        let st_id = golden.conv_committed == strided.conv_committed;
        let c_out = cos(&as_f32(&golden.conv_out), &as_f32(&strided.conv_out));

        let ctrl_differs = golden.conv_out != bad.conv_out;

        println!(
            "seed {seed:>5}: conv_out identical={out_id}  conv_state identical={st_id}  \
             cos={c_out:.9}  [neg-control differs={ctrl_differs}]"
        );
        if !out_id || !st_id {
            println!("  FAIL: strided output diverges from the per-seq loop");
            all_ok = false;
        }
        if !ctrl_differs {
            println!(
                "  FAIL: plain kernel at batch=N matched golden — the test is NOT \
                 exercising the stride difference, so the positive result is void"
            );
            all_ok = false;
        }
    }

    println!(
        "\n{}",
        if all_ok {
            "PASS — strided batch=N is byte-identical to the per-seq loop, \
             and the unstrided batch=N control is provably wrong."
        } else {
            "FAIL"
        }
    );
    if !all_ok {
        std::process::exit(1);
    }
    Ok(())
}
