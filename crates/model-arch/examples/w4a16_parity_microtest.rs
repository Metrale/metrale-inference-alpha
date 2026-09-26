// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Parity gate for the transposed-B W4A16 (NVFP4) GEMMs at M = 17.
//!
//! On the same random NVFP4 data, each of `w4a16_gemm_t`, `w4a16_gemm_t_k64`,
//! `w4a16_gemm_t_m128` and `w4a16_gemm_t_k64_p3` that the target has (absent
//! ones are skipped) is compared with the base `w4a16_gemm`:
//!
//!   C_base = A · dequant(B)        (`w4a16_gemm`, B as [N, K/2])
//!   C_x    = A · dequant(B_t)      (the kernel under test, B_t as [K/2, N])
//!
//! B_t is the byte transpose `QuantizedWeight::transpose_for_gemm` performs. A
//! kernel passes at cosine >= `PASS_COS`; max |delta| is printed, not gated.
//!
//! Owner: model-arch examples.
//! Invariants: none beyond the types.
//!
//!   cargo run -p metrale-model-arch --release --example w4a16_parity_microtest \
//!       --features cuda,gpu-examples
//!
//! Exit 0 = every tested kernel passes; 1 = at least one fails (named).

use anyhow::Result;
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};

const M: usize = 17;
const GROUP: usize = 16;
const PASS_COS: f64 = 0.999;

/// 2026-09-25: `(label, N, K)`.
const SHAPES: &[(&str, usize, usize)] = &[
    ("ffn_gate/up N=17408 K=5120 ", 17408, 5120),
    ("ffn_down    N=5120  K=17408", 5120, 17408),
    ("attn_q      N=12288 K=5120 ", 12288, 5120),
    ("attn_o      N=5120  K=6144 ", 5120, 6144),
];

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn byte(&mut self) -> u8 {
        (self.next() >> 32) as u8
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
}

fn up(g: &dyn GpuBackend, b: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(b.len().max(1))?;
    g.copy_h2d(b, p)?;
    Ok(p)
}
fn dn_bf16(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<f32>> {
    let mut b = vec![0u8; n * 2];
    g.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(2)
        .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
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

#[allow(clippy::too_many_arguments)]
fn launch(
    g: &dyn GpuBackend,
    kh: KernelHandle,
    grid: [u32; 3],
    a: DevicePtr,
    b: DevicePtr,
    bs: DevicePtr,
    c: DevicePtr,
    n: usize,
    k: usize,
) -> Result<()> {
    KernelLaunch::new(g, kh)
        .grid(grid)
        .block([128, 1, 1])
        .arg_ptr(a)
        .arg_ptr(b)
        .arg_ptr(bs)
        .arg_f32(0.01)
        .arg_ptr(c)
        .arg_u32(M as u32)
        .arg_u32(n as u32)
        .arg_u32(k as u32)
        // 2026-09-25: `ldb`, the transposed-B row stride (`n` when packed), for the
        // kernels that declare a ninth argument; the others do not read it.
        .arg_u32(n as u32)
        .launch(0)
}

fn main() -> Result<()> {
    let g0 = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &g0;

    let base_k = g.kernel("w4a16", "w4a16_gemm")?;
    let t_kernels: Vec<(&str, KernelHandle)> = [
        ("w4a16_gemm_t     ", "w4a16_gemm_t"),
        ("w4a16_gemm_t_k64 ", "w4a16_gemm_t_k64"),
        ("w4a16_gemm_t_m128", "w4a16_gemm_t_m128"),
        ("w4a16_gemm_t_k64_p3", "w4a16_gemm_t_k64_p3"),
    ]
    .into_iter()
    .filter_map(|(name, f)| g.kernel("w4a16", f).ok().map(|h| (name, h)))
    .collect();

    let mut all_ok = true;
    for &(label, n, k) in SHAPES {
        let mut r = Rng(0x517A_C0DE ^ (n as u64) << 20 ^ k as u64);
        let half_k = k / 2;
        let groups = k / GROUP;

        // 2026-09-25: A `[M, K]` BF16: uniform draws in [-0.25, 0.25), rounded to BF16.
        let a_host: Vec<u8> = (0..M * k)
            .flat_map(|_| {
                bf16::from_f32((r.unit() - 0.5) * 0.5)
                    .to_bits()
                    .to_le_bytes()
            })
            .collect();
        // 2026-09-25: B packed E2M1 `[N, K/2]`; every nibble is a valid E2M1 value.
        let b_host: Vec<u8> = (0..n * half_k).map(|_| r.byte()).collect();
        // 2026-09-25: B scales E4M3 `[N, K/16]`, bytes 0x28..=0x47: finite
        // positive values 0.25 to 3.75.
        let bs_host: Vec<u8> = (0..n * groups).map(|_| 0x28 + (r.byte() & 0x1F)).collect();

        let mut bt_host = vec![0u8; n * half_k];
        for i in 0..n {
            for j in 0..half_k {
                bt_host[j * n + i] = b_host[i * half_k + j];
            }
        }
        let mut bst_host = vec![0u8; n * groups];
        for i in 0..n {
            for j in 0..groups {
                bst_host[j * n + i] = bs_host[i * groups + j];
            }
        }

        let a = up(g, &a_host)?;
        let b = up(g, &b_host)?;
        let bs = up(g, &bs_host)?;
        let bt = up(g, &bt_host)?;
        let bst = up(g, &bst_host)?;
        let c_base = g.alloc(M * n * 2)?;
        let c_test = g.alloc(M * n * 2)?;

        launch(
            g,
            base_k,
            [div_ceil(n as u32, 64), div_ceil(M as u32, 64), 1],
            a,
            b,
            bs,
            c_base,
            n,
            k,
        )?;
        g.synchronize(0)?;
        let base_out = dn_bf16(g, c_base, M * n)?;

        for &(name, kh) in &t_kernels {
            let grid = if name.trim_end() == "w4a16_gemm_t_m128" {
                [div_ceil(n as u32, 128), div_ceil(M as u32, 128), 1]
            } else {
                [div_ceil(n as u32, 128), div_ceil(M as u32, 64), 1]
            };
            g.memset(c_test, 0, M * n * 2)?;
            launch(g, kh, grid, a, bt, bst, c_test, n, k)?;
            g.synchronize(0)?;
            let out = dn_bf16(g, c_test, M * n)?;

            let c = cos(&out, &base_out);
            let max_abs_base = base_out.iter().fold(0f32, |m, v| m.max(v.abs()));
            let max_d = out
                .iter()
                .zip(&base_out)
                .fold(0f32, |m, (x, y)| m.max((x - y).abs()));
            let ok = c >= PASS_COS;
            all_ok &= ok;
            eprintln!(
                "{label}  {name}  cos={c:.7}  max|Δ|={max_d:.5} (base max|C|={max_abs_base:.3})  {}",
                if ok {
                    "PASS"
                } else {
                    "FAIL ← kernel disagrees with base"
                }
            );
        }
        eprintln!();
        for p in [a, b, bs, bt, bst, c_base, c_test] {
            let _ = g.free(p);
        }
    }

    eprintln!(
        "W4A16 parity GATE (all transposed kernels vs base, cos≥{PASS_COS}): {}",
        if all_ok { "PASS" } else { "FAIL" }
    );
    if !all_ok {
        std::process::exit(1);
    }
    Ok(())
}
