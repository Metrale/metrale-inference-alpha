// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Byte-parity check of the `fp8_gemm_t_row_scaled_k64` and `_p4` twins
//! against `fp8_gemm_t_row_scaled`.
//!
//! Owner: model-arch examples.
//! Invariants:
//! - Exit 0 only when every present twin is byte-identical to the base at every shape, M
//!   and seed and the negative control fires; exit 1 otherwise; exit 2 when the base
//!   kernel or both twins are absent from this target's module set.
//!
//! `dflash_head/from_weights.rs` picks the DFlash drafter's FP8 GEMM in the order `_k64`,
//! `_p4`, base; `METRALE_DFLASH_FP8_GEMM_P4=1` skips `_k64` and `METRALE_DFLASH_FP8_GEMM_P2=1`
//! pins the base. All three take `(A_bf16, B_fp8, row_scale_f32, C_bf16, M, N, K)` on the
//! grid `ops::fp8_gemm_n128_row_scaled` launches: (ceil(N/128), ceil(M/64), 1) x (128, 1, 1).
//!
//! Run (GPU host):
//!   METRALE_TARGET_HW=gb10 METRALE_TARGET_MODEL=qwen3.6-27b METRALE_TARGET_QUANT=nvfp4 \
//!   cargo run -p metrale-model-arch --release --features cuda,gpu-examples \
//!     --example fp8_gemm_twin_parity

use anyhow::Result;
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};

/// 2026-09-25: Distinct (N, K) projection shapes at Qwen3.6-27B dimensions: hidden 5120,
/// 24 q heads x 256 doubled by the output gate, 4 kv heads x 256 (k and v share a shape),
/// FFN 17408. Every K is a multiple of 64.
const SHAPES: [(&str, usize, usize); 5] = [
    ("drafter q_proj   [12288 x  5120]", 12288, 5120),
    ("drafter k/v_proj [ 1024 x  5120]", 1024, 5120),
    ("drafter o_proj   [ 5120 x  6144]", 5120, 6144),
    ("drafter ffn_g/u  [17408 x  5120]", 17408, 5120),
    ("drafter ffn_down [ 5120 x 17408]", 5120, 17408),
];

/// 2026-09-25: With 64-row M tiles (grid.y = ceil(M/64)): a partial tile, an exact tile,
/// and a full tile plus a partial second one.
const MS: [usize; 3] = [16, 64, 96];
const MAX_M: usize = 96;

struct Lcg(u64);
impl Lcg {
    fn f(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (((self.0 >> 11) as f64) / ((1u64 << 53) as f64)) as f32
    }
    fn r(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.f()
    }
    /// 2026-09-25: A random E4M3 weight byte, with the two NaN codes (0x7F, 0xFF)
    /// replaced by 0x00 so every input is a finite value.
    fn e4m3(&mut self) -> u8 {
        self.f();
        let b = (self.0 >> 24) as u8;
        if b & 0x7F == 0x7F { 0x00 } else { b }
    }
}

fn up(g: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(bytes.len().max(1))?;
    g.copy_h2d(bytes, p)?;
    Ok(p)
}

fn down(g: &dyn GpuBackend, p: DevicePtr, n_bytes: usize) -> Result<Vec<u8>> {
    let mut b = vec![0u8; n_bytes];
    g.copy_d2h(p, &mut b)?;
    Ok(b)
}

/// 2026-09-25: Differing element count, worst |delta| and worst relative delta over
/// BF16 pairs.
fn worst_delta(a: &[u8], b: &[u8]) -> (usize, f32, f32) {
    let (mut n_diff, mut worst, mut rel) = (0usize, 0f32, 0f32);
    for (x, y) in a.chunks_exact(2).zip(b.chunks_exact(2)) {
        if x != y {
            n_diff += 1;
            let fx = bf16::from_bits(u16::from_le_bytes([x[0], x[1]])).to_f32();
            let fy = bf16::from_bits(u16::from_le_bytes([y[0], y[1]])).to_f32();
            worst = worst.max((fx - fy).abs());
            let denom = fx.abs().max(fy.abs()).max(1e-6);
            rel = rel.max((fx - fy).abs() / denom);
        }
    }
    (n_diff, worst, rel)
}

/// 2026-09-25: One launch with the arguments and geometry of
/// `ops::fp8_gemm_n128_row_scaled`.
#[allow(clippy::too_many_arguments)]
fn rs_gemm(
    g: &dyn GpuBackend,
    kh: KernelHandle,
    a: DevicePtr,
    b: DevicePtr,
    scale: DevicePtr,
    c: DevicePtr,
    m: u32,
    n: u32,
    k: u32,
) -> Result<()> {
    KernelLaunch::new(g, kh)
        .grid([div_ceil(n, 128), div_ceil(m, 64), 1])
        .block([128, 1, 1])
        .arg_ptr(a)
        .arg_ptr(b)
        .arg_ptr(scale)
        .arg_ptr(c)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(k)
        .launch(0)
}

fn main() -> Result<()> {
    let backend = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;

    let Ok(base_k) = g.kernel("w4a16", "fp8_gemm_t_row_scaled") else {
        println!("fp8_gemm_t_row_scaled absent from this target set — SKIP");
        std::process::exit(2);
    };
    // 2026-09-25: A twin missing from this target's module set is reported and skipped.
    let twins: Vec<(&str, KernelHandle)> =
        ["fp8_gemm_t_row_scaled_k64", "fp8_gemm_t_row_scaled_p4"]
            .into_iter()
            .filter_map(|name| match g.kernel("w4a16", name) {
                Ok(kh) => Some((name, kh)),
                Err(_) => {
                    println!("{name} absent from this target set — twin not graded");
                    None
                }
            })
            .collect();
    if twins.is_empty() {
        println!("no row-scaled FP8 twin present — nothing to grade — SKIP");
        std::process::exit(2);
    }

    let mut twins_clean = true;
    let mut control_ok = true;
    for seed in [1u64, 99, 12345] {
        for (label, n, k) in SHAPES {
            let mut rng = Lcg(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xF8F8);
            let a_bytes: Vec<u8> = (0..MAX_M * k)
                .flat_map(|_| bf16::from_f32(rng.r(-1.5, 1.5)).to_bits().to_le_bytes())
                .collect();
            let b_bytes: Vec<u8> = (0..n * k).map(|_| rng.e4m3()).collect();
            let scale_bytes: Vec<u8> = (0..n).flat_map(|_| rng.r(0.5, 2.0).to_le_bytes()).collect();

            let a_d = up(g, &a_bytes)?;
            let b_d = up(g, &b_bytes)?;
            let s_d = up(g, &scale_bytes)?;
            let c_base = g.alloc(MAX_M * n * 2)?;
            let c_twin = g.alloc(MAX_M * n * 2)?;

            for m in MS {
                g.memset(c_base, 0, MAX_M * n * 2)?;
                rs_gemm(
                    g, base_k, a_d, b_d, s_d, c_base, m as u32, n as u32, k as u32,
                )?;
                g.synchronize(0)?;
                let cb = down(g, c_base, m * n * 2)?;

                for &(name, kh) in &twins {
                    g.memset(c_twin, 0, MAX_M * n * 2)?;
                    rs_gemm(g, kh, a_d, b_d, s_d, c_twin, m as u32, n as u32, k as u32)?;
                    g.synchronize(0)?;
                    let ct = down(g, c_twin, m * n * 2)?;
                    let identical = ct == cb;
                    let (n_diff, worst, rel) = worst_delta(&ct, &cb);
                    twins_clean &= identical;
                    let pct = 100.0 * n_diff as f32 / (m * n) as f32;
                    println!(
                        "seed {seed:>5}  {label}  {name:<26} M={m:<3} byte-identical={identical:<5} \
                         diff_elems={n_diff:<7} ({pct:5.2}%) max|delta|={worst:.6} max_rel={rel:.6}"
                    );
                }
            }

            // 2026-09-25: Negative control: the base kernel on a perturbed activation
            // against the first graded twin on the original must differ, so a
            // byte-identical verdict cannot come from comparing blank buffers.
            let m = MS[0];
            let mut pert = a_bytes.clone();
            // 2026-09-25: Flip bit 0 of the high byte of one BF16 activation (row 1, column
            // 7). That is bit 8, the second exponent bit, so the value is scaled by 4 or 1/4
            // and the change survives accumulation over K and rounding to BF16.
            pert[2 * (k + 7) + 1] ^= 1;
            let a_pert = up(g, &pert)?;
            g.memset(c_base, 0, MAX_M * n * 2)?;
            g.memset(c_twin, 0, MAX_M * n * 2)?;
            rs_gemm(
                g, base_k, a_pert, b_d, s_d, c_base, m as u32, n as u32, k as u32,
            )?;
            rs_gemm(
                g, twins[0].1, a_d, b_d, s_d, c_twin, m as u32, n as u32, k as u32,
            )?;
            g.synchronize(0)?;
            let differs = down(g, c_base, m * n * 2)? != down(g, c_twin, m * n * 2)?;
            control_ok &= differs;
            println!("seed {seed:>5}  {label}  CONTROL 1-ULP perturbation detected={differs}");
            g.free(a_pert).ok();

            for p in [a_d, b_d, s_d, c_base, c_twin] {
                g.free(p).ok();
            }
        }
    }

    if !control_ok {
        println!("FAIL — negative control did not mismatch; this harness is VACUOUS.");
        std::process::exit(1);
    }
    if twins_clean {
        println!(
            "PASS — every present fp8_gemm_t_row_scaled twin is byte-identical to the \
             original at every drafter shape, M shape, and seed."
        );
        Ok(())
    } else {
        println!(
            "FAIL — a fp8_gemm_t_row_scaled twin is NOT byte-identical to the original. \
             The selector in `dflash_head/from_weights.rs` picks twins on exactly that \
             claim (and METRALE_DFLASH_FP8_GEMM_P2/P4 A/Bs rely on it); do NOT loosen this \
             comparison — fix the twin or demote it in the preference order."
        );
        std::process::exit(1);
    }
}
