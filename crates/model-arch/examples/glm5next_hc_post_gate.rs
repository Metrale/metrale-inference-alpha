// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Bit-identity gate: `glm5next_hc_post` against its oracle `glm5next_hc_post_ref`.
//!
//! Owner: model-arch examples (GLM-5.3 mHC kernels).
//! Invariants:
//! - The run fails unless every case's `out` is byte-identical between the two kernels, and it
//!   fails if the oracle left `out` unwritten.
//!
//! `glm_hc_post` launches `glm5next_hc_post` (grid `(T, ceil(H/256))`, loops to the
//! compile-time `GLM_HC_MAX_MULT`). `glm5next_hc_post_ref` (grid `(T, 1)`, runtime trip counts)
//! is the oracle. The gate compares bytes, not a tolerance: both kernels execute the same
//! arithmetic in the same order, so any difference is a real reordering.
//!
//!   cargo run -p metrale-model-arch --release --example glm5next_hc_post_gate \
//!       --features cuda,gpu-examples
//!
//! The aliased case writes `out` over the residual buffer, as the serve path writes the highway
//! `streams` in place. That is safe because the thread for column `d` reads only `res[i*H+d]`
//! and writes only `o[j*H+d]`, so columns never cross blocks.

use anyhow::{Result, bail};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use metrale_model_arch::glm5next_mhc::{Glm5NextMhcKernels, glm_hc_post};

/// 2026-09-25: `(hidden, hc_mult, tokens)`. The first two use GLM-5.3's hidden size and
/// `hc_mult`; 1000 is not a multiple of the 256-wide block, and 256 is a single block.
const CASES: [(usize, usize, usize); 5] = [
    (4096, 4, 1),
    (4096, 4, 5),
    (5120, 4, 1),
    (1000, 2, 3),
    (256, 4, 1),
];

fn lcg(seed: &mut u64) -> f32 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*seed >> 33) as f32 / (1u64 << 31) as f32) - 1.0
}

fn up_f32(g: &dyn GpuBackend, d: &[f32]) -> Result<DevicePtr> {
    let b: Vec<u8> = d.iter().flat_map(|x| x.to_le_bytes()).collect();
    let p = g.alloc(b.len().max(1))?;
    g.copy_h2d(&b, p)?;
    Ok(p)
}

/// 2026-09-25: Upload as BF16 by truncating the f32 pattern. Only `block_out` goes through
/// this, and both arms read the same bytes, so the rounding mode does not affect the comparison.
fn up_bf16(g: &dyn GpuBackend, d: &[f32]) -> Result<DevicePtr> {
    let b: Vec<u8> = d
        .iter()
        .flat_map(|x| ((x.to_bits() >> 16) as u16).to_le_bytes())
        .collect();
    let p = g.alloc(b.len().max(1))?;
    g.copy_h2d(&b, p)?;
    Ok(p)
}

fn dn(g: &dyn GpuBackend, p: DevicePtr, bytes: usize) -> Result<Vec<u8>> {
    let mut b = vec![0u8; bytes];
    g.copy_d2h(p, &mut b)?;
    Ok(b)
}

fn poison(g: &dyn GpuBackend, p: DevicePtr, bytes: usize) -> Result<()> {
    g.copy_h2d(&vec![0xABu8; bytes], p)
}

/// 2026-09-25: Mean wall microseconds per launch after 10 warm-up calls, synchronised once at
/// each end rather than per launch, which would time the sync instead of the kernel.
fn time_us(g: &dyn GpuBackend, reps: usize, mut f: impl FnMut() -> Result<()>) -> Result<f64> {
    for _ in 0..10 {
        f()?;
    }
    g.synchronize(0)?;
    let t0 = std::time::Instant::now();
    for _ in 0..reps {
        f()?;
    }
    g.synchronize(0)?;
    Ok(t0.elapsed().as_secs_f64() * 1e6 / reps as f64)
}

fn main() -> Result<()> {
    let gpu = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let k = Glm5NextMhcKernels::resolve(&gpu)?;
    let k_ref = gpu.kernel("glm5next_mhc", "glm5next_hc_post_ref")?;
    println!(
        "glm5next hc_post gate — glm5next_hc_post_ref is the ORACLE, the widened kernel must be byte-identical\n"
    );

    for (hid, hc, t) in CASES {
        for alias in [false, true] {
            let mut seed = 0x9051_0000_u64 ^ (hid as u64) << 20 ^ (hc as u64) << 8 ^ t as u64;
            let block_out: Vec<f32> = (0..t * hid).map(|_| lcg(&mut seed)).collect();
            let residual: Vec<f32> = (0..t * hc * hid).map(|_| lcg(&mut seed)).collect();
            let post: Vec<f32> = (0..t * hc).map(|_| lcg(&mut seed) + 1.0).collect();
            let comb: Vec<f32> = (0..t * hc * hc).map(|_| lcg(&mut seed)).collect();

            let d_block = up_bf16(&gpu, &block_out)?;
            let d_post = up_f32(&gpu, &post)?;
            let d_comb = up_f32(&gpu, &comb)?;
            let out_b = t * hc * hid * 4;

            // 2026-09-25: Each arm gets its own residual buffer: under `alias` the kernel writes
            // into it, so a shared one would feed arm B what arm A left behind.
            let ra = up_f32(&gpu, &residual)?;
            let rb = up_f32(&gpu, &residual)?;
            let (oa, ob) = if alias {
                (ra, rb)
            } else {
                let (a, b) = (gpu.alloc(out_b)?, gpu.alloc(out_b)?);
                poison(&gpu, a, out_b)?;
                poison(&gpu, b, out_b)?;
                (a, b)
            };

            KernelLaunch::new(&gpu, k_ref)
                .grid([t as u32, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(d_block)
                .arg_ptr(ra)
                .arg_ptr(d_post)
                .arg_ptr(d_comb)
                .arg_ptr(oa)
                .arg_u32(hid as u32)
                .arg_u32(hc as u32)
                .launch(0)?;
            gpu.synchronize(0)?;
            let a_h = dn(&gpu, oa, out_b)?;

            glm_hc_post(
                &gpu, k.hc_post, d_block, rb, d_post, d_comb, ob, t as u32, hid as u32, hc as u32,
                0,
            )?;
            gpu.synchronize(0)?;
            let b_h = dn(&gpu, ob, out_b)?;

            if a_h.iter().all(|&x| x == 0xAB) {
                bail!("H={hid} hc={hc} T={t} alias={alias}: the ORACLE never wrote `out`");
            }
            if a_h != b_h {
                let i = a_h.iter().zip(&b_h).position(|(x, y)| x != y).unwrap();
                bail!(
                    "H={hid} hc={hc} T={t} alias={alias}: `out` differs at byte {i}: \
                     oracle {:#04x} widened {:#04x}",
                    a_h[i],
                    b_h[i]
                );
            }
            println!("  H={hid:>5} hc={hc} T={t} alias={alias:<5}  out BYTE-IDENTICAL");

            if (hid, hc, t, alias) == (4096, 4, 1, true) {
                let reps = 500;
                let t_ref = time_us(&gpu, reps, || {
                    KernelLaunch::new(&gpu, k_ref)
                        .grid([t as u32, 1, 1])
                        .block([256, 1, 1])
                        .arg_ptr(d_block)
                        .arg_ptr(ra)
                        .arg_ptr(d_post)
                        .arg_ptr(d_comb)
                        .arg_ptr(oa)
                        .arg_u32(hid as u32)
                        .arg_u32(hc as u32)
                        .launch(0)
                })?;
                let t_new = time_us(&gpu, reps, || {
                    glm_hc_post(
                        &gpu, k.hc_post, d_block, rb, d_post, d_comb, ob, t as u32, hid as u32,
                        hc as u32, 0,
                    )
                })?;
                println!(
                    "\n  TIMING (GLM decode shape H=4096 hc=4 T=1, {reps} reps, us/call):\n    \
                     hc_post_ref (1 block) {t_ref:8.1}\n    hc_post     (16 blocks) {t_new:8.1}\n    \
                     speedup                {:8.2}x\n    \
                     per token (90 sites)   {:8.3} ms -> {:8.3} ms\n",
                    t_ref / t_new,
                    t_ref * 90.0 / 1000.0,
                    t_new * 90.0 / 1000.0
                );
            }
        }
    }

    println!("\nhc_post gate PASS — glm5next_hc_post == glm5next_hc_post_ref, byte for byte");
    Ok(())
}
