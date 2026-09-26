// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Bit-identity gate: the split `glm5next_hc_mix` + `glm5next_hc_finish` pair (and
//! `glm5next_hc_mix_bf16`) against the fused oracle `glm5next_hc_pre`.
//!
//! Owner: model-arch examples (GLM-5.3 mHC kernels).
//! Invariants:
//! - The run fails unless `y`, `post` and `comb` are byte-identical between the oracle and the
//!   split, for the F32 `hc_fn` and for the BF16 `hc_fn`, and it fails if the oracle left an
//!   output unwritten.
//!
//! `glm_hc_pre` launches the split pair; the fused kernel is the oracle. The gate compares
//! bytes, not a tolerance: each mix block recomputes the RMS and reduces its row in the fused
//! kernel's order, and `hc_finish` repeats the fused kernel's per-element arithmetic, so any
//! difference is a real reordering.
//!
//!   cargo run -p metrale-model-arch --release --example glm5next_hc_split_gate \
//!       --features cuda,gpu-examples
//!
//! Both arms read the same device inputs, and every output is filled with 0xAB before its
//! launch, so a kernel that fails to write cannot pass as a match.

use anyhow::{Result, bail};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use metrale_model_arch::glm5next_mhc::{
    Glm5NextMhcKernels, Glm5NextMhcSiteWeights, glm_hc_pre, mhc_mix_max_tokens,
};

/// 2026-09-25: `(hidden, hc_mult, tokens)`. The first case uses GLM-5.3's hidden size and
/// `hc_mult`, and it is the only case the timing block runs.
const CASES: [(usize, usize, usize); 5] = [
    (4096, 4, 1),
    (5120, 4, 1),
    (5120, 4, 7),
    (1024, 2, 3),
    (256, 4, 1),
];
const SINKHORN_ITERS: u32 = 20;
const HC_EPS: f32 = 1e-6;
const NORM_EPS: f32 = 1e-5;

fn mix_hc(hc: usize) -> usize {
    (2 + hc) * hc
}

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

/// 2026-09-25: Upload as BF16, the width the GLM-5.3 loader uploads `hc_{site}_fn` at, for the
/// arm that reads `hc_fn` through `glm5next_hc_mix_bf16`.
fn up_bf16(g: &dyn GpuBackend, d: &[f32]) -> Result<DevicePtr> {
    let b: Vec<u8> = d
        .iter()
        .flat_map(|&x| half::bf16::from_f32(x).to_le_bytes())
        .collect();
    let p = g.alloc(b.len().max(1))?;
    g.copy_h2d(&b, p)?;
    Ok(p)
}

/// 2026-09-25: Round through BF16 and back. The BF16 arm can only be compared against an oracle
/// fed values BF16 can hold; otherwise the test measures rounding, not the kernel.
fn to_bf16_exact(d: &[f32]) -> Vec<f32> {
    d.iter()
        .map(|&x| half::bf16::from_f32(x).to_f32())
        .collect()
}

fn dn(g: &dyn GpuBackend, p: DevicePtr, bytes: usize) -> Result<Vec<u8>> {
    let mut b = vec![0u8; bytes];
    g.copy_d2h(p, &mut b)?;
    Ok(b)
}

/// 2026-09-25: Fill `p` with 0xAB so an unwritten output cannot pass as a match with the other
/// arm.
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
    println!(
        "glm5next hc split gate — fused glm5next_hc_pre is the ORACLE, split must be byte-identical\n"
    );

    for (hid, hc, t) in CASES {
        if t > mhc_mix_max_tokens() {
            bail!("case T={t} exceeds mhc_mix_max_tokens()");
        }
        let m = mix_hc(hc);
        let hc_dim = hc * hid;
        let mut seed = 0x5eed_1234_u64 ^ (hid as u64) << 20 ^ (hc as u64) << 8 ^ t as u64;
        let streams: Vec<f32> = (0..t * hc_dim).map(|_| lcg(&mut seed)).collect();
        // 2026-09-25: `hc_fn` is scaled down only to keep the mixes small: each is a dot over
        // hc*H terms, and unscaled it would saturate the sigmoids and the softmax. The byte
        // comparison holds at any scale.
        let hc_fn: Vec<f32> = (0..m * hc_dim).map(|_| lcg(&mut seed) * 0.02).collect();
        let hc_base: Vec<f32> = (0..m).map(|_| lcg(&mut seed)).collect();
        let hc_scale: Vec<f32> = vec![0.7, 1.3, 0.9];

        let d_streams = up_f32(&gpu, &streams)?;
        let d_fn = up_f32(&gpu, &hc_fn)?;
        let d_base = up_f32(&gpu, &hc_base)?;
        let d_scale = up_f32(&gpu, &hc_scale)?;

        let y_b = t * hid * 2;
        let post_b = t * hc * 4;
        let comb_b = t * hc * hc * 4;

        let (ya, pa, ca) = (gpu.alloc(y_b)?, gpu.alloc(post_b)?, gpu.alloc(comb_b)?);
        poison(&gpu, ya, y_b)?;
        poison(&gpu, pa, post_b)?;
        poison(&gpu, ca, comb_b)?;
        KernelLaunch::new(&gpu, k.hc_pre)
            .grid([t as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(d_streams)
            .arg_ptr(d_fn)
            .arg_ptr(d_scale)
            .arg_ptr(d_base)
            .arg_ptr(ya)
            .arg_ptr(pa)
            .arg_ptr(ca)
            .arg_u32(hid as u32)
            .arg_u32(hc as u32)
            .arg_u32(SINKHORN_ITERS)
            .arg_f32(NORM_EPS)
            .arg_f32(HC_EPS)
            .launch(0)?;
        gpu.synchronize(0)?;
        let (ya_h, pa_h, ca_h) = (
            dn(&gpu, ya, y_b)?,
            dn(&gpu, pa, post_b)?,
            dn(&gpu, ca, comb_b)?,
        );

        let w = Glm5NextMhcSiteWeights {
            hc_fn: d_fn,
            hc_fn_bf16: false,
            hc_scale: d_scale,
            hc_base: d_base,
            mix: gpu.alloc(mhc_mix_max_tokens() * m * 4)?,
        };
        let (yb, pb, cb) = (gpu.alloc(y_b)?, gpu.alloc(post_b)?, gpu.alloc(comb_b)?);
        poison(&gpu, yb, y_b)?;
        poison(&gpu, pb, post_b)?;
        poison(&gpu, cb, comb_b)?;
        glm_hc_pre(
            &gpu,
            &k,
            d_streams,
            &w,
            yb,
            pb,
            cb,
            t as u32,
            hid as u32,
            hc as u32,
            SINKHORN_ITERS,
            NORM_EPS,
            HC_EPS,
            0,
        )?;
        gpu.synchronize(0)?;
        let (yb_h, pb_h, cb_h) = (
            dn(&gpu, yb, y_b)?,
            dn(&gpu, pb, post_b)?,
            dn(&gpu, cb, comb_b)?,
        );

        for (name, a, b) in [
            ("y", &ya_h, &yb_h),
            ("post", &pa_h, &pb_h),
            ("comb", &ca_h, &cb_h),
        ] {
            if a.iter().all(|&x| x == 0xAB) {
                bail!(
                    "H={hid} hc={hc} T={t}: `{name}` is still poison — the ORACLE never wrote it"
                );
            }
            if a != b {
                let i = a.iter().zip(b).position(|(x, y)| x != y).unwrap();
                bail!(
                    "H={hid} hc={hc} T={t}: `{name}` differs at byte {i}: oracle {:#04x} split {:#04x}",
                    a[i],
                    b[i]
                );
            }
        }
        println!("  H={hid:>5} hc={hc} T={t}  mix_hc={m:>2}  y/post/comb BYTE-IDENTICAL");

        // 2026-09-25: The BF16 `hc_fn` arm. Widening BF16 to F32 is exact, so the split reading
        // BF16 must match the fused oracle reading the same values as F32; `to_bf16_exact`
        // gives the oracle values BF16 can hold.
        let hc_fn_r = to_bf16_exact(&hc_fn);
        let d_fn_r = up_f32(&gpu, &hc_fn_r)?;
        let d_fn_b = up_bf16(&gpu, &hc_fn_r)?;
        let (yc, pc, cc) = (gpu.alloc(y_b)?, gpu.alloc(post_b)?, gpu.alloc(comb_b)?);
        for (p, n) in [(yc, y_b), (pc, post_b), (cc, comb_b)] {
            poison(&gpu, p, n)?;
        }
        KernelLaunch::new(&gpu, k.hc_pre)
            .grid([t as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(d_streams)
            .arg_ptr(d_fn_r)
            .arg_ptr(d_scale)
            .arg_ptr(d_base)
            .arg_ptr(yc)
            .arg_ptr(pc)
            .arg_ptr(cc)
            .arg_u32(hid as u32)
            .arg_u32(hc as u32)
            .arg_u32(SINKHORN_ITERS)
            .arg_f32(NORM_EPS)
            .arg_f32(HC_EPS)
            .launch(0)?;
        gpu.synchronize(0)?;
        let (yc_h, pc_h, cc_h) = (
            dn(&gpu, yc, y_b)?,
            dn(&gpu, pc, post_b)?,
            dn(&gpu, cc, comb_b)?,
        );

        let wb = Glm5NextMhcSiteWeights {
            hc_fn: d_fn_b,
            hc_fn_bf16: true,
            hc_scale: d_scale,
            hc_base: d_base,
            mix: gpu.alloc(mhc_mix_max_tokens() * m * 4)?,
        };
        let (yd, pd, cd) = (gpu.alloc(y_b)?, gpu.alloc(post_b)?, gpu.alloc(comb_b)?);
        for (p, n) in [(yd, y_b), (pd, post_b), (cd, comb_b)] {
            poison(&gpu, p, n)?;
        }
        glm_hc_pre(
            &gpu,
            &k,
            d_streams,
            &wb,
            yd,
            pd,
            cd,
            t as u32,
            hid as u32,
            hc as u32,
            SINKHORN_ITERS,
            NORM_EPS,
            HC_EPS,
            0,
        )?;
        gpu.synchronize(0)?;
        let (yd_h, pd_h, cd_h) = (
            dn(&gpu, yd, y_b)?,
            dn(&gpu, pd, post_b)?,
            dn(&gpu, cd, comb_b)?,
        );
        for (name, a, b) in [
            ("y", &yc_h, &yd_h),
            ("post", &pc_h, &pd_h),
            ("comb", &cc_h, &cd_h),
        ] {
            if a.iter().all(|&x| x == 0xAB) {
                bail!("H={hid} hc={hc} T={t}: BF16 arm `{name}` — the ORACLE never wrote it");
            }
            if a != b {
                let i = a.iter().zip(b).position(|(x, y)| x != y).unwrap();
                bail!(
                    "H={hid} hc={hc} T={t}: BF16 `hc_fn` `{name}` differs at byte {i}: \
                     f32 oracle {:#04x} bf16 split {:#04x}",
                    a[i],
                    b[i]
                );
            }
        }
        println!("  H={hid:>5} hc={hc} T={t}  mix_hc={m:>2}  BF16 hc_fn  BYTE-IDENTICAL");

        if (hid, hc, t) == (4096, 4, 1) {
            let reps = 300;
            let t_fused = time_us(&gpu, reps, || {
                KernelLaunch::new(&gpu, k.hc_pre)
                    .grid([t as u32, 1, 1])
                    .block([256, 1, 1])
                    .arg_ptr(d_streams)
                    .arg_ptr(d_fn)
                    .arg_ptr(d_scale)
                    .arg_ptr(d_base)
                    .arg_ptr(ya)
                    .arg_ptr(pa)
                    .arg_ptr(ca)
                    .arg_u32(hid as u32)
                    .arg_u32(hc as u32)
                    .arg_u32(SINKHORN_ITERS)
                    .arg_f32(NORM_EPS)
                    .arg_f32(HC_EPS)
                    .launch(0)
            })?;
            let t_mix = time_us(&gpu, reps, || {
                KernelLaunch::new(&gpu, k.hc_mix)
                    .grid([t as u32, m as u32, 1])
                    .block([256, 1, 1])
                    .arg_ptr(d_streams)
                    .arg_ptr(d_fn)
                    .arg_ptr(w.mix)
                    .arg_u32(hid as u32)
                    .arg_u32(hc as u32)
                    .arg_f32(NORM_EPS)
                    .launch(0)
            })?;
            let t_fin = time_us(&gpu, reps, || {
                KernelLaunch::new(&gpu, k.hc_finish)
                    .grid([t as u32, 1, 1])
                    .block([256, 1, 1])
                    .arg_ptr(d_streams)
                    .arg_ptr(w.mix)
                    .arg_ptr(d_scale)
                    .arg_ptr(d_base)
                    .arg_ptr(yb)
                    .arg_ptr(pb)
                    .arg_ptr(cb)
                    .arg_u32(hid as u32)
                    .arg_u32(hc as u32)
                    .arg_u32(SINKHORN_ITERS)
                    .arg_f32(HC_EPS)
                    .launch(0)
            })?;
            let t_fin1 = time_us(&gpu, reps, || {
                KernelLaunch::new(&gpu, k.hc_finish)
                    .grid([t as u32, 1, 1])
                    .block([256, 1, 1])
                    .arg_ptr(d_streams)
                    .arg_ptr(w.mix)
                    .arg_ptr(d_scale)
                    .arg_ptr(d_base)
                    .arg_ptr(yb)
                    .arg_ptr(pb)
                    .arg_ptr(cb)
                    .arg_u32(hid as u32)
                    .arg_u32(hc as u32)
                    .arg_u32(1)
                    .arg_f32(HC_EPS)
                    .launch(0)
            })?;
            // 2026-09-25: `t_fin` launches `hc_finish` on grid (T, 1, 1), where block 0 also does
            // the whole collapse. `t_prod` goes through `glm_hc_pre`: the mix, then `hc_finish`
            // on grid (T, 1 + ceil(H/256)), which spreads the collapse over blocks 1.. .
            let t_prod = time_us(&gpu, reps, || {
                glm_hc_pre(
                    &gpu,
                    &k,
                    d_streams,
                    &w,
                    yb,
                    pb,
                    cb,
                    t as u32,
                    hid as u32,
                    hc as u32,
                    SINKHORN_ITERS,
                    NORM_EPS,
                    HC_EPS,
                    0,
                )
            })?;
            println!("    hc_finish@iters=1 {t_fin1:8.1}  (Sinkhorn cost = (fin - fin1) * 20/19)");
            println!(
                "    PRODUCTION mix+finish (wide collapse) {t_prod:8.1} us/call = {:8.3} ms/token over 90 sites",
                t_prod * 90.0 / 1000.0
            );
            println!(
                "\n  TIMING (GLM shape, {reps} reps, us/call):\n    \
                 fused hc_pre {t_fused:8.1}\n    hc_mix       {t_mix:8.1}\n    \
                 hc_finish    {t_fin:8.1}\n    split total  {:8.1}\n    \
                 per token (90 sites) {:8.3} ms\n",
                t_mix + t_fin,
                (t_mix + t_fin) * 90.0 / 1000.0
            );
        }
    }

    println!(
        "\nsplit gate PASS — glm5next_hc_mix + glm5next_hc_finish == glm5next_hc_pre, byte for byte"
    );
    Ok(())
}
