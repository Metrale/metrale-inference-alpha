// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Device checks of the two RMS-norm weight conventions against an
//! f64 host formula: `rms_norm_vanilla` (out = x * w / rms) and `rms_norm`
//! (out = x * (1 + w) / rms).
//!
//! Owner: model-arch examples (GPU microtests).
//! Invariants:
//! - The run exits 0 only if every check below passed.
//!
//! Checks, by the label each section prints:
//!   V2  `rms_norm_vanilla` against the formula, for every weight range in
//!       `cases`, hidden size and seed: max relative error <= 0.01.
//!   I1  `rms_norm` with a zero weight equals plain normalization exactly.
//!   V3  `rms_norm` with w ~ U(-0.5, 0.5) against x * (1 + w) / rms: max
//!       relative error < 0.01.
//!   V1d per weight range, `rms_norm` on a stored `bf16(w - 1)` against
//!       `rms_norm_vanilla` on the exact `w`, both scored against the formula
//!       with the exact weight: the vanilla arm's max relative error must not
//!       exceed the offset arm's when the latter is above 1e-6.
//!
//!   cargo run -p metrale-model-arch --release --example rmsnorm_vanilla_microtest \
//!     --features cuda,gpu-examples
use anyhow::{Result, bail};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_gpu_runtime::kernel_args::KernelLaunch;

const EPS: f32 = 1e-6;

struct Lcg(u64);
impl Lcg {
    fn f(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
    fn r(&mut self, a: f64, b: f64) -> f64 {
        a + (b - a) * self.f()
    }
}

fn ub(g: &dyn GpuBackend, d: &[bf16]) -> Result<DevicePtr> {
    let b: Vec<u8> = d.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect();
    let p = g.alloc(b.len())?;
    g.copy_h2d(&b, p)?;
    Ok(p)
}
fn db(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<f64>> {
    let mut b = vec![0u8; n * 2];
    g.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(2)
        .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f64())
        .collect())
}

/// 2026-09-25: F64 reference. `w_eff` is the weight the formula applies.
/// Inputs are the exact BF16 values the device sees, so any delta is the
/// kernel's.
fn truth(x: &[bf16], w_eff: &[f64], hidden: usize, tokens: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; tokens * hidden];
    for t in 0..tokens {
        let xs = &x[t * hidden..(t + 1) * hidden];
        let ms: f64 = xs.iter().map(|v| v.to_f64() * v.to_f64()).sum::<f64>() / hidden as f64;
        let rms = 1.0 / (ms + EPS as f64).sqrt();
        for i in 0..hidden {
            // 2026-09-25: Round through BF16, as the kernel's store does.
            out[t * hidden + i] = bf16::from_f32((xs[i].to_f64() * rms * w_eff[i]) as f32).to_f64();
        }
    }
    out
}

fn launch(
    g: &dyn GpuBackend,
    module: &str,
    func: &str,
    x: DevicePtr,
    w: DevicePtr,
    o: DevicePtr,
    tokens: u32,
    hidden: u32,
) -> Result<()> {
    let k = g.kernel(module, func)?;
    KernelLaunch::new(g, k)
        .grid([tokens, 1, 1])
        .block([hidden.min(1024), 1, 1])
        .arg_ptr(x)
        .arg_ptr(w)
        .arg_ptr(o)
        .arg_u32(hidden)
        .arg_f32(EPS)
        .launch(0)?;
    g.synchronize(0)?;
    Ok(())
}

/// 2026-09-25: max |device - truth|, and the max relative error against the
/// truth magnitude (elements with |truth| <= 1e-9 skipped).
fn cmp(dev: &[f64], truth_v: &[f64]) -> (f64, f64) {
    let mut amax = 0.0f64;
    let mut rmax = 0.0f64;
    for (d, t) in dev.iter().zip(truth_v) {
        let a = (d - t).abs();
        amax = amax.max(a);
        if t.abs() > 1e-9 {
            rmax = rmax.max(a / t.abs());
        }
    }
    (amax, rmax)
}

/// 2026-09-25: The effective weight when `bf16(w - 1)` is stored and the
/// offset kernel adds 1 back.
fn old_effective(w: f64) -> f64 {
    1.0 + bf16::from_f32((w - 1.0) as f32).to_f64()
}

struct Case {
    name: &'static str,
    lo: f64,
    hi: f64,
}

fn main() -> Result<()> {
    let gpu = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &gpu;

    let dims = [512usize, 1024, 4096, 7168];
    let seeds = [1u64, 7, 42, 1337, 90210];
    let cases = [
        Case {
            name: "v4_qnorm    (~0.04)",
            lo: 0.0098,
            hi: 0.2402,
        },
        Case {
            name: "v4_attnnorm (~0.03)",
            lo: 0.0084,
            hi: 0.0400,
        },
        Case {
            name: "straddle_zero       ",
            lo: -0.2256,
            hi: 0.2256,
        },
        Case {
            name: "near_one            ",
            lo: 0.6300,
            hi: 0.9900,
        },
        Case {
            name: "large               ",
            lo: 1.0000,
            hi: 3.8281,
        },
        Case {
            name: "negative            ",
            lo: -1.5000,
            hi: -0.0100,
        },
    ];

    let mut fail = 0usize;
    println!("=== V2 — production `rms_norm_vanilla` vs F64 truth (out = x*w/rms) ===");
    println!(
        "{:<22} {:>6} {:>6} {:>12} {:>12}",
        "weight regime", "hidden", "seed", "max |abs|", "max rel"
    );
    println!("{}", "-".repeat(64));
    for c in &cases {
        for &hidden in &dims {
            for &seed in &seeds {
                let tokens = 3usize;
                let mut r = Lcg(seed);
                let x: Vec<bf16> = (0..tokens * hidden)
                    .map(|_| bf16::from_f32(r.r(-3.0, 3.0) as f32))
                    .collect();
                let w: Vec<bf16> = (0..hidden)
                    .map(|_| bf16::from_f32(r.r(c.lo, c.hi) as f32))
                    .collect();
                let w64: Vec<f64> = w.iter().map(|v| v.to_f64()).collect();

                let xp = ub(g, &x)?;
                let wp = ub(g, &w)?;
                let op = g.alloc(tokens * hidden * 2)?;
                launch(
                    g,
                    "rms_norm_vanilla",
                    "rms_norm_vanilla",
                    xp,
                    wp,
                    op,
                    tokens as u32,
                    hidden as u32,
                )?;
                let dev = db(g, op, tokens * hidden)?;
                let truth_v = truth(&x, &w64, hidden, tokens);
                let (a, rel) = cmp(&dev, &truth_v);
                // 2026-09-25: The gate is a relative error of 0.01, just above one
                // BF16 ULP relative (2^-7, about 0.0078).
                let bad = rel > 0.01;
                if bad {
                    fail += 1;
                }
                if seed == seeds[0] && hidden == dims[0] {
                    println!(
                        "{:<22} {:>6} {:>6} {:>12.3e} {:>12.3e}{}",
                        c.name,
                        hidden,
                        seed,
                        a,
                        rel,
                        if bad { "   <<< FAIL" } else { "" }
                    );
                }
                if bad {
                    println!(
                        "  FAIL {:<20} hidden={} seed={} abs={:.3e} rel={:.3e}",
                        c.name, hidden, seed, a, rel
                    );
                }
            }
        }
    }
    println!(
        "V2: {} regimes x {} dims x {} seeds = {} cells, {} failures\n",
        cases.len(),
        dims.len(),
        seeds.len(),
        cases.len() * dims.len() * seeds.len(),
        fail
    );

    // 2026-09-25: I1: the offset kernel with a zero weight must equal plain
    // normalization exactly.
    println!("=== I1 — `rms_norm` + zero weight (norm_unit_w) == pure normalize (unchanged) ===");
    let hidden = 512usize;
    let tokens = 4usize;
    let mut r = Lcg(2026);
    let x: Vec<bf16> = (0..tokens * hidden)
        .map(|_| bf16::from_f32(r.r(-3.0, 3.0) as f32))
        .collect();
    let wz: Vec<bf16> = vec![bf16::from_f32(0.0); hidden];
    let xp = ub(g, &x)?;
    let wp = ub(g, &wz)?;
    let op = g.alloc(tokens * hidden * 2)?;
    launch(
        g,
        "norm",
        "rms_norm",
        xp,
        wp,
        op,
        tokens as u32,
        hidden as u32,
    )?;
    let dev = db(g, op, tokens * hidden)?;
    let ones = vec![1.0f64; hidden];
    let truth_v = truth(&x, &ones, hidden, tokens);
    let (a, rel) = cmp(&dev, &truth_v);
    let i1_ok = a == 0.0;
    println!(
        "  offset kernel, w=0  ->  max|abs| {:.3e}  max rel {:.3e}   {}",
        a,
        rel,
        if i1_ok {
            "BIT-EXACT — invariant holds"
        } else {
            "<<< FAIL"
        }
    );
    if !i1_ok {
        fail += 1;
    }

    // 2026-09-25: V3: the offset kernel computes x * (1 + w) / rms.
    println!("\n=== V3 — regression: `rms_norm` (offset-from-1) is UNCHANGED for non-V4 ===");
    let mut r = Lcg(5150);
    let wo: Vec<bf16> = (0..hidden)
        .map(|_| bf16::from_f32(r.r(-0.5, 0.5) as f32))
        .collect();
    // 2026-09-25: Offset-from-1 semantics: effective weight = 1 + stored.
    let w_eff: Vec<f64> = wo.iter().map(|v| 1.0 + v.to_f64()).collect();
    let wp = ub(g, &wo)?;
    let op = g.alloc(tokens * hidden * 2)?;
    launch(
        g,
        "norm",
        "rms_norm",
        xp,
        wp,
        op,
        tokens as u32,
        hidden as u32,
    )?;
    let dev = db(g, op, tokens * hidden)?;
    let truth_v = truth(&x, &w_eff, hidden, tokens);
    let (a, rel) = cmp(&dev, &truth_v);
    let v3_ok = rel < 0.01;
    println!(
        "  offset kernel, w~U(-0.5,0.5)  ->  max|abs| {:.3e}  max rel {:.3e}   {}",
        a,
        rel,
        if v3_ok {
            "UNCHANGED — offset path intact"
        } else {
            "<<< FAIL"
        }
    );
    if !v3_ok {
        fail += 1;
    }

    // 2026-09-25: V1d: stored bf16(w - 1) on the offset kernel against the exact
    // weight on the vanilla kernel, both scored against the exact-weight
    // reference.
    println!(
        "\n=== V1d — OLD path (1 + bf16(w-1)) vs NEW path (exact w), device, vs F64 truth ==="
    );
    println!(
        "{:<22} {:>13} {:>13} {:>13} {:>13}",
        "weight regime", "OLD max|abs|", "OLD max rel", "NEW max|abs|", "NEW max rel"
    );
    println!("{}", "-".repeat(80));
    for c in &cases {
        let mut r = Lcg(31337);
        let x: Vec<bf16> = (0..tokens * hidden)
            .map(|_| bf16::from_f32(r.r(-3.0, 3.0) as f32))
            .collect();
        let w: Vec<bf16> = (0..hidden)
            .map(|_| bf16::from_f32(r.r(c.lo, c.hi) as f32))
            .collect();
        let w64: Vec<f64> = w.iter().map(|v| v.to_f64()).collect();
        let truth_v = truth(&x, &w64, hidden, tokens);
        let xp = ub(g, &x)?;

        // 2026-09-25: Offset arm: store bf16(w - 1); the offset kernel adds 1.
        let w_old: Vec<bf16> = w
            .iter()
            .map(|v| bf16::from_f32((v.to_f64() - 1.0) as f32))
            .collect();
        let wp_old = ub(g, &w_old)?;
        let op_old = g.alloc(tokens * hidden * 2)?;
        launch(
            g,
            "norm",
            "rms_norm",
            xp,
            wp_old,
            op_old,
            tokens as u32,
            hidden as u32,
        )?;
        let (a_old, r_old) = cmp(&db(g, op_old, tokens * hidden)?, &truth_v);

        // 2026-09-25: Vanilla arm: the exact weight.
        let wp_new = ub(g, &w)?;
        let op_new = g.alloc(tokens * hidden * 2)?;
        launch(
            g,
            "rms_norm_vanilla",
            "rms_norm_vanilla",
            xp,
            wp_new,
            op_new,
            tokens as u32,
            hidden as u32,
        )?;
        let (a_new, r_new) = cmp(&db(g, op_new, tokens * hidden)?, &truth_v);

        // 2026-09-25: The offset arm's output against a reference built from
        // `old_effective`, printed as a check of that model.
        let pred: Vec<f64> = w64.iter().map(|v| old_effective(*v)).collect();
        let pred_truth = truth(&x, &pred, hidden, tokens);
        let (a_pred, _) = cmp(&db(g, op_old, tokens * hidden)?, &pred_truth);

        println!(
            "{:<22} {:>13.3e} {:>13.3e} {:>13.3e} {:>13.3e}   [old==1+bf16(w-1) model: {:.1e}]",
            c.name, a_old, r_old, a_new, r_new, a_pred
        );
        if r_new > r_old && r_old > 1e-6 {
            println!("  <<< FAIL: the new path is not better than the old one");
            fail += 1;
        }
    }

    println!();
    if fail > 0 {
        bail!("{fail} failure(s) — READING R1: STOP AND REPAIR. No behavioral eval.");
    }
    println!("ALL DEVICE CHECKS PASS (V2 + I1 + V3 + V1d).");
    Ok(())
}
