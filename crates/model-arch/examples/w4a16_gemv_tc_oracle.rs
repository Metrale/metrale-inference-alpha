// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Accuracy gate for the tensor-core NVFP4 GEMV (`w4a16_gemv_tc.cu`)
//! as `ops::w4a16_gemv_batchm` routes it (`layers/ops/gemv_tc.rs`), against the
//! CUDA-core `w4a16_gemv_batch{4,8,16}` tiers, for M in `MS` and every shape.
//!
//!   1. Armed: `gemv_tc::tc_kernel` resolves and `W4a16BatchmTiers::kernel(m)`
//!      is nonzero, or the run exits 2. An M above `narrow_gemv_max_rows()`
//!      (9..=16 unless `METRALE_W4A16_TC_WIDE` is set) is skipped, and so are
//!      the fixed-M legs.
//!   2. Against a CPU f64 reference over 64 sampled output features plus the
//!      first and last, every row: the routed max error must not exceed
//!      max(1.25 x the tier's error, half a BF16 ulp of the output range).
//!   3. Against the tier, every element must be finite and within one BF16 ulp
//!      of the larger magnitude plus absmax/1024. Bit-identity is not required.
//!   4. Rows >= M of the output are never written (sentinel row M).
//!   5. Fixed-M, with the widening on: `ops::w4a16_gemv_batch2/3` and
//!      `w4a16_gemv_dual_batch2/3` meet check 3 and differ bitwise from the tier.
//!   6. Some element differs bitwise from the tier, so the routed launch really
//!      ran the tensor-core kernel.
//!
//! Negative controls, each of which must trip check 3:
//!   A. output feature 0's group scales raised by two exponent steps;
//!   B. a launch of M-1 rows into an M-row buffer (the last row stays unwritten).
//!
//! Owner: model-arch examples.
//! Invariants: none beyond the types.
//!
//! Exit: 0 pass, 1 any leg failed or a control stayed silent, 2 kernels not loaded.
//!
//! Run:
//!   cargo run -p metrale-model-arch --release --features cuda,gpu-examples \
//!     --example w4a16_gemv_tc_oracle

use anyhow::Result;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};
use metrale_model_layers::layers::ops;
use metrale_model_layers::weight_map::QuantizedWeight;

const SCALE2: f32 = 0.0023;
const SHAPES: [(&str, u32, u32); 7] = [
    ("gdn qkvz ", 16384, 5120),
    ("attn q   ", 12288, 5120),
    ("attn k/v ", 1024, 5120),
    ("o/out    ", 5120, 6144),
    ("ffn gu   ", 34816, 5120),
    ("ffn down ", 5120, 17408),
    ("lm_head  ", 248077, 5120),
];
const MS: [u32; 10] = [1, 2, 3, 4, 5, 8, 9, 12, 15, 16];
const E2M1: [f64; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

fn e4m3(b: u8) -> f64 {
    let s = if b & 0x80 != 0 { -1.0 } else { 1.0 };
    let (e, m) = ((b >> 3) & 0xF, b & 7);
    if e == 0 {
        s * m as f64 * 2f64.powi(-9)
    } else {
        s * 2f64.powi(e as i32 - 7) * (1.0 + m as f64 / 8.0)
    }
}
fn bf(b: u16) -> f64 {
    f32::from_bits((b as u32) << 16) as f64
}
fn bf_ulp(v: f64) -> f64 {
    if v == 0.0 {
        0.0
    } else {
        2f64.powi(v.abs().log2().floor() as i32 - 7)
    }
}
fn to_u16(b: &[u8]) -> Vec<u16> {
    b.chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

struct Case<'a> {
    g: &'a dyn GpuBackend,
    a: DevicePtr,
    w: QuantizedWeight,
    c: DevicePtr,
    n: u32,
    k: u32,
}

impl Case<'_> {
    /// 2026-09-25: Run `f` into an `[(m+1) x n]` output whose row m is a 0xFFFF sentinel.
    fn run(&self, m: u32, f: impl Fn(DevicePtr) -> Result<()>) -> Result<Vec<u16>> {
        let bytes = (m as usize + 1) * self.n as usize * 2;
        self.g.copy_h2d(&vec![0xFFu8; bytes], self.c)?;
        f(self.c)?;
        self.g.synchronize(0)?;
        let mut out = vec![0u8; bytes];
        self.g.copy_d2h(self.c, &mut out)?;
        Ok(to_u16(&out))
    }
    fn tier(&self, kh: KernelHandle, m: u32) -> Result<Vec<u16>> {
        self.run(m, |c| {
            KernelLaunch::new(self.g, kh)
                .grid([div_ceil(self.n, 4), 1, 1])
                .block([256, 1, 1])
                .arg_ptr(self.a)
                .arg_ptr(self.w.weight)
                .arg_ptr(self.w.weight_scale)
                .arg_f32(SCALE2)
                .arg_ptr(c)
                .arg_u32(m)
                .arg_u32(self.n)
                .arg_u32(self.k)
                .launch(0)
        })
    }
    /// 2026-09-25: The serving launcher `ops::w4a16_gemv_batchm`, which launches the
    /// tensor-core kernel whenever `gemv_tc::tc_kernel` resolves.
    fn routed(
        &self,
        kh: KernelHandle,
        m_launch: u32,
        m_buf: u32,
        w: &QuantizedWeight,
    ) -> Result<Vec<u16>> {
        self.run(m_buf, |c| {
            ops::w4a16_gemv_batchm(self.g, kh, self.a, w, c, m_launch, self.n, self.k, 0)
        })
    }
}

/// 2026-09-25: Checks 3 and 4. Returns the number of violating elements.
fn compare(tc: &[u16], tier: &[u16], m: usize, n: usize) -> usize {
    let absmax = tier[..m * n]
        .iter()
        .map(|&b| bf(b).abs())
        .fold(0.0, f64::max);
    let mut bad = 0;
    for i in 0..m * n {
        let (x, y) = (bf(tc[i]), bf(tier[i]));
        let tol = bf_ulp(x.abs().max(y.abs())) + absmax / 1024.0;
        if !x.is_finite() || (x - y).abs() > tol {
            bad += 1;
        }
    }
    bad + tc[m * n..].iter().filter(|&&b| b != 0xFFFF).count()
}

fn main() -> Result<()> {
    let backend = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let tier = |w: u32| g.kernel("w4a16_gemv", &format!("w4a16_gemv_batch{w}")).ok();
    let (Some(b4), Some(b8), Some(b16)) = (tier(4), tier(8), tier(16)) else {
        eprintln!("batch tiers not in this target's module set");
        std::process::exit(2);
    };
    let tier_for = |m: u32| {
        if m <= 4 {
            b4
        } else if m <= 8 {
            b8
        } else {
            b16
        }
    };
    let sites = metrale_model_layers::layers::w4a16_gemv_tiers::W4a16BatchmTiers::resolve(g);
    let fixed_k = |m: u32| {
        g.kernel("w4a16_gemv", &format!("w4a16_gemv_batch{m}"))
            .expect("batch2/3")
    };
    let dual_k = |m: u32| {
        g.kernel("w4a16_gemv", &format!("w4a16_gemv_dual_batch{m}"))
            .expect("dual2/3")
    };

    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let a_host: Vec<u16> = (0..16 * 17408)
        .map(|_| {
            let v = ((rng.next() >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 6.0;
            (v.to_bits() >> 16) as u16
        })
        .collect();
    let a_bytes: Vec<u8> = a_host.iter().flat_map(|v| v.to_le_bytes()).collect();
    let a = g.alloc(a_bytes.len())?;
    g.copy_h2d(&a_bytes, a)?;
    let c = g.alloc(2 * 17 * 248320 * 2 + 4096)?;

    let mut failures = 0usize;
    let mut controls_ok = true;
    let mut lever_moved = false;
    for (label, n, k) in SHAPES {
        let (nu, ku) = (n as usize, k as usize);
        let wb: Vec<u8> = (0..nu * ku / 2).map(|_| (rng.next() >> 32) as u8).collect();
        let mut sb: Vec<u8> = (0..nu * ku / 16)
            .map(|i| {
                let r = (rng.next() >> 32) as u8;
                // 2026-09-25: Finite E4M3 with exponent field 1..=14; every 97th group
                // has exponent field 0 (subnormal or zero).
                if i % 97 == 0 {
                    r & 0x87
                } else {
                    (r & 0x87) | ((1 + (r >> 3) % 14) << 3)
                }
            })
            .collect();
        let wp = g.alloc(wb.len())?;
        g.copy_h2d(&wb, wp)?;
        let sp = g.alloc(sb.len())?;
        g.copy_h2d(&sb, sp)?;
        let w = QuantizedWeight {
            weight: wp,
            weight_scale: sp,
            weight_scale_2: SCALE2,
            ..QuantizedWeight::null()
        };
        let case = Case { g, a, w, c, n, k };

        let rows: Vec<usize> = (0..64)
            .map(|r| (r * 2654435761usize) % nu)
            .chain([0, nu - 1])
            .collect();
        for m in MS {
            if m > ops::gemv_tc::narrow_gemv_max_rows() {
                println!(
                    "{label} M={m:2}  skipped: 9..16-row edge is opt-in (METRALE_W4A16_TC_WIDE)"
                );
                continue;
            }
            if ops::gemv_tc::tc_kernel(g, m, n, k).is_none() {
                eprintln!("NOT ARMED: tensor-core kernel did not resolve at m={m} n={n} k={k}");
                std::process::exit(2);
            }
            // 2026-09-25: The handle `W4a16BatchmTiers::kernel` gives a dispatch site.
            let site = sites.kernel(m);
            if site.0 == 0 {
                eprintln!("NOT ARMED: no dispatch-site handle at m={m}");
                std::process::exit(2);
            }
            let kh = tier_for(m);
            let mu = m as usize;
            let tv = case.tier(kh, m)?;
            let tc = case.routed(site, m, m, &case.w)?;
            let (mut e_tier, mut e_tc, mut range) = (0f64, 0f64, 0f64);
            for r in 0..mu {
                for &row in &rows {
                    let mut acc = 0f64;
                    for kk in 0..ku {
                        let byte = wb[row * ku / 2 + kk / 2];
                        let nib = if kk & 1 == 1 { byte >> 4 } else { byte & 0xF };
                        acc += bf(a_host[r * ku + kk])
                            * E2M1[nib as usize]
                            * e4m3(sb[row * ku / 16 + kk / 16]);
                    }
                    acc *= SCALE2 as f64;
                    range = range.max(acc.abs());
                    e_tier = e_tier.max((bf(tv[r * nu + row]) - acc).abs());
                    e_tc = e_tc.max((bf(tc[r * nu + row]) - acc).abs());
                }
            }
            let cpu_ok = e_tc <= (1.25 * e_tier).max(bf_ulp(range) / 2.0);
            let bad = compare(&tc, &tv, mu, nu);
            // 2026-09-25: Check 6: count the elements that differ from the tier.
            let differ = tc[..mu * nu]
                .iter()
                .zip(&tv[..mu * nu])
                .filter(|(x, y)| x != y)
                .count();
            lever_moved |= differ > 0;
            let ok = cpu_ok && bad == 0;
            failures += usize::from(!ok);
            println!(
                "{label} N={n:6} K={k:5} M={m:2}  cpu_err tier={e_tier:.3e} tc={e_tc:.3e} (range {range:.1})  \
                 elems_out_of_tol={bad} bits_differ={differ}  {}",
                if ok { "PASS" } else { "FAIL" }
            );
        }

        // 2026-09-25: Check 5. `ops::w4a16_gemv_batch2/3` and `dual_batch2/3` take
        // the tensor-core kernel only with the widening on (`gemv_tc::tc_fixed_m`).
        let fixed_ms: &[u32] = if ops::gemv_tc::wide_rows_enabled() {
            &[2, 3]
        } else {
            &[]
        };
        for &m in fixed_ms {
            let tv = case.tier(tier_for(m), m)?;
            let (fixed, dual) = (fixed_k(m), dual_k(m));
            let one = case.run(m, |c| {
                if m == 2 {
                    ops::w4a16_gemv_batch2(g, fixed, a, &case.w, c, n, k, 0)
                } else {
                    ops::w4a16_gemv_batch3(g, fixed, a, &case.w, c, n, k, 0)
                }
            })?;
            // 2026-09-25: The same weight twice: output1 goes to the sentinel-checked
            // buffer, output0 to a scratch region past it.
            let scratch = c.offset(((m as usize + 1) * nu * 2).next_multiple_of(256));
            let two = case.run(m, |c1| {
                if m == 2 {
                    ops::w4a16_gemv_dual_batch2(g, dual, a, &case.w, scratch, &case.w, c1, n, k, 0)
                } else {
                    ops::w4a16_gemv_dual_batch3(g, dual, a, &case.w, scratch, &case.w, c1, n, k, 0)
                }
            })?;
            let (bad1, bad2) = (
                compare(&one, &tv, m as usize, nu),
                compare(&two, &tv, m as usize, nu),
            );
            let differ = one[..m as usize * nu]
                .iter()
                .zip(&tv[..m as usize * nu])
                .filter(|(x, y)| x != y)
                .count();
            let ok = bad1 == 0 && bad2 == 0 && differ > 0;
            failures += usize::from(!ok);
            println!(
                "{label} FIXED M={m}  batch{m} out_of_tol={bad1}  dual_batch{m} out_of_tol={bad2}  \
                 bits_differ={differ} (0 = tc not armed)  {}",
                if ok { "PASS" } else { "FAIL" }
            );
        }

        // 2026-09-25: Control A: output feature 0's scale exponents +2, capped at 14.
        let m = 4u32;
        let tv = case.tier(tier_for(m), m)?;
        let groups = ku / 16;
        let orig: Vec<u8> = sb[..groups].to_vec();
        for b in &mut sb[..groups] {
            let e = (*b >> 3) & 0xF;
            *b = (*b & 0x87) | ((e + 2).min(14) << 3);
        }
        g.copy_h2d(&sb[..groups], sp)?;
        let tc_bad = case.routed(tier_for(m), m, m, &case.w)?;
        sb[..groups].copy_from_slice(&orig);
        g.copy_h2d(&sb[..groups], sp)?;
        let ctl_a = compare(&tc_bad, &tv, m as usize, nu);
        // 2026-09-25: Control B: M-1 rows launched into an M-row buffer.
        let tc_short = case.routed(tier_for(m), m - 1, m, &case.w)?;
        let ctl_b = compare(&tc_short, &tv, m as usize, nu);
        let this_ok = ctl_a > 0 && ctl_b > 0;
        controls_ok &= this_ok;
        println!(
            "{label} CONTROLS  scale-byte-changed trips={ctl_a}  short-M trips={ctl_b}  {}",
            if this_ok {
                "CONTROLS-FIRE"
            } else {
                "CONTROL-SILENT (gate is blind)"
            }
        );
        g.free(wp)?;
        g.free(sp)?;
    }
    let pass = failures == 0 && controls_ok && lever_moved;
    println!(
        "\nORACLE {} (legs failed: {failures}, controls fire: {controls_ok}, lever moved: {lever_moved})",
        if pass { "PASS" } else { "FAIL" }
    );
    std::process::exit(if pass { 0 } else { 1 });
}
