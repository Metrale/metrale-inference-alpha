// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Byte-identity gate for the W4A4 small-M GEMV variants in
//! `w4a4_gemv_mx.cu`: the activation-reuse twins (`w4a4_gemv_mx16_nt2`,
//! `w4a4_gemv_mx32_nt4`), the persistent activation-staged entries
//! (`w4a4_gemv_mx16_ps`, `w4a4_gemv_mx32_ps`) and the wide entries
//! (`w4a4_gemv_mx64`, `w4a4_gemv_mx64_nt2`).
//!
//! With `METRALE_W4A4_MX_NT` unset, `ops::w4a4_proj` serves 9..=64-row launches
//! with the twins or the persistent entries (`w4a4_proj/mx_plan.rs`), so each
//! must write exactly the bytes of the one-tile kernel it replaces:
//!
//!   1. Armed: the quantiser, the one-tile entries, every twin and every
//!      persistent entry resolve, or the run exits 2.
//!   2. For every shape and every M the twin serves (1..=16 for mx16_nt2,
//!      1..=32 for mx32_nt4, 1..=64 for mx64 and mx64_nt2), the twin's [M, N]
//!      output equals the one-tile kernel's bit for bit, and row M (a 0xFFFF
//!      sentinel) is never written. Above 32 rows the reference is mx32 run on
//!      rows [0, 32) and [32, M).
//!   3. The same for each persistent entry, launched as `ops::w4a4_proj` does
//!      (grid = #SMs, its shared-memory size), at every staging depth among
//!      0, 1 and the whole stripe that fits in `PS_SMEM_MAX`. Every case is a
//!      fresh launch.
//!
//! Negative control A: for each twin and persistent entry, at its widest M
//! and once per shape, one weight byte of row 0 is flipped for the tested run;
//! the comparison must then differ.
//!
//! Owner: model-arch examples.
//! Invariants: none beyond the types.
//!
//! Exit: 0 pass, 1 any case differed or a control did not trip, 2 kernels not
//! loaded.
//!
//! Run:
//!   cargo run -p metrale-model-arch --release --features cuda,gpu-examples \
//!     --example w4a4_gemv_nt_oracle

use anyhow::Result;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};
use metrale_model_layers::layers::ops::w4a4_proj::{
    PS_SMEM_MAX, ps_column_blocks, ps_smem_bytes, ps_stripe_chunks,
};

const MODULE: &str = "w4a4_gemv_mx";
const SCALE2: f32 = 0.37;
const MAX_M: u32 = 64;
/// 2026-09-25: `(name, N, K)` of the projection shapes under test.
const SHAPES: [(&str, u32, u32); 8] = [
    ("ffn gate/up", 17408, 5120),
    ("ffn down   ", 5120, 17408),
    ("gdn qkvz   ", 16384, 5120),
    ("gdn qkv    ", 10240, 5120),
    ("gdn z      ", 6144, 5120),
    ("attn q     ", 12288, 5120),
    ("attn k/v   ", 1024, 5120),
    ("o/out_proj ", 5120, 6144),
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

/// 2026-09-25: A twin, its weight rows per CTA, the one-tile entry it must
/// match, and the widest M it is checked at (every M from 1).
struct Twin {
    name: &'static str,
    rows: u32,
    base: &'static str,
    max_m: u32,
}

const TWINS: [Twin; 4] = [
    Twin {
        name: "w4a4_gemv_mx16_nt2",
        rows: 32,
        base: "w4a4_gemv_mx16",
        max_m: 16,
    },
    Twin {
        name: "w4a4_gemv_mx32_nt4",
        rows: 64,
        base: "w4a4_gemv_mx32",
        max_m: 32,
    },
    Twin {
        name: "w4a4_gemv_mx64",
        rows: 16,
        base: "w4a4_gemv_mx32",
        max_m: 64,
    },
    Twin {
        name: "w4a4_gemv_mx64_nt2",
        rows: 32,
        base: "w4a4_gemv_mx32",
        max_m: 64,
    },
];

/// 2026-09-25: A persistent entry, the one-tile entry it must match, its widest M.
const PERSISTENT: [(&str, &str, u32); 2] = [
    ("w4a4_gemv_mx16_ps", "w4a4_gemv_mx16", 16),
    ("w4a4_gemv_mx32_ps", "w4a4_gemv_mx32", 32),
];

struct Case<'a> {
    g: &'a dyn GpuBackend,
    aq: DevicePtr,
    a_scale: DevicePtr,
    a_gs: DevicePtr,
    wq: DevicePtr,
    ws: DevicePtr,
    c: DevicePtr,
    n: u32,
    k: u32,
}

impl Case<'_> {
    /// 2026-09-25: Run `kh` into an `[(m+1) x n]` output whose row m is a 0xFFFF sentinel.
    fn run(&self, kh: KernelHandle, rows_per_cta: u32, m: u32) -> Result<Vec<u16>> {
        self.run_rows(kh, rows_per_cta, m, &[(0, m)])
    }

    /// 2026-09-25: Run `kh` over each `(first row, rows)` span of an m-row activation.
    fn run_rows(
        &self,
        kh: KernelHandle,
        rows_per_cta: u32,
        m: u32,
        spans: &[(u32, u32)],
    ) -> Result<Vec<u16>> {
        self.launch(kh, div_ceil(self.n, rows_per_cta), 0, None, m, spans)
    }

    /// 2026-09-25: Run a persistent entry serving up to `max_m` rows at staging
    /// depth `sst`. The shared memory follows the entry's column blocks
    /// (`ps_column_blocks(max_m)`), not `m`'s, because mx32_ps is also checked
    /// below 17 rows.
    fn run_ps(&self, kh: KernelHandle, sms: u32, sst: u32, m: u32, max_m: u32) -> Result<Vec<u16>> {
        let smem = ps_smem_bytes(ps_column_blocks(max_m), sst);
        self.launch(kh, sms, smem, Some(sst), m, &[(0, m)])
    }

    fn launch(
        &self,
        kh: KernelHandle,
        grid: u32,
        smem: u32,
        sst: Option<u32>,
        m: u32,
        spans: &[(u32, u32)],
    ) -> Result<Vec<u16>> {
        let bytes = (m as usize + 1) * self.n as usize * 2;
        self.g.copy_h2d(&vec![0xFFu8; bytes], self.c)?;
        for &(r0, rows) in spans {
            let r0 = r0 as usize;
            let launch = KernelLaunch::new(self.g, kh)
                .grid([grid, 1, 1])
                .block([256, 1, 1])
                .shared_mem(smem)
                .arg_ptr(self.aq.offset(r0 * self.k as usize / 2))
                .arg_ptr(self.a_scale.offset(r0 * self.k as usize / 16))
                .arg_ptr(self.a_gs.offset(r0 * 4))
                .arg_ptr(self.wq)
                .arg_ptr(self.ws)
                .arg_f32(SCALE2)
                .arg_ptr(self.c.offset(r0 * self.n as usize * 2))
                .arg_u32(rows)
                .arg_u32(self.n)
                .arg_u32(self.k);
            match sst {
                Some(sst) => launch.arg_u32(sst).launch(0)?,
                None => launch.launch(0)?,
            }
        }
        self.g.synchronize(0)?;
        let mut out = vec![0u8; bytes];
        self.g.copy_d2h(self.c, &mut out)?;
        Ok(out
            .chunks_exact(2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
            .collect())
    }
}

/// 2026-09-25: Elements of `twin` that differ from `base`, plus sentinel-row writes.
fn mismatches(base: &[u16], twin: &[u16], m: u32, n: u32) -> usize {
    let body = (m * n) as usize;
    let diff = base[..body]
        .iter()
        .zip(&twin[..body])
        .filter(|(a, b)| a != b)
        .count();
    diff + twin[body..].iter().filter(|&&v| v != 0xFFFF).count()
}

fn main() -> Result<()> {
    let backend = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let g: &dyn GpuBackend = &backend;
    let h = |f: &str| g.kernel(MODULE, f).ok();
    let quant = h("w4a4_quant_rows");
    let resolved: Vec<_> = TWINS.iter().map(|t| (h(t.base), h(t.name))).collect();
    let Some(quant) = quant else {
        eprintln!("ARMED: w4a4_quant_rows not loaded");
        std::process::exit(2);
    };
    let persistent: Vec<_> = PERSISTENT.iter().map(|(p, b, _)| (h(b), h(p))).collect();
    if resolved
        .iter()
        .chain(&persistent)
        .any(|(b, t)| b.is_none() || t.is_none())
    {
        eprintln!("ARMED: a one-tile entry, twin or persistent entry is not loaded");
        std::process::exit(2);
    }
    let sms = g.sm_count()?;

    let (k_max, n_max) = (17408usize, 17408usize);
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    let act: Vec<u8> = (0..MAX_M as usize * k_max)
        .flat_map(|_| {
            let r = rng.next();
            let mag = if r.is_multiple_of(3) { 40.0 } else { 3.0 };
            let v = ((r >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * mag;
            half_bf16(v as f32).to_le_bytes()
        })
        .collect();
    let a = g.alloc(act.len())?;
    g.copy_h2d(&act, a)?;
    let aq = g.alloc(MAX_M as usize * k_max / 2)?;
    let a_scale = g.alloc(MAX_M as usize * k_max / 16)?;
    let a_gs = g.alloc(MAX_M as usize * 4)?;
    let c = g.alloc((MAX_M as usize + 1) * n_max * 2)?;

    let mut pass = true;
    let mut cases = 0usize;
    let mut control_tripped = false;
    for (label, n, k) in SHAPES {
        let wq_host: Vec<u8> = (0..n as usize * k as usize / 2)
            .map(|_| rng.next() as u8)
            .collect();
        // 2026-09-25: Finite positive E4M3 group scales, bytes 0x28..=0x3F (0.25 to 1.875).
        let ws_host: Vec<u8> = (0..n as usize * k as usize / 16)
            .map(|_| 0x28 + (rng.next() % 24) as u8)
            .collect();
        let wq = g.alloc(wq_host.len())?;
        g.copy_h2d(&wq_host, wq)?;
        let ws = g.alloc(ws_host.len())?;
        g.copy_h2d(&ws_host, ws)?;
        let case = Case {
            g,
            aq,
            a_scale,
            a_gs,
            wq,
            ws,
            c,
            n,
            k,
        };
        let mut bad = 0usize;
        for m in 1..=MAX_M {
            KernelLaunch::new(g, quant)
                .grid([m, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(a)
                .arg_ptr(aq)
                .arg_ptr(a_scale)
                .arg_ptr(a_gs)
                .arg_u32(k)
                .launch(0)?;
            for (t, (base, twin)) in TWINS.iter().zip(&resolved) {
                if m > t.max_m {
                    continue;
                }
                let (base, twin) = (base.expect("armed"), twin.expect("armed"));
                // 2026-09-25: mx32 serves at most 32 rows, so the reference is split
                // above that.
                let want = if m <= 32 {
                    case.run(base, 16, m)?
                } else {
                    case.run_rows(base, 16, m, &[(0, 32), (32, m - 32)])?
                };
                let got = case.run(twin, t.rows, m)?;
                let d = mismatches(&want, &got, m, n);
                cases += 1;
                if d != 0 {
                    bad += 1;
                    pass = false;
                    println!("  DIFF {label} M={m} {}: {d} elements", t.name);
                }
                if m == t.max_m {
                    let mut flipped = wq_host[..64].to_vec();
                    flipped[0] ^= 0x77;
                    g.copy_h2d(&flipped, wq)?;
                    let got = case.run(twin, t.rows, m)?;
                    g.copy_h2d(&wq_host[..64], wq)?;
                    if mismatches(&want, &got, m, n) != 0 {
                        control_tripped = true;
                    } else {
                        pass = false;
                        println!("  CONTROL A did not trip: {label} {}", t.name);
                    }
                }
            }
            for ((name, _, max_m), (base, ps)) in PERSISTENT.iter().zip(&persistent) {
                if m > *max_m {
                    continue;
                }
                let (base, ps) = (base.expect("armed"), ps.expect("armed"));
                let want = case.run(base, 16, m)?;
                let full = ps_stripe_chunks(k);
                let mut depths = vec![0, 1, full];
                depths.retain(|&d| ps_smem_bytes(ps_column_blocks(*max_m), d) <= PS_SMEM_MAX);
                depths.dedup();
                for sst in depths {
                    let got = case.run_ps(ps, sms, sst, m, *max_m)?;
                    let d = mismatches(&want, &got, m, n);
                    cases += 1;
                    if d != 0 {
                        bad += 1;
                        pass = false;
                        println!("  DIFF {label} M={m} {name} sst={sst}: {d} elements");
                    }
                }
                if m == *max_m {
                    let mut flipped = wq_host[..64].to_vec();
                    flipped[0] ^= 0x77;
                    g.copy_h2d(&flipped, wq)?;
                    let got = case.run_ps(ps, sms, 0, m, *max_m)?;
                    g.copy_h2d(&wq_host[..64], wq)?;
                    if mismatches(&want, &got, m, n) != 0 {
                        control_tripped = true;
                    } else {
                        pass = false;
                        println!("  CONTROL A did not trip: {label} {name}");
                    }
                }
            }
        }
        println!("{label} N={n:5} K={k:5}: {bad} differing (M, kernel) cases");
        g.free(wq)?;
        g.free(ws)?;
    }
    pass &= control_tripped;
    println!(
        "{cases} (shape, M, kernel[, depth]) cases vs the one-tile kernels; control A tripped: {control_tripped}; {}",
        if pass { "PASS" } else { "FAIL" }
    );
    std::process::exit(if pass { 0 } else { 1 });
}

/// 2026-09-25: f32 to BF16 bits, round to nearest even (no NaN handling).
fn half_bf16(x: f32) -> u16 {
    let u = x.to_bits();
    ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) as u16
}
