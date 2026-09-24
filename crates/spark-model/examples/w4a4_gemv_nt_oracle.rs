// SPDX-License-Identifier: AGPL-3.0-only
//! ORACLE for the activation-reuse twins of the W4A4 small-M GEMV
//! (`w4a4_gemv_mx.cu`: `w4a4_gemv_mx16_nt2`, `w4a4_gemv_mx32_nt4`) and its
//! persistent activation-staged entries (`w4a4_gemv_mx16_ps`,
//! `w4a4_gemv_mx32_ps`).
//!
//! `ops::w4a4_proj` routes 9..=32-row launches to the twins by default
//! (`METRALE_W4A4_MX_NT=4`). That needs no opt-in only because the twins are
//! BIT-IDENTICAL to the one-tile kernels they replace. This decides it on the
//! production loader path (the PTX the serve JIT-compiles):
//!
//!   1. ARMED: the quantiser, the one-tile entries and every twin resolve.
//!   2. At every real dense-27B projection shape and every M the twin serves
//!      (1..=16 for mx16_nt2, 1..=32 for mx32_nt4), the twin's [M, N] output
//!      equals the one-tile kernel's bit for bit, and row M (a 0xFFFF
//!      sentinel) is never written.
//!   3. The same for each persistent entry, launched through
//!      `ops::w4a4_proj`'s launch contract (grid = #SMs, its shared memory),
//!      at every staging depth that fits: 0, 1 and the whole stripe. Every
//!      case is a fresh launch, so a tile counter that failed to reset would
//!      show as unwritten output.
//!
//! KNOWN-BAD CONTROL (the gate must FAIL it, or it proves nothing):
//!   A. one weight byte of row 0 flipped between the two runs: check 2 must
//!      trip.
//!
//! Exit: 0 pass, 1 any case or the control misbehaved, 2 kernels not loaded.
//!
//! Run (GB10):
//!   cargo run -p spark-model --release --features cuda,gpu-examples \
//!     --example w4a4_gemv_nt_oracle

use anyhow::Result;
use spark_model::layers::ops::w4a4_proj::{
    PS_SMEM_MAX, ps_column_blocks, ps_smem_bytes, ps_stripe_chunks,
};
use spark_runtime::cuda_backend::MetraleCudaBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

const MODULE: &str = "w4a4_gemv_mx";
const SCALE2: f32 = 0.37;
const MAX_M: u32 = 32;
/// (name, N, K) of every dense-27B projection the W4A4 path serves.
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

/// A twin, its rows per CTA, the one-tile entry it must match, and the
/// widest M either serves (checked at every M from 1).
struct Twin {
    name: &'static str,
    rows: u32,
    base: &'static str,
    max_m: u32,
}

const TWINS: [Twin; 2] = [
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
];

/// A persistent entry, the one-tile entry it must match, its widest M.
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
    /// Output [(m+1) x n] with row m as a 0xFFFF sentinel.
    fn run(&self, kh: KernelHandle, rows_per_cta: u32, m: u32) -> Result<Vec<u16>> {
        self.launch(kh, div_ceil(self.n, rows_per_cta), 0, None, m)
    }

    /// A persistent entry serving up to `max_m` rows at staging depth `sst`,
    /// per the launch contract. The shared memory follows the ENTRY's column
    /// blocks, not `m`'s: mx32_ps is also checked below 17 rows.
    fn run_ps(&self, kh: KernelHandle, sms: u32, sst: u32, m: u32, max_m: u32) -> Result<Vec<u16>> {
        let smem = ps_smem_bytes(ps_column_blocks(max_m), sst);
        self.launch(kh, sms, smem, Some(sst), m)
    }

    fn launch(
        &self,
        kh: KernelHandle,
        grid: u32,
        smem: u32,
        sst: Option<u32>,
        m: u32,
    ) -> Result<Vec<u16>> {
        let bytes = (m as usize + 1) * self.n as usize * 2;
        self.g.copy_h2d(&vec![0xFFu8; bytes], self.c)?;
        let launch = KernelLaunch::new(self.g, kh)
            .grid([grid, 1, 1])
            .block([256, 1, 1])
            .shared_mem(smem)
            .arg_ptr(self.aq)
            .arg_ptr(self.a_scale)
            .arg_ptr(self.a_gs)
            .arg_ptr(self.wq)
            .arg_ptr(self.ws)
            .arg_f32(SCALE2)
            .arg_ptr(self.c)
            .arg_u32(m)
            .arg_u32(self.n)
            .arg_u32(self.k);
        match sst {
            Some(sst) => launch.arg_u32(sst).launch(0)?,
            None => launch.launch(0)?,
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

/// Elements of `twin` that differ from `base`, plus sentinel-row writes.
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
        // Finite E4M3 group scales, 2^-2 .. 2^1.
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
                let want = case.run(base, 16, m)?;
                let got = case.run(twin, t.rows, m)?;
                let d = mismatches(&want, &got, m, n);
                cases += 1;
                if d != 0 {
                    bad += 1;
                    pass = false;
                    println!("  DIFF {label} M={m} {}: {d} elements", t.name);
                }
                // Control A, once per shape at the widest M: a flipped weight
                // byte in row 0 must make the twin differ.
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

/// f32 -> bf16 bits, round to nearest even.
fn half_bf16(x: f32) -> u16 {
    let u = x.to_bits();
    ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) as u16
}
