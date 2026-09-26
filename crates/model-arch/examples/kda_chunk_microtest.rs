// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Numeric gate for the GLM-5.3 KDA chunked-prefill kernels (`kda_chunk_prepare`
//! and `kda_chunk_scan`), against the goldens and against the decode kernel
//! `kda_recurrent_decode_f32`.
//!
//! Owner: model-arch examples (GLM-5.3 KDA kernels).
//! Invariants:
//! - The run fails unless every section (A: HF golden, B: chunk against sequential decode,
//!   C: production geometry, D: adversarial mutants) passes; latency is timed only after that.
//!
//! Two kernels: `kda_chunk_prepare` works on each (chunk, head) independently, and
//! `kda_chunk_scan` walks the chunks of a head in order, carrying the recurrent state.
//!
//! The central check (B): `kda_chunk` over `T` tokens must match `T` sequential decode steps
//! for the same q/k, v, gate, beta and initial state; it needs no golden. Section D checks
//! that each `Mutation` of a CPU chunked formulation (per-head decay, decay sign flip, flipped
//! triangle, state updated before the output is read) moves away from the GPU output, and
//! that a poisoned pad tail changes nothing.
//!
//! Every launch asserts its dynamic shared memory is at most `SMEM_CEILING` (48 KiB), a limit
//! this harness sets; see `smem_prepare` and `smem_scan`.
//!
//!   cargo run -p metrale-model-arch --release --example kda_chunk_microtest \
//!       --features cuda,gpu-examples

use anyhow::{Result, bail};
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use serde_json::Value;

#[path = "common/kda_chunk_mutant.rs"]
pub(crate) mod kda_chunk_mutant;
use kda_chunk_mutant::*;

#[path = "common/kda_chunk_cases.rs"]
pub(crate) mod kda_chunk_cases;
use kda_chunk_cases::*;

pub(crate) const FIXTURE: &str = include_str!("../src/glm5next_kda_ref/kda_golden.json");
#[path = "common/golden.rs"]
pub(crate) mod golden;

pub(crate) static PROD: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
    golden::load(
        "crates/model-arch/src/glm5next_kda_ref/kda_chunk_prod_golden.json",
        "gen_kda_chunk_prod_golden.py",
    )
});

pub(crate) const PROD_H: usize = 64;
pub(crate) const PROD_D: usize = 128;
pub(crate) const BLOCK: u32 = 128;

/// 2026-09-25: The harness's limit on dynamic shared memory per launch. The backend itself opts
/// in above 48 KiB when a launch asks for more.
pub(crate) const SMEM_CEILING: usize = 49152;

pub(crate) const MAX_ABS: f64 = 2.0e-6;
pub(crate) const MAX_REL: f64 = 1.0e-4;
pub(crate) const MAX_FLOOR_RATIO: f64 = 2.0;

pub(crate) fn smem_prepare(c: usize, d: usize) -> usize {
    (c * d + c * c + c) * 4
}
pub(crate) fn smem_scan(c: usize, d: usize) -> usize {
    (2 * c * d + c * c) * 4
}

pub(crate) fn up_f32(g: &dyn GpuBackend, d: &[f32]) -> Result<DevicePtr> {
    let b: Vec<u8> = d.iter().flat_map(|x| x.to_le_bytes()).collect();
    let p = g.alloc(b.len().max(1))?;
    g.copy_h2d(&b, p)?;
    Ok(p)
}
pub(crate) fn down_f32(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<f32>> {
    let mut b = vec![0u8; n * 4];
    g.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}
pub(crate) fn arr(v: &Value, section: &str, name: &str) -> Vec<f32> {
    v[section][name]["data"]
        .as_array()
        .unwrap_or_else(|| panic!("missing {section}.{name}"))
        .iter()
        .map(|x| x.as_f64().expect("numeric") as f32)
        .collect()
}

#[derive(Default, Clone)]
pub(crate) struct Err2 {
    pub(crate) max_abs: f64,
    pub(crate) max_rel: f64,
    pub(crate) exact: usize,
    pub(crate) total: usize,
}

pub(crate) const REL_GUARD_FRACTION: f64 = 1e-3;

pub(crate) fn compare(got: &[f32], want: &[f32]) -> Err2 {
    assert_eq!(
        got.len(),
        want.len(),
        "length {} vs {}",
        got.len(),
        want.len()
    );
    let mut e = Err2 {
        total: got.len(),
        ..Default::default()
    };
    let peak = want.iter().fold(0.0f64, |m, w| m.max((*w as f64).abs()));
    let guard = (peak * REL_GUARD_FRACTION).max(1e-30);
    for (g, w) in got.iter().zip(want) {
        let d = (*g as f64 - *w as f64).abs();
        e.max_abs = e.max_abs.max(d);
        if (*w as f64).abs() > guard {
            e.max_rel = e.max_rel.max(d / (*w as f64).abs());
        }
        if g.to_bits() == w.to_bits() {
            e.exact += 1;
        }
    }
    e
}
pub(crate) fn report(label: &str, e: &Err2) {
    println!(
        "  {label:<52} max_abs={:.3e} max_rel={:.3e} exact={}/{}",
        e.max_abs, e.max_rel, e.exact, e.total
    );
}
pub(crate) fn within(e: &Err2) -> bool {
    e.max_abs <= MAX_ABS && e.max_rel <= MAX_REL
}

/// 2026-09-25: Chunked prefill and the sequential recurrence are algebraically equal but round
/// differently: the chunked form accumulates through A, u, w and a per-chunk state jump, the
/// recurrence token by token. That difference is measured on the CPU reference over the same
/// inputs (`floor`), and the GPU error must stay within `MAX_FLOOR_RATIO` times it, or within
/// `MAX_ABS` / `MAX_REL`.
pub(crate) fn within_floor(e: &Err2, floor: &Err2) -> bool {
    e.max_abs <= MAX_ABS.max(floor.max_abs * MAX_FLOOR_RATIO)
        && e.max_rel <= MAX_REL.max(floor.max_rel * MAX_FLOOR_RATIO)
}
pub(crate) fn checksum(s: &[f32]) -> f64 {
    s.iter()
        .enumerate()
        .map(|(i, v)| *v as f64 * (i as f64 + 1.0))
        .sum()
}

pub(crate) struct Lcg(u64);
impl Lcg {
    fn unit(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (((self.0 >> 40) as f32) / ((1u32 << 24) as f32)) * 2.0 - 1.0
    }
    fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.unit()).collect()
    }
}

pub(crate) struct Gpu<'a> {
    pub(crate) g: &'a dyn GpuBackend,
    pub(crate) prepare: KernelHandle,
    pub(crate) scan: KernelHandle,
    pub(crate) recurrent: KernelHandle,
}

/// 2026-09-25: q, k, v and gate are `[T, H, D]`, token-major, and beta is `[T, H]`, as the
/// kernels read them.
pub(crate) struct Inputs {
    pub(crate) q: Vec<f32>,
    pub(crate) k: Vec<f32>,
    pub(crate) v: Vec<f32>,
    pub(crate) gate: Vec<f32>,
    pub(crate) beta: Vec<f32>,
}

impl Gpu<'_> {
    /// 2026-09-25: Chunked prefill. `pad_fill` lets a test poison the padded tail to show it
    /// cannot reach a valid output or the final state.
    fn chunk(
        &self,
        i: &Inputs,
        t: usize,
        h: usize,
        d: usize,
        c: usize,
        state: &mut Vec<f32>,
        pad_fill: f32,
    ) -> Result<Vec<f32>> {
        let g = self.g;
        let nchunks = t.div_ceil(c);
        let tp = nchunks * c;
        let (sp, ss) = (smem_prepare(c, d), smem_scan(c, d));
        assert!(
            sp <= SMEM_CEILING && ss <= SMEM_CEILING,
            "chunk={c} needs {sp}/{ss} B shared, ceiling {SMEM_CEILING}"
        );

        let pad_vec = |src: &[f32], per: usize| -> Vec<f32> {
            let mut o = vec![pad_fill; tp * per];
            o[..t * per].copy_from_slice(&src[..t * per]);
            o
        };
        let (pq, pk, pv, pg) = (
            pad_vec(&i.q, h * d),
            pad_vec(&i.k, h * d),
            pad_vec(&i.v, h * d),
            pad_vec(&i.gate, h * d),
        );
        let pb = pad_vec(&i.beta, h);

        let (dq, dk, dv, dg, db) = (
            up_f32(g, &pq)?,
            up_f32(g, &pk)?,
            up_f32(g, &pv)?,
            up_f32(g, &pg)?,
            up_f32(g, &pb)?,
        );
        let n = tp * h * d;
        let (dgc, du, dw) = (g.alloc(n * 4)?, g.alloc(n * 4)?, g.alloc(n * 4)?);
        let dout = up_f32(g, &vec![0.0f32; n])?;
        let dstate = up_f32(g, state)?;

        KernelLaunch::new(g, self.prepare)
            .grid([nchunks as u32, h as u32, 1])
            .block([BLOCK, 1, 1])
            .shared_mem(sp as u32)
            .arg_ptr(dk)
            .arg_ptr(dv)
            .arg_ptr(dg)
            .arg_ptr(db)
            .arg_ptr(dgc)
            .arg_ptr(du)
            .arg_ptr(dw)
            .arg_u32(h as u32)
            .arg_u32(d as u32)
            .arg_u32(c as u32)
            .arg_u32(t as u32)
            .launch(0)?;

        KernelLaunch::new(g, self.scan)
            .grid([h as u32, 1, 1])
            .block([BLOCK, 1, 1])
            .shared_mem(ss as u32)
            .arg_ptr(dq)
            .arg_ptr(dk)
            .arg_ptr(dgc)
            .arg_ptr(du)
            .arg_ptr(dw)
            .arg_ptr(dstate)
            .arg_ptr(dout)
            .arg_u32(h as u32)
            .arg_u32(d as u32)
            .arg_u32(c as u32)
            .arg_u32(nchunks as u32)
            .arg_u32(t as u32)
            .arg_f32(1.0 / (d as f32).sqrt())
            .launch(0)?;
        g.synchronize(0)?;

        *state = down_f32(g, dstate, h * d * d)?;
        let full = down_f32(g, dout, n)?;
        Ok(full[..t * h * d].to_vec())
    }

    /// 2026-09-25: `T` sequential decode steps through `kda_recurrent_decode_f32`.
    fn recurrent(
        &self,
        i: &Inputs,
        t: usize,
        h: usize,
        d: usize,
        state: &mut Vec<f32>,
    ) -> Result<Vec<f32>> {
        let g = self.g;
        let per = h * d;
        let mut out = Vec::with_capacity(t * per);
        for tok in 0..t {
            let (a, b) = (tok * per, (tok + 1) * per);
            let (dq, dk, dv) = (
                up_f32(g, &i.q[a..b])?,
                up_f32(g, &i.k[a..b])?,
                up_f32(g, &i.v[a..b])?,
            );
            let dg = up_f32(g, &i.gate[a..b])?;
            let db = up_f32(g, &i.beta[tok * h..(tok + 1) * h])?;
            let dstate = up_f32(g, state)?;
            let dout = g.alloc(per * 4)?;
            KernelLaunch::new(g, self.recurrent)
                .grid([h as u32, 1, 1])
                .block([BLOCK.min(d as u32), 1, 1])
                .shared_mem((3 * d * 4) as u32)
                .arg_ptr(dq)
                .arg_ptr(dk)
                .arg_ptr(dv)
                .arg_ptr(dg)
                .arg_ptr(db)
                .arg_ptr(dstate)
                .arg_ptr(dout)
                .arg_u32(h as u32)
                .arg_u32(d as u32)
                .arg_f32(1.0 / (d as f32).sqrt())
                .launch(0)?;
            g.synchronize(0)?;
            *state = down_f32(g, dstate, h * d * d)?;
            out.extend_from_slice(&down_f32(g, dout, per)?);
        }
        Ok(out)
    }
}

fn main() -> Result<()> {
    let g = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let gpu: &dyn GpuBackend = &g;
    let d = Gpu {
        g: gpu,
        prepare: gpu.kernel("kda_chunk", "kda_chunk_prepare")?,
        scan: gpu.kernel("kda_chunk", "kda_chunk_scan")?,
        recurrent: gpu.kernel("kda_recurrent", "kda_recurrent_decode_f32")?,
    };
    println!("kda_chunk: kda_chunk_prepare + kda_chunk_scan resolved from PTX (no fallback)");
    println!(
        "shared-memory ceiling {SMEM_CEILING} B (no cuFuncSetAttribute opt-in in this backend)"
    );
    for c in [8usize, 16, 32, 64] {
        let (p, s) = (smem_prepare(c, PROD_D), smem_scan(c, PROD_D));
        println!(
            "  chunk={c:<3} D=128  prepare={p:>6} B  scan={s:>6} B  {}",
            if p.max(s) <= SMEM_CEILING {
                "ok"
            } else {
                "EXCEEDS -> unusable"
            }
        );
    }

    println!("\nA — HF golden, fixture H=2 D=4 T=6");
    let a = test_a(&d)?;
    println!("\nB — chunk == T sequential decode steps (production H=64 D=128)");
    let b = test_b(&d)?;
    println!("\nC — production geometry vs HF golden");
    let c = test_c(&d)?;
    println!("\nD — adversarial");
    let dd = test_d(&d)?;

    if a && b && c && dd {
        println!("\nisolated latency (correctness passed first)");
        latency(&d)?;
        println!("\nPASS");
        Ok(())
    } else {
        bail!("FAIL — see the lines marked FAIL above");
    }
}
