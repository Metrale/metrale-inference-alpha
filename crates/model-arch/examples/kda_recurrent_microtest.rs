// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Numeric gate for the GLM-5.3 KDA recurrent decode kernels
//! (`kda_recurrent_decode_f32`, `kda_recurrent_decode_bf16`): one token at a time, carrying
//! `state[H, D, D]`.
//!
//! Owner: model-arch examples (GLM-5.3 KDA kernels).
//! Invariants:
//! - The run fails unless the fixture, continuation, both production and adversarial checks
//!   pass.
//!
//! The kernel reads the `kda_gate` output, a log-decay in `[lower_bound, 0]`, and
//! exponentiates it per key channel; the CPU reference does the same
//! (`gate[base + kd].exp()`). The GDN path differs on both counts: `compute_gdn_gates` stores
//! the decay already exponentiated, one value per head, and `gated_delta_rule_decode` uses it
//! as is. Mixing the two conventions gives a double exp or a missing one.
//!
//! Oracles:
//! 1. `kda_golden.json`, H=2, D=4, zero initial state: `core_recurrent`, `state_recurrent`,
//!    `state_after_prefill4`, `core_split_prefill_then_decode`.
//! 2. `kda_recurrent_prod_golden.json`, H=64, D=128, three decode steps from a non-zero state:
//!    the full output per step, a state sample at a prime stride, and an FP64 index-weighted
//!    checksum of the whole state per step.
//! 3. `glm5next_kda_ref::kda_recurrent_prenorm`, the CPU reference.
//!
//! On the FP32 production path each step is gated on the ratio of the GPU error to the CPU
//! reference's error against the same golden, as well as on absolute and relative bounds.
//!
//!   cargo run -p metrale-model-arch --release --example kda_recurrent_microtest \
//!       --features cuda,gpu-examples

use anyhow::{Result, bail};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use metrale_model_arch::glm5next_kda_ref::{KdaDims, kda_recurrent_prenorm};
use serde_json::Value;

#[path = "common/kda_recurrent_checks.rs"]
pub(crate) mod kda_recurrent_checks;
use kda_recurrent_checks::*;

pub(crate) const FIXTURE: &str = include_str!("../src/glm5next_kda_ref/kda_golden.json");
#[path = "common/golden.rs"]
pub(crate) mod golden;

pub(crate) static PROD: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
    golden::load(
        "crates/model-arch/src/glm5next_kda_ref/kda_recurrent_prod_golden.json",
        "gen_kda_recurrent_prod_golden.py",
    )
});

pub(crate) const PROD_H: usize = 64;
pub(crate) const PROD_D: usize = 128;
pub(crate) const BLOCK: u32 = 128;

pub(crate) const MAX_ABS: f64 = 2.0e-6;
/// 2026-09-25: Relative bound over elements above the magnitude guard in `compare`.
pub(crate) const MAX_REL: f64 = 1.0e-4;
/// 2026-09-25: On the FP32 production path, the GPU error may be at most this multiple of the
/// CPU reference's error against the same golden.
pub(crate) const MAX_FLOOR_RATIO: f64 = 2.0;

pub(crate) fn up_f32(g: &dyn GpuBackend, d: &[f32]) -> Result<DevicePtr> {
    let b: Vec<u8> = d.iter().flat_map(|x| x.to_le_bytes()).collect();
    let p = g.alloc(b.len().max(1))?;
    g.copy_h2d(&b, p)?;
    Ok(p)
}

pub(crate) fn up_bf16(g: &dyn GpuBackend, d: &[f32]) -> Result<DevicePtr> {
    let b: Vec<u8> = d
        .iter()
        .flat_map(|x| bf16::from_f32(*x).to_bits().to_le_bytes())
        .collect();
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

/// 2026-09-25: Relative error is scored only over elements within three decades of the
/// tensor's peak magnitude: near zero any absolute error is a huge relative one, which measures
/// the float grid, not the kernel.
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

pub(crate) fn report(label: &str, e: &Err2, dtype: &str) {
    println!(
        "  {label:<48} {dtype:<5} max_abs={:.3e} max_rel={:.3e} exact={}/{}",
        e.max_abs, e.max_rel, e.exact, e.total
    );
}

pub(crate) fn within(e: &Err2) -> bool {
    e.max_abs <= MAX_ABS && e.max_rel <= MAX_REL
}

pub(crate) fn ratio(e: &Err2, floor: &Err2) -> f64 {
    if floor.max_abs > 0.0 {
        e.max_abs / floor.max_abs
    } else if e.max_abs == 0.0 {
        1.0
    } else {
        f64::INFINITY
    }
}

/// 2026-09-25: FP64 index-weighted checksum over the full state, so an indexing or ordering
/// error changes it; the golden stores one number per step instead of the 4 MiB state.
pub(crate) fn checksum(s: &[f32]) -> f64 {
    s.iter()
        .enumerate()
        .map(|(i, v)| *v as f64 * (i as f64 + 1.0))
        .sum()
}

pub(crate) struct Rec<'a> {
    pub(crate) g: &'a dyn GpuBackend,
    pub(crate) k_f32: KernelHandle,
    pub(crate) k_bf16: KernelHandle,
}

impl Rec<'_> {
    /// 2026-09-25: One decode token; `state` is replaced by the device state afterwards.
    #[allow(clippy::too_many_arguments)]
    fn step(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        gate: &[f32],
        beta: &[f32],
        state: &mut Vec<f32>,
        h: usize,
        d: usize,
        bf16_inputs: bool,
    ) -> Result<Vec<f32>> {
        let g = self.g;
        let (dq, dk, dv) = if bf16_inputs {
            (up_bf16(g, q)?, up_bf16(g, k)?, up_bf16(g, v)?)
        } else {
            (up_f32(g, q)?, up_f32(g, k)?, up_f32(g, v)?)
        };
        let dgate = up_f32(g, gate)?;
        let dbeta = up_f32(g, beta)?;
        let dstate = up_f32(g, state)?;
        let dout = g.alloc(h * d * 4)?;
        let scale = 1.0f32 / (d as f32).sqrt();

        KernelLaunch::new(g, if bf16_inputs { self.k_bf16 } else { self.k_f32 })
            .grid([h as u32, 1, 1])
            .block([BLOCK.min(d as u32), 1, 1])
            .shared_mem((3 * d * 4) as u32)
            .arg_ptr(dq)
            .arg_ptr(dk)
            .arg_ptr(dv)
            .arg_ptr(dgate)
            .arg_ptr(dbeta)
            .arg_ptr(dstate)
            .arg_ptr(dout)
            .arg_u32(h as u32)
            .arg_u32(d as u32)
            .arg_f32(scale)
            .launch(0)?;
        g.synchronize(0)?;

        *state = down_f32(g, dstate, h * d * d)?;
        down_f32(g, dout, h * d)
    }
}

/// 2026-09-25: Decode from a zero state over the fixture's tokens. After every token the output
/// is checked against the golden and the state against the CPU reference; the final state is
/// checked against the golden.
pub(crate) fn check_fixture(rec: &Rec) -> Result<bool> {
    let v: Value = serde_json::from_str(FIXTURE)?;
    let f = &v["fixture"];
    let (h, d, t) = (
        f["heads"].as_u64().unwrap() as usize,
        f["head_dim"].as_u64().unwrap() as usize,
        f["tokens"].as_u64().unwrap() as usize,
    );
    let dims1 = KdaDims {
        hidden: 0,
        heads: h,
        head_dim: d,
        tokens: 1,
    };
    let (q, k, vv) = (
        arr(&v, "outputs", "q_l2"),
        arr(&v, "outputs", "k_l2"),
        arr(&v, "inputs", "v_in"),
    );
    let (gate, beta) = (arr(&v, "outputs", "gate"), arr(&v, "outputs", "beta"));
    let want_o = arr(&v, "outputs", "core_recurrent");
    let per = h * d;

    let mut state = vec![0.0f32; h * d * d];
    let mut cpu_state = vec![0.0f32; h * d * d];
    let mut all_o = Vec::new();
    let mut ok = true;

    for tok in 0..t {
        let (a, b) = (tok * per, (tok + 1) * per);
        let o = rec.step(
            &q[a..b],
            &k[a..b],
            &vv[a..b],
            &gate[a..b],
            &beta[tok * h..(tok + 1) * h],
            &mut state,
            h,
            d,
            false,
        )?;
        let cpu_o = kda_recurrent_prenorm(
            &q[a..b],
            &k[a..b],
            &vv[a..b],
            &gate[a..b],
            &beta[tok * h..(tok + 1) * h],
            dims1,
            &mut cpu_state,
        );
        let eo = compare(&o, &want_o[a..b]);
        let es = compare(&state, &cpu_state);
        let ec = compare(&o, &cpu_o);
        println!(
            "  token {tok}: o vs HF max_abs={:.3e}  o vs CPU-ref max_abs={:.3e}  state vs CPU-ref max_abs={:.3e}",
            eo.max_abs, ec.max_abs, es.max_abs
        );
        ok &= within(&eo) && within(&es);
        all_o.extend_from_slice(&o);
    }

    report(
        "fixture: all tokens o vs HF",
        &compare(&all_o, &want_o),
        "f32",
    );
    let ef = compare(&state, &arr(&v, "outputs", "state_recurrent"));
    report("fixture: final state vs HF", &ef, "f32");
    Ok(ok && within(&ef))
}

/// 2026-09-25: Decode the tokens after the fourth, starting from the golden's own
/// `state_after_prefill4`.
pub(crate) fn check_continuation(rec: &Rec) -> Result<bool> {
    let v: Value = serde_json::from_str(FIXTURE)?;
    let f = &v["fixture"];
    let (h, d, t) = (
        f["heads"].as_u64().unwrap() as usize,
        f["head_dim"].as_u64().unwrap() as usize,
        f["tokens"].as_u64().unwrap() as usize,
    );
    let (q, k, vv) = (
        arr(&v, "outputs", "q_l2"),
        arr(&v, "outputs", "k_l2"),
        arr(&v, "inputs", "v_in"),
    );
    let (gate, beta) = (arr(&v, "outputs", "gate"), arr(&v, "outputs", "beta"));
    let split = 4usize;
    let per = h * d;

    let mut state = arr(&v, "outputs", "state_after_prefill4");
    let want = arr(&v, "outputs", "core_split_prefill_then_decode");
    let mut ok = true;
    for tok in split..t {
        let (a, b) = (tok * per, (tok + 1) * per);
        let o = rec.step(
            &q[a..b],
            &k[a..b],
            &vv[a..b],
            &gate[a..b],
            &beta[tok * h..(tok + 1) * h],
            &mut state,
            h,
            d,
            false,
        )?;
        let e = compare(&o, &want[a..b]);
        report(&format!("continuation token {tok} vs HF"), &e, "f32");
        ok &= within(&e);
    }
    let ef = compare(&state, &arr(&v, "outputs", "state_recurrent"));
    report("continuation final state vs HF", &ef, "f32");
    Ok(ok && within(&ef))
}

/// 2026-09-25: The golden generator's LCG: 64-bit integer ops and an exact division by 2^24,
/// so both sides produce identical inputs.
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

fn main() -> Result<()> {
    let g = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let gpu: &dyn GpuBackend = &g;
    let rec = Rec {
        g: gpu,
        k_f32: gpu.kernel("kda_recurrent", "kda_recurrent_decode_f32")?,
        k_bf16: gpu.kernel("kda_recurrent", "kda_recurrent_decode_bf16")?,
    };
    println!("kda_recurrent: both entry points resolved from PTX (no fallback path)\n");

    println!("A/B — fixture H=2 D=4, zero initial state, per-token output AND state");
    let a = check_fixture(&rec)?;

    println!("\nC — prefill(4) -> decode(2) from HF's own carried state");
    let c = check_continuation(&rec)?;

    println!("\nA/B — production H=64 D=128, NON-ZERO carried state, fp32 oracle path");
    let p32 = check_production(&rec, false)?;

    println!("\nproduction, bf16 production-input path");
    let pbf = check_production(&rec, true)?;

    println!("\nD — adversarial semantics");
    let d = check_adversarial(&rec)?;

    println!(
        "\ngrid = (H, 1, 1)  block = ({}, 1, 1)  shared = 3*D*4 bytes  one thread per V",
        BLOCK.min(PROD_D as u32)
    );
    let st = PROD_H * PROD_D * PROD_D * 4;
    println!(
        "state per layer = {} B ({:.2} MiB) fp32; traffic per token = 2 reads + 2 writes = {:.2} MiB",
        st,
        st as f64 / (1024.0 * 1024.0),
        4.0 * st as f64 / (1024.0 * 1024.0)
    );

    if a && c && p32 && pbf && d {
        println!("\nPASS");
        Ok(())
    } else {
        bail!("FAIL — see the lines marked ! above");
    }
}
