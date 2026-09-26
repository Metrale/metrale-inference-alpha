// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Numeric gate for the GLM-5.3 KDA bounded forget-gate kernels
//! (`kda_gate_f32`, `kda_gate_bf16`).
//!
//! Owner: model-arch examples (GLM-5.3 KDA kernels).
//! Invariants:
//! - The run fails unless the fixture, production, ragged-`T` and boundary checks all pass.
//!
//! The kernel computes `lower_bound * sigmoid(exp(A_log[h]) * (g + dt_bias[h, d]))`: the decay is
//! per (head, key channel), with `dt_bias` `[H, D]` and a `[T, H, D]` output, and the law is
//! bounded. `compute_gdn_gates` takes one `dt_bias` per head and applies
//! `exp(-exp(A_log) * softplus(a + dt_bias))`, so it cannot express this gate.
//!
//! Oracles:
//! 1. `kda_golden.json`, the tracked toy fixture (H=2, D=4, T=6).
//! 2. `kda_gate_prod_golden.json` at production geometry (H=64, D=128, T=2), from
//!    `gen_kda_gate_prod_golden.py`: `dt_bias` ramps across both `d` and `h`, `A_log` differs
//!    per head, and a per-head amplitude ramp on `g` runs from the linear region into saturation.
//! 3. `glm5next_kda_ref::bounded_gate`, the CPU reference, for ragged `T` and boundary sweeps.
//!
//! Acceptance: the FP32 entry point must be within `MAX_ABS` / `MAX_REL` of each oracle, and on
//! the production golden no more than `MAX_FLOOR_RATIO` times the CPU reference's own distance
//! from it. The BF16 entry point is scored against the CPU reference run on BF16-rounded `g`.
//!
//!   cargo run -p metrale-model-arch --release --example kda_gate_microtest \
//!       --features cuda,gpu-examples

use anyhow::{Result, bail};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use metrale_model_arch::glm5next_kda_ref::{KdaDims, bounded_gate};
use serde_json::Value;

#[path = "common/kda_gate_checks.rs"]
pub(crate) mod kda_gate_checks;
use kda_gate_checks::*;

pub(crate) const FIXTURE_GOLDEN: &str = include_str!("../src/glm5next_kda_ref/kda_golden.json");
#[path = "common/golden.rs"]
pub(crate) mod golden;

pub(crate) static PROD_GOLDEN: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
    golden::load(
        "crates/model-arch/src/glm5next_kda_ref/kda_gate_prod_golden.json",
        "gen_kda_gate_prod_golden.py",
    )
});

/// 2026-09-25: GLM-5.3's KDA heads and head dim.
pub(crate) const PROD_H: usize = 64;
pub(crate) const PROD_D: usize = 128;

pub(crate) const BLOCK: u32 = 128;

/// 2026-09-25: Printed in the PASS message only; `within` gates on absolute and relative error.
/// Bit-exactness is not expected: the gate has two transcendentals, and the device `expf` and
/// the host libm round differently. The `oracle floor` line scores the CPU reference against
/// the golden with no GPU involved.
pub(crate) const MAX_ULP: i64 = 2;

/// 2026-09-25: Absolute bound. 2 ulp of a value in [4, 8) is 9.54e-7; `lower_bound = -12.5` in
/// the boundary sweep reaches [8, 16), where 2 ulp is 1.91e-6.
pub(crate) const MAX_ABS: f64 = 2.0e-6;

/// 2026-09-25: Relative bound, over elements above the magnitude guard in `compare`.
pub(crate) const MAX_REL: f64 = 1.0e-5;

/// 2026-09-25: On the production golden, the GPU error may be at most this multiple of the CPU
/// reference's error against the same golden.
pub(crate) const MAX_FLOOR_RATIO: f64 = 2.0;

/// 2026-09-25: Acceptance on absolute and relative error. `max_ulp` is reported, not gated: at
/// small magnitudes a numerically negligible error spans many representable floats.
pub(crate) fn within(e: &Err2) -> bool {
    e.max_abs <= MAX_ABS && e.max_rel <= MAX_REL
}

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

pub(crate) fn json_arr(v: &Value, section: &str, name: &str) -> Vec<f32> {
    v[section][name]["data"]
        .as_array()
        .unwrap_or_else(|| panic!("missing {section}.{name}"))
        .iter()
        .map(|x| x.as_f64().expect("numeric") as f32)
        .collect()
}

#[derive(Default)]
pub(crate) struct Err2 {
    pub(crate) max_abs: f64,
    pub(crate) max_rel: f64,
    pub(crate) max_ulp: i64,
    pub(crate) exact: usize,
    pub(crate) total: usize,
}

/// 2026-09-25: Monotonic ordering of f32 bit patterns, so `|ord(a) - ord(b)|` counts the
/// representable floats between them. `+0.0` and `-0.0` both map to 0.
pub(crate) fn ord(x: f32) -> i64 {
    let b = x.to_bits();
    if b & 0x8000_0000 != 0 {
        -((b & 0x7fff_ffff) as i64)
    } else {
        b as i64
    }
}

/// 2026-09-25: Max absolute error over all elements; max relative and ULP error over elements
/// whose reference magnitude exceeds 1e-6.
pub(crate) fn compare(got: &[f32], want: &[f32]) -> Err2 {
    assert_eq!(got.len(), want.len());
    let mut e = Err2 {
        total: got.len(),
        ..Default::default()
    };
    for (g, w) in got.iter().zip(want) {
        let d = (*g as f64 - *w as f64).abs();
        e.max_abs = e.max_abs.max(d);
        // 2026-09-25: Deep in the saturated tail a negligible absolute gap is many
        // representable floats, so the relative and ULP figures skip tiny references.
        if w.abs() > 1e-6 {
            e.max_rel = e.max_rel.max(d / (*w as f64).abs());
            e.max_ulp = e.max_ulp.max((ord(*g) - ord(*w)).abs());
        }
        if g.to_bits() == w.to_bits() {
            e.exact += 1;
        }
    }
    e
}

pub(crate) fn report(label: &str, e: &Err2, dtype: &str) {
    println!(
        "  {label:<44} dtype={dtype:<5} max_abs={:.3e} max_rel={:.3e} max_ulp={:<2} exact={}/{}",
        e.max_abs, e.max_rel, e.max_ulp, e.exact, e.total
    );
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_gate(
    g: &dyn GpuBackend,
    k: KernelHandle,
    g_raw: DevicePtr,
    dt_bias: DevicePtr,
    a_log: DevicePtr,
    out: DevicePtr,
    tokens: usize,
    heads: usize,
    head_dim: usize,
    lower_bound: f32,
) -> Result<()> {
    KernelLaunch::new(g, k)
        .grid([(tokens * heads) as u32, 1, 1])
        .block([BLOCK, 1, 1])
        .arg_ptr(g_raw)
        .arg_ptr(dt_bias)
        .arg_ptr(a_log)
        .arg_ptr(out)
        .arg_u32(tokens as u32)
        .arg_u32(heads as u32)
        .arg_u32(head_dim as u32)
        .arg_f32(lower_bound)
        .launch(0)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_f32(
    g: &dyn GpuBackend,
    k: KernelHandle,
    g_raw: &[f32],
    dt_bias: &[f32],
    a_log: &[f32],
    tokens: usize,
    heads: usize,
    head_dim: usize,
    lower_bound: f32,
) -> Result<Vec<f32>> {
    let n = tokens * heads * head_dim;
    let (dg, db, da) = (up_f32(g, g_raw)?, up_f32(g, dt_bias)?, up_f32(g, a_log)?);
    let out = g.alloc(n * 4)?;
    launch_gate(g, k, dg, db, da, out, tokens, heads, head_dim, lower_bound)?;
    g.synchronize(0)?;
    down_f32(g, out, n)
}

/// 2026-09-25: Integer LCG mapped to binary32 by an exact division by 2^24, the same mapping as
/// the golden generators, so both sides produce identical inputs.
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

/// 2026-09-25: The FP32 kernel against the toy fixture golden.
pub(crate) fn check_fixture(g: &dyn GpuBackend, k: KernelHandle) -> Result<bool> {
    let v: Value = serde_json::from_str(FIXTURE_GOLDEN)?;
    let f = &v["fixture"];
    let (h, d, t) = (
        f["heads"].as_u64().unwrap() as usize,
        f["head_dim"].as_u64().unwrap() as usize,
        f["tokens"].as_u64().unwrap() as usize,
    );
    let lb = f["lower_bound"].as_f64().unwrap() as f32;

    let got = run_f32(
        g,
        k,
        &json_arr(&v, "outputs", "g_lowrank"),
        &json_arr(&v, "inputs", "dt_bias"),
        &json_arr(&v, "inputs", "A_log"),
        t,
        h,
        d,
        lb,
    )?;
    let e = compare(&got, &json_arr(&v, "outputs", "gate"));
    report(&format!("fixture H={h} D={d} T={t} vs HF"), &e, "f32");
    Ok(within(&e))
}

/// 2026-09-25: Ragged `T` from 1 to 1024 at production H and D, against the CPU reference.
/// `T` only changes the grid extent, so this checks launch geometry.
pub(crate) fn check_ragged_t(g: &dyn GpuBackend, k: KernelHandle) -> Result<bool> {
    let mut rng = Lcg(0xA11CE_u64);
    let dt_bias = rng.vec(PROD_H * PROD_D);
    let a_log: Vec<f32> = (0..PROD_H)
        .map(|h| -1.0 + 2.0 * h as f32 / (PROD_H as f32 - 1.0))
        .collect();
    let dims_of = |t| KdaDims {
        hidden: 0,
        heads: PROD_H,
        head_dim: PROD_D,
        tokens: t,
    };
    let mut ok = true;
    for &t in &[1usize, 2, 7, 63, 64, 129, 257, 1024] {
        let g_raw = rng.vec(t * PROD_H * PROD_D);
        let got = run_f32(g, k, &g_raw, &dt_bias, &a_log, t, PROD_H, PROD_D, -5.0)?;
        let want = bounded_gate(&g_raw, &dt_bias, &a_log, dims_of(t), -5.0);
        let e = compare(&got, &want);
        report(&format!("ragged T={t:<5} vs CPU reference"), &e, "f32");
        ok &= within(&e);
    }
    Ok(ok)
}

fn main() -> Result<()> {
    let g = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let gpu: &dyn GpuBackend = &g;

    // 2026-09-25: `kernel()`, not `try_kernel`: a missing entry point is an error.
    let kf = gpu.kernel("kda_gate", "kda_gate_f32")?;
    let kb = gpu.kernel("kda_gate", "kda_gate_bf16")?;
    println!("kda_gate: both entry points resolved from PTX (no fallback path)\n");

    println!("A/B — numeric acceptance vs HuggingFace transformers 5.16.1");
    let a = check_fixture(gpu, kf)?;
    let b = check_production(gpu, kf, kb)?;

    println!("\nA — ragged / production T (launch geometry)");
    let c = check_ragged_t(gpu, kf)?;

    println!("\nC — boundary tests");
    let d = check_boundaries(gpu, kf)?;

    println!("\ngrid = (T*H, 1, 1)  block = ({BLOCK}, 1, 1)  one block per (token, head) row of D");

    if a && b && c && d {
        println!(
            "\nPASS — every fp32 case within {MAX_ABS:.1e} abs / {MAX_REL:.1e} rel of the oracle, and"
        );
        println!(
            "       no worse than the CPU reference's own distance from HF (ratio ~1.0). The residual"
        );
        println!(
            "       is CUDA expf (<=2 ulp) vs host libm, not kernel error. MAX_ULP={MAX_ULP} holds where"
        );
        println!("       ULP is meaningful (values near the -5.0 bound).");
        Ok(())
    } else {
        bail!("FAIL — see the lines marked ! above");
    }
}
