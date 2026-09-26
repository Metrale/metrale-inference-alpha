// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: GLM-5.3-Flash FFN kernels on real weights against HF transformers 5.16.1
//! goldens (`glm5next_moe_ref/gen_ffn_golden.py`); the run is `common/glm5next_ffn_run.rs`.
//!
//! Owner: model-arch examples.
//! Invariants:
//! - Returns an error when any row is above its floor, when a floor-C row is not bit-exact,
//!   or when the `clamp64` regime does not make the SwiGLU clamp fire.
//!
//! Gate 3 covers the weight-only NVFP4 (W4A16) routed experts of layer 3; gate 4 the BF16
//! dense FFN of layer 0 and the BF16 shared expert of layer 3. Rows:
//!   C: `dequant_nvfp4_to_bf16` against the golden's dequantised weights rounded to BF16,
//!      bit for bit.
//!   D: `w4a16_gemm` gate_proj against the golden.
//!   E: the whole gate/up, clamp, SiLU-multiply, down chain against the golden.
//! A non-C row passes when its residual is at most its floor, `bf16_output_floor`.
//!
//! `MOE_PACKET_DIR` names the directory holding the layer packets:
//!   MOE_PACKET_DIR=<packet dir> \
//!   cargo run -p metrale-model-arch --release --example glm5next_ffn_microtest \
//!       --features cuda,gpu-examples

use std::collections::BTreeMap;

use anyhow::{Context, Result, bail};
use half::bf16;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_gpu_runtime::kernel_args::KernelLaunch;
use serde_json::Value;

#[path = "common/glm5next_ffn_run.rs"]
pub(crate) mod glm5next_ffn_run;

#[path = "common/golden.rs"]
pub(crate) mod golden;

pub(crate) static GOLDEN: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
    golden::load(
        "crates/model-arch/src/glm5next_moe_ref/ffn_golden.json",
        "gen_ffn_golden.py",
    )
});
/// 2026-09-25: `(name, T, input_scale)`. `clamp64` takes input scale 8 so the SwiGLU clamp
/// fires; the run errors if it does not.
pub(crate) const REGIMES: [(&str, usize, f32); 4] = [
    ("t1", 1, 0.5),
    ("t7", 7, 0.5),
    ("t64", 64, 0.5),
    ("clamp64", 64, 8.0),
];

pub(crate) fn up_bytes(g: &dyn GpuBackend, b: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(b.len().max(1))?;
    g.copy_h2d(b, p)?;
    Ok(p)
}
pub(crate) fn up_bf16(g: &dyn GpuBackend, d: &[f32]) -> Result<DevicePtr> {
    let b: Vec<u8> = d
        .iter()
        .flat_map(|x| bf16::from_f32(*x).to_bits().to_le_bytes())
        .collect();
    up_bytes(g, &b)
}
pub(crate) fn down_bf16(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<f32>> {
    let mut b = vec![0u8; n * 2];
    g.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(2)
        .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect())
}

pub(crate) struct Lcg(u64);
impl Lcg {
    fn t(&mut self, n: usize, scale: f32) -> Vec<f32> {
        (0..n)
            .map(|_| {
                self.0 = self
                    .0
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((((self.0 >> 40) as f64 / (1u64 << 24) as f64) * 2.0 - 1.0) as f32) * scale
            })
            .collect()
    }
}
pub(crate) fn input(t: usize, hid: usize, scale: f32) -> Vec<f32> {
    Lcg(0x0FFF_5EED).t(t * hid, scale)
}

pub(crate) struct Golden(Value);
impl Golden {
    fn f(&self, k: &str) -> Result<f64> {
        self.0["fixture"][k]
            .as_f64()
            .with_context(|| format!("fixture.{k}"))
    }
    fn get(&self, sec: &str, name: &str) -> Result<(Vec<f32>, usize, usize, f64)> {
        let s = &self.0[sec][name];
        if s.is_null() {
            bail!("golden missing {sec}/{name}");
        }
        Ok((
            s["data"]
                .as_array()
                .context("data")?
                .iter()
                .map(|x| x.as_f64().unwrap_or(f64::NAN) as f32)
                .collect(),
            s["stride"].as_u64().context("stride")? as usize,
            s["n"].as_u64().context("n")? as usize,
            s["ck"].as_f64().context("ck")?,
        ))
    }
}

/// 2026-09-25: Max abs difference against a strided golden row, with the element count
/// checked first.
pub(crate) fn resid(what: &str, got: &[f32], g: &(Vec<f32>, usize, usize, f64)) -> Result<f32> {
    let (want, stride, n, _) = g;
    if got.len() != *n {
        bail!(
            "{what}: golden describes {n} elements, produced {}",
            got.len()
        );
    }
    let mut worst = 0.0f32;
    for (i, w) in want.iter().enumerate() {
        worst = worst.max((got[i * stride] - w).abs());
    }
    Ok(worst)
}
/// 2026-09-25: One BF16 ulp at magnitude `x`. BF16 keeps 8 significand bits, so
/// ulp = 2^(exp-7).
pub(crate) fn bf16_ulp(x: f32) -> f32 {
    if x == 0.0 {
        return f32::MIN_POSITIVE;
    }
    let e = x.abs().log2().floor() as i32;
    (2.0f32).powi(e - 7)
}

/// 2026-09-25: Floor for a BF16-output kernel against an FP32 reference: the larger of the
/// activation floor and `ULP_BUDGET` BF16 ulps at `mag`, which covers the output rounding
/// and a K sum in a different order from the reference's.
pub(crate) const ULP_BUDGET: f32 = 4.0;
pub(crate) fn bf16_output_floor(activation_floor: f32, mag: f32) -> f32 {
    activation_floor.max(ULP_BUDGET * bf16_ulp(mag))
}

pub(crate) fn magnitude(g: &(Vec<f32>, usize, usize, f64)) -> f32 {
    g.0.iter().fold(0.0f32, |a, b| a.max(b.abs()))
}

pub(crate) struct Packet {
    pub(crate) raw: Vec<u8>,
    pub(crate) base: usize,
    pub(crate) hdr: BTreeMap<String, (String, Vec<usize>, usize, usize)>,
}
impl Packet {
    fn open(path: &str) -> Result<Self> {
        let raw = std::fs::read(path).with_context(|| format!("reading {path}"))?;
        let hn = u64::from_le_bytes(raw[..8].try_into().unwrap()) as usize;
        let j: Value = serde_json::from_slice(&raw[8..8 + hn])?;
        let mut hdr = BTreeMap::new();
        for (k, m) in j.as_object().context("hdr")? {
            if k == "__metadata__" {
                continue;
            }
            hdr.insert(
                k.clone(),
                (
                    m["dtype"].as_str().context("dtype")?.to_string(),
                    m["shape"]
                        .as_array()
                        .context("shape")?
                        .iter()
                        .map(|x| x.as_u64().unwrap() as usize)
                        .collect(),
                    m["data_offsets"][0].as_u64().context("o0")? as usize,
                    m["data_offsets"][1].as_u64().context("o1")? as usize,
                ),
            );
        }
        Ok(Self {
            raw,
            base: 8 + hn,
            hdr,
        })
    }
    fn meta(&self, n: &str) -> Result<&(String, Vec<usize>, usize, usize)> {
        self.hdr.get(n).with_context(|| format!("missing {n}"))
    }
    fn bytes(&self, n: &str) -> Result<&[u8]> {
        let (_, _, a, b) = self.meta(n)?;
        Ok(&self.raw[self.base + a..self.base + b])
    }
    fn f32_scalar(&self, n: &str) -> Result<f32> {
        let (dt, ..) = self.meta(n)?;
        if dt != "F32" {
            bail!("{n}: expected F32, got {dt}");
        }
        let b = self.bytes(n)?;
        Ok(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
}

pub(crate) struct Row {
    pub(crate) gate: &'static str,
    pub(crate) what: String,
    pub(crate) regime: &'static str,
    pub(crate) floor: &'static str,
    pub(crate) e: f32,
    pub(crate) b: f32,
    pub(crate) mag: f32,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn w4a16(
    g: &dyn GpuBackend,
    k: KernelHandle,
    a: DevicePtr,
    bp: DevicePtr,
    bs: DevicePtr,
    s2: f32,
    c: DevicePtr,
    m: u32,
    n: u32,
    kk: u32,
) -> Result<()> {
    // 2026-09-25: The grid and block of `ops::w4a16_gemm`: 64 x 64 output tiles, 128 threads.
    KernelLaunch::new(g, k)
        .grid([n.div_ceil(64), m.div_ceil(64), 1])
        .block([128, 1, 1])
        .arg_ptr(a)
        .arg_ptr(bp)
        .arg_ptr(bs)
        .arg_f32(s2)
        .arg_ptr(c)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(kk)
        .launch(0)
}

pub(crate) fn gemm_bf16(
    g: &dyn GpuBackend,
    k: KernelHandle,
    a: DevicePtr,
    b: DevicePtr,
    c: DevicePtr,
    m: u32,
    n: u32,
    kk: u32,
) -> Result<()> {
    KernelLaunch::new(g, k)
        .grid([n.div_ceil(16), m.div_ceil(16), 1])
        .block([16, 16, 1])
        .arg_ptr(a)
        .arg_ptr(b)
        .arg_ptr(c)
        .arg_u32(m)
        .arg_u32(n)
        .arg_u32(kk)
        .launch(0)
}

pub(crate) fn swiglu(
    g: &dyn GpuBackend,
    k: KernelHandle,
    gate: DevicePtr,
    up: DevicePtr,
    out: DevicePtr,
    n: u32,
    limit: f32,
) -> Result<()> {
    KernelLaunch::new(g, k)
        .grid([n.div_ceil(256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(gate)
        .arg_ptr(up)
        .arg_ptr(out)
        .arg_u32(n)
        .arg_f32(limit)
        .launch(0)
}

fn main() -> Result<()> {
    glm5next_ffn_run::run()
}
