// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Numeric gate for the GLM-5.3 mHC (manifold-constrained hyper-connection) kernels
//! on checkpoint `hc_{attn,ffn}_{fn,base,scale}` weights, against the golden
//! `gen_mhc_golden.py` writes.
//!
//! Owner: model-arch examples (GLM-5.3 mHC kernels).
//! Invariants:
//! - The run fails if a packet tensor has the wrong shape or dtype (`fn` BF16, `base` and
//!   `scale` F32), if any stage is above the reference's own BF16 floor, or if the GLM and
//!   DeepSeek-V4 `hc_pre` arms stop differing on the comb column sums.
//!
//! Two arms run the same inputs: GLM's `glm5next_hc_pre` / `glm5next_hc_post`, and DeepSeek-V4's
//! `hyper_connection::hc_pre` / `hc_post`, whose Sinkhorn ends on an exact column projection
//! that the GLM kernel does not apply.
//!
//! Each site is checked at both halves of the residual write:
//!   `hc_pre`  -> `post` [T,hc], `comb` [T,hc,hc], `collapsed` [T,H]
//!   `hc_post` -> `site_out` [T,hc,H]
//! and the ffn site consumes the attn site's output, as the decoder layer chains them, so a
//! per-site pass that does not compose still fails.
//!
//! The packets are read from `MHC_PACKET_DIR` (`mhc_layer{L}.safetensors` for each of `LAYERS`):
//!
//!   MHC_PACKET_DIR=/path/to/packets \
//!   cargo run -p metrale-model-arch --release --example mhc_microtest \
//!       --features cuda,gpu-examples

use std::collections::BTreeMap;

use anyhow::{Context, Result, bail};
use half::bf16;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use serde_json::Value;

#[path = "common/mhc_run.rs"]
pub(crate) mod mhc_run;

#[path = "common/golden.rs"]
pub(crate) mod golden;

pub(crate) static GOLDEN: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
    golden::load(
        "crates/model-arch/src/glm5next_mhc_ref/mhc_golden.json",
        "gen_mhc_golden.py",
    )
});

pub(crate) const LAYERS: [usize; 4] = [0, 3, 22, 44];
pub(crate) const SITES: [&str; 2] = ["attn", "ffn"];
/// 2026-09-25: `(name, T)`. mHC is per token, so the regimes vary only the token count.
pub(crate) const REGIMES: [(&str, usize); 4] = [
    ("decode1", 1),
    ("short7", 7),
    ("medium64", 64),
    ("long2176", 2176),
];

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
pub(crate) fn down_bf16(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<f32>> {
    let mut b = vec![0u8; n * 2];
    g.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(2)
        .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
        .collect())
}

/// 2026-09-25: The golden generator's LCG. The inputs are regenerated here, never read from the
/// golden.
pub(crate) struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed)
    }
    fn u(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (((self.0 >> 40) as f64 / (1u64 << 24) as f64) * 2.0 - 1.0) as f32
    }
    fn t(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.u()).collect()
    }
}

pub(crate) struct Golden(Value);
impl Golden {
    fn load() -> Result<Self> {
        Ok(Golden(serde_json::from_str(&GOLDEN)?))
    }
    fn fixture(&self, k: &str) -> Result<f64> {
        self.0["fixture"][k]
            .as_f64()
            .with_context(|| format!("fixture.{k}"))
    }
    /// 2026-09-25: Returns `(values, stride, n)`: the golden stores every tensor strided, so a
    /// comparison walks the produced tensor with the same stride, after checking `n`.
    fn get(
        &self,
        layer: usize,
        arm: &str,
        regime: &str,
        name: &str,
    ) -> Result<(Vec<f32>, usize, usize)> {
        let sec = &self.0["by_layer"][layer.to_string()][format!("{arm}__{regime}")][name];
        if sec.is_null() {
            bail!("golden missing {layer}/{arm}__{regime}/{name}");
        }
        let n = sec["n"].as_u64().context("n")? as usize;
        let stride = sec["stride"].as_u64().context("stride")? as usize;
        let v = sec["data"]
            .as_array()
            .context("data")?
            .iter()
            .map(|x| x.as_f64().unwrap_or(f64::NAN) as f32)
            .collect();
        Ok((v, stride, n))
    }
}

/// 2026-09-25: Max abs difference between a produced tensor and a strided golden row. A length
/// different from the golden's element count is an error, not a comparison over a prefix.
pub(crate) fn residual(what: &str, got: &[f32], g: &(Vec<f32>, usize, usize)) -> Result<f32> {
    let (want, stride, n) = g;
    if got.len() != *n {
        bail!(
            "{what}: golden describes {n} elements, produced {}",
            got.len()
        );
    }
    let mut worst = 0.0f32;
    for (i, w) in want.iter().enumerate() {
        let d = (got[i * stride] - w).abs();
        if d > worst {
            worst = d;
        }
    }
    Ok(worst)
}

/// 2026-09-25: Floor B for a stage, and its magnitude: the max difference between the golden's
/// BF16 and F32 arms. A residual is read against it.
pub(crate) fn floor_b(g: &Golden, layer: usize, regime: &str, name: &str) -> Result<(f32, f32)> {
    let (a, _, _) = g.get(layer, "bf16", regime, name)?;
    let (b, _, _) = g.get(layer, "f32", regime, name)?;
    let mut worst = 0.0f32;
    let mut mag = 0.0f32;
    for (x, y) in a.iter().zip(&b) {
        worst = worst.max((x - y).abs());
        mag = mag.max(y.abs());
    }
    Ok((worst, mag))
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
        for (k, m) in j.as_object().context("packet header")? {
            if k == "__metadata__" {
                continue;
            }
            let dt = m["dtype"].as_str().context("dtype")?.to_string();
            if dt != "BF16" && dt != "F32" {
                bail!("{k}: mHC params are BF16/F32 only, saw {dt}");
            }
            let shape = m["shape"]
                .as_array()
                .context("shape")?
                .iter()
                .map(|x| x.as_u64().unwrap() as usize)
                .collect();
            let a = m["data_offsets"][0].as_u64().context("off0")? as usize;
            let b = m["data_offsets"][1].as_u64().context("off1")? as usize;
            hdr.insert(k.clone(), (dt, shape, a, b));
        }
        Ok(Self {
            raw,
            base: 8 + hn,
            hdr,
        })
    }
    /// 2026-09-25: A tensor as f32, widening BF16 exactly, with its shape and dtype.
    fn f32s(&self, name: &str) -> Result<(Vec<f32>, Vec<usize>, String)> {
        let (dt, shape, a, b) = self
            .hdr
            .get(name)
            .with_context(|| format!("missing {name}"))?;
        let by = &self.raw[self.base + a..self.base + b];
        let v = match dt.as_str() {
            "BF16" => by
                .chunks_exact(2)
                .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                .collect(),
            _ => by
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        };
        Ok((v, shape.clone(), dt.clone()))
    }
}

pub(crate) struct Row {
    pub(crate) arm: &'static str,
    pub(crate) layer: usize,
    pub(crate) regime: &'static str,
    pub(crate) site: &'static str,
    pub(crate) stage: &'static str,
    pub(crate) e: f32,
    pub(crate) b: f32,
    pub(crate) mag: f32,
}

fn main() -> Result<()> {
    mhc_run::run()
}
