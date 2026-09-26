// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: GLM-5.3-Flash DSA kernels against HF transformers 5.16.1 goldens: the kpool
//! indexer (gate 4) over every fixture layer, then the NoPE MLA (gate 5).
//!
//! Owner: model-arch examples.
//! Invariants:
//! - Exit 1 when any layer of either gate mismatches. A missing golden panics; a missing
//!   packet or `dsa_indexer` entry point, or a fixture that is not NoPE, is an error.
//! - The packet reader refuses any tensor that is not BF16 or F32.
//!
//! Goldens come from `glm5next_dsa_ref/gen_dsa_indexer_golden.py` and `gen_dsa_mla_golden.py`
//! against `LibertAIDAI/GLM-5.3-Flash-NVFP4@9e0d74e3`. The indexer regimes are `short7` (one
//! complete pool), `medium64`, `ragged13` (5 leading pad tokens), `relu_probe` (negated
//! query), `longsparse` (S=2560: 640 pools against a 512-pool budget) and `decode` (one query
//! over the 2560-token state).
//!
//! `KDA_DSA_PACKET_DIR` names the directory holding `dsa_layer<L>.safetensors`:
//!   KDA_DSA_PACKET_DIR=<packet dir> \
//!   cargo run -p metrale-model-arch --release --example dsa_indexer_microtest \
//!       --features cuda,gpu-examples

use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Context, Result, bail};
use half::bf16;
use metrale_gpu_runtime::cuda_backend::MetraleCudaBackend;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use metrale_model_arch::glm5next_dsa_ref::DsaDims;
use serde_json::Value;

#[path = "common/dsa_indexer_layer.rs"]
pub(crate) mod dsa_indexer_layer;
use dsa_indexer_layer::*;

#[path = "common/dsa_indexer_mla.rs"]
pub(crate) mod dsa_indexer_mla;
use dsa_indexer_mla::*;

#[path = "common/golden.rs"]
pub(crate) mod golden;

pub(crate) static IDX_GOLDEN: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
    golden::load(
        "crates/model-arch/src/glm5next_dsa_ref/dsa_indexer_golden.json",
        "gen_dsa_indexer_golden.py",
    )
});
pub(crate) static MLA_GOLDEN: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
    golden::load(
        "crates/model-arch/src/glm5next_dsa_ref/dsa_mla_golden.json",
        "gen_dsa_mla_golden.py",
    )
});

/// 2026-09-25: Shared-memory cap (48 KiB) the top-k and MLA launches are held to.
pub(crate) const SMEM_CEILING: usize = 49_152;

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
pub(crate) fn up_i32(g: &dyn GpuBackend, d: &[i32]) -> Result<DevicePtr> {
    let b: Vec<u8> = d.iter().flat_map(|x| x.to_le_bytes()).collect();
    let p = g.alloc(b.len().max(1))?;
    g.copy_h2d(&b, p)?;
    Ok(p)
}
pub(crate) fn up_u8(g: &dyn GpuBackend, d: &[u8]) -> Result<DevicePtr> {
    let p = g.alloc(d.len().max(1))?;
    g.copy_h2d(d, p)?;
    Ok(p)
}
pub(crate) fn down_f32(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<f32>> {
    let mut b = vec![0u8; n * 4];
    g.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}
pub(crate) fn down_i32(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<i32>> {
    let mut b = vec![0u8; n * 4];
    g.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}
pub(crate) fn down_u8(g: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<u8>> {
    let mut b = vec![0u8; n];
    g.copy_d2h(p, &mut b)?;
    Ok(b)
}

pub(crate) struct Lcg(u64);
impl Lcg {
    fn u(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (((self.0 >> 40) as f32) / ((1u32 << 24) as f32)) * 2.0 - 1.0
    }
    fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.u()).collect()
    }
}
pub(crate) fn r(x: f32) -> f32 {
    bf16::from_f32(x).to_f32()
}
pub(crate) fn round_bf16(v: &[f32]) -> Vec<f32> {
    v.iter().map(|x| r(*x)).collect()
}
pub(crate) fn maxabs(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len(), "len {} vs {}", a.len(), b.len());
    a.iter()
        .zip(b)
        .fold(0.0f64, |m, (x, y)| m.max((*x as f64 - *y as f64).abs()))
}
pub(crate) fn checksum(s: &[f32]) -> f64 {
    s.iter()
        .enumerate()
        .map(|(i, v)| *v as f64 * (i as f64 + 1.0))
        .sum()
}
pub(crate) fn ck_i(s: &[i32]) -> f64 {
    s.iter()
        .enumerate()
        .map(|(i, v)| *v as f64 * (i as f64 + 1.0))
        .sum()
}
pub(crate) fn sample<T: Copy>(v: &[T], stride: usize) -> Vec<T> {
    v.iter().step_by(stride).copied().collect()
}

pub(crate) struct Entry {
    pub(crate) n: usize,
    pub(crate) stride: usize,
    pub(crate) ck: f64,
    pub(crate) data: Vec<f64>,
}
pub(crate) fn entry(v: &Value, key: &str) -> Result<Entry> {
    let e = &v[key];
    if e.is_null() {
        bail!("golden is missing {key}");
    }
    Ok(Entry {
        n: e["n"].as_u64().context("n")? as usize,
        stride: e["stride"].as_u64().context("stride")? as usize,
        ck: e["ck"].as_f64().unwrap_or(0.0),
        data: e["data"]
            .as_array()
            .context("data")?
            .iter()
            .map(|x| x.as_f64().unwrap())
            .collect(),
    })
}
impl Entry {
    /// 2026-09-25: Errors unless the golden's element count equals `want`, so a shape
    /// drift fails outright instead of showing as a small sampled error.
    fn expect_n(&self, want: usize, what: &str) -> Result<&Self> {
        if self.n != want {
            bail!(
                "{what}: golden describes {} elements, produced {want}",
                self.n
            );
        }
        Ok(self)
    }
}

pub(crate) fn scalar(v: &Value, key: &str) -> i64 {
    v[key].as_i64().unwrap_or(-1)
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
        for (k, m) in j.as_object().unwrap() {
            if k == "__metadata__" {
                continue;
            }
            let dt = m["dtype"].as_str().unwrap().to_string();
            if dt != "BF16" && dt != "F32" {
                bail!("{k}: DSA blocks are BF16/F32 only, saw {dt}");
            }
            let shape = m["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap() as usize)
                .collect();
            let a = m["data_offsets"][0].as_u64().unwrap() as usize;
            let b = m["data_offsets"][1].as_u64().unwrap() as usize;
            hdr.insert(k.clone(), (dt, shape, a, b));
        }
        Ok(Self {
            raw,
            base: 8 + hn,
            hdr,
        })
    }
    fn f32s(&self, name: &str) -> Result<Vec<f32>> {
        let (dt, _, a, b) = self
            .hdr
            .get(name)
            .with_context(|| format!("missing {name}"))?;
        let by = &self.raw[self.base + a..self.base + b];
        Ok(match dt.as_str() {
            "BF16" => by
                .chunks_exact(2)
                .map(|c| bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
                .collect(),
            _ => by
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        })
    }
}

/// 2026-09-25: `y = x @ w^T` with fp32 accumulation in ascending k order.
pub(crate) fn gemm(x: &[f32], m: usize, k: usize, w: &[f32], n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0f32;
            for i in 0..k {
                acc += x[row * k + i] * w[col * k + i];
            }
            out[row * n + col] = acc;
        }
    }
    out
}

#[derive(Clone, Copy)]
pub(crate) enum Arm {
    Bf16,
    F32,
}

pub(crate) struct Kernels {
    pub(crate) compress: KernelHandle,
    pub(crate) scores: KernelHandle,
    pub(crate) topk: KernelHandle,
    pub(crate) expand: KernelHandle,
    pub(crate) compact: KernelHandle,
    pub(crate) mask: KernelHandle,
    pub(crate) mla: KernelHandle,
}
impl Kernels {
    fn resolve(g: &dyn GpuBackend) -> Result<Self> {
        Ok(Self {
            compress: g.kernel("dsa_indexer", "dsa_kpool_compress")?,
            scores: g.kernel("dsa_indexer", "dsa_index_scores")?,
            topk: g.kernel("dsa_indexer", "dsa_topk_pools")?,
            expand: g.kernel("dsa_indexer", "dsa_expand_selection")?,
            compact: g.kernel("dsa_indexer", "dsa_compact_pools")?,
            mask: g.kernel("dsa_indexer", "dsa_topk_to_mask")?,
            mla: g.kernel("dsa_indexer", "dsa_mla_masked_attn")?,
        })
    }
}

fn main() -> Result<()> {
    let backend = MetraleCudaBackend::new(0, &metrale_kernels::ptx_modules())?;
    let gpu: &dyn GpuBackend = &backend;
    let iv: Value = serde_json::from_str(&IDX_GOLDEN)?;
    let mv: Value = serde_json::from_str(&MLA_GOLDEN)?;
    let f = &iv["fixture"];
    let mf = &mv["fixture"];

    let dims = DsaDims {
        hidden: f["hidden"].as_u64().unwrap() as usize,
        index_heads: f["index_n_heads"].as_u64().unwrap() as usize,
        index_head_dim: f["index_head_dim"].as_u64().unwrap() as usize,
        index_kpool: f["index_kpool"].as_u64().unwrap() as usize,
        index_topk: f["index_topk"].as_u64().unwrap() as usize,
        always_select_tail: f["always_select_tail"].as_bool().unwrap(),
        q_lora_rank: f["q_lora_rank"].as_u64().unwrap() as usize,
        heads: mf["heads"].as_u64().unwrap() as usize,
        kv_lora_rank: mf["kv_lora_rank"].as_u64().unwrap() as usize,
        qk_nope_head_dim: mf["qk_nope_head_dim"].as_u64().unwrap() as usize,
        qk_rope_head_dim: mf["qk_rope_head_dim"].as_u64().unwrap() as usize,
        v_head_dim: mf["v_head_dim"].as_u64().unwrap() as usize,
    };
    println!("GLM-5.3-Flash DSA — kpool indexer + NoPE MLA vs HF transformers 5.16.1");
    println!("  checkpoint {}", f["checkpoint"]);
    println!(
        "  indexer: heads={} head_dim={} kpool={} topk={} select_k_max={} out_width={} tail={}",
        dims.index_heads,
        dims.index_head_dim,
        dims.index_kpool,
        dims.index_topk,
        dims.index_topk / dims.index_kpool,
        dims.out_width(),
        dims.always_select_tail
    );
    println!(
        "  MLA: heads={} qk_nope={} qk_rope={} v_head={} kv_lora={} q_lora={} scaling={}",
        dims.heads,
        dims.qk_nope_head_dim,
        dims.qk_rope_head_dim,
        dims.v_head_dim,
        dims.kv_lora_rank,
        dims.q_lora_rank,
        mf["scaling"]
    );
    if !dims.is_nope() {
        bail!(
            "this path is NoPE-only; qk_rope_head_dim={}",
            dims.qk_rope_head_dim
        );
    }
    println!(
        "  NoPE confirmed: qk_rope_head_dim = 0, qk_head_dim = {}",
        dims.qk_head_dim()
    );

    let k = Kernels::resolve(gpu)?;
    println!("  7 DSA kernel entry points resolved (no fallback path)");

    let dir = std::env::var("KDA_DSA_PACKET_DIR")
        .unwrap_or_else(|_| "/home/msi1/metrale-scratch/dsa-family".to_string());

    let layers: Vec<usize> = f["layers"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as usize)
        .collect();
    let mut ok = true;

    println!("\n=== GATE 4 — kpool indexer (proven BEFORE the MLA) ===");
    for &l in &layers {
        let pkt = Packet::open(&format!("{dir}/dsa_layer{l}.safetensors"))?;
        ok &= indexer_layer(gpu, &k, &iv, dims, l, &pkt)?;
    }

    println!("\n=== GATE 5 — NoPE MLA over the selected tokens ===");
    for &l in &layers {
        let pkt = Packet::open(&format!("{dir}/dsa_layer{l}.safetensors"))?;
        ok &= mla_layer(gpu, &k, &mv, dims, l, &pkt)?;
    }

    println!(
        "\n{}",
        if ok {
            "RESULT: PASS — indexer selection and NoPE MLA both match HF 5.16.1 on real weights"
        } else {
            "RESULT: FAIL"
        }
    );
    if !ok {
        std::process::exit(1);
    }
    Ok(())
}
