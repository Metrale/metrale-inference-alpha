// SPDX-License-Identifier: MIT OR Apache-2.0
// provenance-id: 526f6e616c6420522e205374657369616b

//! 2026-09-25: DeepSeek-V4.1 Flash tiny-graph reference: the V4.1 graph on the
//! CPU, checked against `ds41_tiny_golden.json`.
//!
//! Nothing here runs on a GPU. The GPU attention (`attn_v41`) uses its RoPE
//! tables, window indices and top-k selections, and `model::forward` is the
//! oracle the GPU tests compare against.
//!
//! # Where the truth comes from
//!
//! `ds41_tiny_golden.json` is produced by `gen_ds41_tiny_golden.py` (beside
//! it), which runs DeepSeek's `inference/model.py` (the revision the golden's
//! `fixture.reference` records) on a tiny synthetic model: hc_mult=4
//! hyper-connections with delayed mixes, engram on two layers with the
//! reference's hash (`engram.py`), shared compressed attention with two kv
//! sources, three index sources and a candidate prefilter whose consumers
//! share its ratio, sqrt-softplus routing with a correction bias,
//! route_scale, the swiglu clamp and a shared expert. The tilelang kernels
//! the reference imports are replaced by `ds41_ref_shims.py` (pure torch).
//!
//! # Determinism, and why only outputs are committed
//!
//! Weights and inputs are RNG-free and order-free: `fixed_value(name, i,
//! scale, offset)` here reproduces the generator's filler (fnv1a64 salt,
//! splitmix64, 24-bit uniform, f64 arithmetic, f32 result, bf16 RNE where the
//! parameter is stored bf16). So the golden carries only outputs: each
//! capture as `{shape, n, stride, ck, data}` where `data` is a prime-strided
//! sample and `ck` is the fp64 index-weighted checksum over the whole tensor.
//! `tests.rs` checks the regeneration against the golden.
//!
//! # The engram hash parameters come from the GGUF
//!
//! Measured 2026-09-15 with `gen_ds41_engram_hash_check.py`: the token map,
//! bucket primes, slot offsets and hash multipliers that `engram.py`
//! computes from the V4.1 tokenizer equal the `deepseek41.engram.*` metadata
//! of the Q2_K GGUF, so the loader builds the engram tables from that
//! metadata alone (`EngramHashTables::from_flat`).
//!
//! Owner: model-arch, DeepSeek-V4.1 reference.
//! Invariants: none beyond the types.

use serde_json::Value;

/// 2026-09-25: The tracked golden, produced by `gen_ds41_tiny_golden.py`.
pub const GOLDEN_JSON: &str = include_str!("ds41_tiny_golden.json");

const GOLD: u64 = 0x9E37_79B9_7F4A_7C15;

/// 2026-09-25: The splitmix64 finaliser of `gen_ds41_tiny_golden.py::splitmix64`.
pub fn splitmix64(z: u64) -> u64 {
    let mut z = z.wrapping_add(GOLD);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// 2026-09-25: FNV-1a 64 over the UTF-8 bytes of `s`; the per-tensor salt.
pub fn fnv1a64(s: &str) -> u64 {
    let mut h: u64 = 0xCBF2_9CE4_8422_2325;
    for b in s.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01B3);
    }
    h
}

/// 2026-09-25: Raw stream value for element `i` of the tensor named `name`.
pub fn fixed_raw(name: &str, i: u64) -> u64 {
    splitmix64(fnv1a64(name) ^ i.wrapping_mul(GOLD))
}

/// 2026-09-25: The generator's filler: `u = (z >> 40) / 2^24`,
/// `f32((2u - 1) * scale + offset)` in f64.
pub fn fixed_value(name: &str, i: u64, scale: f64, offset: f64) -> f32 {
    let u = (fixed_raw(name, i) >> 40) as f64 / (1u64 << 24) as f64;
    ((2.0 * u - 1.0) * scale + offset) as f32
}

/// 2026-09-25: The generator's integer filler (`fixed_ints`): `fixed_raw % modulus`.
pub fn fixed_int(name: &str, i: u64, modulus: u64) -> u64 {
    fixed_raw(name, i) % modulus
}

/// 2026-09-25: Round an f32 to bf16 precision, round-to-nearest-even; NaN passes through.
pub fn to_bf16_rne(x: f32) -> f32 {
    if x.is_nan() {
        return x;
    }
    let b = x.to_bits();
    let lsb = (b >> 16) & 1;
    f32::from_bits(b.wrapping_add(0x7FFF + lsb) & 0xFFFF_0000)
}

/// 2026-09-25: fp64 index-weighted checksum `sum(v[i] * (i + 1))`, the generator's
/// `ck`. Summed sequentially here and by a torch reduction there, so callers
/// compare with a relative tolerance.
pub fn checksum<I: IntoIterator<Item = f64>>(v: I) -> f64 {
    v.into_iter()
        .enumerate()
        .map(|(i, x)| x * (i as f64 + 1.0))
        .sum()
}

/// 2026-09-25: One committed capture.
pub struct GoldenTensor {
    pub shape: Vec<usize>,
    pub n: usize,
    pub stride: usize,
    pub ck: f64,
    pub data: Vec<f64>,
}

/// 2026-09-25: One parameter's regeneration rule, from `weights_meta`.
pub struct WeightMeta {
    pub name: String,
    pub n: usize,
    pub scale: f64,
    pub offset: f64,
    pub kind: String,
    pub dtype: String,
    pub ck: f64,
}

pub struct Golden(Value);

impl Golden {
    pub fn load() -> Self {
        Golden(serde_json::from_str(GOLDEN_JSON).expect("ds41_tiny_golden.json parses"))
    }

    pub fn regimes(&self) -> Vec<String> {
        self.0["fixture"]["regimes"]
            .as_array()
            .expect("fixture.regimes")
            .iter()
            .map(|v| v.as_str().expect("regime name").to_string())
            .collect()
    }

    pub fn fixture_u64(&self, key: &str) -> u64 {
        self.0["fixture"][key]
            .as_u64()
            .unwrap_or_else(|| panic!("fixture.{key} is not an integer"))
    }

    pub fn fixture_f64(&self, key: &str) -> f64 {
        self.0["fixture"][key]
            .as_f64()
            .unwrap_or_else(|| panic!("fixture.{key} is not numeric"))
    }

    pub fn fixture_i64(&self, key: &str) -> i64 {
        self.0["fixture"][key]
            .as_i64()
            .unwrap_or_else(|| panic!("fixture.{key} is not an integer"))
    }

    pub fn fixture_usize_list(&self, key: &str) -> Vec<usize> {
        self.0["fixture"][key]
            .as_array()
            .unwrap_or_else(|| panic!("fixture.{key} is not a list"))
            .iter()
            .map(|v| v.as_u64().expect("usize") as usize)
            .collect()
    }

    pub fn lcg_probe(&self) -> Vec<u64> {
        self.0["fixture"]["lcg_probe"]
            .as_array()
            .expect("fixture.lcg_probe")
            .iter()
            .map(|v| v.as_u64().expect("u64 probe"))
            .collect()
    }

    pub fn capture_names(&self, regime: &str) -> Vec<String> {
        self.0[regime]
            .as_object()
            .unwrap_or_else(|| panic!("regime {regime} missing"))
            .keys()
            .cloned()
            .collect()
    }

    pub fn tensor(&self, regime: &str, name: &str) -> GoldenTensor {
        let t = &self.0[regime][name];
        assert!(!t.is_null(), "missing capture {regime}.{name}");
        let as_usize = |k: &str| {
            t[k].as_u64()
                .unwrap_or_else(|| panic!("{regime}.{name}.{k}")) as usize
        };
        GoldenTensor {
            shape: t["shape"]
                .as_array()
                .expect("shape")
                .iter()
                .map(|v| v.as_u64().expect("dim") as usize)
                .collect(),
            n: as_usize("n"),
            stride: as_usize("stride"),
            ck: t["ck"].as_f64().expect("ck"),
            data: t["data"]
                .as_array()
                .expect("data")
                .iter()
                .map(|v| v.as_f64().expect("numeric"))
                .collect(),
        }
    }

    /// 2026-09-25: One parameter's regeneration rule by name; panics when
    /// `weights_meta` has no such entry.
    pub fn weight_meta(&self, name: &str) -> WeightMeta {
        let m = &self.0["weights_meta"][name];
        assert!(!m.is_null(), "weights_meta has no '{name}'");
        WeightMeta {
            name: name.to_string(),
            n: m["n"].as_u64().expect("n") as usize,
            scale: m["scale"].as_f64().expect("scale"),
            offset: m["offset"].as_f64().expect("offset"),
            kind: m["kind"].as_str().expect("kind").to_string(),
            dtype: m["dtype"].as_str().expect("dtype").to_string(),
            ck: m["ck"].as_f64().expect("ck"),
        }
    }

    pub fn weights_meta(&self) -> Vec<WeightMeta> {
        self.0["weights_meta"]
            .as_object()
            .expect("weights_meta")
            .iter()
            .map(|(name, m)| WeightMeta {
                name: name.clone(),
                n: m["n"].as_u64().expect("n") as usize,
                scale: m["scale"].as_f64().expect("scale"),
                offset: m["offset"].as_f64().expect("offset"),
                kind: m["kind"].as_str().expect("kind").to_string(),
                dtype: m["dtype"].as_str().expect("dtype").to_string(),
                ck: m["ck"].as_f64().expect("ck"),
            })
            .collect()
    }
}

/// 2026-09-25: Regenerate an f32-kind parameter (bf16- or f32-stored) as the
/// generator initialised it: the rule (scale, offset) and the storage dtype
/// come from `weights_meta`. Panics on another kind; the fp8 tables are
/// regenerated by `engram::regen_table`.
pub fn regen_param(g: &Golden, name: &str) -> Vec<f32> {
    let m = g.weight_meta(name);
    assert_eq!(
        m.kind, "f32",
        "{name}: regen_param handles f32-kind parameters only"
    );
    let bf16 = m.dtype == "bfloat16";
    (0..m.n as u64)
        .map(|i| {
            let v = fixed_value(name, i, m.scale, m.offset);
            if bf16 { to_bf16_rne(v) } else { v }
        })
        .collect()
}

/// 2026-09-25: Elementwise max-abs comparison that names the worst index and both values.
#[track_caller]
pub fn assert_close(what: &str, got: &[f64], want: &[f64], tol: f64) {
    assert_eq!(
        got.len(),
        want.len(),
        "{what}: length {} vs {}",
        got.len(),
        want.len()
    );
    let mut worst = 0.0f64;
    let mut at = 0usize;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let d = (g - w).abs();
        if d > worst {
            worst = d;
            at = i;
        }
    }
    assert!(
        worst <= tol,
        "{what}: max abs diff {worst:e} > {tol:e} at index {at} (got {}, want {})",
        got[at],
        want[at]
    );
}

pub mod attn;
pub mod compress;
pub mod engram;
pub mod hc;
pub mod model;
pub mod moe;

/// 2026-09-25: Shared comparison bars for the component tests.
#[cfg(test)]
pub(crate) mod testutil {
    use super::{GoldenTensor, assert_close, checksum};

    // 2026-09-25: `EXACT` allows only f64 summation-order noise; `BF16_CK_REL`
    // is 2^-7, the bf16 relative step, for outputs the reference rounds to
    // bf16; `F32_TOL` is for f32 chains summed in another order (the hc mixes).
    pub const EXACT: f64 = 1e-12;
    pub const BF16_CK_REL: f64 = 0.0078125;
    pub const F32_TOL: f64 = 1e-4;

    /// 2026-09-25: Check the sample elementwise at `tol`, and the whole-tensor
    /// checksum against `ck_rel` times the index-weighted magnitude, so
    /// last-bit accumulation-order differences pass.
    #[track_caller]
    pub fn check_capture(what: &str, got: &[f64], g: &GoldenTensor, tol: f64, ck_rel: f64) {
        assert_eq!(got.len(), g.n, "{what}: numel");
        let sample: Vec<f64> = got.iter().step_by(g.stride).copied().collect();
        assert_close(what, &sample, &g.data, tol);
        let ck = checksum(got.iter().copied());
        let mag: f64 = got
            .iter()
            .enumerate()
            .map(|(i, v)| v.abs() * (i as f64 + 1.0))
            .sum();
        let bound = ck_rel * mag.max(1.0);
        assert!(
            (ck - g.ck).abs() <= bound,
            "{what}: checksum {ck} vs golden {} (|diff| {:e} > bound {:e})",
            g.ck,
            (ck - g.ck).abs(),
            bound
        );
    }

    pub fn bf16_tol(g: &GoldenTensor) -> f64 {
        let m = g.data.iter().fold(0f64, |a, v| a.max(v.abs()));
        2.0 * 2f64.powi(-8) * m + 1e-6
    }

    /// 2026-09-25: A capture stored at full resolution (stride 1), as f32.
    pub fn full_f32(g: &GoldenTensor, what: &str) -> Vec<f32> {
        assert_eq!(g.stride, 1, "{what} must be in FULL_CAPTURES (stride 1)");
        g.data.iter().map(|&v| v as f32).collect()
    }

    pub fn as_f64(v: &[f32]) -> Vec<f64> {
        v.iter().map(|&x| x as f64).collect()
    }
}

#[cfg(test)]
mod tests;
