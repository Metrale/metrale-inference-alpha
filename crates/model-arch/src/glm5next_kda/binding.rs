// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Typed weight binding for the GLM-5.3-Flash KDA attention blocks.
//!
//! Owner: model-arch (GLM-5.3-Flash KDA).
//! Invariants:
//! - [`bind_kda_weights`] returns weights only when a layer's `self_attn.*` tensors are exactly
//!   [`KDA_TENSORS`], each with its listed dtype and shape; anything else is an error.
//! - Bytes are uploaded unchanged: no cast, requantisation or dequantisation.
//!
//! The MTP layer is DSA-shaped (it loads through `glm5next_dsa`, see
//! `weight_loader/glm5_next_mtp.rs`) and is not bindable here; [`classify_attn_block`] tells the
//! two apart from the tensor names alone.
//!
//! The checkpoint stores three rank-3 conv tensors (`[qkv, 1, kernel]`). The binder concatenates
//! them in q, k, v order and uses the result as `[conv_dim, kernel]`: dropping the singleton
//! dimension moves no bytes. `A_log` (per head) and `dt_bias` (per channel) are F32; a BF16 copy
//! of either is refused.

use std::collections::BTreeSet;

use anyhow::{Context, Result, bail};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

use super::{Glm5NextKdaConfig, Glm5NextKdaWeights};
use metrale_model_layers::weight_map::DenseWeight;

/// 2026-09-25: The dtypes the KDA binder accepts.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum KdaDtype {
    Bf16,
    F32,
}

impl KdaDtype {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "BF16" => Some(Self::Bf16),
            "F32" => Some(Self::F32),
            _ => None,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::Bf16 => "BF16",
            Self::F32 => "F32",
        }
    }
}

/// 2026-09-25: One tensor as it sits in the checkpoint: dtype, shape and raw little-endian bytes.
pub struct RawTensor<'a> {
    pub dtype: KdaDtype,
    pub shape: Vec<usize>,
    pub bytes: &'a [u8],
}

/// 2026-09-25: A checkpoint slice scoped to one decoder layer. Names are layer-relative
/// (`self_attn.q_proj.weight`), so the same binder works for any layer index.
pub trait KdaTensorSource {
    fn get(&self, name: &str) -> Option<RawTensor<'_>>;
    /// 2026-09-25: Every layer-relative name present, including non-attention ones.
    fn names(&self) -> Vec<String>;
}

/// 2026-09-25: One entry of [`KDA_TENSORS`]: a `self_attn` tensor's name, dtype and shape.
///
/// Shapes are expressed against [`Glm5NextKdaConfig`], so a geometry mismatch fails at binding
/// rather than at launch. `H` = heads, `D` = head_dim, `Q` = H*D, `X` = hidden, `K` = conv
/// kernel, `One` = 1.
#[derive(Clone, Copy, Debug)]
pub struct TensorSpec {
    pub name: &'static str,
    pub dtype: KdaDtype,
    dims: &'static [Dim],
}

#[derive(Clone, Copy, Debug)]
enum Dim {
    Q,
    X,
    D,
    H,
    K,
    One,
}

use Dim::{D as DD, H as DH, K as DK, One as D1, Q as DQ, X as DX};

pub const KDA_TENSORS: &[TensorSpec] = &[
    TensorSpec {
        name: "self_attn.q_proj.weight",
        dtype: KdaDtype::Bf16,
        dims: &[DQ, DX],
    },
    TensorSpec {
        name: "self_attn.k_proj.weight",
        dtype: KdaDtype::Bf16,
        dims: &[DQ, DX],
    },
    TensorSpec {
        name: "self_attn.v_proj.weight",
        dtype: KdaDtype::Bf16,
        dims: &[DQ, DX],
    },
    TensorSpec {
        name: "self_attn.q_conv1d.weight",
        dtype: KdaDtype::Bf16,
        dims: &[DQ, D1, DK],
    },
    TensorSpec {
        name: "self_attn.k_conv1d.weight",
        dtype: KdaDtype::Bf16,
        dims: &[DQ, D1, DK],
    },
    TensorSpec {
        name: "self_attn.v_conv1d.weight",
        dtype: KdaDtype::Bf16,
        dims: &[DQ, D1, DK],
    },
    TensorSpec {
        name: "self_attn.f_a_proj.weight",
        dtype: KdaDtype::Bf16,
        dims: &[DD, DX],
    },
    TensorSpec {
        name: "self_attn.f_b_proj.weight",
        dtype: KdaDtype::Bf16,
        dims: &[DQ, DD],
    },
    TensorSpec {
        name: "self_attn.g_a_proj.weight",
        dtype: KdaDtype::Bf16,
        dims: &[DD, DX],
    },
    TensorSpec {
        name: "self_attn.g_b_proj.weight",
        dtype: KdaDtype::Bf16,
        dims: &[DQ, DD],
    },
    TensorSpec {
        name: "self_attn.b_proj.weight",
        dtype: KdaDtype::Bf16,
        dims: &[DH, DX],
    },
    // 2026-09-25: F32 on disk and F32 in `kda_gate_bf16`'s signature; uploaded unconverted.
    TensorSpec {
        name: "self_attn.A_log",
        dtype: KdaDtype::F32,
        dims: &[DH],
    },
    TensorSpec {
        name: "self_attn.dt_bias",
        dtype: KdaDtype::F32,
        dims: &[DQ],
    },
    TensorSpec {
        name: "self_attn.o_norm.weight",
        dtype: KdaDtype::Bf16,
        dims: &[DD],
    },
    TensorSpec {
        name: "self_attn.o_proj.weight",
        dtype: KdaDtype::Bf16,
        dims: &[DX, DQ],
    },
];

/// 2026-09-25: Names that identify a DSA (`deepseek_sparse_attention`) block, the MTP layer
/// included, so a caller can classify a layer without its index.
pub const DSA_MARKERS: &[&str] = &[
    "self_attn.kv_a_proj_with_mqa.weight",
    "self_attn.indexer.wk.weight",
];

impl TensorSpec {
    pub fn expected_shape(&self, c: &Glm5NextKdaConfig) -> Vec<usize> {
        self.dims
            .iter()
            .map(|d| match d {
                Dim::Q => c.qkv_dim(),
                Dim::X => c.hidden,
                Dim::D => c.head_dim,
                Dim::H => c.heads,
                Dim::K => c.conv_kernel,
                Dim::One => 1,
            })
            .collect()
    }
}

/// 2026-09-25: What kind of attention block a layer's tensor names describe.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AttnBlockKind {
    Kda,
    /// 2026-09-25: `deepseek_sparse_attention`, and the MTP layer, which is DSA-shaped.
    Dsa,
    Unknown,
}

/// 2026-09-25: Classify from tensor names alone: `Dsa` when every [`DSA_MARKERS`] name is present,
/// else `Kda` when every [`KDA_TENSORS`] name is present, else `Unknown`.
pub fn classify_attn_block(names: &[String]) -> AttnBlockKind {
    let set: BTreeSet<&str> = names.iter().map(String::as_str).collect();
    if DSA_MARKERS.iter().all(|m| set.contains(m)) {
        return AttnBlockKind::Dsa;
    }
    if KDA_TENSORS.iter().all(|t| set.contains(t.name)) {
        return AttnBlockKind::Kda;
    }
    AttnBlockKind::Unknown
}

/// 2026-09-25: Per-layer counts from [`bind_kda_weights`]. Tensors outside `self_attn.` are
/// counted in `non_attn_seen` and never bound.
#[derive(Clone, Debug, Default)]
pub struct KdaBindReport {
    pub layer_idx: usize,
    pub bound: usize,
    pub self_attn_seen: usize,
    pub non_attn_seen: usize,
    pub unknown_self_attn: Vec<String>,
    pub bytes: usize,
}

fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let p = gpu.alloc(bytes.len().max(1))?;
    gpu.copy_h2d(bytes, p)?;
    Ok(p)
}

/// 2026-09-25: Bind one KDA block: exact tensor set, dtypes, shapes and byte counts, or an error.
///
/// Weights are uploaded unchanged: BF16 stays BF16 and F32 stays F32.
pub fn bind_kda_weights(
    gpu: &dyn GpuBackend,
    cfg: &Glm5NextKdaConfig,
    layer_idx: usize,
    src: &dyn KdaTensorSource,
) -> Result<(Glm5NextKdaWeights, KdaBindReport)> {
    cfg.validate()?;
    let names = src.names();
    let mut rep = KdaBindReport {
        layer_idx,
        ..Default::default()
    };

    let known: BTreeSet<&str> = KDA_TENSORS.iter().map(|t| t.name).collect();
    for n in &names {
        if n.starts_with("self_attn.") {
            rep.self_attn_seen += 1;
            if !known.contains(n.as_str()) {
                rep.unknown_self_attn.push(n.clone());
            }
        } else {
            rep.non_attn_seen += 1;
        }
    }
    if !rep.unknown_self_attn.is_empty() {
        bail!(
            "layer {layer_idx}: {} unrecognised self_attn tensor(s): {:?} — a KDA block has \
             exactly {} and this binder refuses to skip anything",
            rep.unknown_self_attn.len(),
            rep.unknown_self_attn,
            KDA_TENSORS.len()
        );
    }

    let mut fetch = |spec: &TensorSpec| -> Result<Vec<u8>> {
        let t = src
            .get(spec.name)
            .with_context(|| format!("layer {layer_idx}: missing {}", spec.name))?;
        if t.dtype != spec.dtype {
            bail!(
                "layer {layer_idx}: {} is {} but a KDA block requires {} — casting it would \
                 change the numerics",
                spec.name,
                t.dtype.name(),
                spec.dtype.name()
            );
        }
        let want = spec.expected_shape(cfg);
        if t.shape != want {
            bail!(
                "layer {layer_idx}: {} has shape {:?}, expected {want:?}",
                spec.name,
                t.shape
            );
        }
        let elem = match spec.dtype {
            KdaDtype::Bf16 => 2,
            KdaDtype::F32 => 4,
        };
        let expect_bytes = want.iter().product::<usize>() * elem;
        if t.bytes.len() != expect_bytes {
            bail!(
                "layer {layer_idx}: {} is {} B, shape {want:?} implies {expect_bytes} B",
                spec.name,
                t.bytes.len()
            );
        }
        rep.bound += 1;
        rep.bytes += t.bytes.len();
        Ok(t.bytes.to_vec())
    };

    let by_name = |n: &str| -> &TensorSpec { KDA_TENSORS.iter().find(|t| t.name == n).unwrap() };
    let mut raw = |n: &str| fetch(by_name(n));

    let q_proj = raw("self_attn.q_proj.weight")?;
    let k_proj = raw("self_attn.k_proj.weight")?;
    let v_proj = raw("self_attn.v_proj.weight")?;
    // 2026-09-25: Concatenate in q, k, v order, the order the projections are packed in. The
    // shape check above pins rank 3 with a middle dimension of 1, and `[dim, 1, ks]` and
    // `[dim, ks]` are the same row-major bytes.
    let mut conv = raw("self_attn.q_conv1d.weight")?;
    conv.extend_from_slice(&raw("self_attn.k_conv1d.weight")?);
    conv.extend_from_slice(&raw("self_attn.v_conv1d.weight")?);
    debug_assert_eq!(conv.len(), cfg.conv_dim() * cfg.conv_kernel * 2);
    let f_a = raw("self_attn.f_a_proj.weight")?;
    let f_b = raw("self_attn.f_b_proj.weight")?;
    let g_a = raw("self_attn.g_a_proj.weight")?;
    let g_b = raw("self_attn.g_b_proj.weight")?;
    let b_proj = raw("self_attn.b_proj.weight")?;
    let a_log = raw("self_attn.A_log")?;
    let dt_bias = raw("self_attn.dt_bias")?;
    let o_norm = raw("self_attn.o_norm.weight")?;
    let o_proj = raw("self_attn.o_proj.weight")?;

    let dw = |b: &[u8]| -> Result<DenseWeight> {
        Ok(DenseWeight {
            weight: upload(gpu, b)?,
        })
    };
    let w = Glm5NextKdaWeights {
        q_proj: dw(&q_proj)?,
        k_proj: dw(&k_proj)?,
        v_proj: dw(&v_proj)?,
        conv: dw(&conv)?,
        f_a: dw(&f_a)?,
        f_b: dw(&f_b)?,
        dt_bias: upload(gpu, &dt_bias)?,
        a_log: upload(gpu, &a_log)?,
        b_proj: dw(&b_proj)?,
        g_a: dw(&g_a)?,
        g_b: dw(&g_b)?,
        o_norm: dw(&o_norm)?,
        o_proj: dw(&o_proj)?,
    };
    if rep.bound != KDA_TENSORS.len() {
        bail!(
            "layer {layer_idx}: bound {} of {} tensors",
            rep.bound,
            KDA_TENSORS.len()
        );
    }
    Ok((w, rep))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> Glm5NextKdaConfig {
        Glm5NextKdaConfig {
            hidden: 4096,
            heads: 64,
            head_dim: 128,
            conv_kernel: 4,
            gate_lower_bound: -5.0,
            rms_norm_eps: 1e-5,
            l2_eps: 1e-6,
            chunk: 32,
        }
    }

    /// 2026-09-25: The spec table at the `glm-5.3-flash` geometry (hidden 4096, 64 heads of 128,
    /// conv kernel 4, as `kernels/gb10/glm-5.3-flash/MODEL.toml` records it).
    #[test]
    fn tensor_spec_matches_the_audited_checkpoint_shapes() {
        let c = cfg();
        let want: &[(&str, &str, &[usize])] = &[
            ("self_attn.q_proj.weight", "BF16", &[8192, 4096]),
            ("self_attn.k_proj.weight", "BF16", &[8192, 4096]),
            ("self_attn.v_proj.weight", "BF16", &[8192, 4096]),
            ("self_attn.q_conv1d.weight", "BF16", &[8192, 1, 4]),
            ("self_attn.k_conv1d.weight", "BF16", &[8192, 1, 4]),
            ("self_attn.v_conv1d.weight", "BF16", &[8192, 1, 4]),
            ("self_attn.f_a_proj.weight", "BF16", &[128, 4096]),
            ("self_attn.f_b_proj.weight", "BF16", &[8192, 128]),
            ("self_attn.g_a_proj.weight", "BF16", &[128, 4096]),
            ("self_attn.g_b_proj.weight", "BF16", &[8192, 128]),
            ("self_attn.b_proj.weight", "BF16", &[64, 4096]),
            ("self_attn.A_log", "F32", &[64]),
            ("self_attn.dt_bias", "F32", &[8192]),
            ("self_attn.o_norm.weight", "BF16", &[128]),
            ("self_attn.o_proj.weight", "BF16", &[4096, 8192]),
        ];
        assert_eq!(
            KDA_TENSORS.len(),
            want.len(),
            "the KDA block has exactly 15 tensors"
        );
        for (n, dt, sh) in want {
            let s = KDA_TENSORS.iter().find(|t| &t.name == n).expect(n);
            assert_eq!(s.dtype.name(), *dt, "{n} dtype");
            assert_eq!(s.expected_shape(&c), sh.to_vec(), "{n} shape");
        }
    }

    /// 2026-09-25: A DSA name set classifies as `Dsa`, never `Kda`.
    #[test]
    fn dsa_and_mtp_blocks_do_not_classify_as_kda() {
        let dsa: Vec<String> = [
            "self_attn.kv_a_proj_with_mqa.weight",
            "self_attn.kv_a_layernorm.weight",
            "self_attn.kv_b_proj.weight",
            "self_attn.q_a_proj.weight",
            "self_attn.q_b_proj.weight",
            "self_attn.indexer.wk.weight",
            "self_attn.indexer.wq_b.weight",
            "self_attn.o_proj.weight",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        assert_eq!(classify_attn_block(&dsa), AttnBlockKind::Dsa);

        let kda: Vec<String> = KDA_TENSORS.iter().map(|t| t.name.to_string()).collect();
        assert_eq!(classify_attn_block(&kda), AttnBlockKind::Kda);

        // 2026-09-25: A KDA block missing one tensor is `Unknown`, not `Kda`.
        assert_eq!(classify_attn_block(&kda[1..]), AttnBlockKind::Unknown);
    }

    #[test]
    fn chunk_width_is_bounded_by_the_shared_memory_ceiling() {
        let mut c = cfg();
        assert!(c.validate().is_ok(), "C=32 must fit");
        assert!(c.smem_scan() <= SMEM_CEILING);
        c.chunk = 64;
        assert!(
            c.validate().is_err(),
            "C=64 needs 81920 B and must be rejected, not truncated"
        );
    }

    use super::super::SMEM_CEILING;
}
