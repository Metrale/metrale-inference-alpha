// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests for `dsa_tensor_specs` and `verify_dsa_block` at the GLM-5.3 geometry
//! of `kernels/gb10/glm-5.3-flash/MODEL.toml`.
//!
//! Owner: model-arch (GLM-5.3 DSA).
//! Invariants: none beyond the types.

use std::collections::BTreeMap;

use super::*;
use crate::glm5next_kda::binding::RawTensor;

fn cfg() -> Glm5NextDsaConfig {
    Glm5NextDsaConfig {
        hidden: 4096,
        index_heads: 32,
        index_head_dim: 128,
        index_kpool: 4,
        index_topk: 2048,
        always_select_tail: true,
        local_heads: 64,
        q_lora_rank: 1536,
        kv_lora_rank: 512,
        qk_nope_head_dim: 256,
        qk_rope_head_dim: 0,
        v_head_dim: 256,
        max_context: 16_384,
    }
}

/// 2026-09-25: A source built from the spec table, which each test then perturbs.
struct FakeSource {
    t: BTreeMap<String, (Dtype, Vec<usize>, Vec<u8>)>,
}

impl FakeSource {
    fn good() -> Self {
        let mut t = BTreeMap::new();
        for s in dsa_tensor_specs(&cfg(), 64) {
            let elems: usize = s.shape.iter().product();
            t.insert(
                s.name.clone(),
                (s.dtype, s.shape.clone(), vec![0u8; elems * 2]),
            );
        }
        Self { t }
    }
    fn drop(mut self, name: &str) -> Self {
        assert!(self.t.remove(name).is_some(), "{name} was not present");
        self
    }
    fn reshape(mut self, name: &str, shape: Vec<usize>) -> Self {
        let e = self.t.get_mut(name).expect("present");
        let elems: usize = shape.iter().product();
        e.1 = shape;
        e.2 = vec![0u8; elems * 2];
        self
    }
    fn add(mut self, name: &str, shape: Vec<usize>) -> Self {
        let elems: usize = shape.iter().product();
        self.t
            .insert(name.to_string(), (Dtype::Bf16, shape, vec![0u8; elems * 2]));
        self
    }
}

impl TensorSource for FakeSource {
    fn get(&self, name: &str) -> Option<RawTensor<'_>> {
        self.t.get(name).map(|(d, s, b)| RawTensor {
            dtype: *d,
            shape: s.clone(),
            bytes: b.as_slice(),
        })
    }
    fn names(&self) -> Vec<String> {
        self.t.keys().cloned().collect()
    }
}

/// 2026-09-25: The spec table's bytes match the checkpoint: `attn_DSA` in
/// `crates/model-engine/tests/fixtures/glm53-nvfp4-9e0d74e3-families.tsv` is 2,748,117,504 B
/// over 11 DSA layers.
#[test]
fn spec_table_matches_the_measured_layer() {
    let specs = dsa_tensor_specs(&cfg(), 64);
    assert_eq!(specs.len(), 14, "DSA block tensor count");
    let bytes: usize = specs
        .iter()
        .map(|s| s.shape.iter().product::<usize>() * 2)
        .sum();
    assert_eq!(bytes, 249_828_864);
    assert_eq!(bytes * 11, 2_748_117_504);
}

#[test]
fn a_correct_block_binds() {
    let r = verify_dsa_block(&cfg(), 64, &FakeSource::good()).expect("should bind");
    assert_eq!(r.bound, 14);
    assert_eq!(r.total_bytes, 249_828_864);
    assert!(r.unclaimed.is_empty());
}

/// 2026-09-25: A checkpoint without the `k_norm` LayerNorm bias fails the bind.
#[test]
fn a_missing_k_norm_bias_is_an_error() {
    let src = FakeSource::good().drop("self_attn.indexer.k_norm.bias");
    let e = verify_dsa_block(&cfg(), 64, &src).unwrap_err();
    assert!(e.to_string().contains("k_norm.bias"), "unexpected: {e}");
    assert!(e.to_string().contains("missing"), "unexpected: {e}");
}

/// 2026-09-25: A `q_b_proj` at `kv_b_proj`'s per-head width (512) is still a well-formed
/// 2-D tensor; the shape check refuses it.
#[test]
fn swapped_per_head_widths_are_rejected() {
    let src = FakeSource::good().reshape("self_attn.q_b_proj.weight", vec![64 * 512, 1536]);
    let e = verify_dsa_block(&cfg(), 64, &src).unwrap_err();
    assert!(e.to_string().contains("q_b_proj"), "unexpected: {e}");
    assert!(e.to_string().contains("shape"), "unexpected: {e}");
}

/// 2026-09-25: NoPE: a 576-wide latent, the DeepSeek-V4-Flash cache width, is refused.
#[test]
fn a_deepseek_v4_shaped_latent_is_rejected() {
    let src = FakeSource::good().reshape("self_attn.kv_a_proj_with_mqa.weight", vec![576, 4096]);
    let e = verify_dsa_block(&cfg(), 64, &src).unwrap_err();
    assert!(
        e.to_string().contains("kv_a_proj_with_mqa"),
        "unexpected: {e}"
    );
}

/// 2026-09-25: The APE table is `[index_kpool, index_head_dim]`. The transpose has the same
/// element count, so only the shape check separates them.
#[test]
fn a_transposed_ape_table_is_rejected() {
    let src =
        FakeSource::good().reshape("self_attn.indexer.index_kpool_compress_ape", vec![128, 4]);
    let e = verify_dsa_block(&cfg(), 64, &src).unwrap_err();
    assert!(e.to_string().contains("compress_ape"), "unexpected: {e}");
}

/// 2026-09-25: An unclaimed `self_attn.*` tensor fails the bind.
#[test]
fn an_unclaimed_self_attn_tensor_is_an_error() {
    let src = FakeSource::good().add("self_attn.wkv_a_rope.weight", vec![64, 4096]);
    let e = verify_dsa_block(&cfg(), 64, &src).unwrap_err();
    assert!(e.to_string().contains("unclaimed"), "unexpected: {e}");
}

/// 2026-09-25: Tensors outside `self_attn.` (mHC, norms, MLP) do not trip the unclaimed
/// check.
#[test]
fn non_attention_tensors_are_ignored() {
    let src = FakeSource::good()
        .add("hc_attn_fn", vec![24, 16384])
        .add("mlp.gate.weight", vec![288, 4096])
        .add("input_layernorm.weight", vec![4096]);
    let r = verify_dsa_block(&cfg(), 64, &src).expect("should still bind");
    assert_eq!(r.bound, 14);
}

/// 2026-09-25: Changing `full_heads` changes the shapes of `q_b_proj`, `kv_b_proj` and
/// `o_proj` only.
#[test]
fn specs_track_the_head_count() {
    let a = dsa_tensor_specs(&cfg(), 64);
    let b = dsa_tensor_specs(&cfg(), 32);
    let differing: Vec<&str> = a
        .iter()
        .zip(b.iter())
        .filter(|(x, y)| x.shape != y.shape)
        .map(|(x, _)| x.name.as_str())
        .collect();
    assert_eq!(
        differing,
        vec![
            "self_attn.q_b_proj.weight",
            "self_attn.kv_b_proj.weight",
            "self_attn.o_proj.weight"
        ]
    );
}
