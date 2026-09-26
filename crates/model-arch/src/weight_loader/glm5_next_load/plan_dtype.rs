// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: F32 on disk where the KDA/DSA attention plan wants BF16: the cast, and
//! nothing else.
//!
//! Owner: model-arch weight loader (GLM-5.3).
//! Invariants:
//! - Only an F32 source whose plan dtype is BF16 is cast (`cast_to_plan_dtype`); BF16,
//!   quantised (U8, F8_E4M3) and plan-F32 tensors (`A_log`, `dt_bias`) are returned as `None`
//!   and no copy is made.
//! - The cast rounds to nearest even via `half::bf16::from_f32`, the same cast
//!   `upload_f32_as_bf16` applies to the norms and mHC weights.
//! - The copy is read only through `LayerSource`'s `KdaTensorSource::get`, which the KDA
//!   binder reads through `KdaShardedSource`. `LayerSource::f32`, which the DSA builder,
//!   norms, mHC and MLP read, keeps returning the checkpoint's own bytes.
//! - A BF16 tensor where the plan wants F32 is not converted; the binder's dtype check
//!   refuses it.
//!
//! The plan is read from the binders' spec tables: `KDA_TENSORS` for KDA names, and for any
//! other `self_attn.*` name BF16, which is what every `dsa_tensor_specs` entry is; the test
//! `every_dsa_spec_is_bf16` pins that.

use anyhow::{Result, bail};
use metrale_model_weights::weights::WeightDtype;

use crate::glm5next_kda::binding::{KDA_TENSORS, KdaDtype};

/// 2026-09-25: The dtype the plan wants for one layer-relative tensor name
/// (`self_attn.q_conv1d.weight`), or `None` for a name outside `self_attn.*`
/// (norms, mHC, MLP), which the loader reads through `LayerSource::f32`.
pub(super) fn plan_dtype(rel: &str) -> Option<KdaDtype> {
    if let Some(spec) = KDA_TENSORS.iter().find(|s| s.name == rel) {
        return Some(spec.dtype);
    }
    // 2026-09-25: Every `dsa_tensor_specs` entry is BF16 (pinned by
    // `every_dsa_spec_is_bf16`). A `self_attn.*` name in neither table still
    // reaches `bind_kda_weights`, which refuses unrecognised `self_attn` tensors.
    if rel.starts_with("self_attn.") {
        return Some(KdaDtype::Bf16);
    }
    None
}

/// 2026-09-25: The plan-dtype copy of one tensor's host bytes, or `None` when
/// no cast applies, in which case `LayerSource`'s `get` hands out the
/// checkpoint's own bytes.
pub(super) fn cast_to_plan_dtype(
    rel: &str,
    on_disk: WeightDtype,
    bytes: &[u8],
) -> Result<Option<(WeightDtype, Vec<u8>)>> {
    if on_disk != WeightDtype::FP32 || plan_dtype(rel) != Some(KdaDtype::Bf16) {
        return Ok(None);
    }
    Ok(Some((
        WeightDtype::BF16,
        f32_bytes_to_bf16_bytes(rel, bytes)?,
    )))
}

/// 2026-09-25: Little-endian F32 bytes to little-endian BF16 bytes, round to
/// nearest even.
///
/// The length is checked first because `chunks_exact` would drop a trailing
/// partial element; a bad length is an error, not a shorter tensor.
fn f32_bytes_to_bf16_bytes(rel: &str, src: &[u8]) -> Result<Vec<u8>> {
    if !src.len().is_multiple_of(4) {
        bail!(
            "{rel}: {} B is not a whole number of F32 elements",
            src.len()
        );
    }
    Ok(src
        .chunks_exact(4)
        .flat_map(|c| {
            half::bf16::from_f32(f32::from_le_bytes([c[0], c[1], c[2], c[3]])).to_le_bytes()
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::glm5next_dsa::Glm5NextDsaConfig;
    use crate::glm5next_dsa::binding::dsa_tensor_specs;

    fn f32_blob(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    fn bf16_blob(v: &[f32]) -> Vec<u8> {
        v.iter()
            .flat_map(|x| half::bf16::from_f32(*x).to_le_bytes())
            .collect()
    }

    /// 2026-09-25: The same geometry as `glm5next_dsa::binding::tests::cfg`.
    fn dsa_cfg() -> Glm5NextDsaConfig {
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

    /// 2026-09-25: The claim `plan_dtype`'s `self_attn.*` fallback rests on:
    /// every DSA spec is BF16.
    #[test]
    fn every_dsa_spec_is_bf16() {
        for s in dsa_tensor_specs(&dsa_cfg(), 64) {
            assert_eq!(s.dtype, KdaDtype::Bf16, "{} is no longer BF16", s.name);
        }
    }

    /// 2026-09-25: The plan is read off the binder's table, F32 entries included.
    #[test]
    fn plan_dtype_matches_the_kda_spec_table() {
        assert_eq!(
            plan_dtype("self_attn.q_conv1d.weight"),
            Some(KdaDtype::Bf16)
        );
        assert_eq!(plan_dtype("self_attn.o_proj.weight"), Some(KdaDtype::Bf16));
        // 2026-09-25: The two KDA entries whose plan dtype is F32.
        assert_eq!(plan_dtype("self_attn.A_log"), Some(KdaDtype::F32));
        assert_eq!(plan_dtype("self_attn.dt_bias"), Some(KdaDtype::F32));
        // 2026-09-25: A name outside KDA_TENSORS: BF16 through the fallback.
        assert_eq!(
            plan_dtype("self_attn.indexer.k_norm.bias"),
            Some(KdaDtype::Bf16)
        );
        // 2026-09-25: Not an attention tensor: no plan entry.
        assert_eq!(plan_dtype("input_layernorm.weight"), None);
        assert_eq!(plan_dtype("hc_attn_fn"), None);
        assert_eq!(plan_dtype("mlp.gate.weight"), None);
    }

    /// 2026-09-25: F32 on disk where the plan says BF16 comes back as the
    /// round-to-nearest-even BF16 bytes.
    #[test]
    fn f32_conv1d_is_cast_to_the_plan_bf16() {
        // 2026-09-25: Two ties: 0x3F81_8000 is halfway between 0x3F81 and 0x3F82
        // and rounds up to the even 0x3F82; 0x3F80_8000 rounds down to 0x3F80.
        let vals = [
            1.0f32,
            -2.5,
            0.0,
            f32::from_bits(0x3F80_8000),
            f32::from_bits(0x3F81_8000),
            1.0e-8,
        ];
        let (dt, out) = cast_to_plan_dtype(
            "self_attn.q_conv1d.weight",
            WeightDtype::FP32,
            &f32_blob(&vals),
        )
        .unwrap()
        .expect("F32 on disk where the plan says BF16 must be cast");
        assert_eq!(dt, WeightDtype::BF16);
        assert_eq!(out.len(), vals.len() * 2, "F32 -> BF16 halves the tensor");
        assert_eq!(out, bf16_blob(&vals));
        // 2026-09-25: Spelled out, so a change of rounding convention cannot
        // pass by agreeing with `bf16_blob`.
        assert_eq!(&out[0..2], &0x3F80u16.to_le_bytes());
        assert_eq!(&out[6..8], &0x3F80u16.to_le_bytes());
        assert_eq!(&out[8..10], &0x3F82u16.to_le_bytes());
    }

    /// 2026-09-25: A BF16 source takes the raw path: no cast, no copy.
    #[test]
    fn bf16_source_takes_the_raw_path() {
        let raw = bf16_blob(&[1.0, -2.5, 0.0, 7.75]);
        assert!(
            cast_to_plan_dtype("self_attn.q_conv1d.weight", WeightDtype::BF16, &raw)
                .unwrap()
                .is_none()
        );
    }

    /// 2026-09-25: F32 where the plan also says F32 (`A_log`, `dt_bias`) takes
    /// the raw path.
    #[test]
    fn f32_stays_raw_where_the_plan_wants_f32() {
        let raw = f32_blob(&[1.0, -2.5, 0.0]);
        for name in ["self_attn.A_log", "self_attn.dt_bias"] {
            assert!(
                cast_to_plan_dtype(name, WeightDtype::FP32, &raw)
                    .unwrap()
                    .is_none(),
                "{name} must not be cast"
            );
        }
    }

    /// 2026-09-25: A tensor outside `self_attn.*` is never cast.
    #[test]
    fn unclaimed_tensors_are_never_cast() {
        let raw = f32_blob(&[1.0, 2.0]);
        for name in ["input_layernorm.weight", "hc_attn_fn", "mlp.gate.weight"] {
            assert!(
                cast_to_plan_dtype(name, WeightDtype::FP32, &raw)
                    .unwrap()
                    .is_none(),
                "{name} must not be cast"
            );
        }
    }

    /// 2026-09-25: Quantised dtypes are never cast, even under a claimed name.
    #[test]
    fn quantised_dtypes_are_left_alone() {
        let raw = vec![0xABu8; 16];
        for dt in [WeightDtype::UInt8, WeightDtype::FP8E4M3] {
            assert!(
                cast_to_plan_dtype("self_attn.q_conv1d.weight", dt, &raw)
                    .unwrap()
                    .is_none(),
                "{dt:?} must not be cast"
            );
        }
    }

    /// 2026-09-25: A truncated F32 blob is an error, not a shorter tensor.
    #[test]
    fn a_partial_f32_element_is_an_error() {
        let err = cast_to_plan_dtype("self_attn.q_conv1d.weight", WeightDtype::FP32, &[0u8; 10])
            .unwrap_err()
            .to_string();
        assert!(err.contains("whole number of F32 elements"), "{err}");
    }
}
