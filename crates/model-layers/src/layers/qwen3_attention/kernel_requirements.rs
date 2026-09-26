// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The startup kernel check for a `--kv-cache-dtype`: the
//! reshape/decode pair from `init_kernel_dispatch`, plus the optional kernels
//! the dtype's dispatch cannot run without (its chunked-prefill kernel and, for
//! a WHT-rotated side, the WHT bookends).
//!
//! Owner: model-layers (attention).
//! Invariants: none beyond the types.

use metrale_cache::kv_cache::KvCacheDtype;

/// 2026-09-25: The optional kernels (`init.rs` resolves them with
/// `try_kernel`) that `kv_dtype` cannot run without: its chunked-prefill
/// paged-attention kernel, and the WHT bookend pair when either side is
/// WHT-rotated and `head_dim` is 128, 256 or 512. The reshape/decode pair is
/// not in this list; `validate_required_kernels` adds it.
pub(super) fn required_optional_kernels_for_dtype(
    kv_dtype: KvCacheDtype,
    head_dim: usize,
) -> Vec<(&'static str, &'static str)> {
    let mut req: Vec<(&'static str, &'static str)> = Vec::new();
    match kv_dtype {
        KvCacheDtype::Turbo2 => {
            req.push(("prefill_paged_turbo2", "attn_prefill_paged_turbo2"));
        }
        KvCacheDtype::Turbo3 => {
            req.push(("prefill_paged_turbo3", "attn_prefill_paged_turbo3_64"));
        }
        KvCacheDtype::Turbo4 => {
            req.push(("prefill_paged_turbo4", "attn_prefill_paged_turbo4_64"));
        }
        KvCacheDtype::Turbo8 => {
            req.push(("prefill_paged_turbo8", "attn_prefill_paged_turbo8_64"));
        }
        KvCacheDtype::Bf16KTurbo3V => {
            req.push((
                "prefill_paged_bf16k_turbo3v",
                "attn_prefill_paged_bf16k_turbo3v_64",
            ));
        }
        KvCacheDtype::Bf16KTurbo4V => {
            req.push((
                "prefill_paged_bf16k_turbo4v",
                "attn_prefill_paged_bf16k_turbo4v_64",
            ));
        }
        KvCacheDtype::Bf16KTurbo2V => {
            req.push((
                "prefill_paged_bf16k_turbo2v",
                "attn_prefill_paged_bf16k_turbo2v_64",
            ));
        }
        KvCacheDtype::Fp8KTurbo3V => {
            req.push((
                "prefill_paged_fp8k_turbo3v",
                "attn_prefill_paged_fp8k_turbo3v_64",
            ));
        }
        KvCacheDtype::Fp8KTurbo4V => {
            req.push((
                "prefill_paged_fp8k_turbo4v",
                "attn_prefill_paged_fp8k_turbo4v_64",
            ));
        }
        KvCacheDtype::Fp8KTurbo2V => {
            req.push((
                "prefill_paged_fp8k_turbo2v",
                "attn_prefill_paged_fp8k_turbo2v_64",
            ));
        }
        KvCacheDtype::Turbo4KTurbo3V => {
            req.push((
                "prefill_paged_turbo4k_turbo3v",
                "attn_prefill_paged_turbo4k_turbo3v_64",
            ));
        }
        KvCacheDtype::Turbo4KTurbo8V => {
            req.push((
                "prefill_paged_turbo4k_turbo8v",
                "attn_prefill_paged_turbo4k_turbo8v_64",
            ));
        }
        KvCacheDtype::Turbo3KTurbo8V => {
            req.push((
                "prefill_paged_turbo3k_turbo8v",
                "attn_prefill_paged_turbo3k_turbo8v_64",
            ));
        }
        KvCacheDtype::Bf16 | KvCacheDtype::Fp8 | KvCacheDtype::Nvfp4 => {}
    }
    // 2026-09-25: At these head dims the prefill path rotates the turbo side's
    // cached-prefix rows and Q with `wht_bf16_inplace`, and un-rotates the
    // output with `wht_bf16_inplace_inv`, unless the weights are pre-rotated
    // (`prefill/cache_skip.rs`, `wht_runtime_active`).
    let (k_dtype, v_dtype) = kv_dtype.kv_pair();
    if (k_dtype.is_wht_rotated() || v_dtype.is_wht_rotated()) && matches!(head_dim, 128 | 256 | 512)
    {
        req.push(("wht_bf16", "wht_bf16_inplace"));
        req.push(("wht_bf16", "wht_bf16_inplace_inv"));
    }
    req
}

/// 2026-09-25: Resolves the reshape/decode pair and
/// `required_optional_kernels_for_dtype` for one dtype, and returns an error
/// naming every kernel that is missing.
pub(super) fn validate_required_kernels(
    gpu: &dyn metrale_gpu_runtime::gpu::GpuBackend,
    kv_dtype: KvCacheDtype,
    head_dim: usize,
) -> anyhow::Result<()> {
    // 2026-09-25: The reshape/decode pair is checked here too.
    // `Qwen3AttentionLayer::new` resolves it with `gpu.kernel(..)?`, whose
    // error ("Kernel lookup <module>::<fn>") does not name the dtype.
    let (reshape_mod, reshape_fn, decode_mod, decode_fn) =
        super::init_kernel_dispatch::kernel_modules_for_dtype(kv_dtype, head_dim);
    let mut required = vec![(reshape_mod, reshape_fn), (decode_mod, decode_fn)];
    required.extend(required_optional_kernels_for_dtype(kv_dtype, head_dim));
    let missing: Vec<String> = required
        .into_iter()
        .filter(|(m, f)| gpu.kernel(m, f).is_err())
        .map(|(m, f)| format!("{m}::{f}"))
        .collect();
    if !missing.is_empty() {
        let note = if is_experimental(kv_dtype) {
            " This KV-cache dtype is EXPERIMENTAL and is not supported on this kernel target; \
             use --kv-cache-dtype fp8, bf16 or nvfp4."
        } else {
            " Rebuild kernels or pick a supported dtype."
        };
        anyhow::bail!(
            "kv-cache-dtype {kv_dtype:?} (head_dim {head_dim}) requires kernel(s) \
             missing from this build: {}.{note}",
            missing.join(", ")
        );
    }
    Ok(())
}

/// 2026-09-25: True when either side of `kv_dtype` is WHT-rotated: the turbo
/// dtypes and the asymmetric pairs that contain one.
fn is_experimental(kv_dtype: KvCacheDtype) -> bool {
    let (k, v) = kv_dtype.kv_pair();
    k.is_wht_rotated() || v.is_wht_rotated()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2026-09-25: Every dtype with a turbo side requires its chunked-prefill
    /// kernel and, at head_dim 256, the WHT bookend pair; plain dtypes require
    /// nothing; the bookends appear only at head_dim 128, 256 and 512.
    #[test]
    fn required_optional_kernels_cover_turbo_variants() {
        const TURBO: &[(KvCacheDtype, &str, &str)] = &[
            (
                KvCacheDtype::Turbo2,
                "prefill_paged_turbo2",
                "attn_prefill_paged_turbo2",
            ),
            (
                KvCacheDtype::Turbo3,
                "prefill_paged_turbo3",
                "attn_prefill_paged_turbo3_64",
            ),
            (
                KvCacheDtype::Turbo4,
                "prefill_paged_turbo4",
                "attn_prefill_paged_turbo4_64",
            ),
            (
                KvCacheDtype::Turbo8,
                "prefill_paged_turbo8",
                "attn_prefill_paged_turbo8_64",
            ),
            (
                KvCacheDtype::Bf16KTurbo3V,
                "prefill_paged_bf16k_turbo3v",
                "attn_prefill_paged_bf16k_turbo3v_64",
            ),
            (
                KvCacheDtype::Bf16KTurbo4V,
                "prefill_paged_bf16k_turbo4v",
                "attn_prefill_paged_bf16k_turbo4v_64",
            ),
            (
                KvCacheDtype::Bf16KTurbo2V,
                "prefill_paged_bf16k_turbo2v",
                "attn_prefill_paged_bf16k_turbo2v_64",
            ),
            (
                KvCacheDtype::Fp8KTurbo3V,
                "prefill_paged_fp8k_turbo3v",
                "attn_prefill_paged_fp8k_turbo3v_64",
            ),
            (
                KvCacheDtype::Fp8KTurbo4V,
                "prefill_paged_fp8k_turbo4v",
                "attn_prefill_paged_fp8k_turbo4v_64",
            ),
            (
                KvCacheDtype::Fp8KTurbo2V,
                "prefill_paged_fp8k_turbo2v",
                "attn_prefill_paged_fp8k_turbo2v_64",
            ),
            (
                KvCacheDtype::Turbo4KTurbo3V,
                "prefill_paged_turbo4k_turbo3v",
                "attn_prefill_paged_turbo4k_turbo3v_64",
            ),
            (
                KvCacheDtype::Turbo4KTurbo8V,
                "prefill_paged_turbo4k_turbo8v",
                "attn_prefill_paged_turbo4k_turbo8v_64",
            ),
            (
                KvCacheDtype::Turbo3KTurbo8V,
                "prefill_paged_turbo3k_turbo8v",
                "attn_prefill_paged_turbo3k_turbo8v_64",
            ),
        ];
        for &(d, prefill_mod, prefill_fn) in TURBO {
            assert_eq!(
                required_optional_kernels_for_dtype(d, 256),
                vec![
                    (prefill_mod, prefill_fn),
                    ("wht_bf16", "wht_bf16_inplace"),
                    ("wht_bf16", "wht_bf16_inplace_inv"),
                ],
                "{d:?}"
            );
        }
        for d in [KvCacheDtype::Bf16, KvCacheDtype::Fp8, KvCacheDtype::Nvfp4] {
            assert!(
                required_optional_kernels_for_dtype(d, 256).is_empty(),
                "{d:?}: plain dtype should require no optional kernels"
            );
        }
        assert_eq!(
            required_optional_kernels_for_dtype(KvCacheDtype::Turbo2, 64),
            vec![("prefill_paged_turbo2", "attn_prefill_paged_turbo2")]
        );
        assert_eq!(
            required_optional_kernels_for_dtype(KvCacheDtype::Turbo2, 128).len(),
            3
        );
        assert_eq!(
            required_optional_kernels_for_dtype(KvCacheDtype::Turbo2, 512).len(),
            3
        );
        assert_eq!(
            required_optional_kernels_for_dtype(KvCacheDtype::Turbo2, 513).len(),
            1
        );
    }

    /// 2026-09-25: Turbo2/3/4/8 are WHT-rotated and Bf16/Fp8/Nvfp4 are not; an
    /// asymmetric pair is judged per side.
    #[test]
    fn turbo2_is_wht_rotated() {
        for d in [
            KvCacheDtype::Turbo2,
            KvCacheDtype::Turbo3,
            KvCacheDtype::Turbo4,
            KvCacheDtype::Turbo8,
        ] {
            assert!(d.is_wht_rotated(), "{d:?} must gate the WHT bookends");
        }
        for d in [KvCacheDtype::Bf16, KvCacheDtype::Fp8, KvCacheDtype::Nvfp4] {
            assert!(!d.is_wht_rotated(), "{d:?} must not gate the WHT bookends");
        }
        let (k, v) = KvCacheDtype::Bf16KTurbo2V.kv_pair();
        assert!(!k.is_wht_rotated() && v.is_wht_rotated());
        let (k, v) = KvCacheDtype::Turbo4KTurbo8V.kv_pair();
        assert!(k.is_wht_rotated() && v.is_wht_rotated());
    }
}
