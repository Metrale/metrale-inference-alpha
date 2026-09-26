// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The reshape and paged-decode kernel names for each
//! `KvCacheDtype`, as a pure function the tests below check without a GPU.
//!
//! Owner: model-layers (attention).
//! Invariants: the match has no `_` arm, so a new `KvCacheDtype` variant does
//! not compile until it has a row.

use metrale_cache::kv_cache::KvCacheDtype;

/// 2026-09-25: `(reshape_mod, reshape_fn, decode_mod, decode_fn)` for
/// `kv_dtype`. `Qwen3AttentionLayer::new_with_gating` resolves the reshape pair
/// into `reshape_cache_k` and the decode pair into `paged_decode_k`, both as
/// hard requirements.
///
/// The turbo arms, and the asymmetric pairs built from them, use module names
/// that a model's KERNEL.toml `[modules]` table assigns (for example
/// `paged_decode_attn_turbo4 = "paged_decode_turbo4"`). A target without that
/// entry never registers the name, and
/// `kernel_requirements::validate_required_kernels` reports the missing pair
/// at startup.
pub(super) fn kernel_modules_for_dtype(
    kv_dtype: KvCacheDtype,
    head_dim: usize,
) -> (&'static str, &'static str, &'static str, &'static str) {
    let hd_le_128 = head_dim <= 128;
    match kv_dtype {
        KvCacheDtype::Nvfp4 => (
            "reshape_and_cache",
            "reshape_and_cache_flash_nvfp4",
            "paged_decode_nvfp4",
            "paged_decode_attn_nvfp4",
        ),
        KvCacheDtype::Turbo4 => (
            "reshape_and_cache_turbo",
            "reshape_and_cache_flash_turbo4",
            if hd_le_128 {
                "paged_decode_turbo4_128"
            } else {
                "paged_decode_turbo4"
            },
            "paged_decode_attn_turbo4",
        ),
        KvCacheDtype::Turbo3 => (
            "reshape_and_cache_turbo",
            "reshape_and_cache_flash_turbo3",
            if hd_le_128 {
                "paged_decode_turbo3_128"
            } else {
                "paged_decode_turbo3"
            },
            "paged_decode_attn_turbo3",
        ),
        KvCacheDtype::Turbo2 => (
            "reshape_and_cache_turbo",
            "reshape_and_cache_flash_turbo2",
            "paged_decode_turbo2_128",
            "paged_decode_attn_turbo2",
        ),
        KvCacheDtype::Turbo8 => (
            "reshape_and_cache_turbo",
            "reshape_and_cache_flash_turbo8",
            if hd_le_128 {
                "paged_decode_turbo8_128"
            } else {
                "paged_decode_turbo8"
            },
            "paged_decode_attn_turbo8",
        ),
        KvCacheDtype::Bf16KTurbo3V => (
            "reshape_and_cache_turbo",
            "reshape_and_cache_flash_bf16k_turbo3v",
            if hd_le_128 {
                "paged_decode_bf16k_turbo3v_128"
            } else {
                "paged_decode_bf16k_turbo3v"
            },
            "paged_decode_attn_bf16k_turbo3v",
        ),
        KvCacheDtype::Bf16KTurbo4V => (
            "reshape_and_cache_turbo",
            "reshape_and_cache_flash_bf16k_turbo4v",
            if hd_le_128 {
                "paged_decode_bf16k_turbo4v_128"
            } else {
                "paged_decode_bf16k_turbo4v"
            },
            "paged_decode_attn_bf16k_turbo4v",
        ),
        KvCacheDtype::Bf16KTurbo2V => (
            "reshape_and_cache_turbo",
            "reshape_and_cache_flash_bf16k_turbo2v",
            if hd_le_128 {
                "paged_decode_bf16k_turbo2v_128"
            } else {
                "paged_decode_bf16k_turbo2v"
            },
            "paged_decode_attn_bf16k_turbo2v",
        ),
        KvCacheDtype::Fp8KTurbo3V => (
            "reshape_and_cache_turbo",
            "reshape_and_cache_flash_fp8k_turbo3v",
            if hd_le_128 {
                "paged_decode_fp8k_turbo3v_128"
            } else {
                "paged_decode_fp8k_turbo3v"
            },
            "paged_decode_attn_fp8k_turbo3v",
        ),
        KvCacheDtype::Fp8KTurbo4V => (
            "reshape_and_cache_turbo",
            "reshape_and_cache_flash_fp8k_turbo4v",
            if hd_le_128 {
                "paged_decode_fp8k_turbo4v_128"
            } else {
                "paged_decode_fp8k_turbo4v"
            },
            "paged_decode_attn_fp8k_turbo4v",
        ),
        KvCacheDtype::Fp8KTurbo2V => (
            "reshape_and_cache_turbo",
            "reshape_and_cache_flash_fp8k_turbo2v",
            if hd_le_128 {
                "paged_decode_fp8k_turbo2v_128"
            } else {
                "paged_decode_fp8k_turbo2v"
            },
            "paged_decode_attn_fp8k_turbo2v",
        ),
        KvCacheDtype::Turbo4KTurbo3V => (
            "reshape_and_cache_turbo",
            "reshape_and_cache_flash_turbo4k_turbo3v",
            "paged_decode_turbo4k_turbo3v_128",
            "paged_decode_attn_turbo4k_turbo3v",
        ),
        KvCacheDtype::Turbo4KTurbo8V => (
            "reshape_and_cache_turbo",
            "reshape_and_cache_flash_turbo4k_turbo8v",
            "paged_decode_turbo4k_turbo8v_128",
            "paged_decode_attn_turbo4k_turbo8v",
        ),
        KvCacheDtype::Turbo3KTurbo8V => (
            "reshape_and_cache_turbo",
            "reshape_and_cache_flash_turbo3k_turbo8v",
            "paged_decode_turbo3k_turbo8v_128",
            "paged_decode_attn_turbo3k_turbo8v",
        ),
        KvCacheDtype::Bf16 => (
            "reshape_and_cache",
            "reshape_and_cache_flash",
            "paged_decode",
            "paged_decode_attn",
        ),
        KvCacheDtype::Fp8 => (
            "reshape_and_cache",
            "reshape_and_cache_flash_fp8",
            "paged_decode_fp8",
            "paged_decode_attn_fp8",
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2026-09-25: Every variant yields four non-empty names at head_dim 128
    /// and 256.
    #[test]
    fn every_variant_returns_non_empty_modules() {
        const ALL: &[KvCacheDtype] = &[
            KvCacheDtype::Bf16,
            KvCacheDtype::Fp8,
            KvCacheDtype::Nvfp4,
            KvCacheDtype::Turbo4,
            KvCacheDtype::Turbo3,
            KvCacheDtype::Turbo2,
            KvCacheDtype::Turbo8,
            KvCacheDtype::Bf16KTurbo3V,
            KvCacheDtype::Bf16KTurbo4V,
            KvCacheDtype::Bf16KTurbo2V,
            KvCacheDtype::Fp8KTurbo3V,
            KvCacheDtype::Fp8KTurbo4V,
            KvCacheDtype::Fp8KTurbo2V,
            KvCacheDtype::Turbo4KTurbo3V,
            KvCacheDtype::Turbo4KTurbo8V,
            KvCacheDtype::Turbo3KTurbo8V,
        ];
        for &d in ALL {
            for &hd in &[128usize, 256] {
                let (rm, rf, dm, df) = kernel_modules_for_dtype(d, hd);
                assert!(!rm.is_empty(), "{d:?} hd={hd}: empty reshape module");
                assert!(!rf.is_empty(), "{d:?} hd={hd}: empty reshape fn");
                assert!(!dm.is_empty(), "{d:?} hd={hd}: empty decode module");
                assert!(!df.is_empty(), "{d:?} hd={hd}: empty decode fn");
            }
        }
    }

    /// 2026-09-25: Each asymmetric variant routes to a reshape entry, decode
    /// module and decode entry whose names contain its dtype-pair token (for
    /// example `bf16k_turbo3v`), so it cannot land on a symmetric kernel.
    #[test]
    fn each_asym_variant_routes_to_dedicated_kernel() {
        let cases: &[(KvCacheDtype, &str)] = &[
            (KvCacheDtype::Bf16KTurbo3V, "bf16k_turbo3v"),
            (KvCacheDtype::Bf16KTurbo4V, "bf16k_turbo4v"),
            (KvCacheDtype::Bf16KTurbo2V, "bf16k_turbo2v"),
            (KvCacheDtype::Fp8KTurbo3V, "fp8k_turbo3v"),
            (KvCacheDtype::Fp8KTurbo4V, "fp8k_turbo4v"),
            (KvCacheDtype::Fp8KTurbo2V, "fp8k_turbo2v"),
            (KvCacheDtype::Turbo4KTurbo3V, "turbo4k_turbo3v"),
            (KvCacheDtype::Turbo4KTurbo8V, "turbo4k_turbo8v"),
            (KvCacheDtype::Turbo3KTurbo8V, "turbo3k_turbo8v"),
        ];
        for &(d, shape) in cases {
            for &hd in &[128usize, 256] {
                let (_rm, rf, dm, df) = kernel_modules_for_dtype(d, hd);
                assert!(
                    rf.contains(shape),
                    "{d:?} hd={hd}: reshape_fn {rf:?} missing shape token {shape:?} \
                     — silently dispatching to a non-asym kernel?"
                );
                assert!(
                    dm.contains(shape),
                    "{d:?} hd={hd}: decode_mod {dm:?} missing shape token {shape:?}"
                );
                assert!(
                    df.contains(shape),
                    "{d:?} hd={hd}: decode_fn {df:?} missing shape token {shape:?}"
                );
            }
        }
    }

    /// 2026-09-25: Each symmetric dtype routes to its own decode entry, and no
    /// symmetric dtype gets an asymmetric name.
    #[test]
    fn sym_variants_route_to_sym_kernels() {
        let cases: &[(KvCacheDtype, &str)] = &[
            (KvCacheDtype::Bf16, "paged_decode_attn"),
            (KvCacheDtype::Fp8, "paged_decode_attn_fp8"),
            (KvCacheDtype::Nvfp4, "paged_decode_attn_nvfp4"),
            (KvCacheDtype::Turbo4, "paged_decode_attn_turbo4"),
            (KvCacheDtype::Turbo3, "paged_decode_attn_turbo3"),
            (KvCacheDtype::Turbo2, "paged_decode_attn_turbo2"),
            (KvCacheDtype::Turbo8, "paged_decode_attn_turbo8"),
        ];
        for &(d, want) in cases {
            for &hd in &[128usize, 256] {
                let (_, _, _, df) = kernel_modules_for_dtype(d, hd);
                assert!(
                    df.contains(want),
                    "{d:?} hd={hd}: decode_fn {df:?} doesn't contain {want:?}"
                );
                for asym_shape in &["bf16k_", "fp8k_", "turbo4k_", "turbo3k_"] as &[&str] {
                    assert!(
                        !df.contains(asym_shape),
                        "{d:?} hd={hd}: sym dtype routed to asym kernel {df:?}"
                    );
                }
            }
        }
    }

    /// 2026-09-25: For Turbo3/4/8 and the Bf16K/Fp8K asymmetric variants,
    /// head_dim 128 selects the `_128` decode module and head_dim 256 does not.
    #[test]
    fn hd_gate_picks_128_or_full_kernel() {
        for dtype in [
            KvCacheDtype::Turbo3,
            KvCacheDtype::Turbo4,
            KvCacheDtype::Turbo8,
            KvCacheDtype::Bf16KTurbo3V,
            KvCacheDtype::Bf16KTurbo4V,
            KvCacheDtype::Bf16KTurbo2V,
            KvCacheDtype::Fp8KTurbo3V,
            KvCacheDtype::Fp8KTurbo4V,
            KvCacheDtype::Fp8KTurbo2V,
        ] {
            let (_, _, dm_128, _) = kernel_modules_for_dtype(dtype, 128);
            let (_, _, dm_256, _) = kernel_modules_for_dtype(dtype, 256);
            assert!(
                dm_128.ends_with("_128"),
                "{dtype:?} hd=128: {dm_128} should end _128"
            );
            assert!(
                !dm_256.ends_with("_128"),
                "{dtype:?} hd=256: {dm_256} should not end _128"
            );
        }
    }
}
