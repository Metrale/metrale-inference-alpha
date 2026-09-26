// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Per-attention-layer KV cache dtypes: the boundary-layer rule and
//! the automatic boundary count for turbo dtypes.
//!
//! Owner: server startup (`met serve`).
//! Invariants: none beyond the types.

/// 2026-09-26: The boundary-layer count used when `--kv-high-precision-layers`
/// resolves to 0, its default (`serve_phases/kv_cache.rs`), for n attention
/// layers: `None` for bf16, fp8 and nvfp4; ceil(4n/5), at least 4, for
/// `turbo2` and `bf16k_turbo3v`; ceil(n/3), at least 2, for the other turbo
/// dtypes.
pub(crate) fn auto_high_precision_layers(
    kv_dtype: metrale_cache::kv_cache::KvCacheDtype,
    num_attention_layers: usize,
) -> Option<usize> {
    use metrale_cache::kv_cache::KvCacheDtype as D;
    match kv_dtype {
        D::Bf16 | D::Fp8 | D::Nvfp4 => None,
        D::Turbo2 | D::Bf16KTurbo3V => Some(((num_attention_layers * 4).div_ceil(5)).max(4)),
        D::Turbo3
        | D::Turbo4
        | D::Turbo8
        | D::Turbo4KTurbo3V
        | D::Turbo4KTurbo8V
        | D::Turbo3KTurbo8V
        | D::Bf16KTurbo4V
        | D::Fp8KTurbo4V
        | D::Fp8KTurbo3V
        | D::Bf16KTurbo2V
        | D::Fp8KTurbo2V => Some(((num_attention_layers as f32 / 3.0).ceil() as usize).max(2)),
    }
}

/// 2026-09-26: Per-attention-layer KV dtypes: the first and last
/// `high_precision_layers` layers (capped at the layer count) take
/// `boundary_dtype`, the rest `kv_dtype`. Empty when `high_precision_layers`
/// is 0 or the two dtypes are equal.
pub(crate) fn build_layer_kv_dtypes(
    kv_dtype: metrale_cache::kv_cache::KvCacheDtype,
    num_attention_layers: usize,
    high_precision_layers: usize,
    boundary_dtype: metrale_cache::kv_cache::KvCacheDtype,
) -> Vec<metrale_cache::kv_cache::KvCacheDtype> {
    if high_precision_layers == 0 || kv_dtype == boundary_dtype {
        return vec![];
    }

    let hp = high_precision_layers.min(num_attention_layers);
    let mut dtypes = vec![kv_dtype; num_attention_layers];

    for i in 0..hp.min(num_attention_layers) {
        dtypes[i] = boundary_dtype;
    }
    for i in num_attention_layers.saturating_sub(hp)..num_attention_layers {
        dtypes[i] = boundary_dtype;
    }

    let hp_count = dtypes.iter().filter(|d| **d == boundary_dtype).count();
    tracing::info!(
        "Selective boundary KV cache: {}/{} attention layers at {}, rest at {}",
        hp_count,
        num_attention_layers,
        boundary_dtype,
        kv_dtype,
    );

    dtypes
}
