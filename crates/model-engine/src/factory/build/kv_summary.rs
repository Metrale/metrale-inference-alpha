// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The `--high-speed-swap` KV summary log line: attention-layer count per KV dtype label.
//!
//! Owner: model-engine factory.
//! Invariants: none beyond the types.

use metrale_cache::kv_cache::{KvCacheConfig, KvCacheDtype};

/// 2026-09-25: Summary label for a KV dtype. Mixed K/V dtypes are counted under their K dtype;
/// `Turbo2` is counted under "Turbo3".
fn dtype_label(dt: KvCacheDtype) -> &'static str {
    match dt {
        KvCacheDtype::Bf16
        | KvCacheDtype::Bf16KTurbo4V
        | KvCacheDtype::Bf16KTurbo3V
        | KvCacheDtype::Bf16KTurbo2V => "BF16",
        KvCacheDtype::Fp8
        | KvCacheDtype::Fp8KTurbo4V
        | KvCacheDtype::Fp8KTurbo3V
        | KvCacheDtype::Fp8KTurbo2V => "FP8",
        KvCacheDtype::Nvfp4 => "NVFP4",
        KvCacheDtype::Turbo3 | KvCacheDtype::Turbo3KTurbo8V | KvCacheDtype::Turbo2 => "Turbo3",
        KvCacheDtype::Turbo4 | KvCacheDtype::Turbo4KTurbo3V | KvCacheDtype::Turbo4KTurbo8V => {
            "Turbo4"
        }
        KvCacheDtype::Turbo8 => "Turbo8",
    }
}

pub(super) fn log_hss_kv_summary(kv_config: &KvCacheConfig) {
    let mut counts: std::collections::BTreeMap<&'static str, usize> =
        std::collections::BTreeMap::new();
    if kv_config.layer_dtypes.is_empty() {
        *counts.entry(dtype_label(kv_config.dtype)).or_default() += kv_config.num_layers;
    } else {
        for dt in &kv_config.layer_dtypes {
            *counts.entry(dtype_label(*dt)).or_default() += 1;
        }
    }
    let total: usize = counts.values().sum();
    let summary: Vec<String> = counts
        .iter()
        .map(|(name, n)| format!("{n} {name}"))
        .collect();
    tracing::info!(
        "--high-speed-swap KV: {} attn layers ({}); HBM-shrink applies to all \
         (Phase 6.2.c proper — host dequant for FP8/NVFP4/Turbo3/Turbo4/Turbo8; \
         predictor scoring uses LRU for non-BF16 layers)",
        total,
        summary.join(" + ")
    );
}
