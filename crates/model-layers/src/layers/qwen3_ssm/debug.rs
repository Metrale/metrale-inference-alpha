// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Debug print helpers and the `METRALE_GDN_DUMP` dumper of GDN prefill intermediates, for per-layer comparison against a CPU reference.
//!
//! Owner: model-layers (qwen3 SSM).
//! Invariants: none beyond the types.
//!
//! `METRALE_GDN_DUMP=<dir>` turns the dumper on; `METRALE_GDN_DUMP_LAYERS`
//! (comma-separated SSM-layer indices, default `0`) picks the layers. Each
//! file is `gdnsub_step0_L{layer}_{stage}.bin`: headerless raw bytes, two per
//! element, of the last token's slice.
//! `bench/longcode/hang-forensics/gdn_chain_diff_a3b.py` reads them as BF16.

use std::sync::atomic::{AtomicBool, AtomicUsize};

use super::*;

/// 2026-09-25: Size of the per-layer arrays below. Qwen3.6-35B-A3B has 30 SSM
/// layers and Qwen3.8-27B has 48 (their `config.json` `layer_types`).
pub(super) const MAX_SSM_LAYERS: usize = 64;

/// 2026-09-25: Incremented by each SSM layer's prefill call, so during one
/// prefill the N SSM layers see N consecutive values; it never resets.
pub(super) static SSM_LAYER_CALL_COUNTER: AtomicUsize = AtomicUsize::new(0);

// 2026-09-25: One flag per SSM layer and stage. `maybe_dump_gdn_buf` takes one
// of these arrays but does not read it.
macro_rules! atomic_bool_array {
    () => {{
        // 2026-09-25: `const ELEM` + `[ELEM; N]` is the fixed-size
        // array-of-atomics idiom; clippy's interior-mutable-const lint does not
        // apply to it.
        #[allow(clippy::declare_interior_mutable_const)]
        const ELEM: AtomicBool = AtomicBool::new(false);
        [ELEM; MAX_SSM_LAYERS]
    }};
}
pub(super) static DUMP_CONV: [AtomicBool; MAX_SSM_LAYERS] = atomic_bool_array!();
pub(super) static DUMP_L2: [AtomicBool; MAX_SSM_LAYERS] = atomic_bool_array!();
pub(super) static DUMP_GDN: [AtomicBool; MAX_SSM_LAYERS] = atomic_bool_array!();
pub(super) static DUMP_GNORM: [AtomicBool; MAX_SSM_LAYERS] = atomic_bool_array!();

fn dump_layers_from_env() -> Vec<usize> {
    std::env::var("METRALE_GDN_DUMP_LAYERS")
        .unwrap_or_else(|_| "0".to_string())
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect()
}

/// 2026-09-25: Write `n_elements` two-byte values starting at
/// `ptr + byte_offset` to `<METRALE_GDN_DUMP>/gdnsub_step0_L{layer}_{stage}.bin`,
/// where `layer` is `layer_idx % METRALE_GDN_DUMP_N_SSM`. Skips when
/// `METRALE_GDN_DUMP` is unset or empty, or that layer is not in
/// `METRALE_GDN_DUMP_LAYERS`. Every call overwrites the file.
pub(super) fn maybe_dump_gdn_buf(
    gpu: &dyn GpuBackend,
    ptr: DevicePtr,
    byte_offset: usize,
    n_elements: usize,
    layer_idx: usize,
    stage: &str,
    _unused_latch: &[AtomicBool; MAX_SSM_LAYERS],
    stream: u64,
) -> Result<()> {
    // 2026-09-25: No latch: under chunked prefill this runs once per chunk per
    // SSM layer and overwrites the file, so the file left on disk is the
    // last chunk's capture. Reducing the ever-growing call counter modulo the
    // SSM layer count maps each chunk's calls back to the same layers.
    let dir = match std::env::var("METRALE_GDN_DUMP") {
        Ok(d) if !d.is_empty() => d,
        _ => return Ok(()),
    };
    // 2026-09-25: The SSM layer count, from `METRALE_GDN_DUMP_N_SSM` (default
    // 30, the Qwen3.6-35B-A3B count; Qwen3.8-27B needs 48).
    let n_ssm: usize = std::env::var("METRALE_GDN_DUMP_N_SSM")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(30);
    let effective_layer = layer_idx % n_ssm.max(1);
    if effective_layer >= MAX_SSM_LAYERS {
        return Ok(());
    }
    let layers = dump_layers_from_env();
    if !layers.contains(&effective_layer) {
        return Ok(());
    }
    gpu.synchronize(stream)?;
    let bytes = n_elements * 2;
    let mut buf = vec![0u8; bytes];
    gpu.copy_d2h(ptr.offset(byte_offset), &mut buf)?;
    let path =
        std::path::Path::new(&dir).join(format!("gdnsub_step0_L{effective_layer}_{stage}.bin"));
    std::fs::create_dir_all(&dir).ok();
    std::fs::write(&path, &buf)?;
    tracing::info!(
        "METRALE_GDN_DUMP: wrote {} ({} bf16 elements, raw_idx={layer_idx}, effective_layer={effective_layer})",
        path.display(),
        n_elements
    );
    Ok(())
}

impl Qwen3SsmLayer {
    /// 2026-09-25: Log the first `n` BF16 values at `ptr`; returns quietly if the
    /// copy fails.
    pub(super) fn debug_bf16(gpu: &dyn GpuBackend, label: &str, ptr: DevicePtr, n: usize) {
        let mut buf = vec![0u8; n * 2];
        if gpu.copy_d2h(ptr, &mut buf).is_err() {
            return;
        }
        let vals: Vec<f32> = (0..n)
            .map(|i| {
                let lo = buf[i * 2];
                let hi = buf[i * 2 + 1];
                f32::from_bits(((lo as u32) | ((hi as u32) << 8)) << 16)
            })
            .collect();
        tracing::info!("  SSM {label}: {:?}", vals);
    }

    /// 2026-09-25: Log the first `n` FP32 values at `ptr`; returns quietly if the
    /// copy fails.
    pub(super) fn debug_f32(gpu: &dyn GpuBackend, label: &str, ptr: DevicePtr, n: usize) {
        let mut buf = vec![0u8; n * 4];
        if gpu.copy_d2h(ptr, &mut buf).is_err() {
            return;
        }
        let vals: Vec<f32> = (0..n)
            .map(|i| {
                f32::from_le_bytes([buf[i * 4], buf[i * 4 + 1], buf[i * 4 + 2], buf[i * 4 + 3]])
            })
            .collect();
        tracing::info!("  SSM {label}: {:?}", vals);
    }
}
