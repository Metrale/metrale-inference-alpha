// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Debug taps: dump the `hc_mult`-wide residual highway and other
//! buffers at named points, for a layer-by-layer diff against the reference.
//!
//! Off unless `METRALE_QWEN4EXP_DUMP` names a directory. `tap_highway` and
//! `tap_f32` write `<dir>/L{layer:02}_{tag}.bin` as raw little-endian FP32;
//! `tap_bf16` writes `<dir>/L{layer:02}_{tag}.bf16.bin` as raw BF16.
//!
//! Owner: model-layers (PLE).
//! Invariants:
//! - A tap never overwrites an existing file.

use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

/// 2026-09-25: Directory from `METRALE_QWEN4EXP_DUMP` (empty means unset),
/// resolved once.
fn dump_dir() -> Option<&'static str> {
    static DIR: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    DIR.get_or_init(|| {
        let d = std::env::var("METRALE_QWEN4EXP_DUMP")
            .ok()
            .filter(|s| !s.is_empty());
        if let Some(ref path) = d {
            let _ = std::fs::create_dir_all(path);
            tracing::warn!(
                "METRALE_QWEN4EXP_DUMP={path}: taping the mHC highway to disk. \
                 This SYNCHRONIZES and copies D2H at every tap — a debug aid, \
                 not a serving mode."
            );
        }
        d
    })
    .as_deref()
}

/// 2026-09-25: Whether `path` is free to write; an existing tap is never
/// overwritten. The SSM taps are labelled by `SSM_LAYER_CALL_COUNTER`, which
/// counts every SSM layer call and never resets, so only the first prefill
/// call after startup carries true layer numbers; later calls write under
/// higher labels.
fn claim(path: &str) -> bool {
    !std::path::Path::new(path).exists()
}

/// 2026-09-25: Tap the FP32 highway, `num_tokens * hc_dim` values. No-op
/// unless the dump directory is set. It synchronizes the stream first, so it
/// must not run inside CUDA-graph capture. Errors are logged or dropped,
/// never returned.
pub fn tap_highway(
    gpu: &dyn GpuBackend,
    streams: DevicePtr,
    layer: usize,
    tag: &str,
    num_tokens: usize,
    hc_dim: usize,
    stream: u64,
) {
    let Some(dir) = dump_dir() else {
        return;
    };
    let path = format!("{dir}/L{layer:02}_{tag}.bin");
    if !claim(&path) {
        return;
    }
    if gpu.synchronize(stream).is_err() {
        return;
    }
    let mut raw = vec![0u8; num_tokens * hc_dim * 4];
    if gpu.copy_d2h(streams, &mut raw).is_err() {
        return;
    }
    let path = format!("{dir}/L{layer:02}_{tag}.bin");
    if let Err(e) = std::fs::write(&path, &raw) {
        tracing::warn!("highway tap {path}: {e}");
    }
}

/// 2026-09-25: Tap a BF16 buffer of `n_elements` the same way.
pub fn tap_bf16(
    gpu: &dyn GpuBackend,
    ptr: DevicePtr,
    layer: usize,
    tag: &str,
    n_elements: usize,
    stream: u64,
) {
    let Some(dir) = dump_dir() else {
        return;
    };
    let path = format!("{dir}/L{layer:02}_{tag}.bf16.bin");
    if !claim(&path) {
        return;
    }
    if gpu.synchronize(stream).is_err() {
        return;
    }
    let mut raw = vec![0u8; n_elements * 2];
    if gpu.copy_d2h(ptr, &mut raw).is_err() {
        return;
    }
    if let Err(e) = std::fs::write(&path, &raw) {
        tracing::warn!("highway tap {path}: {e}");
    }
}

/// 2026-09-25: Tap an FP32 buffer of `n_elements` the same way.
pub fn tap_f32(
    gpu: &dyn GpuBackend,
    ptr: DevicePtr,
    layer: usize,
    tag: &str,
    n_elements: usize,
    stream: u64,
) {
    let Some(dir) = dump_dir() else {
        return;
    };
    let path = format!("{dir}/L{layer:02}_{tag}.bin");
    if !claim(&path) {
        return;
    }
    if gpu.synchronize(stream).is_err() {
        return;
    }
    let mut raw = vec![0u8; n_elements * 4];
    if gpu.copy_d2h(ptr, &mut raw).is_err() {
        return;
    }
    if let Err(e) = std::fs::write(&path, &raw) {
        tracing::warn!("highway tap {path}: {e}");
    }
}
