// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Small device-side helpers: BF16 buffer copy + optional debug dump.
//!
//! Owner: model-layers (vision).
//! Invariants: none beyond the types.

use anyhow::{Context, Result};
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};
use metrale_gpu_runtime::kernel_args::{KernelLaunch, div_ceil};

use super::super::VisionEncoder;

impl VisionEncoder {
    /// 2026-09-25: Copy `n_bytes` of a BF16 device buffer to another with the
    /// `vision_bf16_copy` kernel, which takes a u32 element count (not bytes).
    pub(super) fn gpu_copy_bf16(
        &self,
        gpu: &dyn GpuBackend,
        src: DevicePtr,
        dst: DevicePtr,
        n_bytes: usize,
        stream: u64,
    ) -> Result<()> {
        let n_elts = (n_bytes / 2) as u32;
        KernelLaunch::new(gpu, self.k_copy)
            .grid([div_ceil(n_elts, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(src)
            .arg_ptr(dst)
            .arg_u32(n_elts)
            .launch(stream)
    }

    /// 2026-09-25: Debug hook: when `METRALE_DUMP_VIT=<dir>` is set and non-empty,
    /// synchronise `stream` and write `n_elements` BF16 values from `ptr` to
    /// `<dir>/<label>.bin`, plain little-endian with no header.
    pub(super) fn maybe_dump_buf(
        gpu: &dyn GpuBackend,
        ptr: DevicePtr,
        n_elements: usize,
        label: &str,
        stream: u64,
    ) -> Result<()> {
        let Ok(dir) = std::env::var("METRALE_DUMP_VIT") else {
            return Ok(());
        };
        if dir.is_empty() {
            return Ok(());
        }
        gpu.synchronize(stream)?;
        let bytes = n_elements * 2;
        let mut buf = vec![0u8; bytes];
        gpu.copy_d2h(ptr, &mut buf)?;
        let path = std::path::Path::new(&dir).join(format!("{label}.bin"));
        std::fs::create_dir_all(&dir).ok();
        std::fs::write(&path, &buf).with_context(|| format!("write {}", path.display()))?;
        tracing::info!(
            "METRALE_DUMP_VIT: wrote {} ({} elements)",
            path.display(),
            n_elements
        );
        Ok(())
    }
}
