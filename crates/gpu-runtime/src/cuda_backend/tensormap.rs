// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: TMA descriptors (`CUtensorMap`) for `cp.async.bulk.tensor` loads,
//! encoded with the driver's `cuTensorMapEncodeTiled`.
//!
//! Owner: gpu-runtime.
//! Invariants:
//! - `TensorMap` is 128 bytes aligned to 64, the size of `CUtensorMap` and the
//!   alignment `cuTensorMapEncodeTiled` requires (`cuda.h`); a test pins both.
//! - `tiled_2d_bf16` rejects a global address or row stride that is not 16-byte
//!   aligned, a row stride narrower than `cols`, and a zero box dimension, before
//!   it calls the driver.
//!
//! The driver takes dimensions fastest-varying first and `rank - 1` strides in
//! bytes, the innermost stride being the element size. `tiled_2d_bf16` takes
//! row-major `[rows][cols]` in elements and does that conversion. A descriptor is
//! passed by value to a `__grid_constant__ const CUtensorMap` kernel parameter
//! through `KernelLaunch::arg_tensormap`.

use anyhow::{Result, bail};

use crate::gpu::DevicePtr;

/// 2026-09-25: Enum values from `CUtensorMapDataType` and its neighbours in
/// `/usr/local/cuda/include/cuda.h` (CUDA 13.0): `BFLOAT16` is 9, after
/// `FLOAT64` (8) and before `FLOAT32_FTZ` (10).
const CU_TENSOR_MAP_DATA_TYPE_BFLOAT16: u32 = 9;
const CU_TENSOR_MAP_INTERLEAVE_NONE: u32 = 0;
const CU_TENSOR_MAP_SWIZZLE_NONE: u32 = 0;
const CU_TENSOR_MAP_L2_PROMOTION_L2_128B: u32 = 2;
const CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE: u32 = 0;

/// 2026-09-25: The prototype of `cuTensorMapEncodeTiled` in `cuda.h`. The symbol is
/// found with `dlsym(RTLD_DEFAULT)` in the libcuda the process has already loaded,
/// once, and cached (`tensor_map_encode_tiled`).
type EncodeTiledFn = unsafe extern "C" fn(
    *mut u8,
    u32,
    u32,
    u64,
    *const u64,
    *const u64,
    *const u32,
    *const u32,
    u32,
    u32,
    u32,
    u32,
) -> i32;

fn tensor_map_encode_tiled() -> Option<EncodeTiledFn> {
    static CACHED: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    let addr = (*CACHED.get_or_init(|| {
        // 2026-09-25: `dlsym` is unix-only. Elsewhere the symbol is reported
        // missing and `tiled_2d_bf16` returns an error.
        #[cfg(unix)]
        {
            unsafe extern "C" {
                fn dlsym(handle: *mut std::ffi::c_void, symbol: *const i8)
                -> *mut std::ffi::c_void;
            }
            let name = c"cuTensorMapEncodeTiled";
            let p = unsafe { dlsym(std::ptr::null_mut(), name.as_ptr().cast::<i8>()) };
            if p.is_null() { None } else { Some(p as usize) }
        }
        #[cfg(not(unix))]
        {
            None
        }
    }))?;
    // 2026-09-25: SAFETY: `addr` is libcuda's `cuTensorMapEncodeTiled`, whose
    // prototype in `cuda.h` is `EncodeTiledFn`.
    Some(unsafe { std::mem::transmute::<usize, EncodeTiledFn>(addr) })
}

/// 2026-09-25: A 128-byte TMA descriptor, aligned to 64 bytes.
#[repr(C, align(64))]
#[derive(Clone, Copy)]
pub struct TensorMap([u8; 128]);

impl std::fmt::Debug for TensorMap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("TensorMap(<128 bytes>)")
    }
}

impl TensorMap {
    /// 2026-09-25: The raw bytes, for `KernelLaunch::arg_tensormap`.
    pub fn bytes(&self) -> &[u8; 128] {
        &self.0
    }

    /// 2026-09-25: Describe a row-major bf16 matrix as tiles of `box_rows x box_cols`.
    ///
    /// `rows`, `cols` and `row_stride_elems` are in elements. `row_stride_elems`
    /// is the distance between consecutive rows and may exceed `cols` (a view into
    /// a wider tensor). Out-of-bounds elements of a tile are filled with zero
    /// (`CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE`).
    ///
    /// Errors: a misaligned address or stride, a stride below `cols`, a zero box
    /// dimension, a libcuda without `cuTensorMapEncodeTiled`, or a non-zero
    /// `CUresult` from the encode.
    pub fn tiled_2d_bf16(
        global: DevicePtr,
        rows: u64,
        cols: u64,
        row_stride_elems: u64,
        box_rows: u32,
        box_cols: u32,
    ) -> Result<Self> {
        if !global.0.is_multiple_of(16) {
            bail!("TMA global address {:#x} is not 16-byte aligned", global.0);
        }
        let row_stride_bytes = row_stride_elems * 2;
        if !row_stride_bytes.is_multiple_of(16) {
            bail!(
                "TMA row stride {row_stride_elems} elems ({row_stride_bytes} B) is not \
                 16-byte aligned; bf16 needs a stride that is a multiple of 8 elements"
            );
        }
        if row_stride_elems < cols {
            bail!("TMA row stride {row_stride_elems} is narrower than cols {cols}");
        }
        if box_rows == 0 || box_cols == 0 {
            bail!("TMA box dims must be non-zero, got {box_rows}x{box_cols}");
        }

        // 2026-09-25: Fastest-varying axis first: for row-major [rows][cols]
        // that is cols.
        let global_dim: [u64; 2] = [cols, rows];
        // 2026-09-25: rank - 1 strides, in bytes; the innermost is implicit.
        let global_strides: [u64; 1] = [row_stride_bytes];
        let box_dim: [u32; 2] = [box_cols, box_rows];
        let element_strides: [u32; 2] = [1, 1];

        // 2026-09-25: Resolved at runtime rather than linked, so the crate links
        // against a libcuda that lacks the symbol; such a driver gets this error.
        let f = tensor_map_encode_tiled().ok_or_else(|| {
            anyhow::anyhow!(
                "cuTensorMapEncodeTiled not present in libcuda — TMA needs a CUDA 12.0+ driver"
            )
        })?;
        let mut map = TensorMap([0u8; 128]);
        let rc = unsafe {
            f(
                map.0.as_mut_ptr(),
                CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                2,
                global.0,
                global_dim.as_ptr(),
                global_strides.as_ptr(),
                box_dim.as_ptr(),
                element_strides.as_ptr(),
                CU_TENSOR_MAP_INTERLEAVE_NONE,
                CU_TENSOR_MAP_SWIZZLE_NONE,
                CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
            )
        };
        if rc != 0 {
            bail!(
                "cuTensorMapEncodeTiled failed: CUresult {rc} \
                 (rows={rows} cols={cols} stride={row_stride_elems} box={box_rows}x{box_cols})"
            );
        }
        Ok(map)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2026-09-25: The guards reject before the driver is called, so these tests
    /// need no CUDA context.
    #[test]
    fn misaligned_stride_is_rejected_without_touching_the_driver() {
        // 2026-09-25: A stride of 5 bf16 elements is 10 bytes.
        let e = TensorMap::tiled_2d_bf16(DevicePtr(0x1000), 4, 5, 5, 2, 5).unwrap_err();
        assert!(
            e.to_string().contains("not 16-byte aligned"),
            "expected a stride-alignment error, got: {e}"
        );
    }

    #[test]
    fn misaligned_address_is_rejected() {
        let e = TensorMap::tiled_2d_bf16(DevicePtr(0x1004), 4, 8, 8, 2, 8).unwrap_err();
        assert!(e.to_string().contains("not 16-byte aligned"), "got: {e}");
    }

    #[test]
    fn a_stride_narrower_than_the_row_is_rejected() {
        let e = TensorMap::tiled_2d_bf16(DevicePtr(0x1000), 4, 64, 32, 2, 64).unwrap_err();
        assert!(e.to_string().contains("narrower than cols"), "got: {e}");
    }

    #[test]
    fn the_descriptor_has_the_layout_the_driver_requires() {
        assert_eq!(std::mem::size_of::<TensorMap>(), 128);
        assert_eq!(std::mem::align_of::<TensorMap>(), 64);
    }
}
