// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Unit tests for [`crate::gpu`] against the mock backend. The file
//! is included with `#[path]` as the `gpu::tests` module.
//!
//! Owner: gpu-runtime.
//! Invariants: none beyond the types.

use super::mock::MockGpuBackend;
use super::*;

#[test]
fn test_mock_alloc_free() {
    let gpu = MockGpuBackend::new();
    let ptr = gpu.alloc(1024).unwrap();
    assert!(!ptr.is_null());
    assert_eq!(gpu.alloc_count(), 1);
    gpu.free(ptr).unwrap();
    assert_eq!(gpu.alloc_count(), 0);
}

#[test]
fn mock_free_rejects_interior_and_repeated_pointers() {
    let gpu = MockGpuBackend::new();
    let ptr = gpu.alloc(1024).unwrap();
    let interior = ptr.offset(256);
    assert!(gpu.free(interior).is_err());
    assert_eq!(gpu.alloc_count(), 1, "interior free must preserve owner");
    gpu.free(ptr).unwrap();
    assert!(gpu.free(ptr).is_err());
}

#[test]
fn test_mock_copy_roundtrip() {
    let gpu = MockGpuBackend::new();
    let ptr = gpu.alloc(8).unwrap();
    let src = [1u8, 2, 3, 4, 5, 6, 7, 8];
    gpu.copy_h2d(&src, ptr).unwrap();
    let mut dst = [0u8; 8];
    gpu.copy_d2h(ptr, &mut dst).unwrap();
    assert_eq!(src, dst);
}

#[test]
fn test_device_ptr_offset() {
    let ptr = DevicePtr(0x1000);
    assert_eq!(ptr.offset(256).0, 0x1100);
}
