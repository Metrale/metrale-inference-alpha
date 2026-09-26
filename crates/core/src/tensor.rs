// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `TensorRef`, a non-owning description of a device tensor.
//!
//! Owner: metrale-core.
//! Invariants: none beyond the types.

use crate::dtype::DType;

/// 2026-09-25: A non-owning reference to a device tensor: a raw pointer plus
/// shape, strides and element type. It neither allocates nor frees.
#[derive(Debug, Clone)]
pub struct TensorRef {
    pub ptr: u64,

    pub shape: Vec<usize>,

    /// 2026-09-25: Strides in elements, not bytes.
    pub strides: Vec<usize>,

    pub dtype: DType,
}

impl TensorRef {
    /// 2026-09-25: Create a tensor reference with contiguous row-major strides.
    pub fn new(ptr: u64, shape: Vec<usize>, dtype: DType) -> Self {
        let strides = Self::contiguous_strides(&shape);
        Self {
            ptr,
            shape,
            strides,
            dtype,
        }
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// 2026-09-25: Total size in bytes, rounded up for sub-byte element types.
    pub fn size_bytes(&self) -> usize {
        let bits = self.numel() * self.dtype.element_size_bits();
        bits.div_ceil(8)
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    fn contiguous_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![1usize; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }

    pub fn as_device_ptr<T>(&self) -> *const T {
        self.ptr as *const T
    }

    pub fn as_device_ptr_mut<T>(&self) -> *mut T {
        self.ptr as *mut T
    }
}
