// SPDX-License-Identifier: MIT OR Apache-2.0

use thiserror::Error;

#[derive(Error, Debug)]
pub enum MetraleError {
    #[cfg(feature = "cuda")]
    #[error("CUDA driver error: {0}")]
    CudaDriver(#[from] cudarc::driver::DriverError),

    #[error("CUDA kernel launch failed: {0}")]
    KernelLaunch(String),

    #[error("Shape mismatch: expected {expected}, got {actual}")]
    ShapeMismatch { expected: String, actual: String },

    #[error("Unsupported dtype: {0:?}")]
    UnsupportedDType(crate::dtype::DType),

    #[error("Unsupported configuration: {0}")]
    UnsupportedConfig(String),

    #[error("Device not found: {0}")]
    DeviceNotFound(String),

    #[error("Module load failed: {0}")]
    ModuleLoad(String),
}

pub type Result<T> = std::result::Result<T, MetraleError>;
