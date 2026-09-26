// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: [`MetraleStream`]: a CUDA stream paired with the device it was created on.
//!
//! Owner: gpu-runtime.
//! Invariants: none beyond the types.

use cudarc::driver::CudaStream;
use std::sync::Arc;

use metrale_core::device::MetraleDevice;
use metrale_core::error::Result;

/// 2026-09-25: A CUDA stream and the device whose context created it.
pub struct MetraleStream {
    pub stream: Arc<CudaStream>,
    pub device: MetraleDevice,
}

impl MetraleStream {
    /// 2026-09-25: Create a stream on `device`'s context (cudarc `new_stream`, non-blocking).
    pub fn new(device: &MetraleDevice) -> Result<Self> {
        let stream = device
            .ctx
            .new_stream()
            .map_err(metrale_core::error::MetraleError::CudaDriver)?;
        Ok(Self {
            stream,
            device: device.clone(),
        })
    }
}
