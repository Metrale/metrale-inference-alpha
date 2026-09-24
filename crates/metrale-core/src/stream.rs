// SPDX-License-Identifier: AGPL-3.0-only

use cudarc::driver::CudaStream;
use std::sync::Arc;

use crate::device::MetraleDevice;
use crate::error::Result;

/// CUDA stream wrapper for asynchronous kernel execution.
pub struct MetraleStream {
    pub stream: Arc<CudaStream>,
    pub device: MetraleDevice,
}

impl MetraleStream {
    /// Create a new CUDA stream on the given device.
    pub fn new(device: &MetraleDevice) -> Result<Self> {
        let stream = device
            .ctx
            .new_stream()
            .map_err(crate::error::MetraleError::CudaDriver)?;
        Ok(Self {
            stream,
            device: device.clone(),
        })
    }
}
