// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: GB10 hardware constants (`sm121`) and the cudarc device handle `MetraleDevice`.
//!
//! `sm121` compiles on every feature set; `MetraleDevice` only with the `cuda`
//! feature.
//!
//! Owner: core.
//! Invariants: none beyond the types.

/// 2026-09-25: GB10 (SM 12.1) hardware constants. No code in the workspace
/// reads them; `NUM_SMS` and `MEMORY_BW_GBS` match `sm_count` and
/// `memory_bandwidth_gbps` in `kernels/gb10/HARDWARE.toml`.
pub mod sm121 {
    pub const NUM_SMS: u32 = 48;

    /// 2026-09-25: Shared memory per SM, in bytes.
    pub const SMEM_PER_SM: usize = 99 * 1024;

    pub const MAX_REGS_PER_THREAD: u32 = 255;

    pub const MAX_THREADS_PER_BLOCK: u32 = 1024;

    pub const WARP_SIZE: u32 = 32;

    /// 2026-09-25: Memory bandwidth in GB/s.
    pub const MEMORY_BW_GBS: f64 = 273.0;

    pub const COMPUTE_MAJOR: u32 = 12;
    pub const COMPUTE_MINOR: u32 = 1;
}

#[cfg(feature = "cuda")]
mod cuda_impl {
    use cudarc::driver::CudaContext;
    use std::sync::Arc;

    use crate::error::{MetraleError, Result};

    /// 2026-09-25: A cudarc `CudaContext` and the GPU ordinal it was created on.
    #[derive(Clone)]
    pub struct MetraleDevice {
        pub ctx: Arc<CudaContext>,
        pub ordinal: usize,
    }

    impl MetraleDevice {
        /// 2026-09-25: Create a CUDA context on GPU `ordinal`; a driver failure
        /// is returned as `MetraleError::CudaDriver`.
        pub fn new(ordinal: usize) -> Result<Self> {
            let ctx = CudaContext::new(ordinal).map_err(MetraleError::CudaDriver)?;
            Ok(Self { ctx, ordinal })
        }
    }
}

#[cfg(feature = "cuda")]
pub use cuda_impl::MetraleDevice;
