// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Read-side data for the dashboard: the Library row model and HF
//! cache scan, the kernel table, the metrics sampler, the thermal probe, and
//! free GPU memory.
//!
//! Owner: server tui.
//! Invariants: none beyond the types.

pub mod catalogue;
pub mod kernels;
pub mod library;
pub mod metrics_poll;
pub mod thermal;

/// 2026-09-26: Free GPU memory in bytes from
/// `metrale_gpu_runtime::cuda_backend::cuda_free_memory_bytes`; always `None`
/// in a build without the `cuda` feature, where that module does not exist.
pub fn gpu_free_bytes() -> Option<usize> {
    #[cfg(feature = "cuda")]
    {
        metrale_gpu_runtime::cuda_backend::cuda_free_memory_bytes()
    }
    #[cfg(not(feature = "cuda"))]
    {
        None
    }
}
