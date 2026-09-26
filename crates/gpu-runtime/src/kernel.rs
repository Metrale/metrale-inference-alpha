// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: `KernelModule`, a PTX module loaded through cudarc, plus a 1-D
//! `launch_config` helper and `launch_vector_add`.
//!
//! Owner: gpu-runtime.
//! Invariants: none beyond the types.

use std::sync::Arc;

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::Ptx;

use metrale_core::error::{MetraleError, Result};

/// 2026-09-25: A PTX module loaded into a CUDA context through cudarc.
pub struct KernelModule {
    module: Arc<CudaModule>,
}

impl KernelModule {
    /// 2026-09-25: Load PTX source text into `ctx`.
    pub fn from_ptx_src(ctx: &Arc<CudaContext>, ptx_src: &str) -> Result<Self> {
        let ptx = Ptx::from_src(ptx_src);
        let module = ctx
            .load_module(ptx)
            .map_err(|e| MetraleError::ModuleLoad(format!("PTX load failed: {e}")))?;
        Ok(Self { module })
    }

    /// 2026-09-25: Look up a kernel function by name.
    pub fn get_function(&self, name: &str) -> Result<CudaFunction> {
        self.module
            .load_function(name)
            .map_err(|e| MetraleError::ModuleLoad(format!("Function '{name}' not found: {e}")))
    }
}

/// 2026-09-25: A 1-D launch of `ceil(n / block_size)` blocks of `block_size`
/// threads, with no dynamic shared memory.
pub fn launch_config(n: u32, block_size: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (n.div_ceil(block_size), 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// 2026-09-25: Launch `func` with `(a_ptr, b_ptr, c_ptr, n)` in `ceil(n / 256)`
/// blocks of 256 threads, through cudarc's launch builder.
///
/// # Safety
///
/// All pointers must be valid CUDA device pointers to f32 arrays of length >= n.
pub unsafe fn launch_vector_add(
    stream: &Arc<CudaStream>,
    func: &CudaFunction,
    a_ptr: u64,
    b_ptr: u64,
    c_ptr: u64,
    n: u32,
) -> Result<()> {
    let cfg = launch_config(n, 256);
    unsafe {
        stream
            .launch_builder(func)
            .arg(&a_ptr)
            .arg(&b_ptr)
            .arg(&c_ptr)
            .arg(&n)
            .launch(cfg)
            .map_err(|e| MetraleError::KernelLaunch(format!("vector_add launch failed: {e}")))?;
    }
    Ok(())
}
