// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The process-scoped CUDA context and stream (`CudaHost`), and
//! `release`, which unloads one model's [`crate::registry`] modules.
//!
//! Owner: gpu-runtime.
//! Invariants:
//! - The process has at most one `CudaHost`, on the ordinal of the first `host`
//!   call; a later call with another ordinal is an error, and a failed first
//!   creation is returned as the same error to every later call.
//! - `release` unloads nothing while another `Arc` to the registry is live.

use std::sync::{Arc, OnceLock};

use cudarc::driver::{CudaContext, CudaStream};

use crate::registry::MetraleRegistry;
use metrale_core::error::{MetraleError, Result};

/// 2026-09-25: The CUDA context and stream for the process. Each model's
/// `MetraleRegistry` loads its modules into this one context.
pub struct CudaHost {
    pub ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    ordinal: usize,
}

impl CudaHost {
    pub fn ordinal(&self) -> usize {
        self.ordinal
    }
}

// 2026-09-25: SAFETY: the same argument as for `MetraleRegistry` in
// `registry.rs`: these are handles into a context that is never destroyed.
unsafe impl Send for CudaHost {}
unsafe impl Sync for CudaHost {}

/// 2026-09-25: The process's CUDA host. Static because it belongs to the
/// (process, device) pair, not to a model: it is created once and kept across
/// model loads.
static HOST: OnceLock<std::result::Result<Arc<CudaHost>, String>> = OnceLock::new();

/// 2026-09-25: Get, or on the first call create, the process CUDA host on
/// `ordinal`. The first call fixes the ordinal; asking for another is an error.
pub fn host(ordinal: usize) -> Result<Arc<CudaHost>> {
    let result = HOST.get_or_init(|| {
        let ctx = CudaContext::new(ordinal).map_err(|e| format!("{e}"))?;
        let stream = ctx.new_stream().map_err(|e| format!("{e}"))?;
        Ok(Arc::new(CudaHost {
            ctx,
            stream,
            ordinal,
        }))
    });
    match result {
        Ok(h) if h.ordinal == ordinal => Ok(h.clone()),
        Ok(h) => Err(MetraleError::ModuleLoad(format!(
            "CUDA host already bound to GPU {} — cannot rebind to {ordinal}",
            h.ordinal
        ))),
        Err(msg) => Err(MetraleError::ModuleLoad(msg.clone())),
    }
}

/// 2026-09-25: Unload a model's kernel modules. Errors, unloading nothing, while
/// another strong `Arc` to the registry is live; otherwise drops every module
/// the registry holds.
pub fn release(registry: Arc<MetraleRegistry>) -> Result<()> {
    let outstanding = Arc::strong_count(&registry) - 1;
    if outstanding > 0 {
        return Err(MetraleError::ModuleLoad(format!(
            "cannot release the kernel modules: {outstanding} handle(s) are still live. \
             Something is holding the previous model's registry — find it before swapping, \
             or the next model will run against unloaded modules."
        )));
    }
    let mut owned = Arc::try_unwrap(registry).map_err(|_| {
        MetraleError::ModuleLoad("registry handle count changed during release".to_string())
    })?;
    let failures = owned.unload_raw();
    if !failures.is_empty() {
        return Err(MetraleError::ModuleLoad(format!(
            "{} module(s) failed to unload: {}",
            failures.len(),
            failures.join("; ")
        )));
    }
    Ok(())
}
