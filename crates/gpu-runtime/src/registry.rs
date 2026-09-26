// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: [`MetraleRegistry`]: one model's kernel modules, loaded into the
//! process CUDA context, with raw driver-API helpers for launches, copies and
//! device symbols.
//!
//! Owner: gpu-runtime.
//! Invariants:
//! - Each raw handle in `raw_modules` is a view of the cudarc module stored under
//!   the same name in `modules`; `unload_raw` drains both maps together.
//! - A registry owns neither the context nor the stream; both belong to the
//!   process-scoped [`CudaHost`].

use std::collections::HashMap;
use std::ffi::{CString, c_void};
use std::sync::{Arc, OnceLock};

use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaStream, LaunchConfig};
use cudarc::nvrtc::Ptx;

pub use crate::cuda_host::{CudaHost, host, release};
use metrale_core::error::{MetraleError, Result};

unsafe extern "C" {
    fn cuModuleGetFunction(hfunc: *mut *mut c_void, hmod: *mut c_void, name: *const i8) -> i32;
    fn cuLaunchKernel(
        f: *mut c_void,
        gridDimX: u32,
        gridDimY: u32,
        gridDimZ: u32,
        blockDimX: u32,
        blockDimY: u32,
        blockDimZ: u32,
        sharedMemBytes: u32,
        hStream: *mut c_void,
        kernelParams: *mut *mut c_void,
        extra: *mut *mut c_void,
    ) -> i32;
    fn cuFuncSetAttribute(hfunc: *mut c_void, attrib: i32, value: i32) -> i32;
    fn cuGetErrorName(error: i32, pStr: *mut *const i8) -> i32;
    fn cuGetErrorString(error: i32, pStr: *mut *const i8) -> i32;
    fn cuModuleGetGlobal_v2(
        dptr: *mut u64,
        bytes: *mut usize,
        hmod: *mut c_void,
        name: *const i8,
    ) -> i32;
    fn cuMemcpyHtoDAsync_v2(dst: u64, src: *const c_void, bytes: usize, stream: u64) -> i32;
    fn cuMemcpyDtoHAsync_v2(dst: *mut c_void, src: u64, bytes: usize, stream: u64) -> i32;
    fn cuStreamSynchronize(stream: u64) -> i32;
}

/// 2026-09-25: Format a CUresult as `"<NAME> (<status>): <description>"`, with
/// `CUDA_UNKNOWN` and `(no message)` when the driver has no name or string for it.
pub fn cuda_error_text(status: i32) -> String {
    use std::ffi::CStr;
    let mut name_ptr: *const i8 = std::ptr::null();
    let mut msg_ptr: *const i8 = std::ptr::null();
    let name = unsafe {
        if cuGetErrorName(status, &mut name_ptr) == 0 && !name_ptr.is_null() {
            CStr::from_ptr(name_ptr as *const std::os::raw::c_char)
                .to_string_lossy()
                .into_owned()
        } else {
            "CUDA_UNKNOWN".to_string()
        }
    };
    let msg = unsafe {
        if cuGetErrorString(status, &mut msg_ptr) == 0 && !msg_ptr.is_null() {
            CStr::from_ptr(msg_ptr as *const std::os::raw::c_char)
                .to_string_lossy()
                .into_owned()
        } else {
            "(no message)".to_string()
        }
    };
    format!("{name} ({status}): {msg}")
}

/// 2026-09-25: `CUDA_ERROR_DEINITIALIZED`: the driver has already shut down. The
/// free paths treat it as nothing to do (see [`is_teardown_noop`]).
pub const CUDA_ERROR_DEINITIALIZED: i32 = 4;

/// 2026-09-25: Whether a CUresult means the context is already gone, so there is
/// nothing to free: `CUDA_ERROR_DEINITIALIZED`, `CUDA_ERROR_INVALID_CONTEXT` (201)
/// or `CUDA_ERROR_CONTEXT_IS_DESTROYED` (709).
pub fn is_teardown_noop(status: i32) -> bool {
    matches!(status, CUDA_ERROR_DEINITIALIZED | 201 | 709)
}

/// 2026-09-25: A raw `CUfunction` handle from [`MetraleRegistry::raw_function_cached`].
#[derive(Clone, Copy)]
pub struct RawCudaFunc(pub *mut c_void);
// 2026-09-25: SAFETY: the handle is an opaque pointer that is only copied and
// passed to the driver, never dereferenced in Rust. It is valid only while the
// module it came from is loaded.
unsafe impl Send for RawCudaFunc {}
unsafe impl Sync for RawCudaFunc {}

/// 2026-09-25: The kernel modules of one loaded model.
///
/// The blob set is the model's own (the caller passes the modules for this
/// checkpoint). Obtain one with [`MetraleRegistry::load`] and pass the
/// `Arc<MetraleRegistry>` along; there is no global accessor. Dropping the last
/// handle drops the registry's module handles.
pub struct MetraleRegistry {
    host: Arc<CudaHost>,
    modules: HashMap<&'static str, Arc<CudaModule>>,
    /// 2026-09-25: Raw `CUmodule` views of `modules`, for `cuModuleGetFunction` and
    /// `cuModuleGetGlobal_v2`.
    raw_modules: HashMap<&'static str, *mut c_void>,
}

impl Drop for MetraleRegistry {
    /// 2026-09-25: Drains both module maps (see `unload_raw`); cudarc unloads each
    /// module when its last `Arc<CudaModule>` goes.
    fn drop(&mut self) {
        let failures = self.unload_raw();
        if !failures.is_empty() {
            eprintln!(
                "metrale: {} module(s) failed to unload: {}",
                failures.len(),
                failures.join("; ")
            );
        }
    }
}

// 2026-09-25: SAFETY: `raw_modules` holds raw pointers. They are written only in
// `init` and in `unload_raw`, which takes `&mut self`; shared access only reads
// them and passes them to the driver.
unsafe impl Send for MetraleRegistry {}
unsafe impl Sync for MetraleRegistry {}

impl MetraleRegistry {
    /// 2026-09-25: Load this model's kernel modules into the process CUDA context.
    ///
    /// Each call produces a fresh, independent module set; it shares only the
    /// context and stream with a previously loaded model.
    pub fn load(
        ordinal: usize,
        kernel_blobs: &[(&'static str, &'static [u8])],
    ) -> Result<Arc<Self>> {
        Ok(Arc::new(Self::init(host(ordinal)?, kernel_blobs)?))
    }

    /// 2026-09-25: The process CUDA host this registry's modules live in.
    pub fn host(&self) -> &Arc<CudaHost> {
        &self.host
    }

    pub fn ctx(&self) -> &Arc<CudaContext> {
        &self.host.ctx
    }

    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.host.stream
    }

    /// 2026-09-25: Module names this registry loaded.
    pub fn module_names(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.modules.keys().copied()
    }

    fn init(
        host: Arc<CudaHost>,
        kernel_blobs: &[(&'static str, &'static [u8])],
    ) -> Result<MetraleRegistry> {
        let ctx = &host.ctx;

        let mut modules = HashMap::new();
        let mut raw_modules = HashMap::new();
        for &(name, blob) in kernel_blobs {
            let is_binary = blob.starts_with(b"\x7fELF")
                || blob.starts_with(b"__CLANG_OFFLOAD_BUNDLE__")
                || std::str::from_utf8(&blob[..blob.len().min(64)]).is_err();

            let ptx = if is_binary {
                Ptx::from_binary(blob.to_vec())
            } else {
                let src = std::str::from_utf8(blob).map_err(|e| {
                    MetraleError::ModuleLoad(format!("{name}: PTX not valid UTF-8: {e}"))
                })?;
                Ptx::from_src(src)
            };
            let module = ctx
                .load_module(ptx)
                .map_err(|e| MetraleError::ModuleLoad(format!("{name}: {e}")))?;

            // 2026-09-25: The raw handle is a view of this same module, not a second
            // load. It stays valid while `modules` holds the `Arc<CudaModule>` beside
            // it; `unload_raw` drains both maps together.
            raw_modules.insert(name, module.cu_module_raw() as *mut c_void);
            modules.insert(name, module);
        }

        Ok(MetraleRegistry {
            host,
            modules,
            raw_modules,
        })
    }

    /// 2026-09-25: Look up `func_name` in `module_name` through cudarc. Not memoized;
    /// see [`Self::function_cached`].
    pub fn function(&self, module_name: &str, func_name: &str) -> Result<CudaFunction> {
        let module = self.modules.get(module_name).ok_or_else(|| {
            MetraleError::ModuleLoad(format!("Module '{module_name}' not loaded"))
        })?;
        module
            .load_function(func_name)
            .map_err(|e| MetraleError::ModuleLoad(format!("{module_name}::{func_name}: {e}")))
    }

    /// 2026-09-25: [`Self::function`], memoized in the caller's `OnceLock`.
    pub fn function_cached(
        &self,
        cache: &OnceLock<CudaFunction>,
        module_name: &str,
        func_name: &str,
    ) -> Result<CudaFunction> {
        if let Some(f) = cache.get() {
            return Ok(f.clone());
        }
        let func = self.function(module_name, func_name)?;
        let _ = cache.set(func.clone());
        Ok(func)
    }

    /// 2026-09-25: Whether a module of this name was loaded for this run.
    pub fn has_module(&self, module_name: &str) -> bool {
        self.raw_modules.contains_key(module_name)
    }

    /// 2026-09-25: Look up `func_name` through the raw driver API
    /// (`cuModuleGetFunction`), memoized in the caller's `OnceLock`.
    pub fn raw_function_cached(
        &self,
        cache: &OnceLock<RawCudaFunc>,
        module_name: &str,
        func_name: &str,
    ) -> Result<RawCudaFunc> {
        if let Some(f) = cache.get() {
            return Ok(*f);
        }
        let raw_mod = self.raw_modules.get(module_name).ok_or_else(|| {
            MetraleError::ModuleLoad(format!("Module '{module_name}' not loaded"))
        })?;
        let c_name = CString::new(func_name).map_err(|e| {
            MetraleError::ModuleLoad(format!("{module_name}::{func_name}: CString: {e}"))
        })?;
        let mut func: *mut c_void = std::ptr::null_mut();
        let status =
            // 2026-09-25: `.cast()` because `c_char` is `i8` on x86_64 and `u8` on
            // aarch64; `as *const i8` would be an `unnecessary_cast` on x86_64.
            unsafe { cuModuleGetFunction(&mut func, *raw_mod, c_name.as_ptr().cast()) };
        if status != 0 {
            return Err(MetraleError::ModuleLoad(format!(
                "{module_name}::{func_name}: cuModuleGetFunction failed: {}",
                cuda_error_text(status)
            )));
        }
        let raw = RawCudaFunc(func);
        let _ = cache.set(raw);
        Ok(raw)
    }

    /// 2026-09-25: Drain both module maps, raw views first, so no raw handle
    /// outlives its module. Nothing is unloaded here: cudarc unloads each module
    /// when its last `Arc<CudaModule>` drops. Idempotent; always returns an empty
    /// list.
    pub(crate) fn unload_raw(&mut self) -> Vec<String> {
        self.raw_modules.drain().for_each(drop);
        self.modules.drain().for_each(drop);
        Vec::new()
    }

    /// 2026-09-25: The raw `CUstream` of the process stream held by [`CudaHost`].
    pub fn raw_stream(&self) -> u64 {
        self.host.stream.cu_stream() as u64
    }

    /// 2026-09-25: Resolve a `__device__` symbol of a loaded module to its device
    /// pointer and byte length, for drivers that read or write device globals
    /// without a kernel (the InnerQ calibration state). `symbol` is the
    /// linker-visible name, so C++ namespace symbols are Itanium-mangled
    /// (`_ZN7tq_plus14d_innerq_scaleE`).
    pub fn device_symbol(&self, module_name: &str, symbol: &str) -> Result<(u64, usize)> {
        let raw_mod = self.raw_modules.get(module_name).ok_or_else(|| {
            MetraleError::ModuleLoad(format!("Module '{module_name}' not loaded"))
        })?;
        let c_sym = CString::new(symbol).map_err(|e| {
            MetraleError::ModuleLoad(format!("{module_name}::{symbol}: CString: {e}"))
        })?;
        let mut dptr: u64 = 0;
        let mut bytes: usize = 0;
        let status =
            unsafe { cuModuleGetGlobal_v2(&mut dptr, &mut bytes, *raw_mod, c_sym.as_ptr().cast()) };
        if status != 0 {
            return Err(MetraleError::ModuleLoad(format!(
                "{module_name}::{symbol}: cuModuleGetGlobal_v2 failed: {}",
                cuda_error_text(status)
            )));
        }
        Ok((dptr, bytes))
    }

    /// 2026-09-25: Async H2D copy into a device pointer.
    ///
    /// # Safety
    /// `dst` must be a valid device pointer, and the bytes at `src` must outlive
    /// the copy (keep the host buffer until the next sync on `stream`).
    pub unsafe fn copy_h2d_async(
        &self,
        dst: u64,
        src: *const c_void,
        bytes: usize,
        stream: u64,
    ) -> Result<()> {
        let status = unsafe { cuMemcpyHtoDAsync_v2(dst, src, bytes, stream) };
        if status != 0 {
            return Err(MetraleError::KernelLaunch(format!(
                "cuMemcpyHtoDAsync_v2 failed: {}",
                cuda_error_text(status)
            )));
        }
        Ok(())
    }

    /// 2026-09-25: Async D2H copy from a device pointer.
    ///
    /// # Safety
    /// `dst` must stay alive until `stream` is synchronised.
    pub unsafe fn copy_d2h_async(
        &self,
        dst: *mut c_void,
        src: u64,
        bytes: usize,
        stream: u64,
    ) -> Result<()> {
        let status = unsafe { cuMemcpyDtoHAsync_v2(dst, src, bytes, stream) };
        if status != 0 {
            return Err(MetraleError::KernelLaunch(format!(
                "cuMemcpyDtoHAsync_v2 failed: {}",
                cuda_error_text(status)
            )));
        }
        Ok(())
    }

    /// 2026-09-25: Block the calling thread until all prior work on `stream` completes.
    pub fn stream_synchronize(&self, stream: u64) -> Result<()> {
        let status = unsafe { cuStreamSynchronize(stream) };
        if status != 0 {
            return Err(MetraleError::KernelLaunch(format!(
                "cuStreamSynchronize failed: {}",
                cuda_error_text(status)
            )));
        }
        Ok(())
    }

    /// 2026-09-25: Launch a kernel on the raw CUDA stream `stream_ptr`.
    ///
    /// # Safety
    /// - `kernel_params` must contain valid pointers to arguments matching the kernel signature.
    /// - `stream_ptr` must be a valid CUstream handle, or 0 for the legacy default stream.
    /// - `raw_func` must be a valid CUfunction obtained from `raw_function_cached`.
    pub unsafe fn launch_on_stream(
        &self,
        raw_func: RawCudaFunc,
        cfg: LaunchConfig,
        stream_ptr: u64,
        kernel_params: &mut [*mut c_void],
    ) -> Result<()> {
        // 2026-09-25: The caller's stream is used as given; the registry's own
        // stream is never substituted.
        let stream = stream_ptr;
        // 2026-09-25: Opt in to more than 48 KiB of dynamic shared memory when asked.
        if cfg.shared_mem_bytes > 48 * 1024 {
            const CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES: i32 = 8;
            let attr_status = unsafe {
                cuFuncSetAttribute(
                    raw_func.0,
                    CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    cfg.shared_mem_bytes as i32,
                )
            };
            if attr_status != 0 {
                return Err(MetraleError::KernelLaunch(format!(
                    "cuFuncSetAttribute(MAX_DYNAMIC_SHARED={}) failed: {}",
                    cfg.shared_mem_bytes,
                    cuda_error_text(attr_status)
                )));
            }
        }
        let status = unsafe {
            cuLaunchKernel(
                raw_func.0,
                cfg.grid_dim.0,
                cfg.grid_dim.1,
                cfg.grid_dim.2,
                cfg.block_dim.0,
                cfg.block_dim.1,
                cfg.block_dim.2,
                cfg.shared_mem_bytes,
                stream as *mut c_void,
                kernel_params.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        if status != 0 {
            return Err(MetraleError::KernelLaunch(format!(
                "cuLaunchKernel failed: {} (grid=[{},{},{}], block=[{},{},{}], shared_mem={})",
                cuda_error_text(status),
                cfg.grid_dim.0,
                cfg.grid_dim.1,
                cfg.grid_dim.2,
                cfg.block_dim.0,
                cfg.block_dim.1,
                cfg.block_dim.2,
                cfg.shared_mem_bytes
            )));
        }
        Ok(())
    }
}
