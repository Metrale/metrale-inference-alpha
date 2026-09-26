// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Refuse a GPU whose compute capability cannot run the kernels'
//! architecture, before `MetraleCudaBackend::new` loads any module, with an
//! error that names both ([`metrale_core::arch::ArchMismatch`]); also warn
//! when the device's SM count differs from the target's `sm_count`.
//!
//! The rule is [`metrale_core::arch::ptx_arch_runs_on_device`]; the
//! preflight's device queries are addressed by ordinal (`cuDeviceGet`), not by
//! the calling thread's current context (see [`device_compute_capability_of`]).
//!
//! Owner: gpu-runtime (CUDA backend).
//! Invariants:
//! - `preflight_device_arch_with` returns `Ok` without a device query when
//!   the target records no architecture.
//! - An SM-count mismatch or an unanswered SM-count query never fails the
//!   preflight.

use anyhow::{Result, bail};

use super::{cuCtxGetDevice, cuDeviceGet, cuDeviceGetAttribute};

const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: u32 = 75;
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: u32 = 76;
const CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: u32 = 16;

/// 2026-09-25: One `CUdevice_attribute` of `dev`, or an error with the
/// driver status.
fn device_attribute(attrib: u32, dev: i32) -> Result<i32> {
    let mut value: i32 = 0;
    let status = unsafe { cuDeviceGetAttribute(&mut value, attrib, dev) };
    if status != 0 {
        bail!("cuDeviceGetAttribute({attrib}) failed: status {status}");
    }
    Ok(value)
}

/// 2026-09-25: `(major, minor)` of an already-resolved `CUdevice`. Fails,
/// rather than returning a default, when a query fails or `major <= 0`.
fn compute_capability_of_device(dev: i32) -> Result<(u32, u32)> {
    let major = device_attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev)?;
    let minor = device_attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev)?;
    if major <= 0 {
        bail!("driver reported compute capability {major}.{minor} on device {dev}");
    }
    Ok((major as u32, minor as u32))
}

/// 2026-09-25: `(major, minor)` compute capability of the device of the
/// calling thread's current context; fails without one. `--check-kernels`
/// calls it after the backend is built (`kernel_gate.rs`
/// `current_device_cc`). The preflight uses [`device_compute_capability_of`]
/// instead.
pub fn device_compute_capability() -> Result<(u32, u32)> {
    let mut dev: i32 = 0;
    let status = unsafe { cuCtxGetDevice(&mut dev) };
    if status != 0 {
        bail!("cuCtxGetDevice failed: status {status}");
    }
    compute_capability_of_device(dev)
}

/// 2026-09-25: `(major, minor)` compute capability of GPU `ordinal`, with no
/// current context needed on the calling thread.
///
/// The preflight needs this form. `cuda_host::host` creates the context only
/// on its first call; later calls return the existing host and leave the
/// calling thread's current context alone. A later load can run on another
/// thread (the TUI's `metrale-swap` thread, `tui/lib_state.rs`), where a
/// context-addressed query has no context to read. The ignored GPU test
/// `the_real_preflight_runs_on_a_thread_that_did_not_make_the_host` checks
/// this form on such a thread.
pub fn device_compute_capability_of(ordinal: usize) -> Result<(u32, u32)> {
    let ordinal_i32 = i32::try_from(ordinal)
        .map_err(|_| anyhow::anyhow!("GPU ordinal {ordinal} does not fit a CUdevice ordinal"))?;
    let mut dev: i32 = 0;
    let status = unsafe { cuDeviceGet(&mut dev, ordinal_i32) };
    if status != 0 {
        bail!("cuDeviceGet(ordinal {ordinal}) failed: status {status}");
    }
    compute_capability_of_device(dev)
}

/// 2026-09-25: Streaming multiprocessors on GPU `ordinal`, addressed by
/// ordinal like [`device_compute_capability_of`]. Fails when the query fails
/// or the count is <= 0.
pub fn device_sm_count_of(ordinal: usize) -> Result<u32> {
    let ordinal_i32 = i32::try_from(ordinal)
        .map_err(|_| anyhow::anyhow!("GPU ordinal {ordinal} does not fit a CUdevice ordinal"))?;
    let mut dev: i32 = 0;
    let status = unsafe { cuDeviceGet(&mut dev, ordinal_i32) };
    if status != 0 {
        bail!("cuDeviceGet(ordinal {ordinal}) failed: status {status}");
    }
    let count = device_attribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev)?;
    if count <= 0 {
        bail!("driver reported {count} multiprocessors on device {dev}");
    }
    Ok(count as u32)
}

/// 2026-09-25: Compare the device's SM count with the target's
/// `kernels/<hw>/HARDWARE.toml` `[hardware] sm_count`: `None` when they are
/// equal, otherwise a warning naming both numbers and the file. The caller
/// logs it and carries on (`preflight_device_arch_with`).
pub fn check_sm_count(device_sms: u32, declared_sms: u32) -> Option<String> {
    (device_sms != declared_sms).then(|| {
        format!(
            "this build's kernels are sized for {declared_sms} SMs \
             (kernels/{hw}/HARDWARE.toml [hardware] sm_count) but the device \
             reports {device_sms} — grid sizing that reads it will be off; \
             serving is unaffected",
            hw = metrale_kernels::TARGET_DEFAULTS.hw,
        )
    })
}

/// 2026-09-25: The verdict, without touching a GPU: `Ok(line to log)`, or the
/// [`metrale_core::arch::ArchMismatch`] as the error. Testable with no CUDA.
pub fn check_arch(compiled_arch: &str, device_cc: (u32, u32)) -> Result<String> {
    if let Err(mismatch) = metrale_core::arch::ptx_arch_runs_on_device(compiled_arch, device_cc) {
        // 2026-09-25: Returned typed, so `--check-kernels` can downcast it
        // (`kernel_gate.rs`) instead of parsing the message.
        return Err(mismatch.into());
    }
    Ok(format!(
        "device CC {}.{}, kernels built for {compiled_arch}",
        device_cc.0, device_cc.1
    ))
}

/// 2026-09-25: The architecture string the preflight judges: the target's
/// `ptx_arch` (`[hardware].arch` verbatim, e.g. `sm_90a`), not
/// `target.arch`, which has the `a`/`f` suffix stripped. The suffix decides
/// which devices the PTX runs on: plain `sm_90` passes on CC 10.0 and 12.1,
/// `sm_90a` does not (`ptx_arch_runs_on_device`).
///
/// `None` when `ptx_arch` is empty; the preflight then warns and skips.
pub fn preflight_arch(ptx_set: &metrale_kernels::TargetPtxSet) -> Option<&'static str> {
    Some(ptx_set.ptx_arch).filter(|a| !a.is_empty())
}

/// 2026-09-25: Fail if kernels built for `compiled_arch` cannot run on GPU
/// `ordinal`. Call it before `MetraleCudaBackend::new`, which loads the
/// modules (`init_gpu_backend` does, through `kernel_gate::gate_device_arch`).
///
/// `compiled_arch` is `None` when the target records no architecture
/// ([`preflight_arch`]); that is logged as a warning and returns `Ok` without
/// a device query.
pub fn preflight_device_arch(ordinal: usize, compiled_arch: Option<&str>) -> Result<()> {
    preflight_device_arch_with(ordinal, compiled_arch, &DriverDeviceQuery)
}

/// 2026-09-25: The driver calls the preflight makes, behind a trait so a
/// fake driver can test their order and the thread each runs on without a
/// GPU (`arch_preflight_tests.rs`).
pub(crate) trait DeviceQuery {
    /// 2026-09-25: Initialise the process CUDA host on `ordinal`. In
    /// [`DriverDeviceQuery`] this is `cuda_host::host`, which creates the
    /// context on its first call only.
    fn init_host(&self, ordinal: usize) -> Result<()>;

    /// 2026-09-25: `(major, minor)` compute capability of GPU `ordinal`.
    /// [`DriverDeviceQuery`] answers by ordinal, without a current context.
    fn compute_capability(&self, ordinal: usize) -> Result<(u32, u32)>;

    /// 2026-09-25: Streaming multiprocessors on GPU `ordinal`, for
    /// [`check_sm_count`].
    fn sm_count(&self, ordinal: usize) -> Result<u32>;
}

/// 2026-09-25: The production `DeviceQuery`: `cuda_host::host`, then
/// ordinal-addressed attribute queries (`cuDeviceGet`).
pub(crate) struct DriverDeviceQuery;

impl DeviceQuery for DriverDeviceQuery {
    fn init_host(&self, ordinal: usize) -> Result<()> {
        crate::cuda_host::host(ordinal).map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok(())
    }

    fn compute_capability(&self, ordinal: usize) -> Result<(u32, u32)> {
        device_compute_capability_of(ordinal)
    }

    fn sm_count(&self, ordinal: usize) -> Result<u32> {
        device_sm_count_of(ordinal)
    }
}

/// 2026-09-25: [`preflight_device_arch`] against an injected driver.
pub(crate) fn preflight_device_arch_with(
    ordinal: usize,
    compiled_arch: Option<&str>,
    query: &dyn DeviceQuery,
) -> Result<()> {
    let Some(compiled_arch) = compiled_arch else {
        tracing::warn!(
            "this build recorded no kernel architecture, so the GPU compute-capability \
             preflight is skipped — expected under METRALE_SKIP_BUILD=1, a defect otherwise"
        );
        return Ok(());
    };
    // 2026-09-25: The host is initialised first; `MetraleRegistry::load`
    // later uses the same process host (`cuda_host::host`). The queries below
    // are addressed by ordinal, so they do not need a context current on
    // this thread.
    query.init_host(ordinal)?;
    let device_cc = query.compute_capability(ordinal)?;
    tracing::info!("{}", check_arch(compiled_arch, device_cc)?);
    // 2026-09-25: The SM-count check only warns; a failed query is logged at
    // debug level and the preflight still returns `Ok`.
    match query.sm_count(ordinal) {
        Ok(device_sms) => {
            if let Some(warning) = check_sm_count(device_sms, metrale_kernels::TARGET_SM_COUNT) {
                tracing::warn!("{warning}");
            }
        }
        Err(e) => tracing::debug!("SM-count cross-check skipped: {e}"),
    }
    Ok(())
}

#[cfg(test)]
#[path = "arch_preflight_tests.rs"]
mod tests;
