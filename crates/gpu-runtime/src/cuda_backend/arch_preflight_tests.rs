// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests for the device preflight: which arch string is judged,
//! the arch verdicts, the SM-count warning, and the order and thread of the
//! driver calls against a fake driver. All but one run without a GPU.
//!
//! Owner: gpu-runtime (CUDA backend).
//! Invariants: none beyond the types.

use super::{
    DeviceQuery, DriverDeviceQuery, check_arch, check_sm_count, device_compute_capability_of,
    preflight_arch, preflight_device_arch, preflight_device_arch_with,
};
use anyhow::{Result, bail};
use metrale_core::target::KernelTarget;
use metrale_kernels::{ModelBehavior, SamplingPresets, TargetPtxSet};
use std::sync::Mutex;
use std::thread::{self, ThreadId};

/// 2026-09-25: A `TargetPtxSet` for a hopper target: `KernelTarget.arch` is
/// the stripped base SM `sm_90`, `ptx_arch` is the given `[hardware].arch`.
fn a_hopper_target(ptx_arch: &'static str) -> TargetPtxSet {
    TargetPtxSet {
        target: KernelTarget {
            arch: "sm_90",
            model: "nemotron-super-120b-a12b",
            quant: "nvfp4",
        },
        ptx_arch,
        modules: Vec::new(),
        sampling: SamplingPresets::default(),
        behavior: ModelBehavior::default(),
        model_type_matches: Vec::new(),
        match_names: &[],
        dflash: None,
        shadowed_dropped: &[],
        expected_absent: &[],
    }
}

/// 2026-09-25: The preflight judges `ptx_arch`, not the stripped base SM.
/// `kernels/hopper/HARDWARE.toml` declares `arch = "sm_90a"`, which runs only
/// on CC 9.0 (`ptx_arch_runs_on_device`), while plain `sm_90` passes on
/// CC 10.0.
#[test]
fn the_preflight_judges_the_verbatim_arch_not_the_stripped_base_sm() {
    let hopper = a_hopper_target("sm_90a");
    assert_eq!(
        preflight_arch(&hopper),
        Some("sm_90a"),
        "the preflight must be handed the arch nvcc compiled for"
    );
    let err = check_arch(
        preflight_arch(&hopper).expect("hopper records an arch"),
        (10, 0),
    )
    .expect_err("sm_90a cannot load on CC 10.0");
    let msg = format!("{err}");
    assert!(msg.contains("sm_90a"), "{msg}");
    assert!(msg.contains("compute capability 10.0"), "{msg}");
    // 2026-09-25: The base SM passes on the same device, so judging it
    // would accept a device the PTX cannot run on.
    assert!(
        check_arch(hopper.target.arch, (10, 0)).is_ok(),
        "sm_90 is plain PTX and passes on CC 10.0 — that is the bug, not a \
         property to rely on"
    );
}

/// 2026-09-25: An empty `ptx_arch` selects nothing, and the preflight then
/// skips without touching CUDA.
#[test]
fn a_target_that_records_no_arch_selects_nothing_to_check() {
    let stub = a_hopper_target("");
    assert_eq!(preflight_arch(&stub), None);
    // 2026-09-25: The chain as `kernel_gate::gate_device_arch` runs it.
    preflight_device_arch(0, preflight_arch(&stub)).expect("a stub build has nothing to check");
}

/// 2026-09-25: `kernels/gb10/HARDWARE.toml` declares `arch = "sm_121f"` and
/// `compute_capability = "12.1"`; that pairing passes, and the logged line
/// names both.
#[test]
fn a_matching_device_logs_both_the_device_and_the_compiled_arch() {
    let line = check_arch("sm_121f", (12, 1)).expect("gb10 kernels run on a gb10");
    assert_eq!(line, "device CC 12.1, kernels built for sm_121f");
}

/// 2026-09-25: `kernels/hopper/HARDWARE.toml` declares `arch = "sm_90a"`
/// and `compute_capability = "9.0"`; that pairing passes.
#[test]
fn hopper_kernels_pass_on_a_hopper_device() {
    let line = check_arch("sm_90a", (9, 0)).expect("hopper kernels run on hopper");
    assert_eq!(line, "device CC 9.0, kernels built for sm_90a");
}

/// 2026-09-25: gb10 kernels (`sm_121f`) on a CC 9.0 device are refused with
/// a message naming the arch, the device's compute capability and the target
/// to rebuild for.
#[test]
fn the_gb10_image_on_a_hopper_device_fails_with_the_operator_message() {
    let err = check_arch("sm_121f", (9, 0)).expect_err("sm_121f cannot load on CC 9.0");
    let msg = format!("{err}");
    assert!(msg.contains("sm_121f"), "{msg}");
    assert!(msg.contains("compute capability 9.0"), "{msg}");
    assert!(msg.contains("METRALE_TARGET_HW=hopper"), "{msg}");
}

/// 2026-09-25: A fake driver in which only the first `init_host` makes a
/// context current, on its calling thread; later calls change nothing,
/// matching `cuda_host::host`'s `OnceLock`.
struct FakeDriver {
    /// 2026-09-25: The thread the first `init_host` ran on.
    ctx_current_on: Mutex<Option<ThreadId>>,
    /// 2026-09-25: Every call, as `(operation, thread, ordinal)`.
    calls: Mutex<Vec<(&'static str, ThreadId, usize)>>,
    /// 2026-09-25: `true` models a `cuCtxGetDevice` query, which fails on a
    /// thread with no current context; `false` models `cuDeviceGet`.
    reads_current_context: bool,
}

impl FakeDriver {
    fn new(reads_current_context: bool) -> Self {
        Self {
            ctx_current_on: Mutex::new(None),
            calls: Mutex::new(Vec::new()),
            reads_current_context,
        }
    }

    fn log(&self, op: &'static str, ordinal: usize) {
        self.calls
            .lock()
            .expect("fake driver lock")
            .push((op, thread::current().id(), ordinal));
    }
}

impl DeviceQuery for FakeDriver {
    fn init_host(&self, ordinal: usize) -> Result<()> {
        self.log("init_host", ordinal);
        let mut current = self.ctx_current_on.lock().expect("fake driver lock");
        if current.is_none() {
            *current = Some(thread::current().id());
        }
        Ok(())
    }

    fn compute_capability(&self, ordinal: usize) -> Result<(u32, u32)> {
        self.log("compute_capability", ordinal);
        if self.reads_current_context
            && *self.ctx_current_on.lock().expect("fake driver lock")
                != Some(thread::current().id())
        {
            // 2026-09-25: 201 is CUDA_ERROR_INVALID_CONTEXT.
            bail!("cuCtxGetDevice failed: status 201");
        }
        Ok((9, 0))
    }

    /// 2026-09-25: Always fails, so the tests below that expect `Ok` also
    /// show that a failed SM-count query does not fail the preflight.
    fn sm_count(&self, ordinal: usize) -> Result<u32> {
        self.log("sm_count", ordinal);
        bail!("this fake driver does not answer the SM count");
    }
}

/// 2026-09-25: Run `preflight_device_arch_with` on a new thread.
fn preflight_on_a_fresh_thread(driver: &FakeDriver, ordinal: usize) -> Result<()> {
    thread::scope(|scope| {
        scope
            .spawn(|| preflight_device_arch_with(ordinal, Some("sm_90a"), driver))
            .join()
            .expect("the preflight thread must not panic")
    })
}

/// 2026-09-25: Control: with a context-addressed query, the preflight fails
/// on a thread other than the one that initialised the host.
#[test]
fn a_context_addressed_query_fails_on_a_thread_that_did_not_make_the_host() {
    let driver = FakeDriver::new(true);
    driver.init_host(3).expect("thread A creates the host");
    let err = preflight_on_a_fresh_thread(&driver, 3)
        .expect_err("no context is current on the swap thread");
    assert!(
        format!("{err}").contains("201"),
        "expected CUDA_ERROR_INVALID_CONTEXT, got: {err}"
    );
}

/// 2026-09-25: With ordinal-addressed queries, the preflight passes on a
/// thread other than the one that initialised the host.
#[test]
fn the_ordinal_addressed_query_preflights_from_any_thread() {
    let driver = FakeDriver::new(false);
    driver.init_host(3).expect("thread A creates the host");
    preflight_on_a_fresh_thread(&driver, 3)
        .expect("an ordinal-addressed query needs no context of its own");

    let calls = driver.calls.lock().expect("fake driver lock");
    // 2026-09-25: Thread A's init, then thread B's sequence: `init_host`,
    // the capability query, then the SM-count query, all with the
    // requested ordinal.
    let [
        (_, thread_a, 3),
        ("init_host", thread_b, 3),
        ("compute_capability", queried_on, 3),
        ("sm_count", _, 3),
    ] = calls[..]
    else {
        panic!("unexpected driver call sequence: {calls:?}");
    };
    assert_eq!(thread_b, queried_on, "both ran on the swap thread");
    // 2026-09-25: The two threads differ, the case the control above fails.
    assert_ne!(
        thread_a, thread_b,
        "the defect only bites when these differ"
    );
}

/// 2026-09-25: The real driver, on a thread that did not create the host.
/// Needs a CUDA device, so it is `#[ignore]`d.
#[test]
#[ignore = "requires a free CUDA device"]
fn the_real_preflight_runs_on_a_thread_that_did_not_make_the_host() {
    DriverDeviceQuery
        .init_host(0)
        .expect("this thread creates the process CUDA host");
    thread::spawn(|| {
        let (major, minor) =
            device_compute_capability_of(0).expect("cuDeviceGet needs no current context");
        assert!(major > 0, "driver reported CC {major}.{minor}");
        // 2026-09-25: The device's own arch, so the test passes on any card.
        let arch = format!("sm_{major}{minor}");
        preflight_device_arch(0, Some(arch.as_str()))
    })
    .join()
    .expect("the preflight thread must not panic")
    .expect("a device's own compute capability must pass its preflight");
}

/// 2026-09-25: With no arch, the preflight returns `Ok` before touching CUDA,
/// so this passes without a GPU.
#[test]
fn a_build_that_recorded_no_arch_skips_the_check_without_a_gpu() {
    preflight_device_arch(0, None).expect("a stub build has nothing to check");
}

/// 2026-09-25: Equal SM counts produce no warning.
#[test]
fn a_matching_sm_count_says_nothing() {
    assert_eq!(check_sm_count(132, 132), None);
    assert_eq!(check_sm_count(48, 48), None);
}

/// 2026-09-25: A mismatch warns, naming both numbers and the `sm_count`
/// declaration, and says serving continues.
#[test]
fn a_mismatched_sm_count_warns_naming_both_numbers() {
    let warning = check_sm_count(132, 48).expect("48 declared, 132 present must warn");
    assert!(warning.contains("132"), "{warning}");
    assert!(warning.contains("48"), "{warning}");
    assert!(warning.contains("sm_count"), "{warning}");
    assert!(warning.contains("serving is unaffected"), "{warning}");
}

/// 2026-09-25: The warning fires whichever count is larger.
#[test]
fn the_cross_check_fires_in_both_directions() {
    assert!(check_sm_count(48, 132).is_some());
    assert!(check_sm_count(132, 48).is_some());
}
