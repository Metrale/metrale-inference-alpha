// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The kernel-resolution audit: every lookup the CUDA backend's
//! `GpuBackend::kernel(module, func)` makes, whether it resolved, and the call
//! site that issued it; plus the boot gate's seal.
//!
//! A failed optional lookup (`try_kernel` returns handle 0) raises no error by
//! itself; the audit records it. The boot gate
//! (`serve_phases::kernel_gate::audit_and_gate`) prints
//! [`render_kernel_table`] and then calls [`seal`]. After the seal a failed
//! lookup aborts the process or, with
//! `--dangerously-allow-unresolved-kernel-lookups`, warns once per seal.
//!
//! Owner: telemetry.
//! Invariants: between [`seal`] and [`unseal`], every failed lookup passed to
//! [`record`] adds 1 to [`unresolved_lookups`] before anything else happens.

mod report;

pub use report::{render_kernel_table, unresolved_report};

use std::collections::BTreeMap;
use std::panic::Location;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// 2026-09-26: True from [`seal`] until [`unseal`].
static SEALED: AtomicBool = AtomicBool::new(false);
/// 2026-09-26: `--dangerously-allow-unresolved-kernel-lookups`, as handed to [`seal`].
static ALLOW_UNRESOLVED: AtomicBool = AtomicBool::new(false);
/// 2026-09-26: Unresolved lookups for the live model: the gate's count, plus
/// every failed lookup after the seal.
static UNRESOLVED: AtomicU64 = AtomicU64::new(0);
/// 2026-09-26: Latch so an allowed late miss warns once per seal.
static LATE_WARNED: AtomicBool = AtomicBool::new(false);

/// 2026-09-26: One deduped kernel-resolution row.
#[derive(Clone, Debug)]
pub struct AuditRow {
    pub module: String,
    pub func: String,
    /// 2026-09-26: True if any lookup of this `(module, func)` resolved.
    pub loaded: bool,
    /// 2026-09-26: Call site of the first lookup of this pair, captured through
    /// `#[track_caller]`.
    pub site: &'static Location<'static>,
}

impl AuditRow {
    /// 2026-09-26: `module::func`.
    pub fn name(&self) -> String {
        format!("{}::{}", self.module, self.func)
    }
}

/// 2026-09-26: Record one kernel lookup; the CUDA backend's
/// `GpuBackend::kernel` calls it.
///
/// `site` is passed in, not taken with `#[track_caller]` here, because this
/// function's caller is the backend, not the call site that asked.
pub fn record(module: &str, func: &str, loaded: bool, site: &'static Location<'static>) {
    if !loaded && SEALED.load(Ordering::Acquire) {
        late_miss(module, func, site);
    }
    if let Ok(mut v) = crate::run_metrics::metrics().kernel_audit.lock() {
        v.push((module.to_string(), func.to_string(), loaded, site));
    }
}

/// 2026-09-26: A kernel lookup that failed after the seal, which the boot gate
/// could not see. It is counted; then the process aborts, or with the allow
/// flag warns once per seal.
fn late_miss(module: &str, func: &str, site: &'static Location<'static>) {
    UNRESOLVED.fetch_add(1, Ordering::Relaxed);
    if ALLOW_UNRESOLVED.load(Ordering::Relaxed) {
        if !LATE_WARNED.swap(true, Ordering::Relaxed) {
            tracing::warn!(
                "kernel lookup {module}::{func} at {}:{} failed AFTER the boot audit sealed. \
                 This lookup is not eager, so the boot gate could not see it. Continuing \
                 because --dangerously-allow-unresolved-kernel-lookups was passed. \
                 Performance may be seriously degraded. We recommend you open a GitHub issue \
                 and/or open a PR to solve this issue. \
                 (metrale_kernel_lookups_unresolved counts every occurrence.)",
                site.file(),
                site.line(),
            );
        }
        return;
    }
    // 2026-09-26: Abort, not panic: a panic would unwind only the calling
    // thread and leave the process serving.
    tracing::error!(
        "kernel lookup {module}::{func} at {}:{} failed AFTER the boot audit sealed — a \
         non-eager lookup that the boot gate could not see. The dispatch that asked for it is \
         now on a silent fallback path. Aborting. Pass \
         --dangerously-allow-unresolved-kernel-lookups to downgrade this to a warning.",
        site.file(),
        site.line(),
    );
    std::process::abort();
}

/// 2026-09-26: Seal the audit for this run. `unresolved` is the gate's count
/// of failed lookups not declared expected-absent; `allow` is
/// `--dangerously-allow-unresolved-kernel-lookups`.
pub fn seal(unresolved: u64, allow: bool) {
    UNRESOLVED.store(unresolved, Ordering::Relaxed);
    ALLOW_UNRESOLVED.store(allow, Ordering::Relaxed);
    LATE_WARNED.store(false, Ordering::Relaxed);
    SEALED.store(true, Ordering::Release);
}

/// 2026-09-26: Re-open the audit for a new model load. Called from
/// [`crate::run_metrics::reset_for_new_run`].
pub fn unseal() {
    SEALED.store(false, Ordering::Release);
    UNRESOLVED.store(0, Ordering::Relaxed);
    LATE_WARNED.store(false, Ordering::Relaxed);
}

/// 2026-09-26: Unresolved kernel lookups for the live model. `/metrics`
/// exports it as `metrale_kernel_lookups_unresolved`.
pub fn unresolved_lookups() -> u64 {
    UNRESOLVED.load(Ordering::Relaxed)
}

/// 2026-09-26: The resolution rows: one per `(module, func)`, sorted, `loaded`
/// true if any lookup of that pair resolved.
pub fn audit_rows() -> Vec<AuditRow> {
    let mut resolved: BTreeMap<(String, String), (bool, &'static Location<'static>)> =
        BTreeMap::new();
    if let Ok(v) = crate::run_metrics::metrics().kernel_audit.lock() {
        for (m, f, ok, site) in v.iter() {
            let e = resolved
                .entry((m.clone(), f.clone()))
                .or_insert((false, *site));
            e.0 = e.0 || *ok;
        }
    }
    resolved
        .into_iter()
        .map(|((module, func), (loaded, site))| AuditRow {
            module,
            func,
            loaded,
            site,
        })
        .collect()
}

/// 2026-09-26: The failed lookups, split by whether the operator must act.
/// The boot gate, its log table and the TUI kernel table all split through
/// [`split_failures`].
#[derive(Clone, Debug, Default)]
pub struct FailureSplit {
    /// 2026-09-26: Not declared in `[expected_absent]`. The boot gate refuses
    /// to serve while any is present, unless the allow flag is set.
    pub required: Vec<AuditRow>,
    /// 2026-09-26: Declared in this target's MODEL.toml `[expected_absent]`.
    /// Never fatal.
    pub expected: Vec<AuditRow>,
}

/// 2026-09-26: Split [`audit_rows`]'s failures against a target's
/// `[expected_absent]` declaration (`TargetPtxSet::expected_absent`).
pub fn classify_failures(expected_absent: &[(&str, &str)]) -> FailureSplit {
    split_failures(&audit_rows(), expected_absent)
}

/// 2026-09-26: [`classify_failures`] over rows the caller already has.
pub fn split_failures(rows: &[AuditRow], expected_absent: &[(&str, &str)]) -> FailureSplit {
    let (expected, required): (Vec<AuditRow>, Vec<AuditRow>) =
        rows.iter().filter(|r| !r.loaded).cloned().partition(|r| {
            expected_absent
                .iter()
                .any(|(em, ef)| *em == r.module.as_str() && *ef == r.func.as_str())
        });
    FailureSplit { required, expected }
}
