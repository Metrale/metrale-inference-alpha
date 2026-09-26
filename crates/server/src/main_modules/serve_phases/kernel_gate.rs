// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Startup phase 7: the kernel-resolution audit printed after
//! `build_model`, the boot gate on unresolved lookups, `--check-kernels`, and
//! the device-arch refusal `init_gpu_backend` runs before the backend exists.
//!
//! Owner: server startup (`met serve`).
//! Invariants:
//! - `audit_and_gate` seals the audit before it decides, on every path.
//! - It returns `Ok` only when no required lookup is unresolved or
//!   `--dangerously-allow-unresolved-kernel-lookups` is set; under
//!   `--check-kernels` it exits the process instead of returning.

use anyhow::Result;

use crate::cli;

/// 2026-09-26: Exit statuses are 8 bits, so an unclamped count of 256 would
/// read as 0. `check_and_exit` announces the clamp whenever it applies.
const MAX_EXIT_CODE: usize = 255;

/// 2026-09-26: Refuse a device this build's PTX cannot run on, before the
/// backend is constructed (`init_gpu_backend` calls this first). The arch
/// string judged is `preflight_arch`'s choice: the verbatim `ptx_arch`, not
/// `target.arch`, which has its `a`/`f` suffix stripped.
#[cfg(feature = "cuda")]
pub(crate) fn gate_device_arch(
    checking: bool,
    ptx_set: &metrale_kernels::TargetPtxSet,
    gpu_ordinal: usize,
) -> Result<()> {
    use metrale_gpu_runtime::cuda_backend::arch_preflight;
    gate_arch_preflight(
        checking,
        ptx_set,
        arch_preflight::preflight_device_arch(gpu_ordinal, arch_preflight::preflight_arch(ptx_set)),
    )
}

/// 2026-09-26: Returns `result` unchanged. Under `--check-kernels` an
/// `ArchMismatch` also prints the JSON result line, with null audit counts
/// because no kernel lookup has run yet.
#[cfg(feature = "cuda")]
pub(crate) fn gate_arch_preflight(
    checking: bool,
    ptx_set: &metrale_kernels::TargetPtxSet,
    result: Result<()>,
) -> Result<()> {
    if let Err(error) = &result
        && let Some(line) = arch_refusal_json(
            checking,
            ptx_set.target.model,
            ptx_set.target.quant,
            ptx_set.modules.len(),
            error,
        )
    {
        use std::io::Write as _;
        println!("{line}");
        let _ = std::io::stdout().flush();
    }
    result
}

#[cfg(any(feature = "cuda", test))]
fn arch_refusal_json(
    checking: bool,
    model: &str,
    quant: &str,
    modules_embedded: usize,
    error: &anyhow::Error,
) -> Option<String> {
    let mismatch = error.downcast_ref::<metrale_core::arch::ArchMismatch>()?;
    checking.then(|| {
        serde_json::json!({
            "metrale_kernel_check": {
                "model": model,
                "arch": mismatch.compiled_arch,
                "compiled_arch": mismatch.compiled_arch,
                "device_cc": mismatch.device_cc,
                "quant": quant,
                "kernel_set_hash": metrale_kernels::KERNEL_SET_HASH,
                "modules_embedded": modules_embedded,
                "lookups": null,
                "unresolved": null,
                "expected_absent": null,
                "unresolved_kernels": null,
                "ok": false,
                "exit_code": 1,
                "failing_stage": "arch_preflight",
                "error": error.to_string(),
            }
        })
        .to_string()
    })
}

/// 2026-09-26: Print the audit, seal it, and gate on it.
///
/// Under `--check-kernels` this does not return: it exits the process with
/// the unresolved count (clamped to [`MAX_EXIT_CODE`]) as the status.
pub(crate) fn audit_and_gate(
    args: &cli::ServeArgs,
    ptx_set: &metrale_kernels::TargetPtxSet,
) -> Result<()> {
    tracing::info!(
        "{}",
        metrale_telemetry::kernel_audit::render_kernel_table(
            &ptx_set.modules,
            metrale_kernels::KERNEL_SET_HASH,
            ptx_set.shadowed_dropped,
            ptx_set.expected_absent,
        )
    );

    let rows = metrale_telemetry::kernel_audit::audit_rows();
    let split = metrale_telemetry::kernel_audit::split_failures(&rows, ptx_set.expected_absent);
    let allowed = args.dangerously_allow_unresolved_kernel_lookups;
    let target = &ptx_set.target;

    // 2026-09-26: Sealed before the decision, so a lookup after this point
    // aborts or warns (`kernel_audit::record`) even when the operator chose to
    // serve anyway.
    metrale_telemetry::kernel_audit::seal(split.required.len() as u64, allowed);

    if args.check_kernels {
        check_and_exit(&rows, &split, ptx_set);
    }

    if split.required.is_empty() {
        return Ok(());
    }
    let report = metrale_telemetry::kernel_audit::unresolved_report(
        &split,
        ptx_set.shadowed_dropped,
        target.model,
        target.arch,
        target.quant,
        allowed,
    );
    if allowed {
        tracing::warn!("{report}");
        return Ok(());
    }
    Err(anyhow::anyhow!("{report}"))
}

/// 2026-09-26: `--check-kernels`: print the report, print the JSON line, exit
/// with the unresolved count (clamped to [`MAX_EXIT_CODE`]). Never returns.
fn check_and_exit(
    rows: &[metrale_telemetry::kernel_audit::AuditRow],
    split: &metrale_telemetry::kernel_audit::FailureSplit,
    ptx_set: &metrale_kernels::TargetPtxSet,
) -> ! {
    use std::io::Write as _;

    let target = &ptx_set.target;
    let n = split.required.len();
    if n == 0 {
        tracing::info!(
            "kernel check PASSED for ({}, {}, {}): {} lookups, {} expected-absent",
            target.model,
            target.arch,
            target.quant,
            rows.len(),
            split.expected.len(),
        );
    } else {
        // 2026-09-26: `--check-kernels` ignores
        // `--dangerously-allow-unresolved-kernel-lookups`: the report is the
        // not-allowed form and the exit status is the count.
        tracing::error!(
            "{}",
            metrale_telemetry::kernel_audit::unresolved_report(
                split,
                ptx_set.shadowed_dropped,
                target.model,
                target.arch,
                target.quant,
                false,
            )
        );
    }
    let code = exit_code_for(n);
    if code != n {
        // 2026-09-26: On stdout and in the log: the exit status under-reports.
        let msg = format!("{n} unresolved kernels (exit code clamped to {MAX_EXIT_CODE})");
        tracing::error!("{msg}");
        println!("{msg}");
    }
    // 2026-09-26: The machine-readable result on one line, after the human
    // report, so a sweep over many targets does not parse prose.
    println!("{}", check_json(rows, split, ptx_set, code));
    // 2026-09-26: `exit` runs no destructors, so flush stdout first.
    let _ = std::io::stdout().flush();
    std::process::exit(code as i32);
}

/// 2026-09-26: The process status for `n` unresolved kernels: `n`, clamped to
/// [`MAX_EXIT_CODE`].
fn exit_code_for(n: usize) -> usize {
    n.min(MAX_EXIT_CODE)
}

/// 2026-09-26: Everything the one-line `--check-kernels` result reports. Tests
/// build it without a GPU or a `TargetPtxSet`.
struct CheckSummary<'a> {
    model: &'a str,
    /// 2026-09-26: The arch the kernels were compiled for: `ptx_arch`, the
    /// verbatim `kernels/<hw>/HARDWARE.toml` `[hardware].arch`.
    compiled_arch: &'a str,
    quant: &'a str,
    modules_embedded: usize,
    lookups: usize,
    expected_absent: usize,
    exit_code: usize,
    unresolved: Vec<serde_json::Value>,
    /// 2026-09-26: `(major, minor)` of the GPU, or `None` when it cannot be
    /// queried (`current_device_cc`).
    device_cc: Option<(u32, u32)>,
}

/// 2026-09-26: One compact JSON object summarising the check. `ok` is true
/// exactly when `exit_code` is 0. `arch` repeats `compiled_arch` under the
/// older field name.
fn check_json_from(summary: &CheckSummary) -> String {
    serde_json::json!({
        "metrale_kernel_check": {
            "model": summary.model,
            "arch": summary.compiled_arch,
            "compiled_arch": summary.compiled_arch,
            "device_cc": summary.device_cc,
            "quant": summary.quant,
            "kernel_set_hash": metrale_kernels::KERNEL_SET_HASH,
            "modules_embedded": summary.modules_embedded,
            "lookups": summary.lookups,
            "unresolved": summary.unresolved.len(),
            "expected_absent": summary.expected_absent,
            "ok": summary.unresolved.is_empty(),
            // 2026-09-26: The status this process exits with; it differs from
            // `unresolved` only when clamped.
            "exit_code": summary.exit_code,
            "unresolved_kernels": summary.unresolved,
        }
    })
    .to_string()
}

/// 2026-09-26: The GPU's compute capability, or `None` in a build without the
/// `cuda` feature or when the driver refuses the query. On a served target
/// `--check-kernels` runs after the backend is up, so this is the real device.
fn current_device_cc() -> Option<(u32, u32)> {
    #[cfg(feature = "cuda")]
    {
        metrale_gpu_runtime::cuda_backend::arch_preflight::device_compute_capability().ok()
    }
    #[cfg(not(feature = "cuda"))]
    {
        None
    }
}

fn check_json(
    rows: &[metrale_telemetry::kernel_audit::AuditRow],
    split: &metrale_telemetry::kernel_audit::FailureSplit,
    ptx_set: &metrale_kernels::TargetPtxSet,
    exit_code: usize,
) -> String {
    let unresolved: Vec<serde_json::Value> = split
        .required
        .iter()
        .map(|r| {
            serde_json::json!({
                "kernel": r.name(),
                "site": format!("{}:{}", r.site.file(), r.site.line()),
            })
        })
        .collect();
    check_json_from(&CheckSummary {
        model: ptx_set.target.model,
        // 2026-09-26: `ptx_arch`, not `target.arch`: the field reports the
        // verbatim `[hardware].arch`, suffix included.
        compiled_arch: ptx_set.ptx_arch,
        quant: ptx_set.target.quant,
        modules_embedded: ptx_set.modules.len(),
        lookups: rows.len(),
        expected_absent: split.expected.len(),
        exit_code,
        unresolved,
        device_cc: current_device_cc(),
    })
}

#[cfg(test)]
mod tests {
    use super::{CheckSummary, MAX_EXIT_CODE, arch_refusal_json, check_json_from, exit_code_for};

    fn a_clean_gb10_check() -> CheckSummary<'static> {
        CheckSummary {
            model: "qwen3.6-27b",
            compiled_arch: "sm_121f",
            quant: "nvfp4",
            modules_embedded: 42,
            lookups: 300,
            expected_absent: 2,
            exit_code: 0,
            unresolved: Vec::new(),
            device_cc: Some((12, 1)),
        }
    }

    /// 2026-09-26: The one JSON line names the compiled arch and the device.
    /// Values from `kernels/gb10/HARDWARE.toml`: `arch = "sm_121f"`,
    /// `compute_capability = "12.1"`.
    #[test]
    fn the_check_line_reports_the_compiled_arch_and_the_device() {
        let v: serde_json::Value =
            serde_json::from_str(&check_json_from(&a_clean_gb10_check())).expect("valid JSON");
        let c = &v["metrale_kernel_check"];
        assert_eq!(c["compiled_arch"], "sm_121f");
        assert_eq!(c["device_cc"], serde_json::json!([12, 1]));
        assert_eq!(c["arch"], "sm_121f");
    }

    /// 2026-09-26: With no device the line reports `device_cc: null` and still
    /// carries the rest.
    #[test]
    fn a_host_with_no_device_reports_a_null_device_cc() {
        let summary = CheckSummary {
            device_cc: None,
            ..a_clean_gb10_check()
        };
        let v: serde_json::Value =
            serde_json::from_str(&check_json_from(&summary)).expect("valid JSON");
        assert!(v["metrale_kernel_check"]["device_cc"].is_null());
        assert_eq!(v["metrale_kernel_check"]["compiled_arch"], "sm_121f");
    }

    /// 2026-09-26: Every other field of the result line, with one unresolved
    /// kernel.
    #[test]
    fn the_pre_existing_fields_are_unchanged() {
        let summary = CheckSummary {
            exit_code: 3,
            unresolved: vec![serde_json::json!({"kernel": "m::k", "site": "a.rs:1"})],
            ..a_clean_gb10_check()
        };
        let v: serde_json::Value =
            serde_json::from_str(&check_json_from(&summary)).expect("valid JSON");
        let c = &v["metrale_kernel_check"];
        assert_eq!(c["model"], "qwen3.6-27b");
        assert_eq!(c["quant"], "nvfp4");
        assert_eq!(c["modules_embedded"], 42);
        assert_eq!(c["lookups"], 300);
        assert_eq!(c["unresolved"], 1);
        assert_eq!(c["expected_absent"], 2);
        assert_eq!(c["ok"], false);
        assert_eq!(c["exit_code"], 3);
        assert_eq!(c["unresolved_kernels"][0]["kernel"], "m::k");
        assert_eq!(c["device_cc"], serde_json::json!([12, 1]));
        assert_eq!(c["compiled_arch"], "sm_121f");
    }

    /// 2026-09-26: A real `check_arch` refusal of sm_90a / sm_100a PTX on a
    /// 12.1 device becomes the JSON line with `failing_stage = arch_preflight`;
    /// without `--check-kernels` nothing is printed.
    #[test]
    #[cfg(feature = "cuda")]
    fn early_arch_refusals_report_the_same_numeric_device_contract() {
        for arch in ["sm_90a", "sm_100a"] {
            let error =
                metrale_gpu_runtime::cuda_backend::arch_preflight::check_arch(arch, (12, 1))
                    .expect_err("architecture-specific PTX must refuse GB10");
            let line = arch_refusal_json(true, "nano", "nvfp4", 42, &error)
                .expect("--check-kernels must retain preflight mismatch JSON");
            let value: serde_json::Value = serde_json::from_str(&line).unwrap();
            let check = &value["metrale_kernel_check"];
            assert_eq!(check["compiled_arch"], arch);
            assert_eq!(check["arch"], arch);
            assert_eq!(check["device_cc"], serde_json::json!([12, 1]));
            assert_eq!(check["ok"], false);
            assert_eq!(check["exit_code"], 1);
            assert_eq!(check["failing_stage"], "arch_preflight");
            assert!(check["lookups"].is_null());
            assert!(check["unresolved"].is_null());
            assert!(check["error"].as_str().unwrap().contains("12.1"));
            assert!(check["error"].as_str().unwrap().contains(arch));
            assert!(arch_refusal_json(false, "nano", "nvfp4", 42, &error).is_none());
        }
    }

    #[test]
    fn an_unrelated_error_does_not_invent_device_facts() {
        let error = anyhow::anyhow!("CUDA initialization failed before the device query");
        assert!(arch_refusal_json(true, "nano", "nvfp4", 42, &error).is_none());
    }

    #[test]
    fn the_exit_code_is_the_unresolved_count() {
        // 2026-09-26: The exit status equals the unresolved count below 256.
        for n in [0usize, 1, 2, 15, 42, 254, 255] {
            assert_eq!(exit_code_for(n), n, "exit code must equal the count");
        }
    }

    #[test]
    fn a_count_of_256_does_not_report_as_a_clean_pass() {
        // 2026-09-26: A count at or above 256 must stay non-zero after the
        // 8-bit truncation.
        assert_eq!(exit_code_for(256), MAX_EXIT_CODE);
        assert_eq!(exit_code_for(1000), MAX_EXIT_CODE);
        for n in [256usize, 512, 4096] {
            assert_ne!(exit_code_for(n) % 256, 0, "{n} must not read as success");
        }
    }
}
