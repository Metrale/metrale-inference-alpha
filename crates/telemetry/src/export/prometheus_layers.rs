// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The L1–L5 sections of the telemetry `/metrics` text: scheduler
//! and cache, speculation and requests, and (level `Kernel`) GPU spans.
//!
//! Owner: telemetry.
//! Invariants: none beyond the types.

use std::fmt::Write;

use crate::hub::Telemetry;
use crate::snapshot::TelemetrySnapshot;

use super::prometheus_text::{escape, head, histogram, histogram_series, one};

pub(super) fn render_sched(out: &mut String, t: &Telemetry, snap: &TelemetrySnapshot) {
    let s = &snap.sched;
    let sh = &s.shape;
    for (name, help, v) in [
        (
            "metrale_sched_pending",
            "Requests drained but not yet admitted",
            sh.pending,
        ),
        ("metrale_sched_active", "Sequences decoding", sh.active),
        (
            "metrale_sched_prefilling",
            "Sequences prefilling",
            sh.prefilling,
        ),
        (
            "metrale_sched_swapped",
            "Sequences parked (spilled or requeued)",
            sh.swapped,
        ),
        (
            "metrale_kv_blocks_free",
            "KV-cache blocks free",
            sh.kv_blocks_free,
        ),
        (
            "metrale_kv_blocks_total",
            "KV-cache blocks total",
            sh.kv_blocks_total,
        ),
        (
            "metrale_ssm_slots_used",
            "SSM snapshot slots in use",
            sh.ssm_slots_used,
        ),
        (
            "metrale_ssm_slots_total",
            "SSM snapshot slots total",
            sh.ssm_slots_total,
        ),
    ] {
        one(out, name, "gauge", help, v);
    }
    one(
        out,
        "metrale_sched_ticks_total",
        "counter",
        "Scheduler ticks this run",
        sh.ticks,
    );
    one(
        out,
        "metrale_gpu_stream_syncs_total",
        "counter",
        "Blocking stream synchronisations",
        s.stream_syncs,
    );
    one(
        out,
        "metrale_gpu_blocking_d2h_total",
        "counter",
        "Blocking device-to-host copies",
        s.blocking_d2h,
    );
    one(
        out,
        "metrale_cuda_graph_replays_total",
        "counter",
        "CUDA graph replays (cache hits)",
        s.graph_replays,
    );
    one(
        out,
        "metrale_cuda_graph_captures_total",
        "counter",
        "CUDA graph captures (cache misses)",
        s.graph_captures,
    );
    histogram(
        out,
        "metrale_sched_batch_decode_rows",
        "Decode rows per tick with decode work",
        &t.sched.decode_rows.snapshot(),
        1.0,
    );
    histogram(
        out,
        "metrale_sched_batch_prefill_seqs",
        "Prefilling sequences per tick with prefill work",
        &t.sched.prefill_seqs.snapshot(),
        1.0,
    );
    let names = t.phase_names();
    if !names.is_empty() {
        head(
            out,
            "metrale_sched_phase_seconds",
            "histogram",
            "Scheduler loop phase wall time",
        );
        for (i, name) in names.iter().enumerate() {
            let h = t.sched.phase[i].snapshot();
            if h.count > 0 {
                histogram_series(
                    out,
                    "metrale_sched_phase_seconds",
                    &format!("phase=\"{name}\""),
                    &h,
                    1e9,
                );
            }
        }
    }
}
pub(super) fn render_spec_and_requests(out: &mut String, t: &Telemetry, snap: &TelemetrySnapshot) {
    if !snap.spec.is_empty() {
        head(
            out,
            "metrale_spec_verify_steps_total",
            "counter",
            "Verify steps by drafts checked and drafts accepted",
        );
        for row in &snap.spec {
            for a in 0..=row.drafts {
                let _ = writeln!(
                    out,
                    "metrale_spec_verify_steps_total{{drafts=\"{}\",accepted=\"{a}\"}} {}",
                    row.drafts,
                    t.spec.steps(row.drafts, a)
                );
            }
        }
        head(
            out,
            "metrale_spec_acceptance_by_depth",
            "gauge",
            "Fraction of verify steps accepting at least `depth` drafts",
        );
        for row in &snap.spec {
            for (i, p) in row.acceptance_by_depth.iter().enumerate() {
                let _ = writeln!(
                    out,
                    "metrale_spec_acceptance_by_depth{{drafts=\"{}\",depth=\"{}\"}} {p}",
                    row.drafts,
                    i + 1
                );
            }
        }
    }
    let r = &t.requests;
    one(
        out,
        "metrale_request_output_tokens_total",
        "counter",
        "Output tokens of finished requests, as their terminal frames report",
        r.output_tokens.get(),
    );
    one(
        out,
        "metrale_requests_finished_total",
        "counter",
        "Requests that reached a terminal frame",
        snap.requests.finished,
    );
    histogram(
        out,
        "metrale_request_tpot_seconds",
        "Mean time per output token after the first, per request",
        &r.tpot.snapshot(),
        1e9,
    );
    histogram(
        out,
        "metrale_request_e2e_seconds",
        "Arrival to terminal frame, per request",
        &r.e2e.snapshot(),
        1e9,
    );
}
pub(super) fn render_kernel(
    out: &mut String,
    t: &Telemetry,
    kernel_name: &dyn Fn(u64) -> Option<String>,
) {
    let k = &t.kernel;
    head(
        out,
        "metrale_gpu_lane_seconds",
        "histogram",
        "GPU time per serve lane per step (CUDA events)",
    );
    for (i, lane) in t.lane_names().iter().enumerate() {
        histogram_series(
            out,
            "metrale_gpu_lane_seconds",
            &format!("lane=\"{lane}\""),
            &k.lane[i].snapshot(),
            1e9,
        );
    }
    let kernels = k.kernels();
    if !kernels.is_empty() {
        head(
            out,
            "metrale_gpu_kernel_seconds_total",
            "counter",
            "GPU time of sampled eager kernel launches",
        );
        for kt in &kernels {
            let name = kernel_name(kt.func).unwrap_or_else(|| format!("fn@{:#x}", kt.func));
            let _ = writeln!(
                out,
                "metrale_gpu_kernel_seconds_total{{kernel=\"{}\"}} {}",
                escape(&name),
                kt.total_ns as f64 / 1e9
            );
        }
        head(
            out,
            "metrale_gpu_kernel_samples_total",
            "counter",
            "Sampled eager kernel launches",
        );
        for kt in &kernels {
            let name = kernel_name(kt.func).unwrap_or_else(|| format!("fn@{:#x}", kt.func));
            let _ = writeln!(
                out,
                "metrale_gpu_kernel_samples_total{{kernel=\"{}\"}} {}",
                escape(&name),
                kt.samples
            );
        }
    }
    one(
        out,
        "metrale_gpu_spans_dropped_total",
        "counter",
        "GPU spans not started because the event ring was full",
        k.spans_dropped.get(),
    );
    one(
        out,
        "metrale_gpu_span_failures_total",
        "counter",
        "GPU spans whose events could not be recorded or read",
        k.span_failures.get(),
    );
    one(
        out,
        "metrale_gpu_kernel_table_full_total",
        "counter",
        "Kernel spans lost to a full kernel table",
        k.table_full.get(),
    );
}
