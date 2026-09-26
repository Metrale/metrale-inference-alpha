// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Prometheus text-format writers shared by the telemetry exporter.
//!
//! Owner: telemetry.
//! Invariants: none beyond the types.

use std::fmt::Write;

use crate::instrument::HistogramSnapshot;

pub(super) fn head(out: &mut String, name: &str, kind: &str, help: &str) {
    let _ = writeln!(out, "# HELP {name} {help}\n# TYPE {name} {kind}");
}

pub(super) fn one(out: &mut String, name: &str, kind: &str, help: &str, v: impl std::fmt::Display) {
    head(out, name, kind, help);
    let _ = writeln!(out, "{name} {v}");
}

pub(super) fn opt(out: &mut String, name: &str, help: &str, v: Option<f64>) {
    if let Some(v) = v {
        one(out, name, "gauge", help, v);
    }
}

/// 2026-09-26: One labelled histogram series; `scale` divides raw values into
/// the exported unit.
pub(super) fn histogram_series(
    out: &mut String,
    name: &str,
    labels: &str,
    h: &HistogramSnapshot,
    scale: f64,
) {
    let sep = if labels.is_empty() { "" } else { "," };
    for (bound, c) in h.upper_bounds.iter().zip(&h.cumulative) {
        let le = *bound as f64 / scale;
        let _ = writeln!(out, "{name}_bucket{{{labels}{sep}le=\"{le}\"}} {c}");
    }
    let _ = writeln!(out, "{name}_bucket{{{labels}{sep}le=\"+Inf\"}} {}", h.count);
    let braces = if labels.is_empty() {
        String::new()
    } else {
        format!("{{{labels}}}")
    };
    let _ = writeln!(out, "{name}_sum{braces} {}", h.sum as f64 / scale);
    let _ = writeln!(out, "{name}_count{braces} {}", h.count);
}

pub(super) fn histogram(
    out: &mut String,
    name: &str,
    help: &str,
    h: &HistogramSnapshot,
    scale: f64,
) {
    head(out, name, "histogram", help);
    histogram_series(out, name, "", h, scale);
}

/// 2026-09-26: Escape a label value: backslash, double quote and newline.
pub(super) fn escape(v: &str) -> String {
    v.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}
