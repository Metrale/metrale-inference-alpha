// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Rendering for `met benchmark`: the suite list, a parameter
//! schema, run history, one record, and run progress.
//!
//! Owner: server CLI (`met benchmark`).
//! Invariants:
//! - The list, schema, history and record go to stdout; `StdoutReporter`'s
//!   progress goes to stderr. The plain log subscriber writes to stdout too,
//!   except under `benchmark certify --json` (`main.rs`), so stdout carries
//!   payload and log lines, never progress.

use anyhow::Result;
use metrale_bench::headless::{RunReporter, RunRequest};
use metrale_bench::params::ParamSpec;
use metrale_bench::{BenchmarkResult, PluginEvent, RunRecord, VerdictKind, registry};

use super::bench_args::OutputFormat;

pub fn print_suite(format: OutputFormat) -> Result<()> {
    let all = registry::all();
    if format == OutputFormat::Json {
        let rows: Vec<_> = all
            .iter()
            .map(|d| {
                serde_json::json!({
                    "id": d.id, "name": d.name, "summary": d.summary,
                    "duration_hint": d.duration_hint,
                    "needs_confirmation": d.needs_confirmation,
                    "intended_for": d.intended_for.map(|e| e.families),
                    // 2026-09-26: True for a group id: certified by a complete
                    // partition of shard runs (`--param shard=i/n`) at one commit.
                    "group": metrale_bench::gate::group::find(d.id).is_some(),
                })
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&rows)?);
        return Ok(());
    }
    let width = all.iter().map(|d| d.id.len() + 2).max().unwrap_or(0);
    for d in all {
        let mark = if d.needs_confirmation { " (--yes)" } else { "" };
        let group = if metrale_bench::gate::group::find(d.id).is_some() {
            " [group: certified by shards, --param shard=i/n]"
        } else {
            ""
        };
        println!(
            "{:width$}  {:<10}  {}{}{}",
            d.id, d.duration_hint, d.summary, mark, group
        );
    }
    Ok(())
}

/// 2026-09-26: One benchmark's parameter schema, the keys `--param` accepts.
pub fn print_schema(id: &str, format: OutputFormat) -> Result<()> {
    let descriptor = super::bench_run::find(id)?;
    let specs = descriptor.build().parameters();
    if format == OutputFormat::Json {
        let rows: Vec<_> = specs
            .iter()
            .map(|s| {
                serde_json::json!({
                    "key": s.key, "label": s.label, "help": s.help,
                    "default": s.default.to_edit_string(),
                    "domain": s.kind.domain_hint(),
                })
            })
            .collect();
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "id": descriptor.id, "name": descriptor.name,
                "detail": descriptor.detail, "parameters": rows,
            }))?
        );
        return Ok(());
    }
    println!("{}  —  {}", descriptor.id, descriptor.name);
    println!("{}\n", descriptor.detail);
    // 2026-09-26: The checkpoint families the numbers are defined on, printed
    // before the parameters.
    if let Some(expect) = descriptor.intended_for {
        println!("  defined on   {}", expect.families.join(" | "));
        println!("  {}\n", expect.note);
    }
    print_variants(descriptor.id);
    print_specs(&specs);
    Ok(())
}

/// 2026-09-26: The benchmark's model variants from its baseline, the values
/// `--checkpoint` accepts. Prints nothing when there is no repo root or the
/// baseline cannot be read.
fn print_variants(benchmark_id: &str) {
    let Ok(root) = super::bench_run::repo_root() else {
        return;
    };
    let Ok(baseline) = metrale_bench::gate::read_baseline(&root, benchmark_id) else {
        return;
    };
    for (hardware, hw) in &baseline.hardware {
        for (checkpoint, entry) in &hw.models {
            let default = if *checkpoint == hw.default {
                "  (default — runs when no --checkpoint is passed)"
            } else {
                "  (select with --checkpoint under --pull-request-gate)"
            };
            let label = if entry.label.is_empty() {
                String::new()
            } else {
                format!("  — {}", entry.label)
            };
            println!("  variant [{hardware}]  {checkpoint}{label}{default}");
        }
    }
    println!();
}

fn print_specs(specs: &[ParamSpec]) {
    let width = specs.iter().map(|s| s.key.len()).max().unwrap_or(0);
    for s in specs {
        println!(
            "  --param {:width$} = {:<14} {}  [{}]",
            s.key,
            s.default.to_edit_string(),
            s.help,
            s.kind.domain_hint()
        );
    }
}

pub fn print_history(records: &[RunRecord], format: OutputFormat) -> Result<()> {
    if format == OutputFormat::Json {
        println!("{}", serde_json::to_string_pretty(records)?);
        return Ok(());
    }
    if records.is_empty() {
        println!("no runs recorded yet");
        return Ok(());
    }
    for r in records {
        let verdict = match r.verdict_kind() {
            Some(VerdictKind::Pass) => "PASS",
            Some(VerdictKind::Fail) => "FAIL",
            Some(VerdictKind::Info) => "info",
            None => "—",
        };
        let source = if r.is_legacy() {
            "legacy".to_string()
        } else {
            format!("{:?}", r.source).to_lowercase()
        };
        println!(
            "{}  {:<20} {:<6} {:<6} {}",
            r.run_id,
            r.benchmark_id,
            verdict,
            source,
            r.age_text()
        );
    }
    Ok(())
}

pub fn print_record(record: &RunRecord, format: OutputFormat) -> Result<()> {
    if format == OutputFormat::Json {
        println!("{}", serde_json::to_string_pretty(record)?);
        return Ok(());
    }
    println!("{}  {}", record.run_id, record.benchmark_name);
    println!("  when     {} ({})", record.recorded_at, record.age_text());
    println!("  target   {} · {}", record.target_url, record.target_model);
    println!(
        "  source   {:?} · metrale {}",
        record.source, record.metrale_version
    );
    if !record.params.is_empty() {
        println!("  params");
        for (k, v) in &record.params {
            println!("    {k} = {v}");
        }
    }
    println!();
    print_frame(&record.frame);
    Ok(())
}

/// 2026-09-26: The measurement itself: stats, table, verdict.
pub fn print_frame(frame: &BenchmarkResult) {
    for stat in &frame.summary {
        let unit = &stat.unit;
        println!("  {:<24} {}{}", stat.label, stat.value, unit);
    }
    if let Some(table) = &frame.table {
        println!();
        let headers: Vec<&str> = table.columns.iter().map(|c| c.title.as_str()).collect();
        let mut widths: Vec<usize> = headers.iter().map(|h| h.len()).collect();
        for row in &table.rows {
            for (i, cell) in row.iter().enumerate() {
                if i < widths.len() {
                    widths[i] = widths[i].max(cell.text.len());
                }
            }
        }
        let line = |cells: Vec<String>| {
            let padded: Vec<String> = cells
                .iter()
                .enumerate()
                .map(|(i, c)| format!("{:width$}", c, width = widths.get(i).copied().unwrap_or(0)))
                .collect();
            println!("  {}", padded.join("  "));
        };
        line(headers.iter().map(|h| h.to_string()).collect());
        for row in &table.rows {
            line(row.iter().map(|c| c.text.clone()).collect());
        }
    }
    if let Some(v) = &frame.verdict {
        println!("\n  {:?}: {}", v.kind, v.reason);
    }
}

/// 2026-09-26: Progress to stderr. A phase, status or frame log line prints
/// only when it differs from the previous one; `quiet` keeps the start line
/// and warnings and errors, and drops the rest.
pub struct StdoutReporter {
    pub quiet: bool,
    last_phase: String,
    last_status: String,
    /// 2026-09-26: The last frame log line printed, so a repeated frame does
    /// not repeat it.
    last_frame_log: String,
}

impl StdoutReporter {
    pub fn new(quiet: bool) -> Self {
        Self {
            quiet,
            last_phase: String::new(),
            last_status: String::new(),
            last_frame_log: String::new(),
        }
    }
}

impl RunReporter for StdoutReporter {
    fn started(&mut self, request: &RunRequest) {
        eprintln!(
            "{} → {} · {}",
            request.descriptor.name, request.target.base_url, request.target.model
        );
    }

    fn event(&mut self, event: &PluginEvent) {
        match event {
            // 2026-09-26: Warnings and errors print even when quiet.
            PluginEvent::Log(line)
                if !self.quiet
                    || matches!(
                        line.level,
                        metrale_bench::LogLevel::Warn | metrale_bench::LogLevel::Error
                    ) =>
            {
                eprintln!("  {:?}: {}", line.level, line.text);
            }
            PluginEvent::Status(s) if !self.quiet && *s != self.last_status => {
                self.last_status = s.clone();
                eprintln!("  {s}");
            }
            _ => {}
        }
    }

    fn frame(&mut self, frame: &BenchmarkResult) {
        // 2026-09-26: Frame log lines print before the phase check, so a
        // warning does not depend on whether its frame also changed phase.
        for line in &frame.log {
            if !self.quiet
                || matches!(
                    line.level,
                    metrale_bench::LogLevel::Warn | metrale_bench::LogLevel::Error
                )
            {
                let text = format!("  {:?}: {}", line.level, line.text);
                if text != self.last_frame_log {
                    self.last_frame_log = text.clone();
                    eprintln!("{text}");
                }
            }
        }
        if self.quiet || frame.phase == self.last_phase {
            return;
        }
        self.last_phase = frame.phase.clone();
        let progress = match frame.progress {
            Some((done, total)) => format!(" [{done}/{total}]"),
            None => String::new(),
        };
        eprintln!(
            "  [{:>6.1}s] {}{progress}",
            frame.elapsed.as_secs_f64(),
            frame.phase
        );
    }
}
