// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: OTLP/HTTP metrics push (cargo feature `otlp`, not a default
//! feature).
//!
//! Owner: server (telemetry).
//! Invariants: none beyond the types.
//!
//! The target is `OTEL_EXPORTER_OTLP_METRICS_ENDPOINT` as given, else
//! `OTEL_EXPORTER_OTLP_ENDPOINT` + `/v1/metrics`; the period is
//! `OTEL_METRIC_EXPORT_INTERVAL` in ms. With neither endpoint set nothing is
//! exported, and a log line says so.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};

/// 2026-09-26: The period when `OTEL_METRIC_EXPORT_INTERVAL` is unset.
const SPEC_EXPORT_INTERVAL: Duration = Duration::from_millis(60_000);

fn endpoint() -> Option<String> {
    if let Ok(url) = std::env::var("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT") {
        return Some(url);
    }
    std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT")
        .ok()
        .map(|base| format!("{}/v1/metrics", base.trim_end_matches('/')))
}

fn interval() -> Result<Duration> {
    match std::env::var("OTEL_METRIC_EXPORT_INTERVAL") {
        Ok(ms) => Ok(Duration::from_millis(ms.parse().with_context(|| {
            format!("OTEL_METRIC_EXPORT_INTERVAL={ms:?} is not milliseconds")
        })?)),
        Err(_) => Ok(SPEC_EXPORT_INTERVAL),
    }
}

fn unix_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_nanos()).unwrap_or(u64::MAX))
}

/// 2026-09-26: Start the push thread when an endpoint is configured. An
/// unparseable `OTEL_METRIC_EXPORT_INTERVAL` is an error.
pub(super) fn spawn() -> Result<()> {
    let Some(url) = endpoint() else {
        tracing::info!(
            "telemetry: OTLP export compiled in but no OTEL_EXPORTER_OTLP_(METRICS_)ENDPOINT \
             is set; not exporting"
        );
        return Ok(());
    };
    let every = interval()?;
    let start = unix_ns();
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(10)))
        .build()
        .into();
    tracing::info!("telemetry: OTLP metrics to {url} every {every:?}");
    std::thread::Builder::new()
        .name("metrale-telemetry-otlp".into())
        .spawn(move || {
            loop {
                std::thread::sleep(every);
                let snap = metrale_telemetry::global().snapshot();
                let body = metrale_telemetry::export::otlp::metrics_body(&snap, start, unix_ns());
                if let Err(e) = agent
                    .post(&url)
                    .header("Content-Type", "application/json")
                    .send(body)
                {
                    tracing::warn!("telemetry: OTLP push to {url} failed: {e}");
                }
            }
        })
        .context("spawning the OTLP push thread")?;
    Ok(())
}
