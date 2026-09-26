// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Process start of telemetry: the `--telemetry` level, the
//! loop-phase names, the NVML device sampler and, with the `otlp` feature,
//! the OTLP push thread. `serve` calls it once, before the first model load.
//!
//! Owner: server (telemetry).
//! Invariants:
//! - At `--telemetry off` no sampler or OTLP thread starts and nothing is
//!   logged.

use std::sync::OnceLock;

use anyhow::{Context, Result, anyhow};
use metrale_telemetry::sampler::SamplerHandle;
use metrale_telemetry::{Level, TelemetryConfig};

use crate::cli;

/// 2026-09-26: The device sampler, kept for the process lifetime.
static SAMPLER: OnceLock<SamplerHandle> = OnceLock::new();

/// 2026-09-26: Configure telemetry from `args`. `serve` calls it after
/// `validate_serve_args`, which checks the level string.
pub(crate) fn start(args: &cli::ServeArgs) -> Result<()> {
    let level: Level = args.telemetry.parse().map_err(|e| anyhow!("{e}"))?;
    // 2026-09-26: The CUDA ordinal is passed as the NVML device index; with
    // one visible GPU both are 0.
    let device = u32::try_from(args.gpu_ordinal).context("--gpu-ordinal exceeds u32")?;
    let cfg = TelemetryConfig::serving(level, device);
    let t = metrale_telemetry::global();
    t.set_phase_names(&crate::scheduler::mtp_timing::NAMES);
    match metrale_telemetry::sampler::start(t, &cfg) {
        Ok(Some(handle)) => {
            // 2026-09-26: On a second call the first sampler is kept, and
            // dropping this handle stops the new one.
            let _ = SAMPLER.set(handle);
            tracing::info!(
                "telemetry: level {level}, NVML device {device} sampled every {:?}; \
                 /metrics and /v1/events carry it",
                cfg.device_period()
            );
        }
        Ok(None) => {}
        Err(why) => tracing::warn!(
            "telemetry: level {level} WITHOUT the device layer (no power or energy): {why}"
        ),
    }
    #[cfg(feature = "otlp")]
    if level != Level::Off {
        super::telemetry_otlp::spawn()?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::*;

    fn serve_args(extra: &[&str]) -> cli::ServeArgs {
        let mut argv = vec!["met", "serve", "dummy/model"];
        argv.extend_from_slice(extra);
        match cli::Cli::parse_from(argv).command {
            cli::Command::Serve(a) => a,
            _ => unreachable!("a serve command"),
        }
    }

    #[test]
    fn the_flag_defaults_to_off_and_off_starts_nothing() {
        let args = serve_args(&[]);
        assert_eq!(args.telemetry, "off");
        start(&args).unwrap();
        let t = metrale_telemetry::global();
        assert_eq!(t.level(), Level::Off);
        assert_eq!(t.device_state(), metrale_telemetry::DeviceState::NotStarted);
        assert!(SAMPLER.get().is_none(), "no sampler thread at off");
    }

    #[test]
    fn an_unknown_level_is_refused_by_validation_naming_the_flag() {
        let err = cli::validate_serve_args(&serve_args(&["--telemetry", "verbose"]))
            .expect_err("verbose is not a level");
        assert!(err.contains("--telemetry"), "{err}");
        assert!(
            err.contains("kernel"),
            "the refusal lists the levels: {err}"
        );
    }
}
