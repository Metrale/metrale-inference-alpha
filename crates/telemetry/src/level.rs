// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The telemetry [`Level`] and the validated [`TelemetryConfig`].
//!
//! Serving takes the level from `--telemetry` and the NVML device from
//! `--gpu-ordinal`; the cadences are the constants below, through
//! [`TelemetryConfig::serving`].
//!
//! Owner: telemetry.
//! Invariants: a [`TelemetryConfig`]'s device period lies in
//! [`MIN_DEVICE_PERIOD`]..=[`MAX_DEVICE_PERIOD`] and its kernel span stride is
//! at least 1.

use std::time::Duration;

/// 2026-09-26: The measurement level. Each level measures everything the
/// levels below it measure.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum Level {
    /// 2026-09-26: Nothing is measured. Every hot-path entry point is one
    /// relaxed atomic load and a return: no clock read, no allocation.
    Off = 0,
    /// 2026-09-26: Device sampling (L0), scheduler, cache, spec and request
    /// instruments (L2–L5), and energy attribution.
    Basic = 1,
    /// 2026-09-26: Basic plus GPU event spans (L1): every serve lane each
    /// step, and every eager kernel launch on one step in
    /// [`TelemetryConfig::kernel_span_every`].
    Kernel = 2,
}

impl Level {
    /// 2026-09-26: The accepted spellings, in level order. The `--telemetry`
    /// validator checks against this list.
    pub const NAMES: [&'static str; 3] = ["off", "basic", "kernel"];

    pub const fn as_str(self) -> &'static str {
        Self::NAMES[self as usize]
    }

    /// 2026-09-26: Inverse of `self as u8`; any other byte reads as `Off`.
    pub const fn from_u8(v: u8) -> Self {
        match v {
            1 => Self::Basic,
            2 => Self::Kernel,
            _ => Self::Off,
        }
    }
}

impl std::fmt::Display for Level {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl std::str::FromStr for Level {
    type Err = ConfigError;

    /// 2026-09-26: Exact, lower-case spellings only: `Basic`, `on` and `1`
    /// are refused.
    fn from_str(s: &str) -> Result<Self, ConfigError> {
        match s {
            "off" => Ok(Self::Off),
            "basic" => Ok(Self::Basic),
            "kernel" => Ok(Self::Kernel),
            other => Err(ConfigError::UnknownLevel(other.to_string())),
        }
    }
}

/// 2026-09-26: Device sampling cadence when serving: 10 Hz.
pub const DEVICE_SAMPLE_PERIOD: Duration = Duration::from_millis(100);
/// 2026-09-26: When serving at level `Kernel`, per-kernel spans are recorded on
/// one step in this many.
pub const KERNEL_SPAN_EVERY: u32 = 64;
/// 2026-09-26: The shortest device period [`TelemetryConfig::new`] accepts.
pub const MIN_DEVICE_PERIOD: Duration = Duration::from_millis(10);
/// 2026-09-26: The longest device period [`TelemetryConfig::new`] accepts.
pub const MAX_DEVICE_PERIOD: Duration = Duration::from_secs(10);

/// 2026-09-26: Why a level or configuration was refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ConfigError {
    UnknownLevel(String),
    DevicePeriodOutOfRange(Duration),
    ZeroKernelSpanStride,
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownLevel(s) => write!(
                f,
                "unknown telemetry level {s:?}: expected one of {}",
                Level::NAMES.join(", ")
            ),
            Self::DevicePeriodOutOfRange(p) => write!(
                f,
                "device sample period {p:?} is outside [{MIN_DEVICE_PERIOD:?}, {MAX_DEVICE_PERIOD:?}]"
            ),
            Self::ZeroKernelSpanStride => {
                f.write_str("kernel span stride must be at least 1 (every step)")
            }
        }
    }
}

impl std::error::Error for ConfigError {}

/// 2026-09-26: A validated telemetry configuration. Its fields are private,
/// and [`TelemetryConfig::new`] (which `serving` calls) is the only constructor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TelemetryConfig {
    level: Level,
    device_period: Duration,
    kernel_span_every: u32,
    device_index: u32,
}

impl TelemetryConfig {
    pub fn new(
        level: Level,
        device_period: Duration,
        kernel_span_every: u32,
        device_index: u32,
    ) -> Result<Self, ConfigError> {
        if !(MIN_DEVICE_PERIOD..=MAX_DEVICE_PERIOD).contains(&device_period) {
            return Err(ConfigError::DevicePeriodOutOfRange(device_period));
        }
        if kernel_span_every == 0 {
            return Err(ConfigError::ZeroKernelSpanStride);
        }
        Ok(Self {
            level,
            device_period,
            kernel_span_every,
            device_index,
        })
    }

    /// 2026-09-26: The serving configuration: `level` at
    /// [`DEVICE_SAMPLE_PERIOD`] and [`KERNEL_SPAN_EVERY`], sampling NVML device
    /// `device_index`.
    pub fn serving(level: Level, device_index: u32) -> Self {
        Self::new(level, DEVICE_SAMPLE_PERIOD, KERNEL_SPAN_EVERY, device_index)
            .expect("the pinned cadences are within range")
    }

    pub fn level(&self) -> Level {
        self.level
    }
    pub fn device_period(&self) -> Duration {
        self.device_period
    }
    pub fn kernel_span_every(&self) -> u32 {
        self.kernel_span_every
    }
    pub fn device_index(&self) -> u32 {
        self.device_index
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_listed_name_parses_to_its_own_level() {
        for name in Level::NAMES {
            let level: Level = name.parse().unwrap();
            assert_eq!(level.as_str(), name);
            assert_eq!(Level::from_u8(level as u8), level);
        }
    }

    #[test]
    fn near_miss_spellings_are_refused_not_guessed() {
        for bad in ["Basic", "on", "1", "", "kernels", " off"] {
            assert_eq!(
                bad.parse::<Level>(),
                Err(ConfigError::UnknownLevel(bad.to_string())),
                "{bad:?}"
            );
        }
    }

    #[test]
    fn an_unknown_level_byte_reads_as_off() {
        assert_eq!(Level::from_u8(3), Level::Off);
        assert_eq!(Level::from_u8(255), Level::Off);
    }

    #[test]
    fn a_period_outside_the_range_is_refused_at_both_ends() {
        let fast = Duration::from_millis(9);
        let slow = Duration::from_millis(10_001);
        assert_eq!(
            TelemetryConfig::new(Level::Basic, fast, 1, 0),
            Err(ConfigError::DevicePeriodOutOfRange(fast))
        );
        assert_eq!(
            TelemetryConfig::new(Level::Basic, slow, 1, 0),
            Err(ConfigError::DevicePeriodOutOfRange(slow))
        );
        assert!(TelemetryConfig::new(Level::Basic, MIN_DEVICE_PERIOD, 1, 0).is_ok());
        assert!(TelemetryConfig::new(Level::Basic, MAX_DEVICE_PERIOD, 1, 0).is_ok());
    }

    #[test]
    fn a_zero_kernel_stride_is_refused() {
        assert_eq!(
            TelemetryConfig::new(Level::Kernel, DEVICE_SAMPLE_PERIOD, 0, 0),
            Err(ConfigError::ZeroKernelSpanStride)
        );
    }

    #[test]
    fn the_serving_config_is_ten_hertz() {
        let c = TelemetryConfig::serving(Level::Kernel, 0);
        assert_eq!(c.device_period(), Duration::from_millis(100));
        assert_eq!(c.kernel_span_every(), KERNEL_SPAN_EVERY);
        assert_eq!(c.level(), Level::Kernel);
    }
}
