// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The device sampler thread, the I/O half of L0 and energy. It
//! reads a [`DeviceSource`] at the configured period and hands each reading to
//! [`Telemetry::absorb_device`]. It runs on its own thread, so a slow driver
//! call delays the next sample, not a scheduler step.
//!
//! Owner: telemetry.
//! Invariants: stopping or dropping a [`SamplerHandle`] stops its thread and
//! joins it.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crate::device::{DeviceSource, NvmlSource};
use crate::energy::CounterTracker;
use crate::hub::Telemetry;
use crate::level::{Level, TelemetryConfig};
use metrale_gpu_sys::nvml::NvmlUnavailable;

/// 2026-09-26: A running sampler. Dropping it stops the thread and joins it.
pub struct SamplerHandle {
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl SamplerHandle {
    /// 2026-09-26: Stop and join.
    pub fn stop(mut self) {
        self.shutdown();
    }

    fn shutdown(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(t) = self.thread.take() {
            t.thread().unpark();
            // 2026-09-26: A panicked sampler has already stopped; its join error
            // is ignored.
            let _ = t.join();
        }
    }
}

impl Drop for SamplerHandle {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// 2026-09-26: Sample `source` into `t` every `period` until stopped.
pub fn spawn(
    t: &'static Telemetry,
    mut source: Box<dyn DeviceSource>,
    period: Duration,
) -> std::io::Result<SamplerHandle> {
    let stop = Arc::new(AtomicBool::new(false));
    let flag = stop.clone();
    let thread = std::thread::Builder::new()
        .name("metrale-telemetry-device".into())
        .spawn(move || {
            let mut tracker = CounterTracker::default();
            let mut next = Instant::now();
            while !flag.load(Ordering::Acquire) {
                let reading = source.read();
                t.absorb_device(&mut tracker, &reading, source.read_errors());
                next += period;
                let now = Instant::now();
                if next <= now {
                    // 2026-09-26: Fell behind: the next sample is one period
                    // from now.
                    next = now + period;
                }
                std::thread::park_timeout(next - now);
            }
        })?;
    Ok(SamplerHandle {
        stop,
        thread: Some(thread),
    })
}

/// 2026-09-26: Configure `t` and, for a level that samples, start the NVML
/// sampler.
///
/// `Ok(None)` when the level is `Off`: nothing is started or allocated.
/// `Err(reason)` when the device cannot be read (NVML missing, no such
/// device, no thread): `t` is marked unavailable with the reason, which is
/// also returned.
pub fn start(
    t: &'static Telemetry,
    cfg: &TelemetryConfig,
) -> Result<Option<SamplerHandle>, String> {
    start_with(t, cfg, |index| {
        NvmlSource::open(index).map(|s| Box::new(s) as Box<dyn DeviceSource>)
    })
}

/// 2026-09-26: [`start`] over an explicit device opener.
pub fn start_with(
    t: &'static Telemetry,
    cfg: &TelemetryConfig,
    open: impl FnOnce(u32) -> Result<Box<dyn DeviceSource>, NvmlUnavailable>,
) -> Result<Option<SamplerHandle>, String> {
    t.configure(cfg);
    if cfg.level() == Level::Off {
        return Ok(None);
    }
    let failed = |why: String| {
        t.mark_device_unavailable(why.clone());
        Err(why)
    };
    match open(cfg.device_index()) {
        Ok(src) => match spawn(t, src, cfg.device_period()) {
            Ok(h) => Ok(Some(h)),
            Err(e) => failed(format!("device sampler thread did not start: {e}")),
        },
        Err(e) => failed(e.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clock::MonotonicClock;
    use crate::device::DeviceReading;
    use crate::hub::DeviceState;

    struct Counting {
        counter_mj: u64,
    }

    impl DeviceSource for Counting {
        fn read(&mut self) -> DeviceReading {
            self.counter_mj += 1_000;
            DeviceReading {
                energy_counter_mj: Some(self.counter_mj),
                ..Default::default()
            }
        }
        fn read_errors(&self) -> u64 {
            0
        }
    }

    #[test]
    fn the_thread_samples_until_stopped_then_stops() {
        static CLOCK: MonotonicClock = MonotonicClock;
        let t: &'static Telemetry = Box::leak(Box::new(Telemetry::new(&CLOCK)));
        t.configure(&TelemetryConfig::new(Level::Basic, Duration::from_millis(10), 1, 0).unwrap());
        let h = spawn(
            t,
            Box::new(Counting { counter_mj: 0 }),
            Duration::from_millis(10),
        )
        .unwrap();
        let deadline = Instant::now() + Duration::from_secs(10);
        while t.device_reading().map_or(0, |(_, _, n)| n) < 3 {
            assert!(
                Instant::now() < deadline,
                "the sampler never produced 3 samples"
            );
            std::thread::sleep(Duration::from_millis(5));
        }
        h.stop();
        let after_stop = t.device_reading().unwrap().2;
        std::thread::sleep(Duration::from_millis(50));
        assert_eq!(
            t.device_reading().unwrap().2,
            after_stop,
            "no sample after stop"
        );
        assert_eq!(t.device_state(), DeviceState::Available);
        // 2026-09-26: Each sample after the first adds 1000 mJ.
        assert_eq!(t.energy_mj.get(), (after_stop - 1) * 1_000);
    }

    #[test]
    fn level_off_starts_nothing_and_touches_no_driver() {
        static CLOCK: MonotonicClock = MonotonicClock;
        let t: &'static Telemetry = Box::leak(Box::new(Telemetry::new(&CLOCK)));
        let h = start(t, &TelemetryConfig::serving(Level::Off, 0)).unwrap();
        assert!(h.is_none());
        assert_eq!(t.device_state(), DeviceState::NotStarted);
    }
}
