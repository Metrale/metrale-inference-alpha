// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The dashboard-mode tracing subscriber, the tee log file and the stdout switch.
//!
//! In plain mode `main.rs` installs its own `fmt` subscriber and never calls
//! [`install_tty_subscriber`]. That function installs a registry with three
//! layers:
//!  1. an `fmt` layer (`ansi=false`) writing through [`SwitchableWriter`], to
//!     the tee file and, while [`TUI_ACTIVE`] is clear, to stdout;
//!  2. [`super::log_ring::LogRingLayer`], the lines for the log pane;
//!  3. [`super::capture_layer::ProgressCaptureLayer`], filtered to
//!     `metrale_telemetry::progress::TARGET` and independent of `RUST_LOG`.
//!
//! Layers 1 and 2 each get an `EnvFilter` built from the same `RUST_LOG`
//! spec (default `info`).
//!
//! Owner: server tui.
//! Invariants:
//! - `TEE` is set at most once per process (a `OnceLock`); if the file cannot
//!   be created, logging proceeds without it.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;

use parking_lot::Mutex;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::Layer as _;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

use super::capture_layer::{ProgressCaptureLayer, ProgressEvent};
use super::log_ring::LogRingLayer;

/// 2026-09-26: True while the dashboard owns the screen; [`SwitchableIo`] skips stdout while it is set.
///
/// `tui::start` sets it before spawning the render thread, and the render
/// thread's [`ActiveClaim`] clears it on the way out.
pub static TUI_ACTIVE: AtomicBool = AtomicBool::new(false);

/// 2026-09-26: Clears [`TUI_ACTIVE`] when dropped.
///
/// `events::run` takes one before anything that can return early, so every
/// return and an unwind out of the render thread send logs to stdout again.
pub struct ActiveClaim;

impl ActiveClaim {
    /// 2026-09-26: Set [`TUI_ACTIVE`]; setting it when already set is harmless.
    pub fn claim() -> Self {
        TUI_ACTIVE.store(true, Ordering::SeqCst);
        Self
    }
}

impl Drop for ActiveClaim {
    fn drop(&mut self) {
        TUI_ACTIVE.store(false, Ordering::SeqCst);
    }
}

/// 2026-09-26: The tee log file: its writer, its path, and its raw fd (`-1` off unix).
struct Tee {
    writer: Mutex<BufWriter<File>>,
    path: String,
    fd: i32,
}

// 2026-09-26: A process-wide static: the tracing writer, the panic hook and
// the terminal guard's stderr redirect all reach it without an owner to ask.
static TEE: OnceLock<Tee> = OnceLock::new();

/// 2026-09-26: The tee file's raw fd, if one is open on unix.
pub fn tee_raw_fd() -> Option<i32> {
    match TEE.get().map(|t| t.fd) {
        None | Some(-1) => None,
        Some(fd) => Some(fd),
    }
}

/// 2026-09-26: `$METRALE_TUI_LOG_FILE`, else `$HOME/.cache/metrale/logs/met-serve-<pid>-<unix secs>.log`
/// with `/tmp` standing in for an unset `HOME`.
fn tee_path() -> PathBuf {
    if let Ok(p) = std::env::var("METRALE_TUI_LOG_FILE") {
        return PathBuf::from(p);
    }
    let base = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    PathBuf::from(base)
        .join(".cache/metrale/logs")
        .join(format!("met-serve-{}-{ts}.log", std::process::id()))
}

/// 2026-09-26: The tee file's path, once it is open.
pub fn tee_file_path() -> Option<&'static str> {
    TEE.get().map(|t| t.path.as_str())
}

/// 2026-09-26: Flush the tee file, ignoring errors.
pub fn flush_tee() {
    if let Some(t) = TEE.get().map(|t| &t.writer) {
        let _ = t.lock().flush();
    }
}

/// 2026-09-26: `MakeWriter` whose writers tee to the log file and, while [`TUI_ACTIVE`] is clear, to stdout.
#[derive(Clone, Copy)]
pub struct SwitchableWriter;

pub struct SwitchableIo;

impl Write for SwitchableIo {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        if let Some(t) = TEE.get().map(|t| &t.writer) {
            let _ = t.lock().write_all(buf);
        }
        if !TUI_ACTIVE.load(Ordering::Relaxed) {
            let _ = std::io::stdout().write_all(buf);
        }
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        flush_tee();
        if !TUI_ACTIVE.load(Ordering::Relaxed) {
            std::io::stdout().flush()?;
        }
        Ok(())
    }
}

impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for SwitchableWriter {
    type Writer = SwitchableIo;
    fn make_writer(&'a self) -> Self::Writer {
        SwitchableIo
    }
}

fn env_filter() -> EnvFilter {
    EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into())
}

/// 2026-09-26: Open the tee file and install the dashboard-mode subscriber; progress events go to `progress_tx`.
pub fn install_tty_subscriber(progress_tx: Sender<ProgressEvent>) {
    let path = tee_path();
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    if let Ok(f) = File::create(&path) {
        // 2026-09-26: Off unix the tee still records the tracing stream; only the
        // stderr redirect in `terminal_guard` needs the fd.
        #[cfg(unix)]
        let fd = {
            use std::os::fd::AsRawFd;
            f.as_raw_fd()
        };
        #[cfg(not(unix))]
        let fd = -1;
        let _ = TEE.set(Tee {
            writer: Mutex::new(BufWriter::new(f)),
            path: path.display().to_string(),
            fd,
        });
    }
    let fmt_layer = tracing_subscriber::fmt::layer()
        .with_ansi(false)
        .with_writer(SwitchableWriter)
        .with_filter(env_filter());
    let ring_layer = LogRingLayer.with_filter(env_filter());
    let progress_layer =
        ProgressCaptureLayer::new(progress_tx).with_filter(tracing_subscriber::filter::filter_fn(
            |meta| meta.target() == metrale_telemetry::progress::TARGET,
        ));
    tracing_subscriber::registry()
        .with(fmt_layer)
        .with(ring_layer)
        .with(progress_layer)
        .init();
}

#[cfg(test)]
#[path = "init_tests.rs"]
mod tests;
