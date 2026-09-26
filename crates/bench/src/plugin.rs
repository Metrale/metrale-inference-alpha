// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: [`Plugin`], the abstraction a [`crate::Benchmark`]
//! specialises, with its [`PluginHandle`] and the [`TargetEndpoint`] it
//! drives.
//!
//! [`Plugin::load`] receives the handle: status line, log, progress and glow
//! as [`PluginEvent`]s sent over a channel, plus the artifact store, the
//! endpoint and the cancellation flag.
//!
//! Owner: bench (plugin API).
//! Invariants:
//! - `TargetEndpoint::base_url` never ends in `/` when built by `new` or
//!   `local`.
//! - Sending an event never fails the plugin: a closed channel is ignored.

use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;

use anyhow::{Result, bail};

use crate::artifacts::ArtifactStore;
use crate::metadata::PluginMetadata;
use crate::result::{LogLevel, LogLine};

/// 2026-09-26: The served endpoint a benchmark drives. In the TUI it starts
/// as the local server, and the Benchmarks pane can point it at another URL
/// or model.
///
/// `Default` is the empty endpoint: `host_port` rejects it, so
/// `http::probe` fails on it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TargetEndpoint {
    /// 2026-09-26: Base URL with no trailing slash, e.g.
    /// `http://127.0.0.1:8888`.
    pub base_url: String,
    /// 2026-09-26: The `model` field sent in requests.
    pub model: String,
    /// 2026-09-26: The serve overrides the caller started this target under.
    /// The headless run record takes its `serve_overrides` from here, and the
    /// concurrency sweep reads `ssm_cache_slots` from here. Empty when the
    /// caller did not start the server, which means "not stated", never "no
    /// overrides".
    pub serve_overrides: std::collections::BTreeMap<String, String>,
}

impl TargetEndpoint {
    pub fn new(base_url: impl Into<String>, model: impl Into<String>) -> Self {
        let base = base_url.into();
        Self {
            base_url: base.trim_end_matches('/').to_string(),
            model: model.into(),
            serve_overrides: std::collections::BTreeMap::new(),
        }
    }

    /// 2026-09-26: The same endpoint, with the overrides it was started under.
    #[must_use]
    pub fn with_serve_overrides(
        mut self,
        serve_overrides: std::collections::BTreeMap<String, String>,
    ) -> Self {
        self.serve_overrides = serve_overrides;
        self
    }

    /// 2026-09-26: An override the server was started with, parsed as an
    /// integer. `None` when it was not stated or does not parse.
    pub fn serve_override_usize(&self, key: &str) -> Option<usize> {
        self.serve_overrides.get(key)?.trim().parse().ok()
    }

    /// 2026-09-26: `http://127.0.0.1:<port>`.
    pub fn local(port: u16, model: impl Into<String>) -> Self {
        Self::new(format!("http://127.0.0.1:{port}"), model)
    }

    /// 2026-09-26: Is this endpoint served on this box?
    ///
    /// The energy meter records the local GPU rail only for a loopback
    /// target; for any other host the local reading describes another
    /// machine. An unparseable target is not local.
    pub fn is_loopback(&self) -> bool {
        let Ok((host, _)) = self.host_port() else {
            return false;
        };
        host == "localhost"
            || host
                .trim_start_matches('[')
                .trim_end_matches(']')
                .parse::<std::net::IpAddr>()
                .is_ok_and(|ip| ip.is_loopback())
    }

    /// 2026-09-26: Split into `(host, port)` for a raw TCP connect. Only
    /// `http://` is accepted; the port defaults to 80 when absent.
    pub fn host_port(&self) -> Result<(String, u16)> {
        let rest = self.base_url.strip_prefix("http://").ok_or_else(|| {
            anyhow::anyhow!("only http:// targets are supported: {}", self.base_url)
        })?;
        let authority = rest.split('/').next().unwrap_or(rest);
        match authority.rsplit_once(':') {
            Some((host, port)) => {
                let port: u16 = port
                    .parse()
                    .map_err(|_| anyhow::anyhow!("bad port in {}", self.base_url))?;
                Ok((host.to_string(), port))
            }
            None => Ok((authority.to_string(), 80)),
        }
    }
}

/// 2026-09-26: A message from a running plugin to the terminal.
#[derive(Clone, Debug)]
pub enum PluginEvent {
    Log(LogLine),
    /// 2026-09-26: Replace the one-line status.
    Status(String),
    Progress {
        done: u64,
        total: u64,
    },
    /// 2026-09-26: Turn the terminal's work-in-flight glow on or off.
    Glow(bool),
}

/// 2026-09-26: The plugin's view of its host.
#[derive(Clone)]
pub struct PluginHandle {
    /// 2026-09-26: Distinguishes this run from the other runs of the same
    /// executor, whose counter assigns it.
    ///
    /// Cold TTFT and the serve-matrix probes fold it, with the process id,
    /// into their prompt prefix tags (`benchmarks::unique_prefix_tag`), so
    /// two runs do not share a cached prefix.
    run_id: u64,
    target: TargetEndpoint,
    artifacts: ArtifactStore,
    events: Sender<PluginEvent>,
    cancel: Arc<AtomicBool>,
}

impl PluginHandle {
    /// 2026-09-26: This run's id; see the field doc.
    pub fn run_id(&self) -> u64 {
        self.run_id
    }

    pub fn new(
        run_id: u64,
        target: TargetEndpoint,
        artifacts: ArtifactStore,
        events: Sender<PluginEvent>,
        cancel: Arc<AtomicBool>,
    ) -> Self {
        Self {
            run_id,
            target,
            artifacts,
            events,
            cancel,
        }
    }

    pub fn target(&self) -> &TargetEndpoint {
        &self.target
    }

    pub fn artifacts(&self) -> &ArtifactStore {
        &self.artifacts
    }

    /// 2026-09-26: Emit an event. A closed channel means the receiver is
    /// gone; that is not an error for the plugin.
    fn emit(&self, event: PluginEvent) {
        let _ = self.events.send(event);
    }

    pub fn log(&self, level: LogLevel, text: impl Into<String>) {
        self.emit(PluginEvent::Log(LogLine {
            level,
            text: text.into(),
        }));
    }
    pub fn info(&self, text: impl Into<String>) {
        self.log(LogLevel::Info, text);
    }
    pub fn warn(&self, text: impl Into<String>) {
        self.log(LogLevel::Warn, text);
    }
    pub fn status(&self, text: impl Into<String>) {
        self.emit(PluginEvent::Status(text.into()));
    }
    pub fn progress(&self, done: u64, total: u64) {
        self.emit(PluginEvent::Progress { done, total });
    }
    pub fn set_glow(&self, on: bool) {
        self.emit(PluginEvent::Glow(on));
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }

    /// 2026-09-26: `Err("cancelled")` once the run is cancelled. Call it at
    /// every await point a benchmark controls: otherwise cancellation takes
    /// effect only when the executor checks the flag, after each frame.
    pub fn check_cancelled(&self) -> Result<()> {
        if self.is_cancelled() {
            bail!("cancelled");
        }
        Ok(())
    }
}

/// 2026-09-26: The plugin abstraction. Every implementor today is a
/// benchmark.
pub trait Plugin {
    /// 2026-09-26: Who wrote this plugin, where it came from, and where to
    /// report it.
    fn metadata(&self) -> &'static PluginMetadata;

    /// 2026-09-26: First step. Acquire resources, provision artifacts, verify
    /// the host has what this plugin needs. An `Err` ends the run with a
    /// failed `setup` frame, so its message must name what is missing.
    fn load(&mut self, handle: PluginHandle) -> impl Future<Output = Result<()>> + Send;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_url_loses_its_trailing_slash() {
        let t = TargetEndpoint::new("http://box:8888/", "m");
        assert_eq!(t.base_url, "http://box:8888");
    }

    #[test]
    fn host_port_splits_ipv4_and_defaults_to_80() {
        assert_eq!(
            TargetEndpoint::local(8888, "m").host_port().unwrap(),
            ("127.0.0.1".to_string(), 8888)
        );
        assert_eq!(
            TargetEndpoint::new("http://dgx3", "m").host_port().unwrap(),
            ("dgx3".to_string(), 80)
        );
        assert!(TargetEndpoint::new("https://x", "m").host_port().is_err());
    }

    #[test]
    fn cancellation_is_visible_through_the_handle() {
        let (tx, rx) = std::sync::mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let h = PluginHandle::new(
            1,
            TargetEndpoint::local(8888, "m"),
            ArtifactStore::with_root("/tmp/metrale-test"),
            tx,
            cancel.clone(),
        );
        h.check_cancelled().unwrap();
        h.status("warming up");
        cancel.store(true, Ordering::Relaxed);
        assert!(h.check_cancelled().is_err());
        assert!(matches!(rx.try_recv(), Ok(PluginEvent::Status(s)) if s == "warming up"));
    }
}
