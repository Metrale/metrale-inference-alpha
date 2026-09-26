// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The seam the scheduler-equivalence gate re-serves the checkpoint
//! through: the serving process installs a [`RouterHost`], because this crate
//! starts no server.
//!
//! Owner: bench, scheduler-equivalence gate.
//! Invariants: none beyond the types.

use std::collections::BTreeMap;
use std::sync::Arc;

use anyhow::Result;
use futures::future::BoxFuture;
use parking_lot::RwLock;

use crate::plugin::TargetEndpoint;

/// 2026-09-26: The device router a serve runs under: the `--scheduler-config` value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Router {
    Sync,
    Async,
}

impl Router {
    pub fn flag_value(self) -> &'static str {
        match self {
            Router::Sync => "sync",
            Router::Async => "async",
        }
    }
}

/// 2026-09-26: The speculation lane a serve is pinned to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Lane {
    /// 2026-09-26: No `--speculative`: plain decode only.
    SpecOff,
    /// 2026-09-26: `--speculative --mtp-gate force`: no runtime MTP gate is
    /// built, so the MTP step runs whenever speculation is eligible.
    MtpForce,
}

impl Lane {
    pub fn label(self) -> &'static str {
        match self {
            Lane::SpecOff => "spec-off",
            Lane::MtpForce => "mtp-force",
        }
    }
}

/// 2026-09-26: One serve the host brings up: a router under a lane.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ServeVariant {
    pub router: Router,
    pub lane: Lane,
}

impl ServeVariant {
    pub fn label(self) -> String {
        format!("{}/{}", self.router.flag_value(), self.lane.label())
    }
}

/// 2026-09-26: Re-serves the current checkpoint under a variant; implemented by
/// the serving process.
pub trait RouterHost: Send + Sync {
    /// 2026-09-26: Re-serve the current checkpoint under `variant` and return
    /// the endpoint once it answers.
    fn serve(&self, variant: ServeVariant) -> BoxFuture<'_, Result<TargetEndpoint>>;

    /// 2026-09-26: The asynchronous router's cumulative counters. The driver
    /// subtracts the reading before a lane's async legs from the one after.
    /// A host without counters returns an empty map.
    fn diagnostics(&self) -> Result<BTreeMap<String, f64>>;

    /// 2026-09-26: Put back whatever was serving before the gate started.
    /// Called from the benchmark's `cleanup`.
    fn restore(&self) -> BoxFuture<'_, Result<()>>;
}

/// 2026-09-26: The installed host. A static because the registry builds a
/// benchmark from `BenchmarkDescriptor::ctor`, a `fn()` that takes no handle;
/// `serve_matrix::host` holds its host the same way.
static HOST: RwLock<Option<Arc<dyn RouterHost>>> = RwLock::new(None);

/// 2026-09-26: The serving process calls this as it starts; a later call
/// replaces the earlier host.
pub fn install(host: Arc<dyn RouterHost>) {
    *HOST.write() = Some(host);
}

pub fn installed() -> Option<Arc<dyn RouterHost>> {
    HOST.read().clone()
}

/// 2026-09-26: What to tell the operator when nothing is installed.
pub const NO_HOST: &str = "the scheduler-equivalence gate needs the Metrale Engine server that \
    serves the checkpoint: it re-serves the model under each router itself. Run it from the \
    dashboard's Benchmarks pane, or headless with `met bench run scheduler-equivalence \
    --pull-request-gate`, which starts the server in this process. Pointing it at a foreign \
    endpoint with --url cannot work: the endpoint would stay on one router.";
