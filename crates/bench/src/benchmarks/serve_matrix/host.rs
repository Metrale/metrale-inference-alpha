// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The seam the serve matrix boots checkpoints through: the serving
//! process installs a [`ServeHost`], because this crate starts no server.
//!
//! Owner: bench, serve matrix.
//! Invariants: none beyond the types.

use std::sync::Arc;

use anyhow::Result;
use futures::future::BoxFuture;
use parking_lot::RwLock;

use crate::plugin::TargetEndpoint;

/// 2026-09-26: Why a cached checkpoint cannot take part in the matrix. An absent
/// checkpoint is skipped and never booted; a planned one that fails to boot
/// fails its round.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Absence {
    /// 2026-09-26: A cache entry exists but its weights are not on disk.
    NoWeights,
    /// 2026-09-26: Downloaded, but its `config.json` could not be read or
    /// parsed, so the architecture is unknown. Kept apart from
    /// [`Absence::NoKernels`], which names an architecture.
    NoConfig,
    /// 2026-09-26: Downloaded and parsed, and this build compiled no kernels for it.
    NoKernels,
}

impl Absence {
    pub fn reason(self) -> &'static str {
        match self {
            Absence::NoWeights => "weights not fully downloaded",
            Absence::NoConfig => "config.json unreadable — architecture unknown",
            Absence::NoKernels => "no compiled kernels for this architecture",
        }
    }
}

/// 2026-09-26: One checkpoint the box knows about. Two quants of a model are two
/// candidates, and so two rounds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ServeCandidate {
    /// 2026-09-26: HF id, `org/name`.
    pub model: String,
    /// 2026-09-26: Quantization as the host reports it; the plan uses it in the
    /// round label and order.
    pub quant: String,
    /// 2026-09-26: `None` means the round is planned.
    pub absent: Option<Absence>,
}

impl ServeCandidate {
    pub fn ready(model: impl Into<String>, quant: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            quant: quant.into(),
            absent: None,
        }
    }

    pub fn absent(model: impl Into<String>, quant: impl Into<String>, why: Absence) -> Self {
        Self {
            model: model.into(),
            quant: quant.into(),
            absent: Some(why),
        }
    }
}

/// 2026-09-26: How one round is served.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ServeOptions {
    pub max_seq_len: usize,
    /// 2026-09-26: Serve with speculative decoding; the `speculative` parameter,
    /// off by default.
    pub speculative: bool,
}

/// 2026-09-26: Boots checkpoints and puts the previous one back; implemented by
/// the serving process.
pub trait ServeHost: Send + Sync {
    /// 2026-09-26: Every checkpoint the box holds, each marked servable or absent.
    fn roster(&self) -> Result<Vec<ServeCandidate>>;

    /// 2026-09-26: Bring `model` up and return the endpoint once it answers.
    fn serve(&self, model: &str, opts: ServeOptions) -> BoxFuture<'_, Result<TargetEndpoint>>;

    /// 2026-09-26: Put back whatever was serving before the matrix started.
    /// Called from the benchmark's `cleanup` once a plan exists.
    fn restore(&self) -> BoxFuture<'_, Result<()>>;
}

/// 2026-09-26: The installed host. A static because the registry builds a
/// benchmark from `BenchmarkDescriptor::ctor`, a `fn()` that takes no handle.
static HOST: RwLock<Option<Arc<dyn ServeHost>>> = RwLock::new(None);

/// 2026-09-26: The serving process calls this as it starts; a later call
/// replaces the earlier host.
pub fn install(host: Arc<dyn ServeHost>) {
    *HOST.write() = Some(host);
}

pub fn installed() -> Option<Arc<dyn ServeHost>> {
    HOST.read().clone()
}

/// 2026-09-26: What to tell the operator when nothing is installed; `load`
/// fails with it.
pub const NO_HOST: &str = "the serve matrix needs the Metrale Engine server that hosts this dashboard: it \
                           boots each checkpoint in-process. Run it from `met serve`'s \
                           dashboard rather than a standalone harness.";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_absence_reads_differently_because_each_needs_a_different_fix() {
        assert_eq!(
            [
                (Absence::NoWeights, Absence::NoWeights.reason()),
                (Absence::NoConfig, Absence::NoConfig.reason()),
                (Absence::NoKernels, Absence::NoKernels.reason()),
            ],
            [
                (Absence::NoWeights, "weights not fully downloaded"),
                (
                    Absence::NoConfig,
                    "config.json unreadable — architecture unknown",
                ),
                (
                    Absence::NoKernels,
                    "no compiled kernels for this architecture",
                ),
            ]
        );
    }

    #[test]
    fn a_ready_candidate_is_distinguishable_from_an_absent_one() {
        assert_eq!(
            ServeCandidate::ready("org/ready", "nvfp4"),
            ServeCandidate {
                model: "org/ready".into(),
                quant: "nvfp4".into(),
                absent: None,
            }
        );
        assert_eq!(
            ServeCandidate::absent("org/absent", "fp8", Absence::NoWeights),
            ServeCandidate {
                model: "org/absent".into(),
                quant: "fp8".into(),
                absent: Some(Absence::NoWeights),
            }
        );
    }
}
