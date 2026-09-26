// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The Serve Matrix's [`ServeHost`] and the scheduler-equivalence gate's `RouterHost`, backed by this process's model host.
//!
//! `metrale-bench` declares these seams so that it needs no GPU or server
//! code; this module supplies them.
//!
//! - The roster is derived: `library::scan` reports what is in the HF cache,
//!   and [`classify`] names why a checkpoint cannot run.
//! - A round is an in-process swap onto the port this server already bound,
//!   and `serve` returns only once `/v1/models` names the new checkpoint.
//!
//! Owner: server tui.
//! Invariants:
//! - `original` holds the argv serving at the first `serve` call; `restore`
//!   attempts a swap back to it, and does nothing when it is `None`.

use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use futures::future::BoxFuture;
use metrale_bench::TargetEndpoint;
use metrale_bench::benchmarks::scheduler_equivalence::host::{Lane, RouterHost, ServeVariant};
use metrale_bench::benchmarks::serve_matrix::host::{
    Absence, ServeCandidate, ServeHost, ServeOptions,
};

use crate::main_modules::model_host::ModelHost;

/// 2026-09-26: How long a checkpoint gets to load before the round is reported as a boot failure.
///
/// It bounds the report, not the load: dropping a `spawn_blocking` handle does
/// not stop the task, so an overrunning `model_swap::swap` keeps going and the
/// next round's swap waits behind it on `ModelHost::swap_guard`.
const BOOT_TIMEOUT: Duration = Duration::from_secs(600);

pub struct TuiServeHost {
    host: Arc<ModelHost>,
    /// 2026-09-26: The argv that was serving at the first round, so `restore` can put the box back.
    /// `None` when nothing was serving then.
    original: parking_lot::Mutex<Option<crate::cli::ServeArgs>>,
    cache_dir: Option<std::path::PathBuf>,
}

impl TuiServeHost {
    pub fn new(host: Arc<ModelHost>, cache_dir: Option<std::path::PathBuf>) -> Self {
        Self {
            host,
            original: parking_lot::Mutex::new(None),
            cache_dir,
        }
    }

    /// 2026-09-26: Where this server is listening; the matrix opens no port of its own.
    fn endpoint(&self, model: &str) -> Result<TargetEndpoint> {
        let (_, port) = self
            .host
            .bound()
            .context("this server has not finished binding its port yet")?;
        Ok(TargetEndpoint::local(port, model))
    }

    /// 2026-09-26: Build the argv for one round: the model, this server's port, `--max-seq-len`,
    /// `--speculative` when asked and the cache override. Everything else is what
    /// `met serve <model>` would use.
    fn argv_for(&self, model: &str, opts: ServeOptions) -> Result<crate::cli::ServeArgs> {
        use clap::Parser as _;
        let (_, port) = self
            .host
            .bound()
            .context("this server has not finished binding its port yet")?;
        let mut argv = vec![
            "met".to_string(),
            "serve".to_string(),
            model.to_string(),
            "--port".to_string(),
            port.to_string(),
            "--max-seq-len".to_string(),
            opts.max_seq_len.to_string(),
        ];
        if opts.speculative {
            argv.push("--speculative".to_string());
        }
        if let Some(dir) = &self.cache_dir {
            argv.push("--cache-dir".to_string());
            argv.push(dir.display().to_string());
        }
        let cli = crate::cli::Cli::try_parse_from(&argv).with_context(|| {
            format!("serve matrix produced an invalid command line for {model}")
        })?;
        let crate::cli::Command::Serve(args) = cli.command else {
            bail!("serve matrix did not produce a serve command");
        };
        crate::cli::validate_serve_args(&args).map_err(|e| anyhow!("{model}: {e}"))?;
        Ok(args)
    }

    /// 2026-09-26: Run a swap on a blocking thread: `model_swap::swap` blocks while it loads a model,
    /// which would hold a runtime worker for the whole load.
    async fn swap_blocking(&self, args: crate::cli::ServeArgs) -> Result<()> {
        let host = self.host.clone();
        let handle = tokio::task::spawn_blocking(move || {
            crate::main_modules::model_swap::swap(&host, args).map(|_| ())
        });
        match tokio::time::timeout(BOOT_TIMEOUT, handle).await {
            Ok(Ok(r)) => r,
            Ok(Err(e)) => bail!("the loader thread failed: {e}"),
            Err(_) => bail!(
                "did not load within {}s — the load is still running and the next round will \
                 queue behind it",
                BOOT_TIMEOUT.as_secs()
            ),
        }
    }

    /// 2026-09-26: Wait up to 60 s, polling every 500 ms, until `/v1/models` names `model`.
    async fn await_serving(&self, target: &TargetEndpoint) -> Result<()> {
        let deadline = Instant::now() + Duration::from_secs(60);
        let mut last = String::new();
        while Instant::now() < deadline {
            match metrale_bench::http::list_models(target, Duration::from_secs(10)).await {
                Ok(served) if served.iter().any(|m| m == &target.model) => return Ok(()),
                Ok(served) => last = format!("serving {:?}", served),
                Err(e) => last = format!("{e:#}"),
            }
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
        bail!(
            "{} never reported {} as loaded ({last})",
            target.base_url,
            target.model
        )
    }
}

/// 2026-09-26: Why a scanned checkpoint cannot take part, or `None` when it can.
///
/// Missing weights are checked first: a checkpoint that is not all on disk
/// cannot be judged for kernel support.
pub fn classify(e: &crate::tui::data::library::LibraryEntry) -> Option<Absence> {
    if !e.has_weights {
        Some(Absence::NoWeights)
    } else if e.model_type.is_empty() {
        // 2026-09-26: A config without a `model_type` names no architecture, so it is not reported as
        // `NoKernels`. `library::scan` writes "?" (not empty) when `config.json` cannot be parsed.
        Some(Absence::NoConfig)
    } else if !e.optimized {
        Some(Absence::NoKernels)
    } else {
        None
    }
}

impl ServeHost for TuiServeHost {
    fn roster(&self) -> Result<Vec<ServeCandidate>> {
        Ok(crate::tui::data::library::scan(self.cache_dir.as_deref())
            .into_iter()
            .map(|e| match classify(&e) {
                Some(why) => ServeCandidate::absent(e.id, e.quant, why),
                None => ServeCandidate::ready(e.id, e.quant),
            })
            .collect())
    }

    fn serve(&self, model: &str, opts: ServeOptions) -> BoxFuture<'_, Result<TargetEndpoint>> {
        let model = model.to_string();
        Box::pin(async move {
            // 2026-09-26: Captured at the first round, not at construction: the dashboard may have
            // swapped models since.
            {
                let mut original = self.original.lock();
                if original.is_none() {
                    *original = self.host.args();
                }
            }
            let args = self.argv_for(&model, opts)?;
            let target = self.endpoint(&model)?;
            self.swap_blocking(args).await?;
            self.await_serving(&target).await?;
            Ok(target)
        })
    }

    fn restore(&self) -> BoxFuture<'_, Result<()>> {
        Box::pin(async move {
            let Some(args) = self.original.lock().clone() else {
                // 2026-09-26: Nothing was serving at the first round, so there is no argv to restore;
                // the last round stays loaded.
                return Ok(());
            };
            // 2026-09-26: No short-circuit on the model id: a round may have left the same model at a
            // different `--max-seq-len` or `--speculative`, and `model_swap::swap` skips only an identical argv.
            self.swap_blocking(args)
                .await
                .context("could not put the previous model back")
        })
    }
}

#[cfg(test)]
#[path = "bench_host_tests.rs"]
mod tests;

impl TuiServeHost {
    /// 2026-09-26: The argv that is serving now, with the gate's `--scheduler-config` and lane applied.
    /// Everything else is unchanged, so both routers run under one configuration.
    fn argv_for_variant(&self, variant: ServeVariant) -> Result<crate::cli::ServeArgs> {
        let mut args = self
            .host
            .args()
            .context("no checkpoint is being served, so there is nothing to re-serve")?;
        args.scheduler_config = variant.router.flag_value().to_string();
        match variant.lane {
            Lane::SpecOff => args.speculative = false,
            Lane::MtpForce => args.pin_mtp_force(),
        }
        crate::cli::validate_serve_args(&args).map_err(|e| anyhow!("{}: {e}", variant.label()))?;
        Ok(args)
    }
}

impl RouterHost for TuiServeHost {
    fn serve(&self, variant: ServeVariant) -> BoxFuture<'_, Result<TargetEndpoint>> {
        Box::pin(async move {
            {
                let mut original = self.original.lock();
                if original.is_none() {
                    *original = self.host.args();
                }
            }
            let args = self.argv_for_variant(variant)?;
            let model = args
                .model
                .clone()
                .context("the serving argv names no model")?;
            let target = self.endpoint(&model)?;
            self.swap_blocking(args).await?;
            self.await_serving(&target).await?;
            Ok(target)
        })
    }

    fn diagnostics(&self) -> Result<std::collections::BTreeMap<String, f64>> {
        Ok(crate::scheduler::io::async_device::STATS.snapshot())
    }

    fn restore(&self) -> BoxFuture<'_, Result<()>> {
        ServeHost::restore(self)
    }
}
