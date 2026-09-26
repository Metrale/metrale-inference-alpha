// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: What a gate run would serve: the recipe its baseline names, the
//! checkpoint that recipe must agree on, the override set and the `METRALE_*`
//! levers the record will state.
//!
//! Owner: server CLI (`met benchmark`).
//! Invariants:
//! - `bench_selfstart::serve_for` and `bench_lease::acquire` both plan through
//!   [`plan_serve`], so a self-started and a leased server are the same plan.
//! - A plan's recipe serves the model its baseline entry names; any other is
//!   refused before a model loads.

use std::collections::BTreeMap;

use anyhow::{Context, Result, bail};
use metrale_bench::{gate, serve_env};

use super::bench_resolve::Resolved;

/// 2026-09-26: A resolved serve: everything but the port.
pub struct ServePlan {
    pub model: String,
    pub recipe_id: String,
    pub recipe: crate::recipe::Recipe,
    /// 2026-09-26: The served variant's baseline entry; see `SelfServed::baseline_entry`.
    pub entry: gate::ModelBaseline,
    /// 2026-09-26: The merged `[benchmarks.serve_overrides]` pin and
    /// `--serve-override` set, after `--hermetic` expansion: what the record states.
    pub requested: BTreeMap<String, String>,
    /// 2026-09-26: The `METRALE_*` serve levers the server runs under: the
    /// recipe's `env:` block with the entry's `[benchmarks.serve_env]` pin on
    /// top, both validated by `serve_env::declared`. A leased server's
    /// `env_sha256` must fingerprint to it.
    pub serve_env: BTreeMap<String, String>,
    /// 2026-09-26: The box class the run is for, and its declared limits
    /// (`kernels/<hw>/HARDWARE.toml` `[benchmarks.limits]`): the free-memory
    /// floor checked before a server starts, and its boot timeout.
    pub hardware: String,
    pub limits: metrale_bench::hardware::limits::Limits,
}

impl ServePlan {
    /// 2026-09-26: The argv `met serve` is started with on `port`: the recipe's
    /// rendering, which a reused server must also fingerprint to.
    /// `["met", "serve", <model>, …]`; callers skip the program name.
    pub fn argv(&self, port: u16) -> Result<Vec<String>> {
        let mut overrides = self.requested.clone();
        overrides.insert("port".to_string(), port.to_string());
        self.recipe.argv(&overrides).with_context(|| {
            format!(
                "rendering serve args from recipe {:?} (port override {port})",
                self.recipe_id
            )
        })
    }

    pub fn serve_args(&self, port: u16) -> Result<crate::cli::ServeArgs> {
        let mut overrides = self.requested.clone();
        overrides.insert("port".to_string(), port.to_string());
        self.recipe.serve_args(&overrides).with_context(|| {
            format!(
                "rendering serve args from recipe {:?} (port override {port})",
                self.recipe_id
            )
        })
    }

    /// 2026-09-26: What the gate record discloses about this serve: the knobs
    /// as the server resolves them (`ServeArgs::mtp_gate_force`, so `--hermetic`
    /// counts), keyed as `gate::record_serve` names them. Read off
    /// [`Self::serve_args`], the rendering a leased server's argv digest must
    /// match.
    pub fn disclosed(&self, port: u16) -> Result<BTreeMap<String, String>> {
        Ok(disclosed_from(&self.serve_args(port)?))
    }

    /// 2026-09-26: Reconcile the declared lever set with the levers this process
    /// carries (`serve_env::reconcile`): an undeclared lever, or a declared one
    /// at another value, is refused by name. Both server paths call it before
    /// probing or starting a server.
    pub fn reconcile_env(&self) -> Result<serve_env::Reconciled> {
        serve_env::reconcile(
            &format!("recipe {}", self.recipe_id),
            &self.serve_env,
            &serve_env::process_levers(),
        )
    }
}

/// 2026-09-26: The disclosure for one rendered, validated serve; the body of
/// [`ServePlan::disclosed`], free so a test can call it without a plan.
pub(crate) fn disclosed_from(args: &crate::cli::ServeArgs) -> BTreeMap<String, String> {
    gate::record_serve::disclosure(
        args.mtp_gate_force(),
        args.speculative,
        args.prefill_codispatch,
        args.w4a4_downcast,
    )
}

pub fn plan_serve(
    benchmark_id: &str,
    hardware: Option<&str>,
    checkpoint: Option<&str>,
    overrides: BTreeMap<String, String>,
) -> Result<ServePlan> {
    let root = super::bench_run::repo_root()?;
    // 2026-09-26: A shard runs under its group's id, so it serves the group's
    // baseline entry.
    let serve_id = gate::group::serve_baseline_id(benchmark_id);
    let baseline = gate::read_baseline(&root, serve_id)?;
    let Resolved {
        model,
        recipe_id,
        entry,
        hardware,
    } = super::bench_resolve::resolve(&baseline, serve_id, hardware, checkpoint)?;
    let Some(limits) = metrale_bench::hardware::limits::limits(&root, &hardware)? else {
        bail!(
            "kernels/{hardware}/HARDWARE.toml declares no [benchmarks.limits]: a gate run on this \
             class has no memory floor to check the box against and no boot timeout for its \
             server. Measure them and declare the tables (see kernels/gb10/HARDWARE.toml)."
        );
    };

    let store = metrale_bench::ArtifactStore::discover()?;
    let index = crate::recipe::fetch::cached(store.root());
    let recipe = index
        .recipes
        .iter()
        .find(|r| r.id == recipe_id)
        .with_context(|| {
            format!(
                "recipe {recipe_id:?} is not in the local index ({} cached). The index is read \
                 from {}.{} Populate it with:\n    met sync-recipes\n\
                 (this used to say \"open the TUI Library once\", which a CI runner, a \
                 container, or a machine reached over ssh cannot do.)",
                index.recipes.len(),
                // 2026-09-26: Named through `cache_dir` so the path printed is
                // the path read.
                crate::recipe::fetch::cache_dir(store.root())
                    .join("index.json")
                    .display(),
                // 2026-09-26: Why the index could not be used, when the index
                // layer knows: an unreadable index otherwise reads as one never
                // written, and `sync-recipes` would not fix it.
                index
                    .offline
                    .as_deref()
                    .map(|why| format!(" That index could not be used: {why}."))
                    .unwrap_or_default()
            )
        })?
        .clone();

    // 2026-09-26: The baseline and the recipe must agree on the checkpoint, or
    // the run would be scored against another checkpoint's thresholds. Refused
    // here, before a model load.
    if recipe.model != model {
        bail!(
            "recipe {recipe_id:?} serves {:?} but {benchmark_id}'s baseline is defined on \
             {model:?}. Scoring one checkpoint against another's thresholds is not a lenient \
             comparison, it is a meaningless one.",
            recipe.model
        );
    }

    // 2026-09-26: The recipe's `env:` block with the entry's
    // `[benchmarks.serve_env]` pin on top (the pin wins a clash). Each is
    // validated by `serve_env::declared` before a model loads.
    let serve_env = serve_env::merge_declared(
        serve_env::declared(&format!("recipe {recipe_id}"), &recipe.env)?,
        serve_env::declared(
            &format!("{benchmark_id}'s baseline entry for {model} ([benchmarks.serve_env])"),
            &entry.serve_env,
        )?,
    );
    if !serve_env.is_empty() {
        let shown = serve_env
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join(" ");
        eprintln!("gate: recipe {recipe_id} is measured under serve env: {shown}");
    }

    // 2026-09-26: `--hermetic` expands into the keys it closes before the recipe
    // renders, so a recipe default that turns one on does not contradict it.
    // See `cli::hermetic::CLOSED_KEYS`.
    let requested = crate::cli::hermetic::expand(gate::merge_serve_overrides(
        entry.serve_overrides.clone(),
        overrides,
    ));
    if !requested.is_empty() {
        let shown = requested
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join(" ");
        tracing::warn!(
            "serving recipe {recipe_id} with OVERRIDES: {shown} — this run does not measure the \
             recipe as pinned; the gate record will say so"
        );
    }
    Ok(ServePlan {
        model,
        recipe_id,
        recipe,
        entry,
        requested,
        serve_env,
        hardware,
        limits,
    })
}

#[cfg(test)]
#[path = "bench_serve_plan_tests.rs"]
mod tests;
